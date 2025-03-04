from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from prometheus_client import Counter, Histogram, start_http_server
import spacy
import asyncio

# Set up FastAPI app
app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Set up Redis for rate limiting and caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_health_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up authentication
SECRET_KEY = "your-secret-key"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Set up SpaCy for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Prometheus metrics
REQUESTS = Counter('chat_requests_total', 'Total chat requests')
ERRORS = Counter('chat_errors_total', 'Total chat errors')
RESPONSE_TIME = Histogram('chat_response_time_seconds', 'Response time in seconds')

# Start Prometheus metrics server
start_http_server(8000)

class User(BaseModel):
    username: str
    email: str
    full_name: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str = None

class ChatInput(BaseModel):
    message: str

class ChatOutput(BaseModel):
    response: str

class Feedback(BaseModel):
    chat_id: str
    rating: int
    comment: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def is_health_related(text):
    doc = nlp(text)
    health_entities = ["DISEASE", "SYMPTOM", "TREATMENT", "MEDICATION"]
    return any(ent.label_ in health_entities for ent in doc.ents)

def filter_response(response):
    sentences = response.split('.')
    filtered_sentences = [s for s in sentences if is_health_related(s)]
    return '. '.join(filtered_sentences)

class DeepSeekWrapper:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def generate(self, input_text):
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = await asyncio.to_thread(
                    self.model.generate,
                    input_ids,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7
                )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return filter_response(response)
        except Exception as e:
            logger.error(f"Error in model generation: {str(e)}")
            return "I'm sorry, but I'm having trouble understanding. Could you please rephrase your question?"

model_wrapper = DeepSeekWrapper("./fine_tuned_health_model")

@app.post("/api/chat", response_model=ChatOutput)
@limiter.limit("5/minute")
async def chat(request: Request, chat_input: ChatInput, current_user: User = Depends(get_current_active_user)):
    REQUESTS.inc()
    start_time = time.time()
    try:
        if not chat_input.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")

        if not is_health_related(chat_input.message):
            raise HTTPException(status_code=400, detail="Please ask a health-related question")

        cache_key = f"chat:{chat_input.message}"
        cached_response = redis_client.get(cache_key)
        if cached_response:
            return ChatOutput(response=cached_response.decode('utf-8'))

        response = await model_wrapper.generate(chat_input.message)

        max_response_length = 150
        if len(response) > max_response_length:
            response = response[:max_response_length] + "..."

        redis_client.setex(cache_key, 3600, response)  # Cache for 1 hour

        logger.info(f"User {current_user.username} input: {chat_input.message}")
        logger.info(f"Chatbot response: {response}")

        return ChatOutput(response=response)

    except Exception as e:
        ERRORS.inc()
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")
    finally:
        RESPONSE_TIME.observe(time.time() - start_time)

@app.post("/api/feedback")
async def submit_feedback(feedback: Feedback, current_user: User = Depends(get_current_active_user)):
    try:
        # Store feedback in Redis (as an example, you might want to use a more permanent storage in production)
        feedback_key = f"feedback:{feedback.chat_id}"
        feedback_data = f"{current_user.username}|{feedback.rating}|{feedback.comment}"
        redis_client.set(feedback_key, feedback_data)
        
        logger.info(f"Feedback received for chat {feedback.chat_id}: User: {current_user.username}, Rating: {feedback.rating}, Comment: {feedback.comment}")
        return {"status": "Feedback received"}
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your feedback")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
