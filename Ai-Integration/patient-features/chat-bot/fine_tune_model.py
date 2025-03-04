import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the DeepSeek model and tokenizer
model_name = "deepseek-ai/deepseek-llm-1.5b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and preprocess the dataset
def load_health_dataset():
    # Load a real health Q&A dataset (example using a CSV file)
    df = pd.read_csv("health_qa_dataset.csv")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

dataset = load_health_dataset()

def preprocess_function(examples):
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    fp16=True,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_health_model")
tokenizer.save_pretrained("./fine_tuned_health_model")
