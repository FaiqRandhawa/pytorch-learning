from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np

# load small subset for speed (full dataset takes hours on CPU)
dataset = load_dataset("stanfordnlp/imdb")
small_train = dataset['train'].shuffle(seed=42).select(range(500))
small_test  = dataset['test'].shuffle(seed=42).select(range(100))

# tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=128)

small_train = small_train.map(tokenize, batched=True)
small_test  = small_test.map(tokenize, batched=True)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# training arguments
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    logging_steps=10,
)

# accuracy metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train,
    eval_dataset=small_test,
    compute_metrics=compute_metrics,
)

trainer.train()