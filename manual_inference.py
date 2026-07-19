from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# load tokenizer and model separately
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model     = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model.eval()

texts = [
    "I love building AI from scratch",
    "This is boring and I hate it"
]




for text in texts:
    inputs  = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted     = torch.argmax(probabilities, dim=1).item()
    confidence    = probabilities[0][predicted].item()
    
    label = "POSITIVE" if predicted == 1 else "NEGATIVE"
    print(f"{text[:40]} → {label} ({confidence:.2%})")