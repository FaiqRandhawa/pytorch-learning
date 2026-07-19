from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text = "I am Faiq and I am learning AI in Islamabad"

# tokenize
tokens = tokenizer.tokenize(text)
print("tokens:", tokens)

# convert to ids
ids = tokenizer.encode(text)
print("ids:", ids)

# decode back to text
decoded = tokenizer.decode(ids)
print("decoded:", decoded)

# what the model actually receives
inputs = tokenizer(text, return_tensors="pt")
print("input shape:", inputs['input_ids'].shape)
print("inputs:", inputs)