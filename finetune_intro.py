from datasets import load_dataset
from transformers import AutoTokenizer

# load real dataset
dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
print(dataset['train'][0])