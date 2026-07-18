from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier([
    "I love building AI from scratch",
    "This is boring and I hate it",
    "Islamabad is a beautiful city"
])

for result in results:
  print(result)
