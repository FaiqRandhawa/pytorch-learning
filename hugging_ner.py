from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

text = "Faiq is studying at FAST NUCES and doing internship "

results = ner(text)
for r in results:
    print(f"{r['entity_group']} — {r['word']} ({r['score']:.2f})")