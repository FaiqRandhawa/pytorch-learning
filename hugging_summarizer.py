from transformers import pipeline

summarizer = pipeline("summarization")

text = """
Pakistan's technology sector has been growing rapidly over the past decade. 
Many young developers are now pursuing careers in artificial intelligence and 
machine learning. Universities across the country have started offering specialized 
courses in data science and deep learning. The government has also launched 
several initiatives to promote technology education and create opportunities 
for skilled professionals in the global market.
"""

result = summarizer(text, max_length=50, min_length=20)
print(result[0]['summary_text'])