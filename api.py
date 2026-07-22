from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "text": input.text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }

from fastapi.responses import HTMLResponse

@app.get("/")
def home():
    return HTMLResponse("""
    <html>
    <body>
        <h2>Sentiment Analyzer</h2>
        <textarea id="text" rows="4" cols="50"></textarea><br><br>
        <button onclick="analyze()">Analyze</button>
        <p id="result"></p>
        <script>
        async function analyze() {
            const text = document.getElementById('text').value;
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text})
            });
            const data = await res.json();
            document.getElementById('result').innerText = 
                data.sentiment + ' (' + (data.confidence * 100).toFixed(1) + '%)';
        }
        </script>
    </body>
    </html>
    """)    