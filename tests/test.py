import requests

texts = [
    "this is wonderful!",
    "i feel sick of this",
    "fuck you man, imma kill all of your family",
    "this is the best day ever",
    "im so happy im gonna cry",
    "this man deserves to die, he's what's wrong with this world"
]


for t in texts:
    r = requests.post("http://127.0.0.1:8000/predict", json={"text":t})
    print(r.status_code, r.json())
