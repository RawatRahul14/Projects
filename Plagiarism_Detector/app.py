from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(open("models/model.pickle", "rb"))
vectors = pickle.load(open("models/vectors.pickle", "rb"))

def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)

    return text
def detect(input_text):
    input_text = preprocess_text(input_text)
    input_vector = vectors.transform([input_text])
    result = model.predict(input_vector)
    return "Plagiarism Detected" if result[0] == 1 else "Plagiarism Not Detected"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect_plagiarism():
    input_text = request.form["text"]
    detection_result = detect(input_text)
    return render_template("index.html", result = detection_result)

if __name__ == "__main__":
    app.run(debug = True)