from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

fake_words = ["shocking", "breaking", "secret", "exclusive"]

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy_value = f.read()

# Load confusion matrix
with open("confusion_matrix.txt", "r") as f:
    cm = f.read()

# Load report
with open("report.txt", "r") as f:
    report = f.read()

@app.route('/')
def home():
    return render_template("index.html",
                           accuracy=accuracy_value,
                           cm=cm,
                           report=report)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = round(max(prob) * 100, 2)

    found_words = [word for word in fake_words if word in text.lower()]

    return render_template("index.html",
                           result="Fake News" if prediction == 0 else "Real News",
                           confidence=confidence,
                           suspicious=found_words,
                           accuracy=accuracy_value,
                           cm=cm,
                           report=report)

if __name__ == "__main__":
    app.run(debug=True)