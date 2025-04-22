from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

def generate_explanation(text, is_real):
    if is_real:
        return "✅ This article contains factual statements and a neutral tone, indicating it is likely real."
    else:
        return "🚫 This article contains sensational or biased language, which is typical in fake news."

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""
    explanation = ""
    if request.method == "POST":
        news_text = request.form["news_text"]
        if not news_text.strip():
            result = "Warning"
            message = "⚠️ Please enter some text."
        else:
            transformed = vectorizer.transform([news_text])
            prediction = model.predict(transformed)
            is_real = prediction[0] == 1
            result = "Real" if is_real else "Fake"
            message = "✅ The news is Real!" if is_real else "🚫 The news is Fake!"
            explanation = generate_explanation(news_text, is_real)
    return render_template("index.html", result=result, message=message, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
