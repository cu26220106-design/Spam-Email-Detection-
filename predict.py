import os
import sys
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE, "models", "spam_classifier.joblib")
VECT = os.path.join(BASE, "models", "tfidf_vectorizer.joblib")

if not os.path.exists(MODEL) or not os.path.exists(VECT):
    print("Model not found. Run: python src/train.py")
    raise SystemExit(1)

text = " ".join(sys.argv[1:]).strip()
if not text:
    text = input("Enter email text: ").strip()

model = joblib.load(MODEL)
vectorizer = joblib.load(VECT)
prediction = model.predict(vectorizer.transform([text]))[0]
probability = max(model.predict_proba(vectorizer.transform([text]))[0])

print("Prediction:", "SPAM" if prediction == "spam" else "HAM (Not Spam)")
print("Confidence:", f"{probability:.2%}")
