# Spam Email Detection — Machine Learning Project

A complete, self-contained machine learning project that classifies an email as **Spam** or **Ham (Not Spam)**.

## Features
- TF-IDF text feature extraction
- Logistic Regression classifier
- Train/test split with stratification
- Accuracy, precision, recall, F1-score and confusion matrix
- Saves the trained model and vectorizer
- Command-line prediction for new emails
- No viva questions included

## Project structure
```
spam_email_detection_project/
├── data/emails.csv
├── models/                  # generated after training
├── src/
│   ├── train.py
│   └── predict.py
├── spam_email_detection.ipynb
├── requirements.txt
├── README.md
└── run_project.py
```

## Installation
Use Python 3.9+.

```bash
pip install -r requirements.txt
```

## Train the model
```bash
python src/train.py
```

This creates:
- `models/spam_classifier.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/confusion_matrix.png`

## Predict a new email
```bash
python src/predict.py "Congratulations, you won a free prize. Click here now!"
```

Example:
```text
Prediction: SPAM
```

## Run the interactive program
```bash
python run_project.py
```

Paste an email and press Enter.

## Machine learning workflow
1. Load labeled email text
2. Split data into training and testing sets
3. Convert text into TF-IDF numerical features
4. Train Logistic Regression
5. Evaluate the classifier
6. Save model and vectorizer
7. Predict new emails

## Dataset
`data/emails.csv` is a small educational dataset included with the project so it can run offline. It contains examples labeled `spam` and `ham`. For a production-quality system, replace it with a much larger real-world email dataset.

## Note
This is an educational college project. The included dataset is intentionally compact and self-contained.
