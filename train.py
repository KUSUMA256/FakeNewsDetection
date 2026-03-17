import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake['label'] = 0
true['label'] = 1

# Combine + shuffle
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)

# Preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['text'] = df['text'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
report = classification_report(y_test, y_pred)

# Print
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save accuracy
with open("accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))

# Save confusion matrix
with open("confusion_matrix.txt", "w") as f:
    f.write(str(cm))

# Save classification report
with open("report.txt", "w") as f:
    f.write(report)

print("✅ Model + Evaluation saved successfully!")