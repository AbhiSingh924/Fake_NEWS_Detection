import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1. Load the updated dataset
df = pd.read_csv(r"E:\Fake_NEWS_Detection\fake_news_dataset.csv")

# 2. Keep only relevant columns
df = df[['text', 'label']]

# 3. Drop rows with missing values
df.dropna(subset=['text', 'label'], inplace=True)

# 4. Convert label to numeric if needed
if df['label'].dtype == 'object':
    df['label'] = df['label'].str.strip().str.upper().map({'FAKE': 0, 'REAL': 1})

# 5. Drop any rows that couldn't be mapped
df.dropna(subset=['label'], inplace=True)

# 6. Convert label to int
df['label'] = df['label'].astype(int)

# 7. Print size and label distribution
print("✅ Final data size:", df.shape)
print("✅ Label distribution:\n", df['label'].value_counts())

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# 9. Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.75)),
    ('clf', MultinomialNB())
])

# 10. Hyperparameter tuning
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [1.0, 0.5, 0.1]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# 11. Evaluation
y_pred = grid.predict(X_test)
print(f"\n✅ Best Parameters: {grid.best_params_}")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 12. Save model
joblib.dump(grid.best_estimator_, "fake_news_model.pkl")
print
