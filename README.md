# 🕵️ Fake News Detector (Streamlit App)

This project is a simple and interactive web app built using **Streamlit** that uses a **Logistic Regression model** to classify news content as either **Real** or **Fake**.

---

## 📁 Folder Contents

```
fake-news-detector/
│
├── app.py                  # Streamlit frontend UI
├── fake_news_model.pkl     # Pretrained machine learning model
├── requirements.txt        # Required Python packages
└── README.md               # Instructions (this file)
```

---

## 🚀 How to Run the App Locally

### 1. ✅ Install Python (3.8+ recommended)

Make sure Python is installed. You can verify with:

```bash
python --version
```

---

### 2. 📦 Install Dependencies

Navigate to the project folder and run:

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit`
- `scikit-learn`
- `pandas`
- `joblib`

---

### 3. ▶️ Run the App

Launch the app locally with:

```bash
streamlit run app.py
```

It will open in your browser at [http://localhost:8501](http://localhost:8501)

---

## ✍️ How to Use

1. Type or paste a news article, headline, or paragraph into the text box.
2. Click the **Detect** button.
3. The model will predict if the content is **REAL** or **FAKE**.

---

## 📂 Notes

- The model (`fake_news_model.pkl`) was trained on a cleaned dataset using a TF-IDF vectorizer + Logistic Regression via a pipeline.
- It uses GridSearchCV to find the best parameters before saving only the best estimator.

---

## 📤 Want to Deploy?

You can deploy this app easily on [Streamlit Cloud](https://streamlit.io/cloud):
1. Push this folder to GitHub
2. Go to Streamlit Cloud and connect your repo
3. Set `app.py` as the entry point

---

## 👨‍💻 Author

**Abhishek Anand**  
Built with ❤️ using Python & Streamlit
