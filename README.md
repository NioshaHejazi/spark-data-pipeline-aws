# 🚀 Spark-Based Text Classification Pipeline (Local Version)

A modular pipeline that uses PySpark to process raw text, extract features, and classify documents using a logistic regression model trained on real-world data (20 Newsgroups).  
Fully local, clean, and ready for cloud deployment (EMR/S3 support coming soon).

---

## 📁 Project Structure
```
spark-data-pipeline-aws/
├── jobs/                  # Spark jobs
│   ├── preprocess.py      # Tokenization + cleaning
│   ├── embedding.py       # TF-IDF embedding
│   ├── train_classifier.py# Offline training with sklearn
│   └── classify.py        # Predict new input
├── models/                # Saved model + vectorizer
├── utils/                 # (planned) S3/logging utils
├── config/                # (planned) YAML config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 Features

- ✅ Local Spark jobs (run in WSL)
- ✅ Real tokenization and TF-IDF with PySpark
- ✅ Trained logistic regression model with real dataset
- ✅ Predicts new inputs with clean output
- 🔜 EMR/S3 integration (coming soon)

---

## 🧠 Model Info

- Dataset: `20 Newsgroups` (sci.space vs comp.graphics)
- Vectorizer: `TfidfVectorizer` (max_features=5000)
- Classifier: Logistic Regression
- Accuracy: ~92% on test split

---

## 🛠️ Usage (Local Mode)

### 🔹 Preprocess input
```bash
   python3 jobs/preprocess.py
```
### 🔹 Generate embeddings

```bash
python3 jobs/embedding.py
```

### 🔹 Train model

```bash
python3 jobs/train_classifier.py
```

### 🔹 Classify input

```bash
python3 jobs/classify.py
```

---

## 📦 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```


## 📤 Planned: Cloud Integration (EMR + S3)

These features are in progress and will be added soon:

- Upload/download to **Amazon S3**
- Run Spark jobs on **AWS EMR**
- Use `run_pipeline.py` to switch between local and cloud modes
- Bootstrap scripts to set up EMR clusters

---

## ✍️ Author

Built with ❤️ and Spark by Niosha Hejazi  

---

## ✅ Status

**Project:** ✅ Local complete  
**Cloud:** 🔜 S3 + EMR in progress
