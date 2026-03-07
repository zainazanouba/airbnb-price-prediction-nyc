# Airbnb Price Prediction – New York

## 📌 Project Overview
This project focuses on predicting Airbnb listing prices in New York using machine learning techniques.

The workflow includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation
- Deployment through a simple Streamlit application

---

## 📂 Project Structure

```
AIRBNB-main/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_model_training.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧠 Model
The trained model is generated automatically when training is performed
and saved locally as:
```
model.joblib
```
This file is not included in the repository.
```

---

## 📊 Dataset
The dataset is not included in this repository due to size constraints.

To reproduce the project:
1. Download the Airbnb New York dataset from Kaggle.
2. Place the raw dataset inside:
```
data/raw/
```

---

## 🚀 How to Run the Application

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the Streamlit app:
```
streamlit run app/streamlit_app.py
```

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 👩‍💻 Contributors
This project was developed collaboratively as part of an academic machine learning project.

Primary contributor (repository maintainer):
- Zainab Elkamit