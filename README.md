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

```text
AIRBNB-main/
│
├── app/
│   ├── streamlit_app.py
│   └── model.joblib
│
├── data/
│   ├── raw/
│   │   └── Airbnb_Open_Data.csv
│   └── processed/
│       └── airbnb_processed2.pkl
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_model_training.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt


---

🧠 Model

The trained model is generated during training and saved locally as:

app/model.joblib

This file is included so the prediction page can run directly without retraining.


---

📊 Dataset

The dataset used in this project is included in the repository.

Files used:

Raw dataset: data/raw/Airbnb_Open_Data.csv

Processed dataset: data/processed/airbnb_processed2.pkl


This allows the application and notebooks to run with the existing project structure.


---

🚀 How to Run the Application

1. Install dependencies:



pip install -r requirements.txt

2. Run the Streamlit app:



streamlit run app/streamlit_app.py


---

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Joblib



---

👩‍💻 Contributors

This project was developed collaboratively as part of an academic machine learning project.

Primary contributor (repository maintainer):

Zainab Elkamit
