Airbnb Price Prediction – New York

📌 Project Overview

This project focuses on predicting Airbnb listing prices in New York using machine learning techniques.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation
- Deployment through a simple Streamlit application

---

📂 Project Structure

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

The trained machine learning model used by the Streamlit application is included in the repository.

Location:

app/model.joblib

---

📊 Dataset

The dataset used for this project is already included in the repository.

Location:

data/raw/Airbnb_Open_Data.csv

---

🚀 How to Run the Application

1. Install dependencies:

pip install -r requirements.txt

2. Run the Streamlit app:

streamlit run app/streamlit_app.py

---

🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

👩‍💻 Contributors

This project was developed collaboratively as part of an academic machine learning project.

Primary contributor (repository maintainer):

- Zainab Elkamit