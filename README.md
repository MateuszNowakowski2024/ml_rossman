# Store Sales Forecasting App

This project is an interactive web application built with **Streamlit** that forecasts daily sales for individual retail stores using historical data and machine learning. It is designed to help businesses make data-informed decisions around inventory planning, staffing, and promotional strategies.

---

## 📌 Project Overview

The application leverages **XGBoost**, a powerful gradient boosting framework, to model time series data from 2013 to 2015. By incorporating features such as holidays, promotions, and calendar effects, it provides accurate forecasts tailored to each store. Users can explore predictions through an intuitive web interface, selecting specific store IDs and visualizing trends over time.

---

## 💡 Problem Statement

Retailers face significant challenges in predicting future sales, which can result in overstocking, missed revenue opportunities, or poor staffing decisions. This project aims to solve this problem by building a scalable machine learning solution that provides reliable sales forecasts using historical performance data.

---

## 🔧 Features

- **Interactive Streamlit web UI** for exploring predictions by store  
- **Time series forecasting** with XGBoost  
- **Feature engineering** with calendar and event-based variables  
- **Daily sales forecasts** visualized alongside historical data  
- **Persistent storage** of user interactions and inputs to local disk and AWS S3  

---

## 🛠️ Tech Stack

- **Languages & Libraries**: Python, pandas, NumPy, scikit-learn, XGBoost  
- **Visualization & UI**: Streamlit, Matplotlib  
- **Cloud & Storage**: AWS S3, OS file handling  
- **Development Tools**: Visual Studio Code, Git  

---

## 🚀 Getting Started

1. **Clone this repository**:
    ```bash
    git clone https://github.com/your-username/store-sales-forecasting-app.git
    cd store-sales-forecasting-app
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Launch the app**:
    ```bash
    streamlit run app.py
    ```

---

## 📁 Project Structure

```plaintext
store-sales-forecasting-app/
│
├── app.py                     # Streamlit frontend
├── model/                     # Trained models and training scripts
├── data/                      # Historical sales data
├── utils/                     # Helper functions for preprocessing, plotting, etc.
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📦 Output & Storage

- Forecast results are generated in real time and visualized in the app.  
- All user inputs and conversations can be saved locally or to an S3 bucket for reproducibility and traceability.

---

## 📬 Contact

Feel free to connect with me via **LinkedIn** or reach out if you have questions or suggestions!