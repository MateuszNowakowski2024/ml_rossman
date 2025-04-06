import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import io

@st.cache_data
def load_data():
    df_all = pd.read_csv("data/combined_stores.csv")
    df_all['StateHoliday'] = df_all['StateHoliday'].astype(str).replace({'0': 0, 'a': 1, 'b': 1, 'c': 1}).astype(int)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    return df_all

def prepare_store_data(df_all, store_number):
    df = df_all[df_all['Store'] == store_number].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Sales_Lag1"] = df["Sales"].shift(1)
    df["Sales_Lag7"] = df["Sales"].shift(7)
    df["Sales_Lag14"] = df["Sales"].shift(14)
    df["Sales_Rolling7"] = df["Sales"].shift(1).rolling(window=7).mean()
    df["Sales_Rolling14"] = df["Sales"].shift(1).rolling(window=14).mean()
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df.dropna().reset_index(drop=True)

def train_model(df_model):
    features = [
        "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
        "Sales_Lag1", "Sales_Lag7", "Sales_Lag14",
        "Sales_Rolling7", "Sales_Rolling14",
        "Day", "Month", "Year", "WeekOfYear", "DayOfYear"
    ]
    target = "Sales"
    X = df_model[features]
    y = df_model[target]
    split_index = int(len(df_model) * 0.8)
    X_train = X[:split_index]
    y_train = y[:split_index]
    model = XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1, 
                         objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model

def forecast_future(model, df_model, days_ahead, promo_dates, holiday_dates):
    features = [
        "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
        "Sales_Lag1", "Sales_Lag7", "Sales_Lag14",
        "Sales_Rolling7", "Sales_Rolling14",
        "Day", "Month", "Year", "WeekOfYear", "DayOfYear"
    ]
    last_date = df_model["Date"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
    history = df_model.copy()
    future_predictions = []

    for future_date in future_dates:
        row = {
            "Date": future_date,
            "DayOfWeek": future_date.weekday() + 1,
            "Promo": int(future_date in promo_dates),
            "StateHoliday": int(future_date in holiday_dates),
            "SchoolHoliday": 0,
            "Day": future_date.day,
            "Month": future_date.month,
            "Year": future_date.year,
            "WeekOfYear": future_date.isocalendar().week,
            "DayOfYear": future_date.timetuple().tm_yday
        }
        for lag in [1, 7, 14]:
            row[f"Sales_Lag{lag}"] = history["Sales"].iloc[-lag]
        row["Sales_Rolling7"] = history["Sales"].iloc[-7:].mean()
        row["Sales_Rolling14"] = history["Sales"].iloc[-14:].mean()
        row_df = pd.DataFrame([row])
        X_future = row_df[features]
        row["Sales"] = model.predict(X_future)[0]
        future_predictions.append(row)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    forecast_df = pd.DataFrame(future_predictions)
    return forecast_df[["Date", "Sales"]]

# --- Streamlit App ---
st.set_page_config(page_title="Store Sales Forecasting", layout="wide")
st.title("Store Sales Forecasting")
st.write("""
This interactive app forecasts future store sales using an XGBoost model trained on historical data.
Compare sales and forecasts across up to 5 different stores.
""")

store_options = [145, 260, 280, 347, 391, 436, 507, 538, 554, 593, 637, 658, 682, 742, 951, 982, 986, 993, 1089, 1108]  # Store selection
selected_stores = st.sidebar.multiselect("Select up to 5 Store Numbers to Compare", options=store_options, default=[145], max_selections=5)

if selected_stores:
    df_all = load_data()

    # Filter historical period selection
    st.subheader("📊 Historical Sales Comparison")
    view_mode = st.sidebar.radio("Select data range:", ["All Data", "Specific Month", "Specific Week"])

    date_filter = None
    if view_mode == "Specific Month":
        month_year = st.sidebar.date_input("Choose a month (1st of any month)", value=pd.to_datetime("2014-01-01"))
        date_filter = lambda df: df[(df["Date"].dt.month == month_year.month) & (df["Date"].dt.year == month_year.year)]
    elif view_mode == "Specific Week":
        week_year = st.sidebar.date_input("Choose a date in the week", value=pd.to_datetime("2014-01-01"))
        week_num = week_year.isocalendar().week
        date_filter = lambda df: df[(df["Date"].dt.isocalendar().week == week_num) & (df["Date"].dt.year == week_year.year)]

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    for store in selected_stores:
        df_model = prepare_store_data(df_all, store)
        if date_filter:
            df_model = date_filter(df_model)
        ax1.plot(df_model["Date"], df_model["Sales"], label=f"Store {store}")
    ax1.set_title("Actual Sales per Store")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot forecasted sales separately
    st.subheader("🔮 Forecasted Sales")
    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30)
    promo_input = st.sidebar.text_input("Enter promo dates (comma-separated, e.g. 2025-04-10, 2025-04-15)", "")
    holiday_input = st.sidebar.text_input("Enter public holiday dates (comma-separated, e.g. 2025-04-12)", "")

    promo_dates = set(pd.to_datetime(promo_input.split(","), errors='coerce').dropna())
    holiday_dates = set(pd.to_datetime(holiday_input.split(","), errors='coerce').dropna())
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for store in selected_stores:
        df_model = prepare_store_data(df_all, store)
        model = train_model(df_model)
        forecast_df = forecast_future(model, df_model, forecast_days, promo_dates, holiday_dates)
        ax2.plot(forecast_df["Date"], forecast_df["Sales"], linestyle="--", label=f"Store {store} Forecast")
    ax2.set_title("Forecasted Sales per Store")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.caption("Developed as a portfolio project showcasing time series forecasting in retail analytics.")
