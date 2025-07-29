import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="📊 AI for Demand Forecasting")

st.title("📊 AI for Predictive Demand Forecasting")

st.header("🌟 Business Context: The Power of Predicting Future Demand")
st.markdown("""
Accurate demand forecasting is the bedrock of an efficient supply chain. It directly impacts:
-   **Inventory Levels:** Avoids overstocking (high holding costs, obsolescence) and understocking (lost sales, customer dissatisfaction).
-   **Production Planning:** Ensures manufacturing lines produce the right quantities at the right time.
-   **Logistics & Transportation:** Optimizes warehouse space and delivery routes.
-   **Resource Allocation:** Helps manage staffing and raw material procurement.

Inaccurate forecasts, on the other hand, lead to significant inefficiencies and financial losses.
""")

st.subheader("💡 How Machine Learning Drives Forecasting:")
st.markdown("""
We frame demand forecasting as a **supervised learning regression problem**:
1.  **Historical Data:** We use past sales data, along with relevant features (like date components or promotions), as our training set.
2.  **Feature Engineering:** Extract meaningful patterns from dates (e.g., month, day of week, quarter) and create 'lagged' features (e.g., demand from the previous week/month).
3.  **Model Training:** A **Random Forest Regressor** learns the complex relationships and patterns within this historical data. It's a powerful ensemble model that combines many decision trees to make robust predictions.
4.  **Prediction:** The trained model then takes future dates and their corresponding features to predict demand for those periods.

This data-driven approach allows businesses to move beyond simple averages, capturing seasonality, trends, and other hidden patterns to make more informed decisions.
""")

st.divider()

st.header("🔬 Simulation: Demand Forecasting with Random Forest")
st.info("This simulation generates synthetic historical demand data with controllable patterns (base, trend, seasonality, noise) and then uses a Random Forest Regressor to forecast future demand.")

# --- Demand Data Generation ---
@st.cache_data(show_spinner="Generating historical demand data...")
def generate_demand_data(
    start_date, end_date,
    base_demand, trend_strength,
    weekly_seasonality_amplitude, monthly_seasonality_amplitude,
    noise_level
):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'Date': date_range})
    df['DayOfWeek'] = df['Date'].dt.dayofweek # 0=Monday, 6=Sunday
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Year'] = df['Date'].dt.year
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days

    # Base demand
    demand = np.full(len(df), base_demand)

    # Trend
    demand += trend_strength * df['DaysSinceStart'] / 365.0 # Annual trend

    # Weekly Seasonality (e.g., higher on weekends)
    weekly_pattern = np.sin(df['DayOfWeek'] * (2 * np.pi / 7)) # Simple sine wave for week
    demand += weekly_seasonality_amplitude * weekly_pattern

    # Monthly Seasonality (e.g., higher at month end/beginning)
    monthly_pattern = np.sin(df['Month'] * (2 * np.pi / 12)) # Simple sine wave for month
    demand += monthly_seasonality_amplitude * monthly_pattern

    # Add random noise
    demand += np.random.normal(0, noise_level, len(df))

    # Ensure demand is non-negative
    df['Demand'] = np.maximum(0, demand).round(0).astype(int)
    return df

# --- Feature Engineering for ML Model ---
def create_features(df, lags):
    df_features = df.copy()
    
    # Time-based features
    df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
    df_features['Month'] = df_features['Date'].dt.month
    df_features['WeekOfYear'] = df_features['Date'].dt.isocalendar().week.astype(int)
    df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
    df_features['Quarter'] = df_features['Date'].dt.quarter
    
    # Lagged Demand features
    for lag in lags:
        df_features[f'Demand_Lag_{lag}'] = df_features['Demand'].shift(lag)
    
    df_features = df_features.dropna() # Drop rows with NaN (due to lags)
    return df_features

# --- Model Training ---
@st.cache_resource(show_spinner="Training Demand Forecasting Model...")
def train_forecasting_model(features_df, target_column='Demand'):
    X = features_df.drop(['Date', target_column], axis=1)
    y = features_df[target_column]

    # Use a fixed train/test split for reproducibility within cached function
    # For time series, a time-based split is crucial
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return model, mae, rmse, X_test.index, y_test, predictions

# --- Forecasting Function ---
def forecast_future_demand(model, historical_df, forecast_horizon_days, lags):
    last_historical_date = historical_df['Date'].max()
    future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                                 periods=forecast_horizon_days, freq='D')
    
    forecast_df = pd.DataFrame({'Date': future_dates})
    
    # Initialize features for forecasting based on last historical values
    # We'll need a way to carry forward lagged features
    
    # Create a combined dataframe for feature generation
    # We only need the very end of historical_df to generate lags for the first few forecast points
    # A more robust way would be to iteratively predict and use predictions as new lags.
    # For simplicity, let's create features for the whole combined range and then slice.

    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    combined_df_features = create_features(combined_df, lags)
    
    # Ensure all required lag features exist even if they are NaN for early rows
    for lag in lags:
        if f'Demand_Lag_{lag}' not in combined_df_features.columns:
            combined_df_features[f'Demand_Lag_{lag}'] = np.nan # Add if missing

    # Select only the features needed for the model and future dates
    X_forecast = combined_df_features[combined_df_features['Date'].isin(future_dates)].drop(['Date', 'Demand'], axis=1)
    
    # Fill any NaN lags at the start of forecast period with the last known historical demand
    # This is a critical step for time series forecasting with lagged features
    for col in X_forecast.columns:
        if 'Demand_Lag_' in col:
            # Find the value from historical_df to fill this lag
            lag_val = int(col.split('_')[-1])
            if last_historical_date - pd.Timedelta(days=lag_val) in historical_df['Date'].values:
                X_forecast[col] = X_forecast[col].fillna(
                    historical_df[historical_df['Date'] == (last_historical_date - pd.Timedelta(days=lag_val))]['Demand'].iloc[0]
                )
            else:
                # If historical data doesn't cover the full lag period, fill with mean or 0
                X_forecast[col] = X_forecast[col].fillna(0) # Or historical_df['Demand'].mean()

    # Make predictions
    future_predictions = model.predict(X_forecast)
    forecast_df['Forecasted_Demand'] = np.maximum(0, future_predictions).round(0).astype(int)
    
    return forecast_df

# --- Streamlit App UI ---

st.sidebar.header("Historical Data Generation Parameters")
start_date = st.sidebar.date_input("Start Date of Historical Data", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date of Historical Data", datetime.date(2024, 12, 31))

st.sidebar.subheader("Demand Patterns")
base_demand = st.sidebar.slider("Base Daily Demand", 50, 500, 150, step=10)
trend_strength = st.sidebar.slider("Annual Trend Strength", -20, 50, 10)
weekly_seasonality_amplitude = st.sidebar.slider("Weekly Seasonality Amplitude", 0, 100, 30)
monthly_seasonality_amplitude = st.sidebar.slider("Monthly Seasonality Amplitude", 0, 100, 20)
noise_level = st.sidebar.slider("Random Noise Level", 0, 50, 15)

st.sidebar.header("Forecasting Parameters")
forecast_horizon_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 365, 90)
lags_to_use = st.sidebar.multiselect(
    "Demand Lags to Use (Days)",
    options=[7, 14, 28, 30, 60, 90, 180, 365],
    default=[7, 14, 28]
)
if not lags_to_use:
    st.sidebar.warning("Please select at least one demand lag.")
    lags_to_use = [7] # Default if none selected


# Generate Data
historical_demand_df = generate_demand_data(
    start_date, end_date,
    base_demand, trend_strength,
    weekly_seasonality_amplitude, monthly_seasonality_amplitude,
    noise_level
)

# Prepare features for model
# Important: `create_features` drops NaNs, so it implicitly sets the start of the training period.
features_df = create_features(historical_demand_df, lags=lags_to_use)

# Train Model
model, mae, rmse, test_indices, y_test_actual, y_test_pred = train_forecasting_model(
    features_df, target_column='Demand'
)

st.subheader("1. Historical Demand Data (Generated)")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(historical_demand_df['Date'], historical_demand_df['Demand'], label='Historical Demand', color='dodgerblue')
ax1.set_title('Simulated Historical Demand')
ax1.set_xlabel('Date')
ax1.set_ylabel('Demand Units')
ax1.legend()
st.pyplot(fig1)

st.subheader("2. Model Training & Evaluation")
st.markdown(f"""
The Random Forest model was trained on historical data.
-   **Mean Absolute Error (MAE):** `{mae:.2f}` units
-   **Root Mean Squared Error (RMSE):** `{rmse:.2f}` units
""")
st.info("MAE indicates the average absolute difference between actual and predicted demand. RMSE penalizes larger errors more heavily.")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(historical_demand_df['Date'].iloc[test_indices], y_test_actual, label='Actual Test Demand', color='orange')
ax2.plot(historical_demand_df['Date'].iloc[test_indices], y_test_pred, label='Predicted Test Demand', color='green', linestyle='--')
ax2.set_title('Model Performance on Test Set (Out-of-Sample)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Demand Units')
ax2.legend()
st.pyplot(fig2)


st.subheader(f"3. Future Demand Forecast ({forecast_horizon_days} Days Horizon)")
future_forecast_df = forecast_future_demand(model, historical_demand_df, forecast_horizon_days, lags=lags_to_use)

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(historical_demand_df['Date'], historical_demand_df['Demand'], label='Historical Demand', color='dodgerblue')
ax3.plot(future_forecast_df['Date'], future_forecast_df['Forecasted_Demand'], label='Forecasted Demand', color='red', linestyle='--')
ax3.axvline(x=historical_demand_df['Date'].max(), color='gray', linestyle=':', label='Forecast Start')
ax3.set_title('Historical Demand vs. Future Forecast')
ax3.set_xlabel('Date')
ax3.set_ylabel('Demand Units')
ax3.legend()
st.pyplot(fig3)

st.subheader("4. Raw Forecast Data")
st.dataframe(future_forecast_df.head(10)) # Show first 10 rows of forecast

st.divider()

st.header("🚀 Beyond This Demo: Real-World Demand Forecasting")
st.markdown("""
This simulation provides a foundational understanding. Real-world demand forecasting systems involve:
-   **More Sophisticated Models:** ARIMA, Prophet, Neural Networks (LSTMs, Transformers) for complex patterns.
-   **Exogenous Variables:** Incorporating external factors like promotions, holidays, competitor actions, economic indicators, weather.
-   **Granularity:** Forecasting at different levels (SKU, store, region, channel).
-   **Probabilistic Forecasting:** Predicting not just a single value, but a range of possible demand outcomes (e.g., 95% confidence intervals).
-   **Automated Feature Engineering:** Tools that automatically create relevant features from time series data.
-   **Model Monitoring:** Continuously tracking model performance and retraining as needed.

By integrating these advanced techniques, businesses can build highly accurate and resilient forecasting systems, leading to significant improvements across the entire supply chain.
""")
