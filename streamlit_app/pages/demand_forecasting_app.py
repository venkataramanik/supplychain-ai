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
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days

    # Base demand
    demand = np.full(len(df), base_demand)

    # Trend
    demand += trend_strength * df['DaysSinceStart'] / 365.0 # Annual trend

    # Weekly Seasonality (e.g., higher on weekends)
    weekly_pattern = np.sin(df['Date'].dt.dayofweek * (2 * np.pi / 7)) # Simple sine wave for week
    demand += weekly_seasonality_amplitude * weekly_pattern

    # Monthly Seasonality (e.g., higher at month end/beginning)
    # Using day of month for monthly seasonality
    demand += monthly_seasonality_amplitude * np.sin(df['Date'].dt.day * (2 * np.pi / df['Date'].dt.days_in_month))
    
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
    # CORRECTED: Removed .astype(int) here as .week already returns an integer type from isocalendar() Series
    df_features['WeekOfYear'] = df_features['Date'].dt.isocalendar().week 
    df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
    df_features['Quarter'] = df_features['Date'].dt.quarter
    df_features['Year'] = df_features['Date'].dt.year

    # Lagged Demand features
    for lag in lags:
        df_features[f'Demand_Lag_{lag}'] = df_features['Demand'].shift(lag)
    
    return df_features

# --- Model Training ---
@st.cache_resource(show_spinner="Training Demand Forecasting Model...")
def train_forecasting_model(historical_df, lags, target_column='Demand'):
    # Create features for the entire historical dataset
    features_df_raw = create_features(historical_df, lags=lags)
    
    # Handle NaNs specifically here for training by dropping rows with any NaN feature
    # This ensures a clean training set for the model.
    features_df_cleaned = features_df_raw.dropna(subset=[col for col in features_df_raw.columns if col != 'Date']) 

    # Ensure there's enough data after dropping NaNs
    if features_df_cleaned.empty:
        st.error("Not enough historical data or selected lags are too large to create features. Please adjust parameters.")
        return None, 0, 0, [], [], [], [] # Return None for model if training fails

    X = features_df_cleaned.drop(['Date', target_column], axis=1)
    y = features_df_cleaned[target_column]

    # For time series, a time-based split is crucial, so we split based on index
    train_size = int(len(X) * 0.8)
    
    # Ensure train_size is at least 1 and test_size is at least 1 if possible
    if train_size < 1 and len(X) >= 1:
        train_size = 1
    elif train_size == 0 and len(X) == 0:
        st.error("No data available for training after feature creation and NaN removal. Adjust parameters.")
        return None, 0, 0, [], [], [], []

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Handle cases where test set might be empty (e.g., very short historical data)
    if X_test.empty:
        st.warning("Test set is empty. Model training and evaluation will be based on training data only (less reliable). Please extend historical data or reduce lags.")
        X_test, y_test = X_train, y_train # Use training data for 'test' evaluation for demo if test set is empty

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Return the columns used for training to ensure consistency in forecasting
    return model, mae, rmse, X_test.index, y_test, predictions, X.columns.tolist()

# --- Forecasting Function ---
def forecast_future_demand(model, historical_df, forecast_horizon_days, lags, feature_names):
    forecast_df = pd.DataFrame(columns=['Date', 'Forecasted_Demand'])
    
    # We need a mutable series of demand data that grows with predictions
    # This includes historical data relevant for lags for the first forecast days.
    max_lag = max(lags) if lags else 0
    
    # Ensure `demand_history_for_lags` always contains enough data for the longest lag
    # This list will be updated with new predictions in each iteration.
    demand_history_for_lags = historical_df['Demand'].iloc[-max_lag:].tolist() if max_lag > 0 else []
    
    # Start date for the first prediction
    next_date = historical_df['Date'].max() + pd.Timedelta(days=1)

    for i in range(forecast_horizon_days):
        # Create an empty DataFrame row with the expected feature columns from training
        X_predict_row_df = pd.DataFrame(columns=feature_names)
        
        # Populate time-based features for the current date
        X_predict_row_df.loc[0, 'DayOfWeek'] = next_date.dayofweek
        X_predict_row_df.loc[0, 'Month'] = next_date.month
        # CORRECTED: Removed .astype(int) here as .week already returns an int.
        X_predict_row_df.loc[0, 'WeekOfYear'] = next_date.isocalendar().week
        X_predict_row_df.loc[0, 'DayOfYear'] = next_date.dayofyear
        X_predict_row_df.loc[0, 'Quarter'] = next_date.quarter
        X_predict_row_df.loc[0, 'Year'] = next_date.year 

        # Populate lagged features using the `demand_history_for_lags` list
        for lag in lags:
            lag_feature_name = f'Demand_Lag_{lag}'
            if lag_feature_name in feature_names: # Check if this lag was used in training
                if len(demand_history_for_lags) >= lag:
                    X_predict_row_df.loc[0, lag_feature_name] = demand_history_for_lags[-lag]
                else:
                    # If historical data isn't long enough for this specific lag,
                    # fill with a fallback (e.g., 0 or mean from training data)
                    # This ensures no NaNs for prediction input.
                    X_predict_row_df.loc[0, lag_feature_name] = 0 
        
        # Ensure all columns are converted to the correct numeric type (float for RandomForest)
        X_predict_row_df = X_predict_row_df.astype(float) 

        # Predict
        predicted_demand = model.predict(X_predict_row_df)[0]
        
        # Append prediction to forecast_df and to demand_history_for_lags for next iteration
        forecast_df.loc[len(forecast_df)] = [next_date, max(0, predicted_demand)]
        demand_history_for_lags.append(max(0, predicted_demand))

        # Move to the next day
        next_date += pd.Timedelta(days=1)
    
    forecast_df['Forecasted_Demand'] = forecast_df['Forecasted_Demand'].round(0).astype(int)
    return forecast_df

# --- Streamlit App UI ---

st.sidebar.header("Historical Data Generation Parameters")
start_date = st.sidebar.date_input("Start Date of Historical Data", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date of Historical Data", datetime.date(2024, 12, 31))

st.sidebar.subheader("Demand Patterns")
base_demand = st.sidebar.slider("Base Daily Demand", 50, 500, 150, step=10)
trend_strength = st.sidebar.slider("Annual Trend Strength", -20, 50, 10)
weekly_seasonality_amplitude = st.sidebar.slider("Weekly Seasonality Amplitude", 0, 100, 30)
monthly_seasonality_amplitude = st.sidebar.slider("Monthly Seasonality Amplitude (Day of Month)", 0, 100, 20)
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

# Train Model
model, mae, rmse, test_indices, y_test_actual, y_test_pred, feature_names = train_forecasting_model(
    historical_demand_df, lags=lags_to_use, target_column='Demand'
)

if model is None: # Handle cases where training failed (e.g., not enough data)
    st.error("Model training failed. This can happen if the selected historical data range is too short or if the lags chosen are too large, resulting in insufficient data for training after feature creation. Please adjust start/end dates or reduce the number/size of lags.")
else:
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
    # Pass feature_names to forecasting function
    future_forecast_df = forecast_future_demand(model, historical_demand_df, forecast_horizon_days, lags=lags_to_use, feature_names=feature_names)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(historical_demand_df['Date'], historical_demand_df['Demand'], label='Historical Demand', color='dodgerblue')
    ax3.plot(future_forecast_df['Date'], future_forecast_df['Forecasted_Demand'], label='Forecasted Demand', color='red', linestyle='--')
    ax3.axvline(x=historical_demand_df['Date'].max(), color='gray', linestyle=':', label='Forecast Start')
    ax3.set_title('Historical Demand vs. Future Forecast')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Demand Units')
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("4. Raw Forecast Data (First 10 Days)")
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
