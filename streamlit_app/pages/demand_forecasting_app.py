# --- Forecasting Function ---
# MODIFIED: Implemented robust iterative forecasting for lagged features and column consistency
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
        
        # --- FIX IS HERE ---
        X_predict_row_df.loc[0, 'WeekOfYear'] = next_date.isocalendar().week # Removed .astype(int)
        # --- END FIX ---
        
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
