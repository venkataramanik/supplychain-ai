import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import random
import time # For dynamic seed

# Set Streamlit page configuration
st.set_page_config(page_title="SupplyChain.ai — Demand Volatility", layout="wide")
st.title("📊 AI-Powered Demand Volatility & Predictability Assessment")

# ---------------------------------------------------------
# Data Generation (Simulated)
# ---------------------------------------------------------

@st.cache_data
def generate_demand_data(num_skus: int, num_weeks: int, seed: int = None) -> pd.DataFrame:
    """Generates simulated historical demand data for multiple SKUs."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))

    all_data = []
    product_types = ["Electronics", "Apparel", "Home Goods", "Consumables", "Industrial"]

    for i in range(num_skus):
        sku_id = f"SKU-{i+1:04d}"
        product_type = random.choice(product_types)
        
        # Base demand and trend
        base_demand = np.random.randint(50, 500)
        weekly_trend = np.random.uniform(-2, 5)
        
        # Seasonality (e.g., yearly cycle)
        seasonality_amplitude = base_demand * np.random.uniform(0.1, 0.4)
        
        # Promotional activity (random spikes)
        promo_frequency = random.choice([0, 0, 0, 1, 2]) # More likely no promo, some with 1-2
        promo_weeks = random.sample(range(num_weeks), promo_frequency) if promo_frequency > 0 else []
        
        # External factor impact (e.g., economic indicator)
        external_factor_impact = np.random.uniform(0.8, 1.2) # Multiplier

        sku_data = []
        for week in range(num_weeks):
            # Base + Trend
            demand = base_demand + (weekly_trend * week)
            
            # Add seasonality (simple sine wave for yearly cycle)
            demand += seasonality_amplitude * np.sin(2 * np.pi * (week % 52) / 52)
            
            # Add promotional spikes
            if week in promo_weeks:
                demand += base_demand * np.random.uniform(0.3, 0.8) # 30-80% uplift
            
            # Add noise and external factor
            demand *= external_factor_impact + np.random.normal(0, 0.05) # 5% noise
            demand = max(0, int(demand)) # Ensure non-negative demand

            # Simulate a simple 'promotion' flag
            is_promotion = 1 if week in promo_weeks else 0

            all_data.append({
                "SKUID": sku_id,
                "ProductType": product_type,
                "Week": week + 1,
                "Demand": demand,
                "IsPromotion": is_promotion,
                "Month": (week % 52 // 4) + 1 # Simple month approximation
            })
    
    df = pd.DataFrame(all_data)
    return df

# ---------------------------------------------------------
# Demand Volatility & Predictability Analysis
# ---------------------------------------------------------

@st.cache_data
def analyze_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates volatility metrics and assesses predictability."""
    
    # Calculate Coefficient of Variation (CV) for each SKU
    sku_summary = df.groupby('SKUID').agg(
        AvgDemand=('Demand', 'mean'),
        StdDevDemand=('Demand', 'std'),
        TotalDemand=('Demand', 'sum'),
        ProductType=('ProductType', 'first')
    ).reset_index()
    
    sku_summary['CV'] = (sku_summary['StdDevDemand'] / sku_summary['AvgDemand']).fillna(0) # Handle zero demand
    
    # Predictability Score (lower CV = higher predictability)
    # Normalize CV to a 0-1 scale and invert for predictability (higher is better)
    if sku_summary['CV'].max() > 0:
        sku_summary['NormalizedCV'] = sku_summary['CV'] / sku_summary['CV'].max()
        sku_summary['PredictabilityScore'] = (1 - sku_summary['NormalizedCV']) * 100 # 0-100 scale
    else:
        sku_summary['PredictabilityScore'] = 100 # If CV is 0, perfectly predictable

    # Categorize predictability
    bins = [0, 40, 70, 101] # 101 to include 100
    labels = ["Low Predictability", "Medium Predictability", "High Predictability"]
    sku_summary['PredictabilityLevel'] = pd.cut(sku_summary['PredictabilityScore'], bins=bins, labels=labels, right=False)

    # Simple Linear Regression for feature importance (predictability drivers)
    # This is a simplified model to demonstrate feature influence, not a robust forecast.
    feature_importance_data = []
    for sku_id in sku_summary['SKUID'].unique():
        sku_df = df[df['SKUID'] == sku_id].copy()
        
        # Create features: Lagged demand, Week of Year (seasonality), IsPromotion
        sku_df['Lag1Demand'] = sku_df['Demand'].shift(1)
        sku_df['WeekOfYear'] = sku_df['Week'] % 52 # Simple seasonality proxy
        
        # Drop rows with NaN from lagging
        sku_df.dropna(inplace=True)

        if len(sku_df) > 1: # Need at least 2 data points for regression
            X = sku_df[['Lag1Demand', 'WeekOfYear', 'IsPromotion']]
            y = sku_df['Demand']

            # Scale features for consistent coefficient interpretation (optional but good practice)
            scaler_X = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Store coefficients (feature importance)
            for feature, coef in zip(X.columns, model.coef_):
                feature_importance_data.append({
                    "SKUID": sku_id,
                    "Feature": feature,
                    "Coefficient": coef
                })
        else:
            # Handle SKUs with too little data for regression
            for feature in ['Lag1Demand', 'WeekOfYear', 'IsPromotion']:
                 feature_importance_data.append({
                    "SKUID": sku_id,
                    "Feature": feature,
                    "Coefficient": 0 # No influence if no data
                })


    feature_importance_df = pd.DataFrame(feature_importance_data)
    
    return sku_summary, feature_importance_df

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_skus = st.slider("Number of SKUs", 10, 100, 30, 5)
    num_weeks = st.slider("Historical Weeks", 52, 208, 104, 52) # 1-4 years
    
    # Data seed control (0 for truly random, any number for reproducible)
    data_seed = st.number_input("Data Random Seed (0 for new data each refresh)", value=0, step=1, help="Enter a number for reproducible data, or set to 0 for new data on each refresh.")
    
    st.markdown("---")
    st.markdown("### Predictability Thresholds")
    low_pred_threshold = st.slider("Low Predictability Max Score", 0, 100, 40, 5)
    medium_pred_threshold = st.slider("Medium Predictability Max Score", 0, 100, 70, 5)

# Generate and analyze data
current_seed = data_seed if data_seed != 0 else None
demand_df = generate_demand_data(num_skus, num_weeks, current_seed)
sku_summary_df, feature_importance_df = analyze_demand(demand_df)

# Update predictability levels based on slider
@st.cache_data
def update_predictability_levels(df, low_thresh, med_thresh):
    df_updated = df.copy()
    bins = [0, low_thresh, med_thresh, 101]
    labels = ["Low Predictability", "Medium Predictability", "High Predictability"]
    df_updated['PredictabilityLevel'] = pd.cut(df_updated['PredictabilityScore'], bins=bins, labels=labels, right=False)
    return df_updated

sku_summary_df = update_predictability_levels(sku_summary_df, low_pred_threshold, medium_pred_threshold)


st.subheader("📊 Demand Predictability Overview")

# KPIs
total_skus = len(sku_summary_df)
high_pred_skus = sku_summary_df[sku_summary_df['PredictabilityLevel'] == 'High Predictability'].shape[0]
medium_pred_skus = sku_summary_df[sku_summary_df['PredictabilityLevel'] == 'Medium Predictability'].shape[0]
low_pred_skus = sku_summary_df[sku_summary_df['PredictabilityLevel'] == 'Low Predictability'].shape[0]

avg_cv = sku_summary_df['CV'].mean()
avg_pred_score = sku_summary_df['PredictabilityScore'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total SKUs", total_skus)
with col2:
    st.metric("High Predictability", high_pred_skus, delta=f"{(high_pred_skus/total_skus)*100:.1f}%")
with col3:
    st.metric("Low Predictability", low_pred_skus, delta=f"{(low_pred_skus/total_skus)*100:.1f}%")
with col4:
    st.metric("Avg. CV (Volatility)", f"{avg_cv:.2f}")
with col5:
    st.metric("Avg. Predictability Score", f"{avg_pred_score:.1f}")

st.markdown("---")

# Visualizations
st.subheader("📈 Predictability Distribution & Drivers")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("#### Predictability Level Distribution")
    pred_counts = sku_summary_df['PredictabilityLevel'].value_counts().reset_index()
    pred_counts.columns = ['PredictabilityLevel', 'Count']
    # Define a specific order for the categories
    order = ["High Predictability", "Medium Predictability", "Low Predictability"]
    pred_counts['PredictabilityLevel'] = pd.Categorical(pred_counts['PredictabilityLevel'], categories=order, ordered=True)
    pred_counts = pred_counts.sort_values('PredictabilityLevel')

    chart_pred_dist = alt.Chart(pred_counts).mark_bar().encode(
        x=alt.X('PredictabilityLevel:N', sort=order, title="Predictability Level"),
        y=alt.Y('Count:Q', title="Number of SKUs"),
        color=alt.Color('PredictabilityLevel:N', scale=alt.Scale(domain=order, range=['#2ca02c', '#ff7f0e', '#d62728']), legend=None),
        tooltip=['PredictabilityLevel', 'Count']
    ).properties(
        height=300
    )
    st.altair_chart(chart_pred_dist, use_container_width=True)

with chart_col2:
    st.write("#### Average Influence of Demand Drivers")
    # Average absolute coefficients across all SKUs
    avg_feature_importance = feature_importance_df.groupby('Feature')['Coefficient'].mean().reset_index()
    avg_feature_importance['AbsCoefficient'] = avg_feature_importance['Coefficient'].abs()
    avg_feature_importance = avg_feature_importance.sort_values('AbsCoefficient', ascending=False)

    chart_features = alt.Chart(avg_feature_importance).mark_bar().encode(
        x=alt.X('AbsCoefficient:Q', title="Average Influence (Absolute Coefficient)"),
        y=alt.Y('Feature:N', sort='-x', title="Demand Driver"),
        tooltip=['Feature', alt.Tooltip('Coefficient', format=".2f", title="Avg. Coefficient")]
    ).properties(
        height=300
    )
    st.altair_chart(chart_features, use_container_width=True)

st.markdown("---")

# Detailed SKU Predictability Table
st.subheader("📋 Detailed SKU Predictability & Volatility")

# Filter options
product_type_filter = st.multiselect(
    "Filter by Product Type",
    options=sku_summary_df['ProductType'].unique(),
    default=sku_summary_df['ProductType'].unique()
)

predictability_level_filter = st.multiselect(
    "Filter by Predictability Level",
    options=["High Predictability", "Medium Predictability", "Low Predictability"],
    default=["High Predictability", "Medium Predictability", "Low Predictability"]
)

filtered_sku_summary_df = sku_summary_df[
    (sku_summary_df['ProductType'].isin(product_type_filter)) &
    (sku_summary_df['PredictabilityLevel'].isin(predictability_level_filter))
]

st.dataframe(filtered_sku_summary_df.sort_values(by='PredictabilityScore', ascending=True), use_container_width=True)

st.markdown("---")

# Individual SKU Demand Pattern Visualization
st.subheader("📈 Individual SKU Demand Pattern")
selected_sku = st.selectbox(
    "Select an SKU to view its demand pattern:",
    options=demand_df['SKUID'].unique()
)

if selected_sku:
    selected_sku_demand = demand_df[demand_df['SKUID'] == selected_sku]
    
    # Simple moving average for trend visualization
    selected_sku_demand['SMA_4_Week'] = selected_sku_demand['Demand'].rolling(window=4).mean()

    chart_sku_demand = alt.Chart(selected_sku_demand).mark_line().encode(
        x=alt.X('Week:Q', title="Week"),
        y=alt.Y('Demand:Q', title="Demand"),
        tooltip=['Week', 'Demand']
    ).properties(
        title=f"Demand Pattern for {selected_sku}"
    )

    chart_sku_sma = alt.Chart(selected_sku_demand).mark_line(color='orange', strokeDash=[5,5]).encode(
        x='Week:Q',
        y='SMA_4_Week:Q',
        tooltip=['Week', alt.Tooltip('SMA_4_Week', format=".1f", title="4-Week Avg")]
    )
    
    st.altair_chart(chart_sku_demand + chart_sku_sma, use_container_width=True)


# ---------------------------------------------------------
# Business Context & Highlights
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

### **Problem Statement: The Hidden Costs of Demand Volatility**
In my years leading supply chain operations, I've seen firsthand that not all demand is created equal. While we strive for accurate forecasts, the underlying **volatility and predictability of demand** vary wildly across products and customer segments. Treating all demand the same leads to significant inefficiencies:
- **High-volatility products:** Often result in excessive safety stock, increased holding costs, or frequent stockouts due to unpredictable spikes/drops.
- **Highly predictable products:** May be over-forecasted or managed with outdated methods, tying up capital unnecessarily.
- **Lack of insight:** Without understanding *why* demand behaves the way it does, our inventory, production, and logistics planning become reactive and inefficient.

### **Why This Solution Matters: Smarter Planning & Resource Allocation**
This AI-powered prototype provides a critical lens into demand dynamics. From a supply chain practitioner's perspective, this tool enables us to:
- **Optimize Inventory Strategies:** Tailor inventory policies (e.g., safety stock levels, reorder points) to the true predictability of each SKU, reducing costs and improving service.
- **Refine Production Planning:** Align production schedules more closely with predictable demand, and allocate flexible capacity or buffer stock for volatile items.
- **Enhance Sales & Operations Planning (S&OP):** Provide data-driven insights to S&OP discussions, highlighting where forecasting efforts should be intensified or where promotional strategies might be creating unnecessary volatility.
- **Improve Logistics Efficiency:** Better demand predictability leads to more stable transportation and warehousing needs, reducing expedited shipping and storage costs.

### **How It Works: My Approach to Unpacking Demand Data**
My goal was to build a system that not only tells us *what* demand is doing, but *why* it's behaving that way, using a blend of traditional analytics and AI:

1.  **Data Ingestion & Structuring (using Pandas):** We start with historical sales data for various SKUs over time. Using **Pandas**, a robust data manipulation library in **Python**, I prepare this data, ensuring it's clean and ready for analysis. This step is crucial for transforming raw sales figures into meaningful time-series data.
2.  **Volatility Measurement (Statistical Analysis):** To quantify how 'bumpy' demand is, I calculate the **Coefficient of Variation (CV)** for each SKU. This statistical measure, easily computed with **NumPy** in **Python**, tells us the ratio of standard deviation to the mean demand. A higher CV indicates greater volatility and lower predictability.
3.  **Predictability Assessment:** Based on the CV, I assign a **Predictability Score** to each SKU. This score is then used to categorize SKUs into "High," "Medium," or "Low Predictability" levels, allowing for quick segmentation and strategic grouping.
4.  **Identifying Demand Drivers (using Scikit-learn's Linear Regression):** To understand *what influences* demand, I apply a simple **Linear Regression model** from **Scikit-learn**, a key **Python** ML library. This model helps us see the average impact of factors like past demand (`Lag1Demand`), **seasonality** (`WeekOfYear`), and **promotional activities** (`IsPromotion`). While not a full-blown forecasting model, its primary purpose here is to **interpret the influence of these drivers** on demand, guiding our understanding of predictability.
5.  **Interactive Dashboard & Visualization (Streamlit & Altair):** All these insights are presented in an interactive dashboard built with **Streamlit**. This **Python** framework allows for rapid web application development. I use **Altair** for dynamic visualizations, enabling users to explore the overall predictability distribution, understand the average influence of demand drivers, and even drill down into the historical demand pattern of individual SKUs.

### **Key Performance Indicators (KPIs) This Solution Improves**
- **Reduced Inventory Holding Costs:** By aligning safety stock with true predictability.
- **Improved Forecast Accuracy:** By focusing efforts where they matter most.
- **Optimized Production Schedules:** Better alignment with demand patterns.
- **Enhanced S&OP Effectiveness:** Data-driven insights for strategic planning.

### **Tech Stack: The Tools I Used**
- **Python:** The core programming language for all logic and analytics.
- **Pandas:** Essential for data preparation, cleaning, and manipulation.
- **NumPy:** For numerical operations and statistical calculations.
- **Scikit-learn:** For implementing machine learning models (Linear Regression) to identify demand drivers.
- **Altair:** For creating clear, interactive, and insightful data visualizations.
- **Streamlit:** For rapidly building and deploying the interactive web-based dashboard.

### **Next Steps & Strategic Vision**
This prototype provides a strong analytical foundation. Looking forward, the next strategic steps would involve:
- **Advanced Forecasting Models:** Integrating more sophisticated time-series models (e.g., Prophet, ARIMA, XGBoost) for actual demand forecasting, building upon the predictability insights.
- **Prescriptive Inventory Recommendations:** Automatically suggesting optimal safety stock and reorder points based on calculated predictability and volatility.
- **Integration with External Data:** Incorporating real-world external factors like economic indicators, social media trends, or competitor data.
- **Scenario Planning:** Enabling 'what-if' analyses to understand the impact of various demand scenarios on the supply chain.
- **Root Cause Analysis:** Developing deeper analytics to pinpoint the exact reasons for high volatility in specific SKUs or segments.
""")
