import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import random
import time # Import time for dynamic seed

# Set Streamlit page configuration
st.set_page_config(page_title="SupplyChain.ai — Supplier Risk Profiling", layout="wide")
st.title("🛡️ AI-Powered Supplier Performance & Risk Profiling")

# ---------------------------------------------------------
# Data Generation (Simulated)
# ---------------------------------------------------------

@st.cache_data
def generate_supplier_data(num_suppliers: int, seed: int = None) -> pd.DataFrame:
    """Generates simulated supplier performance data with some anomalies."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else: # If no seed, ensure true randomness by re-seeding with current time
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))


    data = []
    supplier_categories = ["Raw Material", "Component", "Logistics", "Packaging", "Service"]

    for i in range(num_suppliers):
        supplier_id = f"SUP-{i+1:04d}"
        category = random.choice(supplier_categories)

        # Base performance metrics (generally good)
        lead_time_days = max(5, int(np.random.normal(15, 5))) # Days
        on_time_delivery_rate = min(100, max(70, np.random.normal(95, 5))) # %
        quality_defect_rate = min(10, max(0.1, np.random.normal(1, 0.5))) # %
        communication_score = min(5, max(1, np.random.normal(4.5, 0.5))) # 1-5 scale
        financial_stability_score = min(5, max(1, np.random.normal(4, 0.7))) # 1-5 scale

        # Introduce anomalies for a subset of suppliers
        if random.random() < 0.15:  # 15% chance of being an "at-risk" supplier
            risk_type = random.choice(["high_lt", "low_otd", "high_defect", "poor_comm", "fin_risk"])
            if risk_type == "high_lt":
                lead_time_days = max(30, int(np.random.normal(40, 10)))
            elif risk_type == "low_otd":
                on_time_delivery_rate = min(80, max(40, np.random.normal(60, 10)))
            elif risk_type == "high_defect":
                quality_defect_rate = min(20, max(5, np.random.normal(10, 5)))
            elif risk_type == "poor_comm":
                communication_score = min(3, max(1, np.random.normal(2, 0.8)))
            elif risk_type == "fin_risk":
                financial_stability_score = min(3, max(1, np.random.normal(2.5, 1)))

        data.append({
            "SupplierID": supplier_id,
            "Category": category,
            "LeadTimeDays": lead_time_days,
            "OnTimeDeliveryRate": on_time_delivery_rate,
            "QualityDefectRate": quality_defect_rate,
            "CommunicationScore": communication_score,
            "FinancialStabilityScore": financial_stability_score
        })

    df = pd.DataFrame(data)
    return df

# ---------------------------------------------------------
# Anomaly Detection & Risk Scoring
# ---------------------------------------------------------

@st.cache_data
def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Isolation Forest for anomaly detection and calculates a composite risk score.
    """
    df_copy = df.copy()

    # Features for anomaly detection
    features = [
        "LeadTimeDays",
        "OnTimeDeliveryRate",
        "QualityDefectRate",
        "CommunicationScore",
        "FinancialStabilityScore"
    ]

    # Invert scores where lower is better (e.g., lower OTD is worse, so invert for risk)
    df_copy['Inv_OnTimeDeliveryRate'] = 100 - df_copy['OnTimeDeliveryRate']
    df_copy['Inv_CommunicationScore'] = 5 - df_copy['CommunicationScore']
    df_copy['Inv_FinancialStabilityScore'] = 5 - df_copy['FinancialStabilityScore']

    # Select features for Isolation Forest (using inverted scores where applicable)
    iso_features = [
        "LeadTimeDays",
        "Inv_OnTimeDeliveryRate",
        "QualityDefectRate",
        "Inv_CommunicationScore",
        "Inv_FinancialStabilityScore"
    ]

    # Scale features for Isolation Forest
    scaler = MinMaxScaler()
    df_copy[iso_features] = scaler.fit_transform(df_copy[iso_features])

    # Isolation Forest for anomaly detection
    # contamination: proportion of outliers in the data set. Adjust based on expected risk level.
    model = IsolationForest(random_state=42, contamination=0.15)
    model.fit(df_copy[iso_features])
    
    # Anomaly score: lower score indicates higher anomaly (outlier)
    df_copy['AnomalyScore'] = model.decision_function(df_copy[iso_features])
    
    # Convert anomaly score to a risk scale (higher is riskier)
    # Normalize AnomalyScore to be between 0 and 1, then invert
    min_score = df_copy['AnomalyScore'].min()
    max_score = df_copy['AnomalyScore'].max()
    df_copy['NormalizedAnomalyScore'] = (df_copy['AnomalyScore'] - min_score) / (max_score - min_score)
    df_copy['RiskFromAnomaly'] = 1 - df_copy['NormalizedAnomalyScore'] # Higher value = higher risk

    # Define simple business rules for additional risk factors (weighted)
    df_copy['RuleBasedRisk'] = 0
    df_copy.loc[df_copy['OnTimeDeliveryRate'] < 85, 'RuleBasedRisk'] += 0.3
    df_copy.loc[df_copy['QualityDefectRate'] > 5, 'RuleBasedRisk'] += 0.4
    df_copy.loc[df_copy['CommunicationScore'] < 3, 'RuleBasedRisk'] += 0.2
    df_copy.loc[df_copy['FinancialStabilityScore'] < 2.5, 'RuleBasedRisk'] += 0.5
    df_copy.loc[df_copy['LeadTimeDays'] > 30, 'RuleBasedRisk'] += 0.2

    # Combine anomaly score and rule-based risk into a composite RiskScore
    # Weights can be tuned based on business priorities
    df_copy['CompositeRiskScore'] = (df_copy['RiskFromAnomaly'] * 0.6) + (df_copy['RuleBasedRisk'] * 0.4)
    
    # Normalize CompositeRiskScore to a 0-100 scale for easier interpretation
    scaler_final = MinMaxScaler(feature_range=(0, 100))
    df_copy['FinalRiskScore'] = scaler_final.fit_transform(df_copy[['CompositeRiskScore']])

    # Categorize risk levels
    bins = [0, 30, 60, 100]
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    df_copy['RiskLevel'] = pd.cut(df_copy['FinalRiskScore'], bins=bins, labels=labels, right=False)

    return df_copy.drop(columns=['Inv_OnTimeDeliveryRate', 'Inv_CommunicationScore',
                                 'Inv_FinancialStabilityScore', 'AnomalyScore',
                                 'NormalizedAnomalyScore', 'RuleBasedRisk', 'CompositeRiskScore'])

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_suppliers = st.slider("Number of Suppliers", 50, 500, 100, 10)
    # Changed default value to None for true randomization on refresh, but allows user to set a seed for reproducibility
    data_seed = st.number_input("Data Random Seed (set to 0 for new data each refresh)", value=None, step=1, help="Enter a number for reproducible data, or leave blank/set to 0 for new data on each refresh.")
    
    st.markdown("---")
    st.markdown("### Risk Thresholds")
    low_risk_threshold = st.slider("Low Risk Max Score", 0, 100, 30, 5)
    medium_risk_threshold = st.slider("Medium Risk Max Score", 0, 100, 60, 5)
    
    # Update risk categorization based on slider
    @st.cache_data
    def update_risk_levels(df, low_thresh, med_thresh):
        df_updated = df.copy()
        bins = [0, low_thresh, med_thresh, 100]
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        df_updated['RiskLevel'] = pd.cut(df_updated['FinalRiskScore'], bins=bins, labels=labels, right=False)
        return df_updated

# Generate and process data
# If data_seed is None or 0, pass None to generate_supplier_data to ensure new data each time
current_seed = data_seed if data_seed is not None and data_seed != 0 else None
suppliers_df = generate_supplier_data(num_suppliers, current_seed)
suppliers_df_risked = calculate_risk_scores(suppliers_df)
suppliers_df_risked = update_risk_levels(suppliers_df_risked, low_risk_threshold, medium_risk_threshold)


st.subheader("📊 Supplier Risk Overview")

# KPIs
total_suppliers = len(suppliers_df_risked)
high_risk_suppliers = suppliers_df_risked[suppliers_df_risked['RiskLevel'] == 'High Risk'].shape[0]
medium_risk_suppliers = suppliers_df_risked[suppliers_df_risked['RiskLevel'] == 'Medium Risk'].shape[0]
low_risk_suppliers = suppliers_df_risked[suppliers_df_risked['RiskLevel'] == 'Low Risk'].shape[0]

avg_otd = suppliers_df_risked['OnTimeDeliveryRate'].mean()
avg_defect = suppliers_df_risked['QualityDefectRate'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Suppliers", total_suppliers)
with col2:
    st.metric("High Risk", high_risk_suppliers, delta=f"{(high_risk_suppliers/total_suppliers)*100:.1f}%")
with col3:
    st.metric("Medium Risk", medium_risk_suppliers, delta=f"{(medium_risk_suppliers/total_suppliers)*100:.1f}%")
with col4:
    st.metric("Avg. On-Time Delivery", f"{avg_otd:.1f}%")
with col5:
    st.metric("Avg. Quality Defects", f"{avg_defect:.1f}%")

st.markdown("---")

# Visualizations
st.subheader("📈 Risk Distribution & Performance")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("#### Risk Level Distribution")
    risk_counts = suppliers_df_risked['RiskLevel'].value_counts().reset_index()
    risk_counts.columns = ['RiskLevel', 'Count']
    # Define a specific order for the categories
    order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts['RiskLevel'] = pd.Categorical(risk_counts['RiskLevel'], categories=order, ordered=True)
    risk_counts = risk_counts.sort_values('RiskLevel')

    chart_risk_dist = alt.Chart(risk_counts).mark_bar().encode(
        x=alt.X('RiskLevel:N', sort=order, title="Risk Level"),
        y=alt.Y('Count:Q', title="Number of Suppliers"),
        color=alt.Color('RiskLevel:N', scale=alt.Scale(domain=order, range=['#2ca02c', '#ff7f0e', '#d62728']), legend=None),
        tooltip=['RiskLevel', 'Count']
    ).properties(
        height=300
    )
    st.altair_chart(chart_risk_dist, use_container_width=True)

with chart_col2:
    st.write("#### OTD vs. Quality Defect Rate by Risk")
    chart_scatter = alt.Chart(suppliers_df_risked).mark_circle(size=60).encode(
        x=alt.X('OnTimeDeliveryRate:Q', title="On-Time Delivery Rate (%)"),
        y=alt.Y('QualityDefectRate:Q', title="Quality Defect Rate (%)"),
        color=alt.Color('RiskLevel:N', scale=alt.Scale(domain=order, range=['#2ca02c', '#ff7f0e', '#d62728'])),
        tooltip=[
            'SupplierID', 'Category', 'LeadTimeDays', 'OnTimeDeliveryRate',
            'QualityDefectRate', 'CommunicationScore', 'FinancialStabilityScore',
            alt.Tooltip('FinalRiskScore', format=".1f", title="Risk Score"), 'RiskLevel'
        ]
    ).properties(
        height=300
    ).interactive() # Enable zooming and panning
    st.altair_chart(chart_scatter, use_container_width=True)

st.markdown("---")

# Detailed Supplier Table
st.subheader("📋 Detailed Supplier Performance & Risk")

# Filter options
category_filter = st.multiselect(
    "Filter by Category",
    options=suppliers_df_risked['Category'].unique(),
    default=suppliers_df_risked['Category'].unique()
)

risk_level_filter = st.multiselect(
    "Filter by Risk Level",
    options=["Low Risk", "Medium Risk", "High Risk"],
    default=["Low Risk", "Medium Risk", "High Risk"]
)

filtered_df = suppliers_df_risked[
    (suppliers_df_risked['Category'].isin(category_filter)) &
    (suppliers_df_risked['RiskLevel'].isin(risk_level_filter))
]

st.dataframe(filtered_df.sort_values(by='FinalRiskScore', ascending=False), use_container_width=True)

# ---------------------------------------------------------
# Business Context & Highlights
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

### **Problem Statement: The Challenge of Reactive Supplier Risk**
In my 20+ years in supply chain and logistics, a recurring and costly problem has been the **reactive nature of supplier risk management.** We often found ourselves scrambling to address issues – like unexpected late deliveries, sudden quality defects, or even a supplier's financial distress – *after* they had already impacted our production, customer commitments, or bottom line. Manually monitoring a vast network of suppliers is simply not scalable, leading to blind spots and significant operational disruptions.

### **Why This Solution Matters: Proactive Resilience & Strategic Advantage**
This AI-powered prototype is designed to transform that reactive approach into a **proactive, intelligent risk management system.** As a supply chain practitioner, I see immense value in its ability to:
- **Spot Risks Early:** It acts as an early warning system, identifying potential issues before they cause widespread disruption.
- **Build Supply Chain Resilience:** By mitigating risks proactively, we can prevent costly delays and maintain continuity of supply.
- **Inform Sourcing Decisions:** The insights gained allow for more strategic choices in supplier selection, diversification, and relationship management.
- **Boost Operational Efficiency:** It frees up valuable time spent on manual monitoring, allowing teams to focus on strategic initiatives and problem-solving rather than firefighting.
- **Drive Tangible Cost Savings:** Avoiding disruptions, quality issues, and emergency sourcing directly translates into significant cost reductions.

### **How It Works: My Approach to AI-Driven Risk Assessment**
My goal was to build a system that thinks like a seasoned supply chain professional, but with the speed and analytical power of AI. Here's how I approached it:

1.  **Gathering & Preparing the Data:** We start by bringing together key supplier performance data – things like their historical **Lead Time**, **On-Time Delivery Rate**, **Quality Defect Rate**, and even qualitative aspects like **Communication Score** and **Financial Stability Score**. Using **Pandas**, a powerful data manipulation tool in **Python**, I prepare this data, ensuring it's clean and structured for analysis. This also involves some smart transformations, like inverting certain metrics so that a lower 'On-Time Delivery' percentage correctly signals higher risk.
2.  **AI-Powered Anomaly Detection:** The core intelligence comes from an AI technique called **Anomaly Detection**. I've employed the **Isolation Forest** algorithm from **Scikit-learn**, a leading **Python** machine learning library. This algorithm is incredibly effective at automatically spotting unusual patterns or 'outliers' in the data – suppliers whose performance deviates significantly from what's considered 'normal.' This helps us uncover hidden risks that might be missed by simple thresholds.
3.  **Integrating My Business Rules:** Based on my years of experience, I know that certain performance levels are inherently risky, even if they aren't statistical anomalies. So, I've embedded these **business rules directly into the Python logic** of the system. For example, if a supplier's on-time delivery consistently dips below 85%, or their defect rate creeps above 5%, these factors are explicitly weighted into their risk profile. This ensures the AI aligns with critical operational realities.
4.  **Calculating a Comprehensive Risk Score:** All these insights – from the AI's anomaly detection and my defined business rules – are combined. Using **Python's** numerical capabilities and scaling techniques, we generate a single, easy-to-understand **Composite Risk Score** for each supplier, ranging from 0 to 100. This provides a holistic view, where a higher score means higher risk.
5.  **Interactive Dashboard for Action:** To make these powerful insights accessible and actionable for supply chain managers, I've built an intuitive, interactive dashboard using **Streamlit**. This **Python** framework allows for rapid web application development. I use **Altair** for dynamic visualizations, which let users quickly see risk distributions, identify high-risk suppliers at a glance, and filter the data by category or risk level to prioritize their focus.

### **Key Performance Indicators (KPIs) This Solution Improves**
- **Reduced High-Risk Supplier Count:** Direct measure of proactive risk mitigation.
- **Enhanced Supplier Performance Visibility:** Clear, data-driven insights into OTD, Quality, and Lead Times across the network.
- **Increased Operational Efficiency:** Less time spent on reactive problem-solving, more on strategic supplier management.
- **Improved Supply Chain Resilience:** Quantifiable reduction in potential disruption events.

### **Tech Stack: The Tools I Used**
- **Python:** The foundational programming language for all logic and analytics.
- **Pandas:** Essential for data preparation, cleaning, and manipulation.
- **Scikit-learn:** For implementing robust machine learning algorithms, specifically Isolation Forest for anomaly detection.
- **Altair:** For creating clear, interactive, and insightful data visualizations.
- **Streamlit:** For rapidly building and deploying the interactive web-based dashboard.

### **Next Steps & Strategic Vision**
This prototype lays a strong foundation. Looking forward, the next strategic steps would involve:
- **Real-time Data Integration:** Connecting directly to ERP, SRM, or external data sources (e.g., news feeds for geopolitical risk, financial APIs) for live, continuous risk monitoring.
- **Predictive & Prescriptive Actions:** Moving beyond just identifying risk to recommending specific mitigation actions (e.g., "Increase safety stock for products from this supplier," "Engage in a supplier development program").
- **Time-Series Analysis:** Implementing models that detect anomalies in performance *trends* over time, not just static snapshots.
- **Explainable AI (XAI):** Providing even more transparent explanations for *why* a specific supplier is flagged as high-risk, building trust and facilitating quicker action.
- **Network Resilience Simulation:** Simulating the ripple effect of a supplier disruption across the entire supply chain network to assess overall vulnerability.
- **Continuous Learning Loop:** Establishing a feedback mechanism where user actions and outcomes refine the AI model over time.
""")
