import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import random

# Set Streamlit page configuration
st.set_page_config(page_title="SupplyChain.ai — Supplier Risk Profiling", layout="wide")
st.title("🛡️ AI-Powered Supplier Performance & Risk Profiling")

# ---------------------------------------------------------
# Data Generation (Simulated)
# ---------------------------------------------------------

@st.cache_data
def generate_supplier_data(num_suppliers: int, seed: int = 42) -> pd.DataFrame:
    """Generates simulated supplier performance data with some anomalies."""
    np.random.seed(seed)
    random.seed(seed)

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
    data_seed = st.number_input("Data Random Seed", value=42, step=1)
    
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
suppliers_df = generate_supplier_data(num_suppliers, data_seed)
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

### **Problem Statement**
In complex global supply chains, managing supplier risk is paramount. Disruptions due to poor supplier performance (e.g., late deliveries, quality issues, financial instability) can lead to significant operational delays, increased costs, reputational damage, and lost revenue. Traditional manual monitoring methods are often reactive, time-consuming, and prone to human error, making it difficult to proactively identify and mitigate emerging risks.

### **Why This Solution Matters (Business Value)**
This AI-powered prototype demonstrates a proactive approach to supplier risk management. By leveraging machine learning and data analytics, it allows supply chain professionals to:
- **Proactively Identify At-Risk Suppliers:** Move from reactive firefighting to predictive risk identification.
- **Improve Supply Chain Resilience:** Mitigate potential disruptions before they impact operations.
- **Optimize Sourcing Decisions:** Inform strategic choices about supplier relationships and diversification.
- **Enhance Operational Efficiency:** Reduce time spent on manual risk assessments and supplier performance monitoring.
- **Drive Cost Savings:** Avoid costs associated with delays, quality issues, and emergency sourcing.

### **How It Works (Simplified ML Explanation, Referencing Tools)**
This application uses a combination of statistical analysis and a machine learning model to identify anomalies in supplier performance data.

1.  **Data Ingestion & Preparation (using Pandas):** We start by gathering key supplier performance data – things like Lead Time, On-Time Delivery Rate, and Quality Defect Rate. Using **Pandas**, a powerful data manipulation tool in **Python**, we then prepare this data, ensuring it's clean and structured for analysis. This step also involves transforming some metrics (e.g., inverting 'On-Time Delivery' so lower percentages indicate higher risk) to ensure all factors contribute consistently to the overall risk assessment.
2.  **AI-Powered Anomaly Detection (using Scikit-learn's Isolation Forest):** The core of the solution lies in an AI technique called **Anomaly Detection**. Specifically, we employ the **Isolation Forest** algorithm from **Scikit-learn**, a leading **Python** machine learning library. This algorithm is excellent at automatically spotting unusual patterns or 'outliers' in the data – suppliers whose performance deviates significantly from the norm, potentially signaling a hidden risk.
3.  **Integrating Business Rules (Python Logic):** Beyond just statistical anomalies, our experience tells us that certain performance thresholds are inherently risky. We've incorporated these **business rules directly into the Python logic** of the system. For instance, if a supplier's on-time delivery falls below a certain percentage, or their defect rate exceeds a threshold, these factors contribute to their risk profile, even if they aren't extreme statistical outliers.
4.  **Calculating a Comprehensive Risk Score (Python & Scaling):** All these factors – the AI-detected anomalies and our business-defined risk triggers – are then combined. Through **Python's** numerical capabilities and scaling techniques, we generate a single, easy-to-understand **Composite Risk Score** for each supplier, ranging from 0 to 100. A higher score means higher risk.
5.  **Interactive Dashboard & Visualization (Streamlit & Altair):** Finally, to make these insights actionable for supply chain managers, we present everything in an interactive dashboard built with **Streamlit**. This **Python** framework allows us to quickly create web applications. We use **Altair** for dynamic visualizations, enabling users to explore risk distributions, identify high-risk suppliers at a glance, and filter the data to focus on specific categories or risk levels.

### **Key Performance Indicators (KPIs) Improved**
- **Proactive Risk Identification:** Number of high-risk suppliers flagged.
- **Supplier Performance Visibility:** Clear overview of OTD, Quality, Lead Times.
- **Operational Efficiency:** Reduced manual effort in risk assessment.

### **Tech Stack**
- **Python:** Core programming language.
- **Pandas:** Data manipulation and analysis.
- **Scikit-learn:** For the Isolation Forest machine learning model.
- **Altair:** For interactive data visualizations (scatter plots, bar charts).
- **Streamlit:** For building the interactive web application and dashboard.

### **Next Steps & Advanced Considerations**
- **Real-time Data Integration:** Connect to ERP, SRM, or external data sources (e.g., news feeds for geopolitical risk, financial APIs) for live updates.
- **Time-Series Anomaly Detection:** Implement models that detect anomalies in *trends* over time (e.g., a gradual decline in OTD).
- **Prescriptive Analytics:** Beyond identifying risk, recommend specific mitigation actions (e.g., "Increase safety stock for products from this supplier," "Initiate supplier development program").
- **Predictive Maintenance for Assets:** Apply similar anomaly detection to predict equipment failures in warehouses or fleets.
- **Network Resilience Simulation:** Simulate the impact of a supplier disruption on the entire supply chain network.
- **Explainable AI (XAI):** Provide more detailed explanations for *why* a specific supplier is flagged as high-risk (e.g., using SHAP or LIME values).
- **Feedback Loop:** Allow users to provide feedback on risk classifications to continuously improve the model.
""")
