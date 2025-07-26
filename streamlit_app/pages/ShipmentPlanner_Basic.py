import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rating & Mode Selection", layout="wide")
st.title("🚚 Rating & Mode Selection (Cost Engine)")

st.markdown("""
This pilot showcases **Parcel vs. LTL cost optimization** for multiple shipments.
Click **Solve** to calculate the cheapest mode for each shipment using a transparent cost model.
""")

# --- Generate 50 Random Shipments ---
np.random.seed(42)
origins = ["Atlanta, GA", "Dallas, TX", "Chicago, IL", "Denver, CO", "Seattle, WA",
           "Boston, MA", "Miami, FL", "San Jose, CA", "Phoenix, AZ", "Nashville, TN"]
destinations = ["New York, NY", "Houston, TX", "Los Angeles, CA", "Chicago, IL", "Atlanta, GA",
                "Miami, FL", "Seattle, WA", "Boston, MA", "Dallas, TX", "Denver, CO"]

shipments = pd.DataFrame({
    "ShipmentID": [f"SHP-{i:03d}" for i in range(1, 51)],
    "Origin": np.random.choice(origins, 50),
    "Destination": np.random.choice(destinations, 50),
    "Distance (miles)": np.random.randint(100, 3000, 50),
    "Weight (lbs)": np.random.randint(50, 1000, 50)
})

st.subheader("📦 Shipments Data (50 rows)")
st.dataframe(shipments, use_container_width=True)

# --- Solve Button ---
if st.button("Solve"):
    st.subheader("🔍 Cost Optimization Results")

    # Rate data
    modes = pd.DataFrame([
        {"Mode": "Parcel", "Base": 10, "Cost_per_Mile": 0.50, "Cost_per_Lb": 0.05},
        {"Mode": "LTL", "Base": 30, "Cost_per_Mile": 0.40, "Cost_per_Lb": 0.02}
    ])

    # Calculate cost for each shipment & mode
    results = []
    for _, row in shipments.iterrows():
        dist, wt = row["Distance (miles)"], row["Weight (lbs)"]
        for _, mode in modes.iterrows():
            total_cost = mode["Base"] + mode["Cost_per_Mile"] * dist + mode["Cost_per_Lb"] * wt
            results.append({
                "ShipmentID": row["ShipmentID"],
                "Origin": row["Origin"],
                "Destination": row["Destination"],
                "Mode": mode["Mode"],
                "Total Cost ($)": round(total_cost, 2)
            })
    results_df = pd.DataFrame(results)

    # Pick cheapest mode per shipment
    optimal = results_df.loc[results_df.groupby("ShipmentID")["Total Cost ($)"].idxmin()]

    st.markdown("**Optimized Modes (Cheapest per shipment):**")
    st.dataframe(optimal, use_container_width=True)

    # --- Cost Summary Chart ---
    st.subheader("📊 Total Cost Summary by Mode")
    cost_summary = results_df.groupby("Mode")["Total Cost ($)"].sum().reset_index()

    fig, ax = plt.subplots()
    ax.bar(cost_summary["Mode"], cost_summary["Total Cost ($)"], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Parcel vs LTL Total Cost (All Shipments)")
    st.pyplot(fig)

# --- Optional Custom Scenario ---
st.subheader("🧪 Custom Scenario (Optional)")
with st.form("custom_scenario"):
    weight = st.number_input("Weight (lbs)", min_value=10, max_value=5000, value=200)
    distance = st.number_input("Distance (miles)", min_value=10, max_value=3000, value=500)
    submitted = st.form_submit_button("Calculate Cost")

    if submitted:
        parcel_cost = 10 + 0.50 * distance + 0.05 * weight
        ltl_cost = 30 + 0.40 * distance + 0.02 * weight
        st.write(f"**Parcel Cost:** ${parcel_cost:.2f}")
        st.write(f"**LTL Cost:** ${ltl_cost:.2f}")
        st.success(f"Recommended Mode: {'Parcel' if parcel_cost < ltl_cost else 'LTL'}")

# --- Business Explanation ---
st.markdown("## 📝 How This Works")
st.markdown("""
1. **Shipment Data:** We simulate 50 shipments with realistic distances and weights.  
2. **Cost Model:** For each shipment, both **Parcel** and **LTL** costs are computed using:  
   - `Total Cost = Base + (Cost per Mile × Distance) + (Cost per Pound × Weight)`.  
3. **Mode Selection:** The cheapest mode is chosen per shipment, simulating a **rate-shopping engine**.  
4. **Analytics:** We aggregate total cost by mode and visualize it using a bar chart.  
5. **Custom Scenario:** Users can test "what-if" costs for a single shipment.

### **Why It Matters**
- **Parcel vs. LTL Trade-off:** LTL is cheaper for heavier shipments over longer distances due to economies of scale.  
- **Real-World Relevance:** Logistics companies like UPS and FedEx use similar logic, enhanced with **zone-based pricing, dimensional weight, and service-level guarantees**.  
- **Scalability:** The logic can easily scale to 10,000+ shipments with vectorized calculations or Spark/PySpark pipelines.
""")

st.info("""
**Interesting Note:** This model could be extended with **machine learning** to predict the *best carrier* based on historical cost, SLA performance, and transit time, or even integrate with **real-time carrier APIs**.
""")
