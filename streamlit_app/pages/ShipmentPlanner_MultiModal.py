import random
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="Multi-Modal Shipment Planner", layout="wide")
st.title("🚢 Pilot 4 — Multi-Modal & Multi-Leg Shipment Planner")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_shipments = st.slider("Number of shipments", 5, 30, 20, 1)
    random_seed = st.number_input("Random seed", value=42, step=1)

random.seed(int(random_seed))
np.random.seed(int(random_seed))

# ---------------------------------------------------------
# Define Modes and Legs
# ---------------------------------------------------------
modes = {
    "Truck": {"cost_per_mile": 2.0, "speed_mph": 50},
    "Rail": {"cost_per_mile": 1.2, "speed_mph": 35},
    "Air": {"cost_per_mile": 4.5, "speed_mph": 500},
    "Ocean": {"cost_per_mile": 0.8, "speed_mph": 20}
}

routes = [
    ["Truck"],
    ["Truck", "Rail"],
    ["Truck", "Air"],
    ["Truck", "Ocean", "Rail"]
]

# ---------------------------------------------------------
# Generate Shipment Data
# ---------------------------------------------------------
CITIES = ["Atlanta", "Chicago", "Dallas", "Denver", "Los Angeles", "Miami", "New York", "Seattle", "San Francisco"]

def generate_shipments(n):
    data = []
    for i in range(n):
        origin, destination = random.sample(CITIES, 2)
        distance = random.randint(200, 3000)  # miles
        deadline = random.randint(24, 120)  # delivery time in hours
        data.append({
            "ShipmentID": f"S{i+1:03d}",
            "Origin": origin,
            "Destination": destination,
            "Distance": distance,
            "Deadline (hrs)": deadline
        })
    return pd.DataFrame(data)

shipments_df = generate_shipments(num_shipments)
st.subheader("📦 Shipment Data")
st.dataframe(shipments_df, use_container_width=True)

# ---------------------------------------------------------
# Calculate Cost & Transit Time for Routes
# ---------------------------------------------------------
def evaluate_routes(distance, deadline):
    best_route = None
    best_cost = float('inf')
    best_time = None

    for route in routes:
        total_time = 0
        total_cost = 0
        segment_distance = distance / len(route)  # rough split among legs
        for leg in route:
            cost = segment_distance * modes[leg]["cost_per_mile"]
            time = (segment_distance / modes[leg]["speed_mph"])  # hours
            total_cost += cost
            total_time += time

        if total_time <= deadline and total_cost < best_cost:
            best_cost = total_cost
            best_time = total_time
            best_route = " → ".join(route)

    return best_route, round(best_cost, 2), round(best_time, 1)

# Evaluate each shipment
results = []
for _, row in shipments_df.iterrows():
    route, cost, time = evaluate_routes(row["Distance"], row["Deadline (hrs)"])
    results.append({
        "ShipmentID": row["ShipmentID"],
        "Chosen Route": route if route else "No feasible route",
        "Total Cost ($)": cost if route else None,
        "Transit Time (hrs)": time if route else None
    })

results_df = pd.DataFrame(results)
st.subheader("🚛 Selected Routes & Costs")
st.dataframe(results_df, use_container_width=True)

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
valid_shipments = results_df[results_df["Chosen Route"] != "No feasible route"]
total_cost = valid_shipments["Total Cost ($)"].sum()
avg_time = valid_shipments["Transit Time (hrs)"].mean()

c1, c2 = st.columns(2)
c1.metric("Total Cost of Shipments", f"${total_cost:,.2f}")
c2.metric("Average Transit Time", f"{avg_time:.1f} hrs")

# ---------------------------------------------------------
# Route Distribution Chart
# ---------------------------------------------------------
if not valid_shipments.empty:
    chart_data = valid_shipments.groupby("Chosen Route").size().reset_index(name="Count")
    st.subheader("📊 Route Distribution")
    chart = alt.Chart(chart_data).mark_bar().encode(
        x="Chosen Route:N",
        y="Count:Q",
        color="Chosen Route:N",
        tooltip=["Chosen Route", "Count"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

**Problem Statement:**  
Multi-modal shipments often combine modes (e.g., Truck → Rail → Truck) to reduce cost while meeting delivery deadlines.  
The challenge is to choose the **cheapest feasible multi-leg route** for each shipment.

**Key KPIs:**  
- Total cost across all shipments.  
- Average transit time (hours).  
- Distribution of chosen routes.

**Tech Stack & Tools:**  
- Python (data generation & logic).  
- Pandas & Numpy for computation.  
- Altair for visualization.  
- Streamlit for interactive dashboards.

**Next Steps:**  
- Add CO₂ emissions as a decision factor.  
- Include dynamic pricing by distance and weight.  
- Extend to multi-stop consolidations.
""")
