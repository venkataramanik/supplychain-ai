import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit.components.v1 import html
import random
import math
import time # For dynamic seed

st.set_page_config(page_title="SupplyChain.ai — Multi-Modal Shipment Planner", layout="wide")
st.title("🚢 AI-Powered Multi-Modal & Multi-Leg Shipment Planner")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_shipments = st.slider("Number of Shipments", 5, 50, 20, 1)
    # Data seed control (0 for truly random, any number for reproducible)
    data_seed = st.number_input("Data Random Seed (0 for new data each refresh)", value=0, step=1, help="Enter a number for reproducible data, or set to 0 for new data on each refresh.")
    
    st.markdown("---")
    st.subheader("Cost & Time Weights")
    cost_weight = st.slider("Prioritize Cost (vs. Time)", 0.0, 1.0, 0.7, 0.1)
    # This slider will influence the decision making in evaluate_route,
    # higher cost_weight means cost is more important than time.

# Ensure true randomization if seed is 0 or None
current_seed = data_seed if data_seed != 0 else None
if current_seed is not None:
    random.seed(int(current_seed))
    np.random.seed(int(current_seed))
else:
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))


# ---------------------------------------------------------
# Global Locations & Haversine Distance
# ---------------------------------------------------------
GLOBAL_HUBS = {
    "Shanghai, China": [31.2304, 121.4737],
    "Rotterdam, Netherlands": [51.9244, 4.4777],
    "New York, USA": [40.7128, -74.0060],
    "Los Angeles, USA": [34.0522, -118.2437],
    "Hamburg, Germany": [53.5511, 9.9937],
    "Singapore": [1.3521, 103.8198],
    "Chicago, USA": [41.8781, -87.6298],
    "London, UK": [51.5074, -0.1278],
    "Tokyo, Japan": [35.6895, 139.6917],
    "Sydney, Australia": [-33.8688, 151.2093],
    "Dubai, UAE": [25.2048, 55.2708]
}

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Calculates the distance between two points on Earth in miles."""
    R = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

# ---------------------------------------------------------
# Define Mode Characteristics & Route Templates
# ---------------------------------------------------------
# Mode characteristics (per unit of weight/volume per mile)
MODES = {
    "Truck": {"cost_per_mile_per_kg": 0.005, "cost_per_mile_per_cbm": 0.5, "speed_mph": 50, "co2_per_kg_mile": 0.0001},
    "Rail": {"cost_per_mile_per_kg": 0.002, "cost_per_mile_per_cbm": 0.2, "speed_mph": 30, "co2_per_kg_mile": 0.00005},
    "Air": {"cost_per_mile_per_kg": 0.05, "cost_per_mile_per_cbm": 5.0, "speed_mph": 500, "co2_per_kg_mile": 0.005},
    "Ocean": {"cost_per_mile_per_kg": 0.0005, "cost_per_mile_per_cbm": 0.05, "speed_mph": 18, "co2_per_kg_mile": 0.00001}
}

# Route templates with typical mode usage and distance proportions
# These proportions are simplified heuristics for global multi-modal routes
ROUTE_TEMPLATES = {
    "Truck (Domestic)": {"modes": ["Truck"], "distance_proportions": {"Truck": 1.0}},
    "Truck-Rail-Truck (Domestic)": {"modes": ["Truck", "Rail", "Truck"], "distance_proportions": {"Truck": 0.4, "Rail": 0.6}},
    "Ocean-Truck (Standard)": {"modes": ["Ocean", "Truck"], "distance_proportions": {"Ocean": 0.9, "Truck": 0.1}},
    "Ocean-Rail-Truck (Intermodal)": {"modes": ["Ocean", "Rail", "Truck"], "distance_proportions": {"Ocean": 0.8, "Rail": 0.15, "Truck": 0.05}},
    "Air-Truck (Expedited)": {"modes": ["Air", "Truck"], "distance_proportions": {"Air": 0.95, "Truck": 0.05}},
    "Air-Air-Truck (Critical)": {"modes": ["Air", "Air", "Truck"], "distance_proportions": {"Air": 0.9, "Truck": 0.1}} # Assumes 2 air legs for transshipment
}

# Transshipment costs/times (simplified, fixed per transfer)
TRANSSHIPMENT_COST_PER_LEG = 50 # Cost per transfer point
TRANSSHIPMENT_TIME_PER_LEG = 12 # Hours per transfer point

# ---------------------------------------------------------
# Generate Shipment Data
# ---------------------------------------------------------
@st.cache_data
def generate_shipments(n: int) -> pd.DataFrame:
    """Generates randomized global shipment data."""
    data = []
    hub_names = list(GLOBAL_HUBS.keys())
    urgency_levels = ["Standard", "Expedited", "Critical"]

    for i in range(n):
        origin_name, destination_name = random.sample(hub_names, 2)
        origin_coords = GLOBAL_HUBS[origin_name]
        destination_coords = GLOBAL_HUBS[destination_name]
        
        distance = haversine_miles(
            origin_coords[0], origin_coords[1],
            destination_coords[0], destination_coords[1]
        )
        
        weight_kg = random.randint(50, 5000) # kg
        volume_cbm = random.uniform(0.5, 50) # cubic meters
        urgency = random.choice(urgency_levels)

        # Base deadline calculation based on distance and urgency
        # Longer distance -> longer deadline
        # Critical urgency -> shorter deadline
        base_deadline_days = distance / 500 # Approx days at average speed
        if urgency == "Standard":
            deadline_hours = base_deadline_days * 24 * random.uniform(1.0, 1.5)
        elif urgency == "Expedited":
            deadline_hours = base_deadline_days * 24 * random.uniform(0.7, 1.0)
        else: # Critical
            deadline_hours = base_deadline_days * 24 * random.uniform(0.4, 0.7)
        
        deadline_hours = max(24, int(deadline_hours)) # Minimum 24 hours

        data.append({
            "ShipmentID": f"S{i+1:03d}",
            "Origin": origin_name,
            "Destination": destination_name,
            "Distance (miles)": round(distance, 1),
            "Weight (kg)": weight_kg,
            "Volume (CBM)": round(volume_cbm, 1),
            "Urgency": urgency,
            "Deadline (hrs)": deadline_hours
        })
    return pd.DataFrame(data)

shipments_df = generate_shipments(num_shipments)
st.subheader("📦 Global Shipment Data")
st.dataframe(shipments_df, use_container_width=True)

# ---------------------------------------------------------
# Evaluate Routes (Optimization Logic)
# ---------------------------------------------------------
@st.cache_data
def evaluate_routes_for_shipment(shipment_row, cost_weight_param):
    """
    Evaluates all possible multi-modal routes for a single shipment
    and selects the best one based on a weighted cost-time objective.
    """
    shipment_distance = shipment_row["Distance (miles)"]
    shipment_weight = shipment_row["Weight (kg)"]
    shipment_volume = shipment_row["Volume (CBM)"]
    shipment_deadline = shipment_row["Deadline (hrs)"]
    shipment_urgency = shipment_row["Urgency"]

    best_route_name = "No feasible route"
    best_total_cost = float('inf')
    best_total_time = float('inf')
    best_total_co2 = float('inf')
    
    # Objective: Minimize (cost_weight * cost) + ((1 - cost_weight) * time_penalty)
    # Time penalty increases sharply as deadline is approached/exceeded
    
    # Urgency-based time factor: Critical shipments have higher time sensitivity
    urgency_time_factor = {
        "Standard": 1.0,
        "Expedited": 1.5,
        "Critical": 2.5
    }.get(shipment_urgency, 1.0)

    for route_name, template in ROUTE_TEMPLATES.items():
        total_route_cost = 0
        total_route_time = 0
        total_route_co2 = 0
        
        modes_in_route = template["modes"]
        distance_proportions = template["distance_proportions"]
        
        # Calculate transshipment costs/times
        num_transshipments = len(modes_in_route) - 1
        total_route_cost += num_transshipments * TRANSSHIPMENT_COST_PER_LEG
        total_route_time += num_transshipments * TRANSSHIPMENT_TIME_PER_LEG

        for mode in modes_in_route:
            # Calculate distance for this mode based on its proportion of total shipment distance
            mode_distance = shipment_distance * distance_proportions.get(mode, 0) # Use 0 if mode not in proportions

            if mode_distance > 0:
                cost_per_mile_kg = MODES[mode]["cost_per_mile_per_kg"]
                cost_per_mile_cbm = MODES[mode]["cost_per_mile_per_cbm"]
                speed_mph = MODES[mode]["speed_mph"]
                co2_per_kg_mile = MODES[mode]["co2_per_kg_mile"]

                # Cost calculation: consider both weight and volume, take the higher one or sum them
                # For simplicity, let's assume cost is based on max of weight-based or volume-based cost
                mode_cost_kg = mode_distance * cost_per_mile_kg * shipment_weight
                mode_cost_cbm = mode_distance * cost_per_mile_cbm * shipment_volume
                
                # A common industry practice is to charge based on chargeable weight/volume
                # For this demo, let's just sum them or take the higher, summing is simpler for now
                mode_cost = mode_cost_kg + mode_cost_cbm 
                
                mode_time = mode_distance / speed_mph
                mode_co2 = mode_distance * co2_per_kg_mile * shipment_weight # CO2 based on weight

                total_route_cost += mode_cost
                total_route_time += mode_time
                total_route_co2 += mode_co2
        
        # Add a penalty for time if it exceeds deadline, scaled by urgency
        time_penalty = 0
        if total_route_time > shipment_deadline:
            time_over_deadline = total_route_time - shipment_deadline
            # Exponential penalty for exceeding deadline, scaled by urgency
            time_penalty = (time_over_deadline / shipment_deadline) * 100 * urgency_time_factor # Example penalty factor

        # Combined objective function: weighted sum of cost and time penalty
        # This is a simplified "optimization" for demo purposes.
        # In a real scenario, this would be a more complex LP/MIP model.
        objective_value = (cost_weight_param * total_route_cost) + ((1 - cost_weight_param) * time_penalty)
        
        # Check feasibility: Must meet deadline, and have a lower objective value
        # Also, ensure that if time_penalty is too high, it's considered infeasible
        if total_route_time <= shipment_deadline or time_penalty < 50: # Allow some minor penalty if it's the only option
            if objective_value < best_total_cost: # We're minimizing objective value
                best_total_cost = total_route_cost
                best_total_time = total_route_time
                best_total_co2 = total_route_co2
                best_route_name = " → ".join(modes_in_route)

    return best_route_name, round(best_total_cost, 2), round(best_total_time, 1), round(best_total_co2, 4)

# Evaluate each shipment
results = []
for index, row in shipments_df.iterrows():
    route, cost, time, co2 = evaluate_routes_for_shipment(row, cost_weight)
    results.append({
        "ShipmentID": row["ShipmentID"],
        "Origin": row["Origin"],
        "Destination": row["Destination"],
        "Distance (miles)": row["Distance (miles)"],
        "Weight (kg)": row["Weight (kg)"],
        "Volume (CBM)": row["Volume (CBM)"],
        "Urgency": row["Urgency"],
        "Deadline (hrs)": row["Deadline (hrs)"],
        "Chosen Route": route,
        "Total Cost ($)": cost,
        "Transit Time (hrs)": time,
        "CO2 Emissions (kg)": co2
    })

results_df = pd.DataFrame(results)
st.subheader("🚛 Selected Routes & Costs")
st.dataframe(results_df, use_container_width=True)

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
valid_shipments_results = results_df[results_df["Chosen Route"] != "No feasible route"]
total_cost_all = valid_shipments_results["Total Cost ($)"].sum()
total_co2_all = valid_shipments_results["CO2 Emissions (kg)"].sum()
avg_transit_time = valid_shipments_results["Transit Time (hrs)"].mean()
avg_cost_per_kg = (total_cost_all / valid_shipments_results["Weight (kg)"].sum()) if valid_shipments_results["Weight (kg)"].sum() > 0 else 0

st.subheader("📈 Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Shipments Optimized", f"{len(valid_shipments_results)} / {num_shipments}")
c2.metric("Total Optimized Cost", f"${total_cost_all:,.2f}")
c3.metric("Total CO2 Emissions", f"{total_co2_all:,.2f} kg")
c4.metric("Avg. Cost per kg", f"${avg_cost_per_kg:,.2f}")
st.metric("Average Transit Time", f"{avg_transit_time:.1f} hrs")

# ---------------------------------------------------------
# Route Distribution Chart
# ---------------------------------------------------------
if not valid_shipments_results.empty:
    chart_data = valid_shipments_results.groupby("Chosen Route").size().reset_index(name="Count")
    st.subheader("📊 Route Distribution")
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Chosen Route:N", sort="-y"), # Sort by count
        y="Count:Q",
        color="Chosen Route:N",
        tooltip=["Chosen Route", "Count"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# Global Map Visualization
# ---------------------------------------------------------
st.subheader("🗺 Global Shipment Routes")

@st.cache_data
def render_global_map(df):
    # Center map on average of all origins/destinations, or a global view
    center_lat = np.mean([GLOBAL_HUBS[o][0] for o in df['Origin'].unique()] + [GLOBAL_HUBS[d][0] for d in df['Destination'].unique()])
    center_lon = np.mean([GLOBAL_HUBS[o][1] for o in df['Origin'].unique()] + [GLOBAL_HUBS[d][1] for d in df['Destination'].unique()])
    
    # Adjust zoom for global view
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='CartoDB Positron')

    # Add origin and destination markers
    for city, coords in GLOBAL_HUBS.items():
        folium.CircleMarker(
            location=coords,
            radius=5,
            color="darkblue",
            fill=True,
            fill_opacity=0.8,
            popup=city
        ).add_to(m)

    # Add routes
    for idx, row in df.iterrows():
        if row["Chosen Route"] != "No feasible route":
            origin_coords = GLOBAL_HUBS[row['Origin']]
            destination_coords = GLOBAL_HUBS[row['Destination']]
            
            # Color based on urgency or mode type
            color_map = {"Standard": "green", "Expedited": "orange", "Critical": "red"}
            line_color = color_map.get(row['Urgency'], 'blue')
            
            folium.PolyLine(
                locations=[origin_coords, destination_coords],
                color=line_color,
                weight=3,
                tooltip=f"ID: {row['ShipmentID']}<br>From: {row['Origin']}<br>To: {row['Destination']}<br>Route: {row['Chosen Route']}<br>Cost: ${row['Total Cost ($)']:,}<br>Time: {row['Transit Time (hrs)']:.1f} hrs<br>CO2: {row['CO2 Emissions (kg)']:.2f} kg"
            ).add_to(m)
    
    # Add a legend for urgency
    legend_html = """
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: 100px; 
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; opacity:0.9;">
           &nbsp; <b>Route Urgency</b> <br>
           &nbsp; <i style="background:green; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Standard <br>
           &nbsp; <i style="background:orange; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Expedited <br>
           &nbsp; <i style="background:red; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Critical
         </div>
         """
    m.get_root().html.add_child(folium.Element(legend_html))

    html(m._repr_html_(), height=500)

render_global_map(valid_shipments_results)


# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

### **Problem Statement: Navigating Global Multi-Modal Complexity**
In my 20+ years in supply chain, a constant challenge has been optimizing **global multi-modal shipments**. It's rarely a simple point-to-point journey. Goods move across continents, often transitioning between ocean, air, rail, and truck, each with its own cost, speed, and carbon footprint. The goal is to select the **most efficient combination of modes and legs** that meets delivery deadlines while minimizing overall cost and increasingly, environmental impact. Manually evaluating these complex trade-offs for every shipment is impractical and sub-optimal.

### **Why This Solution Matters: Strategic & Operational Efficiency**
This AI-powered prototype demonstrates how we can intelligently select the best multi-modal route for each shipment. From a supply chain practitioner's perspective, this tool enables us to:
- **Optimize Trade-offs:** Balance the often-conflicting objectives of cost, speed, and sustainability for every shipment.
- **Enhance Delivery Reliability:** Ensure shipments meet their deadlines by selecting feasible routes.
- **Reduce Logistics Costs:** Identify the most cost-effective mode combinations for various global lanes and urgency levels.
- **Improve Sustainability:** Account for CO2 emissions in routing decisions, supporting corporate environmental goals.
- **Automate Decision-Making:** Move away from manual, heuristic-based routing to data-driven, optimized choices.

### **How It Works: My Approach to Multi-Modal Route Optimization**
My goal was to build a system that can quickly evaluate complex global routing options, much like an experienced logistics planner, but at scale. Here’s how I approached it:

1.  **Global Data & Realistic Distances (using Pandas & NumPy):** I started by defining a network of major global shipping hubs (cities and ports) with their actual geographic coordinates. Using **Pandas** and **NumPy** in **Python**, the system then calculates realistic **Haversine distances** between these global origins and destinations. This moves beyond simple random distances to reflect actual geographical challenges.
2.  **Defining Mode Characteristics & Route Templates:** For each transportation mode (Truck, Rail, Air, Ocean), I've defined key characteristics: their typical **cost per unit of weight/volume per mile**, **average speed**, and **CO2 emissions per unit of weight per mile**. I then created **route templates** – common multi-modal pathways (e.g., "Ocean-Truck", "Air-Rail-Truck") – each with a predefined sequence of modes and typical proportions of the total distance assigned to each leg. This allows for a more realistic simulation of multi-leg journeys.
3.  **Accounting for Transshipment:** A crucial aspect of multi-modal is the overhead when switching modes (e.g., from ocean to rail at a port). The system incorporates fixed **transshipment costs and times** for each transfer point, reflecting the operational reality of intermodal movements.
4.  **Intelligent Route Evaluation (Python Logic):** For every shipment, the system evaluates all predefined multi-modal route templates. It calculates the total cost, transit time, and CO2 emissions for each potential route, considering the shipment's weight, volume, and urgency. The "optimization" then selects the best route by balancing cost and time, with a heavier weighting towards meeting deadlines for urgent shipments. This decision-making logic is built using **Python**.
5.  **Interactive Dashboard & Global Visualization (Streamlit & Folium):** All these insights are presented in an interactive dashboard built with **Streamlit**. This **Python** framework allows for rapid web application development. I use **Altair** for charts (like route distribution) and **Folium** to visualize the chosen global routes on an interactive map, making it easy to see the flow of goods across continents.

### **Key Performance Indicators (KPIs) This Solution Improves**
- **Total Optimized Cost:** Direct financial savings from efficient routing.
- **Average Transit Time:** Ensuring delivery speed and meeting deadlines.
- **Total CO2 Emissions:** Quantifiable environmental impact reduction.
- **Route Utilization:** Insights into which multi-modal pathways are most frequently chosen.

### **Tech Stack: The Tools I Used**
- **Python:** The core programming language for all logic and analytics.
- **Pandas & NumPy:** Essential for data generation, manipulation, and numerical calculations.
- **Altair:** For creating clear, interactive charts and visualizations.
- **Folium:** For interactive global map visualizations of shipment routes.
- **Streamlit:** For rapidly building and deploying the interactive web-based dashboard.

### **Next Steps & Strategic Vision**
This prototype lays a strong foundation for advanced multi-modal optimization. Looking forward, the next strategic steps would involve:
- **Dynamic Pricing Integration:** Connecting to real-time carrier pricing APIs for more accurate cost calculations.
- **Capacity & Constraint Modeling:** Incorporating actual vehicle/container capacities and network constraints (e.g., port congestion).
- **Advanced Optimization Algorithms:** Implementing more sophisticated solvers (e.g., using OR-Tools or PuLP) for true constrained optimization across a network.
- **Real-time Tracking & Re-routing:** Integrating live tracking data to enable dynamic re-routing in case of disruptions.
- **Service Level Agreement (SLA) Compliance:** More granular modeling of penalties for missed deadlines.
- **Multi-Stop Consolidation:** Optimizing full truckload/container load utilization across multiple shipments.
""")
