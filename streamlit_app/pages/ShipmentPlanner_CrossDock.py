import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit.components.v1 import html
import random
import math
import time

st.set_page_config(page_title="SupplyChain.ai — Cross-Dock Routing", layout="wide")

# ---------------------------------------------------------
# Haversine distance function (moved to top for global accessibility)
# ---------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

# ---------------------------------------------------------
# Header Image
# ---------------------------------------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Warehouse_icon.png",
          caption="Cross-docking and multi-echelon logistics", use_container_width=True)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🔄 Cross-Dock & Multi-Echelon Routing")

# ---------------------------------------------------------
# Business Context (Practitioner's View)
# ---------------------------------------------------------
st.header("Business Problem: The Cross-Docking Conundrum")
st.markdown("""
In my 20+ years in supply chain, a common challenge is optimizing shipments that don't go direct from origin to final destination. Instead, they pass through **intermediate hubs or cross-dock facilities** for consolidation or de-consolidation. The core problem is deciding, for each shipment, whether to send it **directly** (often faster but potentially more expensive for partial loads) or via a **cross-dock** (cheaper line-haul but adds handling time and cost). Optimizing these "line-haul (Plant → Hub)" and "last-mile (Hub → Customer)" routes together is significantly more complex than simple direct deliveries.

**Key KPIs Impacted by this Decision:**
- **Total Transportation Cost:** Balancing direct vs. multi-leg costs.
- **Cross-Docking Cost & Throughput:** Managing operational expenses and efficiency at hubs.
- **Lead Time & Order Cycle Time:** Ensuring timely delivery while optimizing costs.
- **Inventory Holding Cost:** Cross-docking reduces the need for long-term storage, impacting inventory levels.
""")

# ---------------------------------------------------------
# Why This Solution Matters (Practitioner's View)
# ---------------------------------------------------------
st.header("Why This Optimization Approach Matters?")
st.markdown("""
This approach to multi-echelon routing and cross-dock optimization is critical because it allows us to:
- **Reduce Total Logistics Costs:** By leveraging the cost efficiencies of different modes and consolidation opportunities at cross-docks.
- **Improve Delivery Speed & Reliability:** Balancing transit times with cost objectives to meet customer expectations.
- **Minimize Inventory Holding:** Cross-docking reduces the need for large, static inventories at distribution centers.
- **Enhance Network Flow:** Ensures coordinated movement of goods across multiple layers of the supply chain network (e.g., Plant → DC → Cross-Dock → Customer).
- **Make Data-Driven Routing Decisions:** Move beyond intuition to optimize end-to-end network flow based on real-time trade-offs.
""")

# ---------------------------------------------------------
# Tools Used
# ---------------------------------------------------------
st.header("Tools Used")
st.markdown("""
- **Python (Pandas, NumPy, Math):** For data generation, distance calculations, and core optimization logic.
- **Folium:** For interactive geographical visualization of the network and routes.
- **Streamlit:** For building the interactive web application and dashboard.
- *(Future: OR-Tools / PuLP for more complex multi-echelon VRPs; NetworkX for complex network visualization.)*
""")

# ---------------------------------------------------------
# Math Behind It (Practitioner's View)
# ---------------------------------------------------------
st.header("The Logic Behind the Optimization")
st.markdown(r"""
At its core, this problem is about finding the **"least cost path"** through a network that might involve intermediate stops, while respecting constraints like delivery deadlines. Conceptually, it's a type of **network flow problem** or a **shortest path problem with costs and time constraints**.

For each shipment, the system evaluates options by calculating:
1.  **Direct Route Cost & Time:** The cost and time to go directly from Plant to Customer.
2.  **Cross-Dock Route Cost & Time:** For each potential cross-dock, it calculates:
    * Cost & time from Plant to Cross-Dock.
    * **Plus:** Cross-dock handling cost and time.
    * **Plus:** Cost & time from Cross-Dock to Customer.

The optimization then selects the route (either direct or via a cross-dock) that has the **lowest total cost** while still ensuring the shipment **arrives before its deadline**.

This can be generalized as:
\[
\text{Minimize Total Cost} = \sum_{\text{segments}} (\text{Transportation Cost}) + \sum_{\text{cross-docks}} (\text{Handling Cost})
\]
Subject to:
\[
\text{Total Transit Time} \le \text{Delivery Deadline}
\]
\[
\text{All Shipments Delivered}
\]
Where:
-   **Transportation Cost** depends on distance, volume/weight, and mode.
-   **Handling Cost** is incurred at cross-dock facilities.
-   The system implicitly decides on the "flow" ($x_{ij}$) of a shipment through specific nodes (Plant, Cross-Dock, Customer) to achieve the optimal outcome.
""")

# ---------------------------------------------------------
# AI & ML Angle
# ---------------------------------------------------------
st.header("How AI & ML Enhance Cross-Docking")
st.markdown("""
While this demo uses deterministic logic, in a real-world scenario, AI/ML can significantly enhance cross-dock operations:
-   **Predictive Models:** ML models can predict hub congestion, inbound/outbound volume imbalances, or demand spikes to proactively adjust cross-docking strategies.
-   **Dynamic Slotting:** AI can optimize real-time allocation of dock doors and staging areas based on predicted arrival/departure times.
-   **Reinforcement Learning:** Agents can learn optimal consolidation and transfer strategies over time, adapting to changing network conditions and demand patterns.
-   **LLMs:** Could assist in scenario generation for network design, quickly evaluating the impact of adding new cross-docks or changing network topology.
""")

# ---------------------------------------------------------
# Fun Fact
# ---------------------------------------------------------
st.header("Fun Fact")
st.markdown("""
> **Walmart** pioneered modern cross-docking in the 1980s, enabling them to move goods from suppliers to stores with minimal storage, a key factor in their legendary efficiency and low prices. Their entire distribution network is built around this multi-echelon concept.
""")

# ---------------------------------------------------------
# DEMO
# ---------------------------------------------------------
st.header("Interactive Demo: Cross-Docking in Action")
st.markdown("""
Adjust the parameters below to see how changes in cross-docking costs and times influence the optimal routing decisions for a set of shipments.
""")

# --- Demo Data & Logic ---
# Define fixed locations (simplified network for demo)
PLANTS = {
    "Plant A (LA)": [34.0522, -118.2437],
    "Plant B (Chicago)": [41.8781, -87.6298],
    "Plant C (Dallas)": [32.7767, -96.7970]
}

CROSS_DOCKS = {
    "CD 1 (Denver)": [39.7392, -104.9903],
    "CD 2 (Atlanta)": [33.7490, -84.3880]
}

CUSTOMERS = {
    "Cust 1 (SF)": [37.7749, -122.4194],
    "Cust 2 (NY)": [40.7128, -74.0060],
    "Cust 3 (Miami)": [25.7617, -80.1918],
    "Cust 4 (Seattle)": [47.6062, -122.3321],
    "Cust 5 (Houston)": [29.7604, -95.3698]
}

ALL_LOCATIONS = {**PLANTS, **CROSS_DOCKS, **CUSTOMERS}

# Base travel speed (mph)
TRAVEL_SPEED_MPH = 50

# Differentiated Costs for Direct vs. Cross-Dock Legs
DIRECT_COST_PER_MILE_PER_CBM = 0.8 # Higher cost for direct (simulating LTL or less efficient)
CROSS_DOCK_LINE_HAUL_COST_PER_MILE_PER_CBM = 0.3 # Lower cost for cross-dock legs (simulating FTL/rail efficiency)


# --- Shipment Generation ---
# Removed @st.cache_data from this function
def generate_cross_dock_shipments(n: int, seed: int = None) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

    data = []
    plant_names = list(PLANTS.keys())
    customer_names = list(CUSTOMERS.keys())

    for i in range(n):
        origin = random.choice(plant_names)
        destination = random.choice(customer_names)
        volume_cbm = random.uniform(1.0, 10.0) # Cubic meters
        
        # Calculate direct distance for a reasonable deadline
        direct_dist = haversine_miles(PLANTS[origin][0], PLANTS[origin][1],
                                      CUSTOMERS[destination][0], CUSTOMERS[destination][1])
        
        # Deadline based on direct distance, allowing some flexibility for cross-dock
        deadline_hours = (direct_dist / TRAVEL_SPEED_MPH) * random.uniform(1.2, 2.0) # 20% to 100% buffer
        deadline_hours = max(24, int(deadline_hours)) # Min 24 hours

        data.append({
            "ShipmentID": f"CD{i+1:03d}",
            "Origin": origin,
            "Destination": destination,
            "Volume (CBM)": round(volume_cbm, 1),
            "Deadline (hrs)": deadline_hours
        })
    return pd.DataFrame(data)

# --- Optimization Logic ---
def evaluate_cross_dock_routes(shipments_df, cross_dock_cost_per_cbm, cross_dock_time_hrs):
    results = []
    for idx, shipment in shipments_df.iterrows():
        origin_coords = PLANTS[shipment["Origin"]]
        destination_coords = CUSTOMERS[shipment["Destination"]]
        volume = shipment["Volume (CBM)"]
        deadline = shipment["Deadline (hrs)"]

        best_route_type = "No feasible route"
        best_total_cost = float('inf')
        best_total_time = float('inf')
        best_path_nodes = [] # Stores actual node names for map drawing

        # Option 1: Direct Route
        direct_dist = haversine_miles(origin_coords[0], origin_coords[1],
                                      destination_coords[0], destination_coords[1])
        direct_cost = direct_dist * DIRECT_COST_PER_MILE_PER_CBM * volume # Use DIRECT_COST
        direct_time = direct_dist / TRAVEL_SPEED_MPH

        if direct_time <= deadline:
            best_route_type = "Direct"
            best_total_cost = direct_cost
            best_total_time = direct_time
            best_path_nodes = [shipment["Origin"], shipment["Destination"]]

        # Option 2: Via Cross-Dock
        for cd_name, cd_coords in CROSS_DOCKS.items():
            # Plant to Cross-Dock leg
            plant_to_cd_dist = haversine_miles(origin_coords[0], origin_coords[1],
                                               cd_coords[0], cd_coords[1])
            plant_to_cd_cost = plant_to_cd_dist * CROSS_DOCK_LINE_HAUL_COST_PER_MILE_PER_CBM * volume # Use CROSS_DOCK_LINE_HAUL_COST
            plant_to_cd_time = plant_to_cd_dist / TRAVEL_SPEED_MPH

            # Cross-Dock to Customer leg
            cd_to_customer_dist = haversine_miles(cd_coords[0], cd_coords[1],
                                                  destination_coords[0], destination_coords[1])
            cd_to_customer_cost = cd_to_customer_dist * CROSS_DOCK_LINE_HAUL_COST_PER_MILE_PER_CBM * volume # Use CROSS_DOCK_LINE_HAUL_COST
            cd_to_customer_time = cd_to_customer_dist / TRAVEL_SPEED_MPH

            # Total for cross-dock route
            total_cd_cost = plant_to_cd_cost + cd_to_customer_cost + (cross_dock_cost_per_cbm * volume)
            total_cd_time = plant_to_cd_time + cd_to_customer_time + cross_dock_time_hrs

            if total_cd_time <= deadline and total_cd_cost < best_total_cost:
                best_route_type = f"Via {cd_name}"
                best_total_cost = total_cd_cost
                best_total_time = total_cd_time
                best_path_nodes = [shipment["Origin"], cd_name, shipment["Destination"]]

        results.append({
            "ShipmentID": shipment["ShipmentID"],
            "Origin": shipment["Origin"],
            "Destination": shipment["Destination"],
            "Volume (CBM)": shipment["Volume (CBM)"],
            "Deadline (hrs)": shipment["Deadline (hrs)"],
            "Chosen Route Type": best_route_type,
            "Total Cost ($)": round(best_total_cost, 2) if best_route_type != "No feasible route" else None,
            "Transit Time (hrs)": round(best_total_time, 1) if best_route_type != "No feasible route" else None,
            "Path Nodes": best_path_nodes # Store nodes for map drawing
        })
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.markdown("---")
st.subheader("Simulate Cross-Docking Scenarios")

col_params, col_buttons = st.columns([0.7, 0.3])

with col_params:
    num_shipments = st.slider("Number of Shipments", 5, 50, 20, 1)
    # Adjusted default values to encourage cross-docking
    cross_dock_cost_per_cbm = st.slider("Cross-Docking Handling Cost per CBM ($)", 0.0, 50.0, 5.0, 1.0) # Lowered default
    cross_dock_time_hrs = st.slider("Cross-Docking Handling Time (hrs)", 0.0, 24.0, 2.0, 1.0) # Lowered default
    
    # Data seed control
    data_seed = st.number_input("Data Random Seed (0 for new data each refresh)", value=0, step=1, help="Enter a number for reproducible data, or set to 0 for new data on each refresh.")

with col_buttons:
    st.markdown("<br>", unsafe_allow_html=True) # Add some space
    if st.button("Run Optimization"):
        st.session_state.rerun_optimization = True
    if st.button("Reset Simulation"):
        st.session_state.rerun_optimization = False
        st.cache_data.clear() # Clear cache to reset all data

# Initialize session state for re-running optimization
if 'rerun_optimization' not in st.session_state:
    st.session_state.rerun_optimization = False

# Generate and run optimization
current_seed = data_seed if data_seed != 0 else None
shipments_df = generate_cross_dock_shipments(num_shipments, current_seed)

# Only run optimization if explicitly triggered or on initial load
# This helps control when the optimization logic re-runs, improving responsiveness
if st.session_state.rerun_optimization or not st.session_state.get('results_df_cached'):
    results_df = evaluate_cross_dock_routes(shipments_df, cross_dock_cost_per_cbm, cross_dock_time_hrs)
    st.session_state.results_df_cached = results_df # Cache results in session state
else:
    results_df = st.session_state.results_df_cached # Use cached results

st.subheader("📦 Generated Shipments")
st.dataframe(shipments_df, use_container_width=True)

st.subheader("🚛 Optimized Shipment Routes")
st.dataframe(results_df, use_container_width=True)

# --- KPIs ---
st.subheader("📈 Key Performance Indicators")
valid_results = results_df[results_df["Chosen Route Type"] != "No feasible route"]

total_transport_cost = valid_results["Total Cost ($)"].sum()
total_cross_dock_shipments = valid_results[valid_results["Chosen Route Type"].str.contains("Via")].shape[0]
avg_transit_time = valid_results["Transit Time (hrs)"].mean()
num_direct_shipments = valid_results[valid_results["Chosen Route Type"] == "Direct"].shape[0]

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Total Optimized Cost", f"${total_transport_cost:,.2f}")
with col_kpi2:
    st.metric("Avg. Transit Time", f"{avg_transit_time:.1f} hrs")
with col_kpi3:
    st.metric("Shipments via Cross-Dock", f"{total_cross_dock_shipments} ({total_cross_dock_shipments/len(valid_results)*100:.1f}%)")
with col_kpi4:
    st.metric("Shipments Direct", f"{num_direct_shipments} ({num_direct_shipments/len(valid_results)*100:.1f}%)")


# --- Map Visualization ---
st.subheader("🗺 Visualizing Optimized Routes")

# No @st.cache_data here to ensure dynamic updates
def render_cross_dock_map(results_df_map, all_locations_map, plants_map, cross_docks_map, customers_map):
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4) # Centered on USA

    # Add Plants
    for name, coords in plants_map.items():
        folium.Marker(
            location=coords,
            popup=name,
            icon=folium.Icon(color='green', icon='industry', prefix='fa')
        ).add_to(m)
    
    # Add Cross-Docks
    for name, coords in cross_docks_map.items():
        folium.Marker(
            location=coords,
            popup=name,
            icon=folium.Icon(color='orange', icon='retweet', prefix='fa') # Retweet icon for cross-dock
        ).add_to(m)

    # Add Customers
    for name, coords in customers_map.items():
        folium.Marker(
            location=coords,
            popup=name,
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)

    # Draw routes
    for idx, row in results_df_map.iterrows():
        if row["Chosen Route Type"] != "No feasible route":
            path_nodes = row["Path Nodes"]
            path_coords = [all_locations_map[node] for node in path_nodes]
            
            line_color = "darkblue"
            line_weight = 3
            line_opacity = 0.7
            line_dash = None

            if "Via" in row["Chosen Route Type"]:
                # Draw segments separately for cross-dock routes
                # Plant to CD
                folium.PolyLine(
                    locations=[all_locations_map[path_nodes[0]], all_locations_map[path_nodes[1]]],
                    color="purple", # Line-haul to CD
                    weight=line_weight,
                    opacity=line_opacity,
                    dash_array="5, 5", # Dashed for line-haul
                    tooltip=f"Line-haul: {path_nodes[0]} → {path_nodes[1]}<br>Shipment: {row['ShipmentID']}<br>Cost: ${row['Total Cost ($)']:.2f}"
                ).add_to(m)
                # CD to Customer
                folium.PolyLine(
                    locations=[all_locations_map[path_nodes[1]], all_locations_map[path_nodes[2]]],
                    color="purple", # Last-mile from CD
                    weight=line_weight,
                    opacity=line_opacity,
                    tooltip=f"Last-mile: {path_nodes[1]} → {path_nodes[2]}<br>Shipment: {row['ShipmentID']}<br>Cost: ${row['Total Cost ($)']:.2f}"
                ).add_to(m)
            else: # Direct route
                folium.PolyLine(
                    locations=path_coords,
                    color="green", # Direct route
                    weight=line_weight,
                    opacity=line_opacity,
                    tooltip=f"Direct: {path_nodes[0]} → {path_nodes[1]}<br>Shipment: {row['ShipmentID']}<br>Cost: ${row['Total Cost ($)']:.2f}"
                ).add_to(m)
    
    # Add a legend
    legend_html = """
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 180px; height: 120px; 
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; opacity:0.9;">
           &nbsp; <b>Route Legend</b> <br>
           &nbsp; <i style="background:green; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Direct Route <br>
           &nbsp; <i style="background:purple; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Cross-Dock Route <br>
           &nbsp; <i style="background:orange; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Cross-Dock Facility <br>
           &nbsp; <i style="background:green; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Plant <br>
           &nbsp; <i style="background:blue; opacity:0.7;">&nbsp;&nbsp;&nbsp;&nbsp;</i> Customer
         </div>
         """
    m.get_root().html.add_child(folium.Element(legend_html))


    html(m._repr_html_(), height=500)

render_cross_dock_map(results_df, ALL_LOCATIONS, PLANTS, CROSS_DOCKS, CUSTOMERS)

st.markdown("---")
