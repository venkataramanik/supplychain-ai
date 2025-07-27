import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit.components.v1 import html
import random
import math
import time

st.set_page_config(page_title="Dynamic Re-Routing", layout="wide")

# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

# ---------------------------------------------------------
# Header Image
# ---------------------------------------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/Map_icon.png",
          caption="Dynamic re-routing and real-time optimization", use_container_width=True)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("⚡ Dynamic Re-Routing (Real-Time Optimization)")

# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.header("Business Problem")
st.markdown("""
Static transportation plans often fail due to **real-time disruptions** like traffic jams, weather, 
vehicle breakdowns, or last-minute orders. Companies such as **ride-sharing platforms and 
e-commerce giants** rely on **dynamic routing engines** to adjust plans on the fly.

**KPIs Impacted:**
- **Average delivery delay time.**
- **Fleet utilization %.**
- **Customer satisfaction (NPS).**
""")

# ---------------------------------------------------------
# Why This Approach
# ---------------------------------------------------------
st.header("Why This Approach?")
st.markdown("""
Dynamic re-routing enables:
- **Real-time adjustment of routes** as conditions change.
- **Proactive avoidance** of delays and SLA violations.
- **Higher asset utilization** and customer service improvements.
""")

# ---------------------------------------------------------
# Tools Used
# ---------------------------------------------------------
st.header("Tools Used")
st.markdown("""
- **OR-Tools (Dynamic VRP):** For continuous re-optimization.
- **SimPy or simulation libraries:** To simulate dynamic events.
- **pandas:** To handle route and vehicle state data.
- **Folium:** For interactive map visualization.
""")

# ---------------------------------------------------------
# Math Behind It (LaTeX for formulas)
# ---------------------------------------------------------
st.header("Math Behind It")
st.markdown(r"""
Dynamic VRP is often modeled using **rolling horizon optimization** or **reinforcement learning (RL):**

\[
\pi^* = \arg\max_{\pi} \mathbb{E}[ \sum_{t=0}^{T} r_t ]
\]
Where:
- **\(\pi\)** is the routing policy.
- **\(r_t\)** is the reward (e.g., minimized delay or cost) at time step t.
""")

# ---------------------------------------------------------
# AI & ML Angle
# ---------------------------------------------------------
st.header("How AI & ML Enhance Dynamic Routing")
st.markdown("""
- **Predictive Models:** Real-time ETA predictions using ML models (XGBoost, LSTMs).
- **Reinforcement Learning:** Continuously learns optimal routing adjustments.
- **LLMs:** Generate quick "what-if" re-routing scenarios when disruptions occur.
""")

# ---------------------------------------------------------
# Fun Fact
# ---------------------------------------------------------
st.header("Fun Fact")
st.markdown("""
> **Uber's dispatch engine** dynamically reassigns drivers every few seconds, 
solving a complex VRP variant with real-time events and stochastic demands.
""")

# ---------------------------------------------------------
# DEMO
# ---------------------------------------------------------
st.header("Interactive Demo: Dynamic Re-Routing in Action")
st.markdown("""
This simplified demo illustrates how routes can be dynamically adjusted in response to a simulated traffic disruption.
Watch how the system re-calculates the optimal path to minimize impact.
""")

# --- Demo Data & Logic ---
# Define fixed locations (simplified network for demo)
DEMO_LOCATIONS = {
    "Warehouse (A)": [34.0522, -118.2437], # Los Angeles
    "Customer 1 (B)": [34.0901, -118.3587], # Santa Monica
    "Customer 2 (C)": [33.9213, -118.0264], # Downey
    "Customer 3 (D)": [33.8068, -117.9179], # Anaheim
    "Customer 4 (E)": [34.2000, -118.5000]  # San Fernando Valley
}

# Haversine distance function (miles)
def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

# Simplified travel time matrix (base time in hours for 50mph)
@st.cache_data
def get_base_travel_times(locations):
    times = {}
    loc_names = list(locations.keys())
    for i in range(len(loc_names)):
        for j in range(len(loc_names)):
            if i != j:
                loc1_name = loc_names[i]
                loc2_name = loc_names[j]
                dist = haversine_miles(locations[loc1_name][0], locations[loc1_name][1],
                                       locations[loc2_name][0], locations[loc2_name][1])
                times[(loc1_name, loc2_name)] = dist / 50 # Base speed 50 mph
            else:
                times[(loc_names[i], loc_names[j])] = 0
    return times

BASE_TRAVEL_TIMES = get_base_travel_times(DEMO_LOCATIONS)

# --- Simulation Logic ---
@st.cache_data(ttl=1) # Cache for 1 second to allow quick updates
def run_dynamic_routing_scenario(disruption_active, disruption_segment, disruption_factor):
    current_travel_times = BASE_TRAVEL_TIMES.copy()
    
    # Apply disruption if active
    if disruption_active and disruption_segment:
        segment_start, segment_end = disruption_segment
        if (segment_start, segment_end) in current_travel_times:
            current_travel_times[(segment_start, segment_end)] *= disruption_factor
            current_travel_times[(segment_end, segment_start)] *= disruption_factor # Bidirectional

    # Define initial vehicle and orders
    # Vehicle 1: A -> B -> C -> A
    # Vehicle 2: A -> D -> E -> A
    
    # Simplified VRP solver (finding shortest path for fixed sequence)
    # In a real scenario, this would be OR-Tools or a more complex algorithm
    
    routes_info = []
    
    # Vehicle 1 Route: Warehouse (A) -> Customer 1 (B) -> Customer 2 (C) -> Warehouse (A)
    v1_path = ["Warehouse (A)", "Customer 1 (B)", "Customer 2 (C)", "Warehouse (A)"]
    v1_total_time = 0
    v1_total_dist = 0
    v1_segments = []
    for i in range(len(v1_path) - 1):
        start_node = v1_path[i]
        end_node = v1_path[i+1]
        segment_time = current_travel_times.get((start_node, end_node), float('inf'))
        segment_dist = haversine_miles(DEMO_LOCATIONS[start_node][0], DEMO_LOCATIONS[start_node][1],
                                       DEMO_LOCATIONS[end_node][0], DEMO_LOCATIONS[end_node][1])
        v1_total_time += segment_time
        v1_total_dist += segment_dist
        v1_segments.append((start_node, end_node, segment_time))
    routes_info.append({"Vehicle": "Vehicle 1", "Path": v1_path, "TotalTime": v1_total_time, "TotalDist": v1_total_dist, "Segments": v1_segments})

    # Vehicle 2 Route: Warehouse (A) -> Customer 3 (D) -> Customer 4 (E) -> Warehouse (A)
    v2_path = ["Warehouse (A)", "Customer 3 (D)", "Customer 4 (E)", "Warehouse (A)"]
    v2_total_time = 0
    v2_total_dist = 0
    v2_segments = []
    for i in range(len(v2_path) - 1):
        start_node = v2_path[i]
        end_node = v2_path[i+1]
        segment_time = current_travel_times.get((start_node, end_node), float('inf'))
        segment_dist = haversine_miles(DEMO_LOCATIONS[start_node][0], DEMO_LOCATIONS[start_node][1],
                                       DEMO_LOCATIONS[end_node][0], DEMO_LOCATIONS[end_node][1])
        v2_total_time += segment_time
        v2_total_dist += segment_dist
        v2_segments.append((start_node, end_node, segment_time))
    routes_info.append({"Vehicle": "Vehicle 2", "Path": v2_path, "TotalTime": v2_total_time, "TotalDist": v2_total_dist, "Segments": v2_segments})

    # Simulate re-routing for Vehicle 1 if disruption affects its path
    # For this demo, we'll simplify: if disruption is on A-B, Vehicle 1 considers A-C-B-A
    # This is a very basic heuristic, not a full VRP re-optimization
    if disruption_active and disruption_segment == ("Warehouse (A)", "Customer 1 (B)"):
        st.info(f"Disruption detected on {disruption_segment[0]} to {disruption_segment[1]}! Re-routing Vehicle 1...")
        # New alternative path for Vehicle 1: A -> C -> B -> A (example alternative)
        v1_alt_path = ["Warehouse (A)", "Customer 2 (C)", "Customer 1 (B)", "Warehouse (A)"]
        v1_alt_total_time = 0
        v1_alt_total_dist = 0
        v1_alt_segments = []
        for i in range(len(v1_alt_path) - 1):
            start_node = v1_alt_path[i]
            end_node = v1_alt_path[i+1]
            segment_time = current_travel_times.get((start_node, end_node), float('inf'))
            segment_dist = haversine_miles(DEMO_LOCATIONS[start_node][0], DEMO_LOCATIONS[start_node][1],
                                           DEMO_LOCATIONS[end_node][0], DEMO_LOCATIONS[end_node][1])
            v1_alt_total_time += segment_time
            v1_alt_total_dist += segment_dist
            v1_alt_segments.append((start_node, end_node, segment_time))
        
        # Compare original vs alternative
        if v1_alt_total_time < v1_total_time: # If alternative is faster, choose it
            routes_info[0]["Path"] = v1_alt_path
            routes_info[0]["TotalTime"] = v1_alt_total_time
            routes_info[0]["TotalDist"] = v1_alt_total_dist
            routes_info[0]["Segments"] = v1_alt_segments
            routes_info[0]["Re-routed"] = True
        else:
            routes_info[0]["Re-routed"] = False # Stick to original if alt is worse
    else:
        routes_info[0]["Re-routed"] = False

    return routes_info

# --- Streamlit Demo UI ---
st.markdown("---")
st.subheader("Simulate a Disruption")

col_disrupt, col_reset = st.columns([0.7, 0.3])

with col_disrupt:
    disruption_segment_choice = st.selectbox(
        "Select a road segment to simulate traffic/closure:",
        options=[
            ("None", "No Disruption"),
            ("Warehouse (A)", "Customer 1 (B)"),
            ("Customer 1 (B)", "Customer 2 (C)"),
            ("Warehouse (A)", "Customer 3 (D)"),
            ("Customer 3 (D)", "Customer 4 (E)")
        ],
        format_func=lambda x: x[1] if isinstance(x, tuple) else x # Display second element of tuple
    )
    disruption_segment_tuple = None
    if disruption_segment_choice != ("None", "No Disruption"):
        disruption_segment_tuple = (disruption_segment_choice[0], disruption_segment_choice[1])

    disruption_factor = st.slider(
        "Traffic/Closure Impact (Multiplier on Travel Time)",
        min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help="1.0 = No impact, 2.0 = Double travel time, 5.0 = Five times travel time (near closure)"
    )

with col_reset:
    st.markdown("<br>", unsafe_allow_html=True) # Add some space
    if st.button("Apply Disruption / Update Routes"):
        st.session_state.disruption_active = True
        st.session_state.disruption_segment = disruption_segment_tuple
        st.session_state.disruption_factor = disruption_factor
    if st.button("Reset Simulation"):
        st.session_state.disruption_active = False
        st.session_state.disruption_segment = None
        st.session_state.disruption_factor = 1.0
        st.cache_data.clear() # Clear cache to reset all data

# Initialize session state for disruption
if 'disruption_active' not in st.session_state:
    st.session_state.disruption_active = False
if 'disruption_segment' not in st.session_state:
    st.session_state.disruption_segment = None
if 'disruption_factor' not in st.session_state:
    st.session_state.disruption_factor = 1.0

# Run the simulation logic
routes_data = run_dynamic_routing_scenario(
    st.session_state.disruption_active,
    st.session_state.disruption_segment,
    st.session_state.disruption_factor
)

# --- Display KPIs ---
st.subheader("KPIs: Impact of Disruption")
initial_total_time = sum(BASE_TRAVEL_TIMES.get((s,e),0) for v_info in routes_data for s,e,_ in v_info["Segments"])
current_total_time = sum(v_info["TotalTime"] for v_info in routes_data)

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric("Initial Total Transit Time", f"{initial_total_time:.1f} hrs")
with col_kpi2:
    st.metric("Current Total Transit Time", f"{current_total_time:.1f} hrs", delta=f"{current_total_time - initial_total_time:.1f} hrs")

# --- Map Visualization ---
st.subheader("Visualizing Routes")

@st.cache_data
def render_dynamic_map(routes_info, locations, disruption_segment_viz, disruption_factor_viz):
    m = folium.Map(location=[locations["Warehouse (A)"][0], locations["Warehouse (A)"][1]], zoom_start=11)

    # Add location markers
    for name, coords in locations.items():
        icon_color = "blue" if "Customer" in name else "green"
        folium.Marker(
            location=coords,
            popup=name,
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(m)

    # Add routes
    vehicle_colors = {"Vehicle 1": "purple", "Vehicle 2": "darkred"}
    for v_info in routes_info:
        path_coords = [locations[node] for node in v_info["Path"]]
        is_rerouted = v_info.get("Re-routed", False)
        
        # Draw each segment individually to highlight disrupted one
        for i in range(len(v_info["Path"]) - 1):
            start_node = v_info["Path"][i]
            end_node = v_info["Path"][i+1]
            segment_coords = [locations[start_node], locations[end_node]]
            
            segment_is_disrupted = (disruption_segment_viz == (start_node, end_node) or
                                    disruption_segment_viz == (end_node, start_node))
            
            line_color = vehicle_colors[v_info["Vehicle"]]
            line_weight = 4
            line_opacity = 0.8
            line_dash = None # Solid line by default

            if segment_is_disrupted:
                line_color = "red" # Highlight disrupted segment
                line_weight = 6
                line_opacity = 1.0
                line_dash = "5, 5" # Dashed for disruption

            tooltip_text = f"{v_info['Vehicle']}: {start_node} → {end_node}<br>Time: {v_info['Segments'][i][2]:.1f} hrs"
            if segment_is_disrupted:
                tooltip_text += f"<br>**DISRUPTED (x{disruption_factor_viz:.1f})**"
            
            folium.PolyLine(
                locations=segment_coords,
                color=line_color,
                weight=line_weight,
                opacity=line_opacity,
                dash_array=line_dash,
                tooltip=tooltip_text
            ).add_to(m)
        
        # Add a marker at the current position (simplified, just show start for now)
        # In a real demo, this would move along the path
        folium.CircleMarker(
            location=locations[v_info["Path"][0]], # Start of route
            radius=7,
            color=vehicle_colors[v_info["Vehicle"]],
            fill=True,
            fill_opacity=1.0,
            popup=f"{v_info['Vehicle']} (Total Time: {v_info['TotalTime']:.1f} hrs)"
        ).add_to(m)

    html(m._repr_html_(), height=500)

render_dynamic_map(
    routes_data,
    DEMO_LOCATIONS,
    st.session_state.disruption_segment,
    st.session_state.disruption_factor
)

st.subheader("Route Details")
for v_info in routes_data:
    st.write(f"**{v_info['Vehicle']}**")
    st.write(f"Current Path: {' → '.join(v_info['Path'])}")
    st.write(f"Total Estimated Time: {v_info['TotalTime']:.1f} hrs")
    st.write(f"Total Estimated Distance: {v_info['TotalDist']:.1f} miles")
    if v_info.get("Re-routed"):
        st.markdown("<span style='color: green; font-weight: bold;'>*Route dynamically adjusted!*</span>", unsafe_allow_html=True)
    st.markdown("---")
