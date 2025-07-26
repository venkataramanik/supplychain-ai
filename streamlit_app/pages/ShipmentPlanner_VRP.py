import math
import random
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="VRP — Multi‑Vehicle Route Optimization", layout="wide")
st.title("🚐 Pilot 2 — Vehicle Routing Problem (CVRP) with Step-by-step ‘Animation’")

# ---------------------------------------------------------
# 0) Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_customers = st.slider("Number of customers", 8, 40, 15, 1)
    num_vehicles = st.slider("Number of vehicles", 1, 8, 3, 1)
    vehicle_capacity = st.slider("Vehicle capacity (units)", 50, 500, 150, 10)
    seed = st.number_input("Random seed (reproducibility)", value=42, step=1)
    animate = st.checkbox("Enable step-by-step route playback (slider)", value=True)

random.seed(seed)
np.random.seed(seed)

# ---------------------------------------------------------
# 1) Generate data
# ---------------------------------------------------------
# We'll place customers randomly in the continental US-ish bounding box
# and fix the depot around Kansas City to visually sit near center.
def generate_points(n: int, seed: int):
    rng = np.random.default_rng(seed)
    # Rough US bounding box
    lats = rng.uniform(28.0, 48.5, n)     # latitude
    lons = rng.uniform(-122.0, -70.0, n)  # longitude
    return list(zip(lats, lons))

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    # Haversine distance in miles
    R = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2.0)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2)
    return 2 * R * math.asin(math.sqrt(a))

# Depot fixed near Kansas City, MO
depot_coord = (39.0997, -94.5786)
customers_coord = generate_points(num_customers, seed=seed)

# Random demands between 5 and 30 units
demands = np.random.randint(5, 31, size=num_customers)

# Build DataFrame for display
customers_df = pd.DataFrame({
    "CustomerID": [f"C{i:03d}" for i in range(num_customers)],
    "Lat": [c[0] for c in customers_coord],
    "Lon": [c[1] for c in customers_coord],
    "Demand": demands
})

st.subheader("📦 Customers")
st.dataframe(customers_df, use_container_width=True)

# ---------------------------------------------------------
# 2) OR-Tools CVRP model
# ---------------------------------------------------------
def build_distance_matrix(coords: List[Tuple[float, float]], depot: Tuple[float, float]):
    # Node 0 = depot
    all_nodes = [depot] + coords
    n = len(all_nodes)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist = 0
            else:
                dist = haversine_miles(all_nodes[i][0], all_nodes[i][1],
                                       all_nodes[j][0], all_nodes[j][1])
            dist_matrix[i][j] = dist
    return dist_matrix

def solve_cvrp(dist_matrix, demands, capacity, num_vehicles):
    """Solve CVRP with OR-Tools. Node 0 = depot."""
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(dist_matrix[f][t] * 1000)  # scale to int

    transit_cb_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    # Capacity constraint (demands apply to customers; depot demand=0)
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        if node == 0:
            return 0
        return int(demands[node-1])  # node-1 because customers start at 1

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index, 0, [capacity]*num_vehicles, True, "Capacity"
    )

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None, manager, routing

    # Extract routes
    routes = []
    total_distance = 0
    total_loads = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = [0]  # start at depot
        load = 0
        distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                load += demands[node-1]
            next_index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(next_index):
                next_node = manager.IndexToNode(next_index)
                distance += dist_matrix[node][next_node]
            route.append(manager.IndexToNode(next_index))
            index = next_index
        routes.append(route)
        total_distance += distance
        total_loads.append(load)

    return {
        "routes": routes,
        "total_distance": total_distance,
        "vehicle_loads": total_loads,
    }, manager, routing

dist_matrix = build_distance_matrix(customers_coord, depot_coord)
solution, manager, routing = solve_cvrp(dist_matrix, demands, vehicle_capacity, num_vehicles)

if solution is None:
    st.error("No solution found. Try increasing vehicle capacity or vehicles.")
    st.stop()

routes = solution["routes"]
total_distance = solution["total_distance"]
veh_loads = solution["vehicle_loads"]

# ---------------------------------------------------------
# 3) KPIs
# ---------------------------------------------------------
st.subheader("📈 KPIs")

# Naive single-vehicle TSP distance (just chain customers in order for reference)
# (not a true TSP optimum, but a deterministic “baseline”)
def naive_chain_distance(coords, depot):
    order = [0] + list(range(1, len(coords)+1)) + [0]
    dist = 0.0
    all_nodes = [depot] + coords
    for i in range(len(order)-1):
        a, b = order[i], order[i+1]
        dist += haversine_miles(all_nodes[a][0], all_nodes[a][1],
                                all_nodes[b][0], all_nodes[b][1])
    return dist

baseline_distance = naive_chain_distance(customers_coord, depot_coord)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles used", f"{sum(1 for r in routes if len(r) > 2)} / {num_vehicles}")
c2.metric("Total optimized miles", f"{total_distance:,.1f}")
c3.metric("Baseline miles (naive chain)", f"{baseline_distance:,.1f}")
c4.metric("Distance reduction vs baseline", f"{baseline_distance - total_distance:,.1f} mi",
          f"{(baseline_distance - total_distance)/baseline_distance:.1%}")

# ---------------------------------------------------------
# 4) Route tables
# ---------------------------------------------------------
def routes_to_df(routes, demands):
    rows = []
    for v, r in enumerate(routes):
        # r is list of node indices including depot=0
        load = 0
        for i in range(len(r)-1):
            fr, to = r[i], r[i+1]
            fr_label = "Depot" if fr == 0 else f"C{fr:03d}"
            to_label = "Depot" if to == 0 else f"C{to:03d}"
            if to != 0:
                load += demands[to-1]
            rows.append({
                "Vehicle": v,
                "From": fr_label,
                "To": to_label,
                "Leg #": i,
                "Cum Load": load
            })
    return pd.DataFrame(rows)

routes_df = routes_to_df(routes, demands)
st.subheader("📋 Route detail (order of visits)")
st.dataframe(routes_df, use_container_width=True)

# ---------------------------------------------------------
# 5) “Animation” – interactive step slider
# ---------------------------------------------------------
# Build a structure: for each vehicle, list the (lat, lon) in visiting order
all_nodes = [depot_coord] + customers_coord

def route_coords(route):
    return [(all_nodes[n][0], all_nodes[n][1]) for n in route]

veh_paths = [route_coords(r) for r in routes]

# The number of steps is the max legs among vehicles
max_legs = max(len(p) - 1 for p in veh_paths)

if animate:
    step = st.slider("🕒 Step through the routes", 0, max_legs, 0)
else:
    step = max_legs  # show full routes if animation is off

# ---------------------------------------------------------
# 6) Plotly “animated” map (driven by the slider)
# ---------------------------------------------------------
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

fig = go.Figure()

# Draw depot
fig.add_trace(go.Scattergeo(
    lon=[depot_coord[1]], lat=[depot_coord[0]],
    mode="markers",
    marker=dict(size=12, color="black", symbol="star"),
    name="Depot",
    hoverinfo="text",
    text=["Depot"],
))

# Draw customer points
fig.add_trace(go.Scattergeo(
    lon=[c[1] for c in customers_coord],
    lat=[c[0] for c in customers_coord],
    mode="markers",
    marker=dict(size=6, color="gray"),
    name="Customers",
    hoverinfo="text",
    text=[f"C{i:03d} | demand={demands[i]}" for i in range(num_customers)]
))

# Draw polylines per vehicle up to current step
for v, path in enumerate(veh_paths):
    color = colors[v % len(colors)]
    # Plot segments up to 'step'
    segments = min(step, len(path)-1)
    for i in range(segments):
        lat_pair = [path[i][0], path[i+1][0]]
        lon_pair = [path[i][1], path[i+1][1]]
        fig.add_trace(go.Scattergeo(
            lon=lon_pair, lat=lat_pair,
            mode="lines+markers",
            line=dict(width=3, color=color),
            marker=dict(size=4, color=color),
            name=f"Vehicle {v}",
            hoverinfo="text",
            text=[f"V{v}: leg {i}", f"V{v}: leg {i+1}"]
        ))

fig.update_layout(
    geo=dict(
        scope="north america",
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(245, 245, 245)",
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(217, 217, 217)",
        countrycolor="rgb(217, 217, 217)",
    ),
    legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7) Summaries and charts
# ---------------------------------------------------------
st.subheader("📊 Vehicle load utilization")
util_df = pd.DataFrame({
    "Vehicle": [f"V{v}" for v in range(num_vehicles)],
    "Load": [l for l in veh_loads],
    "Capacity": [vehicle_capacity] * num_vehicles
})
util_df["Utilization %"] = util_df["Load"] / util_df["Capacity"]

c_chart1, c_chart2 = st.columns(2)

c_chart1.altair_chart(
    alt.Chart(util_df).mark_bar().encode(
        x="Vehicle:N",
        y=alt.Y("Load:Q", title="Load (units)"),
        color="Vehicle:N",
        tooltip=["Vehicle", "Load", "Capacity", alt.Tooltip("Utilization %:Q", format=".1%")]
    ).properties(title="Load per vehicle", width=350, height=280),
    use_container_width=False
)

c_chart2.altair_chart(
    alt.Chart(util_df).mark_bar().encode(
        x="Vehicle:N",
        y=alt.Y("Utilization %:Q", axis=alt.Axis(format="%")),
        color="Vehicle:N",
        tooltip=["Vehicle", alt.Tooltip("Utilization %:Q", format=".1%")]
    ).properties(title="Capacity utilization", width=350, height=280),
    use_container_width=False
)

# ---------------------------------------------------------
# 8) Business context & tech explanation
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context

**Problem**  
You have a fleet of vehicles with **limited capacity** that must serve a set of customers with different **demands**.  
The goal is to **minimize total miles (or time/cost)** while respecting **capacity constraints**.

**Where it’s used**  
- Parcel / LTL last-mile routing  
- Retail store replenishment  
- Field service technician routing  
- Milk-run collections in manufacturing

**KPIs you typically track**  
- Total miles / drive time  
- # vehicles used vs available  
- Capacity utilization (how “full” trucks are)  
- Service-level adherence (time windows, SLAs — extensions below)

**Fun fact**  
VRP generalizes the TSP, so it is **NP-hard**. That’s why solvers like **OR-Tools** deploy smart heuristics, local searches, and metaheuristics to get very good solutions quickly.

---

## ⚙️ Tech & Tools

- **Python** — glue language and data handling  
- **Google OR-Tools** — industrial-strength optimization for VRP, CVRP, VRPTW, etc.  
- **Pandas / NumPy** — random data generation, KPIs, tabular route outputs  
- **Plotly** — interactive, “animated” route playback via slider  
- **Streamlit** — fast UI to iterate, explain, and deploy

---

## 🚀 What you could add next

- **Time Windows (VRPTW):** each customer has an [earliest, latest] time to be served  
- **Multiple depots / cross-dock**  
- **Heterogeneous fleet:** different capacities, costs, speeds  
- **Driver shift limits / Hours-of-service**  
- **Pickup & Delivery / Backhauls**  
- **Multi-objective optimization:** (cost, miles, CO₂, SLA penalties)  
- **Learning-augmented routing:** use **ML** to predict service times / ETAs, then optimize  
- **Agentic orchestration:** combine LLMs for “what scenario to run?” with OR-Tools for “how to solve it optimally?”
""")
