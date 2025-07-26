import math
import random
from typing import List, Tuple

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="VRP — Multi‑Vehicle Route Optimization", layout="wide")
st.title("🚐 Vehicle Routing Problem (CVRP)")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_customers = st.slider("Number of customers", 5, 10, 6, 1)
    num_vehicles = st.slider("Number of vehicles", 1, 5, 2, 1)
    vehicle_capacity = st.slider("Vehicle capacity (units)", 50, 500, 150, 10)

# ---------------------------------------------------------
# US City Coordinates
# ---------------------------------------------------------
US_CITIES = [
    ("Atlanta", 33.7490, -84.3880),
    ("Chicago", 41.8781, -87.6298),
    ("Dallas", 32.7767, -96.7970),
    ("Denver", 39.7392, -104.9903),
    ("Los Angeles", 34.0522, -118.2437),
    ("Miami", 25.7617, -80.1918),
    ("New York", 40.7128, -74.0060),
    ("Seattle", 47.6062, -122.3321),
    ("San Francisco", 37.7749, -122.4194),
    ("Washington DC", 38.9072, -77.0369)
]

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

# ---------------------------------------------------------
# Cache Customers and Distance Matrix
# ---------------------------------------------------------
@st.cache_data
def generate_customers(n: int):
    """Randomly select n customers and return names, coords, and demands."""
    selected = random.sample(US_CITIES, n)
    names = [c[0] for c in selected]
    coords = [(c[1], c[2]) for c in selected]
    demands = np.random.randint(5, 31, size=n)
    return names, coords, demands

@st.cache_data
def build_distance_matrix(coords, depot):
    all_nodes = [depot] + coords
    n = len(all_nodes)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine_miles(*all_nodes[i], *all_nodes[j])
    return dist_matrix

@st.cache_data
def solve_vrp(dist_matrix, demands, capacity, num_vehicles):
    """Solve VRP with OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f, t = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(dist_matrix[f][t] * 1000)

    transit_cb_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else int(demands[node - 1])

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_cb_index, 0, [capacity] * num_vehicles, True, "Capacity")

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None

    routes, total_distance, vehicle_loads = [], 0, []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route, load, dist = [0], 0, 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                load += demands[node - 1]
            next_index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(next_index):
                next_node = manager.IndexToNode(next_index)
                dist += dist_matrix[node][next_node]
            route.append(manager.IndexToNode(next_index))
            index = next_index
        routes.append(route)
        total_distance += dist
        vehicle_loads.append(load)
    return {"routes": routes, "total_distance": total_distance, "vehicle_loads": vehicle_loads}

# ---------------------------------------------------------
# Generate Data
# ---------------------------------------------------------
depot_city = "Kansas City"
depot_coord = (39.0997, -94.5786)
customer_names, customers_coord, demands = generate_customers(num_customers)
dist_matrix = build_distance_matrix(customers_coord, depot_coord)
solution = solve_vrp(dist_matrix, demands, vehicle_capacity, num_vehicles)

if not solution:
    st.error("No feasible solution found. Try increasing vehicle capacity or vehicle count.")
    st.stop()

routes = solution["routes"]
total_distance = solution["total_distance"]

customers_df = pd.DataFrame({
    "CustomerID": [f"C{i+1:03d}" for i in range(num_customers)],
    "City": customer_names,
    "Lat": [c[0] for c in customers_coord],
    "Lon": [c[1] for c in customers_coord],
    "Demand": demands
})
st.subheader("📦 Selected Customers (Randomized)")
st.dataframe(customers_df, use_container_width=True)

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
def naive_chain_distance(coords, depot):
    order = [0] + list(range(1, len(coords) + 1)) + [0]
    return sum(haversine_miles(*([depot] + coords)[a], *([depot] + coords)[b])
               for a, b in zip(order[:-1], order[1:]))

baseline_distance = naive_chain_distance(customers_coord, depot_coord)
cost_per_mile = 2.0
cost_savings = (baseline_distance - total_distance) * cost_per_mile

st.subheader("📈 KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles used", f"{sum(1 for r in routes if len(r) > 2)} / {num_vehicles}")
c2.metric("Total optimized miles", f"{total_distance:,.1f}")
c3.metric("Baseline miles (naive chain)", f"{baseline_distance:,.1f}")
c4.metric("Distance reduction vs baseline", f"{baseline_distance - total_distance:,.1f} mi",
          f"{(baseline_distance - total_distance)/baseline_distance:.1%}")
st.markdown(f"**Estimated cost savings:** ${cost_savings:,.0f} (at ${cost_per_mile}/mile)")

# ---------------------------------------------------------
# Folium Map
# ---------------------------------------------------------
def render_folium_map(routes, depot, customers, demands, names):
    m = folium.Map(location=depot, zoom_start=4)
    folium.Marker(location=depot, popup=f"Depot: {depot_city}", icon=folium.Icon(color="black")).add_to(m)

    colors = ["blue", "green", "red", "purple", "orange"]
    all_nodes = [depot] + customers

    for i, c in enumerate(customers):
        folium.Marker(location=c,
                      popup=f"{names[i]} (Demand: {demands[i]})",
                      icon=folium.Icon(color="gray", icon="info-sign")).add_to(m)

    for v, route in enumerate(routes):
        route_coords = [all_nodes[n] for n in route]
        folium.PolyLine(route_coords, color=colors[v % len(colors)], weight=4,
                        tooltip=f"Vehicle {v+1} Route").add_to(m)
    return m

st.subheader("🗺 Optimized Vehicle Routes")
m = render_folium_map(routes, depot_coord, customers_coord, demands, customer_names)
st_folium(m, width=800, height=500)

# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

**Problem Statement:**  
Companies like UPS, FedEx, and Amazon face a **Vehicle Routing Problem (VRP)** daily:  
How do we deliver packages to multiple locations using a limited fleet while minimizing **fuel cost, miles driven, and delivery time**?

**Fun Fact:**  
UPS saved **10 million gallons of fuel per year** by optimizing routes and reducing left turns — a real-world VRP success!

**Key KPIs:**  
- Total optimized miles & vehicles used.  
- Distance & cost savings vs naive plan.  
- Fleet capacity utilization.

**Tech Stack:**  
- Python, Pandas, OR-Tools, Folium, Streamlit.

**Next Steps:**  
- Add **time windows (VRPTW)** and SLA adherence.  
- Include **heterogeneous fleets** (different vehicle types and costs).  
- Integrate **real cost models** (fuel, driver hours, tolls).  
- Extend to **multi-depot and cross-dock scenarios**.
""")
