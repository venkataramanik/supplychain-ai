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
    seed = st.number_input("Random seed (reproducibility)", value=42, step=1)

random.seed(seed)
np.random.seed(seed)

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
# Generate Customer Data
# ---------------------------------------------------------
def generate_customers(n: int) -> List[Tuple[str, float, float]]:
    return random.sample(US_CITIES, n)

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

depot_city = "Kansas City"
depot_coord = (39.0997, -94.5786)
customers_selected = generate_customers(num_customers)
customer_names = [c[0] for c in customers_selected]
customers_coord = [(c[1], c[2]) for c in customers_selected]
demands = np.random.randint(5, 31, size=num_customers)

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
# OR-Tools VRP Solver
# ---------------------------------------------------------
def build_distance_matrix(coords, depot):
    all_nodes = [depot] + coords
    n = len(all_nodes)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine_miles(*all_nodes[i], *all_nodes[j])
    return dist_matrix

def solve_cvrp(dist_matrix, demands, capacity, num_vehicles):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
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
    search_params.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None, manager, routing

    routes, total_distance, vehicle_loads = [], 0, []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = [0]
        load = 0
        dist = 0
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
    return {"routes": routes, "total_distance": total_distance, "vehicle_loads": vehicle_loads}, manager, routing

dist_matrix = build_distance_matrix(customers_coord, depot_coord)
solution, manager, routing = solve_cvrp(dist_matrix, demands, vehicle_capacity, num_vehicles)

if not solution:
    st.error("No feasible solution found. Try increasing vehicle capacity or vehicle count.")
    st.stop()

routes = solution["routes"]
total_distance = solution["total_distance"]

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
def naive_chain_distance(coords, depot):
    order = [0] + list(range(1, len(coords) + 1)) + [0]
    dist = 0.0
    all_nodes = [depot] + coords
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        dist += haversine_miles(*all_nodes[a], *all_nodes[b])
    return dist

baseline_distance = naive_chain_distance(customers_coord, depot_coord)
cost_per_mile = 2.0
cost_baseline = baseline_distance * cost_per_mile
cost_optimized = total_distance * cost_per_mile
cost_savings = cost_baseline - cost_optimized

st.subheader("📈 KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles used", f"{sum(1 for r in routes if len(r) > 2)} / {num_vehicles}")
c2.metric("Total optimized miles", f"{total_distance:,.1f}")
c3.metric("Baseline miles (naive chain)", f"{baseline_distance:,.1f}")
c4.metric("Distance reduction vs baseline", f"{baseline_distance - total_distance:,.1f} mi",
          f"{(baseline_distance - total_distance)/baseline_distance:.1%}")
st.markdown(f"**Estimated cost savings:** ${cost_savings:,.0f} (at ${cost_per_mile}/mile)")

# Distance Comparison Chart
st.subheader("📊 Before vs Optimized Distance")
distance_df = pd.DataFrame({
    "Scenario": ["Naive (Single Tour)", "Optimized VRP"],
    "Miles": [baseline_distance, total_distance]
})
st.altair_chart(
    alt.Chart(distance_df).mark_bar().encode(
        x="Scenario:N",
        y="Miles:Q",
        color="Scenario:N"
    ).properties(width=400, height=300),
    use_container_width=True
)

# ---------------------------------------------------------
# Folium Map
# ---------------------------------------------------------
def render_folium_map(routes, depot, customers, demands, customer_names):
    m = folium.Map(location=depot, zoom_start=4)
    folium.Marker(location=depot, popup=f"Depot: {depot_city}", icon=folium.Icon(color="black")).add_to(m)

    colors = ["blue", "green", "red", "purple", "orange"]
    all_nodes = [depot] + customers

    # Add customer markers
    for i, c in enumerate(customers):
        folium.Marker(location=c,
                      popup=f"{customer_names[i]} (Demand: {demands[i]})",
                      icon=folium.Icon(color="gray", icon="info-sign")).add_to(m)

    # Add routes
    for v, route in enumerate(routes):
        route_coords = [all_nodes[n] for n in route]
        folium.PolyLine(route_coords, color=colors[v % len(colors)], weight=4,
                        tooltip=f"Vehicle {v+1} Route").add_to(m)
    return m

st.subheader("🗺 Optimized Vehicle Routes (Real Map)")
m = render_folium_map(routes, depot_coord, customers_coord, demands, customer_names)
st_folium(m, width=800, height=500)

# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

**Problem Statement:**  
Every day, companies like UPS, FedEx, and Amazon face a **Vehicle Routing Problem (VRP)**:  
How do we deliver packages to multiple locations using a limited fleet while minimizing **fuel cost, miles driven, and delivery time**?

**Baseline Explanation:**  
The "baseline miles" is a naive single-route plan, visiting every city in sequence and returning to the depot. This is not optimized but shows how much savings true optimization can deliver.

**Key KPIs:**  
- **Total optimized miles** and **vehicles used**.  
- **Distance and cost savings vs naive plan**.  
- **Fleet capacity utilization** (how full the trucks are).

**Fun Fact:**  
UPS saved **10 million gallons of fuel per year** by optimizing routes and eliminating unnecessary left turns — a famous real-world VRP success story!

**Tech Stack & Tools:**  
- **Python & Pandas** — data manipulation.  
- **Google OR-Tools** — industry-grade solver for VRP and TSP.  
- **Folium** — interactive, road-style map visualization.  
- **Streamlit** — fast web apps and dashboards.

**Next Steps:**  
- Add **time windows (VRPTW)** and SLA adherence.  
- Include **heterogeneous fleets** (different vehicle types and costs).  
- Integrate **real cost models** (fuel, driver hours, tolls).  
- Extend to **multi-depot and cross-dock scenarios**.
""")
