import random
import math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="VRPTW — Vehicle Routing with Time Windows", layout="wide")
st.title("⏱ Pilot 3 — Vehicle Routing with Time Windows (VRPTW)")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario Controls")
    num_customers = st.slider("Number of customers", 5, 10, 6, 1)
    num_vehicles = st.slider("Number of vehicles", 1, 5, 2, 1)
    vehicle_capacity = st.slider("Vehicle capacity (units)", 50, 500, 150, 10)
    random_seed = st.number_input("Random seed", value=42, step=1)

random.seed(random_seed)
np.random.seed(random_seed)

# ---------------------------------------------------------
# Generate Customer Data with Time Windows
# ---------------------------------------------------------
CITIES = [
    "Atlanta", "Chicago", "Dallas", "Denver",
    "Los Angeles", "Miami", "New York", "Seattle",
    "San Francisco", "Washington DC"
]

def generate_customers(n: int):
    selected = random.sample(CITIES, n)
    demands = np.random.randint(5, 31, size=n)
    start_times = np.random.randint(8, 16, size=n)  # start window between 8AM-4PM
    end_times = start_times + np.random.randint(1, 4, size=n)  # 1-3 hr windows
    service_times = np.random.randint(15, 45, size=n)  # 15-45 min per stop
    return selected, demands, start_times, end_times, service_times

customer_names, demands, start_times, end_times, service_times = generate_customers(num_customers)

customers_df = pd.DataFrame({
    "CustomerID": [f"C{i+1:03d}" for i in range(num_customers)],
    "City": customer_names,
    "Demand": demands,
    "TimeWindow": [f"{s}:00–{e}:00" for s, e in zip(start_times, end_times)],
    "ServiceTime (min)": service_times
})
st.subheader("📦 Customers & Time Windows")
st.dataframe(customers_df, use_container_width=True)

# ---------------------------------------------------------
# VRPTW Solver using OR-Tools
# ---------------------------------------------------------
def create_distance_matrix(n: int):
    # Simple synthetic distances
    rng = np.random.default_rng(random_seed)
    matrix = rng.integers(5, 40, size=(n + 1, n + 1))  # depot + n customers
    np.fill_diagonal(matrix, 0)
    return matrix.tolist()

distance_matrix = create_distance_matrix(num_customers)

def solve_vrptw(distance_matrix, demands, starts, ends, service_times, num_vehicles, capacity):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    # Capacity constraint
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else demands[node - 1]
    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_index, 0, [capacity] * num_vehicles, True, "Capacity")

    # Time Windows
    def time_callback(from_index, to_index):
        return distance_callback(from_index, to_index) + (service_times[manager.IndexToNode(from_index) - 1]
                                                          if manager.IndexToNode(from_index) > 0 else 0)
    time_index = routing.RegisterTransitCallback(time_callback)
    horizon = 24 * 60  # 24 hours in minutes
    routing.AddDimension(time_index, 30, horizon, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for i in range(1, len(distance_matrix)):
        start = starts[i - 1] * 60
        end = ends[i - 1] * 60
        time_dimension.CumulVar(manager.NodeToIndex(i)).SetRange(start, end)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None

    # Extract routes
    routes = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        vehicle_route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            time_min = solution.Min(time_var)
            vehicle_route.append((node, time_min))
            index = solution.Value(routing.NextVar(index))
        routes.append(vehicle_route)
    return routes

routes = solve_vrptw(distance_matrix, demands, start_times, end_times, service_times, num_vehicles, vehicle_capacity)

if not routes:
    st.error("No feasible solution found. Try adjusting vehicle capacity or time windows.")
    st.stop()

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
st.subheader("📈 KPIs")
total_distance = sum(sum(distance_matrix[routes[v][i][0]][routes[v][i+1][0]]
                         for i in range(len(routes[v]) - 1))
                     for v in range(num_vehicles) if len(routes[v]) > 1)
vehicles_used = sum(1 for v in routes if len(v) > 1)
st.metric("Vehicles used", f"{vehicles_used} / {num_vehicles}")
st.metric("Total distance (approx)", f"{total_distance} units")

# ---------------------------------------------------------
# Build Gantt Chart Data
# ---------------------------------------------------------
gantt_data = []
for v, route in enumerate(routes):
    for step in route[1:]:
        node, start_time = step
        if node == 0:
            continue
        gantt_data.append({
            "Vehicle": f"Vehicle {v+1}",
            "Customer": customer_names[node - 1],
            "Start": start_time,
            "End": start_time + service_times[node - 1]
        })

gantt_df = pd.DataFrame(gantt_data)
st.subheader("⏳ Delivery Schedule (Gantt Chart)")
chart = alt.Chart(gantt_df).mark_bar().encode(
    x='Start:Q',
    x2='End:Q',
    y='Vehicle:N',
    color='Vehicle:N',
    tooltip=['Customer', 'Start', 'End']
).properties(height=300)
st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.markdown("""
## 📖 Business Context & Highlights

**Problem Statement:**  
Many delivery services have **strict time windows** (e.g., 9 AM–12 PM), making routing far more complex.  
The goal is to minimize distance while **meeting all time windows**.

**Key KPIs:**  
- Number of vehicles used.  
- Total travel distance.  
- SLA compliance (on-time deliveries).  

**Fun Fact:**  
Amazon’s “Same Day” delivery and grocery logistics are **real-world VRPTW problems** — they require aligning driver routes with customer availability.

**Tech Stack & Tools:**  
- Python (data & algorithms).  
- OR-Tools (VRPTW solver).  
- Altair (Gantt chart visualization).  
- Streamlit (UI).

**Next Steps:**  
- Add penalty costs for late deliveries.  
- Integrate realistic travel times (using mapping APIs).  
- Optimize for driver shifts & breaks.
""")
