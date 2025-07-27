import random
import math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="VRPTW — Vehicle Routing with Time Windows", layout="wide")
st.title("Vehicle Routing with Time Windows (VRPTW)")

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
    start_times = np.random.randint(8, 16, size=n)  # time window start in hours
    end_times = start_times + np.random.randint(1, 4, size=n)  # time window end in hours
    service_times = np.random.randint(15, 45, size=n)  # service time in minutes
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
# Create Distance Matrix
# ---------------------------------------------------------
def create_distance_matrix(n: int):
    rng = np.random.default_rng(random_seed)
    matrix = rng.integers(5, 40, size=(n + 1, n + 1))  # depot + n customers
    np.fill_diagonal(matrix, 0)
    return matrix.tolist()

distance_matrix = create_distance_matrix(num_customers)

# ---------------------------------------------------------
# VRPTW Solver
# ---------------------------------------------------------
def solve_vrptw(distance_matrix,
                demands,
                starts,
                ends,
                service_times,
                num_vehicles,
                capacity,
                horizon_minutes: int = 24 * 60):

    n_customers = len(demands)
    n_nodes = n_customers + 1  # depot + customers

    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(distance_matrix[f][t])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else int(demands[node - 1])
    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index, 0, [int(capacity)] * num_vehicles, True, "Capacity"
    )

    def time_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        travel = distance_matrix[f][manager.IndexToNode(to_index)]
        service = 0 if f == 0 else service_times[f - 1]
        return int(travel + service)

    time_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_index,
        slack_max=30,
        capacity=horizon_minutes,
        start_cumul_to_zero=False,
        name="Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Depot time window
    for v in range(num_vehicles):
        start_idx = routing.Start(v)
        end_idx = routing.End(v)
        time_dim.CumulVar(start_idx).SetRange(0, horizon_minutes)
        time_dim.CumulVar(end_idx).SetRange(0, horizon_minutes)

    # Customer time windows
    for cust in range(1, n_nodes):
        start_min = int(starts[cust - 1] * 60)
        end_min = int(ends[cust - 1] * 60)
        index = manager.NodeToIndex(cust)
        time_dim.CumulVar(index).SetRange(start_min, end_min)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        veh_route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            t = solution.Value(time_dim.CumulVar(idx))
            veh_route.append((node, t))
            idx = solution.Value(routing.NextVar(idx))
        node = manager.IndexToNode(idx)
        t = solution.Value(time_dim.CumulVar(idx))
        veh_route.append((node, t))
        routes.append(veh_route)

    return routes

# ---------------------------------------------------------
# Solve
# ---------------------------------------------------------
routes = solve_vrptw(
    distance_matrix=distance_matrix,
    demands=demands,
    starts=start_times,
    ends=end_times,
    service_times=service_times,
    num_vehicles=num_vehicles,
    capacity=vehicle_capacity
)

if not routes:
    st.error("No feasible solution found. Try adjusting vehicle capacity or time windows.")
    st.stop()

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
st.subheader("📈 KPIs")
total_distance = sum(
    sum(distance_matrix[routes[v][i][0]][routes[v][i + 1][0]]
        for i in range(len(routes[v]) - 1))
    for v in range(num_vehicles) if len(routes[v]) > 1
)
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
