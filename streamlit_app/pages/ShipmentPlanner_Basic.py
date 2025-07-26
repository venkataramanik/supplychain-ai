import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import random

st.set_page_config(page_title="Shipment Planner – Load Optimization", layout="wide")
st.title("🚛 Shipment Planner — Load Optimization (Randomized Data)")

# ------------------------------------------------------------
# 1) City list for random lanes
# ------------------------------------------------------------
city_coords = {
    "Atlanta, GA": [33.749, -84.388],
    "Dallas, TX": [32.7767, -96.797],
    "Chicago, IL": [41.8781, -87.6298],
    "Denver, CO": [39.7392, -104.9903],
    "Seattle, WA": [47.6062, -122.3321],
    "Boston, MA": [42.3601, -71.0589],
    "Miami, FL": [25.7617, -80.1918],
    "Los Angeles, CA": [34.0522, -118.2437],
    "New York, NY": [40.7128, -74.0060],
    "Houston, TX": [29.7604, -95.3698],
}

cities = list(city_coords.keys())

# ------------------------------------------------------------
# 2) Random Shipment Generator
# ------------------------------------------------------------
def generate_shipments(n=10):
    # Ensure 3 parcel shipments (<=150 lbs) and 7 LTL shipments (>150 lbs)
    weights = list(np.random.randint(30, 150, 3)) + list(np.random.randint(200, 800, n-3))
    random.shuffle(weights)

    data = []
    for i in range(n):
        origin, dest = random.sample(cities, 2)
        distance = np.random.randint(100, 2000)
        data.append({
            "ShipmentID": f"ORD-{i+1:03d}",
            "Origin": origin,
            "Destination": dest,
            "Distance (miles)": distance,
            "Weight (lbs)": weights[i],
        })
    return pd.DataFrame(data)

def nmfc_class(weight):
    if weight <= 150: return "70"
    if weight <= 500: return "55"
    return "50"

# ------------------------------------------------------------
# 3) Rating Functions
# ------------------------------------------------------------
def parcel_zone_multiplier(distance):
    if distance <= 150: return 1.0
    if distance <= 600: return 1.1
    if distance <= 1200: return 1.3
    return 1.5

def ltl_class_factor(nmfc):
    return {"50": 0.9, "55": 1.0, "70": 1.2}.get(nmfc, 1.0)

def rate_parcel(weight, distance):
    return (8 + 0.06 * weight) * parcel_zone_multiplier(distance)

def rate_ltl(weight, distance, nmfc):
    return 35 + 0.42 * distance * ltl_class_factor(nmfc)

# ------------------------------------------------------------
# 4) Scenario 1: Before Optimization
# ------------------------------------------------------------
def scenario_1_before(df):
    rows = []
    for _, r in df.iterrows():
        if r["Weight (lbs)"] <= 150:
            mode = "Parcel"
            cost = rate_parcel(r["Weight (lbs)"], r["Distance (miles)"])
        else:
            mode = "LTL"
            cost = rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"])
        rows.append({
            "LoadID": r["ShipmentID"],
            "Origin": r["Origin"],
            "Destination": r["Destination"],
            "Total Weight (lbs)": r["Weight (lbs)"],
            "Mode": mode,
            "Cost ($)": round(cost, 2)
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# 5) Scenario 2: Mode-Preserving Consolidation
# ------------------------------------------------------------
def scenario_2_mode_consolidated(before_df, ship_df):
    parcel = before_df[before_df["Mode"] == "Parcel"].copy()
    ltl = before_df[before_df["Mode"] == "LTL"]

    if len(ltl) == 0:
        return parcel.copy()

    lane = (
        ltl.groupby(["Origin", "Destination"], as_index=False)
            .agg({"Total Weight (lbs)": "sum"})
    )
    lane["Distance (miles)"] = lane.apply(
        lambda r: ship_df[(ship_df["Origin"] == r["Origin"]) &
                          (ship_df["Destination"] == r["Destination"])]["Distance (miles)"].mean(),
        axis=1
    )
    lane["Mode"] = "LTL"
    lane["Cost ($)"] = lane.apply(
        lambda r: round(rate_ltl(r["Total Weight (lbs)"], r["Distance (miles)"], nmfc_class(r["Total Weight (lbs)"])), 2),
        axis=1
    )
    lane["LoadID"] = [f"LTL-{i+1:02d}" for i in range(len(lane))]

    return pd.concat([parcel, lane[parcel.columns]], ignore_index=True)

# ------------------------------------------------------------
# 6) Scenario 3: Cross-Mode Consolidation
# ------------------------------------------------------------
def scenario_3_cross_mode(ship_df, scen2_df):
    lanes = ship_df.groupby(["Origin", "Destination"], as_index=False).size()[["Origin", "Destination"]]
    final_rows = []

    for _, lane_row in lanes.iterrows():
        o, d = lane_row["Origin"], lane_row["Destination"]
        lane_orders = ship_df[(ship_df["Origin"] == o) & (ship_df["Destination"] == d)]

        scen2_cost_lane = scen2_df[(scen2_df["Origin"] == o) &
                                   (scen2_df["Destination"] == d)]["Cost ($)"].sum()

        total_weight = lane_orders["Weight (lbs)"].sum()
        avg_distance = lane_orders["Distance (miles)"].mean()
        scen3_lane_cost_if_convert = rate_ltl(total_weight, avg_distance, nmfc_class(total_weight))

        if scen3_lane_cost_if_convert < scen2_cost_lane:
            final_rows.append({
                "LoadID": f"XMODE-{o[:3].upper()}-{d[:3].upper()}",
                "Origin": o,
                "Destination": d,
                "Total Weight (lbs)": total_weight,
                "Mode": "LTL",
                "Cost ($)": round(scen3_lane_cost_if_convert, 2)
            })
        else:
            final_rows.extend(
                scen2_df[(scen2_df["Origin"] == o) &
                         (scen2_df["Destination"] == d)].to_dict(orient="records")
            )
    return pd.DataFrame(final_rows)

# ------------------------------------------------------------
# 7) Mapping Helpers
# ------------------------------------------------------------
def to_routes_df(loads_df, mode_color):
    rows = []
    for _, r in loads_df.iterrows():
        color = mode_color.get(r["Mode"], [200, 200, 200])
        rows.append({
            "from_lat": city_coords[r["Origin"]][0],
            "from_lon": city_coords[r["Origin"]][1],
            "to_lat": city_coords[r["Destination"]][0],
            "to_lon": city_coords[r["Destination"]][1],
            "color": color,
        })
    return pd.DataFrame(rows)

def render_map(loads_df, mode_color):
    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("ArcLayer",
                      data=to_routes_df(loads_df, mode_color),
                      get_source_position=["from_lon", "from_lat"],
                      get_target_position=["to_lon", "to_lat"],
                      get_width=2,
                      get_source_color="color",
                      get_target_color="color")
        ],
        initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=4, pitch=20),
        height=320
    ))

# ------------------------------------------------------------
# 8) Run Logic
# ------------------------------------------------------------
if st.button("Run Optimization"):
    shipments = generate_shipments()
    shipments["NMFC Class"] = shipments["Weight (lbs)"].apply(nmfc_class)
    st.subheader("📦 Orders (Randomized)")
    st.dataframe(shipments, use_container_width=True)

    scen1 = scenario_1_before(shipments)
    scen2 = scenario_2_mode_consolidated(scen1, shipments)
    scen3 = scenario_3_cross_mode(shipments, scen2)

    cost1, cost2, cost3 = scen1["Cost ($)"].sum(), scen2["Cost ($)"].sum(), scen3["Cost ($)"].sum()
    loads1, loads2, loads3 = len(scen1), len(scen2), len(scen3)

    st.subheader("📈 Scenario KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("1) Before", f"${cost1:,.2f}", f"Loads: {loads1}")
    c2.metric("2) Mode-Consolidated", f"${cost2:,.2f}", f"Loads: {loads2} | Δ ${(cost1 - cost2):,.2f}")
    c3.metric("3) Cross-Mode", f"${cost3:,.2f}", f"Loads: {loads3} | Δ ${(cost2 - cost3):,.2f}")

    st.subheader("📊 Cost & Load Comparison")
    cost_df = pd.DataFrame({
        "Scenario": ["Before", "Mode-Consolidated", "Cross-Mode"],
        "Total Cost ($)": [cost1, cost2, cost3],
        "Total Loads": [loads1, loads2, loads3]
    })
    col1, col2 = st.columns(2)
    col1.altair_chart(
        alt.Chart(cost_df).mark_bar(size=40).encode(
            x="Scenario:N", y="Total Cost ($):Q", color="Scenario:N"
        ).properties(width=280, height=280, title="Total Cost by Scenario"),
        use_container_width=False
    )
    col2.altair_chart(
        alt.Chart(cost_df).mark_bar(size=40).encode(
            x="Scenario:N", y="Total Loads:Q", color="Scenario:N"
        ).properties(width=280, height=280, title="Total Loads by Scenario"),
        use_container_width=False
    )

    st.subheader("🗺 Load Maps")
    tab1, tab2, tab3 = st.tabs(["Before", "Mode-Consolidated", "Cross-Mode"])
    with tab1:
        render_map(scen1, {"Parcel": [0, 102, 255], "LTL": [255, 128, 0]})
    with tab2:
        render_map(scen2, {"Parcel": [0, 102, 255], "LTL": [255, 128, 0]})
    with tab3:
        render_map(scen3, {"LTL": [255, 0, 0]})

    st.subheader("📋 Load Details")
    st.write("**Before Optimization:**")
    st.dataframe(scen1, use_container_width=True)
    st.write("**Mode-Consolidated:**")
    st.dataframe(scen2, use_container_width=True)
    st.write("**Cross-Mode Consolidated:**")
    st.dataframe(scen3, use_container_width=True)
