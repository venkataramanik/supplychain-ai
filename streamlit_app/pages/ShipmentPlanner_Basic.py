import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Shipment Planner – Load Optimization", layout="wide")
st.title("🚛 Shipment Planner – Load Optimization Demo")

# ----------------- Helper Data -----------------
city_coords = {
    "Atlanta, GA": [33.749, -84.388],
    "Dallas, TX": [32.7767, -96.797],
    "Chicago, IL": [41.8781, -87.6298],
    "Denver, CO": [39.7392, -104.9903],
    "Seattle, WA": [47.6062, -122.3321],
    "Boston, MA": [42.3601, -71.0589],
    "Miami, FL": [25.7617, -80.1918],
    "San Jose, CA": [37.3382, -121.8863],
    "Phoenix, AZ": [33.4484, -112.0740],
    "Nashville, TN": [36.1627, -86.7816],
    "New York, NY": [40.7128, -74.0060],
    "Houston, TX": [29.7604, -95.3698],
    "Los Angeles, CA": [34.0522, -118.2437]
}

# ----------------- Generate Sample Data -----------------
np.random.seed(42)
origins = np.random.choice(list(city_coords.keys()), 20)
destinations = np.random.choice(list(city_coords.keys()), 20)
weights = np.concatenate([np.random.randint(20, 150, 10), np.random.randint(200, 800, 10)])
np.random.shuffle(weights)
distances = np.random.randint(100, 2500, 20)

shipments = pd.DataFrame({
    "ShipmentID": [f"ORD-{i:03d}" for i in range(1, 21)],
    "Origin": origins,
    "Destination": destinations,
    "Distance (miles)": distances,
    "Weight (lbs)": weights
})

def get_nmfc_class(weight):
    if weight <= 150:
        return "70"
    elif weight <= 500:
        return "55"
    return "50"

shipments["NMFC Class"] = shipments["Weight (lbs)"].apply(get_nmfc_class)

st.subheader("📦 Orders (Before Optimization)")
st.dataframe(shipments, use_container_width=True)

# ----------------- Rating Functions -----------------
def parcel_zone_multiplier(distance):
    if distance <= 150:
        return 1.0
    elif distance <= 600:
        return 1.1
    elif distance <= 1200:
        return 1.3
    return 1.5

def ltl_class_factor(nmfc_class):
    return {"50": 0.9, "55": 1.0, "70": 1.2}.get(nmfc_class, 1.0)

def rate_parcel(weight, distance):
    return (8 + 0.06 * weight) * parcel_zone_multiplier(distance)

def rate_ltl(weight, distance, nmfc_class):
    return 35 + (0.42 * distance * ltl_class_factor(nmfc_class))

# ----------------- Load-Building Scenarios -----------------
def scenario_1_before(df):
    """Each shipment = 1 load, assigned Parcel or LTL by weight."""
    loads = []
    for _, r in df.iterrows():
        mode = "Parcel" if r["Weight (lbs)"] <= 150 else "LTL"
        cost = rate_parcel(r["Weight (lbs)"], r["Distance (miles)"]) if mode == "Parcel" \
            else rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"])
        loads.append({
            "LoadID": r["ShipmentID"],
            "Origin": r["Origin"],
            "Destination": r["Destination"],
            "Total Weight": r["Weight (lbs)"],
            "Mode": mode,
            "Cost ($)": round(cost, 2)
        })
    return pd.DataFrame(loads)

def scenario_2_consolidated(before_df):
    """Consolidate LTL loads by lane; Parcel stays as separate loads."""
    parcel_loads = before_df[before_df["Mode"] == "Parcel"].copy()
    ltl_df = before_df[before_df["Mode"] == "LTL"]

    ltl_consol = ltl_df.groupby(["Origin", "Destination"], as_index=False).agg(
        {"Total Weight": "sum"}
    )
    ltl_consol["Mode"] = "LTL"
    ltl_consol["Cost ($)"] = ltl_consol.apply(
        lambda r: rate_ltl(
            r["Total Weight"],
            shipments[(shipments["Origin"]==r["Origin"]) & (shipments["Destination"]==r["Destination"])]["Distance (miles)"].mean(),
            get_nmfc_class(r["Total Weight"])
        ), axis=1
    )
    ltl_consol["LoadID"] = [f"LTL-{i+1:02d}" for i in range(len(ltl_consol))]

    final_df = pd.concat([parcel_loads, ltl_consol], ignore_index=True)
    return final_df

def scenario_3_cross_mode(df):
    """Evaluate if moving parcel shipments to LTL and consolidating reduces cost."""
    lane = df.groupby(["Origin", "Destination"], as_index=False).agg({"Weight (lbs)": "sum"})
    lane["Mode"] = "LTL"
    lane["Cost ($)"] = lane.apply(
        lambda r: rate_ltl(
            r["Weight (lbs)"],
            df[(df["Origin"]==r["Origin"]) & (df["Destination"]==r["Destination"])]["Distance (miles)"].mean(),
            get_nmfc_class(r["Weight (lbs)"])
        ), axis=1
    )
    lane.rename(columns={"Weight (lbs)": "Total Weight"}, inplace=True)
    lane["LoadID"] = [f"OPT-{i+1:02d}" for i in range(len(lane))]
    return lane

# ----------------- Map Helper -----------------
def render_map(loads_df, color_map):
    data = []
    for _, r in loads_df.iterrows():
        color = color_map.get(r["Mode"], [200, 200, 200])
        data.append({
            "from_lat": city_coords[r["Origin"]][0],
            "from_lon": city_coords[r["Origin"]][1],
            "to_lat": city_coords[r["Destination"]][0],
            "to_lon": city_coords[r["Destination"]][1],
            "color": color
        })
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ArcLayer", data=data,
                          get_source_position=["from_lon", "from_lat"],
                          get_target_position=["to_lon", "to_lat"],
                          get_width=2, get_source_color="color", get_target_color="color")],
        initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=4, pitch=20),
        height=350
    ))

# ----------------- Run Scenarios -----------------
if st.button("Run Optimization"):
    before = scenario_1_before(shipments)
    mode_consol = scenario_2_consolidated(before)
    cross_mode = scenario_3_cross_mode(shipments)

    cost_before = before["Cost ($)"].sum()
    cost_mode = mode_consol["Cost ($)"].sum()
    cost_cross = cross_mode["Cost ($)"].sum()

    # KPIs
    st.subheader("📈 Scenario KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("1) Before Optimization", f"${cost_before:,.2f}", f"Loads: {len(before)}")
    c2.metric("2) Mode-Consolidated", f"${cost_mode:,.2f}", f"Loads: {len(mode_consol)}")
    c3.metric("3) Cross-Mode Consolidated", f"${cost_cross:,.2f}", f"Loads: {len(cross_mode)}")

    # Combined Data
    cost_df = pd.DataFrame({
        "Scenario": ["Before", "Mode-Consolidated", "Cross-Mode"],
        "Total Cost ($)": [cost_before, cost_mode, cost_cross],
        "Total Loads": [len(before), len(mode_consol), len(cross_mode)]
    })

    # ----------------- Charts -----------------
    st.subheader("📊 Cost and Load Comparison")
    chart_col1, chart_col2 = st.columns(2)

    cost_chart = (
        alt.Chart(cost_df)
        .mark_bar(size=40)
        .encode(
            x="Scenario:N",
            y="Total Cost ($):Q",
            color="Scenario:N",
            tooltip=["Scenario", "Total Cost ($):Q", "Total Loads"]
        )
        .properties(width=280, height=300, title="Total Cost by Scenario")
    )
    chart_col1.altair_chart(cost_chart, use_container_width=False)

    load_chart = (
        alt.Chart(cost_df)
        .mark_bar(size=40)
        .encode(
            x="Scenario:N",
            y="Total Loads:Q",
            color="Scenario:N",
            tooltip=["Scenario", "Total Loads"]
        )
        .properties(width=280, height=300, title="Total Loads by Scenario")
    )
    chart_col2.altair_chart(load_chart, use_container_width=False)

    # ----------------- Maps -----------------
    st.subheader("🗺 Load Maps")
    tab1, tab2, tab3 = st.tabs(["1) Before Optimization", "2) Mode-Consolidated", "3) Cross-Mode"])
    color_map = {"Parcel": [0, 102, 255], "LTL": [255, 128, 0]}
    with tab1:
        render_map(before, color_map)
    with tab2:
        render_map(mode_consol, color_map)
    with tab3:
        render_map(cross_mode, {"LTL": [255, 0, 0]})

    # ----------------- Tables -----------------
    st.subheader("📋 Load Details")
    st.write("**1) Before Optimization:**")
    st.dataframe(before, use_container_width=True)
    st.write("**2) Mode-Consolidated:**")
    st.dataframe(mode_consol, use_container_width=True)
    st.write("**3) Cross-Mode Consolidated:**")
    st.dataframe(cross_mode, use_container_width=True)

# ----------------- Explanation -----------------
st.markdown("## 📖 Business Context & Tech Stack")
st.markdown("""
- **Goal:** Reduce cost and loads by consolidating shipments (orders) into fewer loads.
- **Scenario 1:** No optimization – each order is a separate load.
- **Scenario 2:** LTL shipments consolidated by lane.
- **Scenario 3:** Parcels moved to LTL (if cheaper), then consolidated.

**Tech:**
- **Streamlit** for UI,
- **Pandas/Numpy** for data and cost calculations,
- **Altair** for charts,
- **PyDeck** (free map library) for visualizing routes and loads.
""")
