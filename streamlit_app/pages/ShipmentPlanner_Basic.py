import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Shipment Planner – Parcel vs LTL", layout="wide")
st.title("🚚 Shipment Planner – Parcel vs LTL with Consolidation")

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

# Generate dataset
np.random.seed(42)
origins = np.random.choice(list(city_coords.keys()), 20)
destinations = np.random.choice(list(city_coords.keys()), 20)
weights = np.concatenate([np.random.randint(20, 150, 10), np.random.randint(200, 800, 10)])
distances = np.random.randint(100, 2500, 20)

shipments = pd.DataFrame({
    "ShipmentID": [f"SHP-{i:03d}" for i in range(1, 21)],
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

st.subheader("📦 Shipment Data (20 rows)")
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
    if weight > 150:
        return None
    return (8 + 0.06 * weight) * parcel_zone_multiplier(distance)

def rate_ltl(weight, distance, nmfc_class):
    return 35 + (0.42 * distance * ltl_class_factor(nmfc_class))

# ----------------- Solve -----------------
if st.button("Solve Optimization"):
    results = []
    for _, r in shipments.iterrows():
        parcel_cost = rate_parcel(r["Weight (lbs)"], r["Distance (miles)"])
        ltl_cost = rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"])
        chosen_mode, chosen_cost = None, None

        if parcel_cost is not None and parcel_cost < ltl_cost:
            chosen_mode = "Parcel"
            chosen_cost = parcel_cost
        else:
            chosen_mode = "LTL"
            chosen_cost = ltl_cost

        results.append({
            "ShipmentID": r["ShipmentID"],
            "Origin": r["Origin"],
            "Destination": r["Destination"],
            "Parcel Cost ($)": round(parcel_cost, 2) if parcel_cost else None,
            "LTL Cost ($)": round(ltl_cost, 2),
            "Chosen Mode": chosen_mode,
            "Chosen Cost ($)": round(chosen_cost, 2)
        })
    results_df = pd.DataFrame(results)
    st.subheader("🔍 Optimized Shipment Modes")
    st.dataframe(results_df, use_container_width=True)

    # ----------------- Consolidation -----------------
    consolidated = (
        shipments.groupby(["Origin", "Destination"], as_index=False)
        .agg({"Weight (lbs)": "sum", "Distance (miles)": "mean"})
    )
    consolidated["NMFC Class"] = consolidated["Weight (lbs)"].apply(get_nmfc_class)
    consolidated["LTL Cost ($)"] = consolidated.apply(
        lambda r: rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"]), axis=1
    )

    st.subheader("📦 LTL Consolidation Results")
    st.dataframe(consolidated, use_container_width=True)

    # ----------------- KPIs -----------------
    total_initial = shipments.apply(
        lambda r: rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], get_nmfc_class(r["Weight (lbs)"])),
        axis=1
    ).sum()
    total_optimized = results_df["Chosen Cost ($)"].sum()
    total_consolidated = consolidated["LTL Cost ($)"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Cost (All LTL)", f"${total_initial:,.2f}")
    col2.metric("Optimized Mix", f"${total_optimized:,.2f}", f"Saved {total_initial - total_optimized:,.2f}")
    col3.metric("After Consolidation", f"${total_consolidated:,.2f}", f"Saved {total_optimized - total_consolidated:,.2f}")

    # ----------------- Charts -----------------
    st.subheader("📊 Cost Comparison")
    cost_data = pd.DataFrame({
        "Scenario": ["Initial (All LTL)", "Optimized Mix", "After Consolidation"],
        "Total Cost ($)": [total_initial, total_optimized, total_consolidated]
    })
    cost_chart = alt.Chart(cost_data).mark_bar().encode(
        x="Scenario",
        y="Total Cost ($)",
        color="Scenario",
        tooltip=["Scenario", "Total Cost ($)"]
    ).properties(width=500, height=300)
    st.altair_chart(cost_chart, use_container_width=True)

    # ----------------- Maps -----------------
    def get_routes_df(df, color):
        data = []
        for _, r in df.iterrows():
            data.append({
                "from_lat": city_coords[r["Origin"]][0],
                "from_lon": city_coords[r["Origin"]][1],
                "to_lat": city_coords[r["Destination"]][0],
                "to_lon": city_coords[r["Destination"]][1],
                "color": color
            })
        return pd.DataFrame(data)

    st.subheader("🗺 Maps – Before and After")
    tab1, tab2, tab3 = st.tabs(["As Is (All LTL)", "Optimized Mix", "After Consolidation"])

    with tab1:
        routes1 = get_routes_df(shipments, [150, 150, 150])
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer(
                    "ArcLayer",
                    data=routes1,
                    get_source_position=["from_lon", "from_lat"],
                    get_target_position=["to_lon", "to_lat"],
                    get_width=2,
                    get_tilt=15,
                    get_source_color="color",
                    get_target_color="color",
                    pickable=True
                )
            ],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3, pitch=50)
        ))

    with tab2:
        routes2_parcel = get_routes_df(results_df[results_df["Chosen Mode"] == "Parcel"], [0, 0, 255])
        routes2_ltl = get_routes_df(results_df[results_df["Chosen Mode"] == "LTL"], [255, 128, 0])
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("ArcLayer", data=routes2_parcel, get_source_position=["from_lon", "from_lat"],
                          get_target_position=["to_lon", "to_lat"], get_width=2, get_source_color="color", get_target_color="color"),
                pdk.Layer("ArcLayer", data=routes2_ltl, get_source_position=["from_lon", "from_lat"],
                          get_target_position=["to_lon", "to_lat"], get_width=2, get_source_color="color", get_target_color="color")
            ],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3, pitch=50)
        ))

    with tab3:
        routes3 = get_routes_df(consolidated, [255, 0, 0])
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("ArcLayer", data=routes3, get_source_position=["from_lon", "from_lat"],
                          get_target_position=["to_lon", "to_lat"], get_width=2, get_source_color="color", get_target_color="color")
            ],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3, pitch=50)
        ))

# ----------------- Business Context & Key Insights -----------------
st.markdown("## 📖 Business Context & Key Insights")
st.markdown("""
- **Parcel vs LTL Trade-offs:**  
  Parcel shipments are cost-effective for lighter loads (≤150 lbs), while LTL carriers use **freight classes** and weight breaks for pricing heavier loads. This demo reflects that logic — some shipments automatically go Parcel, others go LTL.

- **Optimization Strategy:**  
  First, we pick the **cheapest mode per shipment** (Parcel vs LTL). Then we **consolidate shipments** on the same Origin-Destination lane, which reduces cost by leveraging economies of scale.

- **Savings Visualization:**  
  The three maps show **cost evolution**:  
  1. **All LTL (baseline)** – every shipment moves individually by LTL.  
  2. **Optimized Mix** – Parcel and LTL are chosen dynamically to minimize cost.  
  3. **After Consolidation** – Shipments with the same lanes are grouped, creating bigger loads and lowering per-mile costs.

- **Key Takeaway:**  
  Even this simple model shows **how proper mode selection and lane consolidation** can reduce total transportation spend. In real-world TMS systems, these optimizations are enhanced by **carrier contracts, dimensional weight rules, service levels, and live API rates.**
""")
