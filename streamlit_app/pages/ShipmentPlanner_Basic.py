import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Shipment Planner — Parcel vs LTL", layout="wide")
st.title("🚚 Shipment Planner — Parcel vs LTL (Consolidation & Reassignment)")

# ----------------- Navigation (optional) -----------------
st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

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

# ----------------- Generate Dataset -----------------
np.random.seed(42)
origins = np.random.choice(list(city_coords.keys()), 20)
destinations = np.random.choice(list(city_coords.keys()), 20)

# Ensure 10 <=150 lbs (Parcel-eligible) and 10 >150 lbs (LTL)
weights = np.concatenate([np.random.randint(20, 150, 10), np.random.randint(200, 800, 10)])
np.random.shuffle(weights)

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

# ----------------- Scenario Solvers -----------------
def solve_as_is(df):
    """Scenario 1: Policy — <=150 lbs -> Parcel, else LTL. No consolidation."""
    rows = []
    for _, r in df.iterrows():
        if r["Weight (lbs)"] <= 150:
            mode = "Parcel"
            cost = rate_parcel(r["Weight (lbs)"], r["Distance (miles)"])
        else:
            mode = "LTL"
            cost = rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"])
        rows.append({
            **r.to_dict(),
            "Chosen Mode": mode,
            "Chosen Cost ($)": round(cost, 2)
        })
    return pd.DataFrame(rows)

def consolidate_ltl_lane_costs(df):
    lane = (
        df.groupby(["Origin", "Destination"], as_index=False)
          .agg({"Weight (lbs)": "sum", "Distance (miles)": "mean"})
    )
    lane["NMFC Class"] = lane["Weight (lbs)"].apply(get_nmfc_class)
    lane["Lane LTL Cost ($)"] = lane.apply(
        lambda r: rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"]),
        axis=1
    )
    return lane, lane["Lane LTL Cost ($)"].sum()

def solve_consolidate_within_mode(as_is_df):
    ltl_shipments = as_is_df[as_is_df["Chosen Mode"] == "LTL"].copy()
    parcel_shipments = as_is_df[as_is_df["Chosen Mode"] == "Parcel"].copy()
    ltl_lane_table, ltl_total_consol = consolidate_ltl_lane_costs(ltl_shipments)
    total_cost = parcel_shipments["Chosen Cost ($)"].sum() + ltl_total_consol
    return ltl_lane_table, total_cost, parcel_shipments

def solve_parcel_to_ltl_consolidated(df):
    lane, total = consolidate_ltl_lane_costs(df)
    return lane, total

# ----------------- Map Helpers -----------------
def get_routes_df(df, mode_col=None):
    rows = []
    for _, r in df.iterrows():
        color = [200, 200, 200]
        if mode_col:
            color = [0, 102, 255] if r[mode_col] == "Parcel" else [255, 128, 0]
        rows.append({
            "from_lat": city_coords[r["Origin"]][0],
            "from_lon": city_coords[r["Origin"]][1],
            "to_lat": city_coords[r["Destination"]][0],
            "to_lon": city_coords[r["Destination"]][1],
            "color": color,
        })
    return pd.DataFrame(rows)

def get_routes_df_from_lanes(lane_df, color):
    rows = []
    for _, r in lane_df.iterrows():
        rows.append({
            "from_lat": city_coords[r["Origin"]][0],
            "from_lon": city_coords[r["Origin"]][1],
            "to_lat": city_coords[r["Destination"]][0],
            "to_lon": city_coords[r["Destination"]][1],
            "color": color,
        })
    return pd.DataFrame(rows)

def get_city_labels_df(df):
    cities = set(df["Origin"].tolist() + df["Destination"].tolist())
    return pd.DataFrame([{"lat": city_coords[c][0], "lon": city_coords[c][1], "city": c} for c in cities])

def render_map(routes_df, labels_df, height=350):
    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer(
                "ArcLayer",
                data=routes_df,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_width=2,
                get_source_color="color",
                get_target_color="color",
                get_height=1.5,
            ),
            pdk.Layer(
                "TextLayer",
                data=labels_df,
                get_position='[lon, lat]',
                get_text='city',
                get_size=14,
                get_color=[0, 0, 0],
            )
        ],
        initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3.5, pitch=20),
        height=height
    ))

# ----------------- RUN -----------------
if st.button("Run Scenarios"):
    # Scenario 1
    as_is_df = solve_as_is(shipments)
    total_as_is = as_is_df["Chosen Cost ($)"].sum()

    # Scenario 2
    ltl_lane_table_2, total_consol_mode, parcel_df_2 = solve_consolidate_within_mode(as_is_df)

    # Scenario 3
    lane_all_3, total_consol_all = solve_parcel_to_ltl_consolidated(shipments)

    # KPIs
    st.subheader("💰 Scenario KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("1) As‑Is (Policy)", f"${total_as_is:,.2f}")
    c2.metric("2) Mode-Consolidated", f"${total_consol_mode:,.2f}", f"Saved {total_as_is - total_consol_mode:,.2f}")
    c3.metric("3) Parcel→LTL Consolidated", f"${total_consol_all:,.2f}", f"Saved {total_consol_mode - total_consol_all:,.2f}")

    # Chart
    st.subheader("📊 Total Cost Comparison")
    cost_data = pd.DataFrame({
        "Scenario": ["1) As‑Is", "2) Mode-Consolidated", "3) Parcel→LTL Consolidated"],
        "Total Cost ($)": [total_as_is, total_consol_mode, total_consol_all]
    })
    cost_chart = (
        alt.Chart(cost_data)
        .mark_bar(size=40)
        .encode(
            x=alt.X("Scenario:N"),
            y=alt.Y("Total Cost ($):Q", axis=alt.Axis(format="$.2f")),
            color="Scenario:N",
            tooltip=["Scenario", alt.Tooltip("Total Cost ($):Q", format="$.2f")]
        )
        .properties(width=600, height=280)
    )
    st.altair_chart(cost_chart, use_container_width=False)

    # Maps
    st.subheader("🗺 Maps — Arcs reduce from 1 ➜ 2 ➜ 3")
    st.markdown("""
**Legend**  
🔵 Parcel (≤150 lbs)  
🟠 LTL  
🔴 Consolidated LTL (Scenario 3)
""")

    tab1, tab2, tab3 = st.tabs(["1) As‑Is", "2) Mode-Consolidated", "3) Parcel→LTL Consolidated"])

    with tab1:
        routes1 = get_routes_df(as_is_df, mode_col="Chosen Mode")
        labels1 = get_city_labels_df(as_is_df)
        render_map(routes1, labels1)

    with tab2:
        routes2_parcel = get_routes_df(parcel_df_2.assign(Chosen_Mode="Parcel"), mode_col="Chosen_Mode")
        routes2_ltl = get_routes_df_from_lanes(ltl_lane_table_2, [255, 128, 0])
        render_map(pd.concat([routes2_parcel, routes2_ltl], ignore_index=True),
                   get_city_labels_df(pd.concat([
                       parcel_df_2[["Origin", "Destination"]],
                       ltl_lane_table_2[["Origin", "Destination"]]
                   ])))

    with tab3:
        routes3 = get_routes_df_from_lanes(lane_all_3, [255, 0, 0])
        render_map(routes3, get_city_labels_df(lane_all_3))

# ----------------- Business Context -----------------
st.markdown("## 📖 Business Context & Key Insights")
st.markdown("""
**Scenarios:**
1. **As‑Is:** Parcel (≤150 lbs) vs LTL, no consolidation.
2. **Mode-Consolidated:** Only LTL shipments consolidated by lane, Parcel stays as-is.
3. **Parcel→LTL Consolidated:** All shipments that benefit move to LTL and consolidate.

**Takeaways:**  
- Costs reduce 1 ➜ 2 ➜ 3 by combining **rate-shopping** and **lane consolidation**.
- Shows how **TMS logic** works in a simplified, transparent way.
""")

# ----------------- Tech Explanation -----------------
st.markdown("## ⚙️ Tech & Python Explanation")
st.markdown("""
- **Python Tools:** Streamlit (UI), Pandas/Numpy (data), Altair (charts), PyDeck (free maps).
- **Why PyDeck?** Interactive ArcLayer and TextLayer for cities — no paid API keys required.
- **Design:** Modular functions (`rate_parcel()`, `rate_ltl()`, scenario solvers) ensure clarity.
- **Next:** Add time windows, dimensional weight, and real carrier rates (CSV/API).
""")
