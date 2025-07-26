You’re right—the last version could *increase* costs and the maps didn’t clearly shrink. Here’s a **fixed, fully working, randomized** version that:

**Guarantees optimization potential & monotonic savings**

1. **Before Optimization** (1 order = 1 load; Parcel ≤150 lbs else LTL).
2. **Mode‑Consolidated**

   * **Only LTL** on the same lane are consolidated.
   * **Never worse** than before: for every lane we take **min(original LTL cost, consolidated LTL cost)**.
3. **Cross‑Mode Consolidated**

   * For every lane, compare **(Scenario‑2 cost on that lane) vs (all orders on that lane as ONE LTL load)**.
   * Pick the cheaper. **Can’t be worse** than Scenario‑2.

**Random, but with built‑in optimization potential**

* We **force** 3 lanes, with a **mixed lane** (both Parcel + LTL), a **heavy-only lane**, and a **parcel-only lane**.
* Distances & weights are randomized every run, but **the structure guarantees savings opportunities**.

**Maps that actually shrink**

* We draw **loads (not orders)**.
* **Scatter + Text labels** for cities.
* Arcs **visibly reduce** across scenarios.

---

### ✅ Replace your `ShipmentPlanner_Basic.py` with this

```python
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import random

st.set_page_config(page_title="Shipment Planner – Load Optimization", layout="wide")
st.title("🚛 Shipment Planner — Load Optimization (Randomized, but always optimizable)")

# ------------------------------------------------------------
# Cities
# ------------------------------------------------------------
CITY_COORDS = {
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
CITIES = list(CITY_COORDS.keys())

# ------------------------------------------------------------
# Random data generator with guaranteed optimization potential
#  - 10 orders
#  - 3 lanes:
#    * Lane A: mix of parcel + LTL (4 orders)  -> enables cross-mode savings
#    * Lane B: heavy-only LTL (3 orders)       -> enables LTL consolidation savings
#    * Lane C: parcel-only (3 orders)          -> won't consolidate but good baseline
# ------------------------------------------------------------
def nmfc_class(weight):
    if weight <= 150: return "70"
    if weight <= 500: return "55"
    return "50"

def gen_lane(orders, min_w, max_w, parcel_ratio=0.0):
    """Generate weights for a lane with a certain ratio of parcel-eligible shipments."""
    n_parcel = int(round(orders * parcel_ratio))
    n_ltl = orders - n_parcel
    ws = list(np.random.randint(30, 150, n_parcel)) + list(np.random.randint(min_w, max_w, n_ltl))
    random.shuffle(ws)
    return ws

def generate_shipments():
    # choose 3 unique lanes
    lanes = []
    while len(lanes) < 3:
        o, d = random.sample(CITIES, 2)
        if (o, d) not in lanes:
            lanes.append((o, d))

    # Lane A: 4 orders, mixed parcel + LTL
    laneA = lanes[0]
    wA = gen_lane(4, 250, 800, parcel_ratio=0.5)  # ~half parcel-eligible
    distA = np.random.randint(400, 1800)

    # Lane B: 3 orders, heavy LTL only
    laneB = lanes[1]
    wB = gen_lane(3, 300, 900, parcel_ratio=0.0)
    distB = np.random.randint(600, 2000)

    # Lane C: 3 orders, parcel only
    laneC = lanes[2]
    wC = list(np.random.randint(40, 140, 3))
    distC = np.random.randint(600, 2000)

    rows = []
    i = 1

    for w in wA:
        rows.append({"ShipmentID": f"ORD-{i:03d}", "Origin": laneA[0], "Destination": laneA[1],
                     "Distance (miles)": distA, "Weight (lbs)": int(w)})
        i += 1
    for w in wB:
        rows.append({"ShipmentID": f"ORD-{i:03d}", "Origin": laneB[0], "Destination": laneB[1],
                     "Distance (miles)": distB, "Weight (lbs)": int(w)})
        i += 1
    for w in wC:
        rows.append({"ShipmentID": f"ORD-{i:03d}", "Origin": laneC[0], "Destination": laneC[1],
                     "Distance (miles)": distC, "Weight (lbs)": int(w)})
        i += 1

    df = pd.DataFrame(rows)
    df["NMFC Class"] = df["Weight (lbs)"].apply(nmfc_class)
    return df

# ------------------------------------------------------------
# Rating
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
# Scenario 1: Before optimization (1 order = 1 load)
# ------------------------------------------------------------
def scenario_1_before(df):
    loads = []
    for _, r in df.iterrows():
        if r["Weight (lbs)"] <= 150:
            mode = "Parcel"
            cost = rate_parcel(r["Weight (lbs)"], r["Distance (miles)"])
        else:
            mode = "LTL"
            cost = rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"])
        loads.append({
            "LoadID": r["ShipmentID"],
            "Origin": r["Origin"],
            "Destination": r["Destination"],
            "Total Weight (lbs)": r["Weight (lbs)"],
            "Mode": mode,
            "Cost ($)": round(cost, 2)
        })
    return pd.DataFrame(loads)

# ------------------------------------------------------------
# Scenario 2: Mode-preserving consolidation (only LTL)
#   For each lane: choose min(original LTL loads cost, consolidated lane LTL cost)
#   -> guarantees not worse than Scenario 1.
# ------------------------------------------------------------
def scenario_2_mode_consolidated(scen1_df, ship_df):
    parcel = scen1_df[scen1_df["Mode"] == "Parcel"].copy()
    ltl = scen1_df[scen1_df["Mode"] == "LTL"].copy()

    result_rows = parcel.to_dict(orient="records")

    # lane-by-lane for LTL
    lanes = ltl.groupby(["Origin", "Destination"], as_index=False).size()[["Origin", "Destination"]]
    for _, lane in lanes.iterrows():
        o, d = lane["Origin"], lane["Destination"]
        lane_ltl = ltl[(ltl["Origin"] == o) & (ltl["Destination"] == d)]
        original_cost = lane_ltl["Cost ($)"].sum()
        total_weight = lane_ltl["Total Weight (lbs)"].sum()
        distance = ship_df[(ship_df["Origin"] == o) & (ship_df["Destination"] == d)]["Distance (miles)"].mean()
        cons_cost = rate_ltl(total_weight, distance, nmfc_class(total_weight))

        if cons_cost < original_cost:
            # use consolidated
            result_rows.append({
                "LoadID": f"LTL-{o[:3]}-{d[:3]}",
                "Origin": o,
                "Destination": d,
                "Total Weight (lbs)": total_weight,
                "Mode": "LTL",
                "Cost ($)": round(cons_cost, 2)
            })
        else:
            # keep originals
            result_rows.extend(lane_ltl.to_dict(orient="records"))

    return pd.DataFrame(result_rows)

# ------------------------------------------------------------
# Scenario 3: Cross-mode consolidation
#   lane-by-lane: compare Scenario 2's lane cost vs 1 big LTL for *all* orders (parcel+LTL)
#   -> pick the cheaper. Guarantees not worse than Scenario 2.
# ------------------------------------------------------------
def scenario_3_cross_mode(ship_df, scen2_df):
    final_rows = []
    lanes = ship_df.groupby(["Origin", "Destination"], as_index=False).size()[["Origin", "Destination"]]

    for _, lane in lanes.iterrows():
        o, d = lane["Origin"], lane["Destination"]
        # scenario 2 cost on this lane (could include parcel + LTL entries)
        scen2_lane = scen2_df[(scen2_df["Origin"] == o) & (scen2_df["Destination"] == d)]
        scen2_lane_cost = scen2_lane["Cost ($)"].sum()

        # if we convert ALL ORDERS in that lane to one LTL:
        lane_orders = ship_df[(ship_df["Origin"] == o) & (ship_df["Destination"] == d)]
        total_weight = lane_orders["Weight (lbs)"].sum()
        distance = lane_orders["Distance (miles)"].mean()
        big_ltl_cost = rate_ltl(total_weight, distance, nmfc_class(total_weight))

        if big_ltl_cost < scen2_lane_cost:
            final_rows.append({
                "LoadID": f"XMODE-{o[:3]}-{d[:3]}",
                "Origin": o,
                "Destination": d,
                "Total Weight (lbs)": total_weight,
                "Mode": "LTL",
                "Cost ($)": round(big_ltl_cost, 2)
            })
        else:
            final_rows.extend(scen2_lane.to_dict(orient="records"))

    return pd.DataFrame(final_rows)

# ------------------------------------------------------------
# Map helpers
# ------------------------------------------------------------
def cities_df_from_loads(loads):
    cs = set(loads["Origin"].tolist() + loads["Destination"].tolist())
    return pd.DataFrame([{"city": c, "lat": CITY_COORDS[c][0], "lon": CITY_COORDS[c][1]} for c in cs])

def to_routes_df(loads_df, mode_color):
    rows = []
    for _, r in loads_df.iterrows():
        color = mode_color.get(r["Mode"], [200, 200, 200])
        rows.append({
            "from_lat": CITY_COORDS[r["Origin"]][0],
            "from_lon": CITY_COORDS[r["Origin"]][1],
            "to_lat": CITY_COORDS[r["Destination"]][0],
            "to_lon": CITY_COORDS[r["Destination"]][1],
            "color": color,
        })
    return pd.DataFrame(rows)

def render_map(loads_df, mode_color):
    routes = to_routes_df(loads_df, mode_color)
    labels = cities_df_from_loads(loads_df)

    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer(
                "ArcLayer",
                data=routes,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_width=3,
                get_source_color="color",
                get_target_color="color",
                get_height=1.5,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=labels,
                get_position='[lon, lat]',
                get_radius=60000,
                get_fill_color=[0, 0, 0, 80],
            ),
            pdk.Layer(
                "TextLayer",
                data=labels,
                get_position='[lon, lat]',
                get_text='city',
                get_size=14,
                get_color=[0, 0, 0],
            )
        ],
        initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=4, pitch=20),
        height=360
    ))

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if st.button("Run Optimization"):
    # Random shipments with guaranteed structure
    shipments = generate_shipments()
    st.subheader("📦 Orders (Randomized)")
    st.dataframe(shipments, use_container_width=True)

    # Scenario 1
    s1 = scenario_1_before(shipments)
    cost1, loads1 = s1["Cost ($)"].sum(), len(s1)

    # Scenario 2
    s2 = scenario_2_mode_consolidated(s1, shipments)
    cost2, loads2 = s2["Cost ($)"].sum(), len(s2)

    # Scenario 3
    s3 = scenario_3_cross_mode(shipments, s2)
    cost3, loads3 = s3["Cost ($)"].sum(), len(s3)

    # KPIs (guaranteed non-increasing)
    st.subheader("📈 Scenario KPIs (cost & loads should decrease stepwise)")
    c1, c2, c3 = st.columns(3)
    c1.metric("1) Before", f"${cost1:,.2f}", f"Loads: {loads1}")
    c2.metric("2) Mode-Consolidated", f"${cost2:,.2f}", f"Loads: {loads2} | Δ ${(cost1 - cost2):,.2f}")
    c3.metric("3) Cross-Mode", f"${cost3:,.2f}", f"Loads: {loads3} | Δ ${(cost2 - cost3):,.2f}")

    # Charts
    st.subheader("📊 Cost & Load Comparison")
    cost_df = pd.DataFrame({
        "Scenario": ["Before", "Mode-Consolidated", "Cross-Mode"],
        "Total Cost ($)": [cost1, cost2, cost3],
        "Total Loads": [loads1, loads2, loads3]
    })
    col1, col2 = st.columns(2)
    col1.altair_chart(
        alt.Chart(cost_df).mark_bar(size=40).encode(
            x=alt.X("Scenario:N", sort=["Before", "Mode-Consolidated", "Cross-Mode"]),
            y=alt.Y("Total Cost ($):Q", axis=alt.Axis(format="$.2f")),
            color="Scenario:N"
        ).properties(width=280, height=280, title="Total Cost by Scenario"),
        use_container_width=False
    )
    col2.altair_chart(
        alt.Chart(cost_df).mark_bar(size=40).encode(
            x=alt.X("Scenario:N", sort=["Before", "Mode-Consolidated", "Cross-Mode"]),
            y=alt.Y("Total Loads:Q"),
            color="Scenario:N"
        ).properties(width=280, height=280, title="Total Loads by Scenario"),
        use_container_width=False
    )

    # Maps
    st.subheader("🗺 Load Maps — arcs reduce from 1 → 2 → 3")
    tab1, tab2, tab3 = st.tabs(["Before", "Mode-Consolidated", "Cross-Mode"])
    with tab1:
        render_map(s1, {"Parcel": [0, 102, 255], "LTL": [255, 128, 0]})
        st.caption("Blue = Parcel, Orange = LTL.")
    with tab2:
        render_map(s2, {"Parcel": [0, 102, 255], "LTL": [255, 128, 0]})
        st.caption("Parcel unchanged; LTL lanes consolidated. Fewer orange arcs.")
    with tab3:
        render_map(s3, {"Parcel": [0, 102, 255], "LTL": [255, 0, 0]})
        st.caption("Cross-mode: where cheaper, Parcel moved into lane LTL (red). Fewest arcs.")

    # Tables
    st.subheader("📋 Load Details")
    st.write("**1) Before Optimization**")
    st.dataframe(s1, use_container_width=True)
    st.write("**2) Mode-Consolidated**")
    st.dataframe(s2, use_container_width=True)
    st.write("**3) Cross-Mode Consolidated**")
    st.dataframe(s3, use_container_width=True)

# ------------------------------------------------------------
# Explanation
# ------------------------------------------------------------
st.markdown("## 📖 What you’re seeing")
st.markdown("""
**Why costs/load count now decrease stepwise:**  
- In Scenario 2, for every lane we **choose the cheaper** of "original LTL loads" vs "one consolidated LTL load".  
- In Scenario 3, for every lane we **choose the cheaper** of Scenario 2 vs "one big LTL load including Parcel".  
So cost can **never go up**.

**Random but repeatably optimizable:**  
- We intentionally create one lane that mixes Parcel + LTL (good for cross-mode savings),  
  one lane with several LTLs (good for consolidation),  
  and a parcel-only lane (baseline behavior).
""")
```

---

**If you still see a non-decreasing cost**, paste the KPI numbers back and I’ll tweak the lane logic again.
Ready to jump to **VRP Pilot** once this is solid.
