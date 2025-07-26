import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Shipment Planner – Parcel vs LTL", layout="wide")
st.title("🚚 Shipment Planner — Parcel vs LTL with Consolidation")

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

# ----------------- Generate Dataset (20 rows) -----------------
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
        return "70"   # higher class, more expensive
    elif weight <= 500:
        return "55"
    return "50"       # heavier -> lower class factor

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
    # Parcel only for <= 150 lbs
    if weight > 150:
        return None
    return (8 + 0.06 * weight) * parcel_zone_multiplier(distance)

def rate_ltl(weight, distance, nmfc_class):
    return 35 + (0.42 * distance * ltl_class_factor(nmfc_class))

# ----------------- Scenario Solvers -----------------
def solve_as_is(df):
    """Policy: <=150 lbs -> Parcel, else LTL."""
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

def solve_optimized(df):
    """Rate-shop: choose cheaper of Parcel vs LTL (Parcel only allowed <=150 lbs)."""
    rows = []
    for _, r in df.iterrows():
        parcel_cost = rate_parcel(r["Weight (lbs)"], r["Distance (miles)"])
        ltl_cost = rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"])

        if parcel_cost is not None and parcel_cost < ltl_cost:
            mode = "Parcel"
            cost = parcel_cost
        else:
            mode = "LTL"
            cost = ltl_cost

        rows.append({
            **r.to_dict(),
            "Parcel Cost ($)": round(parcel_cost, 2) if parcel_cost is not None else None,
            "LTL Cost ($)": round(ltl_cost, 2),
            "Chosen Mode": mode,
            "Chosen Cost ($)": round(cost, 2)
        })
    return pd.DataFrame(rows)

def solve_consolidated(df):
    """
    Force ALL shipments into LTL, consolidate by O/D lane.
    One LTL load per lane (sum weight, mean distance for toy example).
    """
    lane = (
        df.groupby(["Origin", "Destination"], as_index=False)
          .agg({"Weight (lbs)": "sum", "Distance (miles)": "mean"})
    )
    lane["NMFC Class"] = lane["Weight (lbs)"].apply(get_nmfc_class)
    lane["Consolidated LTL Cost ($)"] = lane.apply(
        lambda r: rate_ltl(r["Weight (lbs)"], r["Distance (miles)"], r["NMFC Class"]),
        axis=1
    )
    return lane

# ----------------- Run -----------------
if st.button("Solve All Scenarios"):
    # As-Is (policy)
    as_is_df = solve_as_is(shipments)
    total_as_is = as_is_df["Chosen Cost ($)"].sum()

    # Optimized (rate shop)
    opt_df = solve_optimized(shipments)
    total_opt = opt_df["Chosen Cost ($)"].sum()

    # Consolidated (all LTL grouped by lane)
    cons_df = solve_consolidated(shipments)
    total_cons = cons_df["Consolidated LTL Cost ($)"].sum()

    # ----------------- KPIs -----------------
    st.subheader("💰 Scenario KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("As‑Is (Policy) Cost", f"${total_as_is:,.2f}")
    c2.metric("Optimized Mix Cost", f"${total_opt:,.2f}", f"Saved {total_as_is - total_opt:,.2f}")
    c3.metric("Consolidated LTL Cost", f"${total_cons:,.2f}", f"Saved {total_opt - total_cons:,.2f}")

    # ----------------- Charts -----------------
    st.subheader("📊 Total Cost Comparison")
    cost_data = pd.DataFrame({
        "Scenario": ["As‑Is Policy", "Optimized Mix", "Consolidated LTL"],
        "Total Cost ($)": [total_as_is, total_opt, total_cons]
    })
    cost_chart = (
        alt.Chart(cost_data)
        .mark_bar(size=40)  # control bar thickness
        .encode(
            x=alt.X("Scenario:N", sort=["As‑Is Policy", "Optimized Mix", "Consolidated LTL"]),
            y=alt.Y("Total Cost ($):Q", axis=alt.Axis(format="$.2f")),
            color=alt.Color("Scenario:N", legend=None),
            tooltip=["Scenario", alt.Tooltip("Total Cost ($):Q", format="$.2f")]
        )
        .properties(width=550, height=280)
    )
    st.altair_chart(cost_chart, use_container_width=False)

    # Mode mix (show Parcel shrinking from As-Is to Optimized)
    st.subheader("📊 Mode Mix (Counts)")
    mode_mix = (
        pd.DataFrame({
            "Scenario": ["As‑Is Policy"] * len(as_is_df) + ["Optimized Mix"] * len(opt_df),
            "Mode": list(as_is_df["Chosen Mode"]) + list(opt_df["Chosen Mode"])
        })
        .groupby(["Scenario", "Mode"], as_index=False)
        .size()
    )

    mix_chart = (
        alt.Chart(mode_mix)
        .mark_bar(size=35)
        .encode(
            x=alt.X("Scenario:N", sort=["As‑Is Policy", "Optimized Mix"]),
            y=alt.Y("size:Q", title="Shipments"),
            color=alt.Color("Mode:N", scale=alt.Scale(range=["#007bff", "#ff7f0e"])),
            column=alt.Column("Mode:N", header=alt.Header(title="Mode")),
            tooltip=["Scenario", "Mode", "size"]
        )
        .properties(width=250, height=250)
        .resolve_scale(y='independent')
    )
    st.altair_chart(mix_chart, use_container_width=True)

    # ----------------- Maps -----------------
    st.subheader("🗺 Maps – Before, Optimized, Consolidated")

    legend = """
**Legend:**
- 🔵 **Parcel**
- 🟠 **LTL**
- 🔴 **Consolidated LTL (lane-level)**
- ⚪ **All LTL (As‑Is view only for contrast)**
"""
    st.markdown(legend)

    def get_routes_df(df, mode_col=None):
        """
        Return arcs with color by mode (for As-Is & Optimized).
        If mode_col is None, use gray. Otherwise:
          Parcel -> blue, LTL -> orange.
        """
        rows = []
        for _, r in df.iterrows():
            origin, dest = r["Origin"], r["Destination"]
            color = [200, 200, 200]  # default gray
            if mode_col:
                m = r[mode_col]
                if m == "Parcel":
                    color = [0, 102, 255]     # blue
                else:
                    color = [255, 128, 0]     # orange
            rows.append({
                "from_lat": city_coords[origin][0],
                "from_lon": city_coords[origin][1],
                "to_lat": city_coords[dest][0],
                "to_lon": city_coords[dest][1],
                "color": color,
                "origin": origin,
                "destination": dest
            })
        return pd.DataFrame(rows)

    def get_lane_routes_df(df):
        """
        Consolidated arcs (one per lane), all in red.
        """
        rows = []
        for _, r in df.iterrows():
            origin, dest = r["Origin"], r["Destination"]
            rows.append({
                "from_lat": city_coords[origin][0],
                "from_lon": city_coords[origin][1],
                "to_lat": city_coords[dest][0],
                "to_lon": city_coords[dest][1],
                "color": [255, 0, 0],
                "origin": origin,
                "destination": dest
            })
        return pd.DataFrame(rows)

    def get_city_labels_df(df):
        city_list = set(df["Origin"].tolist() + df["Destination"].tolist())
        return pd.DataFrame([
            {"lat": city_coords[c][0], "lon": city_coords[c][1], "city": c} for c in city_list
        ])

    tab1, tab2, tab3 = st.tabs(["As‑Is (Policy)", "Optimized Mix", "Consolidated LTL"])

    with tab1:
        routes_as_is = get_routes_df(as_is_df, mode_col="Chosen Mode")
        city_labels = get_city_labels_df(as_is_df)
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer(
                    "ArcLayer",
                    data=routes_as_is,
                    get_source_position=["from_lon", "from_lat"],
                    get_target_position=["to_lon", "to_lat"],
                    get_width=2,
                    get_source_color="color",
                    get_target_color="color",
                    get_height=1
                ),
                pdk.Layer(
                    "TextLayer",
                    data=city_labels,
                    get_position='[lon, lat]',
                    get_text='city',
                    get_size=14,
                    get_color=[0, 0, 0],
                )
            ],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3, pitch=30),
            height=400
        ))

    with tab2:
        routes_opt = get_routes_df(opt_df, mode_col="Chosen Mode")
        city_labels = get_city_labels_df(opt_df)
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer(
                    "ArcLayer",
                    data=routes_opt,
                    get_source_position=["from_lon", "from_lat"],
                    get_target_position=["to_lon", "to_lat"],
                    get_width=2,
                    get_source_color="color",
                    get_target_color="color",
                    get_height=1.5
                ),
                pdk.Layer(
                    "TextLayer",
                    data=city_labels,
                    get_position='[lon, lat]',
                    get_text='city',
                    get_size=14,
                    get_color=[0, 0, 0],
                )
            ],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3, pitch=30),
            height=400
        ))

    with tab3:
        routes_cons = get_lane_routes_df(cons_df)
        city_labels = get_city_labels_df(cons_df.rename(columns={"Origin":"Origin","Destination":"Destination"}))
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer(
                    "ArcLayer",
                    data=routes_cons,
                    get_source_position=["from_lon", "from_lat"],
                    get_target_position=["to_lon", "to_lat"],
                    get_width=3,
                    get_source_color="color",
                    get_target_color="color",
                    get_height=2
                ),
                pdk.Layer(
                    "TextLayer",
                    data=city_labels,
                    get_position='[lon, lat]',
                    get_text='city',
                    get_size=14,
                    get_color=[0, 0, 0],
                )
            ],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3, pitch=30),
            height=400
        ))

# ----------------- Business Context & Key Insights -----------------
st.markdown("## 📖 Business Context & Key Insights")
st.markdown("""
- **As‑Is vs Optimized vs Consolidated:**  
  - **As‑Is (Policy):** Parcel for ≤150 lbs, LTL otherwise.  
  - **Optimized Mix:** Rate-shop each shipment (Parcel allowed only when ≤150 lbs) — some shipments switch **from Parcel to LTL** to reduce cost.  
  - **Consolidated LTL:** All shipments are moved as **consolidated lane-level LTL**, leveraging **economies of scale**.

- **What you see on the maps:**  
  - **As‑Is:** Many blue (Parcel) + orange (LTL) arcs.  
  - **Optimized:** **Fewer blue arcs** (Parcel shrinks where LTL is cheaper).  
  - **Consolidated:** **One red arc per lane** — large LTL moves that beat per-shipment costs.

- **Why this matters:**  
  Real TMS engines optimize **both mode selection and consolidation**. This demo shows the mechanics with transparent assumptions:
  - Parcel is **zone-based** and **≤150 lbs only**.  
  - LTL uses **NMFC classes**, **distance**, and **weight breaks**.  
  - Consolidation increases load weight → **lower NMFC factor** → lower $/mile.

- **Extensions you could add next:**  
  - **Dimensional weight** for Parcel.  
  - **Carrier-specific tariffs** (real lane matrices).  
  - **Service levels & SLAs**: add penalties for late deliveries (time windows).  
  - **API-rate ingestion** and **rule parsing with LLMs**.  
""")
