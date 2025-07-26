import streamlit as st
import pandas as pd
import numpy as np

# Try Altair for a nicer chart; fall back to st.bar_chart if not installed
try:
    import altair as alt
    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False

st.set_page_config(page_title="Rating & Mode Selection (with Consolidation)", layout="wide")

# --------- Navigation (optional) ----------
st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

# --------------- Title --------------------
st.title("🚚 Rating & Mode Selection — with LTL Consolidation")
st.markdown("""
This pilot demonstrates a **more realistic rating engine**:
- **Parcel** rated with **zone multipliers** (distance → zone).  
- **LTL** rated with **weight breaks / freight class factors**.  
- **Optimization**: choose **Parcel vs LTL** for each shipment.  
- **Consolidation**: group shipments **on the same O/D lane** to ship as **one LTL load**, and compare costs.
""")

# ----------------- Data -------------------
np.random.seed(42)
origins = ["Atlanta, GA", "Dallas, TX", "Chicago, IL", "Denver, CO", "Seattle, WA",
           "Boston, MA", "Miami, FL", "San Jose, CA", "Phoenix, AZ", "Nashville, TN"]
destinations = ["New York, NY", "Houston, TX", "Los Angeles, CA", "Chicago, IL", "Atlanta, GA",
                "Miami, FL", "Seattle, WA", "Boston, MA", "Dallas, TX", "Denver, CO"]

shipments = pd.DataFrame({
    "ShipmentID": [f"SHP-{i:03d}" for i in range(1, 51)],
    "Origin": np.random.choice(origins, 50),
    "Destination": np.random.choice(destinations, 50),
    "Distance (miles)": np.random.randint(100, 3000, 50),
    "Weight (lbs)": np.random.randint(50, 1000, 50)
})

st.subheader("📦 Shipments Data (50 rows)")
st.dataframe(shipments, use_container_width=True)

# ----------------- Rating Logic -------------------

def parcel_zone_multiplier(distance_miles: float) -> float:
    """Basic zone-like multiplier by distance (toy example)."""
    if distance_miles <= 150:
        return 1.00  # Zone 2-ish
    elif distance_miles <= 600:
        return 1.10  # Zone 3
    elif distance_miles <= 1200:
        return 1.30  # Zone 4
    else:
        return 1.50  # Zone 5+

def ltl_class_factor(weight_lbs: float) -> float:
    """
    Approximate class factor by weight break (toy mapping).
    Lower factor => cheaper rate (heavier shipments → lower classes).
    """
    if weight_lbs < 500:
        return 1.00  # Class ~70
    elif weight_lbs < 1000:
        return 0.90  # Class ~55
    else:
        return 0.85  # Class ~50

def rate_parcel(weight_lbs: float, distance_miles: float) -> float:
    """
    Parcel cost model:
      Cost = (Base + Cost_per_Lb * Weight) * ZoneMultiplier
    No per-mile term here (we're “burying” distance effect in zone multiplier).
    """
    base = 8.0
    cost_per_lb = 0.06
    zone_mult = parcel_zone_multiplier(distance_miles)
    return (base + cost_per_lb * weight_lbs) * zone_mult

def rate_ltl(weight_lbs: float, distance_miles: float) -> float:
    """
    LTL cost model:
      Cost = Base + (Cost_per_Mile * Distance * ClassFactor)
    """
    base = 35.0
    cost_per_mile = 0.42
    cf = ltl_class_factor(weight_lbs)
    return base + (cost_per_mile * distance_miles * cf)

def solve_per_shipment(ship_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Parcel vs LTL for each shipment and pick the cheaper mode."""
    rows = []
    for _, r in ship_df.iterrows():
        dist = r["Distance (miles)"]
        wt = r["Weight (lbs)"]

        parcel_cost = rate_parcel(wt, dist)
        ltl_cost = rate_ltl(wt, dist)

        if parcel_cost <= ltl_cost:
            mode = "Parcel"
            cheapest = parcel_cost
        else:
            mode = "LTL"
            cheapest = ltl_cost

        rows.append({
            "ShipmentID": r["ShipmentID"],
            "Origin": r["Origin"],
            "Destination": r["Destination"],
            "Distance (miles)": dist,
            "Weight (lbs)": wt,
            "Parcel Cost ($)": round(parcel_cost, 2),
            "LTL Cost ($)": round(ltl_cost, 2),
            "Cheapest Mode": mode,
            "Chosen Cost ($)": round(cheapest, 2)
        })
    return pd.DataFrame(rows)

def solve_with_consolidation(ship_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate by (Origin, Destination) lane:
      - Sum weights per lane
      - Use the *mean* distance for that lane (toy assumption)
      - Rate it as a single LTL load
      Parcel is not consolidated.
    """
    consolidated = (
        ship_df
        .groupby(["Origin", "Destination"], as_index=False)
        .agg(
            total_weight=("Weight (lbs)", "sum"),
            avg_distance=("Distance (miles)", "mean"),
            shipments=("ShipmentID", "count")
        )
    )

    consolidated["Consolidated LTL Cost ($)"] = consolidated.apply(
        lambda r: round(rate_ltl(r["total_weight"], r["avg_distance"]), 2),
        axis=1
    )

    return consolidated

# --------------- Solve Button -----------------
if st.button("Solve (Rate & Optimize + Consolidation)"):
    st.subheader("🔍 Per-Shipment Rating (Parcel vs LTL)")
    per_shipment = solve_per_shipment(shipments)
    st.dataframe(per_shipment, use_container_width=True)

    # KPI: total cost without consolidation
    total_no_consol = per_shipment["Chosen Cost ($)"].sum()

    st.subheader("📦 Lane Consolidation as LTL")
    consolidated = solve_with_consolidation(shipments)
    st.dataframe(consolidated, use_container_width=True)

    total_consol = consolidated["Consolidated LTL Cost ($)"].sum()

    # KPI Block
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost (per-shipment optimized)", f"${total_no_consol:,.2f}")
    with col2:
        st.metric("Total Cost (LTL with consolidation)", f"${total_consol:,.2f}")
    with col3:
        savings = total_no_consol - total_consol
        pct = savings / total_no_consol * 100 if total_no_consol > 0 else 0
        st.metric("Savings from Consolidation", f"${savings:,.2f}", f"{pct:.2f}%")

    # -------------- Mode Mix Chart --------------
    mode_mix = per_shipment["Cheapest Mode"].value_counts().reset_index()
    mode_mix.columns = ["Mode", "Shipments"]

    st.subheader("📊 Mode Mix (Chosen per Shipment)")
    if HAS_ALTAIR:
        import altair as alt
        chart = (
            alt.Chart(mode_mix)
            .mark_bar()
            .encode(
                x=alt.X("Mode:N", sort="-y"),
                y=alt.Y("Shipments:Q"),
                color="Mode:N",
                tooltip=["Mode", "Shipments"]
            )
            .properties(width=400, height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.bar_chart(mode_mix.set_index("Mode"))

# ----------------- Explanation -----------------
st.markdown("## 📝 How This Works")
st.markdown("""
**Business framing:**  
We simulate 50 shipments, compute **Parcel vs LTL** cost for each, then **optionally consolidate** LTL by grouping
shipments on the same **Origin–Destination lane**. Consolidation often unlocks **lower LTL class factors and better per-mile economics**.

**Parcel (Zone-Based) Logic**  
We mimic parcel zones by converting distance into a **zone multiplier** (e.g., Zone 2 ≈ 1.0, Zone 5 ≈ 1.5).  
Formula (toy):  
`Cost = (Base + Cost_per_Lb * Weight) * ZoneMultiplier`

**LTL (Freight Class / Weight Breaks) Logic**  
We mimic **NMFC class** dynamics by using a **class factor** that **drops as weight increases**  
(heavier consolidated loads → cheaper per-mile).  
Formula (toy):  
`Cost = Base + (Cost_per_Mile * Distance * ClassFactor)`

**Optimization strategy:**  
- **Without consolidation:** For every shipment, we pick **min(Parcel, LTL)**.  
- **With consolidation:** We group shipments by `(Origin, Destination)` and **rate the entire group as a single LTL load**.  
  Parcel is not consolidated (real world: you *can* partially, but we keep it simple here).

**KPIs shown:**  
- **Total cost without consolidation** (best per shipment).  
- **Total cost with consolidation** (LTL per lane).  
- **Savings %** from consolidation.

---

### Role of Python
This prototype uses:
- **pandas** for fast data transforms and group-bys.
- **NumPy** to generate realistic random test data.
- **Streamlit** to build a production-like UI in minutes.
- **Altair** (or Streamlit native charts) to visualize mode mix & costs.

**Why Python for Supply Chain?**  
Python’s **optimization and OR ecosystem** (e.g., **OR-Tools, PuLP, Pyomo**) plus **ML stacks** (scikit-learn, XGBoost)
make it ideal for **blending optimization + AI** — exactly what modern transportation platforms do.

---

### Where this can go next
- True **zone table lookups** (ZIP-to-zone matrices).  
- **Dimensional weight (DIM)** for parcel.  
- Real LTL rating using **class, weight breaks, lane-based tariffs**.  
- **Carrier-specific rules & APIs**.  
- **Reinforcement Learning** / **ML** to recommend the **best carrier & mode** using historical performance (SLA, cost, on-time %).  
""")
