Understood. Below is a **fully rebuilt, deterministic, 10‑shipment version** that:

* **Never randomizes** (hard-coded data).
* Starts with **“Before Optimization”** (each shipment = a load; Parcel if ≤150 lb, else LTL).
* **Scenario 2 – Mode‑Consolidated**: only **LTL** orders on the same lane are consolidated into **fewer LTL loads**; Parcel stays Parcel.
* **Scenario 3 – Cross‑Mode Consolidated**: **parcel orders are moved into LTL and consolidated per lane only when it reduces cost** — guaranteeing **cost strictly decreases** from 1 → 2 → 3.
* Shows **cost & load-count charts** (clear, thin bars).
* Maps draw **loads (not shipments)**, so you’ll actually see **fewer arcs** across scenarios.

> Paste this over **`ShipmentPlanner_Basic.py`**.

---

```python
import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Shipment Planner – Load Optimization", layout="wide")
st.title("🚛 Shipment Planner — Load Optimization (Deterministic 10 orders)")

# ----------------- Fixed, deterministic input data (10 shipments) -----------------
# We deliberately make 3 Parcel-eligible (<=150 lbs) and 7 LTL (>150 lbs).
# There is ONE lane (Atlanta -> Chicago) that has both LTL + Parcel so Scenario 3 can save more.
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

shipments = pd.DataFrame([
    # lane: Atlanta -> Chicago (one parcel + two LTL)
    {"ShipmentID": "ORD-001", "Origin": "Atlanta, GA", "Destination": "Chicago, IL", "Distance (miles)": 715,  "Weight (lbs)": 400},
    {"ShipmentID": "ORD-002", "Origin": "Atlanta, GA", "Destination": "Chicago, IL", "Distance (miles)": 715,  "Weight (lbs)": 350},
    {"ShipmentID": "ORD-003", "Origin": "Atlanta, GA", "Destination": "Chicago, IL", "Distance (miles)": 715,  "Weight (lbs)": 120},  # Parcel-eligible (<=150)

    # lane: Dallas -> Los Angeles (all LTL)
    {"ShipmentID": "ORD-004", "Origin": "Dallas, TX",  "Destination": "Los Angeles, CA", "Distance (miles)": 1435, "Weight (lbs)": 300},
    {"ShipmentID": "ORD-005", "Origin": "Dallas, TX",  "Destination": "Los Angeles, CA", "Distance (miles)": 1435, "Weight (lbs)": 450},
    {"ShipmentID": "ORD-006", "Origin": "Dallas, TX",  "Destination": "Los Angeles, CA", "Distance (miles)": 1435, "Weight (lbs)": 700},

    # lane: Miami -> New York (both Parcel)
    {"ShipmentID": "ORD-007", "Origin": "Miami, FL",   "Destination": "New York, NY", "Distance (miles)": 1275, "Weight (lbs)": 80},   # Parcel-eligible
    {"ShipmentID": "ORD-008", "Origin": "Miami, FL",   "Destination": "New York, NY", "Distance (miles)": 1275, "Weight (lbs)": 120},  # Parcel-eligible

    # lane: Boston -> Houston (Parcel)
    {"ShipmentID": "ORD-009", "Origin": "Boston, MA",  "Destination": "Houston, TX", "Distance (miles)": 1820, "Weight (lbs)": 90},   # Parcel-eligible

    # lane: Denver -> Seattle (LTL)
    {"ShipmentID": "ORD-010", "Origin": "Denver, CO",  "Destination": "Seattle, WA", "Distance (miles)": 1330, "Weight (lbs)": 600},
])

def nmfc_class(weight):
    if weight <= 150: return "70"
    if weight <= 500: return "55"
    return "50"

shipments["NMFC Class"] = shipments["Weight (lbs)"].apply(nmfc_class)

st.subheader("📦 Orders (Before Optimization)")
st.dataframe(shipments, use_container_width=True)

# ----------------- Rating functions -----------------
def parcel_zone_multiplier(distance):
    if distance <= 150: return 1.0
    if distance <= 600: return 1.1
    if distance <= 1200: return 1.3
    return 1.5

def ltl_class_factor(nmfc):
    return {"50": 0.9, "55": 1.0, "70": 1.2}.get(nmfc, 1.0)

def rate_parcel(weight, distance):
    # Only meaningful if <=150 lbs (we'll enforce outside)
    return (8 + 0.06 * weight) * parcel_zone_multiplier(distance)

def rate_ltl(weight, distance, nmfc):
    return 35 + 0.42 * distance * ltl_class_factor(nmfc)

# ----------------- Scenario 1: Before optimization -----------------
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
            "LoadID": r["ShipmentID"],  # 1 order = 1 load
            "Origin": r["Origin"],
            "Destination": r["Destination"],
            "Total Weight (lbs)": r["Weight (lbs)"],
            "Mode": mode,
            "Cost ($)": round(cost, 2),
        })
    return pd.DataFrame(rows)

# ----------------- Scenario 2: Mode-preserving consolidation -----------------
def scenario_2_mode_consolidated(before_df):
    # Split parcel loads (stay as-is) and ltl loads (consolidate per lane)
    parcel = before_df[before_df["Mode"] == "Parcel"].
```
