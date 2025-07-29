Okay, I understand you want me to regenerate the entire `global_trade_insights_app.py` file. I will provide the *exact same code* again.

**Please pay extremely close attention to the following points when you copy and save this file:**

  * **Copy ALL of the code:** Start from the very first `import streamlit as st` line and go all the way to the very last `"""` (triple quote) of the final markdown block.
  * **Use a Plain Text Editor:** Copy the code into a simple text editor (like Notepad on Windows, TextEdit in Plain Text mode on Mac, or Nano/Vim on Linux) rather than a word processor or a complex IDE initially. This helps avoid hidden characters or unintended formatting.
  * **Check Indentation:** Python is very strict about indentation. Ensure that spaces (typically 4 spaces) are used consistently for indentation, and that there are no mixed tabs and spaces, or incorrect indentation levels.
  * **Save as `.py`:** Make sure the file is saved with the `.py` extension, e.g., `global_trade_insights_app.py`.

-----

Here is the complete and corrected code for `global_trade_insights_app.py`:

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime
from sklearn.ensemble import IsolationForest

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="🌍 Global Trade Insights & Landed Cost")

st.title("🌍 Global Trade Insights, Anomaly Detection & Landed Cost Simulation")

st.header("🌟 Business Context: AI for Global Trade Analysis & Cost Optimization")
st.markdown("""
In today's interconnected world, understanding global trade flows and accurately calculating costs are crucial for businesses, governments, and researchers. AI and Machine Learning can unlock powerful insights from vast trade datasets by:

-   **Identifying Trends & Patterns:** Recognizing long-term growth or decline, and seasonal variations in trade volumes.
-   **Detecting Anomalies:** Flagging unusual spikes or drops in trade that might indicate economic shifts, supply chain disruptions, policy changes, or even data errors/fraud.
-   **Forecasting Future Trade:** Predicting future import/export volumes to inform strategic planning.
-   **Optimizing Logistics & Sourcing:** Identifying key partners and commodities to streamline operations.
-   **Calculating Landed Costs:** Understanding the *true* cost of goods, including tariffs, freight, and other fees, to inform pricing, sourcing, and profitability.

This application demonstrates how we can fetch real-world trade data via the **UN Comtrade API** and apply basic ML techniques like **Anomaly Detection**, alongside a **Landed Cost & Tariff Simulator** to highlight interesting trade patterns and financial implications.
""")

st.subheader("💡 How We're Using AI/ML & Simulation for Insights:")
st.markdown("""
1.  **Data Acquisition:** We fetch annual trade data (e.g., Imports of 'Total Merchandise' for 'USA' from '2010' to '2023') from the UN Comtrade API.
2.  **Trend Visualization:** Basic plots help us see the general direction of trade over time.
3.  **Anomaly Detection (Isolation Forest):**
    * We use Scikit-learn's `IsolationForest` model, an unsupervised algorithm effective for finding outliers in data.
    * It works by isolating observations that are few and different (anomalies) from the rest of the observations.
    * The model assigns an anomaly score; values identified as anomalies are explicitly marked for review. This can help quickly pinpoint unusual trade activity that warrants further investigation.
4.  **Landed Cost & Tariff Simulation:**
    * This tool allows you to input various costs (unit cost, freight, insurance) and, crucially, a **tariff rate**.
    * It calculates the total "landed cost" – the true cost of bringing a product to its destination – which is vital for pricing decisions and profit margin analysis.
    * *While this demo uses a user-inputted tariff rate for simplicity, in a real-world AI solution, ML could potentially predict future tariff changes or optimize sourcing based on complex duty structures.*
""")

st.divider()

st.header("🔬 Explore Global Trade Data & Simulate Costs")

# --- UN Comtrade API Helper Functions ---

# Simplified country mapping (ID: Name). In a real app, this would be fetched from a Comtrade API lookup or a comprehensive list.
COUNTRY_CODES = {
    "USA": 842, "China": 156, "Germany": 276, "Japan": 392, "India": 356,
    "United Kingdom": 826, "Canada": 124, "France": 250, "Brazil": 76,
    "Australia": 36, "South Korea": 410, "Italy": 380, "Netherlands": 528,
    "Singapore": 702, "Mexico": 484, "World (All)": "ALL" # 'ALL' is a special code for partner or reporter
}
COUNTRY_NAMES = {v: k for k, v in COUNTRY_CODES.items()} # Reverse for display

# Trade Flow options for the API
TRADE_FLOWS = {
    "Imports": "imports",
    "Exports": "exports",
    "Re-Imports": "re_imports",
    "Re-Exports": "re_exports"
}

# Common Commodity options (HS codes). User can also input custom HS codes.
COMMODITY_OPTIONS = {
    "Total Merchandise (All Commodities)": "TOTAL",
    "Crude Petroleum": "2709", # HS code for Crude Petroleum
    "Passenger Motor Vehicles": "8703", # HS code
    "Pharmaceutical Products": "3004", # HS code
    "Electronic Integrated Circuits": "8542", # HS code
    "Gold (Non-monetary)": "7108", # HS Code for Gold, non-monetary
}

@st.cache_data(ttl=3600, show_spinner="Fetching data from UN Comtrade API (may take a moment)...")
def fetch_comtrade_data(reporter_id, period, flow_code, commodity_code='TOTAL', partner_id='ALL'):
    """Fetches trade data from UN Comtrade API."""
    base_url = "https://comtrade.un.org/api/get/v1/trade/r"

    params = {
        'r': reporter_id,
        'ps': period, # e.g., '2020,2021,2022'
        'freq': 'A', # Annual data
        'px': 'HS', # Harmonized System
        'cc': commodity_code, # 'TOTAL' or specific HS code
        'flow': flow_code,
        'p': partner_id, # 'ALL' or specific country ID
        'format': 'json'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data and data.get('data'):
            df = pd.DataFrame(data['data'])
            # Select and rename relevant columns for clarity
            df = df[['period', 'rtTitle', 'ptTitle', 'cmdCode', 'cmdDesc', 'flowDesc', 'tradeValue']]
            df.columns = ['Year', 'Reporter', 'Partner', 'Commodity_Code', 'Commodity_Description', 'Trade_Flow', 'Trade_Value_USD']
            df['Year'] = pd.to_datetime(df['Year'].astype(str), format='%Y') # Convert to datetime objects
            df['Trade_Value_USD'] = pd.to_numeric(df['Trade_Value_USD'])
            
            # Aggregate by Year, Reporter, Partner, Flow, Commodity for consistent time series
            df = df.groupby(['Year', 'Reporter', 'Partner', 'Trade_Flow', 'Commodity_Code', 'Commodity_Description'])['Trade_Value_USD'].sum().reset_index()

            return df
        else:
            st.warning("No data returned for the selected parameters. Please try different selections or check if data is available for this combination.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from UN Comtrade API: {e}. Please check your internet connection or API limits.")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error processing JSON data: {e}. Data might be in an unexpected format.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


# --- Streamlit Sidebar UI for Trade Data Inputs ---
st.sidebar.header("Global Trade Data Parameters")

selected_reporter_name = st.sidebar.selectbox(
    "Select Reporter Country:",
    list(COUNTRY_CODES.keys()),
    index=list(COUNTRY_CODES.keys()).index("USA")
)
reporter_id = COUNTRY_CODES[selected_reporter_name]

selected_partner_name = st.sidebar.selectbox(
    "Select Partner Country:",
    list(COUNTRY_CODES.keys()),
    index=list(COUNTRY_CODES.keys()).index("World (All)")
)
partner_id = COUNTRY_CODES[selected_partner_name]

selected_flow_name = st.sidebar.selectbox(
    "Select Trade Flow:",
    list(TRADE_FLOWS.keys()),
    index=0 # Default to Imports
)
flow_code = TRADE_FLOWS[selected_flow_name]

selected_commodity_name = st.sidebar.selectbox(
    "Select Commodity:",
    list(COMMODITY_OPTIONS.keys()),
    index=0 # Default to Total Merchandise
)
commodity_code = COMMODITY_OPTIONS[selected_commodity_name]

# Allow custom HS Code input
custom_hs_code = st.sidebar.text_input("Or enter custom HS Commodity Code (e.g., 8703 for cars):", value="")
if custom_hs_code:
    commodity_code = custom_hs_code


# Time period selection
current_year = datetime.datetime.now().year
# Adjusting end year to be current_year - 2, as Comtrade data often has a lag
default_end_year = current_year - 2 
if default_end_year < 2000: # Ensure it doesn't go too low if current_year is low for testing
    default_end_year = 2000 
default_start_year = default_end_year - 15 
if default_start_year < 1990: # Ensure start is not before 1990 for available data
    default_start_year = 1990

year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=1990,
    max_value=current_year, # Max should be current_year
    value=(default_start_year, default_end_year)
)
period_string = ",".join([str(y) for y in range(year_range[0], year_range[1] + 1)])


# --- Fetch Trade Data ---
trade_data_df = fetch_comtrade_data(reporter_id, period_string, flow_code, commodity_code, partner_id)

# --- Main Content Area: Trade Analysis ---
if not trade_data_df.empty:
    st.subheader(f"1. Raw Trade Data ({selected_flow_name} for {selected_reporter_name})")
    st.write(trade_data_df)

    # --- Trend Analysis ---
    st.subheader(f"2. Trade Trend Analysis: {selected_commodity_name} {selected_flow_name} for {selected_reporter_name}")
    
    # Aggregate data by year for trend analysis
    yearly_trade_df = trade_data_df.groupby('Year')['Trade_Value_USD'].sum().reset_index()
    yearly_trade_df['Year_Int'] = yearly_trade_df['Year'].dt.year # For plotting/models that prefer int years

    if not yearly_trade_df.empty:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=yearly_trade_df, x='Year', y='Trade_Value_USD', marker='o', ax=ax1)
        ax1.set_title(f'{selected_flow_name} of {selected_commodity_name} by {selected_reporter_name} Over Time (USD)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Trade Value (USD)')
        ax1.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}B')) # Format to Billions
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)

        st.markdown(f"""
        The chart above shows the annual {selected_flow_name.lower()} value for **{selected_commodity_name}** by **{selected_reporter_name}** {
            f"with **{selected_partner_name}**" if selected_partner_name != "World (All)" else "with all partners"
        }.
        You can observe overall trends (growth, decline, stability) and year-to-year fluctuations.
        """)

        # --- Anomaly Detection ---
        st.subheader("3. Anomaly Detection in Trade Volume (Using Isolation Forest)")

        X = yearly_trade_df[['Trade_Value_USD']].values

        if len(X) > 1: # Need at least 2 samples for IsolationForest
            try:
                model_if = IsolationForest(random_state=42, contamination='auto') 
                model_if.fit(X)

                yearly_trade_df['anomaly'] = model_if.predict(X)
                yearly_trade_df['anomaly_score'] = model_if.decision_function(X) 

                anomalies_df = yearly_trade_df[yearly_trade_df['anomaly'] == -1]

                if not anomalies_df.empty:
                    st.warning("Anomalies Detected! Review the marked points in the chart below.")
                    st.dataframe(anomalies_df[['Year', 'Trade_Value_USD', 'anomaly_score']].sort_values(by='Year').set_index('Year'))
                else:
                    st.info("No significant anomalies detected for the selected period and commodity based on this model.")

                fig_anomaly, ax_anomaly = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=yearly_trade_df, x='Year', y='Trade_Value_USD', marker='o', label='Normal Trade', ax=ax_anomaly, color='blue')
                if not anomalies_df.empty:
                    sns.scatterplot(data=anomalies_df, x='Year', y='Trade_Value_USD', color='red', s=100, label='Anomaly', ax=ax_anomaly, zorder=5) 
                
                ax_anomaly.set_title(f'Trade Volume with Anomalies Highlighted for {selected_commodity_name}')
                ax_anomaly.set_xlabel('Year')
                ax_anomaly.set_ylabel('Trade Value (USD)')
                ax_anomaly.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}B'))
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                st.pyplot(fig_anomaly)

                st.markdown("""
                The red points in the chart highlight years identified as **anomalies** by the Isolation Forest model.
                These are data points that deviate significantly from the typical trade patterns for this commodity and country.
                Anomalies can point to:
                -   Unexpected economic events
                -   Policy changes affecting trade
                -   Supply chain disruptions
                -   Large one-off transactions
                -   Potential data recording issues.
                """)

            except Exception as e:
                st.error(f"Error during anomaly detection: {e}. This might occur if there's insufficient data points or too little variance for the model to work.")
        else:
            st.info("Not enough data points to perform anomaly detection for the selected parameters. Need at least 2 years of data.")


    # --- Top Partners Analysis ---
    st.subheader(f"4. Top Trading Partners ({selected_flow_name}) for {selected_commodity_name}")

    if selected_partner_name == "World (All)":
        partner_summary_df = trade_data_df.groupby('Partner')['Trade_Value_USD'].sum().reset_index()
        partner_summary_df = partner_summary_df.sort_values(by='Trade_Value_USD', ascending=False).head(10)

        if not partner_summary_df.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Partner', y='Trade_Value_USD', data=partner_summary_df, palette='viridis', ax=ax2)
            ax2.set_title(f'Top 10 Trading Partners for {selected_commodity_name} ({selected_flow_name} by {selected_reporter_name})')
            ax2.set_xlabel('Partner Country')
            ax2.set_ylabel('Total Trade Value (USD)')
            ax2.ticklabel_format(style='plain', axis='y') # Prevent scientific notation
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}B')) # Format to Billions
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)
            st.markdown(f"""
            This chart displays the top 10 trading partners for **{selected_commodity_name}**
            ({selected_flow_name.lower()} by **{selected_reporter_name}**) over the selected period.
            This helps in identifying key partners in global supply chains.
            """)
        else:
            st.info("No partner-specific data to display for the selected parameters.")
    else:
        st.info(f"Top partner analysis is most relevant when 'World (All)' is selected as the partner. Showing data for selected partner: **{selected_partner_name}** in the above trend analysis.")

else:
    st.info("Please select parameters from the sidebar to fetch and analyze trade data.")

st.divider()

# --- Landed Cost & Tariff Simulator Section ---
st.header("5. 💸 Landed Cost & Tariff Simulator")
st.info("Simulate the true cost of bringing a product to market, including tariffs and other fees. This helps in pricing, sourcing, and profitability analysis.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Product & Shipping Details")
    unit_cost = st.number_input("Unit Cost of Goods ($)", min_value=0.01, value=100.00, format="%.2f")
    quantity = st.number_input("Quantity", min_value=1, value=1000)
    
    freight_cost = st.number_input("Freight Cost (Shipping) ($)", min_value=0.00, value=500.00, format="%.2f")
    insurance_cost = st.number_input("Insurance Cost ($)", min_value=0.00, value=50.00, format="%.2f")

    st.markdown("---")
    st.subheader("Tariff & Other Import Fees")
    tariff_rate = st.slider("Tariff Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, format="%.1f")
    other_fees = st.number_input("Other Import Fees (Customs Brokerage, Port Fees, etc.) ($)", min_value=0.00, value=200.00, format="%.2f")

with col2:
    st.subheader("Simulation Parameters (for context)")
    origin_country_sim = st.text_input("Origin Country (e.g., China)", value="China")
    destination_country_sim = st.text_input("Destination Country (e.g., USA)", value="USA")
    hs_code_sim = st.text_input("HS Commodity Code (e.g., 8703 for cars)", value="8703")

    st.markdown("---")
    st.subheader("Calculated Landed Cost")

    # Calculations
    total_goods_value = unit_cost * quantity
    duties = total_goods_value * (tariff_rate / 100)
    
    total_landed_cost = total_goods_value + freight_cost + insurance_cost + duties + other_fees
    cost_per_unit_landed = total_landed_cost / quantity if quantity > 0 else 0

    st.metric("Total Goods Value", f"${total_goods_value:,.2f}")
    st.metric("Calculated Duties (Tariff)", f"${duties:,.2f}")
    st.metric("Total Landed Cost", f"**${total_landed_cost:,.2f}**")
    st.metric("Landed Cost Per Unit", f"**${cost_per_unit_landed:,.2f}**")

    st.markdown("""
    <p style="font-size: small; color: gray;">
    This simulator provides a simplified calculation of landed cost. In reality, duties can be more complex (e.g., specific duties, anti-dumping duties, preferential rates).
    </p>
    """, unsafe_allow_html=True)


st.divider()

st.header("🚀 Beyond This Demo: Advanced AI in Global Trade")
st.markdown("""
This application offers a glimpse into AI's potential in trade analysis and cost management. More advanced applications include:
-   **Predictive Analytics (Forecasting):** Using time series models (ARIMA, Prophet, LSTMs) to forecast future trade volumes and **future landed costs**.
-   **Automated Tariff Classification:** AI models that suggest the correct HS code for a product, reducing errors and saving time.
-   **Optimized Sourcing:** AI recommending the best origin countries based on a holistic view of unit cost, tariffs, lead times, and risk.
-   **Trade Compliance Automation:** AI assisting in checking regulations, sanctions, and documentation requirements.
-   **Network Analysis:** Mapping global trade relationships to identify dependencies and vulnerabilities.
-   **Sentiment Analysis:** Analyzing news and social media to predict trade impacts from geopolitical events.

By combining historical trade data with economic indicators, geopolitical events, and AI, businesses can build resilient and proactive global trade strategies, ensuring both profitability and compliance.
""")
```
