import streamlit as st
import streamlit_shadcn_ui as ui

# --------------------------------------------------------------------------------
# ✅ Streamlit setup
# --------------------------------------------------------------------------------
st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# --------------------------------------------------------------------------------
# 🏷️ Header Section with Shadcn Badges
# --------------------------------------------------------------------------------
ui.badges(
    badge_list=[
        ("SupplyChain.ai", "default"),
        ("AI/ML", "secondary"),
        ("Optimization", "destructive")
    ],
    class_name="mb-2 flex gap-2",
    key="home_badges"
)

st.header("SupplyChain.ai")
st.caption("AI-powered optimization pilots for logistics & supply chain leaders")

# --------------------------------------------------------------------------------
# 🚀 Welcome & CTA Buttons
# --------------------------------------------------------------------------------
st.markdown("### 👋 Welcome")
st.markdown("""
Welcome to **SupplyChain.ai** — a curated set of interactive pilots that demonstrate how  
**AI, Machine Learning, and Optimization** can solve complex supply chain and logistics challenges.
""")

with ui.element("div", className="flex gap-2 mt-2", key="cta_buttons"):
    ui.element("link_button", text="Get Started", url="/TransportationSuite", variant="primary", key="btn_start")
    ui.element("link_button", text="GitHub", url="https://github.com/venkataramanik", variant="outline", key="btn_github")

# --------------------------------------------------------------------------------
# 👤 About Me & Connect
# --------------------------------------------------------------------------------
st.markdown("### 👤 About Me")
st.markdown("""
Hi, I’m **Venkat Krishnan**, a digital transformation and supply chain strategy leader  
with over **20 years of experience** helping enterprises modernize operations, optimize costs,  
and adopt intelligent automation.

🚛 **What I do best**:  
Blending business strategy with **AI/ML, mathematical optimization, and modern ERP platforms**  
to solve problems in **transportation, warehousing, fulfillment, and global trade**.

🧩 I’ve delivered impactful work across:
- Manufacturing & Distribution
- Logistics & Last Mile
- Energy, Utilities, and Tech
- Public Sector & Retail

📈 What I care about:
- Building **data-driven pilots** that deliver real business value
- Framing operational pain points as **ML or optimization problems**
- Rapid prototyping using **open-source tools** (OR-Tools, scikit-learn, Pyomo, Streamlit)
- Translating data into stories that business & technical leaders can act on

💡 This portfolio is my way of showing what’s possible — fast, functional AI solutions that deliver measurable impact.
""")

with st.expander("📬 Let’s Connect"):
    st.markdown("""
I'm always happy to exchange ideas around **AI/ML in supply chain**, or collaborate on digital initiatives.

- 🔗 [**LinkedIn**](https://www.linkedin.com/in/venkrish1/)
- 🧪 [**GitHub Projects**](https://github.com/venkataramanik)
- ✉️ Reach out if you'd like to bring these kinds of ideas to life in your business!
""")

# --------------------------------------------------------------------------------
# 🗂️ What’s Inside
# --------------------------------------------------------------------------------
st.markdown("### 🗂️ What’s Inside")
st.markdown("""
This portfolio includes:
- 🚛 **Transportation Optimization Suite**  
- ⏰ **Time Window & VRP Planning Tools**  
- 🌐 **Network Design & Cross-Docking Models**  
- 📊 **Analytics: Supplier Risk & Demand Volatility**

👉 Use the **sidebar** to explore each solution!
""")

# --------------------------------------------------------------------------------
# 💡 Why I Built This
# --------------------------------------------------------------------------------
st.markdown("### 💡 Why I Built This")
st.markdown("""
I created **SupplyChain.ai** to showcase:
- How to build AI pilots *fast* using Streamlit and Python
- The power of optimization to drive **cost savings and service levels**
- The potential of **open-source libraries** (like OR-Tools, Pyomo, and Pandas) in real-world supply chain use cases
""")

# --------------------------------------------------------------------------------
# 📬 Footer
# --------------------------------------------------------------------------------
st.markdown("---")
st.caption("🔗 [LinkedIn](https://www.linkedin.com/in/venkrish1/) | Built with 💡 using Streamlit & Shadcn UI")
