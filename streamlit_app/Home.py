import streamlit as st

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("SupplyChain.ai")
st.subheader("AI-Powered Optimization Pilots for Supply Chain & Logistics")

st.markdown("""
Welcome to **SupplyChain.ai** — a portfolio of interactive prototypes demonstrating how  
**AI, Machine Learning, and Optimization** can tackle real-world supply chain and logistics challenges.
""")

# ---------------------------------------------------------
# About Me Section
# ---------------------------------------------------------
st.markdown("""
---

### **About Me**
Hi, I’m **Venkat Krishnan**, a supply chain transformation leader with 20+ years of experience.  
I am passionate about blending **business strategy with AI/ML and optimization** to solve  
challenges in transportation, warehousing, and global trade.

This portfolio highlights:
- **Business problem framing** — turning logistics pain points into optimization problems.
- **Rapid prototyping** using open-source AI/ML and optimization libraries.
- **Storytelling with data** — showing how these solutions create measurable business impact.

[**Connect with me on LinkedIn**](https://www.linkedin.com/in/venkrish1/)
""")

# ---------------------------------------------------------
# Why I Built This
# ---------------------------------------------------------
st.markdown("""
---

### **Why I Built This**
I created **SupplyChain.ai** to showcase:
- How quickly we can build **AI-powered pilots** for supply chain problems.
- The **business value of optimization** (cost savings, SLA compliance, network efficiency).
- The potential of **open-source tools** like Python, OR-Tools, and Streamlit for enterprise solutions.
""")

# ---------------------------------------------------------
# Explore Pilots
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## **Explore the Pilots**")
st.page_link("pages/TransportationSuite.py", label="🚛 Go to Transportation Optimization Suite")

st.markdown("""
(More pilots — e.g., **Tariff Impact Simulator** and **Network Design Tool** — coming soon!)
""")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
---

""", unsafe_allow_html=True)
