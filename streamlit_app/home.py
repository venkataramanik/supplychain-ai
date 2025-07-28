import streamlit as st

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# Title & Intro
st.title("🏠 SupplyChain.ai")
st.subheader("AI-powered optimization pilots for supply chain and logistics")

st.markdown("""
Welcome to **SupplyChain.ai**, a demo portfolio showcasing how AI, optimization, and data science  
can tackle real-world supply chain challenges.

Use the sidebar to explore interactive pilots built with:
- 📦 Open-source solvers (OR-Tools, PuLP)
- 📈 Python data libraries (Pandas, scikit-learn)
- 🎯 Streamlit for rapid prototyping
""")

st.markdown("---")

# Optimization Models
st.markdown("### ⚙️ Optimization Models")
st.markdown("""
- 🚛 **Transportation Suite**  
- 📦 **Shipment Planner – Basic**  
- 🛻 **VRP Optimization**  
- ⏱ **VRPTW – Time Windows**  
- 🚢 **Multi-Modal Planner**  
- 🔄 **Cross-Dock Routing**  
- ⚡ **Dynamic Re-Routing**  
- 🌐 **Network Design**
""")

# Analytics Section
st.markdown("### 📊 Analytics")
st.markdown("""
- 📈 **Demand Volatility**  
- 🛡️ **Supplier Risk**
""")

st.markdown("---")

# About Me
st.markdown("### 👤 About Me")
st.markdown("""
Hi, I’m **Venkat Krishnan**, a supply chain transformation leader with 20+ years of experience.  
I am passionate about blending **business strategy with AI/ML and optimization** to solve  
challenges in transportation, warehousing, and global trade.

This portfolio highlights:
- **Business problem framing** — turning logistics pain points into optimization problems.
- **Rapid prototyping** using open-source AI/ML and optimization libraries.
- **Storytelling with data** — showing how these solutions create measurable business impact.
""")

# Why I Built This
st.markdown("### 💡 Why I Built This")
st.markdown("""
I created **SupplyChain.ai** to showcase:
- How quickly we can build **AI-powered pilots** for supply chain problems.
- The **business value of optimization** (cost savings, SLA compliance, network efficiency).
- The potential of **open-source tools** like Python, OR-Tools, and Streamlit for enterprise solutions.
""")

# Footer
st.markdown("---")
st.markdown("🔗 [**Connect with me on LinkedIn**](https://www.linkedin.com/in/venkrish1/)")
st.caption("Built with ❤️ using Streamlit · GitHub: [venkataramanik/supplychain-ai](https://github.com/venkataramanik/supplychain-ai)")
