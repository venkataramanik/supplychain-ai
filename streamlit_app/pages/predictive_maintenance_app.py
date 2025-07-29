# predictive_maintenance_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import datetime

# Set Matplotlib style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid') # A professional-looking seaborn style

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="⚙️ AI for Predictive Maintenance")

st.title("⚙️ AI for Predictive Maintenance in Logistics & Manufacturing")

# --- Business Context & Highlights ---
st.header("🌟 Business Context & Highlights")
st.markdown("""
- **Problem:** Unforeseen equipment failures are a major headache in operations. They cause **expensive downtime**, **disrupt production schedules**, inflate maintenance costs, and can even lead to safety incidents. It's like a truck breaking down mid-route or a key factory machine suddenly stopping – highly disruptive and costly.
- **Solution:** This pilot demonstrates how **Artificial Intelligence (AI)** can predict when equipment is likely to fail *before* it happens. By analyzing sensor data, we move from **reactive repairs** (fixing it after it breaks) to **proactive, smart maintenance** (fixing it just in time).
- **Impact:** Anticipating failures means maintenance can be scheduled during planned downtime, spare parts can be ordered efficiently, and overall operational continuity is significantly improved. It's about maximizing asset uptime and reducing surprises.
""")

st.subheader("💡 Why This Problem Matters")
st.markdown("""
- **Reduced Downtime:** Keeps your trucks moving, your assembly lines running, and your warehouses operating smoothly.
- **Significant Cost Savings:** Avoids costly emergency repairs, overtime for technicians, and wasted inventory due to production halts. Optimizes parts inventory.
- **Extended Asset Lifespan:** Proactive care prevents small issues from becoming catastrophic failures, making your equipment last longer.
- **Enhanced Safety:** Malfunctioning machinery can be dangerous. Predicting failures helps prevent accidents and ensures a safer work environment.
- **Improved Customer Service:** Reliable operations mean consistent product availability and on-time deliveries.
""")

st.subheader("📈 Key Performance Indicators (KPIs) Improved")
st.markdown("""
| KPI Category | Measurement of Improvement |
| :------------------------- | :-------------------------------------------------------------- |
| **Unplanned Downtime** | Reduction in hours/frequency of unexpected equipment stoppages. |
| **Maintenance Costs** | Decrease in total expenditure on repairs and maintenance.        |
| **Asset Utilization** | Increase in the percentage of time equipment is operational.     |
| **Mean Time Between Failures (MTBF)** | Longer intervals between equipment breakdowns.                  |
| **Spare Parts Inventory Optimization** | Reduction in excess inventory and associated carrying costs.    |
""")

st.subheader("🛠️ Core Technologies Used (Pilot)")
st.markdown("""
- **Python:** The programming language orchestrating data handling, AI model training, and the application logic.
- **`pandas` & `numpy`:** Essential libraries for efficient data manipulation and numerical operations.
- **`scikit-learn`:** A powerful and user-friendly library for building and evaluating Machine Learning models.
- **`matplotlib` & `seaborn`:** Libraries for creating clear and insightful data visualizations and charts.
- **Streamlit:** Our framework for turning Python scripts into interactive web applications, making this demo easy to use.
""")

st.subheader("🧠 How the Machine Learns (Non-Technical Explanation of ML Model)")
st.markdown("""
For this pilot, we use a type of AI model called **Logistic Regression**. Think of it like this:

Imagine you have a team of experienced mechanics. Over time, they learn to spot subtle signs that a machine is about to break down – maybe a specific engine temperature, a weird vibration sound, or high pressure readings coupled with many hours of recent use.

Our Logistic Regression model works similarly:
1.  **Learning from History:** We feed the model historical data: "On these days, the machine had these sensor readings and **didn't** fail. On *these other* days, with *these specific* sensor readings, it **did** fail."
2.  **Finding Patterns:** The model then sifts through this historical data to find patterns. It figures out which combinations of temperature, vibration, pressure, and usage hours are most often associated with an upcoming failure. For example, it might learn that high vibration *combined* with rising temperature is a strong indicator.
3.  **Predicting Likelihood:** Once it learns these patterns, when you give it new, real-time sensor data, it can calculate a **probability** – a number between 0% (no chance of failure) and 100% (very high chance of failure) – of how likely that machine is to fail soon.
4.  **Making a Call:** If that probability crosses a certain threshold (e.g., 50% or more), the model flags the machine as "at risk" of failure, prompting proactive action. This allows maintenance teams to step in before an actual breakdown occurs.

It's a straightforward yet effective way for a computer to "learn" from data and help us make smarter decisions about machine upkeep.
""")

st.divider() # Visual separator

st.subheader("📡 IoT Data Ingestion: Where the Data Comes From")
st.markdown("""
In a real-world predictive maintenance scenario, the sensor data (like the temperature, vibration, and pressure readings we're simulating) doesn't just appear. It's collected directly from the physical equipment via **Internet of Things (IoT) sensors and devices**.

Here's a typical flow:

1.  **Physical Sensors:** Devices like accelerometers (for vibration), thermocouples (for temperature), pressure transducers, and hour meters are attached to critical machine components (e.g., motors, pumps, bearings, engines).
2.  **Edge Devices/Gateways:** These sensors transmit their readings to a local "edge device" or "gateway" on the factory floor or in the vehicle. These gateways often perform initial data processing or filtering.
3.  **Communication Protocols:** Data is then sent from the edge to the cloud using various **IoT communication protocols** and APIs. Common ones include:
    * **MQTT (Message Queuing Telemetry Transport):** A lightweight messaging protocol ideal for low-bandwidth, high-latency networks, commonly used in industrial IoT.
    * **AMQP (Advanced Message Queuing Protocol):** Offers more robust messaging with advanced queuing and routing features.
    * **HTTP/HTTPS:** Standard web protocols, also used for IoT data, especially for less frequent data uploads or device management.
4.  **Cloud IoT Platforms:** Major cloud providers offer specialized services designed to securely ingest, process, and manage vast streams of IoT data:
    * **AWS IoT Core:** Amazon's managed cloud service that lets connected devices easily and securely interact with cloud applications and other devices.
    * **Azure IoT Hub:** Microsoft's platform that provides a central message hub for bi-directional communication between your IoT application and the devices it manages.
    * **Google Cloud Pub/Sub (for IoT workloads):** While Google Cloud IoT Core was retired, Google's ecosystem leverages services like Cloud Pub/Sub for high-volume data ingestion and real-time streaming data processing for IoT workloads.

This robust infrastructure ensures that continuous, real-time data is available for the AI models to make timely and accurate predictions about machine health.
""")

st.divider() # Visual separator

st.header("1. Simulated Data Overview")
st.info("We've generated synthetic sensor data (temperature, vibration, pressure, usage hours) for a fleet of machines, including simulated failure events. This data mimics real-world scenarios where equipment shows signs of deterioration before a breakdown.")

@st.cache_data # Cache data generation to avoid re-running on every interaction
def generate_maintenance_data(num_machines=50, num_days=365):
    np.random.seed(42) # for reproducibility

    data = []
    for machine_id in range(1, num_machines + 1):
        # Simulate baseline health for each machine
        baseline_temp = np.random.uniform(70, 90) # F
        baseline_vibration = np.random.uniform(0.5, 1.5) # G's
        baseline_pressure = np.random.uniform(50, 70) # psi
        baseline_usage_rate = np.random.uniform(8, 16) # hours per day

        current_usage_hours = 0
        
        # Simulate a distinct failure pattern for each machine
        failure_imminent_days = np.random.randint(5, 25) # Days before failure when signs appear
        
        # Ensure failure day occurs within the simulation period and allows for pre-failure signs
        failure_day = np.random.randint(failure_imminent_days + 10, num_days - 10) # Failure happens, but not too early or too late

        for day in range(num_days):
            timestamp = datetime.date(2024, 1, 1) + datetime.timedelta(days=day)
            
            # Simulate sensor readings with natural noise
            temp = baseline_temp + np.random.normal(0, 1.5)
            vibration = baseline_vibration + np.random.normal(0, 0.08)
            pressure = baseline_pressure + np.random.normal(0, 0.8)
            
            # Usage hours increase daily
            daily_usage = max(0, baseline_usage_rate + np.random.normal(0, 0.5)) # Ensure non-negative usage
            current_usage_hours += daily_usage

            failure_event = 0 # Default to no failure

            # Introduce gradual signs of deterioration leading up to the failure day
            days_until_failure = failure_day - day
            if 0 < days_until_failure <= failure_imminent_days:
                # Severity of anomaly increases as failure approaches
                anomaly_factor = (failure_imminent_days - days_until_failure) / failure_imminent_days
                
                temp += anomaly_factor * np.random.uniform(5, 20) # Temperature rises significantly
                vibration += anomaly_factor * np.random.uniform(0.5, 3.0) # Vibration increases a lot
                pressure += anomaly_factor * np.random.uniform(3, 15) # Pressure fluctuates more wildly
                
            # Mark actual failure on the specific day
            if day == failure_day:
                failure_event = 1
                # After failure, machine is repaired and usage/sensor data reset for next cycle (simplification)
                current_usage_hours = np.random.uniform(0, 100) # Start new life cycle
                baseline_temp = np.random.uniform(70, 90) # Baselines reset
                baseline_vibration = np.random.uniform(0.5, 1.5)
                baseline_pressure = np.random.uniform(50, 70)

            data.append([machine_id, timestamp, temp, vibration, pressure, current_usage_hours, failure_event])
    
    df = pd.DataFrame(data, columns=['machine_id', 'timestamp', 'temperature', 'vibration', 'pressure', 'usage_hours', 'failure_event'])
    
    # Add a 'predicted_failure_risk' column for later use, initialized to 0
    df['predicted_failure_risk'] = 0.0
    return df

df_raw = generate_maintenance_data(num_machines=50, num_days=365)
st.write(f"Generated data for **{df_raw['machine_id'].nunique()} machines** over **{df_raw['timestamp'].nunique()} days**.")
st.dataframe(df_raw.head())

st.divider()

st.header("2. Machine Learning Model Training")
st.info("Using this historical data, a Logistic Regression model is trained to learn the patterns that precede a machine failure.")

@st.cache_resource # Cache the trained model and its evaluation results
def train_predictive_model(data):
    # Features (X): What the model uses to predict (sensor readings, usage hours)
    # Target (y): What the model tries to predict (failure_event: 1 for failure, 0 for no failure)
    
    X = data[['temperature', 'vibration', 'pressure', 'usage_hours']]
    y = data['failure_event']

    # Split data into training and testing sets (70% for training, 30% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # 'stratify=y' ensures that the proportion of failure events is roughly the same in both train and test sets.

    # Train a Logistic Regression model
    # 'solver='liblinear'' is a good general-purpose solver for smaller datasets.
    # 'class_weight='balanced'' helps the model learn from failures, which are rare compared to non-failures.
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    model.fit(X_train, y_train) # The model learns from the training data

    # Evaluate the model's performance on unseen data (test set)
    y_pred = model.predict(X_test) # Predict outcomes (failure/no-failure)
    y_proba = model.predict_proba(X_test)[:, 1] # Predict the probability of failure

    accuracy = accuracy_score(y_test, y_pred) # Overall correctness
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # Detailed performance metrics

    # ROC Curve: A common way to evaluate classification models, showing trade-off between true positives and false positives
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr) # Area Under the Curve (higher is better)

    return model, accuracy, report, fpr, tpr, roc_auc, X_test, y_test

model, accuracy, report, fpr, tpr, roc_auc, X_test, y_test = train_predictive_model(df_raw)

st.write(f"**Model Overall Accuracy:** `{accuracy:.2f}` (This tells us how often the model was correct in its predictions on unseen data.)")
st.write("**Detailed Model Performance (Classification Report):**")
# Displaying specific key metrics in a user-friendly way
st.info(f"""
- **Precision (for Failures):** `{report['1']['precision']:.2f}` (Out of all times the model predicted a failure, how many were actual failures?)
- **Recall (for Failures):** `{report['1']['recall']:.2f}` (Out of all actual failures, how many did the model correctly identify?)
- **F1-Score (for Failures):** `{report['1']['f1-score']:.2f}` (A balance between Precision and Recall.)
""")

st.write("For more technical details, here's the full report:")
st.json(report)


# Plot ROC Curve
st.subheader("Model Performance: Receiver Operating Characteristic (ROC) Curve")
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr, tpr, color='#007bff', lw=2, label=f'ROC curve (Area = {roc_auc:.2f})') # Using a nice blue color
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
ax_roc.set_ylabel('True Positive Rate (Recall)')
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)
st.markdown("The ROC curve helps visualize the trade-off between correctly identifying failures (True Positive Rate) and incorrectly flagging non-failures (False Positive Rate). A higher Area Under the Curve (AUC) indicates a better performing model.")

st.divider()

st.header("3. Interactive Machine Health Monitoring Dashboard")
st.info("Select a machine to view its historical sensor trends and the model's predicted likelihood of failure over time. This helps maintenance teams proactively identify at-risk assets.")

selected_machine_id = st.selectbox(
    "Select a Machine ID to inspect:",
    df_raw['machine_id'].unique(),
    index=0 # Default to the first machine
)

machine_data = df_raw[df_raw['machine_id'] == selected_machine_id].copy()
# Predict probabilities for the selected machine's data using the trained model
machine_data['predicted_failure_proba'] = model.predict_proba(machine_data[['temperature', 'vibration', 'pressure', 'usage_hours']])[:, 1]

# Plotting sensor trends
st.subheader(f"Sensor Trends for Machine {selected_machine_id}")

# Use Seaborn's lineplot for better visual appeal
fig_sensors, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True) # Increased height
sns.lineplot(x='timestamp', y='temperature', data=machine_data, ax=axes[0], color='red', label='Temperature (°C)')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Temperature Over Time')

sns.lineplot(x='timestamp', y='vibration', data=machine_data, ax=axes[1], color='blue', label='Vibration (G)')
axes[1].set_ylabel('Vibration (G)')
axes[1].set_title('Vibration Over Time')

sns.lineplot(x='timestamp', y='pressure', data=machine_data, ax=axes[2], color='green', label='Pressure (psi)')
axes[2].set_ylabel('Pressure (psi)')
axes[2].set_title('Pressure Over Time')

sns.lineplot(x='timestamp', y='usage_hours', data=machine_data, ax=axes[3], color='purple', label='Usage Hours')
axes[3].set_ylabel('Usage (Hours)')
axes[3].set_xlabel('Date')
axes[3].set_title('Usage Hours Over Time')

# Mark actual failure events on all sensor plots
failure_days = machine_data[machine_data['failure_event'] == 1]['timestamp']
for ax in axes:
    for f_day in failure_days:
        ax.axvline(f_day, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Actual Failure' if f_day == failure_days.iloc[0] else None) # Only label once
    ax.legend(loc='upper left') # Ensure legend shows

plt.tight_layout()
st.pyplot(fig_sensors)


# Plotting predicted failure probability
st.subheader(f"Predicted Failure Probability for Machine {selected_machine_id}")

fig_proba, ax_proba = plt.subplots(figsize=(12, 5))
sns.lineplot(x='timestamp', y='predicted_failure_proba', data=machine_data, ax=ax_proba, color='orange', label='Predicted Failure Probability')
ax_proba.axhline(0.5, color='red', linestyle=':', label='Action Threshold (0.5)') # Example threshold
ax_proba.set_xlabel('Date')
ax_proba.set_ylabel('Probability')
ax_proba.set_title(f'Machine {selected_machine_id} Failure Probability Over Time')
ax_proba.legend()
ax_proba.grid(True, linestyle='--', alpha=0.6)

# Highlight days where probability crosses threshold
threshold_crossings = machine_data[machine_data['predicted_failure_proba'] >= 0.5]
if not threshold_crossings.empty:
    first_warning_day = threshold_crossings['timestamp'].min()
    ax_proba.axvline(first_warning_day, color='darkgreen', linestyle='-', alpha=0.8, linewidth=2, label='First Prediction Above Threshold')
    st.warning(f"**Action Required:** Predicted failure probability for Machine {selected_machine_id} crossed the **0.5 threshold on {first_warning_day.strftime('%Y-%m-%d')}**! This machine is at elevated risk of failure. Consider scheduling proactive maintenance.")
else:
    st.info(f"Machine {selected_machine_id} currently shows no high predicted failure probability (above 0.5) in the simulated data. It appears healthy.")

# Add actual failure markers if any
for f_day in failure_days:
    ax_proba.axvline(f_day, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Actual Failure' if f_day == failure_days.iloc[0] else None)

ax_proba.legend(loc='upper left') # Re-add legend to include actual failure if present
st.pyplot(fig_proba)

st.divider()

st.header("🚀 Next Steps: Scaling with High-Performance AI & Advanced Capabilities")
st.markdown("""
This demonstration uses open-source components that can run on a standard CPU for simplicity.
However, for real-world enterprise applications with massive datasets, complex equipment, and
demanding real-time performance, **specialized hardware and optimized software stacks** are essential.

Here's how this Predictive Maintenance pipeline can be elevated:
""")

st.subheader("Hardware & Performance")
st.markdown("""
-   **Real-time Sensor Data Ingestion:** Integrate with industrial IoT platforms (e.g., AWS IoT, Azure IoT Hub, Google Cloud IoT Core) and high-throughput data streaming solutions (e.g., Apache Kafka) to ingest massive volumes of sensor data in real-time.
-   **Accelerated Model Training:** For more complex, state-of-the-art models (e.g., Deep Learning, advanced Time Series Neural Networks), leverage powerful GPUs (e.g., NVIDIA GPUs) for significantly faster training times, allowing for more frequent model updates and retraining on fresh data.
-   **Edge AI Deployment:** Deploy lightweight, optimized inference models directly on edge devices (e.g., on the machines themselves or in nearby gateways) using technologies like NVIDIA Jetson or TensorRT. This enables immediate, localized failure prediction without constant cloud communication, critical for remote or high-latency environments.
""")

st.subheader("Advanced Data & Models")
st.markdown("""
-   **Complex Time Series Models:** Implement advanced deep learning models (e.g., LSTMs, Transformers, Recurrent Neural Networks) specifically designed to capture intricate temporal patterns and long-term dependencies in sensor data for more accurate long-term failure prediction.
-   **Anomaly Detection:** Incorporate unsupervised learning techniques to detect unusual sensor readings that might indicate emerging, unforeseen issues or novel failure modes, even without historical failure labels.
-   **Multi-modal Data Integration:** Combine sensor data with other crucial data sources like detailed maintenance logs (textual analysis), equipment specifications, external factors (e.g., weather, road conditions for vehicles), and operational schedules for richer feature sets and more comprehensive predictions.
-   **Physics-Informed AI:** Integrate engineering knowledge, physical models, and first principles into the AI model design to improve the robustness, accuracy, and explainability of predictions, especially for complex mechanical systems.
""")

st.subheader("Integration & MLOps")
st.markdown("""
-   **Automated MLOps Pipeline:** Establish robust, end-to-end MLOps pipelines for automated data validation, feature engineering, model retraining, versioning, deployment, and continuous monitoring of model performance in production environments.
-   **Integration with CMMS/ERP:** Seamlessly integrate failure predictions with Computerized Maintenance Management Systems (CMMS) or Enterprise Resource Planning (ERP) systems to automatically trigger work orders, allocate resources, and manage spare parts.
-   **Prescriptive Analytics:** Move beyond simply predicting *when* a machine will fail to prescribing *what specific action* should be taken to prevent it (e.g., "Replace bearing X," "Adjust pressure valve Y," "Perform lubrication on component Z").
-   **AI-Powered Digital Twins:** Create digital replicas of physical assets that integrate real-time sensor data with predictive models, allowing for virtual testing of maintenance strategies and optimization before real-world implementation.
-   **Agentic Workflows:** Develop autonomous AI agents that can not only monitor machine health and predict failures but also initiate maintenance requests, order spare parts automatically, and even coordinate with operational scheduling systems to minimize disruption, acting as an intelligent orchestrator.
""")

st.markdown("""
By leveraging high-performance computing and specialized AI software, enterprises can build highly accurate, scalable, and responsive Predictive Maintenance solutions that truly optimize asset performance, enhance operational resilience, and drive significant cost savings.
""")
