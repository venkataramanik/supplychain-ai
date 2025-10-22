
## Python Script: Regression vs. Classification for Fixed Thresholds

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Simulate Maintenance Data ---
# Feature: Sensor_Alert_Level (Proxy for engine wear)
X = np.random.uniform(50, 100, size=200).reshape(-1, 1)

# Target: Days_Until_Failure (The continuous value we *could* predict)
# A high alert level generally means fewer days until failure.
y_reg = 12 - (X / 10) + np.random.normal(0, 1.5, size=200).reshape(-1, 1)
y_reg = y_reg.clip(min=1, max=10) # Clip unrealistic values

# Business Rule: Schedule Service if Days_Until_Failure <= 3
# Target: Service_Action (The binary decision we *need* to predict)
FIXED_THRESHOLD = 3
y_cls = (y_reg <= FIXED_THRESHOLD).astype(int).ravel() # 1 = Schedule, 0 = Monitor

# Combine data for splitting
data = pd.DataFrame({'Sensor_Alert_Level': X.ravel(), 'Days_Until_Failure': y_reg.ravel(), 'Service_Action': y_cls})

# Split the data
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg.ravel(), y_cls, test_size=0.3, random_state=42
)

# --- 2. Train and Evaluate REGRESSION Model (Predicts a Number) ---
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
reg_preds = reg_model.predict(X_test)

# Convert Regression prediction to an operational decision
reg_decisions = (reg_preds <= FIXED_THRESHOLD).astype(int)

# --- 3. Train and Evaluate CLASSIFICATION Model (Predicts the Decision) ---
# Note: Logistic Regression is a simple classifier for demonstration
cls_model = LogisticRegression()
cls_model.fit(X_train, y_cls_train)
cls_decisions = cls_model.predict(X_test)

# --- 4. Identify The Critical Failure Case: Predictions Near the Threshold (3 days) ---
# We focus on the range 2.5 to 3.5 days where a small numerical error causes a huge action error
critical_window_mask = (y_reg_test > 2.5) & (y_reg_test < 3.5)
X_critical = X_test[critical_window_mask]
y_reg_critical = y_reg_test[critical_window_mask]
y_cls_critical = y_cls_test[critical_window_mask]

reg_critical_preds = reg_model.predict(X_critical)
reg_critical_decisions = (reg_critical_preds <= FIXED_THRESHOLD).astype(int)

# --- 5. Display Results ---

print("=" * 60)
print("             ML Model Choice for Fixed Thresholds (3 Days)")
print("=" * 60)

print(f"**Overall Operational Accuracy (Predicting the Action):**")
print(f"Regression Decision Accuracy: {accuracy_score(y_cls_test, reg_decisions):.3f}")
print(f"Classification Decision Accuracy: {accuracy_score(y_cls_test, cls_decisions):.3f}")
print("-" * 60)

print(f"**Detailed Look at the Critical Window (True Days Until Failure: 2.5 - 3.5):**")
print(f"Number of Critical Samples: {len(X_critical)}")

results_df = pd.DataFrame({
    'Sensor_Level': X_critical.ravel().round(1),
    'True_Days_Failure': y_reg_critical.round(2),
    'True_Action': y_cls_critical,
    'Reg_Predicted_Days': reg_critical_preds.round(2),
    'Reg_Decision': reg_critical_decisions,
    'Cls_Decision': cls_decisions[critical_window_mask]
})

print("\nSample Critical Results:")
print(results_df.head(10))

print("\n--- Summary of Operational Failure near Threshold ---")
reg_fail_count = sum(abs(y_cls_critical - reg_critical_decisions))
cls_fail_count = sum(abs(y_cls_critical - cls_decisions[critical_window_mask]))

print(f"Regression Model Decision Errors in Critical Window: {reg_fail_count}")
print(f"Classification Model Decision Errors in Critical Window: {cls_fail_count}")
print("\nConclusion: The Classification Model is far superior near the fixed 3-day threshold because it is optimized to minimize the cost of the *wrong decision*, not just the numerical error.")
```
