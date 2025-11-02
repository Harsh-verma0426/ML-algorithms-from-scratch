# =============================================================================
# 🧠 SECTION 1: MACHINE LEARNING USING SCIKIT-LEARN
# -----------------------------------------------------------------------------
# In this section:
# - Use sklearn's built-in models (LinearRegression, DecisionTree, etc.)
# - Focus on applying, tuning, and evaluating models
# - Great for comparing results with your own implementation later
# =============================================================================

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Linear_regression/ols/ols_for _1_feature/house_prices_simple.csv")
X = data[['SquareFeet']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Compare
print(f"Intercept (c): {model.intercept_}")
print(f"Slope (m): {model.coef_}")
print("R² Score:", r2_score(y, y_pred))

# Visualization
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Simple Linear Regression (sklearn)")
plt.legend()
plt.show()



# =============================================================================
# ⚙️ SECTION 2: MACHINE LEARNING FROM SCRATCH (OWN IMPLEMENTATION)
# -----------------------------------------------------------------------------
# In this section:
# - Implement ML algorithms manually using NumPy and logic
# - Build a deeper understanding of the math and workflow behind sklearn
# - Compare results with the sklearn version for validation
# =============================================================================

from ols_for_1_feature import Single_OLS

model_scratch = Single_OLS()
model_scratch.fit(X, y)

# Predictions
y_pred_scratch = model_scratch.predict(X)
m, c = model_scratch.coefficients()

# Compare
print(f"Intercept (c): {c}")
print(f"Slope (m): {m}")
print(f"R² Score: {r2_score(y, y_pred_scratch)}")

# Visualization
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred_scratch, color="red", label="Regression Line")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Simple Linear Regression (scratch)")
plt.legend()
plt.show()