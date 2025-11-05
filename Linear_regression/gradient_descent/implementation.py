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

data = pd.read_csv("house_prices_simple.csv")
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

from gradient_descent import LinearRegressionGD

model_scratch = LinearRegressionGD()
model_scratch.iteration(10000).learning_rate(0.01).fit(X, y)

# Predictions
y_pred_scratch = model_scratch.predict(X)

# Compare
params = model_scratch.parameters()
print(f"R² Score: {r2_score(y, y_pred_scratch)}")

# Visualization
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred_scratch, color="red", label="Regression Line")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Simple Linear Regression (scratch)")
plt.legend()
plt.show()