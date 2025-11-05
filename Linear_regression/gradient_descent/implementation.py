# =============================================================================
# 🧠 SECTION 1: MACHINE LEARNING USING SCIKIT-LEARN
# -----------------------------------------------------------------------------
# In this section:
# - Use sklearn's built-in models (LinearRegression, DecisionTree, etc.)
# - Focus on applying, tuning, and evaluating models
# - Great for comparing results with your own implementation later
# =============================================================================

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Linear_regression/gradient_descent/house_prices_simple.csv")
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

x = X.values.flatten()
y = y.values

# ---- SCALE DATA ---- #
x_scaled = (x - np.mean(x)) / np.std(x)
y_scaled = (y - np.mean(y)) / np.std(y)

# Train gradient descent on SCALED data
model_scratch = LinearRegressionGD()
model_scratch.iteration(20000).learning_rate(0.01).fit(x_scaled, y_scaled)

# Predictions
y_pred_scratch = model_scratch.predict(x_scaled)

# Compare
params = model_scratch.parameters()
print("Parameters (scaled):", params)
print(f"R² Score (scratch): {r2_score(y_scaled, y_pred_scratch)}")

# Visualization
plt.scatter(x_scaled, y_scaled, color="blue", label="Actual (scaled)")
plt.plot(x_scaled, y_pred_scratch, color="red", label="Regression Line (scratch)")
plt.title("Simple Linear Regression - From Scratch")
plt.xlabel("SquareFeet (scaled)")
plt.ylabel("Price (scaled)")
plt.legend()
plt.show()