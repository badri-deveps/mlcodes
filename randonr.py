import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🔷 Step 1: Create dataset
data = {
    'Feature1': [1,2,3,4,6,7,8,9],
    'Feature2': [2,3,4,5,7,8,9,10],
    'Target':   [10,15,20,25,40,45,50,55]
}

df = pd.DataFrame(data)

# 🔷 Step 2: Save CSV
file_path = "rf_regression.csv"
df.to_csv(file_path, index=False)
print("CSV created at:", os.path.abspath(file_path))

# 🔷 Step 3: Read CSV
df = pd.read_csv(file_path)

X = df[['Feature1','Feature2']]
y = df['Target']

# 🔷 Step 4: Train model
model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X, y)

# Predictions
yp = model.predict(X)

# 🔷 Step 5: Output
print("\nR2 Score:", r2_score(y, yp))
print("MSE:", mean_squared_error(y, yp))

# 🔷 Step 6: User input (FIXED)
f1, f2 = map(float, input("\nEnter Feature1 Feature2: ").split())

# ✅ Use DataFrame to avoid warning
test = pd.DataFrame([[f1, f2]], columns=['Feature1','Feature2'])

print("Prediction:", model.predict(test)[0])

# 🔷 Step 7: Graph (Actual vs Predicted)
plt.scatter(y, yp)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest Regression")
plt.show()