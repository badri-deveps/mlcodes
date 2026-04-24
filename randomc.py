import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 🔷 Create dataset
data = {
    'Feature1': [1,2,3,4,6,7,8,9],
    'Feature2': [2,3,4,5,7,8,9,10],
    'Class':    [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

# 🔷 Save CSV
file_path = "rf_classification.csv"
df.to_csv(file_path, index=False)
print("CSV created at:", os.path.abspath(file_path))

# 🔷 Read CSV
df = pd.read_csv(file_path)

X = df[['Feature1','Feature2']]
y = df['Class']

# 🔷 Train model
model = RandomForestClassifier(n_estimators=10, random_state=0)
model.fit(X, y)

# Predictions
yp = model.predict(X)

# Output
print("\nAccuracy:", accuracy_score(y, yp))
print("Confusion Matrix:\n", confusion_matrix(y, yp))

# 🔷 User input (FIXED PART)
f1, f2 = map(float, input("\nEnter Feature1 Feature2: ").split())

# ✅ Use DataFrame with column names
test = pd.DataFrame([[f1, f2]], columns=['Feature1','Feature2'])

print("Prediction:", model.predict(test)[0])

# 🔷 Graph (Decision Boundary)
h = 0.1
x_min, x_max = X['Feature1'].min()-1, X['Feature1'].max()+1
y_min, y_max = X['Feature2'].min()-1, X['Feature2'].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],
                               columns=['Feature1','Feature2']))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X['Feature1'], X['Feature2'], c=y)

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Random Forest Classification")
plt.show()
