import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
]
columns = ['Sky','AirTemp','Humidity','Wind','Water','Forecast','EnjoySport']
df = pd.DataFrame(data, columns=columns)
file_path = "ceData.csv"
df.to_csv(file_path, index=False)
print("CSV file created at:", os.path.abspath(file_path))

df = pd.read_csv(file_path)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

S = ['0'] * len(X[0])
G = [['?'] * len(X[0])]

for i in range(len(X)):
    if y[i] == "Yes":
        for j in range(len(S)):
            if S[j] == '0':
                S[j] = X[i][j]
            elif S[j] != X[i][j]:
                S[j] = '?'
        G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(len(S)))]
    else:  # Negative example
        new_G = []
        for g in G:
            for j in range(len(g)):
                if g[j] == '?':
                    new_h = g.copy()
                    new_h[j] = S[j]
                    if new_h not in new_G:
                        new_G.append(new_h)
        G = new_G
print("\nFinal Specific Hypothesis S:")
print(S)
print("\nFinal General Hypothesis G:")
for g in G:
    print(g)

def predict(sample):
    for j in range(len(S)):
        if S[j] != '?' and S[j] != sample[j]:
            return "No"
    return "Yes"
yp = [predict(x) for x in X]
y_true = [1 if val=="Yes" else 0 for val in y]
y_pred = [1 if val=="Yes" else 0 for val in yp]
print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
# 🔷 Graph (simple visualization)
plt.scatter(range(len(y_true)), y_true, label="Actual")
plt.scatter(range(len(y_pred)), y_pred, marker='x', label="Predicted")
plt.legend()
plt.title("Candidate Elimination (Actual vs Predicted)")
plt.show()