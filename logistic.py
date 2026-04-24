import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'X': [1,2,3,2,3,6,7,8,7,8],
    'Y': [1,2,3,3,2,6,7,8,6,9],
    'Class': [0,0,0,0,0,1,1,1,1,1]
}
df = pd.DataFrame(data)

file_path = "logisticData.csv"
df.to_csv(file_path, index=False)
print("CSV file created at:", os.path.abspath(file_path))

df = pd.read_csv(file_path)

X = df[['X','Y']]
y = df['Class']

model = LogisticRegression()
model.fit(X, y)

yp = model.predict(X)

print("\nAccuracy:", accuracy_score(y, yp))
print("Confusion Matrix:\n", confusion_matrix(y, yp))

x1, x2 = map(float, input("\nEnter values: ").split())
test = pd.DataFrame([[x1, x2]], columns=['X','Y'])
print("Prediction:", model.predict(test)[0])
h = 0.1
x_min, x_max = X['X'].min()-1, X['X'].max()+1
y_min, y_max = X['Y'].min()-1, X['Y'].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X['X'], X['Y'], c=y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Logistic Regression Decision Boundary")
plt.show()