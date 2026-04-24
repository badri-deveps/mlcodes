import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'X': [1,2,3,2,3,4,6,7,8,7,8,9],
    'Y': [1,2,3,3,2,4,6,7,8,6,9,8],
    'Class': [0,0,0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(data)
df.to_csv("perceptronData.csv", index=False)
df = pd.read_csv("perceptronData.csv")

X = df[['X','Y']]
y = df['Class']

model = Perceptron()
model.fit(X, y)

yp = model.predict(X)

print("Accuracy:", accuracy_score(y, yp))
print("Confusion Matrix:\n", confusion_matrix(y, yp))
print("Epochs:", model.n_iter_)

x1, x2 = map(float, input("Enter values: ").split())
test = pd.DataFrame([[x1, x2]], columns=['X','Y'])
print("Prediction:", model.predict(test)[0])
w, b = model.coef_[0], model.intercept_[0]
x_vals = np.linspace(X['X'].min(), X['X'].max(), 50)

plt.scatter(X['X'], X['Y'], c=y)
plt.plot(x_vals, -(w[0]*x_vals + b)/w[1], color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Perceptron Decision Boundary")
plt.show()