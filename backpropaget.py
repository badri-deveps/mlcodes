import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data = {
    'X1': [1,2,3,2,3,6,7,8,7,8],
    'X2': [1,2,3,3,2,6,7,8,6,9],
    'Class': [0,0,0,0,0,1,1,1,1,1]
}
df = pd.DataFrame(data)
df.to_csv("bpData.csv", index=False)
print("CSV file created!")

df = pd.read_csv("bpData.csv")
X = df[['X1','X2']]
y = df['Class']
model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=3000, random_state=0)
model.fit(X, y)
yp = model.predict(X)

print("Accuracy:", accuracy_score(y, yp))
print("Confusion Matrix:\n", confusion_matrix(y, yp))

x1, x2 = map(float, input("Enter values: ").split())
test = pd.DataFrame([[x1, x2]], columns=['X1','X2'])
print("Prediction:", model.predict(test)[0])

h = 0.1
x_min, x_max = X['X1'].min()-1, X['X1'].max()+1
y_min, y_max = X['X2'].min()-1, X['X2'].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X['X1'], X['X2'], c=y)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Backpropagation (Neural Network)")
plt.show()