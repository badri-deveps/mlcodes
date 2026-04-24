import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Feature1': [1,2,3,4,6,7,8,9],
    'Feature2': [2,3,4,5,7,8,9,10],
    'Target':   [10,15,20,25,40,45,50,55]
}
df = pd.DataFrame(data)
df.to_csv("dt_regression.csv", index=False)

df = pd.read_csv("dt_regression.csv")

print("Dataset:\n", df)

X = df[['Feature1','Feature2']]
y = df['Target']

model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)

yp = model.predict(X)

print("\n=== Regression ===")
print("R2 Score:", r2_score(y, yp))
print("MSE:", mean_squared_error(y, yp))

f1, f2 = map(float, input("\nEnter Feature1 and Feature2: ").split())
test = pd.DataFrame([[f1, f2]], columns=['Feature1','Feature2'])

print("Predicted Value:", model.predict(test)[0])

plt.figure(figsize=(6,5))
plot_tree(model, filled=True)
plt.title("Decision Tree - Regression")
plt.show()