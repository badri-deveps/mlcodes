import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Experience': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [20000,25000,30000,35000,40000,50000,55000,60000,65000,70000]
}
df = pd.DataFrame(data)
df.to_csv("linearData.csv", index=False)
print("CSV file 'linearData.csv' created!")

df = pd.read_csv("linearData.csv")
X = df[['Experience']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)
yp = model.predict(X)

print("R2 Score:", r2_score(y, yp))
print("Mean Squared Error:", mean_squared_error(y, yp))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

exp = float(input("Enter experience: "))
test = pd.DataFrame([[exp]], columns=['Experience'])
print("Predicted Salary:", model.predict(test)[0])
# 🔷 Graph
plt.scatter(X, y)
plt.plot(X, yp)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()