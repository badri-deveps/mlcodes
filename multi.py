import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Experience': [1,2,3,4,5,6,7,8,9,10],
    'EducationLevel': [1,1,2,2,3,3,4,4,5,5],
    'Salary': [20000,25000,30000,35000,45000,50000,60000,65000,70000,80000]
}
df = pd.DataFrame(data)

df.to_csv("multiLinearData.csv", index=False)
print("CSV file created!")
df = pd.read_csv("multiLinearData.csv")

X = df[['Experience', 'EducationLevel']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)
yp = model.predict(X)
print("R2 Score:", r2_score(y, yp))
print("Mean Squared Error:", mean_squared_error(y, yp))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

exp = float(input("Enter Experience: "))
edu = float(input("Enter Education Level: "))

test = pd.DataFrame([[exp, edu]], columns=['Experience','EducationLevel'])
print("Predicted Salary:", model.predict(test)[0])

# graph
plt.scatter(y, yp)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Multiple Linear Regression")
plt.show()