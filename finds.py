import pandas as pd
import os
data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
]
columns = ['Sky','AirTemp','Humidity','Wind','Water','Forecast','EnjoySport']
df = pd.DataFrame(data, columns=columns)
file_path = "findsData.csv"
df.to_csv(file_path, index=False)
print("CSV file created at:", os.path.abspath(file_path))

df = pd.read_csv(file_path)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

h = ['0'] * len(X[0])

for i in range(len(X)):
    if y[i] == "Yes":
        for j in range(len(h)):
            if h[j] == '0':
                h[j] = X[i][j]
            elif h[j] != X[i][j]:
                h[j] = '?'

print("\nFinal Hypothesis:")
print(h)

test = input("\nEnter values (comma separated): ").split(",")
result = "Yes"
for i in range(len(h)):
    if h[i] != '?' and h[i] != test[i]:
        result = "No"
        break
print("Prediction:", result)