import pandas as pd
data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
]
columns = ['Sky','Temp','Humidity','Wind','Water','Forecast','EnjoySport']
df = pd.DataFrame(data, columns=columns)
df.to_csv("weather.csv", index=False)
print("CSV created!\n")
print(df)
data = pd.read_csv("weather.csv")   # ✅ correct file
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
S = X[0].copy()
G = [["?" for _ in range(len(S))]]
for i in range(len(X)):
    if y[i] == "Yes":
        for j in range(len(S)):
            if S[j] != X[i][j]:
                S[j] = "?"
    else:
        new_G = []
        for g in G:
            for j in range(len(S)):
                if g[j] == "?":
                    if X[i][j] != S[j]:
                        new_h = g.copy()
                        new_h[j] = S[j]
                        new_G.append(new_h)
        G = new_G
print("\nFinal S:", S)
print("Final G:", G)