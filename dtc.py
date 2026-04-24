import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Feature1': [1,2,3,4,6,7,8,9],
    'Feature2': [2,3,4,5,7,8,9,10],
    'Class':    [0,0,0,0,1,1,1,1]
}
df = pd.DataFrame(data)
df.to_csv("dt_classification.csv", index=False)
df = pd.read_csv("dt_classification.csv")

print("Dataset:\n", df)

X = df[['Feature1','Feature2']]
y = df['Class']

model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

yp = model.predict(X)

print("\n=== Classification ===")
print("Accuracy:", accuracy_score(y, yp))
print("Confusion Matrix:\n", confusion_matrix(y, yp))

f1, f2 = map(float, input("\nEnter Feature1 and Feature2: ").split())
test = pd.DataFrame([[f1, f2]], columns=['Feature1','Feature2'])

print("Predicted Class:", model.predict(test)[0])
plt.figure(figsize=(6,5))
plot_tree(model, filled=True)
plt.title("Decision Tree - Classification")
plt.show()