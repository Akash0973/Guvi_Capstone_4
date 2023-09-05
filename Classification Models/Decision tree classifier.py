import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

X = pd.concat(
    [pd.read_csv("Features.csv"), pd.read_csv("Selling Price.csv")],
    axis = 1
    )

with open('status mapping.pkl', 'rb') as Y_map:
    col_map=pickle.load(Y_map)

Y = pd.read_csv("Status.csv")
mask=Y.iloc[:,[col_map['Won'],col_map['Lost']]].sum(axis=1)==1

X=X[mask]
Y=Y[mask].iloc[:,[col_map['Won']]]

X_train , X_test , Y_train , Y_test = train_test_split(
    X , Y , test_size=0.2 , random_state=100 , stratify = Y
    )

model=DecisionTreeClassifier(random_state=100)

model.fit(X_train , Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(Y_test, Y_pred, target_names=['Lost','Won'])
print("\nClassification Report:\n", report)