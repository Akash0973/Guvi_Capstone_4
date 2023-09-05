import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
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

param_grid = {'max_depth': [10, 20, 30, 40],
              'min_samples_split': [1, 2, 3, 5],
              'min_samples_leaf': [1, 2, 3, 5],
              'max_features': [1, 'sqrt', 'log2']}

grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring='accuracy'
    )
grid_search.fit(X_train, Y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(Y_test, Y_pred, target_names=['Lost','Won'])
print("\nClassification Report:\n", report)