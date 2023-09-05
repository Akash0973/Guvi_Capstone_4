import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

X = pd.concat(
    [pd.read_csv("Features.csv"), pd.read_csv("Status.csv")],
    axis = 1
    )

Y = pd.read_csv("Selling Price.csv")

X_train , X_test , Y_train , Y_test = train_test_split(
    X , Y , test_size=0.2 , random_state=100
    )

model=RandomForestRegressor(n_estimators=100, random_state=100)

param_grid = {
    'n_estimators': [100,150],
    'max_depth': [15,20,25],
    'min_samples_split': [2,5],
    'min_samples_leaf': [2,4],
    'max_features': ['sqrt','log2']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

r2 = r2_score(Y_test , Y_pred)
print(r2)