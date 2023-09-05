import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X = pd.concat(
    [pd.read_csv("Features.csv"), pd.read_csv("Status.csv")],
    axis = 1
    )

Y = pd.read_csv("Selling Price.csv")

X_train , X_test , Y_train , Y_test = train_test_split(
    X , Y , test_size=0.2 , random_state=100
    )

model=RandomForestRegressor(n_estimators=150, random_state=100)

model.fit(X_train , Y_train)

Y_pred = model.predict(X_test)

r2 = r2_score(Y_test , Y_pred)
print(r2)