import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = pd.concat(
    [pd.read_csv("Features.csv"), pd.read_csv("Status.csv")],
    axis = 1
    )

Y = pd.read_csv("Selling Price.csv")

X_train , X_test , Y_train , Y_test = train_test_split(
    X , Y , test_size=0.2 , random_state=100
    )

pf = PolynomialFeatures(degree = 2)
pf.fit(X_train)

X_train_pf = pf.transform(X_train)[:,1:]
X_test_pf = pf.transform(X_test)[:,1:]

model=LinearRegression()

model.fit(X_train_pf , Y_train)

Y_pred = model.predict(X_test_pf)

r2 = r2_score(Y_test , Y_pred)
print(r2)