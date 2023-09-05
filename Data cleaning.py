import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle

warnings.filterwarnings("ignore")

raw_data=pd.read_csv("Copper_Set.csv")
cols=raw_data.columns

filtered_data=raw_data[
    raw_data['id'].notnull() &
    raw_data['thickness'].notnull() &
    raw_data['status'].notnull() &
    raw_data['item_date'].notnull() &
    raw_data['customer'].notnull() &
    raw_data['country'].notnull() &
    raw_data['application'].notnull() &
    raw_data['delivery date'].notnull() &
    raw_data['selling_price'].notnull()
    ]

filtered_data['item_date'] = pd.to_datetime(
    filtered_data['item_date'],
    format='%Y%m%d',
    errors='coerce'
    ).dt.date

filtered_data['quantity tons'] = pd.to_numeric(
    filtered_data['quantity tons'],
    errors='coerce'
    )

filtered_data['delivery date'] = pd.to_datetime(
    filtered_data['delivery date'],
    format='%Y%m%d',
    errors='coerce'
    ).dt.date

def replace_with_null(value):
    if isinstance(value,str) and value.startswith('0000'):
        return pd.NA
    return value

filtered_data['material_ref'] = filtered_data['material_ref'].apply(
    replace_with_null
    )

filtered_data['quantity tons'].fillna(
    filtered_data['quantity tons'].mode().iloc[0],
    inplace=True
    )

filtered_data = filtered_data[filtered_data['item_date'].notnull() &
                              filtered_data['delivery date'].notnull()
                              ]
filtered_data = filtered_data[filtered_data['quantity tons']>0]
filtered_data = filtered_data[filtered_data['selling_price']>0]

final_data = filtered_data[
    [
     'quantity tons',
     'country',
     'item type',
     'application',
     'thickness',
     'width',
     'material_ref',
     'product_ref',
     'status',
     'selling_price',
     'customer'
     ]
    ]

final_data['quantity adjusted'] = np.log(final_data['quantity tons'])
final_data['selling_price_converted'] = np.log(final_data['selling_price'])
final_data['customer converted'] = np.log(final_data['customer'])

print(final_data.skew())

X=final_data[
    [
     'quantity adjusted',
     'width',
     'thickness',
     'country',
     'item type',
     'application',
     'product_ref',
     'customer converted'
     ]
    ]

countries=X['country'].unique().tolist()
item_type=X['item type'].unique().tolist()
application=X['application'].unique().tolist()
product_ref=X['product_ref'].unique().tolist()
customer=final_data['customer'].unique().tolist()

Y1=final_data['selling_price_converted']
Y2=final_data['status']

list_dict={
    'countries':countries,
    'item_type':item_type,
    'application':application,
    'product_ref':product_ref,
    'status':Y2.unique().tolist(),
    'customer':customer
    }

with open('list_dict.pkl', 'wb') as f:
    pickle.dump(list_dict,f)

Encoder_X = ColumnTransformer(
    [('onehot', OneHotEncoder(), ['item type'])],
    remainder='passthrough'
)

Encoder_Y2 = ColumnTransformer(
    [('onehot', OneHotEncoder(), ['status'])],
    remainder='passthrough'
)

Encoder_X.fit(X)
Encoder_Y2.fit(pd.DataFrame(Y2))
X_encoded = Encoder_X.transform(X)
Y2_encoded = pd.DataFrame(Encoder_Y2.transform(pd.DataFrame(Y2)).toarray())

X_expanded = pd.DataFrame(X_encoded)

scaler = StandardScaler()
scaler.fit(X_expanded)

X_scaled = pd.DataFrame(scaler.transform(X_expanded))

X_scaled.to_csv("Features.csv",index=False)
Y1.to_csv("Selling Price.csv",index=False)
Y2_encoded.to_csv("Status.csv",index=False)

with open('Data transformation/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('Data transformation/encoder_X.pkl', 'wb') as f:
    pickle.dump(Encoder_X, f)

with open('Data transformation/encoder_Y.pkl', 'wb') as f:
    pickle.dump(Encoder_Y2, f)