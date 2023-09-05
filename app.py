import pandas as pd
import numpy as np
import streamlit as st
import pickle

with open('Classifier_model.pkl', 'rb') as f:
    classifier=pickle.load(f)

with open('Regressor_model.pkl', 'rb') as f:
    regressor=pickle.load(f)

with open('Data transformation/encoder_X.pkl', 'rb') as f:
    encoder_x=pickle.load(f)

with open('Data transformation/encoder_Y.pkl', 'rb') as f:
    encoder_y=pickle.load(f)

with open('Data transformation/scaler.pkl', 'rb') as f:
    scaler=pickle.load(f)

with open('list_dict.pkl', 'rb') as f:
    list_dict=pickle.load(f)

st.write('# Industrical Copper Modeling Project')

tab1,tab2=st.tabs(['predict price','predict status'])

with tab1:
    quantity=np.log(float(st.text_input('Insert Quantity', value=54.15113862)))
    width=float(st.text_input('Insert Width', value=1500))
    thickness=float(st.text_input('Insert Thickness', value=2))
    country=st.selectbox('Select country code',list_dict['countries'])
    item_type=st.selectbox('Select item type',list_dict['item_type'])
    application=st.selectbox('Select application',list_dict['application'])
    product_ref=st.selectbox('Select product_ref',list_dict['product_ref'])
    customer=np.log(float(st.selectbox('Insert Customer ID',list_dict['customer'])))
    status=st.selectbox('Insert Status',list_dict['status'])
    
    data={
        'quantity adjusted':[quantity],
        'width':[width],
        'thickness':[thickness],
        'country':[country],
        'item type':[item_type],
        'application':[application],
        'product_ref':[product_ref],
        'customer converted':[customer]
        }
    X=pd.DataFrame(data)
    X=pd.DataFrame(encoder_x.transform(X))
    X=pd.DataFrame(scaler.transform(X))
    
    Y=pd.DataFrame({'status':[status]})
    Y=pd.DataFrame(encoder_y.transform(Y).toarray())
    
    fin=pd.concat([X,Y],axis=1)
    
    price=np.exp(regressor.predict(fin))[0]
    
    if st.button('Predict Price'):
        st.write('# The predicted price is:',round(price,2))

with tab2:
    quantity=np.log(float(st.text_input('Insert Quantity ', value=54.15113862)))
    width=float(st.text_input('Insert Width ', value=1500))
    thickness=float(st.text_input('Insert Thickness ', value=2))
    country=st.selectbox('Select country code ',list_dict['countries'])
    item_type=st.selectbox('Select item type ',list_dict['item_type'])
    application=st.selectbox('Select application ',list_dict['application'])
    product_ref=st.selectbox('Select product_ref ',list_dict['product_ref'])
    customer=np.log(float(st.selectbox('Insert Customer ID ',list_dict['customer'])))
    price=np.log(float(st.text_input('Insert Price', value=854)))
    
    data={
        'quantity adjusted':[quantity],
        'width':[width],
        'thickness':[thickness],
        'country':[country],
        'item type':[item_type],
        'application':[application],
        'product_ref':[product_ref],
        'customer converted':[customer]
        }
    X_=pd.DataFrame(data)
    X_=pd.DataFrame(encoder_x.transform(X_))
    X_=pd.DataFrame(scaler.transform(X_))
    
    Y_=pd.DataFrame({14:[price]})
    
    fin_=pd.concat([X_,Y_],axis=1)
    status=classifier.predict(fin_)[0]
    
    if status==1:
        status='Won'
    else:
        status='Lost'
    
    if st.button('Predict Status'):
        st.write('# The predicted Status is:',status)