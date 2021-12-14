import sklearn
import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('laptop_price_prediction_model.pickle','rb'))
df = pickle.load(open('columns.pickle','rb'))

st.title("Laptop Predictor")

# brand

company = st.selectbox('Brand',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
# type of laptop


# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

#OS 
os = st.selectbox('OS',df['OpSys'].unique())

#memory
memory = st.selectbox('Memory', df['Memory'].unique())

if st.button('Predict Price'):
    #query
    query = np.array([company,type,ram,memory,os])
    query = query.reshape(1,5)
    # st.title(str(int(np.exp(pipe.predict(query)[0]))))
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))