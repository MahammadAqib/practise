import streamlit as st
import numpy as np
import pandas as pd
import pickle
#from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
         
This app predicts the **Palmer Penguin** species!
""")

st.sidebar.header('User Input Features')

# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# input_df = pd.read_csv(uploaded_file)

def user_input_features():
    island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill Length(mm)',32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth(mm)',13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider('Flipper Length(mm)',172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Body Mass(g)',2700.0,6300.0,4207.0)
    data = {'island':island,
            'sex':sex,
            'bill_length_mm':bill_length_mm,
            'bill_depth_mm':bill_depth_mm,
            'flipper_length_mm':flipper_length_mm,
            'body_mass_g':body_mass_g}
    features = pd.DataFrame(data,index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]

# Displays the user input features
st.subheader('User Input features')

st.write(df[:1])

load_clf  = pickle.load(open('penguin_clf.pkl','rb'))

# Apply model to make predictions
pred = load_clf.predict(df[:1])
pred_prob = load_clf.predict_proba(df[:1])

st.subheader('Prediction')
penguin_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguin_species[pred])

st.subheader('Prediction Probability')
st.write(pred_prob)
