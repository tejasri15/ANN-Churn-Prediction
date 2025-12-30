import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle 
import streamlit as st


model = tf.keras.models.load_model('regression.keras')

#load the encoder and scaler
with open('ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title("Estimated Salary Prediction ")

##user input
geography = st.selectbox('Geography',ohe_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,60)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Products',1,20)
has_credit_card = st.selectbox('has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Memeber ',[0,1])

input_data = {
    'CreditScore':[credit_score],
'Gender':[label_encoder_gender.transform([gender])[0]]
,'Age':[age]
,'Tenure':[tenure]
,'Balance':[balance]
,'NumOfProducts':[num_of_products]
,'HasCrCard':[has_credit_card]
,'IsActiveMember':[is_active_member]
}



input_data = pd.DataFrame(input_data)

#one hot encoded geography
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns = ohe_geo.get_feature_names_out(['Geography']))
geo_encoded_df



#concate the one hot encoded
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scaling the input data
input_data_scaled  = scaler.transform(input_data)


#prediction
prediction_df = model.predict(input_data_scaled)
prediction_proba = prediction_df[0][0]
st.write(prediction_proba)