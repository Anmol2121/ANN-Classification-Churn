import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import pickle
import tensorflow as tf
import keras
from PIL import Image
#from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from tensorflow.keras.models import load_model

import pickle

tf.compat.v1.reset_default_graph()

from tensorflow.keras.utils import get_custom_objects

custom_objects = get_custom_objects()  
model = load_model('Ann_model.h5', custom_objects=custom_objects)



with open('gender_label_encoder.pkl','rb') as f:
    gender_label = pickle.load(f)

with open('geo_onehot_encoder.pkl','rb') as f:
    geo_encoder = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler_data = pickle.load(f)

#model = load_model('Ann_model.h5')

# streamlit app

img = Image.open('ed3cfef53bb5f0136a75a36bb05ab38c.png')
st.set_page_config(page_title="CUSTOMER-CHUNK",page_icon=img)

def load_lotty_image(filepath: str):
    with open(filepath,'r') as f:
        return json.load(f)

selected_option_menu = option_menu(menu_title=None,
                                       options=['Home','Contact'],
                                       default_index=0,
                                       icons=['house','phone'],
                                       orientation='horizontal',
                                       styles={
                                           "container":{"padding":'important',"background-color":'#FEFAE0'},
                                           "nav-link-selected":{'background-color':"#B34180"}})
                                                        
if selected_option_menu == 'Home':                                  

    
    with st.container():
        st.write("---")
        left_column,right_column = st.columns(2)
        with left_column:
            st.header('Purpose of Chunking')
            st.write("""
                    - Explain that chunking customers helps in making more accurate predictions by focusing on specific groups rather than a broad audience. 
                    - Explain what chunk customer prediction is. It refers to the process of predicting customer behavior by dividing customers into segments or "chunks" based on specific attributes such as demographics, purchase history, or interaction patterns.
                    """)
            image_data = load_lotty_image(filepath="Animation - 1724570026258.json")

            st_lottie(image_data,speed=1,loop=True,quality='low',height=300)

            st.subheader("Customer Lifetime Value (CLV) Prediction")
            st.write("""
                    - Explain how predicting the lifetime value of different customer chunks can help in prioritizing resources.
                    """)
            st.subheader("Targeted Marketing Efforts")
            st.write("""
                    - By understanding the predicted CLV of different chunks, businesses can tailor their marketing efforts to match the potential value of each segment. For high CLV chunks, personalized and premium marketing strategies might be employed, while for lower CLV chunks, more generic and cost-effective campaigns can be used.
                    - Companies can prioritize retention efforts for chunks with higher CLV predictions. Since retaining a high-value customer is often more profitable than acquiring a new one, investing in loyalty programs, personalized communication, and exclusive offers for these customers can lead to better resource utilization.
                    """)
        with right_column:
            #if selected_option_menu == 'Home':

            st.write("#### CUSTOMER CHUNK PREDICTION")


            geography = st.selectbox('Geography',geo_encoder.categories_[0])
            gender = st.selectbox('Gender',gender_label.classes_)
            age = st.slider('Age',18,92)
            balance = st.number_input('Balance')
            credits_score = st.number_input('Credit Score')
            estimated_salary = st.number_input("Estimated Salary")
            tenure = st.slider("Tenure",0,10)
            number_product = st.slider("Number Of Product",0,4)
            has_cr_card = st.selectbox('Has Credit Card',[0,1])
            active_member = st.selectbox('Member',[0,1])


            def active_member_data(option):
                if option == 'Active':
                    return 1
                else:
                    return 0
                    
            input_data = pd.DataFrame({
                    'CreditScore':[credits_score],
                    "Gender":[gender_label.transform([gender])[0]],
                    "Age":[age],
                    'Tenure':[tenure],
                    'Balance':[balance],
                    'NumOfProducts':[number_product],
                    'HasCrCard':[has_cr_card],
                    'IsActiveMember':[active_member],
                    'EstimatedSalary':[estimated_salary], 
                })

            geo_data = geo_encoder.transform([[geography]]).toarray()
            geo_data_main = pd.DataFrame(geo_data,columns=geo_encoder.get_feature_names_out(['Geography']))

            final_data_frame = pd.concat([input_data.reset_index(drop=True),geo_data_main],axis=1)

            input_data_scaled = scaler_data.transform(final_data_frame)

            prediction = model.predict(input_data_scaled)
            prediction_prob = prediction[0][0]

            st.write(f"Churn Probablity: {prediction_prob:.2f}")

            if prediction_prob>0.5:
                st.success("The Customer is likely to churn")
            else:
                st.error("The Customer is not likely to churn")



if selected_option_menu == "Contact":
    st.header("Contact")

    with st.form(key='form1',clear_on_submit=True,border=True):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        mail = st.text_input("Mail")
        dob = st.text_area("Message")
        st.form_submit_button('Submit')


