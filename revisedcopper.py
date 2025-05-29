import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
import base64
from PIL import Image


#load the dataset
df=pd.read_csv(r"C:\My Setups\Industrial_copper_modelling\Copper_Set.CSV",low_memory=False)



#streamlit
st.set_page_config(page_title="COPPER MODELING",layout="wide")

st.markdown("<h1 style='text-align: center; color: #3E2723;'>INDUSTRIAL COPPER MODELING</h1>", unsafe_allow_html=True)

options=option_menu("Predictions Based on Machine Learning",["Product Status","Selling Price"],icons=["cash-coin", "award-fill"],
                    orientation="horizontal")


#load models
with open('Regression.pkl', 'rb') as file:
    best_model=pickle.load(file)
with open('scaler_1.pkl', 'rb') as file:
    scaler=pickle.load(file)
with open('item.pkl', 'rb') as file:
    ohe=pickle.load(file)
with open('status.pkl', 'rb') as file:
    ohe2=pickle.load(file)
with open("classifier_model.pkl", 'rb') as file_1:
     b_m_c=pickle.load(file_1)
with open('classifier_scaler.pkl', 'rb') as file:
    scaler_c=pickle.load(file)
with open('classifier_item.pkl', 'rb') as file:
    ohe_c_t=pickle.load(file)



#background image
def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
background_image_path = r'C:\My Setups\Industrial_copper_modelling\.venv\co3.jpg'
base64_image = get_base64_of_bin_file(background_image_path)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    ;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



#Product Status Prediction
if options == "Product Status":
    col1, col2= st.columns(2)

    with col1:
        item_type = st.selectbox('Item Type', df['item type'].unique())
        application = st.selectbox('Application', df['application'].unique())
        country = st.selectbox('Country Number', df['country'].unique())
        quantity = st.text_input("Enter the Quantity in Logs")

        if quantity:
            try:
                quantity = float(quantity)
            except ValueError:
                st.error("Quantity must be a valid number")
                quantity = None

        st.write("Minimum Value=5.983 ,Maximum Value=7.391")
        selling_price = st.text_input("Enter the selling_price in Logs")

        if selling_price:
            try:
                selling_price = float(selling_price)
            except ValueError:
                st.error("selling_price_log must be a valid number")
                selling_price = None

    with col2:
        st.write("Minimum Value=0.16, Maximum Value=2.66")
        thickness_log = st.text_input("Enter the Thickness")

        if thickness_log:
            try:
                thickness_log = float(thickness_log)
            except ValueError:
                st.error("Thickness must be a valid number")
                thickness_log = None

        st.write("Minimum Value=1, Maximum Value=2990")
        width = st.text_input("Enter the Width")

        if width:
            try:
                width = float(width)
            except ValueError:
                st.error("Width must be a valid number")
                width = None

        st.write("Minimum Value=12458, Maximum Value=30408185")
        customer = st.text_input("Enter the Customer ID")

        if customer:
            try:
                customer = float(customer)
            except ValueError:
                st.error("Customer ID must be a valid number")
                customer = None

        st.write("Minimum Value=611728, Maximum Value=1722207579")
        product_ref = st.text_input("Enter the Product Reference")

        if product_ref:
            try:
                product_ref = float(product_ref)
            except ValueError:
                st.error("Product Reference must be a valid number")
                product_ref = None

    if st.button("Predict Product Status"):
        if None in [item_type, application, country, quantity, thickness_log, width, customer, product_ref]:
            st.error("Please fill in all the required fields with valid values.")
        else:
            new_sample = np.array([[quantity, np.log(6.867162), application, thickness_log, width, country, customer, product_ref, item_type]])
            
            new_sample_ohe = ohe_c_t.transform(new_sample[:, [8]])
            new_sample_features = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
            new_sample_scaled = scaler_c.transform(new_sample_features)
            
            new_pred = b_m_c.predict(new_sample_scaled)
            status_label = 'Won' if new_pred[0] == 1 else 'Lost'
            st.write(f"The status is: {status_label}")

#Selling Price Prediction
elif options == "Selling Price":
    col1,col2=st.columns(2)
    
    with col1:
        status = st.selectbox('Status', df['status'].unique())
        item_type = st.selectbox('Item Type', df['item type'].unique())
        application = st.selectbox('Application', df['application'].unique())
        country = st.selectbox('Country Number', df['country'].unique())
        st.write("Minimum value=0.00001,Maximum value=6.807")
        quantity=st.text_input("Enter the Quantity in Logs")

        if quantity:
            try:
                quantity = float(quantity)
            except ValueError:
                st.error("Quantity must be a valid number")
                quantity = None

    with col2:
        st.write("Minimum Value=0.16, Maximum Value=2.66")
        thickness_log = st.text_input("Enter the Thickness")
        
        if thickness_log:
            try:
                thickness_log = float(thickness_log)
            except ValueError:
                st.error("Thickness must be a valid number")
                thickness_log = None
        
        st.write("Minimum Value=1, Maximum Value=2990")
        width = st.text_input("Enter the Width")
        
        if width:
            try:
                width = float(width)
            except ValueError:
                st.error("Width must be a valid number")
                width = None

        st.write("Minimum Value=12458, Maximum Value=30408185")
        customer = st.text_input("Enter the Customer ID")
        
        if customer:
            try:
                customer = float(customer)
            except ValueError:
                st.error("Customer ID must be a valid number")
                customer = None
        
        st.write("Minimum Value=611728, Maximum Value=1722207579")
        product_ref = st.text_input("Enter the Product Reference")
        
        if product_ref:
            try:
                product_ref = float(product_ref)
            except ValueError:
                st.error("Product Reference must be a valid number")
                product_ref = None

    with col1:
        predict_button_1 = st.button("Predict Selling Price")
    
    if predict_button_1:
        if None in [status, item_type, application, country, quantity, thickness_log, width, customer, product_ref]:
            st.error("Please fill in all the required fields with valid values.")
        else:
            input_data = pd.DataFrame({
                'status': [status],
                'item type': [item_type],
                'application': [application],
                'country': [country],
                'quantity': [quantity],
                'thickness_log': [thickness_log],
                'width': [width],
                'customer': [customer],
                'product_ref': [product_ref]
            })

            new_sample_ohe_item_type = ohe.transform(input_data[['item type']])
            new_sample_ohe_status = ohe2.transform(input_data[['status']])
            
            new_sample_encoded = np.concatenate((
                input_data[['quantity', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']].values,
                new_sample_ohe_item_type,
                new_sample_ohe_status
            ), axis=1)
            
            new_sample_scaled = scaler.transform(new_sample_encoded)
            
            new_pred = best_model.predict(new_sample_scaled)
            
            st.write(f"Predicted Selling Price Log: {new_pred[0]:.2f}")



# #Product Status Prediction
# elif options == "Product Status":
#     col1, col2= st.columns(2)

#     with col1:
#         item_type = st.selectbox('Item Type', df['item type'].unique())
#         application = st.selectbox('Application', df['application'].unique())
#         country = st.selectbox('Country Number', df['country'].unique())
#         quantity = st.text_input("Enter the Quantity in Logs")

#         if quantity:
#             try:
#                 quantity = float(quantity)
#             except ValueError:
#                 st.error("Quantity must be a valid number")
#                 quantity = None

#         st.write("Minimum Value=5.983 ,Maximum Value=7.391")
#         selling_price = st.text_input("Enter the selling_price in Logs")

#         if selling_price:
#             try:
#                 selling_price = float(selling_price)
#             except ValueError:
#                 st.error("selling_price_log must be a valid number")
#                 selling_price = None

#     with col2:
#         st.write("Minimum Value=0.16, Maximum Value=2.66")
#         thickness_log = st.text_input("Enter the Thickness")

#         if thickness_log:
#             try:
#                 thickness_log = float(thickness_log)
#             except ValueError:
#                 st.error("Thickness must be a valid number")
#                 thickness_log = None

#         st.write("Minimum Value=1, Maximum Value=2990")
#         width = st.text_input("Enter the Width")

#         if width:
#             try:
#                 width = float(width)
#             except ValueError:
#                 st.error("Width must be a valid number")
#                 width = None

#         st.write("Minimum Value=12458, Maximum Value=30408185")
#         customer = st.text_input("Enter the Customer ID")

#         if customer:
#             try:
#                 customer = float(customer)
#             except ValueError:
#                 st.error("Customer ID must be a valid number")
#                 customer = None

#         st.write("Minimum Value=611728, Maximum Value=1722207579")
#         product_ref = st.text_input("Enter the Product Reference")

#         if product_ref:
#             try:
#                 product_ref = float(product_ref)
#             except ValueError:
#                 st.error("Product Reference must be a valid number")
#                 product_ref = None

#     if st.button("Predict Product Status"):
#         if None in [item_type, application, country, quantity, thickness_log, width, customer, product_ref]:
#             st.error("Please fill in all the required fields with valid values.")
#         else:
#             new_sample = np.array([[quantity, np.log(6.867162), application, thickness_log, width, country, customer, product_ref, item_type]])
            
#             new_sample_ohe = ohe_c_t.transform(new_sample[:, [8]])
#             new_sample_features = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
#             new_sample_scaled = scaler_c.transform(new_sample_features)
            
#             new_pred = b_m_c.predict(new_sample_scaled)
#             status_label = 'Won' if new_pred[0] == 1 else 'Lost'
#             st.write(f"The status is: {status_label}")
