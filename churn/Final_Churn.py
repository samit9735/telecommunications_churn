
import numpy as np
import pickle
import streamlit as st

#loading the model
loaded_model = pickle.load(open('D:\\project\\telecommunications\\model_xgb.pkl','rb'))
    
   
 #def main():
st.title('Telecommunications Churn')

account_length          = st.sidebar.number_input('Account length', min_value = 0)
str_voice_mail_plan     = st.sidebar.radio('Voice Mail Plan', ['Yes','No'])

if str_voice_mail_plan == 'Yes':
    voice_mail_plan     = 1
else:
    voice_mail_plan     = 0
    
#st.write(voice_mail_plan)

voice_mail_messages     = st.sidebar.number_input('Voice Mail Messages',min_value=0)
night_minutes           = st.sidebar.number_input('Night Minutes')
international_minutes   = st.sidebar.number_input('International Minutes')
customer_service_calls  = st.sidebar.number_input('Customer Service Calls', min_value=0)

str_international_plan      = st.sidebar.radio('International Plan', ['Yes','No'])

if str_international_plan == 'Yes':
    international_plan     = 1
else:
    international_plan     = 0

#st.write(international_plan)
    
day_calls               = st.sidebar.number_input('Day Calls', min_value=0)
evening_calls           = st.sidebar.number_input('Evening Calls', min_value=0)
night_calls             = st.sidebar.number_input('Night Calls', min_value=0)
international_calls     = st.sidebar.number_input('International Calls', min_value=0)
total_charge            = st.sidebar.number_input('Total Charge')





def churn_prediction(input_data):
    
    input_data_1 = np.asarray(input_data)
   
    input_data_1_reshaped = input_data_1.reshape(1,-1)
   
    
    #checking the prediction
    prediction_1 = loaded_model.predict(input_data_1_reshaped)
    prediction_proba =loaded_model.predict_proba(input_data_1_reshaped)
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    
    
    if(prediction_1[0] == 0):
        st.write('Congratulation this person is not going to Churn')
        st.balloons()
    else:
        st.write('Unfortunately this person is going to churn')
    
    
    
    
    
    

# code for prediction
churn_status = ''

#creating submit button
if st.button('Predict Churn Status'):
    churn_status= churn_prediction([account_length,voice_mail_plan,voice_mail_messages,
                                    night_minutes,international_minutes,customer_service_calls,international_plan,
                                    day_calls,evening_calls,night_calls,international_calls,total_charge])


st.success(churn_status) 


st.subheader(' Created By : PROJECT GROUP NO 2')






