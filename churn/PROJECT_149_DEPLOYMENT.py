import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st
import re

st.title('Customer Churn Prediction')
st.sidebar.header('Enter values for following Variables and in same order:')
st.sidebar.subheader("1. Account Length")
st.sidebar.subheader("2. Voice mail plan (0 or 1)")
st.sidebar.subheader("3. Voice mail messages")
st.sidebar.subheader("4. day mins")
st.sidebar.subheader("5. evening mins")
st.sidebar.subheader("6. night mins")
st.sidebar.subheader("7. international mins")
st.sidebar.subheader("8. customer service calls")
st.sidebar.subheader("9. international plan (0 or 1)")
st.sidebar.subheader("10. day calls")
st.sidebar.subheader("11. day charge")
st.sidebar.subheader("12. evening calls")
st.sidebar.subheader("13. evening charge")
st.sidebar.subheader("14. night calls")
st.sidebar.subheader("15. night charge")
st.sidebar.subheader("16. international calls ")
st.sidebar.subheader("17. international charge")
st.sidebar.subheader("18. total charge")


collect_numbers = lambda x : [float(i) for i in re.split(";", x) if i != ""]
numbers = st.text_input("Please Enter the Values [ Seperated by ' ; ' ]")
user_input=collect_numbers(numbers)
st.write(user_input)
features=pd.DataFrame([user_input])
if (len(user_input))!=18: 
    st.write("Warning : You have entered {} values,\n Please Enter only 18 values".format(len(user_input)))
else:
    
    if st.button("Process"): 
        # Load Data set
        df = pd.read_csv("D:\\project\\telecommunications\\telecommunications_churn.csv",delimiter=';')
        df1=df.copy()
        # change 'Class' data type to Categoric
        df1['churn']=df1['churn'].astype('category')   

        #Imputing Outliers using Winsorizing
        df2=df1.copy()
        upper_limit_1 = df2['account_length'].quantile(0.99)
        lower_limit_1= df2['account_length'].quantile(0.01)
        df2['account_length'] = np.where(df2['account_length'] >= upper_limit_1,
        upper_limit_1,
        np.where(df2['account_length'] <= lower_limit_1,
        lower_limit_1,
        df2['account_length']))

        upper_limit_2 = df2['voice_mail_messages'].quantile(0.99)
        lower_limit_2= df2['voice_mail_messages'].quantile(0.01)

        df2['voice_mail_messages'] = np.where(df2['voice_mail_messages'] >= upper_limit_2,
        upper_limit_2,
        np.where(df2['voice_mail_messages'] <= lower_limit_2,
        lower_limit_2,
        df2['voice_mail_messages']))


        upper_limit_3 = df2['day_mins'].quantile(0.99)
        lower_limit_3= df2['day_mins'].quantile(0.01)
        df2['day_mins'] = np.where(df2['day_mins'] >= upper_limit_3,
        upper_limit_3,
        np.where(df2['day_mins'] <= lower_limit_3,
        lower_limit_3,
        df2['day_mins']))

        upper_limit_4 = df2['evening_mins'].quantile(0.99)
        lower_limit_4= df2['evening_mins'].quantile(0.01)
        df2['evening_mins'] = np.where(df2['evening_mins'] >= upper_limit_4,
        upper_limit_4,
        np.where(df2['evening_mins'] <= lower_limit_4,
        lower_limit_4,
        df2['evening_mins']))

        upper_limit_5 = df2['night_mins'].quantile(0.99)
        lower_limit_5= df2['night_mins'].quantile(0.01)
        df2['night_mins'] = np.where(df2['night_mins'] >= upper_limit_5,
        upper_limit_5,
        np.where(df2['night_mins'] <= lower_limit_5,
        lower_limit_5,
        df2['night_mins']))


        upper_limit_6 = df2['international_mins'].quantile(0.99)
        lower_limit_6= df2['international_mins'].quantile(0.01)
        df2['international_mins'] = np.where(df2['international_mins'] >= upper_limit_6,
        upper_limit_6,
        np.where(df2['international_mins'] <= lower_limit_6,
        lower_limit_6,
        df2['international_mins']))

        upper_limit_8 = df2['customer_service_calls'].quantile(0.99)
        lower_limit_8= df2['customer_service_calls'].quantile(0.01)
        df2['customer_service_calls'] = np.where(df2['customer_service_calls'] >= upper_limit_8,
        upper_limit_8,
        np.where(df2['customer_service_calls'] <= lower_limit_8,
        lower_limit_8,
        df2['customer_service_calls']))

        upper_limit_9 = df2['day_calls'].quantile(0.99)
        lower_limit_9= df2['day_calls'].quantile(0.01)
        df2['day_calls'] = np.where(df2['day_calls'] >= upper_limit_9,
        upper_limit_9,
        np.where(df2['day_calls'] <= lower_limit_9,
        lower_limit_9,
        df2['day_calls']))

        upper_limit_10 = df2['day_charge'].quantile(0.99)
        lower_limit_10= df2['day_charge'].quantile(0.01)
        df2['day_charge'] = np.where(df2['day_charge'] >= upper_limit_10,
        upper_limit_10,
        np.where(df2['day_charge'] <= lower_limit_10,
        lower_limit_10,
        df2['day_charge']))

        upper_limit_11 = df2['evening_calls'].quantile(0.99)
        lower_limit_11= df2['evening_calls'].quantile(0.01)
        df2['evening_calls'] = np.where(df2['evening_calls'] >= upper_limit_11,
        upper_limit_11,
        np.where(df2['evening_calls'] <= lower_limit_11,
        lower_limit_11,
        df2['evening_calls']))

        upper_limit_12 = df2['evening_charge'].quantile(0.99)
        lower_limit_12= df2['evening_charge'].quantile(0.01)
        df2['evening_charge'] = np.where(df2['evening_charge'] >= upper_limit_12,
        upper_limit_12,
        np.where(df2['evening_charge'] <= lower_limit_12,
        lower_limit_12,
        df2['evening_charge']))

        upper_limit_13 = df2['night_calls'].quantile(0.99)
        lower_limit_13= df2['night_calls'].quantile(0.01)
        df2['night_calls'] = np.where(df2['night_calls'] >= upper_limit_13,
        upper_limit_13,
        np.where(df2['night_calls'] <= lower_limit_13,
        lower_limit_13,
        df2['night_calls']))

        upper_limit_14 = df2['night_charge'].quantile(0.99)
        lower_limit_14= df2['night_charge'].quantile(0.01)
        df2['night_charge'] = np.where(df2['night_charge'] >= upper_limit_14,
        upper_limit_14,
        np.where(df2['night_charge'] <= lower_limit_14,
        lower_limit_14,
        df2['night_charge']))

        upper_limit_15 = df2['international_calls'].quantile(0.99)
        lower_limit_15= df2['international_calls'].quantile(0.01)
        df2['international_calls'] = np.where(df2['international_calls'] >= upper_limit_15,
        upper_limit_15,
        np.where(df2['international_calls'] <= lower_limit_15,
        lower_limit_15,
        df2['international_calls']))
        upper_limit_16 = df2['international_charge'].quantile(0.99)
        lower_limit_16= df2['international_charge'].quantile(0.01)
        df2['international_charge'] = np.where(df2['international_charge'] >= upper_limit_16,
        upper_limit_16,
        np.where(df2['international_charge'] <= lower_limit_16,
        lower_limit_16,
        df2['international_charge']))
        upper_limit_17 = df2['total_charge'].quantile(0.99)
        lower_limit_17= df2['total_charge'].quantile(0.01)
        df2['total_charge'] = np.where(df2['total_charge'] >= upper_limit_17,
        upper_limit_17,
        np.where(df2['total_charge'] <= lower_limit_17,
        lower_limit_17,
        df2['total_charge']))

        #Balancing the Data Set
        churn_data = df2[df2["churn"] == 1]
        no_churn_data  = df2[df2["churn"] == 0]
        churn_upsample = resample(churn_data,
             replace=True,
             n_samples=len(no_churn_data),
             random_state=42)
        data_upsampled = pd.concat([no_churn_data, churn_upsample])

        # Number of Churn and Non-churn obeseravtions are equal
        df_final=data_upsampled.copy()
        # Upsampled data set has 5700 rows

        ##Model Building
        ##SVM

        array = df_final.values
        X8 = array[:,0:18]
        Y8 = array[:,18]
        #best value of C=15 and gamma=50
        clf = SVC(C= 15, gamma = 50)
        clf.fit(X8 ,Y8)

        y_pred = clf.predict(features)

        st.subheader('Predicted Result:')
        if y_pred==1:
            st.write('The Customer will Churn')
        else:
            st.write('The Customer will Not-Churn')
            
        
