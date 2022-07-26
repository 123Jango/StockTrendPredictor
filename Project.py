
from ast import increment_lineno
from inspect import trace
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st
from datetime import date
import plotly.express as px
from matplotlib import pyplot as plt
import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names =["Sayan Chakraborty","KIIT"]
usernames = ["Scarab","Dolphin"]


file_path = Path(__file__).parent/ "hashed_PW.pkl"
with file_path.open('rb') as file:
    hashed_passwords= pickle.load(file)

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,"sales_dashboard","abcdef", cookie_expiry_days=30)    

names, authentication_status,username = authenticator.login("Login","main")

if authentication_status == False:
    st.error("Username or Password is incorrect")

if authentication_status == None:
    st.warning("Please enter your Username and Password")

if authentication_status:    

    start = '2017-07-21'
    end = '2022-07-18'

    st.title('Stock Trend Analyzer and Forecaster')

    user_input = st.text_input('Enter stock Ticker','AAPL')

    df = pdr.DataReader(user_input,'yahoo',start,end)

    st.subheader('Data for  last 5 days')
    st.write(df.tail())
    
    df1=df.reset_index()['Close']

    st.subheader('Plotting the Closing Pricce V/s Number of days : ')
    fig=plt.figure(figsize=(7,5))
    plt.plot(df1)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    st.plotly_chart(fig)
    
    
    #scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    #Spliting data set into train and test
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    ## convert an array of values into a dataset matrix
    import numpy
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


    ## reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)


    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model = load_model('stock.h5')

    import tensorflow as tf

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)


    ### Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error
    math.sqrt(mean_squared_error(y_train,train_predict))

    ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest,test_predict))

    ### Plotting 
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    st.subheader('Plotting Baseline and Predictions : ')
    fig2=plt.figure(figsize=(7,5))
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    st.plotly_chart(fig2)

    x_input=test_data[340:].reshape(1,-1)


    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    print(lst_output)

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    st.subheader('Plotting the future 30 Days Forecast Based on last 100 Days data')
    fig3=plt.figure(figsize=(7,5))
    plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    st.plotly_chart(fig3)
    authenticator.logout("Logout","main")






