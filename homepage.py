import streamlit as st

st.set_page_config(
    page_title="HomePage",
    page_icon="",
)


st.title("Stock Trend Predicter and Forecaster")
st.sidebar.success("Select a page above.")

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

st.subheader('Welcome to our page.Let me give you Some insights of our APP\n  ')
st.subheader('Here I have created a Stock Trend Prediction model It is based on Streamlit for the web page part and  and python for the coding part.At first we are scapping the data from yahoo finanace for the next step we are clearing the data from the data set we received. LSTM are sensitive to the scale of the data so we apply MinMax scaler.Next we split the data into training and testing set.After that the stacked LSTM model is to be Created.Then we are Training the last 100 day data to predict the value for 101th day .Similarly for the 102nd day we will consider data from 1st data to 101st data and so on.This app can Also forecast the Closing Price for next 30 days. Please feel free to go to project Section to try the app yourself. ')



