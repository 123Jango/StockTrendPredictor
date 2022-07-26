

# Stock Trend Predictor and Forecaster 

Here I have created a Stock Trend Prediction model It is based on Streamlit for the web page part and  and python for the coding part.At first we are scapping the data from yahoo finanace for the next step we are clearing the data from the data set we received. LSTM are sensitive to the scale of the data so we apply MinMax scaler.Next we split the data into training and testing set.After that the stacked LSTM model is to be Created.Then we are Training the last 100 day data to predict the value for 101th day .Similarly for the 102nd day we will consider data from 1st data to 101st data and so on.This app can Also forecast the Closing Price for next 30 days. Please feel free to go to project Section to try the app yourself. 




## Deployment

To deploy this project run

```bash
  streamlit run homepage.py
```


## Tech Stack

**packages:** keras,streamlit,numpy,pandas,datareader,matplotlib,pickle


**softwares:** 
Web browser,Virtual studio Code,Anaconda PowerShell Prompt
## Screenshots
### Initially running the app
![Screenshot 2022-07-26 221540](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/Screenshot%202022-07-26%20221540.png?raw=true)
### Home
![homepage](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/homepage.png?raw=true)
### login
![login page](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/login%20page.png?raw=true)
### Project Page
![projectpage1](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/projectpage1.png?raw=true)
### Graph 1
![Graph 1](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/Graph%201.png?raw=true)
### Graph 2
![Graph2](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/Graph2.png?raw=true)
### Graph 3
![graph3](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/graph3.png?raw=true)
### logout
![logout](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/logout.png?raw=true)
### Default Contact us
![defaultContactUs](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/defaultContactUs.png?raw=true)
### Data sent Confirmation
![senderConfirmation](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/senderConfirmation.png?raw=true)
### Received Information
![ReceivedInformation](https://github.com/123Jango/StockTrendPredictor/blob/main/project%20ss/ReceivedInformation.png?raw=true)



