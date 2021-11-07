import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyramid as pm
from pmdarima.arima import auto_arima
from pandas import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


#read
path = 'date_sorted.csv'
dataset = pd.read_csv(path)
print(dataset)
print("dataset type",type(dataset))
dataset_list = list(dataset.columns)
print("dataset list",dataset_list)

#choose where to train
dataframe_arima = dataset[dataset['Month'] >= 6 and dataset['Month' <= 7]]
print(dataframe_arima)

#drop variables
dataset = dataset.drop('Month',axis=1)
dataset = dataset.drop('Time',axis=1)
dataset = dataset.drop('Day',axis=1)
dataset = dataset.drop('Hour',axis=1)
dataset = dataset.drop('Season',axis=1)
dataset = dataset.drop('WeekDay',axis=1)
dataset = dataset.drop('ConsoAMeanDay',axis=1)
dataset = dataset.drop('ConsoAMeanHour',axis=1)
dataset = dataset.drop('ConsoAMeanMonth',axis=1)
print(dataset)

#prepare time series
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date',inplace=True)
dataset.plot()

#split
#get the index of a certain date
split_date = datetime.date(209,4,3)


train = dataset[int(0.8*(len(dataset))):int(0.999*(len(dataset)))]
test = dataset[int(0.997*(len(dataset))):]
plt.plot(train)
plt.plot(test)
plt.title("train and test plot")
#plt.show()

#build the model
arima_model = model = auto_arima(train,start_p=0,d=1,start_q=0,max_p=5,max_d=5,max_q=5,start_P =0,
                                D=1,start_Q=0,max_P=5,max_Q=5,m=12,seasonal=True,error_action='warn',
                                trace=True,supress_warnings=True,stepwise=True,random_state=20,n_fits=50)
                                

arima_model.summary()

#predictions
prediction = pd.DataFrame(arima_model.predict(n_periods=88),index=test.index)
prediction.columns = ['predicted_consoA']
print("Prediction Dataframe")
print(prediction)

plt.figure(figsize=(8,5))
plt.plot(train,label="Training",color='green')
plt.plot(test,label="Test",color='orange')
plt.plot(prediction,label="Predicted")
plt.title("prediction vs Actual")
plt.legend()



from sklearn.metrics import r2_score
test['predicted_consoA'] = prediction
rmse = r2_score(test['ConsoA'],test['predicted_consoA'])
print("RMSE:",rmse)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test['ConsoA'],test['predicted_consoA']))
print("second RMSE:",rms)

print("Test Dataframe")
print(test)
list_test = test['ConsoA'].tolist()
list_consoA = test['predicted_consoA'].tolist()
'''
plt.plot(list_test,color='r')
plt.plot(list_consoA,color='b')
plt.title("predicted v actual")
plt.show()
'''
total_sum = 0
for i in range(len(list_test)):
    curr_squared = (list_test[i]-list_consoA[i])**2
    total_sum += curr_squared

rmse_finalversion = sqrt(total_sum/len(list_test))
print("rmse final",rmse_finalversion)

from sklearn.metrics import mean_absolute_error
print("mean absolute error:",mean_absolute_error(test['ConsoA'],test['predicted_consoA']))
plt.show()