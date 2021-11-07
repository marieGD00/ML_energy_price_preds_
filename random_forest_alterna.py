#----------------Imports-----------------------------#
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
import math
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_linear_regression

#---------------Read in the data set-----------------#
path = "date_sorted.csv"
original_dataframe = pd.read_csv(path, sep=",",dtype=None)
print("DATASET:",original_dataframe)

#---------Functions to choose which type of data you wish to feed into Random Forest----------#
def month_season(month):
    global season
    if month > 0:
        if month <= 4:
            season = 'winter'
        elif month <= 8 and month > 4:
            season = 'summer'
        else:
            season = 'autumn'
    return season

def time_daytime(hour):
    global time_of_day
    if hour > -1:
        if hour <= 6:
             time_of_day = '00-6am'
        elif hour <= 12 and hour > 6:
            time_of_day = '6am-12pm'
        elif hour > 12 and hour <= 18:
            time_of_day = '12pm-18h'
        else:
            time_of_day = '18h-00'
    return time_of_day

def correct_dataframe(string,frame):
    if string == 'Winter':
        frame = frame.loc[(frame['Mois'] <= 4)]
        print(frame)
        return frame
    elif string == 'Summer':
        frame = frame.loc[ (frame['Mois'] > 4) & (frame['Mois'] <= 8)]
        print(frame)
        return frame
    elif string == 'Autumn':
        frame = frame.loc[ (frame['Mois'] > 8) & (frame['Mois'] <= 12)]
        print(frame)
        return frame
    else:
        return frame
#--------------Choose which type of data to us--------------#
'''
options available:
1. All
2. Winter
3. Summer
4. Autumn
'''
choose_data = 'Autumn'
dataframe = correct_dataframe(choose_data,original_dataframe)
#----------Choose which variable to predict---------#
'''
options available:
1. Zonal.Price
2. residus
'''
predict_variable = "residus"
if predict_variable == "Zonal.Price":
    labels = dataframe['Zonal.Price']
elif predict_variable == "residus":
    labels = dataframe['residus']

#-----------------Drop variables----------------------#

dataframe = dataframe.drop('Forecasted.Zonal.Load',axis = 1) # remove Zonal
dataframe = dataframe.drop('Minute',axis=1)
dataframe = dataframe.drop('Posan',axis = 1)
dataframe = dataframe.drop('Time',axis = 1)
dataframe = dataframe.drop('TypeJour',axis = 1)

#----------------print list of columns in dataframe-------#
dataframe_list = list(dataframe.columns)
#print("list dataframe 2:",dataframe_list)

#-------------Add Prix decalees---------------------#

dataframe['prix decalees'] = dataframe['Zonal.Price'].shift(1)
index = dataframe.columns.get_loc('prix decalees')
dataframe.iloc[0,index] = 43.17
#print(dataframe)

#---------------Prepare time series------------------#
dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe.set_index('date',inplace=True)

#------------calculate linear regression--------------#
x = np.array(dataframe['Forecasted.Total.Load']).reshape((-1,1))
y = np.array(dataframe['Zonal.Price']).reshape((-1,1))
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)

print('coefficient of determination:',r_sq)
print('intercept:',model.intercept_[0])
print('slope:',model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
dataframe["lineareg"] = y_pred
dataframe["residus"] = dataframe["Zonal.Price"] - dataframe["lineareg"]

#------------------Plot linear regression results-----------------#

plt.scatter(dataframe['Forecasted.Total.Load'],dataframe['Zonal.Price'] )
x_plot = dataframe['Forecasted.Total.Load'].tolist()
y_plot = model.coef_[0]*x + model.intercept_[0]
plt.plot(x_plot,y_plot,color='orange')
plt.xlabel("Forecasted.Total.Load")
plt.ylabel("Zonal.Price")
plt.show()


#-----------Drop zonal.Price,linear regression, and residus----------#
dataframe = dataframe.drop('Zonal.Price',axis = 1)
dataframe = dataframe.drop('lineareg',axis=1)
dataframe = dataframe.drop('residus',axis=1)


#----------------Split dataframe into test and train----------------------------#
n_rows = len(dataframe)
train_dataframe = dataframe[:int(n_rows*0.95)]
test_dataframe = dataframe[int(n_rows*0.95):]
train_labels = labels[:int(n_rows*0.95)]
test_labels = labels[int(n_rows*0.95):]

print('Training dataframe Shape:', train_dataframe.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing dataframe Shape',test_dataframe.shape)
print('Testing labels Shape',test_labels.shape)

#------------Plot train and test dataframe----------------#
#plt.plot(train_dataframe)
#plt.plot(test_dataframe)
#plt.title("train and test plot")

#---------------Train model--------------------#
rf = RandomForestRegressor(n_estimators=500,random_state=42,oob_score = True)
rf.fit(train_dataframe,train_labels)
predictions = rf.predict(test_dataframe)

#------------------------Results-------------------------------#
print("BEFORE LOOP:")
print('Mean Absolute Error',metrics.mean_absolute_error(test_labels,predictions))
print('Mean Squared Error:' ,metrics.mean_squared_error(test_labels,predictions))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(test_labels,predictions)))

#-----------------------Plot Results-----------------#
#print("test labels")
#print("type test labels",type(test_labels) )

plt.plot(test_labels.index,test_labels.values,label='true values')
plt.plot(test_labels.index,predictions,label='predicted values')
plt.xlabel('Date')
plt.ylabel(predict_variable)
plt.title(choose_data)
plt.legend()
plt.show()

#----------------Print Variable Importances--------------------------#
importances = list(rf.feature_importances_)
feature_importances = [(dataframe,round(importance,2)) for dataframe,importance in zip(dataframe_list,importances)]
feature_importances = sorted(feature_importances,key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#---------------Verification print statements-----------#
testdataframe_list = list(test_dataframe.columns)
index_pd = testdataframe_list.index('prix decalees')
print("test datafram columns:",testdataframe_list)
print("test dataframe")
print(test_dataframe)
print(test_dataframe.iloc[0,8])
print("index_pd",index)
print("number of columns in test",len(test_dataframe.columns))

#---------------Loop-------------------------------#
totalsum_errors = 0
for i in range(len(test_dataframe)):
    n = len(test_dataframe)
    print(test_dataframe.shape)
    pred = rf.predict([test_dataframe.iloc[i]] )
    if i != (n-1) and i != 0:
        #print("i+1",i+1)
        test_dataframe.iloc[i+1,8] = pred 
    current_error = (pred - test_labels[i]) ** 2
    #print("Current error:",current_error[0])
    totalsum_errors += current_error

#---------------Calculate Loop RMSE--------------------#
final_RMSE =  np.sqrt(totalsum_errors/len(test_dataframe))
print("final RMSE:",final_RMSE)


#---------------plot the loop results------------#