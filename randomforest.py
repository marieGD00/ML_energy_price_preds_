
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
path = "newform_two.csv"
dataset = pd.read_csv(path, sep=",",dtype=None)
print("DATASET:",dataset)



#-----------------Data subsets-----------------------#
#choose which data to feed
# 0-not fed into the random forest
# 1-fed into the random forest
prix_decalees_bool = 0
ForecastedTotalLoad_bool = 0
Jour_Ferie_bool = 0
JourSemaine_bool = 0
Posan_bool = 0
Instant_bool = 0
Minute_bool = 0
Heure_bool = 0
Jour_bool = 0
Mois_bool = 0
Annee_bool = 0

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

#---------------Dynamically modify types of data given into the Random Forest---------------#
season = 'All seasons'
month = 0
hour = -1
time_of_day = 'Not specified'

#adjust dataframe according to the seasons
if month == 0 and hour == -1:
    dataframe = dataset
    print(dataframe)
else:
    if month_season(month) == 'winter':
        dataframe = dataset.loc[(dataset['Mois'] <= 4)]
        print(dataframe)

    elif month_season(month) == 'summer':
        dataframe = dataset.loc[ (dataset['Mois'] > 4) & (dataset['Mois'] <= 8)]
        print(dataframe)
    elif month_season(month) == 'autumn':
        dataframe = dataset.loc[ (dataset['Mois'] > 8) & (dataset['Mois'] <= 12)]
        print(dataframe)

    #adjust dataframe according to the time of day
    elif time_daytime(hour) == '00-6am':
        dataframe = dataset.loc[(dataset['Heure'] <= 6)]
        print(dataframe)
    elif time_daytime(hour) == '6am-12pm':
        dataframe = dataset.loc[ (dataset['Heure'] > 6) & (dataset['Heure'] <= 12)]
        print(dataframe)
    elif time_daytime(hour) == '12pm-18h':
        dataframe = dataset.loc[ (dataset['Heure'] > 12) & (dataset['Heure'] <= 18)]
        print(dataframe)
    elif time_daytime(hour) == '18h-00':
        dataframe = dataset.loc[ (dataset['Heure'] > 18) & (dataset['Heure'] <= 24)]
        print(dataframe)

#-----------------Linear regression--------------------------#
'''
x = np.array(dataset['Forecasted.Total.Load']).reshape((-1,1))
y = np.array(dataset['Zonal.Price']).reshape((-1,1))

#intercept,slope,corr_coeff = plot_linear_regression(x,y)
#plt.show()

#create the linear regression model
model = LinearRegression().fit(x,y) # should I normalise? or decide to calculate the fit intercept?
r_sq = model.score(x,y)


print('coefficient of determination:',r_sq)
print('intercept:',model.intercept_[0])
print('slope:',model.coef_)

#plt.scatter(x,y)
#plt.plot(x,model.intercept_[0] + model.coef_[0]*x,color='g')
#plt.show()

#predict results
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
#dataset["linear regression prix"] = y_pred


#write in y_pred into the csv file
with open('use.csv','w',newline='') as file:
    #csv_reader = csv.reader
    csv_writer = csv.writer(file)
    csv_writer.writerow("linear regression prix")
    for row in  y_pred:
        csv_writer.writerow(row)
'''
    
#------------------start preparing Random Forest-------------------#
labels = np.array(dataframe['zonal price - prix regression'])

#drop all of the useless columns
dataframe = dataframe.drop('zonal price - prix regression',axis = 1)
dataframe = dataframe.drop('Forecasted.Zonal.Load',axis = 1) # remove Zonal
dataframe = dataframe.drop('Id',axis = 1)
dataframe = dataframe.drop('Zonal.Price',axis = 1)
dataframe = dataframe.drop('Prix predicted by linear regression',axis=1)
dataframe = dataframe.drop('Minute',axis=1)
dataframe = dataframe.drop('TBM',axis=1)
dataframe = dataframe.drop('Posan',axis = 1)

dataframe_list = list(dataframe.columns)
print("list dataframe 2:",dataframe_list)
dataframe = np.array(dataframe)


#-------------------Split model not time series---------------------------#
'''
enter method = 0 for a normal test split which doesn't take into account the time series
enter method = 1 for a time series test split which makes sure future and past data are not mixed
'''
method = 1

if method == 0:
    train_dataframe,test_dataframe,train_labels, test_labels = train_test_split(dataframe,labels,test_size = 0.3,random_state = 42)
    print("test labels",test_labels)
    print("train labels",train_labels)
elif method == 1:
    train_dataframe = dataframe[:int(dataframe.shape[0]*0.7)]
    test_dataframe = dataframe[int(dataframe.shape[0]*0.7):]
    train_labels = labels[:int(dataframe.shape[0]*0.7)]
    test_labels = labels[int(dataframe.shape[0]*0.7):]


#-------------Various print statements---------------#
np.set_printoptions(suppress=True)
print("columns test df",test_dataframe[0,] )
print("columns train df",train_dataframe[0,])
print("test labels",test_labels)
print("len test labels",len(test_labels))
print('Training dataframe Shape:', train_dataframe.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing dataframe Shape',test_dataframe.shape)
print('Testing labels Shape',test_labels.shape)


#-----------------------TRAIN MODEL---------------------------------#


rf = RandomForestRegressor(n_estimators=500,random_state=42,oob_score = True)
rf.fit(train_dataframe,train_labels)
predictions = rf.predict(test_dataframe)

#more print statements
length = len(predictions)
print("predictions type", type(predictions))
print("test labels type",type(test_labels))
print("length predictions",len(predictions))
print("length test observations:",len(test_dataframe))
print("predictions shape:",predictions.shape)
print("test dataframe shape",test_dataframe)
print("list predictions:",predictions)
print("type predictions:",type(predictions))
print("list test",test_dataframe)
print("type test dataframe:",type(test_dataframe))
list_predictions = predictions.tolist()
print("type list predictions",type(list_predictions))
list_test_values = test_dataframe.tolist()
print("type test values:",type(list_test_values))
for i in range(len(predictions)):
    if i < 10:
        print("curr predictions:",list_predictions[i])
        print("curr test values:",list_test_values[i])


#------------------------Results-------------------------------#

print('Mean Absolute Error:',metrics.mean_absolute_error(test_labels,predictions))
print('Mean Squared Error:',metrics.mean_squared_error(test_labels,predictions))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(test_labels,predictions)))

#----------------Print Variable Importances--------------------------#
importances = list(rf.feature_importances_)
feature_importances = [(dataframe,round(importance,2)) for dataframe,importance in zip(dataframe_list,importances)]
feature_importances = sorted(feature_importances,key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#-----------------Plot Variable importances----------------------#
#this code produces a visual representation of the variable importances
'''
plt.style.use('fivethirtyeight')

x_values = list(range(len(importances)))
plt.bar(x_values,importances,orientation = 'vertical')

plt.xticks(x_values,dataframe_list,rotation='vertical')

#axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
#plt.tight_layout()
#plt.autoscale

plt.show()
'''

#----------------------------Random Forest Loop----------------------------#

print(test_dataframe)
list_errors = []
totalsum_errors = 0
#test_dataframe = test_dataframe['zonal price - prix regression'].reshape(1, -1)
for i in range(len(test_dataframe)):
    print("number:",i)
    n = len(test_dataframe)
    pred = rf.predict([test_dataframe[i]]) 
    if i != (n-1):
        test_dataframe[i+1,8] = pred 
    current_error = (pred - test_labels[i]) ** 2
    print("Current error:",current_error[0])
    list_errors.append(current_error[0])
    totalsum_errors += current_error
    

# calculate the RMSE
final_RMSE =  np.sqrt(totalsum_errors/len(test_dataframe))
print("final RMSE:",final_RMSE)


#-------------------Plot the actual values in relation to the predicted values----------#
'''
#dates of training values
months = dataframe[:,dataframe_list.index('Mois')]
days = dataframe[:,dataframe_list.index('Jour')]
years = dataframe[:,dataframe_list.index('Annee')]

#list and convert to date time object
dates = [str(int(Annee)) + '-' + str(int(Mois)) + '-' + str(int(Jour)) for Annee,Mois,Jour in zip(years,months,days)]
dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data ={'date': dates, 'Zonal.Price':labels}) # check you call it labels

#dates of predictions
months = test_dataframe[:,dataframe_list.index('Mois')]
days = test_dataframe[:,dataframe_list.index('Jour')]
years = test_dataframe[:,dataframe_list.index('Annee')]

test_dates = [str(int(Annee)) + '-' + str(int(Mois)) + '-' + str(int(Jour)) for Annee,Mois,Jour in zip(years,months,days)]
test_dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in test_dates]

#Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction':predictions})

#plot actual values
plt.plot(true_data['date'],true_data['Zonal.Price'],'b-',label = 'Zonal.Price')

#plot the predicted values
plt.plot(predictions_data['date'],predictions_data['prediction'],'ro',label = 'prediction')
plt.xticks(rotation = '60')
plt.legend()

#Graph labels
plt.xlabel('Date')
plt.ylabel('Energy price')
plt.title('Actual and Predicted Values')
plt.show()
'''







