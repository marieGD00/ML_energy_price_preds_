import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'data_A.csv'
dataset = pd.read_csv(path, sep=",",dtype=None)
print(dataset)


col_consoA = dataset['ConsoA']
print(col_consoA)

#
def movingaverage(values,window):
    weights = np.repeat(1.0,window)/window
    smas = np.convolve(values,weights,'valid')
    return smas #np array



print(movingaverage(col_consoA,29))



movingaverage_window29 = movingaverage(col_consoA,29)
movingaverage_window24 = movingaverage(col_consoA,24)
movingaverage_window50 = movingaverage(col_consoA,50)
movingaverage_window7 = movingaverage(col_consoA,7)


#-------------------calculate the MAPE--------------------#

#length of original is 8760
print("length:", len(movingaverage_window29) ) #8732

def adjust_list(list,window):
    reduce = window-1
    new_list = list[reduce:]
    return new_list

def calculate_RMSE(original,moving_averages,window):
    original_adjust = adjust_list(original,window)
    list_original_adjust = original_adjust.tolist()
    list_moving_averages = moving_averages.tolist()
    total_sum = 0

    for i in range(len(list_moving_averages)):
        curr = (list_moving_averages[i] - list_original_adjust[i]) ** 2
        total_sum += curr
    
    RMSE = math.sqrt(total_sum / len(list_moving_averages) )
    return RMSE

def calculate_MAPE(original,moving_averages,window):
    original_adjust = adjust_list(original,window)
    list_use_original = original_adjust.tolist()
    list_moving_averages = moving_averages.tolist()
    total_sum = 0
    
    for i in range(len(list_moving_averages)):
        curr_numerator = abs(list_use_original[i] - list_moving_averages[i])
        curr_denominator = list_use_original[i]
        curr_value = curr_numerator/ curr_denominator
        total_sum += curr_value
    
    
    MAPE = (1/len(moving_averages)) * total_sum * 100
    return MAPE

MAPE_29 = calculate_MAPE(col_consoA,movingaverage_window29,29)
RMSE_29 = calculate_RMSE(col_consoA,movingaverage_window29,29)
MAPE_24 = calculate_MAPE(col_consoA,movingaverage_window24,24)
RMSE_24 = calculate_RMSE(col_consoA,movingaverage_window24,24)
MAPE_50  = calculate_MAPE(col_consoA,movingaverage_window50,50)
RMSE_50 = calculate_RMSE(col_consoA,movingaverage_window50,50)
MAPE_7 = calculate_MAPE(col_consoA,movingaverage_window7,7)
RMSE_7 = calculate_RMSE(col_consoA,movingaverage_window7,7)

print("MAPE window 29",MAPE_29)
print("RMSE window 29", RMSE_29)

print("MAPE window 24",MAPE_24)
print("RMSE window 24", RMSE_24)

print("MAPE window 50",MAPE_50)
print("RMSE window 50", RMSE_50)

print("MAPE window 7",MAPE_7)
print("RMSE window 7", RMSE_7)

#------------plot the actual compared to real-----------#
#plot moving average with windo 24
consoA_adjustlist = adjust_list(col_consoA,24)
plt.plot(consoA_adjustlist,movingaverage_window24)
#plt.plot(col_consoA,)
plt.title("window of 24")
plt.show()

#-----------plot the different MAPE and RMSE given different windows-----#


#plot the moving averages

   