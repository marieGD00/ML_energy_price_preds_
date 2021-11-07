import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
import argparse
from xgboost import XGBRegressor
from copy import deepcopy
import datetime as dt

# %% parse args
def parse_args():
    """Argument parsing, must give one only input file argument. Returns the args object"""
    parser = argparse.ArgumentParser()

    parser.add_argument('ifile', help = 'Input file')

    args = parser.parse_args()
    return args

# %% data processing

def parse_data(ifile, sep=';', char2remove = ['\n']):
    """Parses input file line by line, and return a 2-dimensional list containing the stored information"""
    with open(ifile, 'r') as f:
        vect = []
        line = f.readline()
        while line != '':
            for c in char2remove:
                line = line.replace(c, '')
            vect.append(line.split(sep))
            line = f.readline()
        return vect

def convert_weekday(wd):
    if wd == 'Monday':
        return 1
    elif wd == 'Tuesday':
        return 2
    elif wd == 'Wednesday':
        return 3
    elif wd == 'Thursday':
        return 4
    elif wd == 'Friday':
        return 5
    elif wd == 'Saturday':
        return 6
    elif wd == 'Sunday':
        return 7
    else : 
        raise ValueError('Unrecognized week-day')
def convert_season(s):
    if s == 'Winter':
        return 1
    elif s == 'Spring':
        return 2
    elif s == 'Summer':
        return 3
    elif s == 'Fall':
        return 4
    else:
        raise ValueError('Unrecognized season : '+s)


def convert_data(data):
    """Data preprocessing : removes unusable features, convert categorial ones... Returns (float) np.array"""
    data = data[1:] #header
    new_data = []
    for i in range(len(data)):
        v = []
        for j in range(len(data[i])):
            if j <= 5 and j != 1: # id, year, month, day, hour
                v.append(int(data[i][j]))
            elif j == 6: # season
                v.append(convert_season(data[i][j]))
            elif j == 7: # weekday
                v.append(convert_weekday(data[i][j]))
            elif j >= 8: # prod, means, shift
                v.append(float(data[i][j]))
            # nothing done for date --> ignored
        new_data.append(v)
    return np.array(new_data)

def split_xy(data, index):
    """Cuts the label (index-th column) out of the data."""
    x = np.delete(data, index, axis = 1)
    y = data[:,index]
    return (x, y)

# %% Correlations

def versus(data, i, j, ask4names = False):
    """Plot correlations : one variable versus another"""
    plt.scatter(data[:,i],data[:,j])
    if ask4names:
        xlbl = input('Please enter xlabel : ')
        ylbl = input('Please enter ylabel : ')
        ttl = 'Plot of '+xlbl+' versus '+ylbl
        yn = input('Do you want to use auto-title "'+ttl+'" ?(y/n) ')
        if yn == 'n':
            ttl = input('Please enter figure title : ')
        plt.title(ttl)
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
    else:
        plt.title('Plot of column '+str(i)+' against column '+str(j))
        plt.xlabel('Column '+str(i))
        plt.ylabel('Column '+str(j))
    plt.show()
# %% result

def average(li):
    """Returns the average of a list"""
    s = 0
    for i in li:
        s += i
    return (s/len(li))

def compute_result(real, pred):
    """Computes the three indicators MAPE, RMSE, R-squared and returns them"""
    sse = 0
    ape = 0
    sst = 0
    meanpred = average(pred)
    # print(meanpred)
    for i in range(len(real)):
        sse += (real[i] - pred[i])**2
        ape += abs((pred[i] - real[i])/real[i])
        sst += (real[i] - meanpred)**2
    mse = sse / len(real)
    rmse = np.sqrt(mse)
    mape = ape / len(real)
    rsquared = 1 - sse/sst
    return (mape, rmse, rsquared)

# %% moving average


def moving_average(data, realvar, length, plot = True):
    """Compute moving average prediction (one hour each time) with length as window length. Returns the indicators."""
    window = [realvar[i] for i in range(length)]
    ave = average(window)

    x = [i for i in range(length, len(realvar))]
    real = realvar[length:]
    pred = [ave]

    for i in range(length, len(realvar)-1):
        window = window[1:]+[realvar[i]]
        ave = average(window)
        pred.append(ave)

    if plot:
        # print(x)
        # print(real)
        # print(pred)
        plt.plot(x, real, color = 'green', label = 'Real Production')
        plt.plot(x, pred, color = 'blue', label = 'Prediction')
        plt.title('Moving average prediction using '+str(length)+' last hours')
        plt.legend()
        plt.show()
    return compute_result(real, pred)

def optimal_window_length(data, realvar, lengthmin, lengthmax, use = 'all'):
    """Returns the window length that gives the best result for used indicator, and plot the curves."""
    storeMAPE = []
    minMAPE = -1
    iMinMAPE = -1
    storeRMSE = []
    minRMSE = -1
    iMinRMSE = -1
    storeR2 = []
    maxR2 = -1
    iMaxR2 = -1
    x = [i for i in range(lengthmin, lengthmax)]
    for i in x:
        result = moving_average(data, realvar, i, plot = False)
        storeMAPE.append(result[0])
        storeRMSE.append(result[1])
        storeR2.append(result[2])
        if result[0] < minMAPE or iMinMAPE == -1:
            minMAPE = result[0]
            iMinMAPE = i
        if result[1] < minRMSE or iMinRMSE == -1:
            minRMSE = result[1]
            iMinRMSE = i
        if result[2] > maxR2 or iMaxR2 == -1:
            maxR2 = result[2]
            iMaxR2 = i
    if use == 'mape':
        plt.plot(x, storeMAPE)
        plt.title('MAPE versus length of moving window')
        plt.show()
        return(iMinMAPE, minMAPE, storeRMSE[iMinMAPE-lengthmin], storeR2[iMinMAPE-lengthmin])
    elif use == 'rmse':
        plt.plot(x, storeRMSE)
        plt.title('RMSE versus length of moving window')
        plt.show()
        return(iMinRMSE, storeMAPE[iMinRMSE-lengthmin], minRMSE, storeR2[iMinRMSE-lengthmin])
    elif use == 'R2':
        plt.plot(x, storeR2)
        plt.title('R2 versus length of moving window')
        plt.show()
        return (iMaxR2, storeMAPE[iMaxR2-lengthmin], storeRMSE[iMaxR2-lengthmin], maxR2)
    elif use == 'all':
        plt.plot(x, storeMAPE)
        plt.title('MAPE versus length of moving window')
        plt.show()
        plt.plot(x, storeRMSE)
        plt.title('RMSE versus length of moving window')
        plt.show()
        plt.plot(x, storeR2)
        plt.title('R2 versus length of moving window')
        plt.show()
        return (iMinMAPE, minMAPE, iMinRMSE, minRMSE, iMaxR2, maxR2)
    
# %% random forest

def split_tt(data, realvar, cutindex, length = 24):
    """Split train/test. It cts out the next <length> hours, starting at <cutindex>. the remaining data, is treated so that it considers it to be previous year."""
    if cutindex + length < len(data): # does not change year
        dataCopy = deepcopy(data)
        for i in range(len(data)-24,len(data)):
            p = realvar[i]
            dataCopy[(i+1) % len(data),10] = p
            #
            # Comment these lines if 1 shift only !
            #
            dataCopy[(i+2) % len(data),11] = p
            dataCopy[(i+5) % len(data),12] = p
            dataCopy[(i+12) % len(data),13] = p
            dataCopy[(i+24) % len(data),14] = p
            # 
            # Until There
            #

        xtrain1 = data[:cutindex].tolist()
        xtrain2 = data[cutindex+length:].tolist()
        xtest = data[cutindex : cutindex+length]
        xtrain = np.array(xtrain2 + xtrain1)

        ytrain1 = realvar[:cutindex].tolist()
        ytrain2 = realvar[cutindex+length:].tolist()
        ytest = realvar[cutindex : cutindex+length]
        ytrain = np.array(ytrain2 + ytrain1)
    elif cutindex + length == len(data): #finish at the exact end of the data
        xtrain = data[:cutindex]
        xtest = data[cutindex:]
        ytrain = realvar[:cutindex]
        ytest = realvar[cutindex:]
    else : 
        print('Warning : The period you are trying to predict comes over the end of accessible data. You predict data that is circularly linked to the start but may not be continuous')
        cutindex2 = cutindex+length - len(data)
        dataCopy = deepcopy(data)
        for i in range(len(data)-24,len(data)):
            p = realvar[i]
            if (i+1) % len(data) < cutindex2:
                dataCopy[(i+1) % len(data),10] = p
            #
            # Comment these lines if 1 shift only !
            #
            if (i+2) % len(data) < cutindex2:
                dataCopy[(i+2) % len(data),11] = p
            if (i+5) % len(data) < cutindex2:
                dataCopy[(i+5) % len(data),12] = p
            if (i+12) % len(data) < cutindex2:
                dataCopy[(i+12) % len(data),13] = p
            if (i+24) % len(data) < cutindex2:
                dataCopy[(i+24) % len(data),14] = p
            #
            # Until there
            #

        xtrain = dataCopy[cutindex2 : cutindex]
        xtest1 = dataCopy[cutindex:].tolist()
        xtest2 = dataCopy[:cutindex2].tolist()
        xtest = np.array(xtest1 + xtest2)

        ytrain = realvar[cutindex2 : cutindex]
        ytest1 = realvar[cutindex:].tolist()
        ytest2 = realvar[:cutindex2].tolist()
        ytest = np.array(ytest1 + ytest2)
    return xtrain, ytrain, xtest, ytest

def rf(data, realvar, cutindex = -1, noplot = False, retErr = False):
    """Random Forest prediction. Return results on the sample"""
    model = RandomForestRegressor(n_estimators=100, criterion = 'mse')
    if cutindex == -1:
        cutindex = len(data)-24
    xtrain, ytrain, xtest, ytest = split_tt(data, realvar, cutindex)
    model.fit(xtrain, ytrain)
    pred = []
    for i in range(len(xtest)):
        # print(xtest[i].reshape(1,-1))
        p = model.predict(xtest[i].reshape(1,-1))
        pred.append(p)
        if i < len(xtest)-1:
            xtest[i+1,10] = p #use prediction to predict next hour
            #
            # Comment these lines if 1 shift only !
            #
            if i < len(xtest) - 2:
                xtest[i+2,11] = p
                if i < len(xtest) - 5:
                    xtest[i+5,12] = p
                    if i < len(xtest) - 12:
                        xtest[i+12,13] = p
                        if i < len(xtest) - 24:
                            xtest[i+24,14] = p
            #
            # Until there
            #
        # print(xtest[i], p)
    if not noplot:
        x = [i for i in range(len(pred))]
        plt.plot(x, ytest, color = 'green', label = 'Real value')
        plt.plot(x, pred, color = 'blue', label='Prediction')
        plt.legend()
        plt.title('RF prediction')
        plt.show()

        fimp = model.feature_importances_
        plt.bar([i for i in range(len(fimp))], fimp)
        plt.title('RF feature importance for prediction of the production')
        plt.show()
    
    pred = np.array(pred)
    pred = np.reshape(pred, (len(pred)))
    mape, rmse, rsq = compute_result(ytest, pred)
    if retErr:
        return (mape, rmse, rsq, ytest - pred)
    else:
        return (mape, rmse, rsq)

def rfmultiple(data, realvar, number = 10):
    """Multiple RF predictions, but coded with old version of split_tt, choice of window is therefore different"""
    model = RandomForestRegressor(n_estimators=100, criterion = 'mse')
    xtrain, ytrain, xtest, ytest = split_tt(data, realvar, len(data)-48)# please choose winow there (or recode)
    x = [i for i in range(len(xtest))]
    agr_results = [0,0,0]
    for n in range(number):
        print('\r'+str(n)+'\t\t', end='', flush = True)
        model.fit(xtrain, ytrain)
        pred = []
        for i in range(len(xtest)):
            # print(xtest[i].reshape(1,-1))
            p = model.predict(xtest[i].reshape(1,-1))
            pred.append(p)
            if i < len(xtest)-1:
                xtest[i+1,10] = p #use prediction to predict next hour
                #
                # Comment these lines if 1 shift only !
                #
                if i < len(xtest) - 2:
                    xtest[i+2,11] = p
                    if i < len(xtest) - 5:
                        xtest[i+5,12] = p
                        if i < len(xtest) - 12:
                            xtest[i+12,13] = p
                            if i < len(xtest) - 24:
                                xtest[i+24,14] = p
                #
                # Until there
                #
            # print(xtest[i], p)
        res = compute_result(ytest, pred)
        for i in range(len(agr_results)):
            agr_results[i] += res[i]
        plt.plot(x, pred, label='Prediction '+str(n))
    # plt.legend()
    for i in range(len(agr_results)):
        agr_results[i] = agr_results[i]/len(ytest) #mean 
    print(agr_results)
    plt.plot(x, ytest, color = 'green', linewidth = 5, label = 'Real value')
    plt.title('RF prediction')
    plt.show()

def rf_wholeyear(data, realvar):
    """Iterates RF prediction on each day of the whole year, with each time a previous training on the rest of the dataset"""
    agr_results = [0,0,0]
    agr_err = []
    count = 0
    for cutindex in range(0, len(data)-23, 24):
        print('\r'+str(count+1)+'/'+str(len(data)//24), end = '\t',flush=True)
        mape, rmse, rsquared,err = rf(data, realvar, cutindex, noplot = True, retErr=True) # model is created here inside, suboptimal but more simple to code
        agr_results[0] += mape
        agr_results[1] += rmse
        agr_results[2] += rsquared
        count += 1
        agr_err = agr_err + err.tolist()
    for i in range(len(agr_results)):
        agr_results[i] = agr_results[i]/count
    plt.plot([i for i in range(len(agr_err))],agr_err)
    plt.title('Error of RF prediction')
    plt.show()
    count_res = [0]*201
    for res in agr_err:
        i = int(res/1000) + 100
        i = min(i,200)
        i = max(0,i)
        count_res[i] +=1
    # with open('./store_res.txt','w') as f:
    #     for c in count_res:
    #         f.write(str(c)+'\n')
    
    plt.bar([1000*(i-100) for i in range(len(count_res))],count_res, width = 1000)
    plt.title('Density of residuals')
    plt.show()
    return (agr_results)

# %% XGBoost

def xgb(data, realvar, cutindex = -1, noplot = False, retErr = False):
    """XGBoost preditor : returns the result on given window"""
    model = XGBRegressor(n_estimators = 100)
    if cutindex == -1:
        cutindex = len(data)-24
    xtrain, ytrain, xtest, ytest = split_tt(data, realvar, cutindex)
    model.fit(xtrain, ytrain)
    pred = []
    for i in range(len(xtest)):
        # print(xtest[i].reshape(1,-1))
        p = model.predict(xtest[i].reshape(1,-1))
        pred.append(p)
        if i < len(xtest)-1:
            xtest[i+1,10] = p #use prediction to predict next hour
            #
            # Comment these lines if 1 shift only !
            #
            if i < len(xtest) - 2:
                xtest[i+2,11] = p
                if i < len(xtest) - 5:
                    xtest[i+5,12] = p
                    if i < len(xtest) - 12:
                        xtest[i+12,13] = p
                        if i < len(xtest) - 24:
                            xtest[i+24,14] = p
            #
            # Until there
            #
        # print(xtest[i], p)
    if not noplot:
        x = [i for i in range(len(pred))]
        plt.plot(x, ytest, color = 'green', label = 'Real value')
        plt.plot(x, pred, color = 'blue', label='Prediction')
        plt.legend()
        plt.title('XGBoost prediction')
        plt.show()

        fimp = model.feature_importances_
        plt.bar([i for i in range(len(fimp))], fimp)
        plt.title('XGBoost feature importance for prediction of the production')
        plt.show()
    
    pred = np.array(pred)
    pred = np.reshape(pred, (len(pred)))
    mape, rmse, rsq = compute_result(ytest, pred)
    if retErr:
        return (mape, rmse, rsq, ytest - pred)
    else:
        return (mape, rmse, rsq)

def xgb_wholeyear(data, realvar):
    """Iterates XGBoost prediction on each day of the whole year, with each time a previous training on the rest of the dataset"""
    agr_results = [0,0,0]
    agr_err = []
    count = 0
    for cutindex in range(0, len(data)-23, 24):
        print('\r'+str(count+1)+'/'+str(len(data)//24), end = '\t',flush=True)
        mape, rmse, rsquared,err = xgb(data, realvar, cutindex, noplot = True, retErr=True) # model is created here inside, suboptimal but more simple to code
        agr_results[0] += mape
        agr_results[1] += rmse
        agr_results[2] += rsquared
        count += 1
        agr_err = agr_err + err.tolist()
    for i in range(len(agr_results)):
        agr_results[i] = agr_results[i]/count
    plt.plot([i for i in range(len(agr_err))],agr_err)
    plt.title('Error of XGBoost prediction')
    plt.show()
    count_res = [0]*201
    for res in agr_err:
        i = int(res/1000) + 100
        i = min(i,200)
        i = max(0,i)
        count_res[i] +=1
    # with open('./store_res.txt','w') as f:
    #     for c in count_res:
    #         f.write(str(c)+'\n')
    
    plt.bar([1000*(i-100) for i in range(len(count_res))],count_res, width = 1000)
    plt.title('Density of residuals')
    plt.show()
    return (agr_results)

# multiple tries is useless because they are all equal

# %% double XGBoost

def specific_range_hour(sample):
    """Defines th specific range of hour for model separation"""
    return (sample[4] >= 9 and sample[4] <= 18)

def split_specrange(data, realvar, specrangefunc = specific_range_hour):
    """Splits the training data uing the specified hour range"""
    data0 = []
    data1 = []
    realvar0 = []
    realvar1 = []
    for i in range(len(data)):
        if specrangefunc(data[i]): # special zone
            data1.append(data[i].tolist())
            realvar1.append(realvar[i])
        else:
            data0.append(data[i].tolist())
            realvar0.append(realvar[i])
    data0np = np.array(data0)
    data1np = np.array(data1)
    realvar0np = np.array(realvar0)
    realvar1np = np.array(realvar1)
    return (data0np, data1np, realvar0np, realvar1np)

def double_xgb(data, realvar, cutindex = -1, noplot = False, retErr = False, specrange = specific_range_hour, dist_weights = False, weightsalpha = 2*10**-7):
    """Double-XGBoost predictor. 
        @arg dist_weights : True for including ponderation by distance to actual date using weightsalpha"""
    model0 = XGBRegressor(n_estimators = 100) # out of special interval
    model1 = XGBRegressor(n_estimators = 100) # inside special interval
    if cutindex == -1:
        cutindex = len(data)-24
    xtrain, ytrain, xtest, ytest = split_tt(data, realvar, cutindex) # cut the testing day out of the data
    xtr0, xtr1, ytr0, ytr1 = split_specrange(xtrain, ytrain, specrangefunc = specrange) # split the two subsets for training (no need to split the tesing one)
    if dist_weights:
        # print(data[cutindex,1],data[cutindex,2],data[cutindex,3],data[cutindex,4],0,0)
        date = dt.datetime(int(data[cutindex,1]),int(data[cutindex,2]),int(data[cutindex,3]),int(data[cutindex,4]),0,0)
        wght0 = weights(xtr0, date, alpha = weightsalpha)
        wght1 = weights(xtr1, date, alpha = weightsalpha)
    else:
        wght0 = None
        wght1 = None
    # fit both models separately on specific subsets
    model0.fit(xtr0, ytr0, sample_weight = wght0)
    model1.fit(xtr1, ytr1, sample_weight = wght1)
    # start of the testing phase
    pred = []
    for i in range(len(xtest)):
        # print(xtest[i].reshape(1,-1))
        if specrange(xtest[i]): # belongs to model1 subset
            p = model1.predict(xtest[i].reshape(1,-1))
        else: # belongs to model0 subset
            p = model0.predict(xtest[i].reshape(1,-1))
        pred.append(p)
        if i < len(xtest)-1:
            xtest[i+1,10] = p #use prediction to predict next hour
            #
            # Comment these lines if 1 shift only !
            #
            if i < len(xtest) - 2:
                xtest[i+2,11] = p
                if i < len(xtest) - 5:
                    xtest[i+5,12] = p
                    if i < len(xtest) - 12:
                        xtest[i+12,13] = p
                        if i < len(xtest) - 24:
                            xtest[i+24,14] = p
            #
            # Until there
            #
        # print(xtest[i], p)
    if not noplot:
        x = [i for i in range(len(pred))]
        plt.plot(x, ytest, color = 'green', label = 'Real value')
        plt.plot(x, pred, color = 'blue', label='Prediction')
        plt.legend()
        plt.title('XGBoost prediction')
        plt.show()

        fimp = model0.feature_importances_
        plt.bar([i for i in range(len(fimp))], fimp)
        plt.title('XGBoost model0 feature importance for prediction of the production')
        plt.show()

        fimp = model1.feature_importances_
        plt.bar([i for i in range(len(fimp))], fimp)
        plt.title('XGBoost model1 feature importance for prediction of the production')
        plt.show()
    
    pred = np.array(pred)
    pred = np.reshape(pred, (len(pred)))
    mape, rmse, rsq = compute_result(ytest, pred)
    if retErr:
        return (mape, rmse, rsq, ytest - pred)
    else:
        return (mape, rmse, rsq)

def double_xgb_wholeyear(data, realvar, specrange = specific_range_hour, dist_weights = False, weightsalpha = 2*10**-7, noplot = False):
    """Iterates double-XGBoost prediction on each day of the whole year, with each time a previous training on the rest of the dataset"""
    agr_results = [0,0,0]
    agr_err = []
    count = 0
    for cutindex in range(0, len(data)-23, 24):
        print('\r'+str(count+1)+'/'+str(len(data)//24), end = '\t',flush=True)
        mape, rmse, rsquared,err = double_xgb(data, realvar, cutindex, noplot = True, retErr=True, specrange = specrange, dist_weights = dist_weights, weightsalpha = weightsalpha)
        agr_results[0] += mape
        agr_results[1] += rmse
        agr_results[2] += rsquared
        count += 1
        agr_err = agr_err + err.tolist()
    print('')
    for i in range(len(agr_results)):
        agr_results[i] = agr_results[i]/count
    if not noplot:
        plt.plot([i for i in range(len(agr_err))],agr_err)
        plt.title('Error of XGBoost prediction')
        plt.show()

        count_res = [0]*201
        for res in agr_err:
            i = int(res/1000) + 100
            i = min(i,200)
            i = max(0,i)
            count_res[i] +=1
   
        plt.bar([1000*(i-100) for i in range(len(count_res))],count_res, width = 1000)
        plt.title('Density of residuals')
        plt.show()
    return (agr_results)

# %% add weights by similar season

def weights(data,preddate, alpha = 2*10**-7):
    """Computes weights by distance to first predicted date (considering the shortest distance with previous and next year same date)
        @arg alpha controls the exponential distance, the default one is the one identified as giving the better results"""
    year = preddate.year 
    weights = np.zeros(len(data))
    for i in range(len(data)):
        d = dt.datetime(year,int(data[i,2]),int(data[i,3]),int(data[i,4]),0,0)
        d1 = dt.datetime(year+1,int(data[i,2]),int(data[i,3]),int(data[i,4]),0,0)
        d2 = dt.datetime(year-1,int(data[i,2]),int(data[i,3]),int(data[i,4]),0,0)
        weights[i] = np.exp(- alpha*(min(abs(preddate-d), abs(preddate-d1), abs(preddate-d2)).total_seconds()) )
    plt.plot([i for i in range(len(weights))],weights)
    plt.show()
    return weights

def varying_alpha(data, realvar, alpharange):
    """Plots curve with three indicators on given values of alpha for the distance"""
    rrr = []
    for alpha in alpharange:
        print('Alpha used : '+str(alpha))
        rrr.append(double_xgb_wholeyear(data, realvar, dist_weights=True, weightsalpha = alpha, noplot = True))
    rrrnp = np.array(rrr)
    plt.title('Plot of the three indicators by alpha value')
    plt.plot(alpharange,rrrnp[:,0],label = 'MAPE')
    plt.plot(alpharange,rrrnp[:,1]/30000,label = 'RMSE/30000')
    plt.plot(alpharange,rrrnp[:,2],label = 'Rsquared')
    plt.legend()
    plt.xlabel('alpha')
    plt.show()



# %% launch

if __name__ == '__main__':
    args = parse_args()
    data = parse_data(args.ifile, char2remove = ['\n'], sep = ';')
    data = convert_data(data)
    # for i in data:
    #     print(i)
    # data = normalize(data)
    x, y = split_xy(data, 7)
    # print(moving_average(x, y, 24, plot = True))
    # print(optimal_window_length(x, y, 1, 24*7, use = 'all'))
    # print(rf(x, y))
    # print(rfmultiple(x, y,50))
    print(rf_wholeyear(x, y))
    # print(xgb(x, y))
    # print(xgb_wholeyear(x, y))
    # print(double_xgb(x, y, dist_weights=True))
    # print(double_xgb_wholeyear(x,y, dist_weights=True))
    # varying_alpha(x, y, [10**-7,2*10**-7,3*10**-7])