#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:49:50 2022

@author: poojap

Eating Mode is split between what is being eaten
soup = H
chips = I
pasta = J
drinking = K 36052 - 39658=
sandwhich - L

Handss on machine learning with sktlearn by O'riley '


Cross validation,
Sliding window for each validation set
Then train?

hyperparameter tuning

other algrithms to test ddata on , kcluster, random forest
"""

# To import all necessary packages, please run: pip install -r ./requirements.txt 

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
import os
import shutil
import glob
import statistics
import csv
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import TimeSeriesSplit


def classifyData():
    
    #Most likely need to change pathway when submit, download the dataset 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_parent = os.path.dirname(dir_path)
    data ='DataSetFiles/raw/watch/accel'


    readData = os.path.join(path_parent, data)

    files = os.path.join(readData, "*.txt")

    # list of merged files returned
    joinedfiles = glob.glob(files)

    #First must add a header to each filos.e
    # files = ['data_1607_accel_watch.csv', 'data_1608_accel_watch.csv']
    header = ['SubjectID', 'Class', 'TimeStamp', 'x', 'y', 'z']
    
    inputvalue = 'SubjectID'
    has_header = True
    
    #Make new directory 
    newdirname = "DataSetFiles/HeaderFiles"
    newdir = os.path.join(path_parent, newdirname)
    
    #Remove header directory if it already exists
    if os.path.exists(newdir):
        shutil.rmtree(newdir)

    # if not os.path.exists(newdir):
    os.makedirs(newdir)

    for f in joinedfiles:
        shutil.copy(f, newdir)
    
    joinnew = os.path.join(newdir, "*.txt")
    newfiles = glob.glob(joinnew)
    #Make a new directory to put header files in 
    for filename in joinedfiles:
        with open(filename) as infile:
            text = infile.read()
            reader = csv.reader(infile, delimiter=',' )
    
    for filename in newfiles:
        with open(filename, 'w') as outfile:
            # join the headers into a string with commas and add a newline
            outfile.write(f"{','.join(header)}\n") 
            outfile.write(text)
  
    # else:
        # joinnew = os.path.join(newdir, "*.txt")
        # newfiles = glob.glob(joinnew)

    df = pd.concat(map(pd.read_csv, newfiles), ignore_index=True)
    df['TimeStamp'] = df['TimeStamp'].replace(np.nan, 0)

    # df.to_csv('/Users/poojap/Documents/EatingAlgorithm/prelimEatingDetection/Pooja/my_data.csv', index=False)
    dataframe = pd.DataFrame(df)
    dataframe.fillna(0)

    #Convert CSV file to a dataset.
    dataset = dataframe.to_numpy()
    
    # Columns 3 4 5 are x y z respectively, create triple
    xyz = dataset [ :, (3,4,5)] # add column 5 for thr y value
    
    x = df['x']
    y = df['y']
    z = df['z']
    
    print(len(x), len(z))
    semicolon = ';'
    string = 'hey i dont know'
    
    df['z'] = df['z'].str.replace(';','')

    df.insert(loc=6,
          column='binary_eating',
          value=0)
    
    classes = df['Class']
    datatime  = df['TimeStamp']
    
    df['binary_eating'] = df['Class']
    
    eatingClasses = ['H', 'I', 'J', 'K', 'L']
    noneatingclasses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    
    for value in eatingClasses:
        df.loc[df['Class'] == value, 'binary_eating'] = 1
    for nonval in noneatingclasses:
        df.loc[df['Class'] == nonval, 'binary_eating'] = 0


    target = df['binary_eating']
    classes = classes.to_numpy()
    
    count =0
    # Make the target eating coordinates , these have a value of 1

    
    howmany = 0
    for value in target:
        if value == 1:
            howmany +=1

    time_seconds = []
    #Consider the timestamp 
    # C = 1.0

    return df, dataframe, datatime, target, x, y, z #, datatime

def extract_windows(array, mHZ, seconds, overlap):
    """clearting_time_index is the first value it takes from array
    # subArrayamt is maximum amount of subarray sbeing output - if
    
    # sub window size is how many values in each subarray
    # 50% overlap 
    """
    windows = []
    array = array.to_numpy()
    
    windowSize = mHZ * seconds
    
    # start = clearing_time_index + 1 - sub_window_size + 1
        
    overlapamt = int(windowSize* overlap)
    for i in range(len(array)):
        window = array[ (overlapamt *i) : (overlapamt*i) +windowSize ]
        if len(window)!= 0:
            windows.append(window)

    return windows


def featureEngineering(x_windows, y_windows, z_windows, target_windows):
    
    vector = np.vectorize(float)

    # for j in range(len(x_windows)):
    #     xwindows[j] = x_windows[j].astype(int)
    for j in range(0, len(x_windows)):
        # print("array", j, ": ", x_windows[j])
        x_windows[j] = vector(x_windows[j])
        y_windows[j] = vector(y_windows[j])
        z_windows[j] = vector(z_windows[j])


#Engineer the mean
    #x feature
    x_mean = []
    for i in range(len(x_windows)):
        xavg = np.mean(x_windows[1])
        x_mean.append(xavg)

    #y feature
    y_mean = []
    for i in range(len(y_windows)):
        yavg = np.mean(y_windows[i])
        y_mean.append(yavg)
        
    #z feature 
    z_mean = []
    for i in range(len(z_windows)):
        zavg = np.mean(z_windows[i])
        z_mean.append(zavg)
        
#Engineer standard deviation
    #x std
    x_std = []
    for i in range(len(x_windows)):
        xstd = np.std(x_windows[i])
        x_std.append(xstd)
    #y std
    y_std = []
    for i in range(len(y_windows)):
        ystd = np.std(y_windows[i])
        y_std.append(ystd)
    
    #z std
    z_std = []
    for i in range(len(z_windows)):
        zstd = np.std(z_windows[i])
        z_std.append(zstd)
        
#Engineer absolute average deviation
    x_absavgdev = []
    for i in range(len(x_windows)):
        xaad = np.mean(np.absolute(x_windows[i] - np.mean(x_windows[i])))
        x_absavgdev.append(xaad)
    # np.mean(np.absolute(data - np.mean(data)))
    
    y_absavgdev = []
    for i in range(len(y_windows)):
        yaad = np.mean(np.absolute(y_windows[i] - np.mean(y_windows[i])))
        y_absavgdev.append(yaad)
    
    z_absavgdev = []
    for i in range(len(z_windows)):
        zaad = np.mean(np.absolute(z_windows[i] - np.mean(z_windows[i])))
        z_absavgdev.append(zaad)
        
        
#Engineer minimum value
    x_min = []
    for i in range(len(x_windows)):
        xmin = min(x_windows[i])
        x_min.append(xmin)
        
    y_min = []
    for i in range(len(y_windows)):
        ymin = min(y_windows[i])
        y_min.append(ymin)
        
    z_min = []
    for i in range(len(z_windows)):
        zmin = min(z_windows[i])
        z_min.append(zmin)
        
#Engineer maximum value
    x_max = []
    for i in range(len(x_windows)):
        xmax = min(x_windows[i])
        x_max.append(xmax)
    
    y_max = []
    for i in range(len(y_windows)):
        ymax = min(y_windows[i])
        y_max.append(ymax)
    
    z_max = []
    for i in range(len(z_windows)):
        zmax = min(z_windows[i])
        z_max.append(zmax)
#Median feature

    x_median = []
    for i in range(len(x_windows)):
        xmedian = statistics.median(x_windows[j])
        x_median.append(xmedian)
        
    y_median = []
    for i in range(len(y_windows)):
        ymedian = statistics.median(y_windows[j])
        y_median.append(ymedian)
        
    z_median = []
    print(len(z_windows))
    for i in range(len(z_windows)):
        zmedian = statistics.median(z_windows[i])
        z_median.append(zmedian)
        

#Mode Feature

    x_mode = []
    for i in range(len(x_windows)):
        xstat = statistics.mode(x_windows[i])
        x_mode.append(xstat)
        
    y_mode = []
    for i in range(len(y_windows)):
        ystat = statistics.mode(y_windows[i])
        y_mode.append(ystat)
        
    z_mode = []
    for i in range(len(z_windows)):
        zstat = statistics.mode(z_windows[i])
        z_mode.append(zstat)
        
# Range feature 

    x_range = []
    for i in range(len(x_windows)):
        xrange = np.ptp(x_windows[i])
        x_range.append(xrange)
        
    y_range = []
    for i in range(len(y_windows)):
        yrange = np.ptp(y_windows[i])
        y_range.append(yrange)
        
    z_range = []
    for i in range(len(z_windows)):
        zrange = np.ptp(z_windows[i])
        z_range.append(zrange)
        
        
#Target mode to make even 

    target_mode = []
    for i in range(len(target_windows)):
        targetv = statistics.mode(target_windows[i])
        target_mode.append(targetv)
# # log(x)

#     x_log = []
#     for i in range(len(x_windows)):
#         xlogarr = x_windows[i]
#         for j in range(len(xlogarr)):
#             xlog
#             xlogarr[j] = math.log(abs(xlogarr[j]))
#         x_log.append(xlogarr)
        
#     y_log = []
#     for i in range(len(y_windows)):
#         ylogarr = y_windows[i]
#         for j in range(len(ylogarr)):
#             ylogarr[j] = math.log(abs(ylogarr[j]))
#         y_log.append(ylogarr)
        
#     z_log = [] 
#     for i in range(len(z_windows)):
#         zlogarr = z_windows[i]
#         for j in range(len(zlogarr)):
#             zlogarr[j] = math.log(abs(zlogarr[j]))
#         z_log.append(zlogarr)
            

#sqrt(x)

# x^2


#Standard Normalization
    
#Feature scaling
    # x_scaleall = []
    # for i in range(len( x_windows)):
    #     xdiff = x_max[i] - x_min[i]
    #     specificarr = x_windows[i]
    #     for d in range(len(specificarr)):
    #         specificvalue = specificarr[d]
    #         x_scale = (specificvalue - x_min[i])/ xdiff
    #     x_scaleall.append(x_scale)

    
    return x_mean, y_mean, z_mean, x_std, y_std, z_std, \
            x_absavgdev, y_absavgdev, z_absavgdev, x_min, y_min, z_min, \
            x_max, y_max, z_max, x_median, y_median, z_median, x_mode, \
                y_mode, z_mode, target_mode
    
    
    
def AppendNewFeatures(df, x_windows, y_windows, z_windows, xmean, ymean, zmean, \
                      x_std, y_std, z_std, x_absavgdev, y_absavgdev, z_absavgdev, x_min, y_min, z_min, \
                         x_max, y_max, z_max, x_median, y_median, z_median, x_mode, y_mode, z_mode):
    
    X = [[0, 0], [1,1], [2,2]]
   
    df_features = pd.DataFrame(data = xmean, columns = ['x_mean'])
    df_features['y_mean'] = ymean
    df_features['z_mean'] = zmean
    
    df_features['x_std'] = x_std
    df_features['y_std'] = y_std
    df_features['z_std'] = z_std
    
    df_features['x_absavgdev'] = x_absavgdev
    df_features['y_absavgdev'] = y_absavgdev
    df_features['z_absavgdev'] = z_absavgdev
    
    df_features['x_min'] = x_min
    df_features['y_min'] = y_min
    df_features['z_min'] = z_min
    
    df_features['x_max'] = x_max
    df_features['y_max'] = y_max
    df_features['z_max'] = z_max
    
    df_features['x_median'] = x_median
    df_features['y_median'] = y_median
    df_features['z_median'] = z_median
    
    df_features['x_mode'] = x_mode
    df_features['y_mode'] = y_mode
    df_features['z_mode'] = z_mode
    
    return df_features

def TrainTestData(dataframe, Y):
    
    Y = pd.DataFrame(Y)
    
    X_trainindex , X_testindex, y_trainindex , y_testindex = train_test_split(dataframe.index,Y.index,test_size=0.2, random_state= None)
    X_train = dataframe.iloc[X_trainindex] # return dataframe train
    X_test = dataframe.iloc[X_testindex]
    
    Y_train = Y.iloc[y_trainindex]
    Y_test = Y.iloc[y_testindex]
    
    


    return X_train, X_test, Y_train , Y_test

def SVM(Xtrain, Xtest, Ytrain, Ytest):
    """ Sliding windows can only work with dataframes, so first must convert
    all x_train and y_train and tests to dataframe....."""
    
    # s = pd.Series(range(5))
    
    
    # for xwindow in Xtrain.rolling(window=windowSize):
    #     for ywindow in Ytrain.rolling(window=windowSize):
            
    #         allxwindows.append(xwindow)
    #         allywindows.append(ywindow)
    
    # for i in range(len(Xtrain)-windowSize+1):
    #     batch = Xtrain[i:i+windowSize]
        # print('Window: ', batch)
    Ytrain = Ytrain.values.ravel()
    
    Ytest = Ytest.values.ravel()
    #Error thrown says need to reshape data to be a 2d array, not 1d 
    # Ytest = Ytest.reshape(-1,1)
    
    model = SVC()

    # fit the model to data
    model.fit(Xtrain, Ytrain)
    
    # Make predictions 
    y_pred = model.predict(Xtest)
    
    #Model score 
    print('Model Score: ', model.score(Xtest, Ytest))
    
    #Accuracy Score
    accuracy = accuracy_score(Ytest, y_pred)
    print('Model Accuracy score: ', accuracy)    
    plt.plot(model.predict(Xtest))
    
    plt.plot(Ytest)
    plt.show()

    # pd.DataFrame(Xtrain, columns = ['Column_A','Column_B','Column_C'])
    

    
def main():
    start_time = time.time()
    
    df, dataframe, datatime, target, x, y, z = classifyData()
    
    # CVxtrainall, CVxtestall, CVytrainall, CVytestall, CVtrainIndex, CVtestIndex, CVnsplits = crossValidation(xyz, target)
    
    #Create array of consecutive integers until 10
    # output = list(range(10+1))
    # tryoutput = [ [1,2,3], [4,5,6], [7,8,9]]
    # # Training for 100 data points in each testing blob
    x_windows = extract_windows(x, 20, 10,  .5)
    y_windows = extract_windows(y, 20, 10,  .5)
    z_windows = extract_windows(z, 20, 10, .5)
    target_window = extract_windows(target, 20, 10, .5)
    
    print(np.shape(target_window[0]))
    
 
    xmean, ymean, zmean, x_std, y_std, z_std, \
            x_absavgdev, y_absavgdev, z_absavgdev, x_min, y_min, z_min, \
            x_max, y_max, z_max, x_median, y_median, z_median, x_mode, \
                y_mode, z_mode, target_mode \
                    = featureEngineering(x_windows, y_windows, z_windows, target_window)
            
            
    appended_df = AppendNewFeatures(df, x_windows, y_windows, z_windows, xmean, ymean, zmean, \
                          x_std, y_std, z_std, x_absavgdev, y_absavgdev, z_absavgdev, x_min, y_min, z_min, \
                             x_max, y_max, z_max, x_median, y_median, z_median, x_mode, y_mode, z_mode)
     
    # SVM(tr/ainexamples, CVxtestall, CVytrainall, CVytestall, CVnsplits)
    
    Xtrain , Xtest, Ytrain, Ytest = TrainTestData(appended_df, target_mode)
    print(np.shape(Xtrain))
    SVM(Xtrain, Xtest, Ytrain, Ytest)

    # for window in dataframe.rolling(window=2):
    #     print(window)
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
   
if __name__ == '__main__':
    main()
    
"""

model = svm.SVC(kernel = 'linear')
clf = model.fit(xyz, target)

z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef[0][2]


# x = df ['x']
# y = df['y']
# z = df['z']


# #need to Hard code all of the modes≠≠≠]
# # mode A is 0 - 3606 
# # until mode H is 0 - 25237

# noneat_x = x[:25237]
# noneat_y = y[:25237]
# noneat_z = z[:25237]


# eat_x = x[25238:36052]
# eat_y = y[25238:36052]
# eat_z = z[25238:36052]

# eat_x = eat_x.append( x[39658:43263])
# eat_y = eat_y.append( y[39658:43263])
# eat_z = eat_z.append( z[39658:43263])

# #this is also eating
# drinking_x= x[36052:39658]
# drinking_y= y[36052:39658]
# drinking_z= z[36052:39658]


# noneat_x = noneat_x.append(x[43264:])
# noneat_y = noneat_y.append(y[43264:])
# noneat_z = noneat_z.append(z[43264:])


"""
