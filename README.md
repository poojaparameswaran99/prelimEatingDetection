# prelimEatingDetection
The code here is for preliminary eating detection algorithms. 

The document 'EatingDetectionSVM.ipynb' contains a machine
learning network to detect between eating and non-eating movements.
A support vector machine is utilized as a binary classification method to take in
x,y, and z values and determine whether the movement is eating or non-eating.
The data is presented in a time-series manner with a specified time window.
This algorithm is to be used for incoming phone and watch accelerometer data
presenting x,y,z coordinates demonstrating movement in space.


# Dataset Information

A dataset provided by WISDM from the UCI Machine Learning Repository is used 
to train the network. Here, data from all participants using the watch as an 
accelerometer is used. The watch gathers data at 20 mHz a second. 50 participant's
data was used, and the x,y,z coordinates alongside the class {'A', 'B', 'C',
'D'... 'S'}

The classes {'H' - 'L'}, inclusive, are eating classes. The remaining are
classified as 'non-eating'. This is a binary classification algorithm,
so we will value 'eating' as 1, and 'non-eating' as 0.

The data is present in the directory 'Pooja/DataSetFiles/raw/watch/accel' as 
'.txt' files. 

# Eating Detection Algorithm (SVM)

The SVM classification algorithm is present in the directory
'Pooja/SupportVectorMachines/EatingDetectionSVM.ipynb' as a file to be opened
in a jupyter notebook.

Currently, the data is being interval-classified at a time section of 5 seconds,
from a watch gathering data at 20 mHz a second. However, if another wearable 
sensor is used, and data is being gathered at a different rate and for a
different period of time, please change the parameters mHz in extractWindows() 
appropriately.

Due to the large amount of data being utilized, features were engineered to 
better describe the large dataset. This reduced complexity and created a more 
robust model. When testing new data, please use the sliding window to 
extract windows, and then the 'featureEngineering()' function to accurately
engineer the data to pass through the classification SVM model. 

The model's performance is checked with four metrics: accuracy, precision,
recall, and F1. 

# Installation
To install the necessary packages please run
!pip install -r ./requirements.txt


# Extra Information
To learn more about machine learning networks, Support Vector Machines, 
Classification models, and the dataset, please look into the directories
'Pooja/TimeSeriesFundamentals' and 'Pooja/Papers'.

