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


hyperparameter tuning

other algrithms to test ddata on , kcluster, random forest
"""

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA 
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import time
start_time = time.time()


iris = datasets.load_iris()
# Read the CSV file
df = pd.read_csv('/Users/poojap/Documents/EatingAlgorithm/prelimEatingDetection/Pooja/AccelerometerWatchData_Excel/data_1600_accel_watch.csv')
dataframe = pd.DataFrame(df)

#Convert CSV file to a dataset
dataset = dataframe.to_numpy()

# Columns 3 4 5 are x y z respectively, create triple
xyz = dataset [ :, (3,4)] # add column 5 for thr y value


# Make the target eating coordinates , these have a value of 1
target = (dataset[:, 1] == 'eating').astype(np.float64())

# y = (iris["target"] ==2).astype(np.float64)
print("target", target)

s = set()
for val in target:
    s.add(val)
s = list(s)


C = 1.0

# Training the model 
x_train, x_test, y_train, y_test = train_test_split(xyz, target, random_state = 0, test_size = 0.25)

clf = svm.SVC(kernel='linear', C=1). fit(x_train, y_train)

# Make predictions with the x test data
classifier_predictions = clf.predict(x_test)

# Accuracy between y_Test (target) and classifier_predictions
print (accuracy_score(y_test, classifier_predictions)*100)



svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss = "hinge"))
    ])

print(svm_clf.fit(xyz, target))

svm_clf.predict([[5.5, 1.7]])
print(svm_clf.score(xyz, target))


tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(xyz[target==0,0], xyz[target==0,1], 'ob')
ax.plot3D(xyz[target==1,0], xyz[target==1,1],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()








# poly_kernel_svm_clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=1))
#     ])
# poly_kernel_svm_clf.fit(xyz, target)


# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(projection = '3d')
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(xyz,target, color = 'green')

# print(xyz[:,1])
# # ax.scatter(xyz, target) # plot the point (2,3,4) on the figure

# plt.show()


# # ax.scatter3D(noneat_x, noneat_y, noneat_z, cmap='Greens', color = 'green')

# # for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
# #     # Plot the decision boundary. For that, we will assign a color to each
# #     # point in the mesh [x_min, x_max]x[y_min, y_max].
# #     plt.subplot(2, 2, i + 1)
# #     plt.subplots_adjust(wspace=0.4, hspace=0.4)

# #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
# #     # Put the result into a color plot
# #     Z = Z.reshape(xx.shape)
# #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# #     # Plot also the training points
# #     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
# #     plt.xlabel('Sepal length')
# #     plt.ylabel('Sepal width')
# #     plt.xlim(xx.min(), xx.max())
# #     plt.ylim(yy.min(), yy.max())
# #     plt.xticks(())
# #     plt.yticks(())
# #     plt.title(titles[i])

# # plt.show()


print("--- %s seconds ---" % (time.time() - start_time))


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