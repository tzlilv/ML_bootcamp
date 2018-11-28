# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:19:11 2018

@author: stzli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn as skl
raw_data_1 = {
'Older sibling': [31, 22, 40, 26],
'Younger sibling': [22, 21, 37, 25],
'Times talked': [2, 3, 8, 12]}

data1 = pd.DataFrame(raw_data_1)

X = np.array([data1['Older sibling'],data1['Younger sibling']]).T
Y = np.array([data1['Times talked']]).T
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
[teta]=reg.coef_
[interc]=reg.intercept_

teta_normal_eq=np.linalg.inv(np.dot(X.T,X))
teta_normal_eq=np.dot(teta_normal_eq,X.T)
teta_normal_eq_end=np.dot(teta_normal_eq,Y)

a=np.ones((4,1))
X_new= np.concatenate((X, a), axis=1)
X=X_new
teta_normal_eq=np.linalg.inv(np.dot(X.T,X))
teta_normal_eq=np.dot(teta_normal_eq,X.T)
teta_normal_eq_end2=np.dot(teta_normal_eq,Y)

#q2
linear_regr=pd.read_csv(r'''D:\Downloads\data_for_linear_regression.csv''')
values_data=linear_regr.values
max_points=200

x=values_data[0:200,0]
x=x.reshape(-1,1)
y=values_data[0:200,1]
y=y.reshape(-1,1)

#2.3
plot_save=plt.scatter(x,y)
plt.show
#2.4
#values_data_with= np.concatenate((values_data,np.ones((700,1))), axis=1)
#values_data_with1= np.hstack((values_data[0:200,:],np.ones((200,1))))
a=np.ones((200,1))
#x_re=x.reshape(200,1)
x_1=np.hstack((x,a))
reg = LinearRegression().fit(x, y)
teta=reg.coef_
[interc_2]=reg.intercept_
#2.5
x_random=np.random.choice(x.shape [0], size = 700)
y_pred=teta*x_random+interc_2
plt.scatter(x,y)
plt.scatter(x_random,y_pred)
plt.show
#2.6
x=values_data[0:700,0]
x=x.reshape(-1,1)
y=values_data[0:700,1]
y=y.reshape(-1,1)
plt.xlim((0,100))
plt.ylim(0,100)
plt.scatter(x,y)
plt.scatter(x_random,y_pred)
plt.show



