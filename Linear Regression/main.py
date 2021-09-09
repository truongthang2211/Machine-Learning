import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def predict(new_radio,weight,bias):
    return new_radio*weight + bias
def cost_func(X,Y,weight,bias):
    n = len(X)
    sum = 0
    for i in range(n):
        sum+= (Y[i] - (X[i]*weight + bias))**2
    return sum/n
def update (X,Y,weight,bias,learning_rate):
    weight_temp = 0.0
    bias_temp = 0.0

    n = len(X)
    for i in range(n):
        weight_temp+= -2*X[i]*(Y[i] -(weight*X[i] + bias))
        bias_temp += -2*(Y[i] -(weight*X[i] + bias))
    weight-= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate
    return weight,bias
def training(X,Y,weight,bias,learning_rate,iter):
    cost_his= []
    for i in range(iter):
        weight,bias = update(X,Y,weight,bias,learning_rate)
        cost_his.append(cost_func(X,Y,weight,bias))
    return weight,bias,cost_his




dataFrame = pd.read_csv('Advertising.csv')
print (dataFrame)
X = dataFrame.values[:,2]
Y = dataFrame.values[:,4]
x = np.linspace(-2,50,100)

we,bias,cost_his = training(X,Y,0,0,0.001,50)
print(cost_his)
print(we,'x+',bias)


plt.plot(x, we*x+bias, '-r', label='y=2x+1')
plt.scatter(X,Y)
plt.show()

