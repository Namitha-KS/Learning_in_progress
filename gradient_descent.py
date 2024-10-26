import numpy as np
from tkinter import W
#initialise parameters
x = np.random.rand(10,1)
y = 5*x + np.random.rand()
# parameters
w = 0.0
b = 0.0
#HYPERPARAMETES
learning_rate = 0.01

# print(x.shape[0])

def descend(x,y,w,b,learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    # loss = (y-(wx+b))**2
    for xi,yi in zip(x,y):
        dldw += 2*xi*(yi-(w*xi+b))
        dldb += 2*(yi-(w*xi+b))
        
    w = w-learning_rate*(1/N)*dldw
    b = b-learning_rate*(1/N)*dldb
    
    return w,b
        

for epoch in range(400):
    w,b = descend(x,y,w,b,learning_rate)
    yhat = w*x+b
    loss = np.mean(np.square(y-yhat))
    print(loss)
    print(w,b)

#create gradient descent functions

#make updates
