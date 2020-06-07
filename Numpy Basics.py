# -*- coding: utf-8 -*-
"""
Numpy Basics

"""

import numpy as np
import math

#basic_sigmoid with math
def basic_sigmoid(x):

    s = 1/(1+ (math.exp(-x)))
    
    
    return s

print(basic_sigmoid(3))

#numpy sigmoid. With numpy we can apply sigmoid function to arrays and matrices
def sigmoid(x):

    s = 1/(1+ np.exp(-x))
    
    return s

x = np.array([1, 2, 3])
sigmoid(x)

"""
#Sigmoid Gradient
#sigmoid_derivative

"""
def sigmoid_derivative(x):

    s = sigmoid(x)
    ds = s*(1-s)
    
    return ds

x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))


"""
Reshaping Arrays:
    
X.shape is used to get the shape (dimension) of a matrix/vector X.
X.reshape(...) is used to reshape X into some other dimension.


"""
#Reshaping function
#image2vector
def image2vector(image):
    v = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    return v

#Lets reshape this 3x3x2 array. 
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],
       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

image.shape[0]
image.shape[1]
image.shape[2]

print ("image2vector(image) = " + str(image2vector(image)))



"""

Normalizing Rows

Another common technique we use in Machine Learning and Deep Learning is to normalize our data. 
It often leads to a better performance because gradient descent converges faster after normalization. 
By normalization we mean (dividing each row vector of x by its norm).

"""

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)  
    x = x/x_norm
    return x

x = np.array([[0, 3, 4],[1, 6, 4]])
print("normalizeRows(x) = ", normalizeRows(x))


"""
Broadcasting and the softmax function

Broadcasting is useful for performing mathematical operations between arrays of different shapes. 

"""

def softmax(x):
    x_exp = np.exp(x) 
    print("Shape of x_exp: ", x_exp.shape)
    
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    print("Shape of x_sum: ", x_sum.shape)
    
    #Numpy Broadcasting is needed to compute x_exp/x_sum due to different shape. 
    s = x_exp/x_sum
    print("Shape of s: ", s.shape)
    
    return s

x = np.array([[9, 2, 5, 0, 0],[7, 5, 0, 0 ,0]])


print("softmax(x) = " + str(softmax(x)))



"""

VECTORIZATION

A non-computationally-optimal function can become a huge bottleneck in your algorithm and can result in a model that takes ages to run. 
To make sure that the code is computationally efficient, we use vectorization. 

"""

import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]


"""
Inefficient Classic Methods
"""

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = ", float(1000*(toc - tic)), "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic1 = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc1 = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " , float(1000*(toc - tic)), "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic2 = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc2 = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " , float(1000*(toc - tic)), "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic3 = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc3 = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " , float(1000*(toc - tic)), "ms")

print("Total Time = ",float((1000*(toc2 - tic2))+ float(1000*(toc1 - tic1))+ float(1000*(toc3 - tic3))+ float(1000*(toc - tic))), "ms" )



"""
Efficient Numpy Methods
"""

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic1 = time.process_time()
outer = np.outer(x1,x2)
toc1 = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc1 - tic1)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic2 = time.process_time()
mul = np.multiply(x1,x2)
toc2 = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc2 - tic2)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic3 = time.process_time()
dot = np.dot(W,x1)
toc3 = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc3 - tic3)) + "ms")

print("Total Time = ",float((1000*(toc2 - tic2))+ float(1000*(toc1 - tic1))+ float(1000*(toc3 - tic3))+ float(1000*(toc - tic))), "ms" )




"""

Implement the L1 and L2 loss functions

"""

def L1(yhat, y): #Sum of abosulte (y-yhat)

    loss = np.sum(abs(y-yhat))
    
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))


def L2(yhat, y): #Sum of (y-yhat)**2
    loss = np.dot(y-yhat, y-yhat)
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
