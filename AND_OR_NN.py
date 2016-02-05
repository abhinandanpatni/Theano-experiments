'''
This is an implementation of a neural network with 2 inputs, 1 hidden layer with 3 units, and 2 outputs.
It implements both AND & OR gates.
Expected output:
X Y AND OR
0 0  0  0
0 1  0  1
1 0  0  1
1 1  1  1
'''
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as np

def layerComputation(X,W):  #function to compute sigmoid values between 2 layers. Based on UFLDL tutorial
    dot_product = T.dot(W,X)
    return NN.sigmoid(dot_product)

def sGradientDescent(error, W): #function that calculates gradient descent. Learning factor = 0.1 (researched online, this seemed like a good number)
    return (W - (0.1 * T.grad(error, wrt=W)))

x = T.lvector('x')  #input vector of 2x1
y = T.lvector('y')  #output vector of 2x1

W1 = theano.shared(np.random.random((3,2))) #Creating shared weight matrices with random values so they can be updated after every iteration
W2 = theano.shared(np.random.random((2,3))) #Maybe instead of random, we init to zero matrix and increase learning factor?

hidden_layer = layerComputation(x,W1)   #Computation for hidden layer
output = layerComputation(hidden_layer, W2) #Output of hidden layer is now input
error = T.sum((output - y) ** 2)    #Squared error calculation.

error_func = theano.function([x, y], error, updates=[(W1, sGradientDescent(error,W1)), (W2, sGradientDescent(error,W2))])   #Error calculation and weight updation
run_nn = theano.function([x], output)   #Driver function

train_input = np.array([[0,0],[0,1],[1,0],[1,1]])   #Input and output based on truth tables
train_output = np.array([[0,0],[0,1],[0,1],[1,1]])

initial_error = 0
while True:
    for k in range(len(train_input)):
        current_error = error_func(train_input[k], train_output[k])
    if abs(initial_error - current_error) < 1e-10:  #Continue updating the weight matrix till the error is small enough
        break
    initial_error = current_error

print(run_nn([0,0]))
print(run_nn([0,1]))
print(run_nn([1,0]))
print(run_nn([1,1]))
