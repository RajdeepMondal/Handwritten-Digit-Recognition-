import numpy as np
import matplotlib.pyplot as plt
import math
def initialize_parameters(layer_dims):
    parameters={}
    L=len(layer_dims)
    np.random.seed(1)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters["b"+str(l)]=np.zeros((layer_dims[l],1))
    return parameters

def relu(Z):
    A=np.maximum(0,Z)
    cache=Z
    return A,cache

def softmax(Z):
    T=np.exp(Z-np.max(Z))
    S=np.sum(T,axis=0,keepdims=True)
    A=T/S
    cache=Z
    assert(S.shape==(1,Z.shape[1]))
    assert(A.shape==Z.shape)
    return A,cache

def relu_backward(dA,activation_cache):
    Z=activation_cache
    dZ=np.array(dA,copy=True)
    dZ[Z<=0]=0
    return dZ

def random_minibatches(X,Y,t=64,seed=0):
    np.random.seed(seed)
    m=X.shape[1]
    minibatches=[]
    permutation=list(np.random.permutation(m))
    X_shuffled=X[:,permutation]
    Y_shuffled=Y[:,permutation].reshape((Y.shape[0],m))
    num_complete_minibatches=math.floor(m/t)
    for i in range(num_complete_minibatches):
        minibatch_X=X_shuffled[:,i*t:(i+1)*t]
        minibatch_Y=Y_shuffled[:,i*t:(i+1)*t]
        minibatch=(minibatch_X,minibatch_Y)
        minibatches.append(minibatch)
    if m%t!=0:
        minibatch_X=X_shuffled[:,num_complete_minibatches*t:]
        minibatch_Y=Y_shuffled[:,num_complete_minibatches*t:]
        minibatch=(minibatch_X,minibatch_Y)
        minibatches.append(minibatch)
    return minibatches

def linear_forward(A_prev,W,b):
    Z=np.dot(W,A_prev)+b
    cache=(A_prev,W,b)
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    Z,linear_cache=linear_forward(A_prev,W,b)
    if activation=="relu": A,activation_cache=relu(Z)
    elif activation=="softmax": A,activation_cache=softmax(Z)
    cache=(linear_cache,activation_cache)
    return A,cache

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost=-np.sum(Y*np.log(AL))/m
    return cost

def L_layer_forward(X,parameters):
    L=len(parameters)//2
    caches=[]
    A=X
    for l in range(1,L):
       A_prev=A
       A,current_cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
       caches.append(current_cache)
    AL,current_cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"softmax")
    caches.append(current_cache)
    return AL,caches

def linear_backward(dZ,cache):
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,Y,AL,cache,activation):
    linear_cache,activation_cache=cache
    if activation=="relu": dZ=relu_backward(dA,activation_cache)
    elif activation=="softmax": dZ=AL-Y
    dA_prev,dW,db=linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

def L_layer_backward(AL,Y,caches):
    grads={}
    L=len(caches)
    dA_temp,grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(None,Y,AL,caches[L-1],"softmax")
    for l in range (L-1,0,-1):
        dA_temp,grads["dW"+str(l)],grads["db"+str(l)]=linear_activation_backward(dA_temp,None,None,caches[l-1],"relu")
    return grads

def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)//2
    for l in range(1,L+1):
        parameters["W"+str(l)]-=learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)]-=learning_rate*grads["db"+str(l)]
    return parameters

def L_layer_model(X, Y, layer_dims, t=64,learning_rate = 0.0075, num_epochs = 3000, print_cost=False):
    np.random.seed(1)
    parameters=initialize_parameters(layer_dims)
    costs=[]
    seed=10
    for i in range(num_epochs):
        seed=seed+1
        minibatches=random_minibatches(X,Y,t,seed)
        for j in range(len(minibatches)):
            minibatch_X,minibatch_Y=minibatches[j]
            AL,caches=L_layer_forward(X,parameters)
            cost=compute_cost(AL,Y)
            #print(str(cost))
            grads=L_layer_backward(AL,Y,caches)
            parameters=update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 5 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 5 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


    

    
