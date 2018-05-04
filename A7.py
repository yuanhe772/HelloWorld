import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# # Read data
# data = np.genfromtxt('wbdc.txt', delimiter=',')     
# size = 400

# # Primal data set
# train, test = data[0:size, :], data[size:, :]# divide into trn and tst
# X_O_trn, Y_trn = train[:, 2:], train[:,1]
# X_O_tst, Y_tst = test[:, 2:], test[:,1]
X_O_trn = np.genfromtxt('X-trn-400.csv', delimiter = ',')
Y_trn = np.genfromtxt('Y-trn-400.csv', delimiter = ',')
X_O_tst = np.genfromtxt('X-tst-400.csv', delimiter = ',')
Y_tst = np.genfromtxt('Y-tst-400.csv', delimiter = ',')


# Feature-Normalizing
X_trn = np.array(X_O_trn)
X_max_feature = X_trn.max(axis = 0)
X_trn /= X_max_feature # normalized X-trn
X_tst = np.array(X_O_tst)
X_tst /= X_max_feature # normalized X-tst

# initial W1, W2:
mu, sigma = 0, 0.01 # mean and standard deviation
# W1 = np.random.normal(mu, sigma, (20,30))
# W2 = np.random.normal(mu, sigma, (2,20))


W1 = np.full((20,30), 0.1)
W2 = np.full((2,20), 0.1)

def softmax(ip):
    ip = ip - np.amax(ip, axis = 1, keepdims = True)
    return np.divide(np.exp(ip),np.sum(np.exp(ip),axis=1)[:,None])

def crossLoss(label, score):
    return -(label*np.log(score) + (1-label)*np.log(1-score))

def backward(W1, W2, str_ACT):
    label = Y_trn
    Input1 = X_trn

    
    # Forward Pass:
    
    ## 1st layer:
    ### dot-product with W1
    W1_Input1 = Input1.dot(W1.T)  


    ### Activate
    if str_ACT == "sig":
        h1 = 1/(1+np.exp(-W1_Input1))
    else:
        h1 = np.maximum(W1_Input1,0)


        
    ## 2nd layer:
    W2_h1 = h1.dot(W2.T)
    ### Activate
    score = softmax(W2_h1)[:, 0]

    

    # Loss Func:
    loss = np.average(crossLoss(label, score))



    # backward Pass:(Gradients)
    ## L Gradient:
    S1_g = (-label*(1-score) + (1-label)*score)[:, None]/len(X_trn)

    ## W2 Gradient:
    L_S = np.concatenate((S1_g, -S1_g), axis = 1)
    

 
    S_W2 = h1
    L_W2 = (L_S.T).dot(S_W2)



    L_h1 = np.matmul(L_S, W2)

    print (L_h1)
    sys.exit()


    ## W1 Gradient:    
    ### Activation function:
    if str_ACT == "sig":
        ACT_g = h1 * (1 - h1)
        L_W1_Input = L_h1*ACT_g

    else:
        L_W1_Input = L_h1
        L_W1_Input[W1_Input1<0] = 0

 #        ACT_g = h1

 #       for i in range(len(ACT_g)):
 #           for j in range(len(ACT_g[0])):
 #               if ACT_g[i, j] > 0:
 #                   ACT_g[i, j] = 1
 #               else:
 #                   ACT_g[i, j] = 0
                        
 #   L_h1 = np.transpose(L_S).dot(W2) 
 #   L_W1 = L_11*np.transpose(ACT_g)
   
    L_W1 = (L_W1_Input.T).dot(Input1)

    # L_W1 = L_W1.T

    # Return value
    return (L_W1, L_W2)

def predict(W1, W2, X, Y, str_ACT):
    label = Y
    Input1 = np.transpose(X)

    # Forward Pass:
    ## 1st layer:
    ### dot-product with W1
    W1_Input1 = W1.dot(Input1)  
    ### Activate
    if str_ACT == "sig":
        h1 = 1/(1+np.exp(-W1_Input1))
    else:
        h1 = np.maximum(W1_Input1,0)
        
    ## 2nd layer:
    W2_h1 = W2.dot(h1)

    ### Activate
    score = softmax(W2_h1)[0]  
    
    # Accuracy rate
    mistake = 0
    for i in range(len(score)):
        if score[i]<=0.5:
            score[i] = 0
        else:
            score[i] = 1
        if score[i] != Y_trn[i]:
            mistake += 1
    return (1-mistake/len(Y_trn))


def gradientDescent(W1, W2, res, lr):
    W1 = W1 - lr*res[0]
    W2 = W2 - lr*res[1]
    return (W1, W2)

#Initialize parameters 
epochs = 10000

results = []

#Train and test NN
# for lr in [1,0.1,0.01,0.001,0.1/(np.sqrt(epochs)+1)]:
for lr in [0.1]:
    for i in range(10000):             
#             lr = 0.1/(np.sqrt(i)+1)
#             lr = 0.01      
            # Back Prop:
        delta_res = backward(W1, W2, "")
        # Gradient Descent:
        W_new = gradientDescent(W1, W2, delta_res, lr)
            # Predict:
        W1 = W_new[0]
        W2 = W_new[1]
        accuracy = predict(W1, W2, X_tst, Y_tst, "")
        results.append(accuracy)
#         print(accuracy)
        
        
#    print("finish")
#    print(results)