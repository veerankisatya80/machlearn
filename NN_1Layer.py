import numpy as np

def __sigmoidderv(X,derv=False):
    if(derv == True):
        return X * (1 - X)

    return 1/(1+np.exp(-X))


#define our array

X = np.array([[0,1,1],[1,1,1],[1,0,1],[0,1,1]])

#output dataset
#y = np.array([0,1,1,0])[np.newaxis].T
y = np.array([[0],[1],[1],[0]])

#seed random numbers to make calculation deterministic

np.random.seed(1)

syn0 = 2 * np.random.random((3,1)) - 1


#iterate and calculate the
for iter in range(10000):
        l0 = X
        #forward propogation , multiply layer0 with Theta( syn0) weights and calculate the sigmoid 1/(1+e(-x))
        l1 = __sigmoidderv(np.dot(l0,syn0))

        #once layer 1 is calculated now backprogate , i.e find the error and adjust
        #error calculation
        l1_error = y - l1

        #calculate the derivative of l1
        delta1 = l1_error * __sigmoidderv(l1,True)

        #multiply delta1 to layer 0 :l0 to "adjust"
        syn0 += np.dot(l0.T,delta1)

print("Output after Training",l1)




