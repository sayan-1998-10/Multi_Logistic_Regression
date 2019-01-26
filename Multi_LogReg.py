import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
epsilon = 1e-5

def sigmoid(z): #hypothesis function
    return 1/(1+np.exp(-z))
def dot_product(X,theta):
    return np.dot(X,theta)

def grad_descent_runner(iterations,X,Y,theta,h):

    for i in range(iterations):
         theta= grad_descent(X,Y,theta,h)

    return theta
def grad_descent(X,Y,theta,h):
    alpha = 0.001
    grad = (1/len(Y))*(X.T @ (h-Y))

    theta = theta - alpha*grad
    return theta
def classifier(theta,X):
    #10 different classifiers for 10 different classes (0-9 digits)
    #I have used sigmoid function(0-->1)ra nge
    h1= sigmoid(dot_product(X,theta[0,:].reshape(401,1)))
    h2 = sigmoid(dot_product(X,theta[1,:].reshape(401,1)))
    h3 = sigmoid(dot_product(X,theta[2,:].reshape(401,1)))
    h4 = sigmoid(dot_product(X,theta[3,:].reshape(401,1)))
    h5 = sigmoid(dot_product(X,theta[4,:].reshape(401,1)))
    h6 = sigmoid(dot_product(X,theta[5,:].reshape(401,1)))
    h7 = sigmoid(dot_product(X,theta[6,:].reshape(401,1)))
    h8 = sigmoid(dot_product(X,theta[7,:].reshape(401,1)))
    h9 = sigmoid(dot_product(X,theta[8,:].reshape(401,1)))
    h10= sigmoid(dot_product(X,theta[9,:].reshape(401,1)))
    h = np.array([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10])
    #print(h)  ---->>printing the hypothsis function
    #finding out the max probability out of all classes
    max = h[0]
    for i in range(len(h)):
        if(h[i]>max):
            max=h[i]
            j=i
    return [max,j]

def main():
    #loading data and separating X and Y and then adding bias term to X
    #creatin weights vector -'theta' (initially filled with zeros)
    data = sio.loadmat("C:/Users/europ/Desktop/machine-learning-ex3/ex3/ex3data1.mat")
    X = data['X']
    Y = data['y']
    ones = np.ones((5000,1))
    X = np.append(X,ones,axis=1)
    theta = np.zeros((10,401))  #shape is 10*401 since 10 classes and 401
                                #pixels as features


    h1= sigmoid(dot_product(X,theta[0,:].reshape(401,1)))
    h2 = sigmoid(dot_product(X,theta[1,:].reshape(401,1)))
    h3 = sigmoid(dot_product(X,theta[2,:].reshape(401,1)))
    h4 = sigmoid(dot_product(X,theta[3,:].reshape(401,1)))
    h5 = sigmoid(dot_product(X,theta[4,:].reshape(401,1)))
    h6 = sigmoid(dot_product(X,theta[5,:].reshape(401,1)))
    h7 = sigmoid(dot_product(X,theta[6,:].reshape(401,1)))
    h8 = sigmoid(dot_product(X,theta[7,:].reshape(401,1)))
    h9 = sigmoid(dot_product(X,theta[8,:].reshape(401,1)))
    h10= sigmoid(dot_product(X,theta[9,:].reshape(401,1)))

    iterations = 1000
    #new values of weights for each class
    #shape of each theta[i,:] is (401,) for 400 different pixel features and 1 bias weight
    theta[0,:]=grad_descent_runner(iterations,X,(Y==10).astype(int),theta[0,:].reshape(401,1),h1).reshape(401,)
    theta[1,:]=grad_descent_runner(iterations,X,(Y==1).astype(int),theta[1,:].reshape(401,1),h2).reshape(401,)
    theta[2,:]=grad_descent_runner(iterations,X,(Y==2).astype(int),theta[2,:].reshape(401,1),h3).reshape(401,)
    theta[3,:]=grad_descent_runner(iterations,X,(Y==3).astype(int),theta[3,:].reshape(401,1),h4).reshape(401,)
    theta[4,:]=grad_descent_runner(iterations,X,(Y==4).astype(int),theta[4,:].reshape(401,1),h5).reshape(401,)
    theta[5,:]=grad_descent_runner(iterations,X,(Y==5).astype(int),theta[5,:].reshape(401,1),h6).reshape(401,)
    theta[6,:]=grad_descent_runner(iterations,X,(Y==6).astype(int),theta[6,:].reshape(401,1),h7).reshape(401,)
    theta[7,:]=grad_descent_runner(iterations,X,(Y==7).astype(int),theta[7,:].reshape(401,1),h8).reshape(401,)
    theta[8,:]=grad_descent_runner(iterations,X,(Y==8).astype(int),theta[8,:].reshape(401,1),h9).reshape(401,)
    theta[9,:]=grad_descent_runner(iterations,X,(Y==9).astype(int),theta[9,:].reshape(401,1),h10).reshape(401,)

    #print(theta[1].reshape(401,1)) if you want check the theta for each class after training the classifiers
    max,j=classifier(theta,X[2343])
    print("class:h%d"%j)


if __name__ == '__main__':
    main()
