# Import necessary libraries
import pandas as pd
import numpy as np

def z_standardize(X_inp):
    """
        Z-score Standardization.
        Standardizes the feature matrix, and stores the standarization rule.
        ----------------
        Parameter:
            X_inp: Matrix or 2-D array. Input feature matrix.
        ----------------
        Return:
            Standardized feature matrix.
    """
    toreturn = X_inp.copy()
    for i in range(X_inp.shape[1]):
        std = np.std(X_inp[:, i])               # ------ Find the standard deviation of the feature
        mean = np.mean(X_inp[:, i])             # ------ Find the mean value of the feature
        temp = []
        for j in np.array(X_inp[:, i]):
            temp.append((j - mean) / std)       # ------ Standardize the feature
        toreturn[:, i] = temp
    return toreturn


def sigmoid(x):
    """ 
        Sigmoid Function
        ----------------
        Parameters:
            x: int. Input value.
        ----------------
        Return:
            transformed x.
    """
    return 1 / (1 + np.exp(-x))


class Logistic_Regression():
    def __init__(self):
        """
        Initialize the Logistic Regression model.
        """
        self.model_name = 'Logistic Regression'
        self.theta = None
        self.b = 0


    def __str__(self):
        """
        Return the model name.
        """
        return self.model_name
    

    def fit(self, X_train, y_train):
        """
            Saves the datasets in our model, and normalizes to y_train
            ----------------
            Parameter:
                X_train: Matrix or 2-D array. Input feature matrix.
                Y_train: Matrix or 2-D array. Input target value.
        """
        self.X = X_train
        self.y = y_train
        
        count = 0
        uni = np.unique(y_train)
        for y in y_train:
            if y == min(uni):
                self.y[count] = -1
            else:
                self.y[count] = 1
            count += 1        
        
        n,m = X_train.shape
        self.theta = np.zeros(m)
        self.b = 0
    

    def gradient(self, X_inp, y_inp, theta, b):
        """
            Calculate the gradient of Weight and Bias, given sigmoid_yhat, true label, and data

            Parameter:
                X_inp: Matrix or 2-D array. Input feature matrix.
                y_inp: Matrix or 2-D array. Input target value.
                theta: Matrix or 1-D array. Weight matrix.
                b: int. Bias.

            Return:
                grad_theta: gradient with respect to theta
                grad_b: gradient with respect to b
        """
        m = len(y_inp)
        y_hat = sigmoid(np.dot(X_inp, theta) + b)
        error = np.subtract(y_hat, y_inp)
        grad_theta = np.dot(X_inp.T, error) / m
        grad_b = np.sum(error) / m
        return grad_theta, grad_b


    def gradient_descent_logistic(self, alpha, num_pass, early_stop=0, standardized = True):
        """
            Logistic Regression with gradient descent method

            Parameter:
                alpha: (Hyper Parameter) Learning rate.
                num_pass: Number of iteration
                early_stop: (Hyper Parameter) Least improvement error allowed before stop. 
                            If improvement is less than the given value, then terminate the function and store the coefficents.
                            default = 0.
                standardized: bool, determine if we standardize the feature matrix.
                
            Return:
                self.theta: theta after training
                self.b: b after training
        """
        if standardized:
            self.X = z_standardize(self.X)

        for i in range(num_pass):    
            grad_theta, grad_b = self.gradient(self.X, self.y, self.theta, self.b)
            temp_theta = self.theta - alpha * grad_theta
            temp_b = self.b - alpha * grad_b

            y_hat = sigmoid(np.dot(self.X, temp_theta) + temp_b)
            temp_error = -np.mean(self.y * np.log(y_hat) + (1 - self.y) * np.log(1 - y_hat))

            if i > 0 and abs(prev_error - temp_error) < early_stop:
                break

            self.theta = temp_theta
            self.b = temp_b
            prev_error = temp_error
        return self.theta, self.b
    

    def predict_ind(self, x: list):
        """
            Predict the most likely class label of one test instance based on its feature vector x.

            Parameter:
            x: Matrix, array or list. Input feature point.
            
            Return:
                p: prediction of given data point
        """
        prob = sigmoid(np.dot(x, self.theta) + self.b)
        return 1 if prob >= 0.5 else 0
    

    def predict(self, X):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Parameter:
            x: Matrix, array or list. Input feature point.
            
            Return:
                p: prediction of given data matrix
        """
        return np.array([self.predict_ind(x) for x in X])