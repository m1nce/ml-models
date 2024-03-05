import numpy as np
import pandas as pd

class Linear_Regression():
    def __init__(self, alpha = 1e-10 , num_iter = 10000, early_stop = 1e-50, intercept = True, init_weight = None):  
        """
        Initializes Linear Regression class.
        ---
        Parameters: 
            alpha: Learning Rate, defaults to 1e-10.
            num_iter: Number of Iterations to update coefficient with training data.
            early_stop: Constant control early_stop.
            intercept: Bool, If we are going to fit a intercept, default True.
            init_weight: Matrix (n x 1), input init_weight for testing.
        """
        self.model_name = 'Linear Regression'
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight  ### For testing correctness.
        self.coef = None
        self.loss = []
        

    def __str__(self):
        return self.model_name
    

    def fit(self, X_train, y_train):
        """
        Saves datasets to our model, and performs gradient descent.
        ---
        Parameters:
            X_train: Matrix or 2-D array. Input feature matrix.
            y_train: Matrix or 2-D array. Input target value.
        """
        self.X = np.mat(X_train)
        if isinstance(y_train, pd.core.series.Series):
            self.y = np.reshape(y_train.values, (-1, 1))
        elif y_train.ndim == 1:
            self.y = np.reshape(y_train, (-1, 1))
        else:
            self.y = y_train
        
        # adds column of all 1's to first column of X if intercept is True.
        if self.intercept:
            ones = np.ones((self.X.shape[0], 1))
            self.X = np.hstack((ones, self.X))
        
        if self.init_weight is not None:
            self.coef = self.init_weight
        else:
            # initializes coefficient with uniform from [-1, 1]
            self.coef = np.random.rand(self.X.shape[1]) * 2 - 1
            self.coef = np.reshape(self.coef, (-1, 1))

        # Call gradient_descent function to train.
        self.gradient_descent()
        

    def gradient(self):
        """
        Helper function to calculate the gradient of the cost function.
        Gradient: -2X^T(y - Xw)
        ---
        Returns the gradient of the cost function.
        """
        y_pred = np.dot(self.X, self.coef)
        y_pred = np.reshape(y_pred, (-1, 1))
        error = np.subtract(self.y, y_pred)

        # Return gradient of cost function
        return -2 * np.dot(self.X.T, error)
        

    def gradient_descent(self):
        """
        Performs gradient descent to find the model's optimal coefficients.
        Cost function: SE = \sum^{n}_{i=1}(Y_i - \hat{Y}_i)^2
        """
        for i in range(self.num_iter):
            if i == 0:
                pre_error = np.sum(np.power(np.dot(self.X, self.coef) - self.y, 2))
                
            # Calculate the gradient and temporary coefficient to compare
            temp_coef = self.coef - self.alpha * self.gradient()  

            # Calculate error for early stopping
            y_pred = np.dot(self.X, temp_coef)
            current_error = np.sum(np.power(y_pred - self.y, 2))

            ### This is the early stop, don't modify the following three lines.
            if (abs(pre_error - current_error) < self.early_stop) | (abs(abs(pre_error - current_error) / pre_error) < self.early_stop):
                self.coef = temp_coef
                return self
            
            # Update learning rate and coefficients based on error comparison
            if current_error < pre_error:
                self.coef = temp_coef
                # Increase learning rate if error decreases
                self.alpha *= 1.3 
            else:
                # Decrease learning rate if error increases
                self.alpha *= 0.9 

            # Update previous error for next iteration
            pre_error = min(pre_error, current_error)
            # Add loss to loss list we create
            self.loss.append(min(pre_error, current_error))

            if i % 1000 == 0:
                print('Iteration: ' + str(i))
                print('Coef:      ' + str(self.coef))
                print('Loss:      ' + str(current_error))
                print('Alpha:     ' + str(self.alpha)) 
    
        return self


    def ind_predict(self, x: list):
        """
        Predict the value based on its feature vector x.
        ---
        Parameters:
            x: Matrix, array or list. Input feature point.
        ---
        Returns a prediction of given data point.
        """
        x = np.matrix(x)
        return np.array(np.dot(x, self.coef)).flatten()


    def predict(self, X):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            ---
            Parameters:
                X: Matrix, array or list. Input feature point.
            ---
            Returns a prediction of the given data matrix.
        """
        X = np.matrix(X)

        # Adds ones if intercept is True.
        if self.intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
            
        # Calls ind_predict for each value in the X matrix.
        predictions = np.array([self.ind_predict(x) for x in X]).flatten()
        return predictions

