"""
Naive Bayes classifer
By: Minchan Kim
"""

import pandas as pd
import numpy as np

class Naive_Bayes():
    """
    Naive Bayes classifer
    
    Attributes:
        prior: P(Y)
        likelihood: P(X_j | Y)
    """
    
    def __init__(self):
        """
        Some initializations, if neccesary
        """
        self.model_name = 'Naive Bayes'
    
    
    def fit(self, X_train, y_train):
        """ 
        The fit function fits the Naive Bayes model based on the training data. 
        Here, we assume that all the features are **discrete** features. 
        
        X_train is a matrix or 2-D numpy array, representing training instances. 
        Each training instance is a feature vector. 

        y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        ---
        Parameters:
            X_train : list
                A list of feature vectors, each feature vector is a list.
            y_train : list
                A list of class labels.
        ---
        Return None
        """
        # Compute the prior distribution of all y labels
        self.prior = dict()
        for y in y_train:
            prior_key = f'Y = {y}'
            if prior_key in self.prior.keys():
                self.prior[prior_key] += 1
            else:
                self.prior[prior_key] = 1

        # Normalize the prior distribution
        for key in self.prior.keys():
            self.prior[key] /= len(y_train)

        # Compute the likelihood distribution of all X_j given Y
        self.likelihood = dict()
        for x, y in zip(np.array(X_train), y_train):
            for j in range(len(x)):
                likelihood_key = f'X{j} = {x[j]} | Y = {y}'
                if likelihood_key in self.likelihood.keys():
                    self.likelihood[likelihood_key] += 1
                else:
                    self.likelihood[likelihood_key] = 1

        # Normalize the likelihood distribution
        for key in self.likelihood.keys():
            self.likelihood[key] /= len(y_train)
        
        
    def ind_predict(self, x : list):
        """ 
        Predict the most likely class label of one test instance based on its feature vector x.
        ---
        Parameters:
            x : list
                A feature vector.
        ---
        Return the prediction of the instance.
        """
        best_label, best_prob = None, float('-inf')
    
        # Iterate through each class to compute its posterior log probability
        for y, prior_count in self.prior.items():
            cur_prob = prior_count
            
            # Sum log likelihoods for each feature given the class
            for j, feature_value in enumerate(x):
                likelihood_key = f'X{j} = {feature_value} | {y}'
                if likelihood_key in self.likelihood:
                    cur_prob *= self.likelihood[likelihood_key]
                else:
                    # Handle unknown feature values - Assume very small probability
                    cur_prob *= 10^-9
                    
            if cur_prob > best_prob:
                best_prob = cur_prob
                best_label = y
        
        return best_label[-1]
        
    
    def predict(self, X):
        """
        X is a matrix or 2-D numpy array, represnting testing instances. 
        Each testing instance is a feature vector. 
        ---
        Parameters:
            X : list
                A list of feature vectors, each feature vector is a list.
        ---
        Return the predictions of all instances in a list.
        """
        return [self.ind_predict(x) for x in np.array(X)]