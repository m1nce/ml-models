import pandas as pd
import numpy as np

"""
Decision Tree classifier
By: Minchan Kim
"""

class DecisionTree():
    def __init__(self, max_depth = 1000, size_allowed = 1, n_features = None, n_split = None):
        """
        Initializations for class attributes.
        ---
        Parameters:
            max_depth: Max depth allowed for the tree
            size_allowed : Min_size split, smallest size allowed for split 
            n_features: Number of features to use during building the tree.(Random Forest)
            n_split:  Number of split for each feature. (Random Forest)
        """
        self.root = None
        self.max_depth = max_depth
        self.size_allowed = size_allowed
        self.n_features = n_features
        self.n_split = n_split
    
    
    class Node():
        """
            Node Class for the building the tree.

            Attribute: 
                threshold: The threshold like if x1 < threshold, for spliting.
                feature: The index of feature on this current node.
                left: Pointer to the node on the left.
                right: Pointer to the node on the right.
                pure: Bool, describe if this node is pure.
                predict: Class, indicate what the most common Y on this node.

        """
        def __init__(self, threshold = None, feature = None):
            """
                Initializations for class attributes.
            """
            self.threshold = threshold
            self.feature = feature
            # Initialize left and right children to None
            self.left = None 
            self.right = None
            self.pure = False 
            self.depth = 0
            self.predict = None
    
    
    def entropy(self, lst):
        """
            Function Calculate the entropy given lst.
            
            Attributes: 
                entro: variable store entropy for each step.
                classes: all possible classes. (without repeating terms)
                counts: counts of each possible classes.
                total_counts: number of instances in this lst.
                
            lst is vector of labels.
        """
        entro = 0
        value, counts = np.unique(lst, return_counts=True)
        total_counts = sum(counts)
        for count in counts:
            prob = count / total_counts
            entro -= prob * np.log2(prob)
        return entro


    def information_gain(self, lst, values, threshold):
        """
        
            Function Calculate the information gain, by using entropy function.
            
            lst is vector of labels.
            values is vector of values for individule feature.
            threshold is the split threshold we want to use for calculating the entropy.
            
            
            TODO:
                5. Modify the following variable to calculate the P(left node), P(right node), 
                   Conditional Entropy(left node) and Conditional Entropy(right node)
                6. Return information gain.
                
                
        """
        left_idx = values <= threshold
        right_idx = values > threshold

        left_y = lst[left_idx]
        right_y = lst[right_idx]

        left_prop = len(left_y) / len(lst)
        right_prop = len(right_y) / len(lst)

        left_entropy = self.entropy(left_y)    
        right_entropy = self.entropy(right_y)   

        parent_entropy = self.entropy(lst)
        info_gain = parent_entropy - (left_prop * left_entropy + right_prop * right_entropy)
        return info_gain
    
    
    def find_rules(self, data):
        
        """
        
            Helper function to find the split rules.
            
            data is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

        """
        n, m = data.shape       
        rules = []        
        for i in range(m):          
            unique_value = np.unique(data[:, i])       
            diff = (unique_value[:-1] + unique_value[1:]) / 2  # Mid points
            rules.append(diff)             
        return rules
    
    def next_split(self, data, label):
        """
            Helper function to find the split with most information gain, by using find_rules, and information gain.
            
            data is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 
            
            label contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        rules = self.find_rules(data)             
        max_info = -np.inf          
        num_col = None          
        threshold = None       
        entropy_y = self.entropy(label)      
        index_col = range(data.shape[1])

        for i in index_col:
            for rule in rules[i]:
                info_gain = self.information_gain(label, data[:, i], rule)
                if info_gain > max_info:
                    max_info = info_gain
                    num_col = i
                    threshold = rule
        return threshold, num_col
        
        
    def build_tree(self, X, y, depth):
            """
                Helper function for building the tree.
            """
            if depth > self.max_depth or len(y) <= self.size_allowed:
                node = self.Node()
                node.pure = True
                node.predict = np.bincount(y).argmax()
                return node
            
            if len(np.unique(y)) == 1:
                node = self.Node()
                node.predict = y[0]
                node.pure = True
                return node
                
            threshold, feature = self.next_split(X, y)
            if threshold is None:
                node = self.Node()
                node.pure = True
                node.predict = np.bincount(y).argmax()
                return node
                
            node = self.Node(threshold, feature)
            left_index = X[:, feature] <= threshold
            right_index = X[:, feature] > threshold
            
            if not np.any(left_index) or not np.any(right_index):  
                node.predict = np.bincount(y).argmax()
                node.pure = True
                return node
            
            node.left = self.build_tree(X[left_index], y[left_index], depth + 1)
            node.right = self.build_tree(X[right_index], y[right_index], depth + 1)
            
            return node
    

    def fit(self, X, y): 
        """
            The fit function fits the Decision Tree model based on the training data. 
            
            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        self.root = self.build_tree(X, y, 1)
        return self
            

    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        cur = self.root  
        while not cur.pure:  
            if inp[cur.feature] <= cur.threshold:  
                cur = cur.left
            else:
                cur = cur.right
        return cur.predict
    

    def predict(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Return the predictions of all instances in a list.
        """
        result = [self.ind_predict(inp[i]) for i in range(inp.shape[0])]
        return result
    
