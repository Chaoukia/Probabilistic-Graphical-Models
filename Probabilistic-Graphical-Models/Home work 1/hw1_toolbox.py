# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:50:30 2018

@author: Jarvis
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def error(y, y_pred):
    return (y != y_pred).sum()/len(y)

class LDA:
    
    def __init__(self):
        
        """
        Attributes
        -----------
        pi : float, Bernoulli parameter p(y = 1)
        mu_0 : (p,) np.array, mean of X|{y = 0}
        mu_1 : (p,) np.array, mean of X|{y = 1}
        Sigma : (p, p) np.array, shared covariance matrix of X|{y = 0} and X|{y = 1}
        a : float
        b : (p,) np.array
        
        Note
        -----------
        p : int, space dimension of the data
        a & b: Logistic parameters We get when computing the MLE.
        """
        self.pi = None
        self.mu_0 = None
        self.mu_1 = None
        self.Sigma = None
        self.a = None
        self.b = None
        
    def fit(self, X, y):
        """
        Description
        -----------
        Update the attributes
        
        Parameters
        -----------
        X : (n, p) np.array, data matrix
        y : (n,) np.array, labels in {0, 1}
        """
        n = X.shape[0] # Number of observations

        sum_1 = np.sum(y) # Number of observations of class 1
        sum_0 = n - sum_1   # Number of observations of class 0

        self.pi = sum_1/n       # The estimated parameter pi
        self.mu_1 = np.dot(X.T, y)/sum_1
        self.mu_0 =np.dot(X.T, (1 - y))/sum_0
        Sigma_sample_1 = np.dot((X - self.mu_1).T, y[:,np.newaxis]*(X - self.mu_1))/sum_1
        Sigma_sample_0 = np.dot((X - self.mu_0).T, (1 - y)[:,np.newaxis]*(X - self.mu_1))/sum_0
        self.Sigma = (sum_1*Sigma_sample_1 + sum_0*Sigma_sample_0)/n
        Precision = np.linalg.inv(self.Sigma)
        self.a = -(np.dot(self.mu_1.T, np.dot(Precision, self.mu_1)) - np.dot(self.mu_0.T, np.dot(Precision, self.mu_0)))/2 + np.log(self.pi/(1 - self.pi))
        self.b = np.dot(Precision, self.mu_1 - self.mu_0)
        
    def predict_proba(self, X):
        """
        Description
        -----------
        Compute the probabilities p(y = 0|X) and p(y = 1|X)
        
        Parameters
        -----------
        X : (n, p) np.array, data matrix
        
        Returns
        -----------
        y_proba : (n, 2) np.array, 1st column p(y = 0|X) and 2nd column p(y = 1|X)
        
        """
        n = X.shape[0]
        y_proba = np.zeros((n,  2))
        y_proba[:, 1] = sigmoid(self.a + np.dot(X, self.b))
        y_proba[:, 0] = 1 - y_proba[:, 1]
        return y_proba
    
    def predict(self, X):
        """
        Description
        -----------
        Compute the labels
        
        Parameters
        -----------
        X : (n, p) np.array, data matrix
        
        Returns
        -----------
        y_pred: (n,) np.array, predicted labels of data matrix X
        
        """
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis = 1)
        return y_pred
    
    def plot_line(self, X, y, title = "LDA", save = False, path = ""):
        """
        Description
        -----------
        Scatter plot of data X along with plotting the line p(y=1|X) = 0.5
        
        Parameters:
        -----------
        X : (n, p) np.array, data matrix
        y : (n,) np.array, labels in {0, 1}
        title : String, the plot titme
        save : Boolean, if True save the figure
        path : String, only used when save is True, the path of the figure to save
        """
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c = "b")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c = "r")
        plt.title(title)
        plt.ylim((min(X[:, 1]), max(X[:, 1])))
        x, y = np.meshgrid(np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 100), np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5, 100))
        z = self.a + self.b[0]*x + y*self.b[1]
        plt.ylim((X[:, 1].min() - 2, X[:, 1].max() + 2))
        plt.contour(x, y, z, 0)
        plt.contourf(x, y, z, 0, alpha = 0.5, cmap = plt.cm.inferno)
        if save:
            plt.savefig(path)
        
class QDA:
    
    def __init__(self):
        
        """
        Attributes
        -----------
        pi : float, Bernoulli parameter p(y = 1)
        mu_0 : (p,) np.array, mean of X|{y = 0}
        mu_1 : (p,) np.array, mean of X|{y = 1}
        Sigma : (p, p) np.array, shared covariance matrix of X|{y = 0} and X|{y = 1}
        a : float
        b : (p,) np.array
        
        Note
        -----------
        p : int, space dimension of the data
        a & b: Logistic parameters We get when computing the MLE.
        """
        self.pi = None
        self.mu_0 = None
        self.mu_1 = None
        self.Sigma_0 = None
        self.Sigma_1 = None
        self.a = None
        self.b = None
        self.c = None
        
    def fit(self, X, y):
        """
        Description
        -----------
        Update the attributes
        
        Parameters
        -----------
        X : (n, p) np.array, data matrix
        y : (n,) np.array, labels in {0, 1}
        """
        n = X.shape[0] # Number of observations

        sum_1 = np.sum(y) # Number of observations of class 1
        sum_0 = n - sum_1   # Number of observations of class 0

        self.pi = sum_1/n       # The estimated parameter pi
        self.mu_1 = np.dot(X.T, y)/sum_1
        self.mu_0 =np.dot(X.T, (1 - y))/sum_0
        self.Sigma_1 = np.dot((X - self.mu_1).T, y[:,np.newaxis]*(X - self.mu_1))/sum_1
        self.Sigma_0 = np.dot((X - self.mu_0).T, (1 - y)[:,np.newaxis]*(X - self.mu_1))/sum_0
        Precision_1 = np.linalg.inv(self.Sigma_1)
        Precision_0 = np.linalg.inv(self.Sigma_0)
        self.a = -(np.dot(self.mu_1.T, np.dot(Precision_1, self.mu_1)) - np.dot(self.mu_0.T, np.dot(Precision_0, self.mu_0)))/2 + np.log(self.pi/(1 - self.pi)) + np.log(np.linalg.det(Precision_1)/np.linalg.det(Precision_0))/2
        self.b = np.dot(Precision_1, self.mu_1) - np.dot(Precision_0, self.mu_0)
        self.c = (Precision_0 - Precision_1)/2
        
    def predict_proba(self, X):
        """
        Description
        -----------
        Compute the probabilities p(y = 0|X) and p(y = 1|X)
        
        Parameters
        -----------
        X : (n, p) np.array, data matrix
        
        Returns
        -----------
        y_proba : (n, 2) np.array, 1st column p(y = 0|X) and 2nd column p(y = 1|X)
        
        """
        n = X.shape[0]
        y_proba = np.zeros((n,  2))
        y_proba[:, 1] = sigmoid(self.a + np.dot(X, self.b) + np.sum(np.dot(X, self.c)*X, axis = 1))
        y_proba[:, 0] = 1 - y_proba[:, 1]
        return y_proba
    
    def predict(self, X):
        """
        Description
        -----------
        Compute the labels
        
        Parameters
        -----------
        X : (n, p) np.array, data matrix
        
        Returns
        -----------
        y_pred: (n,) np.array, predicted labels of data matrix X
        
        """
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis = 1)
        return y_pred
    
    def plot_line(self, X, y, title = "QDA", save = False, path = ""):
        """
        Description
        -----------
        Scatter plot of data X along with plotting the line p(y=1|X) = 0.5
        
        Parameters:
        -----------
        X : (n, p) np.array, data matrix
        y : (n,) np.array, labels in {0, 1}
        """
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c = "b")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c = "r")
        plt.title(title)
        x, y = np.meshgrid(np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 100), np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5, 100))
        z = self.a + self.b[0]*x + self.b[1]*y + self.c[0, 0]*(x**2) + (self.c[0, 1] + self.c[1, 0])*x*y + self.c[1, 1]*(y**2)
        plt.ylim((X[:, 1].min() - 2, X[:, 1].max() + 2))
        plt.contour(x, y, z, 0)
        plt.contourf(x, y, z, 0, cmap = plt.cm.inferno, alpha = 0.5)
        if save:
            plt.savefig(path)
        
class LogisticRegression():
    """
    Description
    -----------
    Class of a pure logistic regression without any regularization

    Parameters:
    -----------
    max_iter : the maximum  iterations 
    tol : the relative tolerance for the loglikelihood convergence
    rng : random number generation for result reproduction
    weights : the weights of our model
    
    """
    
    def __init__(self, max_iter = 1000 , tol =1e-6, rng = np.random.RandomState(1)):
        
        self.max_iter = max_iter #max iterations 
        self.tol = tol # the relative tolerance for the loglikelihood convergence
        self.rng = rng # random number generation for result reproduction
        self.weights = None # the weights of our model
        self.loglike_history = [] # the evolution of the loglikelihood to check convergence
        
    def loglike(self, X, y ,beta):
        
        """loglikelihood computations to check convergence"""
        
        linear = np.dot(X,beta)
        return np.dot(y.T, np.log(sigmoid(linear))) + np.dot(1-y.T,np.log(sigmoid(-linear)))
    
    def IRLS_update(self, X, y, beta):
        
        """the IRLS update"""
        
        nu = np.dot(X,beta)
        mu = sigmoid(nu)
        s = np.maximum(mu*(1-mu), 1e-6)
        z = s*nu + (y - mu)
        A, b = np.dot(X.T, s.reshape(-1,1)*X), np.dot(X.T, z)
        return np.linalg.solve(A,b)
    
    def fit(self, X, y):
        """
        Description
        -----------
        fit on training observations
        
        Parameters:
        -----------
        X : (n, p) np.array, training data matrix
        Y : (n, ) np.array, training labels
        """

        n,p = X.shape
        X_intercept = np.concatenate((np.ones((n,1)), X), axis=1)
        self.weights = self.rng.normal(0, 1e-3, p+1)
        convergence = False
        self.iter = 0
        
        while (not convergence):
            
            weights_t = self.weights
            
            # we compute the loglikelihood, the gradient and the hessian
            logl = self.loglike(X_intercept, y, weights_t)

            #we update the weights 
            self.weights = self.IRLS_update(X_intercept, y, weights_t)
            
            
            #we check the convergence of our algorithm
            convergence = (np.abs(logl - self.loglike(X_intercept, y, self.weights)) <= self.tol).all() or (self.iter > self.max_iter)
            self.iter += 1
        
        return self.weights
    
    def predict(self, X):
        """
        Description
        -----------
        Predict on new observations
        
        Parameters:
        -----------
        X : (n, p) np.array, new data matrix
        """
            
        n,p = X.shape
        X_intercept = np.concatenate((np.ones((n,1)), X), axis=1)
        return np.where(np.dot(X_intercept, self.weights) >= 0, 1 , 0)
    
    
    
    def plot_line(self, X, y, title = "Logistic_Regression", save = False, path = ""):
        """
        Description
        -----------
        Scatter plot of data X along with plotting the line p(y=1|X) = 0.5
        
        Parameters:
        -----------
        X : (n, p) np.array, data matrix
        y : (n,) np.array, labels in {0, 1}
        title : String, the plot titme
        save : Boolean, if True save the figure
        path : String, only used when save is True, the path of the figure to save
        """
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c = "b")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c = "r")
        plt.title(title)
        plt.ylim((min(X[:, 1]), max(X[:, 1])))
        x, y = np.meshgrid(np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 100), np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5, 100))
        z = self.weights[0] + self.weights[1]*x + y*self.weights[2]
        plt.ylim((X[:, 1].min() - 2, X[:, 1].max() + 2))
        plt.contour(x, y, z, 0)
        plt.contourf(x, y, z, 0, alpha = 0.5, cmap = plt.cm.inferno)
        if save:
            plt.savefig(path)
        
class LinearRegression():
    """
    Description
    -----------
    Class of a pure linear regression without any regularization    
    
    Parameters:
    -----------
    weights : the weights of our model        
    
    """
    def __init__(self):
        
        self.weights = None 

    def fit(self, X, y):
        """
        Description
        -----------
        fit on training observations
        
        Parameters:
        -----------
        X : (n, p) np.array, training data matrix
        Y : (n, ) np.array, training labels
        
        """

        n,p = X.shape
        X_intercept = np.concatenate((np.ones((n,1)), X), axis=1)
        self.weights = np.dot(np.linalg.pinv(X_intercept), y)

        return self.weights

    def predict(self, X):
        """
        Description
        -----------
        Predict on new observations
        
        Parameters:
        -----------
        X : (n, p) np.array, new data matrix
        """
        n,p = X.shape
        X_intercept = np.concatenate((np.ones((n,1)), X), axis=1)
        return np.where(np.dot(X_intercept, self.weights)>=0.5, 1,0)


    def plot_line(self, X, y, title = "Linear_Regression", save = False, path = ""):
        """
        Description
        -----------
        Scatter plot of data X along with plotting the line p(y=1|X) = 0.5

        Parameters:
        -----------
        X : (n, p) np.array, data matrix
        y : (n,) np.array, labels in {0, 1}
        title : String, the plot titme
        save : Boolean, if True save the figure
        path : String, only used when save is True, the path of the figure to save
        """
        
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c = "b")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c = "r")
        plt.title(title)
        plt.ylim((min(X[:, 1]), max(X[:, 1])))
        x, y = np.meshgrid(np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 100), np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5, 100))
        z = self.weights[0]-0.5 + self.weights[1]*x + y*self.weights[2]
        plt.ylim((X[:, 1].min() - 2, X[:, 1].max() + 2))
        plt.contour(x, y, z, 0)
        plt.contourf(x, y, z, 0, alpha = 0.5, cmap = plt.cm.inferno)
        if save:
            plt.savefig(path)
        
        
    