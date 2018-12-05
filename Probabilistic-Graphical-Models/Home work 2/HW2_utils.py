import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2





#Kmeans class

class Kmeans():
    
    """Class for the Kmeans algorithm"""
    
    def __init__(self, k, random_seed=1):
        '''
        Attributes:
        
        k_: integer
            number of clusters
        centers_: np.array
            array containing the cluster centers
        distorition_: float
            the distortion measure of the data  
        labels_: (n, ) np.array
            labels for data points
        '''
        self.k_ = k
        self.centers_ = None
        self.distortion_ = None
        self.labels_ = None
        self.iterations_ = 0
        self.random_seed_ = random_seed
        
    
    def compute_distortion(self, X, labels, centers):
        
        '''
        compute the distortion measure
        returns : float
        '''
        return np.sum(np.linalg.norm(X - centers[labels])**2)
    
    def find_labels(self, X, centers):
        '''
        find the labels that minimizes the distortion
        returns : np.array (n, )
        '''
        return np.argmin(np.linalg.norm((centers - X[:,None,:]), axis = 2), axis = 1)
        
    def fit(self, X, max_iter=100):
        """ Find the labels that minimize the distortion
            on the data knowing the number of clusters
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self
        """
        
        n,p = X.shape
        rng = np.random.RandomState(self.random_seed_)
        
        
        self.initial_centers = X[rng.choice(np.arange(n), size=self.k_, replace=False)]
        self.initial_labels = self.find_labels(X, self.initial_centers)
        
        self.centers_, self.labels_ = self.initial_centers.copy(), self.initial_labels.copy()
        
        self.distortion_ = self.compute_distortion(X, self.labels_, self.centers_)
        self.iterations_ = 0
                                                                           
        converge = False                              
        while not(converge):
                                
            self.iterations_ += 1
            
            old_distortion = self.distortion_
            
            for cluster in range(self.k_) : self.centers_[cluster] = np.mean(X[self.labels_ == cluster], axis=0)
            self.labels_ = self.find_labels(X, self.centers_)
            
            self.distortion_ = self.compute_distortion(X , self.labels_, self.centers_)
            
            converge = np.isclose(old_distortion, self.distortion_, rtol=1e-5) or (self.iterations_ > max_iter)
            
            


            
# EM - Gaussian mixture model            
            
class GMM_isotropic():
    
    """class for the isotropic Gaussian mixture model"""
    
    def __init__(self, k, random_seed=1):
        '''
        Attributes:
        
        k_: integer
            number of components
        mu_: np.array 
            means of our gaussian vectors
        sigma_: np.array
            the sigmas of our isotropic guassians
        Tau_i_k_: (n, K) np.array
            conditional probabilities z|x for all data points 
        labels_: (n, ) np.array
            labels for data points
        '''
        self.k_ = k
        self.mus_ = None
        self.sigmas_ = None
        self.Tau_i_k_ = None
        self.pi_k_ = None
        self.labels_ = None
        self.iterations_ = 0
        self.random_seed_ = random_seed
        self.expectation_ = np.inf
        
    
    def compute_Tau_i_k(self, X, mus, sigmas, pi_k):
        '''Compute the conditional probability matrix Tau_i_k
        shape: (n, K)
        '''
        n, p = X.shape
        Tau_i_k = np.zeros((n ,self.k_))
        for i in range(self.k_) : Tau_i_k[:,i] = multivariate_normal.pdf(X, mus[i], sigmas[i]*np.eye(p))
        Tau_i_k *= pi_k 
        Tau_i_k /= np.sum(Tau_i_k, axis= 1, keepdims=True)
        return Tau_i_k

            
    def E_step(self, X, mus, sigmas, pi_k, Tau_i_k):
        '''Compute the expectation to check convergence'''
        
        n, p = X.shape
        multinomial_part = np.sum(np.dot(Tau_i_k, np.log(pi_k)))
        gaussian_probs = np.zeros((n,self.k_))
        for i in range(self.k_) : gaussian_probs[:,i] = multivariate_normal.pdf(X, mus[i], sigmas[i]*np.eye(p))
        gaussian_part = np.sum(Tau_i_k * np.log(gaussian_probs))
        
        return  multinomial_part +  gaussian_part
    
    def M_Step(self, X, Tau_i_k):
        '''Compute the new parameters'''
        
        n, p = X.shape
        
        pi_k_new = np.mean(Tau_i_k, axis=0)
        mus_new = np.dot(Tau_i_k.T, X)/(np.sum(Tau_i_k, axis=0)[:, None])
        
        diff = X[:,:,None] - mus_new.T
        sigmas_new = np.sum(Tau_i_k*np.einsum('ijk, ijk -> ik', diff, diff), axis=0)/(p*(np.sum(Tau_i_k, axis=0)))
        
        return pi_k_new, mus_new, sigmas_new
        
        
    def fit(self, X):
        """ Find the parameters mus_ and sigmas_ and pi_k_
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self
        """
        
        n,p = X.shape
        rng = np.random.RandomState(self.random_seed_)
        
        self.iterations_ = 0
              
            
        model = Kmeans(self.k_, self.random_seed_)
        model.fit(X)
        
        self.pi_k_ = np.unique(model.labels_, return_counts=True)[1]/n
        self.mus_ = model.centers_
        self.sigmas_ = rng.uniform(low= 1,high=20, size=self.k_)            
                                                   
        converge = False                              
        while not(converge):
            
            expectation_t = self.expectation_
            
            self.Tau_i_k_ = self.compute_Tau_i_k(X, self.mus_, self.sigmas_, self.pi_k_)
            self.pi_k_, self.mus_, self.sigmas_ = self.M_Step(X, self.Tau_i_k_)
            self.expectation_ = self.E_step(X, self.mus_, self.sigmas_, self.pi_k_, self.Tau_i_k_)
            
            converge = np.isclose(self.expectation_, expectation_t, rtol=1e-5)
            self.iterations_ += 1

        
        self.labels_ = np.argmax(self.Tau_i_k_, axis=1)

        return self
    
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        return np.argmax(self.compute_Tau_i_k(X, self.mus_, self.sigmas_, self.pi_k_), axis=1)
    
    def compute_complete_log_likelihood(self, X):
        
        '''Compute the complete loglikelihood of the data for this model'''
        n, p = X.shape
        labels = self.predict(X)
        one_hot = np.eye(self.k_)[labels]
        return self.E_step(X, self.mus_, self.sigmas_, self.pi_k_, one_hot)           
            
            
            

            
            
            
# EM - Gaussian Mixture model

class GMM_general():
    
    """class for the general Gaussian mixture model"""
    
    def __init__(self, k, random_seed=1):
        '''
        Attributes:
        
        k_: integer
            number of components
        mus_: np.array 
            means of our gaussian vectors
        sigmas_: np.array
            the sigmas of our isotropic guassians
        Tau_i_k_: (n, K) np.array
            conditional probabilities z|x for all data points 
        labels_: (n, ) np.array
            labels for data points
        '''
        self.k_ = k
        self.mus_ = None
        self.sigmas_ = None
        self.Tau_i_k_ = None
        self.pi_k_ = None
        self.labels_ = None
        self.iterations_ = 0
        self.random_seed_ = random_seed
        self.expectation_ = np.inf
        
    
    def compute_Tau_i_k(self, X, mus, sigmas, pi_k):
        '''Compute the conditional probability matrix Tau_i_k
        shape: (n, K)
        '''
        n, p = X.shape
        Tau_i_k = np.zeros((n ,self.k_))
        for i in range(self.k_) : Tau_i_k[:,i] = multivariate_normal.pdf(X, mus[i], sigmas[:,:,i])
        Tau_i_k *= pi_k 
        Tau_i_k /= np.sum(Tau_i_k, axis= 1, keepdims=True)
        return Tau_i_k

            
    def E_step(self, X, mus, sigmas, pi_k, Tau_i_k):
        '''Compute the expectation to check convergence'''
        
        n, p = X.shape
        multinomial_part = np.sum(np.dot(Tau_i_k, np.log(pi_k)))
        gaussian_probs = np.zeros((n,self.k_))
        for i in range(self.k_) : gaussian_probs[:,i] = multivariate_normal.pdf(X, mus[i], sigmas[:,:,i])
        gaussian_part = np.sum(Tau_i_k * np.log(gaussian_probs))
        
        return  multinomial_part +  gaussian_part
    
    def M_Step(self, X, Tau_i_k):
        '''Compute the new parameters'''
        
        n, p = X.shape
        
        pi_k_new = np.mean(Tau_i_k, axis=0)
        mus_new = np.dot(Tau_i_k.T, X)/(np.sum(Tau_i_k, axis=0)[:, None])
        
        sigmas_new = np.zeros((p,p,self.k_))
        for i in range(self.k_) : sigmas_new[:,:,i] = np.dot((X - mus_new[i]).T, (Tau_i_k[:,[i]]*(X - mus_new[i])))/(np.sum(Tau_i_k, axis=0)[i])
        
        return pi_k_new, mus_new, sigmas_new
        
        
    def fit(self, X):
        """ Find the parameters mus_ and sigmas_ and pi_k_
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self
        """
        
        n,p = X.shape
        rng = np.random.RandomState(self.random_seed_)
        
        self.iterations_ = 0
              
            
        model = Kmeans(self.k_, self.random_seed_)
        model.fit(X)
        
        self.pi_k_ = np.unique(model.labels_, return_counts=True)[1]/n
        self.mus_ = model.centers_
        self.sigmas_ = rng.uniform(low= 1,high=20, size=self.k_)[None, None,:]*np.eye(p)[:,:,None]         
                                                   
        converge = False                              
        while not(converge):
            
            expectation_t = self.expectation_
            
            self.Tau_i_k_ = self.compute_Tau_i_k(X, self.mus_, self.sigmas_, self.pi_k_)
            self.pi_k_, self.mus_, self.sigmas_ = self.M_Step(X, self.Tau_i_k_)
            self.expectation_ = self.E_step(X, self.mus_, self.sigmas_, self.pi_k_, self.Tau_i_k_)
            
            converge = np.isclose(self.expectation_, expectation_t, rtol=1e-5)
            self.iterations_ += 1

        
        self.labels_ = np.argmax(self.Tau_i_k_, axis=1)

        return self
    
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        return np.argmax(self.compute_Tau_i_k(X, self.mus_, self.sigmas_, self.pi_k_), axis=1)
    
    def compute_complete_log_likelihood(self, X):
        
        '''Compute the complete loglikelihood of the data for this model'''
        n, p = X.shape
        labels = self.predict(X)
        one_hot = np.eye(self.k_)[labels]
        return self.E_step(X, self.mus_, self.sigmas_, self.pi_k_, one_hot)
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#plot utilities

#kmeans plots

def plot_kmeans(data, random_seeds, save=False):
    for i in random_seeds:
        clustering = Kmeans(k=4, random_seed=i)
        clustering.fit(data)
        _ = plt.figure(figsize = (10,5))
        plt.subplot(1,2,1)
        plt.scatter(data[:,0], data[:,1], marker='+',s=50, c = clustering.initial_labels)
        plt.scatter(clustering.initial_centers[:,0], clustering.initial_centers[:,1] , s=100, c='red')
        plt.title('Initialisation of our clusters')
        plt.text(x=2.8, y=-13.5, s='The final distortion is : '+str(round(clustering.distortion_,3)), fontsize=12)
        plt.subplot(1,2,2)
        plt.scatter(data[:,0], data[:,1],marker='+',s=50, c = clustering.labels_)
        plt.scatter(clustering.centers_[:,0], clustering.centers_[:,1], s=100, c='red')
        plt.title('Kmeans clusters after convergence')
        if save: plt.savefig('Kmeans_%d'%(i+1), bbox_inches='tight')
        plt.show()

def distortion_variability(data, random_seeds):
    distortions = []
    for random_seed in random_seeds:
        clustering = Kmeans(4, random_seed)
        clustering.fit(data)
        distortions.append(clustering.distortion_)
    return distortions
        
            
#GMM plots


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plot_ellipse(means, covs, conf, ax):
    
    classes, p = means.shape
    quantile = chi2.ppf(conf/100, 2)
    
    for k in range(classes):
        cov, mean = covs[:,:,k], means[k]
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 2*np.sqrt(vals*quantile)
        ell = Ellipse(xy=(mean[0], mean[1]),
                  width=w, height=h,
                  angle=theta, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)

def plot_conf_interval(train, test, model, conf=90, case = 'Isotropic', save=False):
    
    means = model.mus_
    covs = model.sigmas_ if len(model.sigmas_.shape) == 3 else model.sigmas_[None,None,:]*np.eye(2)[...,None]
    n_train, n_test = train.shape[0], test.shape[0]    
    
    fig = plt.figure(figsize = (10,5))
    ax1 = plt.subplot(1,2,1)
    train_mini, train_maxi = np.min(train) - 1, np.max(train) +1
    ax1.set_xlim([train_mini, train_maxi])
    ax1.set_ylim([train_mini, train_maxi])
    plt.scatter(train[:,0], train[:,1], marker='+',s=100, c = model.labels_)
    plt.scatter(means[:,0], means[:,1], marker='o',s=100, c='red')
    plot_ellipse(means, covs, conf, ax1)
    plt.title('Train data')
    log_train = model.compute_complete_log_likelihood(train)/n_train
    plt.text(np.min(train[:,0]),1.35* np.min(train[:,1]),'The complete loglikelihood: %.4f'%log_train, fontsize= 12)
    
    ax2 = plt.subplot(1,2,2)
    test_mini, test_maxi = np.min(train) -1, np.max(train) +1
    ax2.set_xlim([test_mini, test_maxi])
    ax2.set_ylim([test_mini, test_maxi])
    plt.scatter(test[:,0], test[:,1], marker='+',s=100, c = model.predict(test))
    plt.scatter(means[:,0], means[:,1], marker='o',s=100, c='red')
    plot_ellipse(means, covs, conf, ax2)
    plt.title('Test data')
    log_test = model.compute_complete_log_likelihood(test)/n_test
    plt.text(np.min(train[:,0]),1.35* np.min(train[:,1]),'The complete loglikelihood: %.4f'%log_test, fontsize= 12)
    
    
    plt.text(-23 + np.min(train[:,0]),1.3* np.max(train[:,1]),'Centers, Clusters and Confidence intervals at ' +str(conf) +'% for '+ case + ' EM', fontsize= 13)
    if save : plt.savefig(case + '_EM', bbox_inches='tight')
    plt.show()
            
            
            
            
            
            
            
            
            