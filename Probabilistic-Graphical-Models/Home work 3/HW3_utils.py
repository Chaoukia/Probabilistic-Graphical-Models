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
        
            
class GaussianHMM:
    
    def __init__(self, pi, proba_trans, mus, Sigmas):
        
        """
        Description
        -------------
        Constructor of an HMM with Gaussian emission probabilities
        
        Attributes
        -------------
        pi               : 1D np.array of length the number of states K. Parameter vector of the multinomial distribution 
                           of the first latent variable.
        proba_trans      : 2D np.array of shape K*K, the transition matrix of the latent Markov chain.
                           1st dimension for the next state.
                           2nd dimension for the current state.
        mus              : 2D np.array of shape K*2, the means of the Gaussian in the emission distributions. The 2nd
                           dimension is 2 here because the observed variables exist on the 2D plane.
        Sigmas           : 3D np.array of shape K*2*2, 1st dimension for the states, 2nd and 3rd for the covariance matrices.
        probas_singleton : 2D np.array of shape T*K holding the probabilities p(q_t | u) for all times and states.
        probas_pair      : 3D np.array of shape T*K*K holding the probabilities (p(q_t, q_{t+1} | u)
                           1st dimension for time t.
                           2nd dimension for q_{t+1}
                           3rd dimension for q_t
        K                : Int, the number of states.
        """
        
        self.pi = pi
        self.proba_trans = proba_trans
        self.mus = mus
        self.Sigmas = Sigmas
        self.K = len(self.pi)
        self.converged = False
        
    def alpha_recursion_logs(self, u):
        
        """
        Description
        -------------
        Perform alpha recusrion to compute all the alpha-messages of the forward step in the logarithmic scale.
        
        Parameters
        -------------
        u : 2D np.array of shape T*2, the observations.
        
        Returns
        -------------
        alphas : 2D np.array of shape T*K
        """
        
        T = u.shape[0]
        K = self.K
        alphas = np.zeros((T, K))
        alphas[0, :] = np.log(self.pi) + np.log(np.array([multivariate_normal.pdf(u[0, :], self.mus[k, :], self.Sigmas[k, :, :]) for k in range(K)]))
        
        for t in range(1, T):
            # Compute the emission probability distribution of observation u_t given one of the K states each time
            emissions = np.log(np.array([multivariate_normal.pdf(u[t, :], self.mus[k, :], self.Sigmas[k, :, :]) for k in range(K)]))
            a = np.log(self.proba_trans) + alphas[t - 1, :].reshape((1, -1))
            l_max = np.max(a, axis = 1)
            alphas[t, :] = emissions + l_max + np.log(np.exp(a - l_max.reshape((-1, 1))).sum(axis = 1))
            
        return alphas
    
    def beta_recursion_logs(self, u):
        
        """
        Description
        -------------
        Perform beta recusrion to compute all the beta-messages of the backward step in the logarithmic scale.
        
        Parameters
        -------------
        u : 2D np.array of shape T*2, the observations.
        
        Returns
        -------------
        betas : 2D np.array of shape T*K
        """
        
        T = u.shape[0]
        K = self.K
        betas = np.zeros((T, K))
        betas[-1, :] = 0 # 1 in logarithmic scale.
        
        for t in range(T - 2, -1, -1):
            # Compute the emission probability distribution of observation u_{t+1} given one of the K states each time
            emissions = np.log(np.array([multivariate_normal.pdf(u[t + 1, :], self.mus[k, :], self.Sigmas[k, :, :]) for k in range(K)]))
            a = emissions.reshape((-1, 1)) + np.log(self.proba_trans) + betas[t + 1, :].reshape((-1, 1))
            l_max = np.max(a, axis = 0)
            betas[t, :] = l_max + np.log(np.exp(a - l_max.reshape((1, -1))).sum(axis = 0))
                        
        return betas
    
    def probas_singleton_logs(self, u):
        
        """
        Description
        -------------
        Compute the probabilities p(q_t | u) for all times and states.
        
        Parameters
        -------------
        u : 2D np.array of shape T*2, the observations.
        
        Returns
        -------------
        probas : 2D np.array of shape T*K holding the probabilities p(q_t | u) for all times and states.
        """
        
        alphas = self.alpha_recursion_logs(u)
        betas = self.beta_recursion_logs(u)
                
        term1 = alphas + betas
        
        l_max = np.max(term1, axis = 1)
        term2 = (l_max + np.log(np.exp(term1 - l_max.reshape((-1, 1))).sum(axis = 1))).reshape((-1, 1))
        
        probas = term1 - term2
        
        return np.exp(probas)
        
    def probas_pair_logs(self, u):
        
        """
        Description
        -------------
        Compute the probabilities p(q_t, q_{t+1} | u) for all times and K*K possible states.
        
        Parameters
        -------------
        u : 2D np.array of shape T*2, the observations.
        
        Returns
        -------------
        probas : 3D np.array of shape T*K*K holding the probabilities p(q_t, q_{t+1} | u)
                 1st dimension for time t.
                 2nd dimension for q_{t+1}
                 3rd dimension for q_t
        """
        
        T = u.shape[0]
        K = self.K
        alphas = self.alpha_recursion_logs(u)
        betas = self.beta_recursion_logs(u)
        
        emissions = np.zeros((T - 1, K))
        for k in range(K):
            emissions[:, k] = multivariate_normal.pdf(u[1:, :], self.mus[k, :], self.Sigmas[k, :, :])
        
        # We never use the last row of alpha when computing the numerator of any p(q_t, q_{t+1} | u)
        # We never use the first row of beta when computing the numerator of any p(q_t, q_{t+1} | u)
        term1 = alphas[:-1, :].reshape((T - 1, 1, K)) + (betas[1:, :].reshape((T - 1, K, 1)) + np.log(emissions).reshape((T - 1, K, 1)))
        term1 += np.log(self.proba_trans).reshape(1, K, K)
                
        l_max = np.max(alphas[:-1, :] + betas[:-1, :], axis = 1)
        term2 = (l_max + np.log(np.exp(alphas[:-1, :] + betas[:-1, :] - l_max.reshape((-1, 1))).sum(axis = 1))).reshape((T - 1, 1, 1))
        
        probas = term1 - term2
        
        return np.exp(probas)
        
    def estimate_pi(self, probas_singleton):
        
        """
        Description
        -------------
        Estimate parameter pi in the M-step.
        
        Parameters
        -------------
        probas_singleton : the return of method probas_singleton_logs
        
        Returns
        -------------
        updated parameter pi : 1D np.array of length K
        """
        
        return probas_singleton[0, :]
    
    def estimate_proba_trans(self, probas_pair):
        
        """
        Description
        -------------
        Estimate parameter probability transition matrix in the M-step.
        
        Parameters
        -------------
        probas_pair : the return of method probas_pair_logs
        
        Returns
        -------------
        updated parameter proba_trans : 2D np.array of shape K*K
        """
                
        term1 = np.sum(probas_pair, axis = 0)
        
        term2 = (np.sum(probas_pair, axis = (0, 1))).reshape((1, -1))
                
        return term1/term2
        
    def estimate_mus(self, u, probas_singleton):
        
        """
        Description
        -------------
        Estimate parameter mus in the M-step
        
        Parameters
        -------------
        u                : 2D np.array of shape T*2, the observations
        probas_singleton : the return of method probas_singleton_logs
        
        Returns
        -------------
        updated parameter mus : 2D np.array of shape K*2
        """
        
        T = u.shape[0]
        K = self.K
        
        probas_u = (probas_singleton).reshape((T, K, 1))*(u.reshape((T, 1, 2)))
        term1 = np.sum(probas_u, axis = 0)
        
        term2 = (np.sum(probas_singleton, axis = 0)).reshape((-1, 1))
                
        return term1/term2
        
    def estimate_sigmas(self, u, probas_singleton):
        
        """
        Description
        -------------
        Estimate parameter Sigmas in the M-step.
        
        Parameters
        -------------
        u                : 2D np.array of shape T*2, the observations.
        probas_singleton : the return of method probas_singleton_logs
        
        Returns
        -------------
        updated parameter Sigmas : 3D np.array of shape K*2*2
        """
        
        T = u.shape[0]
        K = self.K
        
        obs_centered = u.reshape((T, 1, 2)) - (self.mus).reshape((1, K, 2))
        obs_centered = (obs_centered.reshape((T, K, 2, 1)))*(obs_centered.reshape((T, K, 1, 2)))
        
        probas_obs_centered = (probas_singleton.reshape((T, K, 1, 1)))*obs_centered
        term1 = np.sum(probas_obs_centered, axis = 0)
        
        term2 = (np.sum(probas_singleton, axis = 0)).reshape((-1, 1, 1))
        
        return term1/term2
    
    def E_step(self, u, delta_i, delta_ij):
        
        """
        Description
        -------------
        Compute the log-likelihood.
        
        Parameters
        -------------
        u        : 2D np.array of shape T*2, the observations.
        delta_i  : 2D np.array of shape T*K holding the probabilities p(q_t | u) for all times and states.
        delta_ij : 3D np.array of shape (T-1)*K*K
                   1st dimension for time t.
                   2nd dimension for q_{t+1}
                   3rd dimension for q_t
        
        Returns
        -------------
        
        """
        
        T = u.shape[0]
        K = self.K
        term_pi = np.dot(delta_i[0, :], np.log(self.pi))
        term_proba_trans = np.sum(delta_ij*(np.log(self.proba_trans.reshape((1, self.K, self.K)))))
        logpdfs = np.zeros((T, K))
        for k in range(K):
            logpdfs[:, k] = np.log(multivariate_normal.pdf(u, mean = self.mus[k, :], cov = self.Sigmas[k, :, :]))
            
        term_gaussian = np.sum(delta_i*logpdfs)
        
        return term_pi + term_proba_trans + term_gaussian
        
                            
    def EM(self, u, max_iter = 100):
        
        """
        Description
        -------------
        Apply EM algorithm to estimate the parameters pi, proba_trans, mus and Sigmas
        
        Parameters
        -------------
        u        : 2D np.array of shape T*2, the observations.
        max_iter : Int, the maximum number of iterations.
        
        Returns
        -------------
        self
        """
        
        i = 1
        self.converged = False
        probas_singleton = self.probas_singleton_logs(u)
        probas_pair = self.probas_pair_logs(u)
        likelihood_t = self.E_step(u, probas_singleton, probas_pair)
        while (i <= max_iter) and (not self.converged):
            print('iteration : ', i)
            print('log_likelihood : \n', likelihood_t)
            self.pi = self.estimate_pi(probas_singleton) + 1e-80
            self.proba_trans = self.estimate_proba_trans(probas_pair)
            self.mus = self.estimate_mus(u, probas_singleton)
            self.Sigmas = self.estimate_sigmas(u, probas_singleton)
            if i%10 == 0:
                print('pi : \n', self.pi)
                print('proba_trans : \n', self.proba_trans)
                print('mus : \n', self.mus)
                print('Sigmas : \n', self.Sigmas)
                print('\n')
                
            probas_singleton = self.probas_singleton_logs(u)
            probas_pair = self.probas_pair_logs(u)
            likelihood_t1 = self.E_step(u, probas_singleton, probas_pair)
            self.converged = np.isclose(likelihood_t1, likelihood_t, rtol = 1e-5)
            likelihood_t = likelihood_t1
            i += 1
            
    def Viterbi_algorithm(self, obs):
        
        """
        Description
        -------------
        Compute the most likely hidden state for each observation
        
        Parameters
        -------------
        obs : the observed data array Tx2
        
        Returns
        -------------
        labels
        """
        T = obs.shape[0]
        max_messages = np.zeros((T,self.K))
        argmax = np.zeros((T,self.K))
        emissions = np.zeros((T, self.K))
        for k in range(self.K): emissions[:, k] = multivariate_normal.pdf(obs, self.mus[k, :], self.Sigmas[k, :, :])
        max_messages[0,:] = np.log(self.pi) + np.log(emissions[0])
        
        for t in range(1,T):
            max_messages[t,:] = np.max(max_messages[t-1,:].reshape(1,-1) + np.log(self.proba_trans), axis=1) + np.log(emissions[t])
            argmax[t,:] = np.argmax(max_messages[t-1,:].reshape(1,-1) +  np.log(self.proba_trans), axis=1)
        
        labels = np.zeros(T, dtype=int)
        labels[-1] = np.argmax(max_messages[-1,:])
        for t in range(T-1,0,-1): labels[t-1] = argmax[t, labels[t]]

        return labels
    
    def compute_complete_log_likelihood(self, u):
        '''
        Compute the complete loglikelihood of the data for this model
        '''
        n, _ = u.shape
        labels = self.Viterbi_algorithm(u)
        one_hot = np.eye(self.K)[labels]
        def coord_to_point(x,y):
            matrix = np.zeros((self.K, self.K))
            matrix[x,y] = 1
            return matrix
        double_hot = np.vectorize(coord_to_point, signature='(m),(m)->(k,k)')(labels[:-1].reshape(-1,1), labels[1:].reshape(-1,1))
        
        return self.E_step(u, one_hot, double_hot)
            
    
        
            
            
            
########################################################################################################################################################
            
#plot utilities

        
            
#confidence interval plots


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

def plot_conf_interval(train, test, model, conf=90, case = 'GMM', save=False):
    if case =='GMM':
        means = model.mus_
        covs = model.sigmas_ 
        tr_labels, ts_labels = model.predict(train), model.predict(test)
    else : 
        means = model.mus
        covs = np.transpose(model.Sigmas, axes=(1,2,0))
        tr_labels, ts_labels = model.Viterbi_algorithm(train), model.Viterbi_algorithm(test)
        
    n_train, n_test = train.shape[0], test.shape[0]    
    
    fig = plt.figure(figsize = (10,5))
    ax1 = plt.subplot(1,2,1)
    train_mini, train_maxi = np.min(train) - 1, np.max(train) +1
    ax1.set_xlim([train_mini, train_maxi])
    ax1.set_ylim([train_mini, train_maxi])
    plt.scatter(train[:,0], train[:,1], marker='+',s=100, c =tr_labels)
    plt.scatter(means[:,0], means[:,1], marker='o',s=100, c='red')
    plot_ellipse(means, covs, conf, ax1)
    plt.title('Train data')
    log_train = model.compute_complete_log_likelihood(train)/n_train
    plt.text(np.min(train[:,0]),1.35* np.min(train[:,1]),'The complete loglikelihood: %.4f'%log_train, fontsize= 12)
    
    ax2 = plt.subplot(1,2,2)
    test_mini, test_maxi = np.min(train) -1, np.max(train) +1
    ax2.set_xlim([test_mini, test_maxi])
    ax2.set_ylim([test_mini, test_maxi])
    plt.scatter(test[:,0], test[:,1], marker='+',s=100, c = ts_labels)
    plt.scatter(means[:,0], means[:,1], marker='o',s=100, c='red')
    plot_ellipse(means, covs, conf, ax2)
    plt.title('Test data')
    log_test = model.compute_complete_log_likelihood(test)/n_test
    plt.text(np.min(train[:,0]),1.35* np.min(train[:,1]),'The complete loglikelihood: %.4f'%log_test, fontsize= 12)
    
    
    plt.text(-23 + np.min(train[:,0]),1.3* np.max(train[:,1]),'Centers, Clusters and Confidence intervals at ' +str(conf) +'% for '+ case + ' with EM', fontsize= 13)
    if save : plt.savefig(case + '_EM', bbox_inches='tight', dpi=200)
    plt.show()
            
            
            
            
            
            
            
            
            