# Libraries imports
import sys

from covariance import *
from optimizers import *
# Helper functions
from plot import *

from sklearn.datasets import make_classification
import autograd.numpy as np
from numpy.linalg import inv
from autograd import grad

###################
# MODEL SELECTION #
###################

# First argument of this python script is the Model, to be passed in command line !!!!!
model_name = (sys.argv[1]) # Model can be "LinReg", "PolyReg", "Gaussian"
model_list = ["jaakkola", "SVI"]
if model_name not in model_list:
    raise NameError(f"Unknown model. Try one of those: {model_list}")

# Model parameters (as global variables)

d = 2 # size of the param vector to be estimated
n = 100 # size of train/test batch
m = 1000 # size of truth batch
mc = 1999 # number of Monte_Carlo samples
boot = 10 # number of bootstraps draws
EM_iter = 30
SGD_iter = 200

mu_0 = np.full((d, 1), 0.0) # Prior Gaussian mean in float, must be 0
sigma_0 = np.full(d, 1.0) # Pior SVI std dev, must be I
S_0 = np.identity(d)*20 # Prior Gaussian variance matrix

Z_space = .2 # variance of the generator of the input Z (.5 was ok)
theta = generate_theta((d,1)) # True parameter theta

H = int(3 * n ) # grid [search] size
G = H/200 # grid [search] resolution

ten = 3 # Number of tentatives of each strategy to be averaged together

alpha_bayes = np.clip(n, 0, H) # Value of bayesian alpha for plotting


####################
# Helper functions #
####################

def sigmoid(x):
    """
    Sigmoid helper function.
    """
    return 1. / (1. + np.exp(-x))


def reparam_fwd(rho_P):
    """
    Reparam rho_P into sigma_P
    """
    return np.log(1 + np.exp(rho_P))


def reparam_bwd(sigma_P):
    """
    Reparam sigma intro rho
    """
    return np.log(np.exp(sigma_P) - 1)


####################
# Generate batches #
####################

def generate_batch(testing=True, scikit=True):
    """
    Generates a new train/test/validation set batch, for a new tentative.
    """
    if(scikit):
        Z, Y = make_classification(
            n_samples=n,
            n_features=d,
            n_informative=d,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=2,
            flip_y=.1
        )
        Y = Y.reshape(-1,1)

        Z_3, Y_3 = make_classification(
            n_samples=m,
            n_features=d,
            n_informative=2,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=2
        )
        Y_3 = Y_3.reshape(-1, 1)

        plot_logreg_data(Z, Y)

    else:
        if testing: np.random.seed(37)
        Z = np.random.normal(0, Z_space, (n,d) ) # Input matrix of the linear regression, gaussian
        if testing: np.random.seed(32)
        Y = np.random.binomial( n=1 , p=sigmoid(Z @ theta), size=(n,1) ) #computes y ~ P(Y=y|Z, theta) = B(1, σ(theta*Z) )

        if testing: np.random.seed(36)
        Z_3 = np.random.normal(0, Z_space, (m,d) ) # validation batch
        if testing: np.random.seed(35)
        Y_3 = np.random.binomial( n=1 , p=sigmoid(Z_3 @ theta), size=(m,1) ) #computes y ~ P(Y=y|Z, theta) = B(1, σ(theta*Z) )

        import time # stops random.seed to work
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)

    Z_1 = Z[0:int(n / 2)]  # train batch
    Z_2 = Z[int(n / 2):n]  # test batch
    Y_1 = Y[0:int(n / 2)]  # train batch
    Y_2 = Y[int(n / 2):n]  # test batch

    X = (Z, Y)
    X_1 = (Z_1,Y_1)
    X_2 = (Z_2,Y_2)
    X_3 = (Z_3,Y_3)

    return X, X_1, X_2, X_3


def generate_bootstrap(batch):
    """
    Generate a bootstrap version of the batch (random draw with replacement of its entries)
    """
    batch_bootstrap_indices = np.random.choice(batch[0].shape[0], batch[1].size, replace=True) # choose bootstrap indices
    return batch[0][batch_bootstrap_indices], batch[1][batch_bootstrap_indices]  # create bootstrap matrix


def loss(batch, theta):
    """
    Loss function.
    """
    Z, Y = batch  # unwrap X into (Z,Y)
    return -Z@theta * Y - np.log( sigmoid(-Z@theta) ) # element-wise product Z@theta * Y


def empirical_risk(batch, theta):
    """
    Empirical risk r^batch(theta) for some batch (Z,Y), with theta. theta MUST be a matrix of vectors
    (each vector being one possible theta sampled from the posterior)
    Return: a vector of size _mc_, with the empirical risk computed over the
    whole batch X, but with a different sample of theta for each entry of the output vector.
    """
    return np.mean( loss(batch, theta), axis=0)


############################
# Posterior Gaussian samples
############################

def mu_P(alpha, batch, S_P):
    """
    jaakkola: Mean of the Gaussian posterior estimation, computed with given batch.
    S_P is the cov matrix as an argument to gain time.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return S_P @ ( inv(S_0) @ mu_0 + alpha * np.mean( (Y-0.5) * Z , axis=0 ).reshape(-1, 1) )


def S_P(alpha, batch, lambda_v):
    """
    jaakkola: Variance of the multivariate Gaussian posterior estimation.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    lv1 = lambda_v.reshape(-1) # make lambda_v 1 dimensional, otherwise diag will not make it a matrix
    return inv( inv(S_0) + 2*alpha/Y.size * Z.T @ np.diag(lv1) @ Z )


def lambda_v(batch, mu_P, S_P):
    """
    jaakkola: Compute the variational vector parameter lambda(v)
    """
    Z, Y = batch # unwrap X into (Z,Y)
    v = np.sqrt(np.sum(Z.T * ((S_P + mu_P @ mu_P.T) @ Z.T), axis=0)) # fast diag(Z @ A @ Z.T)
    return 1/(2*v) * (sigmoid(v) - 0.5) # return lambda(v)


def f_ELBO(theta, alpha, batch, mu_P, rho_P):
    """
    SVI: Function f as described in the paper.
    Returns a scalar
    """
    sigma_P = np.squeeze( reparam_fwd(rho_P) ) # convert back into sigma (1 dimensional vector)

    log_posterior = - np.sum(np.log(sigma_P)) - 0.5 * (theta - mu_P).T * (1/sigma_P**2) @ (theta - mu_P)
    log_prior = - 0.5 * theta.T @ theta
    log_likelihood = - alpha * empirical_risk(batch, theta)

    return log_posterior - log_prior - log_likelihood


df_dtheta = grad(f_ELBO, 0) # df / d theta
df_dmu_P = grad(f_ELBO, 3) # df / d mu_P
df_drho_P = grad(f_ELBO, 4) # df / d rho_P


def theta_sample(alpha, batch, lr=.2, EM_iter=EM_iter, SGD_iter=SGD_iter):
    """
    Returns mc random samples of the posterior mean estimator, for doing Monte Carlo approx.
    This uses EM algorithm with _EM_iter_ iterations to converge to approx posterior.
    The output is an matrix of _d_ sampled means, times _mc_ samples. Shape = (d, mc)
    lr must be maximum 0.25 otherwise explosion occurs with reparam
    """
    if model_name == "jaakkola":
        lv1 = np.abs(np.random.normal(0, 1, batch[1].shape))  # POSITIVE initial value of lambda(v) before maximization, same shape as Y
        for i in range(EM_iter):  # EM algorithm updating (mu_P,S_P) and v in turn.
            S_P1 = S_P(alpha, batch, lv1)  # Cov matrix (d x d)
            mu_P1 = mu_P(alpha, batch, S_P1)  # Mean vector (d x 1)
            lv1 = lambda_v(batch, mu_P1, S_P1)  # lambda(v) vector (n x 1)
        return np.random.multivariate_normal(mu_P1.reshape(-1), S_P1, mc).T

    if model_name == "SVI": # Prior MUST BE N(0,I)
        mu_P1 = np.copy(mu_0) # init posterior = prior N(0,I)
        rho_P1 = reparam_bwd(sigma_0) # init posterior = prior N(0,I)

        gradients_mu = np.empty((d,SGD_iter-1)) # init plotting

        for j in range(1, SGD_iter):
            epsilon = np.random.multivariate_normal(np.zeros(d), np.identity(d)).reshape(-1, 1) # generate noise ~ N(0,I)
            theta = np.nan_to_num( mu_P1 + reparam_fwd(rho_P1.reshape(-1,1)) * epsilon ) # nan to num is used to fix Autograd bug with sqrt(0)
            df_dthetha1 = np.nan_to_num( df_dtheta(theta, alpha, batch, mu_P1, rho_P1) )
            grad_mu_P = np.nan_to_num( df_dthetha1 + df_dmu_P(theta, alpha, batch, mu_P1, rho_P1) )
            grad_rho_P = np.nan_to_num( df_dthetha1 * (epsilon/(1 + np.exp(-rho_P1.reshape(-1,1)))) + df_drho_P(theta, alpha, batch, mu_P1, rho_P1).reshape(-1,1) )

            mu_P1  -= lr/np.sqrt(j) * grad_mu_P # gradient descent
            rho_P1 -= lr/np.sqrt(j) * np.squeeze(grad_rho_P)

            # Plotting the SGD
            gradients_mu[:,j-1] = np.squeeze(mu_P1)

        if(False and alpha==0):
            plt.plot(gradients_mu.T)
            plt.xlabel("iteration")
            plt.ylabel("mu_P's values")
            plt.savefig("../plots/SGD_SVI.png")
            plt.show()
            plt.clf()

        return np.random.multivariate_normal(mu_P1.reshape(-1), np.diag(reparam_fwd(rho_P1)**2), mc).T


############
# D and dD #
############

def D(alpha, E_batch, r_batch):
    """
    Estimator of C(alpha), corresponding to E_batch [ r^(r_batch) ], with batch input Z and output Y
    """
    local_theta_samples = theta_sample(alpha, E_batch) #[0] # Get _mc_ new samples of the posterior
    return np.mean( empirical_risk(r_batch, local_theta_samples) ) # Approximate expectation using MC


D_vec = np.vectorize(D, excluded=[1,2]) # Vectorized version of D taking an array of alphas!


def dD_MC(alpha, E_batch, r_batch):
    """
    Monte-Carlo estimator of the derivative of D(alpha) with E_batch [ r^(r_batch) ]
    """
    local_theta_samples = theta_sample(alpha, E_batch) # Get _mc_ new samples of the posterior
    return MC_risk_covariance(E_batch, r_batch, empirical_risk, local_theta_samples)


##############
# Strategies #
##############

def bayes(alpha, X, X_1, X_2, X_3):
    """
    Selects alpha using the traditional Bayesian setup.
    Likelihood and prior are always given the same importance.
    Returns a least squares with minimum at the Bayesian alpha.
    """
    return (alpha - n)**2


def naive_grid(alpha, X, X_1, X_2, X_3):
    """
    Stupid strategy using only the training set
    """
    return D(alpha, X, X)


def optimal_grid(alpha, X, X_1, X_2, X_3):
    """
    Optimal strategy using the large validation set.
    """
    return D(alpha, X, X_3)


def sample_split_grid(alpha, X, X_1, X_2, X_3):
    """
    Strategy Sample-split
    Negative covariance MC estimation of the derivative of D(alpha), using mc different samples theta
    """
    return D(alpha, X_1, X_2)


def bootstrap_grid(alpha, X, X_1, X_2, X_3):
    """
    Strategy Bootstrap
    Same as strategy 2, but using bootstrap X and full X as batches, instead of first and second halves of X.
    We compute the bootstrap boot times and average over the results.
    """
    return np.mean(np.array([D(alpha, generate_bootstrap(X), X) for i in range(boot)]))


def safeBayes_grid(alpha, X, X_1, X_2, X_3, start=1):
    """
    Grunwald strategy for grid search to compute the expectation
    """
    Z, Y = X # unwrap X into (Z,Y)
    global_error = 0 # init d/da S(a) = 0 before summing

    for t in np.arange(start, Y.size-1) : # go from t=start to n-1
        batch_t = (Z[:t], Y[:t])  # batch from 0 to t
        local_theta_samples = theta_sample(alpha, batch_t) # (d*mc) Get _mc_ new samples of the posterior^(t)
        loss_values = loss( (Z[t+1], Y[t+1]) , local_theta_samples) # Values of loss t+1 for all sampled thetas
        global_error += np.mean(loss_values)

    return global_error


def naive_MC(alpha, X, X_1, X_2, X_3):
    """
    Stupid strategy using only the training set
    """
    return dD_MC(alpha, X, X)


def optimal_MC(alpha, X, X_1, X_2, X_3):
    """
    Optimal strategy using the large validation set.
    """
    return dD_MC(alpha, X, X_3)
    

def sample_split_MC(alpha, X, X_1, X_2, X_3):
    """
    Strategy Sample-split
    Negative covariance MC estimation of the derivative of D(alpha), using mc different samples theta
    """
    return dD_MC(alpha, X_1, X_2)


def bootstrap_MC(alpha, X, X_1, X_2, X_3):
    """
    Strategy Bootstrap
    Same as strategy 2, but using bootstrap X and full X as batches, instead of first and second halves of X.
    We compute the bootstrap boot times and average over the results.
    """
    return np.mean(np.array([dD_MC(alpha, generate_bootstrap(X), X) for i in range(boot)]))


def safeBayes_MC(alpha, X, X_1, X_2, X_3, start=1):
    """
    Grunwald strategy with MC sampling to compute the expectation
    """
    Z, Y = X # unwrap X into (Z,Y)
    global_error = 0 # init d/da S(a) = 0 before summing

    for t in range(start, Y.size-1) : # go from t=start to n-1
        batch_t = (Z[:t], Y[:t]) # batch from 0 to t
        local_theta_samples = theta_sample(alpha, batch_t) # (d*mc) Get _mc_ new samples of the posterior^(t)
        global_error += MC_d_local_error(batch_t, (Z[t+1], Y[t+1]), empirical_risk, loss, local_theta_samples) # derivative of the error

    return global_error


#########################################################
# Choice of the optimizer type and the list of strategies
#########################################################

optimizer = scipy_GD
optimizer_name = "scipy"
strategies = np.hstack([naive_MC, sample_split_MC, bootstrap_MC, safeBayes_MC, bayes, optimal_MC])
if(optimizer_name=="grid"):
    strategies = np.hstack([naive_grid, sample_split_grid, bootstrap_grid, safeBayes_grid, bayes, optimal_grid])
if(optimizer_name=="scipy"):
    strategies = np.hstack([naive_MC, sample_split_MC, bootstrap_MC, safeBayes_MC, bayes, optimal_MC])
strat_names = np.array(["Naive", "Sample-split", "Bootstrap", "SafeBayes", "Bayes", r'$ \mathcal{R}(\tilde{\alpha})$'])

# PLotting the pdfs for one example
#plot_joy(generate_batch, theta_vec)

# Actual optimization (everything happens here)
risks, strats_alphas, D_X_1, D_X_2, D_X_3, grid, all_X_3 = optimize(optimizer, generate_batch, D_vec, strategies, H, G, ten)
theta_min = np.min( [empirical_risk(X_3, theta) for X_3 in all_X_3]) # risk of last tentative (all should have same risk)

np.save(f"../data/{model_name}.npy", [model_name, optimizer_name, strat_names,
risks,
strats_alphas, theta_min,
D_X_1, D_X_2, D_X_3,
grid])

# Plotting each stretegy independently, each with all tentatives
plot_all(model_name, optimizer_name, strat_names, 
    risks,
    strats_alphas, theta_min,
    D_X_1, D_X_2, D_X_3,
    grid)