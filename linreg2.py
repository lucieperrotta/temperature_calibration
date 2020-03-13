# Libraries imports
import sys

import numpy as np  # Thinly-wrapped numpy
#from jax import elementwise_grad as egrad  # for functions that vectorize over inputs
from numpy.linalg import inv
from scipy.stats import invgamma
from scipy.special import digamma, gamma

# Helper functions
from covariance import *
from optimizers import *
from plot import *

###################
# MODEL SELECTION #
###################

# First argument of this python script is the Model, to be passed in command line !!!!!
model_name = (sys.argv[1]) # Model can be "LinReg", "PolyReg", "Gaussian"
model_list = ["LinReg2", "PolyReg2", "Gaussian2"]
if model_name not in model_list:
    raise NameError(f"Unknown model. Try one of those: {model_list}")

data="uniform"

####################
# MODEL PARAMETERS #
####################

d = 30 # size of the param vector to be estimated
n = 35 # size of train/test batch
m = 5000 # size of truth batch
mc = 20 # number of Monte_Carlo samples
boot = 5 # number of bootstraps draws

a = 2  # True secret parameter a (Must be > 1 !!!)
b = 2  # True secret parameter b (increase this to inscrease the variance)
sigma = 1 # generate_sigma(a, b) # secret variance !
delta = .01 # Alternative variance
p = .5 # Probability of using delta instead of sigma in noise generation

Z_space = .5 # variance of the generator of the input Z
theta = generate_theta((d,1)) # true parameter theta

uniform_noise = 3

if model_name== "LinReg2":
    mu_0 = np.full((d, 1), 0) # Prior Gaussian 0 mean
    S_0 = np.identity(d)/1 # Prior Gaussian cov identity matrix
    a_0 = 2 # Prior inv gamma a (Must be > 1 !!!)
    b_0 = 2 # Prior inv gamma boot

elif model_name== "PolyReg2":
    mu_0 = np.full((d, 1), 0) # Prior Gaussian mean vector
    S_0 = np.diag(np.array([0.5**k for k in range(d)])) # Prior Gaussian cov matrix !
    a_0 = 2 # Prior inv gamma a (Must be > 1 !!!)
    b_0 = 8 # Prior inv gamma boot

elif model_name== "Gaussian2":
    mu_0 = np.array([0]) # Prior scalar 0 mean
    S_0 = np.array([1]).reshape(1, 1) # Prior scalar cov matrix
    a_0 = 2 # Prior inv gamma a (Must be > 1 !!!)
    b_0 = 1 # Prior inv gamma boot

H = int(3 * n ) # grid [search] size
G = H/300 # grid [search] resolution

ten = 1 # Number of tentatives of each strategy to be averaged together

alpha_bayes = np.clip(n, 0, H) # Value of bayesian alpha for plotting


def polynomial_basis(zeta):
    """
    Helper function to turn a vector beta into a matrix Z.
    Z is composed of the vector beta at all powers from 0 to d-1.
    """
    return np.vander(zeta.reshape(-1), N=d, increasing=True) # Vandermonde matrix


def generate_batch(testing=False):
    """
    Generates a new train/test/validation set batch, for a new tentative.
    If testing is true, the same values are always generated. Otherwise, it's random.
    """
    if data=="well": # well specified data
        if testing: np.random.seed(76)
        epsilon_n = np.random.normal(0, sigma, (n,1))  # Noise vector to be added to train/test
        if testing: np.random.seed(54)
        epsilon_m = np.random.normal(0, sigma, (m,1))  # Noise vector to be added to validation

    if data=="GMM": # GMM noise
        if testing: np.random.seed(76)
        epsilon_n = p*np.random.normal(0, delta, (n,1)) + (1-p)*np.random.normal(0, sigma, (n,1)) # Noise vector to be added to train/test
        if testing: np.random.seed(54)
        epsilon_m = p*np.random.normal(0, delta, (m,1)) + (1-p)*np.random.normal(0, sigma, (m,1))  # Noise vector to be added to validation

    if data=="uniform": # Uniform noise
        epsilon_n = np.random.uniform(-uniform_noise, uniform_noise, (n,1))
        epsilon_m = np.random.uniform(-uniform_noise, uniform_noise, (m, 1))

    if model_name== "LinReg2":
        if testing: np.random.seed(2)
        Z = np.random.normal(0, Z_space, (n,d)) # Input matrix of the linear regression, gaussian
        Z[:, 0] = 1.  # intercept value
        Z_3 = np.random.normal(0, Z_space, (m,d)) # validation batch
        Z_3[:, 0] = 1.  # intercept value

        if data == "norm":
            epsilon_n = np.random.normal( np.zeros((n,1)) , 0.1*np.linalg.norm(Z, axis=1).reshape(-1,1) , (n, 1) )
            epsilon_m = np.random.normal( np.zeros((m,1)) , 0.1*np.linalg.norm(Z_3, axis=1).reshape(-1,1) , (m, 1) )

        if data!="random":
            Y = Z @ theta + epsilon_n  # Noisy output of the linear regression
            Y_3 = Z_3 @ theta + epsilon_m # validation batch

        if data == "random":  # data outputs generated randomly
            Y = np.random.normal(0, Z_space, (n, 1))
            Y_3 = np.random.normal(0, Z_space, (m, 1))

    elif model_name== "PolyReg2":
        def f(zeta):
            """
            Hidden generator of the data, non-linear function.
            """
            return zeta ** 2 + 5

        if testing: np.random.seed(3)
        zeta = np.random.uniform(-Z_space, Z_space, (n, 1)) # Input vector, gaussian
        #zeta = np.linspace(-Z_space, Z_space, n).reshape(-1,1)
        #zeta = np.random.normal(0, Z_space, (n, 1))  # Input vector, gaussian
        epsilon_n = np.random.normal(0, .5, (n, 1))  # Noise vector to be added to train/test
        Y = f(zeta) + epsilon_n # Noisy TRUE output
        #Y = f(zeta)   # Noisy TRUE output
        Z = polynomial_basis(zeta) # Model matrix Z (WRONG MODEL ON PURPOSE)

        if testing: np.random.seed(99)
        #zeta_3 = np.random.normal(0, Z_space, (m,1)) # Input vector, gaussian
        zeta_3 = np.random.uniform(-Z_space, Z_space, (m, 1))  # Input vector, gaussian
        #Y_3 = f(zeta_3) + epsilon_m # Noisy TRUE output
        Y_3 = f(zeta_3)  # TRUE output
        Z_3 = polynomial_basis(zeta_3) # Model matrix Z (WRONG MODEL ON PURPOSE)

        plot_polyreg_batches(zeta, Y, zeta_3, Y_3) # plot the actual function's shape

    elif model_name== "Gaussian2":
        Z = np.ones( (n,1) ) # Input vector Z ALWAYS EQUAL TO 1
        Y = Z @ theta + epsilon_n # Noisy output of the linear regression

        Z_3 = np.ones( (m,1) ) # Validation vector Z ALWAYS EQUAL TO 1
        Y_3 = Z_3 @ theta + epsilon_m # validation batch

    Z_1 = Z[0:int(n/2)] # train batch
    Z_2 = Z[int(n/2):n] # test batch

    Y_1 = Y[0:int(n/2)] # train batch
    Y_2 = Y[int(n/2):n] # test batch

    X   = (Z,Y) # Wrap
    X_1 = (Z_1,Y_1) # Wrap
    X_2 = (Z_2,Y_2) # Wrap
    X_3 = (Z_3,Y_3) # Wrap

    return X, X_1, X_2, X_3


def generate_bootstrap(batch):
    """
    Generate a bootstrap version of the batch (random draw with replacement of its entries)
    """
    batch_bootstrap_indices = np.random.choice(batch[0].shape[0], batch[1].size, replace=True) # choose bootstrap indices
    return batch[0][batch_bootstrap_indices], batch[1][batch_bootstrap_indices]  # create bootstrap matrix


def loss(batch, theta, sigma):
    Z, Y = batch  # unwrap X into (Z,Y)
    return 1 / (2 * sigma) * (Y.reshape(-1, 1) - Z.reshape(1,-1) @ theta) ** 2 + 0.5 * np.log(2 * np.pi * sigma)


def empirical_risk(batch, theta, sigma):
    """
    Empirical risk r^batch(theta) for some batch (Z,Y), with theta. theta MUST be a matrix of vectors
    (each vector being one possible theta sampled from the posterior)
    Return: a vector of size _mc_, with the empirical risk computed over the
    whole batch X, but with a different sample of theta for each entry of the output vector.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return 1/(2*sigma) * np.mean( (Y.reshape(-1, 1) - Z @ theta)**2, axis=0) + 0.5 * np.log(2*np.pi*sigma)


############################
# Posterior Gaussian samples
############################

def S_P(alpha, batch):
    """
    Variance of the mutlivariate NIG posterior estimation.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return inv( alpha/Y.size * Z.T @ Z + inv(S_0) )


def mu_P(alpha, batch, S_P):
    """
    Mean of the NIG posterior estimation, computed with given batch.
    S_P is the cov matrix as an argument to gain time.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return S_P @ ( alpha/Y.size * Z.T @ Y + inv(S_0) @ mu_0 )


def a_P(alpha):
    """
    a (shape) param of the inv gamma posterior estimation, computed with given batch.
    """
    return a_0 + alpha/2


def b_P(alpha, batch, mu_P, S_P):
    """
    boot (scale) param of the inv gamma posterior estimation, computed with given batch.
    """
    Z, Y = batch  # unwrap X into (Z,Y)
    return ( b_0 + 0.5 * ( mu_0.T @ inv(S_0) @ mu_0 - mu_P.T @ inv(S_P) @ mu_P + alpha/Y.size * Y.T @ Y ) ).item() # item() to convert to float


def theta_sample(alpha, batch):
    """
    Generates n random samples from β where β ~ NIG(mu, V, a, boot), i.e.:
    β | σ^2 ~ N(mu, σ^2 * V)
        σ^2 ~ IG(a, boot)
    The output is an matrix of _d_ sampled means, times _mc_ samples. Shape = ((d, mc) , (1, mc))
    """
    S_P1 = S_P(alpha, batch) # Cov matrix
    mu_P1 = mu_P(alpha, batch, S_P1).reshape(-1)  # reshape for becoming only 1 dimensional
    a_P1 = a_P(alpha)
    b_P1 = b_P(alpha, batch, mu_P1, S_P1)

    sigma = invgamma.rvs(a=a_P1, scale=b_P1, size=mc)
    theta = np.apply_along_axis(lambda x: np.random.multivariate_normal(mu_P1, x * S_P1), 0, sigma[None, :]).T
    return theta.T, sigma


theta_vec = np.vectorize(theta_sample, excluded=[1, 2])  # Vectorized version of D taking an array of alphas!


############
# D and dD #
############

def D(alpha, E_batch, r_batch):
    """
    Estimator of C(alpha), corresponding to E_batch [ r^(r_batch) ], with batch input Z and output Y
    """
    local_theta_samples, sigmas = theta_sample(alpha, E_batch) #[0] # Get _mc_ new samples of the posterior
    return np.mean( empirical_risk(r_batch, local_theta_samples, sigmas) ) # Approximate expectation using MC


def D1(alpha, E_batch, r_batch):
    """
    Estimator of C(alpha), corresponding to E_batch [ r^(r_batch) ], with batch input Z and output Y
    """
    S_P1 = S_P(alpha, E_batch) # cov matrix
    mu_P1 = mu_P(alpha, E_batch, S_P1) # mean vector
    a_P1 = a_P(alpha)
    b_P1 = b_P(alpha, E_batch, mu_P1, S_P1)

    Z_r, Y_r = r_batch  # unwrap r_batch into (Z,Y)

    thetaoversigma = mu_P1 / gamma(a_P1) * gamma( a_P1 + (d-1)/2 ) / b_P1**((d-1)/2)
    thetaZZthetaoversigma = 1/gamma(a_P1) * ( gamma(a_P1 + d/2) * np.trace( Z_r @ S_P1 @ Z_r.T) / b_P1**(d/2)
                                            + gamma(a_P1 + d/2 -1) * mu_P1.T @ Z_r.T @ Z_r @ mu_P1 / b_P1**(d/2-1) )

    return np.squeeze( 1/(2*Y_r.size) * (
                a_P1 / b_P1 * Y_r.T @ Y_r
              - 2 * Y_r.T @ Z_r @ thetaoversigma
              + thetaZZthetaoversigma )
              + np.log(b_P1) - digamma(a_P1) )


D_vec = np.vectorize(D, excluded=[1,2]) # Vectorized version of D taking an array of alphas!


def dD_MC(alpha, E_batch, r_batch):
    """
    Monte-Carlo estimator of the derivative of D(alpha) with E_batch [ r^(r_batch) ]
    """
    local_theta_samples, local_sigma_samples = theta_sample(alpha, E_batch) # Get _mc_ new samples of the posterior
    return MC_risk_covariance(E_batch, r_batch, empirical_risk, local_theta_samples, local_sigma_samples)


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


def naive(alpha, X, X_1, X_2, X_3):
    """
    Stupid strategy using only the training set
    """
    return D(alpha, X, X)


def optimal(alpha, X, X_1, X_2, X_3):
    """
    Optimal strategy using the large validation set.
    """
    return D(alpha, X, X_3)


def sample_split(alpha, X, X_1, X_2, X_3):
    """
    Exact computation of the sample-split strategy (no MC).
    """
    return D(alpha, X_1, X_2)


def safeBayes(alpha, X, X_1, X_2, X_3, start=1):
    """
    Strategy SafeBayes from Grunwald
    _start_ is the batch number from which we begin to compute (typically 3 or 4)
    """
    Z, Y = X # unwrap X into (Z,Y)
    global_error = 0 # init d/da S(a) = 0 before summing

    for t in np.arange(start, Y.size-1) : # go from t=start to n-1
        batch_t = (Z[:t], Y[:t]) # batch from 0 to t
        S_Pt = S_P(alpha, batch_t)
        mu_Pt = mu_P(alpha, batch_t, S_Pt)
        a_Pt = a_P(alpha)
        b_Pt = b_P(alpha, batch_t, mu_Pt, S_Pt)

        Zt1 = Z[t+1].reshape(-1,1) # make Z[t+1] into a matrix (d,1) for matrix multiplication

        global_error += a_Pt/(2*b_Pt) * ( Y[t+1]**2
                                         + np.trace( Zt1 @ Zt1.T @ S_Pt ) \
                                         + mu_Pt.T @ Zt1 @ Zt1.T @ mu_Pt \
                                         - 2 * Y[t+1] * Zt1.T @ mu_Pt ) \
                                + 1/2 * ( np.log(b_Pt) - digamma(a_Pt) )

    return global_error # S(a)


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
        local_theta_samples, sigmas = theta_sample(alpha, batch_t) # (d*mc) Get _mc_ new samples of the posterior^(t)
        global_error += MC_d_local_error(batch_t, (Z[t+1], Y[t+1]), empirical_risk, loss, local_theta_samples, sigmas) # derivative of the error

    return global_error


#########################################################
# Choice of the optimizer type and the list of strategies
#########################################################

optimizer = scipy_GD
optimizer_name = "scipy"
strategies = np.array([naive_MC, sample_split_MC, bootstrap_MC, safeBayes_MC, bayes, optimal_MC])
strat_names = np.array(["Naive", "Sample split", "Bootstrap", "SafeBayes", "Bayes", r'$\mathcal{R}(\alpha^*)$'])

# PLotting the pdfs for one example
if(d==1): plot_joy(generate_batch, theta_vec)

# Actual optimization (everything happens here)
risks, strats_alphas, D_X_1, D_X_2, D_X_3, grid, all_X_3 = optimize(optimizer, generate_batch, D_vec, strategies, H, G, ten)
#theta_min = np.min( [empirical_risk(X_3, theta, sigma) for X_3 in all_X_3]) # risk of last tentative (all should have same risk)
theta_min = np.min( [ inv(X_3[0].T @ X_3[0]) @ X_3[0].T @ X_3[1] for X_3 in all_X_3] ) # mean squares with no true theta

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