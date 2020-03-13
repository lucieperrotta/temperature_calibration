# Libraries imports
import random
import sys

from numpy.linalg import inv

# Helper functions
from covariance import *
from optimizers import *
from plot import *

###################
# MODEL SELECTION #
###################

# First argument of this python script is the Model, to be passed in command line !!!!!
model_name = (sys.argv[1]) # Model can be "LinReg", "PolyReg", "Gaussian"
model_list = ["LinReg", "PolyReg", "Gaussian"]
if model_name not in model_list:
    raise NameError(f"Unknown model. Try one of those: {model_list}")


####################
# MODEL PARAMETERS #
####################

d = 11 # size of the param vector to be estimated
n = 11 # size of train/test batch
m = 10000 # size of truth batch
mc = 1 # number of Monte_Carlo samples
boot = 1 # number of bootstraps draws

sigma = .0001 # Variance of the noise added on the output
Z_space = 1 # variance of the generator of the input Z
theta = generate_theta((d,1)) # True parameter theta


if model_name== "LinReg":
    mu_0 = np.full((d, 1), 0)   # Prior Gaussian 0 mean
    S_0 = np.identity(d)*1 # Prior Gaussian cov identity matrix

elif model_name== "PolyReg":
    mu_0 = np.full((d, 1), 0) # Prior Gaussian mean vector
    S_0 = np.diag(np.array([0.5**k for k in range(d)]))*1 # Prior Gaussian cov matrix !
    #S_0 = np.identity(d)

elif model_name== "Gaussian":
    mu_0 = np.array([0]) # Prior scalar 0 mean
    S_0 = np.array([1]).reshape(1, 1) # Prior scalar cov matrix

H = int(3 * n ) # grid [search] size
G = H/300 # grid [search] resolution

ten = 1 # Number of tentatives of each strategy to be averaged together

alpha_bayes = np.clip(n, 0, H) # Value of bayesian alpha for plotting


def f(zeta):
    """
    Hidden generator of the data, non-linear function.
    """
    # return np.sin(5 * zeta)
    # return .5*(.2+.2*zeta + np.sin(1.125*2*2*np.pi*zeta) )
    return zeta ** 2 + 5
    # return np.sin(300*zeta) * zeta**2


def polynomial_basis(zeta):
    """
    Helper function to turn a vector beta into a matrix Z.
    Z is composed of the vector beta at all powers from 0 to d-1.
    """
    return np.vander(zeta.reshape(-1), N=d, increasing=True) # Vandermonde matrix


def generate_batch(testing=True, return_theta=False):
    """
    Generates a new train/test/validation set batch, for a new tentative.
    If testing is true, the same values are always generated. Otherwise, it's random.
    """
    if testing: np.random.seed(76)
    epsilon_n = np.random.normal(0, sigma, (n, 1)) # Noise vector to be added to train/test
    if testing: np.random.seed(54)
    epsilon_m = np.random.normal(0, sigma, (m, 1)) # Noise vector to be added to validation

    if model_name== "LinReg":
        if testing: np.random.seed(2)
        Z = np.random.normal(0, Z_space, (n,d)) # Input matrix of the linear regression, gaussian
        Z[:, 0] = 1. # intercept value
        Y = Z @ theta + epsilon_n # Noisy output of the linear regression

        if testing: np.random.seed(7)
        Z_3 = np.random.normal(0, Z_space, (m,d)) # validation batch
        Z_3[:, 0] = 1.  # intercept value
        Y_3 = Z_3 @ theta + epsilon_m # validation batch

    elif model_name== "PolyReg":
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

    elif model_name== "Gaussian":
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

    if return_theta:
        return X, X_1, X_2, X_3, theta
    else:
        return X, X_1, X_2, X_3


def generate_bootstrap(batch):
    """
    Generate a bootstrap version of the batch (random draw with replacement of its entries)
    """
    batch_bootstrap_indices = np.random.choice(batch[0].shape[0], batch[1].size, replace=True) # choose bootstrap indices
    return batch[0][batch_bootstrap_indices], batch[1][batch_bootstrap_indices]  # create bootstrap matrix


def loss(batch, theta):
    Z, Y = batch # unwrap X into (Z,Y)
    return 1/(2*sigma) * (Y.reshape(-1, 1) - Z @ theta)**2


def empirical_risk(batch, theta):
    """
    Empirical risk r^batch(theta) for some batch (Z,Y), with theta. theta MUST be a matrix of vectors
    (each vector being one possible theta sampled from the posterior)
    Return: a vector of size _mc_, with the empirical risk computed over the
    whole batch X, but with a different sample of theta for each entry of the output vector.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return 1/(2*sigma) * np.mean( (Y.reshape(-1, 1) - Z @ theta)**2, axis=0)


############################
# Posterior Gaussian samples
############################

def S_P(alpha, batch):
    """
    Variance of the mutlivariate Gaussian posterior estimation.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return inv( alpha/(sigma*Y.size) * Z.T @ Z + inv(S_0) )


def mu_P(alpha, batch, S_P):
    """
    Mean of the Gaussian posterior estimation, computed with given batch.
    S_P is the cov matrix as an argument to gain time.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return S_P @ ( alpha/(sigma*Y.size) * Z.T @ Y + inv(S_0) @ mu_0 )


def dS_P(batch, S_P):
    """
    Derivative of variance of the multivariate Gaussian posterior estimation.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return - 1/(sigma*Y.size) * S_P @ Z.T @ Z @ S_P # Derivative of S_P wrt alpha


def dmu_P(batch, mu_P, S_P):
    """
    Derivative of the mean of the mutlivariate Gaussian posterior estimation.
    """
    Z, Y = batch # unwrap X into (Z,Y)
    return 1/(sigma*Y.size) * S_P @ Z.T @ (Y - Z @ mu_P) # Derivative of mu_P wrt alpha


def theta_sample(alpha, batch, return_params=False):
    """
    Returns mc random samples of the posterior mean estimator, for doing Monte Carlo approx.
    The output is an matrix of _d_ sampled means, times _mc_ samples. Shape = (d, mc)
    """
    S_P1 = S_P(alpha, batch) # Cov matrix
    mu_P1 = mu_P(alpha, batch, S_P1).reshape(-1)  # reshape for becoming only 1 dimensional

    if return_params :
        return mu_P1, S_P1
    else:
        return np.random.multivariate_normal(mu_P1, S_P1, mc).T


theta_vec = np.vectorize(theta_sample, excluded=[1, 2])  # Vectorized version of D taking an array of alphas!


############
# D and dD #
############

def D2(alpha, E_batch, r_batch):
    """
    Estimator of C(alpha), corresponding to E_batch [ r^(r_batch) ], with batch input Z and output Y
    """
    local_theta_samples = theta_sample(alpha, E_batch) # Get _mc_ new samples of the posterior
    return np.mean( empirical_risk(r_batch, local_theta_samples) ) # Approximate expectation using MC


def D(alpha, E_batch, r_batch):
    """
    Estimator of C(alpha), corresponding to E_batch [ r^(r_batch) ], with batch input Z and output Y
    """
    Z_r, Y_r = r_batch # unwrap r_batch into (Z,Y)    
    S_P1 = S_P(alpha, E_batch) # cov matrix
    mu_P1 = mu_P(alpha, E_batch, S_P1) # mean vector

    return np.squeeze( 1/(2*sigma*Y_r.size) * (
                Y_r.T @ Y_r 
              - 2 * Y_r.T @ Z_r @ mu_P1
              + np.trace( Z_r.T @ Z_r @ S_P1 )
              + mu_P1.T @ Z_r.T @ Z_r @ mu_P1 ) )


D_vec = np.vectorize(D, excluded=[1,2]) # Vectorized version of D taking an array of alphas!


def dD(alpha, E_batch, r_batch):
    """
    Helper function for computing the exact derivative of D(alpha) with E_batch [ r^(r_batch) ]
    """
    Z_r, Y_r = r_batch # Unwrap r_batch

    S_P1 = S_P(alpha, E_batch) # get the value of S_P
    mu_P1 = mu_P(alpha, E_batch, S_P1) # get the value of mu_P

    dS_P1 = dS_P(E_batch, S_P1) # Derivative of S_P wrt alpha
    dmu_P1 = dmu_P(E_batch, mu_P1, S_P1) # Derivative of mu_P wrt alpha

    # if model_name == "PolyReg": plt.plot(np.sort(Z_r[:, 1]), np.squeeze(Z_r @ mu_P1)[np.argsort(Z_r[:, 1])])  # Plot for each alpha the estimated f(zeta)

    return np.squeeze( 1/(2*sigma*Y_r.size) * (
            - 2 * Y_r.T @ Z_r @ dmu_P1
            + np.trace( Z_r.T @ Z_r @ dS_P1 )
            + 2 * mu_P1.T @ Z_r.T @ Z_r @ dmu_P1 ) )


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
    # Plot example when d=2
    # if d==2 and model_name=="LinReg" and alpha==1 : plot_linreg(X, mu_P, S_P, n, sigma)
    # if model_name == "PolyReg" and alpha == 1: plot_polyreg_animate(X, mu_P, S_P, n, sigma, d, f)
    return D(alpha, X_1, X_2)


def bootstrap_MC(alpha, X, X_1, X_2, X_3):
    """
    Gradient of the bootstrap strategy (no MC). To be used with handmade SGD.
    """
    return np.mean( np.array([dD(alpha, generate_bootstrap(X), X) for i in range(boot)]) )


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

        Zt1 = Z[t+1].reshape(-1,1) # make Z[t+1] into a matrix (d,1) for matrix multiplication

        global_error += 1/(2*sigma) * ( Y[t+1]**2
                                         + np.trace( Zt1 @ Zt1.T @ S_Pt ) \
                                         + mu_Pt.T @ Zt1 @ Zt1.T @ mu_Pt \
                                         - 2 * Y[t+1] * Zt1.T @ mu_Pt )

    return global_error # S(a)


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
strategies = np.array([naive, sample_split, bootstrap_MC, safeBayes, bayes, optimal])
strat_names = np.array(["Naive", "Sample split", "Bootstrap", "SafeBayes", "Bayes", r'$\mathcal{R}(\alpha^*)$'])

# PLotting the pdfs for one example
if(d==1): plot_joy(generate_batch, theta_vec)

# Actual optimization (everything happens here)
risks, strats_alphas, D_X_1, D_X_2, D_X_3, grid, all_X_3 = optimize(optimizer, generate_batch, D_vec, strategies, H, G, ten)

if(model_name=="PolyReg"): theta =  inv(all_X_3[0][0].T @ all_X_3[0][0]) @ all_X_3[0][0].T @ all_X_3[0][1]  # mean squares with no true theta
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