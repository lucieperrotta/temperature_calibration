# Imports
import numpy as np
from scipy.stats import invgamma

def generate_theta(shape, seed=23):
    """
    Args:
        shape: the shape of the parameter theta
        seed: the seed to be used for pseudo random generation
    Returns: a vector pf parameter theta (always the same with same seed)
    """
    np.random.seed(seed)
    theta = np.random.normal(0, 1, shape)  # Secret parameter vector of the log regression

    import time  # stops random.seed to work
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)

    return theta


def generate_sigma(a, b, seed=23):
    """
    Args:
        shape:  shape of sigma
        a: first param for inverse gamma
        b: second param of inverse gamma
        seed: pseudo random seed
    Returns: sigma variance for linreg2
    """
    np.random.seed(seed)
    sigma = invgamma.rvs(a=a, scale=b) # Variance (inv gamma) of the noise added on the output

    import time  # stops random.seed to work
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)

    return sigma


def MC_d_local_error(batch_t, X_t_1, empirical_risk, loss, *args):
    """
    Compute a MC approxmiation of the derivative of the local error.
    Used for computing the derivative of the SafeBayes strategy.
    """
    risk_t = empirical_risk(batch_t, *args) # risk of observations 0 to t
    loss_t_1 = loss(X_t_1, *args) # loss of observation t+1
    return -np.mean(risk_t * loss_t_1) + (np.mean(risk_t) * np.mean(loss_t_1)) # covariance


def MC_risk_covariance(batch1, batch2, empirical_risk, *args):
    """
    Computes the approximate negative covariance between the risks of 2 batches, using MC estimation.
    The expected value is computed with respect to the distribution _theta_.
    _theta_ is a vector, hence a MC approximation is used over all values of theta.
    """
    risk1 = empirical_risk(batch1, *args)
    risk2 = empirical_risk(batch2, *args)
    return -np.mean(risk1 * risk2) + (np.mean(risk1) * np.mean(risk2))


def MC_risk_variance(batch, empirical_risk, *args):
    """
    Computes the approximate variance of the risk of a batch using MC estimation.
    """
    risk = empirical_risk(batch, *args)
    return np.mean( risk**2 ) - np.mean(risk)**2