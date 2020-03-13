"""
This helper file contains different optimization methods for finding optimal_MC alphas with different strategies.
For instance: grid search, gradient descent, etc.
"""

import matplotlib
matplotlib.use('Agg') # fix bug on Raiden
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def optimize(optimizer, generate_batch, D_vec, strategies, H, G, ten):
    """
    Big pipeline for computing everything, each strategy with all tentatives, get all alphas, everything.

    Outputs:
    - optis[strategy, tentative]: each entry is a tuple of the covariance array, and the optimal_MC alpha found.
    - risks[strategy, tentative]: each entry is the validation risk value of a strategy's result for a given tentative
    - strats_gradients[strategy, tentative]: each entry is the covariance array (first half of optis)
    - strats_alphas[strategy, tentative]: each entry is the found alpha (second half of optis)
    - D_X_1,2,3[tentative]: the curve of the train/test/validation risk for a given tentative
    - alpha_naive[tentative]: the train alpha for each tentative
    - alpha_optimal[tentative]: the validation alpha for each tentative
    - grid: an array of all the points of the grid
    """
    print("Optimizing...")
    nb_strat = strategies.size # number of different strategies

    risks = np.empty([nb_strat, ten], dtype="object") # Validation risks computed for each alpha given by each tentative and each strat
    # strats_gradients = np.empty([nb_strat, ten], dtype="object") # each complete gradient curve, for each tentative of each strategy
    strats_alphas = np.empty([nb_strat, ten], dtype="object") # each found alpha of each tentative of each strategy
    all_X_3 = np.empty([ten], dtype="object") # all batches X_3
    
    D_X_1 = np.empty([ten], dtype=object) # DX1 for each tentative
    D_X_2 = np.empty([ten], dtype=object) # DX2 for each tentative
    D_X_3 = np.empty([ten], dtype=object) # DX3 for each tentative

    # Do _ten_ tentatives with various datasets
    for tentative in range(ten): 

        print(f"  Tentative {tentative} - Running strategy ", end="", flush=True)
        X, X_1, X_2, X_3 = generate_batch() # dataset generation
        all_X_3[tentative] = X_3
        grid, D_X_1[tentative], D_X_2[tentative], D_X_3[tentative] = grid_generation(X, X_1, X_2, X_3, D_vec, H, G)

        # Quick plot
        plt.plot(grid, D_X_1[tentative], color="red", label=r'$ {E}_{\theta\sim \pi_\alpha}[r_n(\theta)] $')  # plot D(alpha, X_1) train
        plt.plot(grid, D_X_2[tentative], color="purple", label=r'$ {E}_{\theta\sim \pi_\alpha^{(1)}}[r_n^{(2)}(\theta)] $')  # plot D(alpha, X_2) test
        plt.plot(grid, D_X_3[tentative], color="black", label=r'$ {E}_{\theta\sim \pi_\alpha}[r_n^{(3)}(\theta)] $')  # plot D(alpha, X_3) validation
        plt.savefig("../plots/risks.png")
        plt.show()
        plt.clf()

        # Do all strategies
        for strat in range(nb_strat):
            # Completing the covariances and alphas value
            print(f"{strat} ", end="", flush=True)
            # SHOULD BE REMOVED FOR FASTER COMPUTATION
            #strats_gradients[strat, tentative] = grid_alphas(strategies[strat], grid, X, X_1, X_2, X_3) # values of all alphas over the grid
            #plt.title(f"{strategies[strat].__name__}")
            #plt.show()
            strats_alphas[strat, tentative] = optimizer(strategies[strat], H, grid, X, X_1, X_2, X_3)[0]
            risks[strat, tentative] = D_vec(strats_alphas[strat, tentative], X, X_3).item()
        print("", flush=True)

    return risks, strats_alphas, D_X_1, D_X_2, D_X_3, grid, all_X_3


def grid_generation(X, X_1, X_2, X_3, D_vec, H, G):
    """
    Generates the train/test/validation curves over the given grid.
    This works for one tentative at the time.
    Also generates the training alpha_naive and the validation alpha_optimal.

    Outputs:
    - Alpha_naive is the optimal_MC alpha of the training set.
    - Alpha_optimal is the optimla alpha of the (very large) validation set.
    """
    # Creation of grid search over alpha
    grid = np.arange(0, H, G) # grid search init
    grid_size = grid.size

    # Train, test, and validation curves: E_alpha1 [ r^(X_1,2,3) ]
    D_X_1 = np.array( D_vec(grid, X, X) ) # train
    D_X_2 = np.array( D_vec(grid, X_1, X_2) ) # test
    D_X_3 = np.array( D_vec(grid, X, X_3) ) # validation

    return grid, D_X_1, D_X_2, D_X_3


def grid_alphas(strategy, grid, X, X_1, X_2, X_3):
    """
    Local function going over the grid, and returning an array of the value of alpha for each node of the grid.
    Note that the strategy is averaged using averaged_slope to make the GD work.
    """
    strategy_vec = np.vectorize(strategy, excluded=[1,2,3,4])
    #return np.array( averaged_slope(strategy_vec, grid, X, X_1, X_2, X_3) ) # values of all alphas on grid
    return np.array( strategy_vec(grid, X, X_1, X_2, X_3) ) # bypass the averaging
    #return np.array( [strategy(aaa, X, X_1, X_2, X_3) for aaa in grid] ) # bypass the averaging


def averaged_slope(strategy_vec, alpha, X, X_1, X_2, X_3, width=0.2, MA=10):
    """
    Computes the averaged slope of the function strategy at value alpha.
    This is a denoiser function.
    Warning: this is pretty slow when MA is more than 2.
    This is like a moving average working for a given function instead of a given array.
    _width_ is the width of the interval on which to average.
    _MA_ is the number of uniformly distributed points where to compute the function.
    """
    local_points = np.linspace(alpha-width/2, alpha+width/2, MA)
    return np.mean( strategy_vec(local_points, X, X_1, X_2, X_3) )


##############
# OPTIMIZERS #
##############

def grid_search(strategy, H, grid, X, X_1, X_2, X_3):
    """
    Runs a grid search over H for a given strategy, then takes the minimum alpha.
    The result is always different due to the random nature of the strategies.
    """
    strat_gradients = grid_alphas(strategy, grid, X, X_1, X_2, X_3)  # values of all alphas over the grid
    alpha_tilde = np.argmin(np.abs(strat_gradients)) * H / grid.size  # find the optimal_MC value of alpha (closest to 0)
    return [alpha_tilde] # put in list so that in can access it using [0] later


def SGD(strategy, H, grid, X, X_1, X_2, X_3, learning_rate=2, max_iter=300):
    """
    Constrained "stochastic" gradient descent, taking the strategy = the gradient function as argument.
    The returned optimal_MC value is bound between 0 and H (the grid size).
    The result is always different (stochastic) due to the random nature of the strategies.
    learning_rate : 50 for log reg
    """
    temp_alpha = H/3 # initial alpha
    gradients = [temp_alpha]
    for i in range(1, max_iter):
        lr = (H*learning_rate/i**0.5) # parameter H in the learning rate is for scaling its speed to the grid size
        temp_alpha = np.clip( temp_alpha - lr * strategy(temp_alpha, X, X_1, X_2, X_3),  0, H)
        gradients.append(temp_alpha)


    plt.plot(gradients, label=strategy.__name__)
    plt.legend(loc="best", prop={'size': 10})
    plt.xlabel("iteration")
    plt.ylabel("alpha")
    plt.savefig("../plots/SGD.png")
    return temp_alpha, np.array(gradients)


def scipy_GD(strategy, H, grid, X, X_1, X_2, X_3, initial_alpha=1):
    """
    Use scipy's minimizer to compute the optimal alpha.
    Takes the function D directly as strategy and not the derivative!!!
    Doesnt work for noisy function such as bootstrap, so we special case them.
    """
    if "MC" in strategy.__name__ : # Noisy functions (MC) special case
        return SGD(strategy, H, grid, X, X_1, X_2, X_3)
    else: # Non-noisy case
        return minimize(strategy, x0=initial_alpha, args=(X, X_1, X_2, X_3), bounds=((0, H),) ).get("x")