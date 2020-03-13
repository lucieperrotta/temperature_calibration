"""
This helper files contains all functions for plotting.
For example, plotting the optimal_MC alphas, plotting all alphas values as violin plots, etc.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # fix bug on Raiden
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
from scipy.stats import norm
import joypy
import gif

# Global variables
graph_size = (16,10) # Size of all graphs
colors = ["tomato", "darkorchid", "darkorange", "silver", "pink", "limegreen", "lightgreen", "peru", "brown"]
opacity=.5
linthreshy=0.1 # Limit of log scale around 0
linewidth=3 # width of the curves
legend_size = 21

plt.rcParams.update({'font.size': 22}) # font size
fontname = "Helvetica"


def plot_from_file(file):
	"""
	Plot all the graphs from a saved numpy file.
	"""
	model_name, optimizer_name, strat_names, risks, strats_alphas, theta_min, D_X_1, D_X_2, D_X_3, grid = np.load(file, allow_pickle=True)

	strat_names = np.array(
		["Naive", "Sample-split", "Bootstrap", "SafeBayes", "Bayes", r'$ \mathcal{R}(\alpha^*)$'])

	plot_all(model_name, optimizer_name, strat_names, risks, strats_alphas, theta_min, D_X_1, D_X_2, D_X_3, grid)
	

def plot_all(model_name, optimizer_name, strat_names, risks, strats_alphas, theta_min, D_X_1, D_X_2, D_X_3, grid):
	"""
	Run all the following methods for one model, given strategies, many tentatives.
	"""
	print("Plotting...")

	plot_violin(risks, strat_names, theta_min, name=f"../plots/{model_name} - All boxplots ({optimizer_name}).png")
	plot_overfit(grid, D_X_1[0], D_X_2[0], D_X_3[0], name=f"../plots/{model_name} - Overfit graph ({optimizer_name}).png")
	plot_strategies(grid, D_X_1[0], D_X_2[0], D_X_3[0],	strats_alphas.T[0], strat_names,
					name=f"../plots/{model_name} - All strategies risks comparison ({optimizer_name}).png")

	#for i in range(strat_names.size):
		# plot_one_strategy(grid, D_X_1, D_X_2, D_X_3, alpha_naive, alpha_optimal, alpha_bayes, strats_alphas[i], strat_names[i], risks[i], name=f"../plots/{model_name} - Risks for strategy {strat_names[i]} ({optimizer_name}).png")
		# plot_covariances_one_strategy(grid, strats_gradients[i], strats_alphas[i], alpha_optimal, alpha_bayes, strat_names[i], name=f"../plots/{model_name} - Covariances for strategy {strat_names[i]} ({optimizer_name}).png")

	# Ploting one tentative (the first) of all strategies
	# plot_all_covariances(grid, alpha_optimal[0], alpha_bayes, strats_gradients.T[0], strats_alphas.T[0], strat_names, name=f"../plots/{model_name} - All strategies covariances comparison ({optimizer_name}).png")


def plot_overfit(grid, D_X_1, D_X_2, D_X_3, name):
	"""
	Only plot
	"""
	fig, ax = plt.subplots(figsize=graph_size)
	ax.ticklabel_format(useOffset=False, style='plain')
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
	ax.grid(True, 'minor', 'y', ls='--', lw=.5, c='k', alpha=.3)

	n = int( (grid[-1]+1) / 3)

	plt.plot(grid/n, D_X_1, linewidth = linewidth, color="tomato", label=r'$ {E}_{\theta\sim \pi_\alpha}[r_n(\theta)] $') # plot D(alpha, X_1) train
	plt.plot(grid/n, D_X_3, linewidth = linewidth, color="limegreen", label=r'$ {E}_{\theta\sim \pi_\alpha}[R(\theta)] $') # plot D(alpha, X_3) validation

	plt.xlabel(r"$\alpha / n$")
	plt.ylabel(r'Risk')
	plt.legend(loc="best", prop={'size': legend_size})
	#plt.yscale("symlog", basey=2, subsy=[2,3,4,5,6,7,8,9], linthreshy=linthreshy)
	plt.yscale("log")
	plt.savefig(name)
	plt.show()
	plt.clf()


def plot_strategies(grid, D_X_1, D_X_2, D_X_3, strats_alphas, strat_names, name):
	"""
	Plot the train/test/validation curves of the dataset, as well as the different alphas computed by the strategies,
	and the optimal_MC alpha.
	"""
	fig, ax = plt.subplots(figsize=graph_size)
	ax.ticklabel_format(useOffset=False, style='plain')
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
	ax.grid(True, 'minor', 'y', ls='--', lw=.5, c='k', alpha=.3)

	n = int((grid[-1] + 1) / 3)

	plt.plot(grid/n, D_X_1, linewidth = linewidth, color="tomato", label=r'$ {E}_{\theta\sim \pi_\alpha}[r_n(\theta)] $') # plot D(alpha, X_1) train
	# plt.plot(grid, D_X_2, linewidth = linewidth, color="darkorchid", label=r'$ {E}_{\theta\sim \pi_\alpha^{(1)}}[r_n^{(2)}(\theta)] $') # plot D(alpha, X_2) test
	plt.plot(grid/n, D_X_3, linewidth = linewidth, color="limegreen", label=r'$ {E}_{\theta\sim \pi_\alpha}[R(\theta)] $') # plot D(alpha, X_3) validation

	# Plot training, validation and bayesian alpha
	# ax.axvline(x=alpha_naive, color="red", alpha=opacity, label=r'Naive $ \alpha $')
	# ax.axvline(x=alpha_optimal, color="black", alpha=opacity, label=r'Optimal $ \alpha $')
	#ax.axvline(x=alpha_bayes, color="grey", alpha=opacity, label=r'Bayesian $ \alpha $')

	for i in range(strat_names.size):
		ax.axvline(x=strats_alphas[i]/n, linestyle="--", linewidth = linewidth, color=colors[i], alpha=opacity, label=rf'{strat_names[i]}')

	#plt.xlim(0.03, grid[-1]+0.1) # Not plot the extreme values around alpha=0
	plt.xlabel(r"$\alpha / n$")
	plt.ylabel(r'Risk')
	# plt.title("Optimal values of alpha with different strategies for one tentative")
	plt.legend(loc="best", prop={'size': legend_size})
	plt.yscale("log")
	#plt.yscale("symlog", basey=2, subsy=[2,3,4,5,6,7,8,9], linthreshy=linthreshy)
	plt.savefig(name)
	plt.show()
	plt.clf()


def plot_one_strategy(grid, D_X_1, D_X_2, D_X_3, alpha_naive, alpha_optimal, alpha_bayes, alphas, strat_name, all_risks, name):
	"""
	Plot all the train/test/validation curves of the dataset for one strategy, as well as the different alphas computed for each tentative.
	We also plot the optimal_MC alpha, and report all risks on the left on the Y axis, and draw a violin plot of them.
	"""

	# Set up the axes with gridspec
	fig = plt.figure(figsize=graph_size)
	plotgrid = plt.GridSpec(4, 10, hspace=0, wspace=0)
	main_ax = fig.add_subplot(plotgrid[:, 1:])
	y_hist = fig.add_subplot(plotgrid[:, 0], xticklabels=[], sharey=main_ax)
	total = fig.add_subplot(plotgrid[:, :], xticklabels=[], sharey=main_ax)

	# MAIN PLOT WITH CURVES AND STUFF
	main_ax.axvline(x=alpha_bayes, linewidth = linewidth, color="grey", alpha=opacity, label=r'Bayesian $ \alpha $')

	for ten in range(alpha_optimal.size): # iterate over all tentatives
		#main_ax.plot(grid, D_X_1[ten], color="red", label=r'$ {E}_{\theta\sim \pi_\alpha^{(1)}}[r_n^{(1)}(\theta)] $') # plot D(alpha, X_1) train
		#main_ax.plot(grid, D_X_2[ten], color="purple", label=r'$ {E}_{\theta\sim \pi_\alpha^{(1)}}[r_n^{(2)}(\theta)] $') # plot D(alpha, X_2) test
		main_ax.plot(grid, D_X_3[ten], linewidth = linewidth, color="black", alpha=opacity, label=r'$ {E}_{\theta\sim \pi_\alpha^{(1)}}[r_n^{(3)}(\theta)] $') # plot D(alpha, X_3) validation

		# Plot training and validation alphas, and strategy alphas
		#main_ax.axvline(x=alpha_hat[ten],  color="red", alpha=opacity, label=r'Training $ \hat{\alpha} $')
		# main_ax.axvline(x=alpha_optimal[ten], color="black", alpha=opacity, label=r'Optimal $ \alpha $')
		main_ax.axvline(x=alphas[ten], linewidth = linewidth, color=colors[0], alpha=opacity, label=rf'{strat_name} $ \hat{{\alpha}} $')

		# Plot the risk of each found alpha
		total.patch.set_facecolor('none')
		total.axhline(y=all_risks[ten], linewidth = linewidth, color=colors[0], alpha=opacity)

	#main_ax.legend(loc="best", prop={'size': legend_size})
	#main_ax.set_xlim(0.01, grid[-1]+0.1) # Not plot the extreme values around alpha=0
	main_ax.set_xlabel("alpha")
	main_ax.axes.get_yaxis().set_visible(False)

	# LEFT PLOT WITH THE VIOLIN
	y_hist.patch.set_facecolor('none')
	y_hist = sns.boxplot(data=all_risks.T, ax=y_hist, color=colors[0])
	y_hist.set_ylabel(r'Risk value $ D(\alpha) $')
	y_hist.axes.get_xaxis().set_visible(False)
	#y_hist.axis(ymin=0.99*np.min(all_risks), ymax=1.01*np.max(all_risks), option='on') # have a nice scale, otherwise all crushed

	main_ax.set_title(f"Optimal values of alpha for {strat_name} for all tentatives")
	plt.yscale("symlog", basey=2, subsy=[2,3,4,5,6,7,8,9], linthreshy=linthreshy)
	plt.savefig(name)
	#plt.show()
	plt.clf()


def plot_violin(all_risks, strat_names, theta_min, name):
	"""
	Generate a violin plot for _ten_ values of the risk, for a each _strategy_.
	"""
	fig, ax = plt.subplots(figsize=graph_size)
	ax.ticklabel_format(useOffset=False, style='plain')
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
	ax.grid(True, 'minor', 'y', ls='--', lw=.5, c='k', alpha=.3)

	plt.axhline(y=theta_min, label=r'$\min_\theta R (\theta)$', linestyle="--", color="rosybrown", linewidth=2) # plot the minimum risk for optimal theta

	dataframe = pd.DataFrame(data=all_risks.T, columns=strat_names) # Turn into dataframe to put name to each strategy
	ax = sns.boxplot(data=dataframe, ax=ax, linewidth = 2, palette=sns.cubehelix_palette(8))

	plt.ylim(bottom=0.8*theta_min.item())
	plt.xlabel("Strategy")
	plt.ylabel('Risk')
	plt.legend(loc="upper right", prop={'size': legend_size})
	#plt.title("Risk values of all tentatives for each strategy")
	plt.savefig(name)
	#plt.show()
	plt.clf()


def plot_logreg_data(Z,Y):
	fig , ax = plt.subplots(figsize=graph_size)
	ax.ticklabel_format(useOffset=False, style='plain')
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	plt.scatter(Z[:, 0], Z[:, 1], c=np.squeeze(Y.astype(float)/2+1/4), cmap=cm.autumn, norm=cm.colors.Normalize(vmax=.7, vmin=.3))

	plt.xlabel(r"$Z_0$")
	plt.ylabel(r"$Z_1$")
	#plt.title("Risk values of all tentatives for each strategy")
	plt.savefig("../plots/logreg_data.png", bbox_inches='tight')
	#plt.show()
	plt.clf()


def plot_polyreg_batches(zeta, Y, zeta_3, Y_3):
	"""
	Plot the batches (train/test, and validation)
	"""
	fig , ax = plt.subplots(figsize=graph_size)
	ax.ticklabel_format(useOffset=False, style='plain')
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	n = zeta.size

	plt.scatter(zeta_3, Y_3, alpha=.5, color="cadetblue", label="Original function")
	plt.scatter(zeta, Y, alpha=1, label="Noisy observations", linewidth=3, c="darkorange", marker="X")

	plt.xlabel(r"$\zeta$")
	plt.ylabel(r"$Y$")
	plt.legend(loc="best", prop={'size': legend_size})
	plt.savefig("../plots/polyreg_data.png", bbox_inches='tight')
	plt.show()
	plt.clf()


def plot_polyreg_batches_old(zeta, Y, zeta_3, Y_3):
	"""
	Plot the batches (train/test, and validation)
	"""
	fig = plt.figure(figsize=graph_size)
	n = zeta.size
	zeta_1 = np.squeeze(zeta[0:int(n / 2)])  # train batch
	zeta_2 = np.squeeze(zeta[int(n / 2):n]) # test batch

	Y_1 = np.squeeze(Y[0:int(n / 2)])  # train batch
	Y_2 = np.squeeze(Y[int(n / 2):n])  # test batch

	plt.scatter(zeta_3, Y_3, alpha=.5, color="cadetblue", label="Validation batch")
	plt.plot(np.sort(zeta_1), Y_1[np.argsort(zeta_1)], alpha=1, label="Train batch", c="darkorange", marker="X")
	plt.plot(np.sort(zeta_2), Y_2[np.argsort(zeta_2)], alpha=1, label="Test batch", c="crimson", marker="X")
	plt.title("Generated data for PolyReg")
	plt.legend(loc="best", prop={'size': legend_size})
	plt.savefig("../plots/PolyReg - Generated data.png", bbox_inches='tight')
	plt.show()
	plt.clf()


def plot_joy(generate_batch, theta_vec):
	"""
	Plots the pdf of different posteriors taken between alpha=0 and alpha=multiple of bayes
	Works for the linreg only.
	"""
	X, X_1, X_2, X_3, true_theta = generate_batch(return_theta=True)  # dataset generation
	Z, Y = X  # unwrap X into (Z,Y)
	MAP_theta = np.linalg.inv(Z.T @ Z) @ Z.T @ Y # find the MAP estimation of theta fir LinReg1
	n = X[1].size # get size of batch

	number_x_labels = 9
	x_max_value = max(0, true_theta.item(0)*1.2)
	x_min_value = min(0, true_theta.item(0)*1.2)
	x_axis = np.linspace(x_min_value, x_max_value, 500) # resolution of the x axis for thePDFs
	x_axis_mini = np.squeeze( np.round( np.linspace(x_min_value, x_max_value, num=number_x_labels) , decimals=1)) # ugly trick to set x axis correctly
	x_range = list(range(x_axis.size)) # ugly trick to set x axis correctly
	x_range_mini = np.linspace(0, x_axis.size, num=number_x_labels) # ugly trick to set x axis correctly

	scaled_true_theta =  (x_axis.size-1) / 1.2 if true_theta.item(0) > 0 else (x_axis.size-1) - (x_axis.size-1) / 1.2
	scaled_MAP_theta = scaled_true_theta * MAP_theta / true_theta if true_theta.item(0) > 0 else (x_axis.size-1) - (x_axis.size-1) / 1.2 * MAP_theta / true_theta

	alphas = np.linspace(4*n, 0, num=21)  # generate different values of alpha (lin scale)
	#alphas = np.append(np.round(np.geomspace(10*n, 0.1, num=21), decimals=2), 0.) # generate different values of alpha (log scale)
	muS, Ss = theta_vec(alphas, X, return_params=True) # obtain all means and variances
	pdfs = norm.pdf(x_axis.reshape(-1, 1), muS, Ss) # matrix of x_axis * alphas
	dataframe = pd.DataFrame(data=pdfs, columns=alphas, index=x_axis) # Turn into dataframe to put name to each strategy

	n = 50

	fig, ax = joypy.joyplot(dataframe, kind="values", x_range=x_range, linewidth=.5, labels=alphas, grid=alphas, range_style='own', figsize=graph_size, colormap=cm.autumn)
	ax[-1].set_xticks(x_range_mini) # ugly trick to set x axis correctly
	ax[-1].set_xticklabels(x_axis_mini) # ugly trick to set x axis correctly
	ax[-1].axvline(x=scaled_true_theta, linewidth=linewidth, color="black", label='True theta') # plot true theta
	ax[-1].axvline(x=scaled_MAP_theta, linewidth=linewidth, color="red", label='MAP theta')  # plot map theta

	plt.xlabel(r"$\theta$")
	ax[-1].yaxis.set_label_position("right")
	ax[-1].set_ylabel(r"$\alpha/n$")
	ax[-1].yaxis.set_visible(True)
	ax[-1].yaxis.set_ticks([])
	plt.legend()
	plt.savefig("../plots/JoyPlot_posterior.png")
	plt.show()
	plt.clf()


@gif.frame
def plot_linreg_internal(alpha, E_batch, mu_P, S_P, n, sigma):
	"""
	Plots the points of linreg and the line too.
	"""

	S_P1 = S_P(alpha, E_batch)  # cov matrix
	mu_P1 = mu_P(alpha, E_batch, S_P1)  # mean vector
	X, Y = E_batch  # unwrap
	fig, ax = plt.subplots(figsize=graph_size)
	ax.ticklabel_format(useOffset=False, style='plain')
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	a = .5

	x_grid = np.linspace(np.min(X), np.max(X), num=100)[:, None]
	Phi_X = np.hstack((np.ones((x_grid.shape[0], 1)), x_grid))

	stdev_pred = np.sqrt(np.sum(np.dot(Phi_X, S_P1) * Phi_X, 1)[:, None])
	upper_bound = Phi_X @ mu_P1 + stdev_pred
	lower_bound = Phi_X @ mu_P1 - stdev_pred

	stdev_pred1 = np.sqrt(np.sum(np.dot(Phi_X, S_P1) * Phi_X, 1)[:, None] + sigma)
	upper_bound1 = Phi_X @ mu_P1 + stdev_pred1
	lower_bound1 = Phi_X @ mu_P1 - stdev_pred1

	a = .2

	plt.fill_between(np.squeeze(x_grid[:, 0]), np.squeeze(lower_bound1), np.squeeze(upper_bound1), color=cm.autumn(.5), alpha=a,
					 label='Std dev of predictions')

	plt.plot(x_grid, Phi_X @ mu_P1, '-', color=cm.autumn(1), label='Mean of predictions')

	plt.plot(X[:, 1], Y, 'xk', label='Observations')
	#plt.ylim(2 * np.min(Y), 2 * np.max(Y))
	plt.legend()


def plot_linreg(E_batch, mu_P, S_P, n, sigma):
	i=0
	plt.clf()

	for alpha in [0, n / 3, n, 99999999]:

		plot_linreg_internal(alpha, E_batch, mu_P, S_P, n, sigma)

		plt.savefig(f"../plots/linreg{i}.png", bbox_inches='tight')
		plt.show()
		plt.clf()


def plot_linreg_animate(E_batch, mu_P, S_P, n, sigma):
	frames = []
	for alpha in np.geomspace(start=0.0001, stop=10*n, num=100):
	#for alpha in np.linspace(start=0, stop=3 * n, num=40):
		frame = plot_linreg_internal(alpha, E_batch, mu_P, S_P, n, sigma)
		frames.append(frame)

	gif.save(frames, "../plots/linreg_anim.gif", duration=50)


@gif.frame
def plot_polyreg_internal(alpha, E_batch, mu_P, S_P, n, sigma, d, f):

		x_grid = np.linspace(-2.5, 2.5, 300)
		f_grid = f(x_grid) # true function f
		X_grid = np.vstack(x_grid ** i for i in range(d)).T
		S_P1 = S_P(alpha, E_batch)  # cov matrix
		mu_P1 = mu_P(alpha, E_batch, S_P1)  # mean vector
		X, Y = E_batch  # unwrap
		fig, ax = plt.subplots(figsize=graph_size)
		ax.ticklabel_format(useOffset=False, style='plain')
		# Hide the right and top spines
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		# Only show ticks on the left and bottom spines
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')

		a=.5

		stdev_pred = np.sqrt(np.sum(np.dot(X_grid, S_P1) * X_grid, 1)[:, None])
		upper_bound = X_grid @ mu_P1 + 2*stdev_pred
		lower_bound = X_grid @ mu_P1 - 2*stdev_pred

		stdev_pred1 = np.sqrt(np.sum(np.dot(X_grid, S_P1) * X_grid, 1)[:, None] + sigma )
		upper_bound1 = X_grid @ mu_P1 + 2*stdev_pred1
		lower_bound1 = X_grid @ mu_P1 - 2*stdev_pred1

		color_list = sns.cubehelix_palette(6)

		plt.plot(x_grid, f_grid, '-', linewidth=3, color="blue", alpha=.1, label=r"Noiseless function $f(\zeta)$")

		plt.fill_between(np.squeeze(x_grid), np.squeeze(lower_bound1), np.squeeze(upper_bound1), color=color_list[1], alpha=a, label='Double std dev of predictions')

		#plt.fill_between(np.squeeze(x_grid), np.squeeze(lower_bound), np.squeeze(upper_bound), color=color_list[3],	 alpha=a, label='Std dev of posterior')
		#plt.fill_between(np.squeeze(x_grid), np.squeeze(lower_bound), np.squeeze(lower_bound1), color=color_list[1], alpha=a, label='Std dev of the data uncertainty')
		#plt.fill_between(np.squeeze(x_grid), np.squeeze(upper_bound1), np.squeeze(upper_bound), color=color_list[1], alpha=a)

		plt.plot(x_grid, X_grid @ mu_P1, '-', linewidth=3, color=color_list[3], label="Mean of predictions")

		plt.plot(X[:,1], Y, 'xk', label="Observations")

		plt.ylim(-2*np.min(Y), 2*np.max(Y))
		plt.ylim(0 , 10)
		#plt.ylim(-100, 100)

		plt.xlabel(r"$\zeta$")
		plt.ylabel(r"$Y$")
		plt.title(f"a/n={np.format_float_scientific( alpha/n, precision=2) }")

		handles, labels = ax.get_legend_handles_labels()
		order = [0,2,1,3]
		plt.legend(np.asarray(handles)[order], np.asarray(labels)[order], loc="lower center", prop={'size': legend_size})


def plot_polyreg(E_batch, mu_P, S_P, n, sigma, d, f):

	i=0
	plt.clf()

	for alpha in [0, n*1e-5, n, 1e+10]:

		plot_polyreg_internal(alpha, E_batch, mu_P, S_P, n, sigma, d, f)

		plt.savefig(f"../plots/polyreg{i}.png", bbox_inches='tight')
		plt.show()
		plt.clf()
		i += 1


def plot_polyreg_animate(E_batch, mu_P, S_P, n, sigma, d, f):

	frames = []
	for alpha in np.geomspace(start=1e-5, stop=1e+10, num=600):
	#for alpha in np.linspace(start=0, stop=3 * n, num=40):
		frame = plot_polyreg_internal(alpha, E_batch, mu_P, S_P, n, sigma, d, f)
		frames.append(frame)

	gif.save(frames, "../plots/polyreg_anim.gif", duration=50)