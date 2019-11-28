#!/usr/bin/env python
# coding: utf-8

# In[8]:



from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools

from setup_problem import load_problem

class RidgeRegression(BaseEstimator, RegressorMixin):
	""" ridge regression"""

	def __init__(self, l2reg=1):
		if l2reg < 0:
			raise ValueError('Regularization penalty should be at least 0.')
		self.l2reg = l2reg

	def fit(self, X, y=None):
		n, num_ftrs = X.shape
		# convert y to 1-dim array, in case we're given a column vector
		y = y.reshape(-1)
		def ridge_obj(w):
			predictions = np.dot(X,w)
			residual = y - predictions
			empirical_risk = np.sum(residual**2) / n
			l2_norm_squared = np.sum(w**2)
			objective = empirical_risk + self.l2reg * l2_norm_squared
			return objective
		self.ridge_obj_ = ridge_obj

		w_0 = np.zeros(num_ftrs)
		self.w_ = minimize(ridge_obj, w_0).x
		return self

	def predict(self, X, y=None):
		try:
			getattr(self, "w_")
		except AttributeError:
			raise RuntimeError("You must train classifer before predicting data!")
		return np.dot(X, self.w_)

	def score(self, X, y):
		# Average square error
		try:
			getattr(self, "w_")
		except AttributeError:
			raise RuntimeError("You must train classifer before predicting data!")
		residuals = self.predict(X) - y
		return np.dot(residuals, residuals)/len(y)



def compare_our_ridge_with_sklearn(X_train, y_train, l2_reg=1):
	# First run sklearn ridge regression and extract the coefficients
	from sklearn.linear_model import Ridge
	# Fit with sklearn -- need to multiply l2_reg by sample size, since their
	# objective function has the total square loss, rather than average square
	# loss.
	n = X_train.shape[0]
	sklearn_ridge = Ridge(alpha=n*l2_reg, fit_intercept=False, normalize=False)
	sklearn_ridge.fit(X_train, y_train)
	sklearn_ridge_coefs = sklearn_ridge.coef_

	# Now run our ridge regression and compare the coefficients to sklearn's
	ridge_regression_estimator = RidgeRegression(l2reg=l2_reg)
	ridge_regression_estimator.fit(X_train, y_train)
	our_coefs = ridge_regression_estimator.w_

	print("Hoping this is very close to 0:{}".format(np.sum((our_coefs - sklearn_ridge_coefs)**2)))

def do_grid_search_ridge(X_train, y_train, X_val, y_val):
	# Now let's use sklearn to help us do hyperparameter tuning
	# GridSearchCv.fit by default splits the data into training and
	# validation itself; we want to use our own splits, so we need to stack our
	# training and validation sets together, and supply an index
	# (validation_fold) to specify which entries are train and which are
	# validation.
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

	# Now we set up and do the grid search over l2reg. The np.concatenate
	# command illustrates my search for the best hyperparameter. In each line,
	# I'm zooming in to a particular hyperparameter range that showed promise
	# in the previous grid. This approach works reasonably well when
	# performance is convex as a function of the hyperparameter, which it seems
	# to be here.
	param_grid = [{'l2reg':np.unique(np.concatenate((10.**np.arange(-6,1,1),
										   np.arange(1,3,.3)
											 ))) }]

	ridge_regression_estimator = RidgeRegression()
	grid = GridSearchCV(ridge_regression_estimator,
						param_grid,
						return_train_score=True,
						cv = PredefinedSplit(test_fold=val_fold),
						refit = True,
						scoring = make_scorer(mean_squared_error,
											  greater_is_better = False))
	grid.fit(X_train_val, y_train_val)

	df = pd.DataFrame(grid.cv_results_)
	# Flip sign of score back, because GridSearchCV likes to maximize,
	# so it flips the sign of the score if "greater_is_better=FALSE"
	df['mean_test_score'] = -df['mean_test_score']
	df['mean_train_score'] = -df['mean_train_score']
	cols_to_keep = ["param_l2reg", "mean_test_score","mean_train_score"]
	df_toshow = df[cols_to_keep].fillna('-')
	df_toshow = df_toshow.sort_values(by=["param_l2reg"])
	return grid, df_toshow

def compare_parameter_vectors(pred_fns):
	# Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
	# "coefs" key
	fig, axs = plt.subplots(len(pred_fns),1, sharex=True)
	num_ftrs = len(pred_fns[0]["coefs"])
	for i in range(len(pred_fns)):
		title = pred_fns[i]["name"]
		coef_vals = pred_fns[i]["coefs"]
		axs[i].bar(range(num_ftrs), coef_vals)
		axs[i].set_xlabel('Feature Index')
		axs[i].set_ylabel('Parameter Value')
		axs[i].set_title(title)

	fig.subplots_adjust(hspace=0.3)
	return fig

def plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best"):
	# Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
	# "preds" key. The value corresponding to the "preds" key is an array of
	# predictions corresponding to the input vector x. x_train and y_train are
	# the input and output values for the training data
	fig, ax = plt.subplots()
	ax.set_xlabel('Input Space: [0,1)')
	ax.set_ylabel('Action/Outcome Space')
	ax.set_title("Prediction Functions")
	plt.scatter(x_train, y_train, label='Training data')
	for i in range(len(pred_fns)):
		ax.plot(x, pred_fns[i]["preds"], label=pred_fns[i]["name"])
	legend = ax.legend(loc=legend_loc, shadow=True)
	return fig

def plot_confusion_matrix(cm, title, classes):      
	 plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)     
	 plt.title(title)       
	 plt.colorbar()     
	 tick_marks = np.arange(len(classes))       
	 plt.xticks(tick_marks, classes, rotation=45)       
	 plt.yticks(tick_marks, classes)        

	 thresh = cm.max() / 2.        
	 for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):     
		 plt.text(j, i, format(cm[i, j], 'd'),      
				  horizontalalignment="center",     
				  color="white" if cm[i, j] > thresh else "black")      

	 plt.tight_layout()        
	 plt.ylabel('True label')       
	 plt.xlabel('Predicted label')

    
    
lasso_data_fname = "lasso_data.pickle"
x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

# Generate features
X_train = featurize(x_train)
X_val = featurize(x_val)

#Visualize training data
fig, ax = plt.subplots()
ax.imshow(X_train)
ax.set_title("Design Matrix: Color is Feature Value")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Example Number")
plt.show(block=False)

# Compare our RidgeRegression to sklearn's.
compare_our_ridge_with_sklearn(X_train, y_train, l2_reg = 1.5)

# Do hyperparameter tuning with our ridge regression
grid, results = do_grid_search_ridge(X_train, y_train, X_val, y_val)
print(results)

# Plot validation performance vs regularization parameter
fig, ax = plt.subplots()
#    ax.loglog(results["param_l2reg"], results["mean_test_score"])
ax.semilogx(results["param_l2reg"], results["mean_test_score"])
ax.grid()
ax.set_title("Validation Performance vs L2 Regularization")
ax.set_xlabel("L2-Penalty Regularization Parameter")
ax.set_ylabel("Mean Squared Error")
fig.show()

# Let's plot prediction functions and compare coefficients for several fits
# and the target function.
pred_fns = []
x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
name = "Target Parameter Values (i.e. Bayes Optimal)"
pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })

l2regs = [0, grid.best_params_['l2reg'], 1]
# 	l2regs = [0, grid.best_params_['l2reg'], 0.1, 1]
X = featurize(x)
for l2reg in l2regs:
    ridge_regression_estimator = RidgeRegression(l2reg=l2reg)
    ridge_regression_estimator.fit(X_train, y_train)
    name = "Ridge with L2Reg="+str(l2reg)
    pred_fns.append({"name":name,
                     "coefs":ridge_regression_estimator.w_,
                     "preds": ridge_regression_estimator.predict(X) })

f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
f.show()

f = compare_parameter_vectors(pred_fns)
f.show()


##Sample code for plotting a matrix
## Note that this is a generic code for confusion matrix
## You still have to make y_true and y_pred by thresholding as per the insturctions in the question.
# y_true = [1, 0, 1, 1, 0, 1]
# y_pred = [0, 0, 1, 1, 0, 1]
# eps = 1e-1;
# cnf_matrix = confusion_matrix(y_true, y_pred)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(eps), classes=["Zero", "Non-Zero"])
# plt.show()





# ### 2.1  Run ridge regression on the provided training dataset. 
# Choose the λ that minimizes the empirical risk (i.e. the average square loss) on the validation set. Include a table of the parameter values you tried and the validation performance for each. Also include a plot of the results

# In[27]:


lasso_data_fname = "lasso_data.pickle"
x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

# Generate features
X_train = featurize(x_train)
X_val = featurize(x_val)


# Do hyperparameter tuning with our ridge regression
grid, results = do_grid_search_ridge(X_train, y_train, X_val, y_val)
print(results)

# Plot validation performance vs regularization parameter
fig, ax = plt.subplots()
#    ax.loglog(results["param_l2reg"], results["mean_test_score"])
ax.semilogx(results["param_l2reg"], results["mean_test_score"])
ax.grid()
ax.set_title("Validation Performance vs L2 Regularization")
ax.set_xlabel("L2-Penalty Regularization Parameter")
ax.set_ylabel("Mean Squared Error")
fig.show()


# The graph shows that the Validation Performance is best when our L2 is almost $10^{-2}$, which is equal to 0.01. At this point, the Mean Square Error is 0.141887.

# ### 2.2 Now we want to visualize the prediction functions. 
# On the same axes, plot the following: the
# training data, the target function, an unregularized least squares fit (still using the featurized
# data), and the prediction function chosen in the previous problem. Next, along the lines of the
# bar charts produced by the code in compare_parameter_vectors, visualize the coefficients for
# each of the prediction functions plotted, including the target function. Describe the patterns,
# including the scale of the coefficients, as well as which coefficients have the most weight

# In[9]:


# Let's plot prediction functions and compare coefficients for several fits
# and the target function.
pred_fns = []
x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
name = "Target Parameter Values (i.e. Bayes Optimal)"
pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })

l2regs = [0, grid.best_params_['l2reg'], 1]
# 	l2regs = [0, grid.best_params_['l2reg'], 0.1, 1]
X = featurize(x)
weights = []
for l2reg in l2regs:
    ridge_regression_estimator = RidgeRegression(l2reg=l2reg)
    ridge_regression_estimator.fit(X_train, y_train)
    name = "Ridge with L2Reg="+str(l2reg)
    pred_fns.append({"name":name,
                     "coefs":ridge_regression_estimator.w_,
                     "preds": ridge_regression_estimator.predict(X) })
    weights.append(ridge_regression_estimator.w_)

f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
f.show()

f = compare_parameter_vectors(pred_fns)
f.show()


# The highest weights belong to the unregularized equation. As we increase the value of lambda, the sparcity of the weights decreases, and the weights decrease. When lambda is 0, our weights are the highest, and our sparcity is the highest as well. This means that only a few features (parameters) are given importance in our equation. 
# 
# The bar graphs show that the coefficients with the most weight are near 50, between 150-200. The lowest weights, on the other hand, are more concentrated around 250-350. 
# 
# 

# 2.3 

# In[13]:


#pass
# lets choose lambda 0.01


epsilons = [1e-6, 1e-3, 1e-1]

# choose lambda = 0.01
ridge_regression_estimator_001 = RidgeRegression(l2reg=0.01)
estimator = ridge_regression_estimator_001.fit(X_train, y_train).w_


for epsilon in range(len(epsilons)): 
    y_true = [1 if abs(coef) > 0 else 0 for coef in coefs_true] 
    y_pred = [1 if abs(e) > epsilons[epsilon] else 0 for e in estimator]
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(epsilon), classes=["Zero", "Non-Zero"])
    plt.show()
    
    


# ### 3.1 Experiments with the shooting algorithm

# 3.1.1 The algorithm as described above is not ready for a large dataset (at least if it has being implemented in Python) because of the implied loop in the summation signs for the expressions
# for aj and cj . Give an expression for computing aj and cj using matrix and vector operations,
# without explicit loops. This is called “vectorization” and can lead to dramatic speedup when
# implemented in languages such as Python, Matlab, and R. Write your expressions using X,
# w, y = (y1, . . . , yn)
# T
# (the column vector of responses), X·j (the jth column of X, represented
# as a column matrix), and wj (the jth coordinate of w – a scalar).

#  $$a_{j} = 2|X_j|^{2}$$

# $$c_{j} = 2*X_j(y-w^{T}X + w_{j}X_{j})$$

# ### 3.1.2 Write a function that computes the Lasso solution for a given λ using the shooting algorithm described above. 
# For convergence criteria, continue coordinate descent until a pass through
# the coordinates reduces the objective function by less than 10−8, or you have taken 1000 passes through the coordinates. Compare performance of cyclic coordinate descent to
# randomized coordinate descent, where in each round we pass through the coordinates in a
# different random order (for your choices of λ). Compare also the solutions attained
# (following the convergence criteria above) for starting at 0 versus starting at the ridge
# regression solution suggested by Murphy (again, for your choices of λ). If you like, you may
# adjust the convergence criteria to try to attain better results (or the same results faster).

# In[14]:


# implement Lasso for a given lambda/ 
# continue coordinate descent until a pass through the coordinates reduces the objective function by less than 10-8, or 100 iterations have been made


# In[15]:


def soft(a,b):
    if a < 0:
        return -max(abs(a) - b, 0)
    else:
        return max(abs(a) - b, 0)

def compute_lasso(X, y, l, ridge = False, shuffle = False, maxIterations = 1000, initialW = None):
    
        
    
    num_features = len(X[0])
    objectives = []
    
    if initialW is not None:
        w = initialW
        
    elif ridge:
        x_x = np.matmul(X.transpose(), X)
        w = np.dot(np.linalg.inv(x_x + l*np.identity(num_features)), np.dot(X.transpose(),y))
    else:
        w = np.zeros(num_features)
        
    
    #shuffle vs in order
    order = np.arange(num_features)
    if shuffle:
        np.random.shuffle(order)
    
        
    prev_obj = sum((np.dot(w.transpose(), X.transpose())-y)**2 + l*np.linalg.norm(w,1))
    objectives.append(prev_obj)
    
    for i in range(maxIterations):       
        for j in order:
            X_j = X.transpose()[j]
            a_j = 2*(np.dot(X_j, X_j))
            if a_j == 0:
                w[j] = 0
                continue
            c_j = 2 * (np.dot(X_j.transpose(), y) - np.dot(w.transpose(), 
                    np.dot(X.transpose(), X_j)) + 
                     w[j] * np.dot(X_j.transpose(), X_j))
            
            w[j] = soft((c_j/a_j), (l/a_j))
            

            
        obj = sum((np.dot(w.transpose(), X.transpose())-y)**2 + l*np.linalg.norm(w,1))
        objectives.append(obj)
        if abs(prev_obj - obj) < 1e-8:
            return w,objectives
        prev_obj = obj
            
    return w, objectives 



# In[16]:


for l in [0.001, 0.01, 0.1, 1, 2]: 
    
    plt.figure(figsize = (10, 10))
    
    w_cyclic, obj_cyclic = compute_lasso(X_train, y_train, l) 
    w_shuffle, obj_shuffle = compute_lasso(X_train, y_train, l, shuffle = True) 
    w_cyclic_ridge, obj_cyclic_ridge = compute_lasso(X_train, y_train, l, ridge = True) 
    w_shuffle_ridge, obj_shuffle_ridge = compute_lasso(X_train, y_train, l, ridge = True, shuffle = True)  

    plt.plot(range(len(obj_cyclic)), obj_cyclic, label = 'Cyclic, Zero-Weights, Lambda = ' + str(l))  
    plt.plot(range(len(obj_shuffle)), obj_shuffle, label = 'Shuffle, Zero-Weights, Lambda = ' + str(l))  
    plt.plot(range(len(obj_cyclic_ridge)), obj_cyclic_ridge, label = 'Cyclic, Ridge-Weights, Lambda = ' + str(l))  
    plt.plot(range(len(obj_shuffle_ridge)), obj_shuffle_ridge, label = 'Shuffle, Ridge-Weights, Lambda = ' + str(l))
    
    plt.xlabel('Iteration') 
    plt.ylabel('Objective Function') 
    plt.legend() 


# The different graphs show that the best configuration is one in which lambda is 0.01. In addition, the cyclic, ridge-weights configuration is the one reaching converging first, which leads me to believe that it is the best configuration. 
# 

# ### 3.1.3 Run your best Lasso configuration on the training dataset provided, and select the λ that minimizes the square error on the validation set. 
# Include a table of the parameter values you
# tried and the validation performance for each. Also include a plot of these results. Include
# also a plot of the prediction functions, just as in the ridge regression section, but this time
# add the best performing Lasso prediction function and remove the unregularized least
# squares fit. Similarly, add the lasso coefficients to the bar charts of coefficients generated in
# the ridge regression setting. Comment on the results, with particular attention to parameter
# sparsity and how the ridge and lasso solutions compare. What’s the best model you found,
# and what’s its validation performance?

# In[36]:


lambdas = [0.001, 0.01, 0.1, 1, 2]
errors = []
weights = []
predictions = []
print('Lamba \t\t\t Mean Squared Error')

lasso_dict = dict()


for l in lambdas: 
    w_shuffle_ridge, obj_shuffle_ridge = compute_lasso(X_train, y_train, l, ridge = True, shuffle = True)
    mse = mean_squared_error(y_val, np.dot(w_shuffle_ridge.transpose(), X_val.transpose()))
    errors.append(mse)
    weights.append(w_shuffle_ridge)
    print(str(l) + ' \t\t\t ' + str(mse))
    lasso_dict[l] = mse
    
# also graph the lambda over loss fcn
plt.figure(figsize = (20, 10))
plt.plot(lambdas, errors)
plt.xlabel("Lambdas")
plt.ylabel("Mean Squared Errors")


# In[29]:


plt.figure(figsize = (20, 10))

for l in range(len(lambdas)):
    #create dictionary for key = input, val = pred
    pred_fcns = dict(zip(list(x_val), list(np.dot(weights[l], X_val.transpose()))))
    x_vals = sorted(pred_fcns) 
    y_vals = [pred_fcns[x] for x in x_vals]
    plt.plot(x_vals, y_vals,  label = 'Lambda = ' + str(l))
plt.scatter(x_val, y_val, s = 0.5) 
plt.legend()


# In[30]:


plt.figure(figsize = (20, 10)) 
for i in range(len(lambdas)): 
    plt.subplot(str(len(lambdas)) + '1' + str(i+1))
    plt.bar(range(len(weights[i])), weights[i], label = 'Lambda = ' + str(lambdas[i]))
    plt.ylabel('Coefficient of weight') 
    plt.legend() 
plt.xlabel('Parameters') 


# The results from lasso regression have greater sparcity. This is evident when we compare the results from Lambda = 1 in both  bar graphs. The bar graph for lasso regression shows much greater weights, but with more sparcity. However, in terms of which solution is best, it is hard to tell since both prediction graphs show very similar behaviors. 
# 

# ### 3.1.4 Homotopy

# In[31]:


def homotopy(X_train, y_train, X_val, y_val):
    
    num_features = len(X[0])
    x_y = np.dot(X_train.transpose(), y_train)
    lambdas = []
    max_lambda = 2 * np.linalg.norm(x_y, np.inf)
    avg_loss = []
    
    #initialize weights
    w = np.zeros(num_features)
    
    
    for i in range(30):
        lambdas.append(max_lambda*(0.8 ** i))
        
    prev_w = w
    for l in range(len(lambdas)):
        # calculate weights with lasso
        #since we are using ridge, we will let our initial W be what the ridge configuration gives
        if l == 0:
            # on our first iteration, we will not initialize W since lasso will use the ridge configuration
            w, obj = compute_lasso(X_train, y_train, lambdas[l], ridge = True, shuffle = True)
        else:
            # we will use our last optimal solution by giving an initial value of weights
            w, obj = compute_lasso(X_train, y_train, lambdas[l], ridge = True, shuffle = True, initialW = prev_w)
            
        
        #calculate loss
        mse = mean_squared_error(y_val, np.dot(w.transpose(), X_val.transpose()))
        avg_loss.append(mse / len(y_val))
        prev_w = w
        
        
    plt.figure(figsize = (20, 10)) 
    plt.xlabel("Lambda")
    plt.ylabel("Average Loss")
    plt.plot(lambdas, avg_loss)
        
        


# In[32]:


homotopy(X_train, y_train, X_val, y_val)


# ### 3.2 Deriving $\nabla_{max}$
# 

# ### 3.2.1 Compute $J^{\prime}(0 ; v)$

# In[ ]:





# ### 4 SGD via Variable Splitting

# We can define the gradient as: 
# $$ 
# \nabla_\theta J = 2 \sum_{i=1}^{m}\left(h_{\theta^{+}, \theta^{-}}\left(x_{i}\right)-y_{i}\right)x_i+\lambda
# $$ 

# In[34]:


def theta_pos(X, y, theta_pos, theta_neg, l):
    pred = np.dot((theta_pos - theta_neg), X)
    grad = 2 * ((pred-y)*X)
    grad += l
    return grad
    

def theta_neg(X, y, theta_pos, theta_neg, l):
    pred = np.dot((theta_pos - theta_neg), X)
    grad = 2 * ((pred-y)*-X)
    grad += l
    return grad

def variable_SGD(X, y, l, num_epoch = 1000):
    num_instances = len(X)
    num_features = len(X[0])
    theta_p = np.zeros(num_features)
    theta_n = np.zeros(num_features)
    alpha = 0.001
    # keep track of validation errors to plot later
    errors = []
    for epoch in range(num_epoch):
        for i in range(num_instances):
            
            theta_p += -alpha * theta_pos(X[i], y[i], theta_p, theta_n, l/96)
            theta_n += -alpha*theta_neg(X[i], y[i], theta_p, theta_n, l/96)
#             zero values
            for j in range(len(theta_p)):
                if theta_p[j] < 0:
                    theta_p[j] = 0
                if theta_n[j] < 0:
                    theta_n[j] = 0
            theta = theta_p - theta_n
            
            errors.append(mean_squared_error(y_val, np.dot(theta.transpose(), X_val.transpose())))
    return theta, errors





# In[37]:


plt.figure(figsize = (20, 10)) 
sgd_dict = dict()
for l in lambdas:
    w, losses = variable_SGD(X_train, y_train, l)    
    mse = mean_squared_error(y_val, np.dot(w.transpose(), X_val.transpose()))
    pred_fcns = dict(zip(list(x_val), list(np.dot(w, X_val.transpose()))))
    x_vals = sorted(pred_fcns) 
    y_vals = [pred_fcns[x_val] for x_val in x_vals] 
    plt.plot(x_vals, y_vals) 
    plt.xlabel('Input Space') 
    plt.ylabel('Action/Outcome Space') 
    plt.scatter(x_val, y_val, s = 0.5, label = 'Lambda' + str(l))
    sgd_dict[l] = mse
plt.legend()


# In[40]:


# we have two dictionaries
#lasso_dict
#sdg_dict
plt.figure(figsize = (20, 10)) 
y_vals_shooting = [lasso_dict[l] for l in lambdas]
y_vals_sdg = [sgd_dict[l] for l in lambdas]
plt.plot(lambdas, y_vals_shooting,  label = 'Lambda = ' + "Shooting Algo")
plt.plot(lambdas, y_vals_sdg,  label = 'Lambda = ' + "SGD")
plt.xlabel('Lambdas') 
plt.ylabel('Mean Squared Error') 
plt.legend()


# ### 4.1 
# The performance of the Shooting Algorithm seems to be best. It has a lower mean Squared Error. 

# ### 4.2
# We can see that the Projected SGD via variable splitting method has greater sparcity because the graph has less peaks, suggesting less effects from the weights. More weights would cause the graph to have greater, more constant changes.

# In[ ]:




