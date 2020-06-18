import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data
from scipy import stats
import scipy
import random
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)
from sklearn.preprocessing import normalize as sk_normalize
from scipy.special import expit as sigmoid  # logistic function


# Inspired from https://berndplumhoff.gitbook.io/sulprobil/excel/excel-worksheet-function-solutions/iman-conover
# And https://stats.stackexchange.com/questions/271686/iman-conover-implementation-for-correlated-randoms-in-python-with-numpy-scipy



def ic_m(n, d):
    a = np.arange(1, (n+1))
    p = stats.norm.ppf(a/(n+1))
    p = normalize(p)
    score = np.zeros((n, d))
    for j in range(0, score.shape[1]):
        score[:, j] = np.random.permutation(p)
    return score

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    return v/norm

def rank(N):
    rank = np.zeros((N.shape[0], N.shape[1]))
    for j in range(0, N.shape[1]):
        rank[:, j] = stats.rankdata(N[:, j], method='ordinal')
    return rank.astype(int) - 1

def reorder(rank, samples):
    rank_samples = np.zeros((samples.shape[0], samples.shape[1]))
    for j in range(0, samples.shape[1]):
        s = np.sort(samples[:, j])
        rank_samples[:, j] = s[rank[:,j]]
    return rank_samples

def generate_correlated_feature(X, correlation_rate):
    if len(X.shape) == 1:
        d = 2
    else:
        d = X.shape[1] * 2
    n = X.shape[0]
    S = np.array(([1., correlation_rate],
              [correlation_rate, 1.]))
    C = scipy.linalg.cholesky(S)
    M = ic_m(n,d)
    D = (1./n) * np.dot(M.T, M)
    E = scipy.linalg.cholesky(D)
    N = np.dot(np.dot(M, np.linalg.inv(E)), C)
    R = rank(N)
    A = np.array([X, X])
    #A = np.array([X,stats.norm.ppf(np.random.uniform(0.0, 1.0, n), loc=0, scale=1)])
    #A = (np.array([X, stats.uniform.ppf(np.random.uniform(0.0, 1.0, n), 0, 1)]))
    dists = reorder(R, A.T)
    np.corrcoef(dists.T)
    
    # Search mapping
    A = np.transpose(A)
    index = np.argsort(dists[:,0])
    sorted_x = dists[:,0][index]
    sorted_index = np.searchsorted(sorted_x, A[:,0])

    yindex = np.take(index, sorted_index, mode="clip")
    mask = dists[:,0][yindex] != A[:,0]

    result = np.ma.array(yindex, mask=mask)
 
    B = np.transpose(np.array([A[:,0], dists[:,1][result]]))
    return B[:, 1]

def check_feature_correlation(x1, x2, plot_=False):
    coef, p = scipy.stats.spearmanr(x1,x2)#(X[0:1000,0], y)
    print('Spearmans correlation coefficient: %.3f' % coef)
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.8f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.8f' % p)
    if plot_:
        plt.scatter(x1, x2)
        plt.show()

def binary_sensitive_to_numerical(X, mu1=2, sigma1=1.2, mu0=-2, sigma0=0.8):
    # Prepare two new distributions.
    nv1 = multivariate_normal(mean = mu1, cov = sigma1)
    nv0 = multivariate_normal(mean = mu0, cov = sigma0)
    def f(x):
        if x == 0:
            return nv0.rvs(1)
        else:
            return nv1.rvs(1)
    def vectorize(x):
        return np.vectorize(f)(x)
    return vectorize(X)


def get_bins(col, nb_bin):
    min_ = np.min(col)
    max_ = np.max(col)
    print(min_, max_)
    bin_size = ((max_) - (min_)) / nb_bin
    bin_list = []
    bin_name = []
    thresh_ = min_
    i = 0
    while thresh_ <= max_:
        bin_list.append(thresh_)
        bin_name.append("cat_" + str(i))
        i+=1
        thresh_ += bin_size
    if max_ not in bin_list:
        bin_list.append(max_)
    else:
        bin_name.pop()
    return bin_list, bin_name


def generate_synthetic_dataframe(X,y,x_control_list, x_correlated_list, x_uncorrelated_list, x_cat_uncorrelated_list, x_cat_correlated_list):
    data = {}
    # Regular features.
    nb_feat = X.shape[1]
    for i in range(nb_feat):
        data["x_" + str(i)] = X[:,i]
    # Correlated features.
    if len(x_correlated_list) >0:
        for i in range(len(x_correlated_list)):
            data["x_corr_" + str(i)] = x_correlated_list[i]
    # Protected features.
    for i in range(len(x_control_list)):
        data["x_sens_" + str(i)] = x_control_list[i]
    # Labels.
    data["label"] = y
    # Additional uncorrelated features.
    if len(x_uncorrelated_list) >0:
        for i in range(len(x_uncorrelated_list)):
            data["x_uncorr_" + str(i)] = x_uncorrelated_list[i]
    # Additional uncorrelated categorical features.
    if len(x_cat_uncorrelated_list) >0:
        for i in range(len(x_cat_uncorrelated_list)):
            data["x_uncorr_cat_" + str(i)] = x_cat_uncorrelated_list[i]
    # Additional correlated (with the sensitive attribute) categorical features.
    if len(x_cat_correlated_list) >0:
        for i in range(len(x_cat_correlated_list)):
            data["x_corr_cat_" + str(i)] = x_cat_correlated_list[i]
    
    return pd.DataFrame(data) 
    

def generate_synthetic_data(nb_data, mu1, sigma1, mu2, sigma2, discri_factor_protected_features, correlation_feature_scores_protected_features, add_random_features, add_categorical_features, plot_data=False):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

    n_samples = int(nb_data/2) # generate these many data points per class
    #disc_factor = math.pi / 2 # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0,n_samples*2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]

    def gen_sensitive_feature(X, disc_factor, nv1, nv2):
        rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
        X_aux = np.dot(X, rotation_mult)
        x_control = [] # this array holds the sensitive feature value
        for i in range (0, len(X)):
            x = X_aux[i]

            # probability for each cluster that the point belongs to it
            p1 = nv1.pdf(x)
            p2 = nv2.pdf(x)

            # normalize the probabilities from 0 to 1
            s = p1+p2
            p1 = p1/s
            p2 = p2/s

            r = np.random.uniform() # generate a random number from 0 to 1

            if r < p1: # the first cluster is the positive class
                x_control.append(1.0) # 1.0 means its male
            else:
                x_control.append(0.0) # 0.0 -> female

        x_control = np.array(x_control)
        return x_control
    
    """ Generate the sensitive feature here """
    x_control_list = []
    for sensitive_feature_characteristic in discri_factor_protected_features:
        x_control_list.append(gen_sensitive_feature(X, sensitive_feature_characteristic, nv1, nv2))

    """ Generate additional features correlated with the sensitive ones """ 
    x_correlated_list = []
    sensitive_feature_index = 0
    for list_feature_details in correlation_feature_scores_protected_features:
        if len(list_feature_details) > 0:
            for feature_correlation in list_feature_details:
                print("Intended coefficient: ", feature_correlation)
                # Create a new feature.
                x1 = x_control_list[sensitive_feature_index] # Sensitive feature
                # We transform the sensitive feature into a numerical one,\
                # otherwise the correlation does not work.
                x_sens_num = binary_sensitive_to_numerical(x1)
                x2 = generate_correlated_feature(x_sens_num, feature_correlation)
                check_feature_correlation(x_sens_num, x2, False)
                x_correlated_list.append(x2)
        sensitive_feature_index += 1
        
    """ Generate additional uncorrelated features with the predictions """
    x_uncorrelated_list = []
    for feat_index in range(add_random_features):
        mean_in = random.uniform(-10, 10)
        cov_in = random.uniform(0, 5)
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        x_uncorrelated_list.append(nv.rvs(nb_data))
        
    """ Generate additional categorical features, correlated or not with the sensitive attributes """
    x_cat_uncorrelated_list = []
    x_cat_correlated_list = []
    # Add the uncorrelated features.
    # WE could take them from many different disributions.
    #X = stats.binom(10, 0.2) # Declare X to be a binomial random variable
    #print(X.rvs(10))
    #Y = stats.poisson(2) # Declare Y to be a poisson random variable
    #print(Y.rvs(10))
    #X = stats.geom(0.75) # Declare X to be a geometric random variable
    #print(X.rvs(10))        # Get a random sample from Y
    if len(add_categorical_features) > 0: 
	    for feat_index in range(add_categorical_features[0]):
	        mean_ = random.randrange(0, 10)
	        prob = random.uniform(0, 1)
	        nv = stats.binom(mean_, prob) # Binomial random variable
	        x_cat_uncorrelated_list.append(nv.rvs(nb_data))
        
    #print(Y.rvs(10))
    # Add the correlated features with the first sensitive attribute.
    if len(add_categorical_features) > 1:
	    for feat_index in range(add_categorical_features[0]):
	        x1 = x_control_list[0] # Sensitive feature
	        # We transform the sensitive feature into a numerical one,\
	        # otherwise the correlation does not work.
	        x_sens_num = binary_sensitive_to_numerical(x1)
	        x2 = generate_correlated_feature(x_sens_num, feature_correlation)
	        # We transform x2 into a categorical attribute.
	        df = pd.DataFrame(x2)
	        bin_, bin_name = get_bins(x2, 10)
	        df['col2'] = pd.cut(df[0], bins=bin_, labels=bin_name)
	        x2 = np.array(df["col2"])
	        x_cat_correlated_list.append(x2)
        
    """ Show the data """
    if plot_data:
        num_to_draw = 200 # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]
        plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "Prot. +ve")
        plt.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "Prot. -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "Non-prot. +ve")
        plt.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "Non-prot. -ve")

        
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15,10))
        plt.ylim((-10,15))
        #plt.savefig("img/data.png")
        plt.show()

    
    return X,y,x_control_list, x_correlated_list, x_uncorrelated_list, x_cat_uncorrelated_list, x_cat_correlated_list


# Define method.
# Define on which variables to synthesize, 
# define which are the dependent variable, 
# define missing value rate.
# Inspired from https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values.

# Generate MCAR (Missing Completely at Random)
def ampute_mcar(X_complete, missing_rate = .2):
    # Mask completly at random some values

    M = np.random.binomial(1, missing_rate, size = X_complete.shape)
    X_obs = X_complete.copy()
    np.putmask(X_obs, M, np.nan)
    print('Percentage of newly generated mising values: {}'.\
        format(np.round(np.sum(pd.isnull(X_obs))/X_obs.size,3)))
    """
    # warning if a full row is missing
    for row in X_obs:
        if np.all(np.isnan(row)):
            warnings.warn('Some row(s) contains only nan values.')
            break

    # warning if a full col is missing
    for col in X_obs.T:
        if np.all(np.isnan(col)):
            warnings.warn('Some col(s) contains only nan values.')
            break
    """

    return X_obs
    
        
        
# These last two functions only work for numerical data... not categorical.
# Missing not at Random (MNAR)
def ampute_mnar(X_complete, missing_rate= .2, print_=True):
    """ ampute X_complete with censoring (Missing Not At Random)

    The missingness depends on the values.
    This will tends to "censor" X[i,j] where X[i,j] is high 
    comparing to its column X[:,j]
    """
    X_complete = X_complete.reshape(X_complete.shape[0], 1)
    # M depends on X_complete values
    #print(X_complete.shape)
    M_proba = np.random.normal(X_complete)
    #print(np.linalg.norm(M_proba,1))
    #print(np.sum((M_proba / np.linalg.norm(M_proba,1))))
    #M_proba = sk_normalize(M_proba, norm='l1')
    M_proba = M_proba / np.linalg.norm(M_proba,1)
    #print(M_proba)

    # compute thresold wrt missing_rate
    threshold = np.percentile(M_proba.ravel(), 100 * (1- missing_rate))
    #print(threshold)
    M = M_proba > threshold

    X_obs = X_complete.copy()
    np.putmask(X_obs, M, np.nan)
    if print_:
        print('Percentage of newly generated mising values: {}'.\
            format(np.sum(np.isnan(X_obs))/X_obs.size))
    return X_obs

def ampute_MNAR_categorical(df_complete, missing_rate= .2):
    a = pd.get_dummies(df_complete.astype("str")).copy(deep=True)
    list_col = list(a.columns)
    # Get a random number between 1 and the number of columns - 1.
    rand_n = random.randint(1, len(list_col) -1)
    # Get rand_n random of these columns.
    col_to_transform = [list_col[i] for i in random.sample(range(0, len(list_col)), rand_n)]
    print("We transform ", rand_n, " categories: ", col_to_transform)
    col_not_to_transform = [i for i in list_col if i not in col_to_transform ]
    # We need to compute the missing value per column. It's a combination of distributions of existing columns.
    missing_rate = missing_rate / rand_n
    # For each of the column to modify, we add MV_rate missing values.
    list_col_with_missing_val = []
    for col in col_to_transform:
        list_col_with_missing_val.append(ampute_mnar(a[[col]].astype("float").values, missing_rate, False))

    new_dataframe = a[col_not_to_transform]
    for i,j  in zip(col_to_transform, list_col_with_missing_val):
        new_dataframe[i] = j
    new_dataframe['nan_cat'] = new_dataframe[col_to_transform].apply(lambda x: np.nan if (np.isnan( (np.array(list(x)))).any() ) else 0, axis=1)
    a_back = new_dataframe.idxmax(axis=1, skipna=False)
    a_back.value_counts(dropna=False)

    print('Percentage of newly generated mising values: {}'.\
            format(a_back.isna().sum()/a_back.size))
    return a_back

# Missing (Conditionally) at Random (MAR).
# We need to specify the right W to define which feature influence which other one!
def ampute_mar(X_complete, missing_rate=.2, W=None, plot_=True):
    """ Observed values will censor the missing ones

    The proba of being missing: M_proba = X_obs.dot(W)
    So for each sample, some observed feature (P=1) will influence 
    the missingness of some others features (P=0) w.r.t to the weight 
    matrix W (shape n_features x n_features).

    e.g. during a questionnary, those who said being busy (X_obs[:,0] = 1) 
    usually miss to fill the last question (X_obs[:,-1] = np.nan)
    So here W[0,-1] = 1
    """
    X_obs = X_complete.copy()
    M_proba = np.zeros(X_obs.shape)

    if W is None:
        # generate the weigth matrix W
        W = np.random.randn(X_complete.shape[1], X_complete.shape[1])
    if plot_:
        print(W)
    # Severals iteration to have room for high missing_rate
    for i in range(X_obs.shape[1]*2):
        # Sample a pattern matrix P
        # P[i,j] = 1 will correspond to an observed value
        # P[i,j] = 0 will correspond to a potential missing value
        P = np.random.binomial(1, .5, size=X_complete.shape)

        # potential missing entry do not take part of missingness computation
        X_not_missing = np.multiply(X_complete,P)
        #print(X_not_missing)

        # sample from the proba X_obs.dot(W)
        sigma = np.var(X_not_missing)
        M_proba_ = np.random.normal(X_not_missing.dot(W), scale = sigma)

        # not missing should have M_proba = 0
        M_proba_ = np.multiply(M_proba_, 1-P)  # M_proba[P] = 0

        M_proba += M_proba_

    thresold = np.percentile(M_proba.ravel(), 100 * (1 - missing_rate))
    M = M_proba > thresold

    np.putmask(X_obs, M, np.nan)
    if plot_:
        print('Percentage of newly generated mising values: {}'.\
            format(np.sum(np.isnan(X_obs))/X_obs.size))
    return X_obs

def ampute_MAR_categorical(df_complete, df_condition, missing_rate= .2):
    perc_missing = -100
    while abs(perc_missing - missing_rate) > 0.02:
        input_ = pd.get_dummies(df_complete.copy(deep=True))
        list_col = list(input_.columns)
        # Get a random number between 1 and the number of columns - 1.
        rand_n = random.randint(1, len(list_col) -1) #random.randint(1, len(list_col) -1)
        # Get rand_n random of these columns.
        col_to_transform = [list_col[i] for i in random.sample(range(0, len(list_col)), rand_n)]
        #print("We transform ", rand_n, " categories: ", col_to_transform)
        col_not_to_transform = [i for i in list_col if i not in col_to_transform ]

        input_to_transform = input_[col_to_transform]
        input_to_transform[list(df_condition.columns)[0]] = df_condition.copy(deep=True)

        W = np.zeros((len(list(input_to_transform.columns)), len(list(input_to_transform.columns))))
        W[-1, 0:-1] = 1.0#/rand_n
        input_to_transform = input_to_transform.values
        output_ = ampute_mar(input_to_transform, missing_rate/rand_n, W, False)
        output_ = output_[:, 0:-1]


        new_dataframe = input_[col_not_to_transform]
        new_dataframe_2 = pd.DataFrame(output_, columns=col_to_transform)
        new_dataframe.reset_index(drop=True, inplace=True)
        new_dataframe_2.reset_index(drop=True, inplace=True)

        new_dataframe = pd.concat( [new_dataframe, new_dataframe_2], axis=1) 
        #print("New df ", new_dataframe)
        new_dataframe['nan_cat'] = new_dataframe.apply(lambda x: np.nan if (np.isnan( (np.array(list(x)))).any() ) else 0, axis=1)
        a_back = new_dataframe.idxmax(axis=1, skipna=False)
        a_back.value_counts(dropna=False)
        perc_missing = a_back.isna().sum()/a_back.size
        print('Percentage of newly generated mising values: {}'.\
                format(perc_missing))
    return a_back
    #print(a_back)


    #print(a_back)
    #print(df[["x_corr_cat_1", "x_corr_1"]])
    #c = df[["x_corr_cat_1","x_corr_1"]]
    #c["testtt"] = a_back
    #c.groupby(['x_corr_1', 'testtt']).size().unstack(fill_value=0)

    #d = c.loc[~c.notnull().all(1)]
    #d = d.replace(np.nan, "NA")
    #print(c)
    #print(d)
    #bins = [-2, 0, 2]
    #groups = d.groupby(['x_corr_cat_1', pd.cut(d.x_corr_1, 2)])
    #print(groups.size().unstack(fill_value="NA"))

    #print(d['x_corr_cat_1'].value_counts())

def generate_synthetic_errors(data, missing_value_list):    
    
    # Get the data that won't change.
    not_transformed_attributes = list(data.columns)
    for missing_element in missing_value_list:
        not_transformed_attributes.remove(missing_element[1])
    #print(not_transformed_attributes)
    # Safely copy these attributes.
    transformed_data = data[not_transformed_attributes].copy(deep=True)
    
    # Apply the transformations for the other attributes and add them to the new dataframe.
    for missing_element in missing_value_list:
        if missing_element[0] == "MCAR":
            new_attribute = ampute_mcar(data[[missing_element[1]]].values, missing_element[2])
            
        elif missing_element[0] == "MAR":
            # This needs to be checked.
            W = np.zeros((2, 2))
            W[1, 0] = 1.0
            new_attribute = ampute_mar(data[[missing_element[1], missing_element[2]]].values, missing_element[3], W)
            new_attribute = new_attribute[:, 0]
        elif missing_element[0] == "MNAR":
            new_attribute = ampute_mnar(data[[missing_element[1]]].values, missing_element[2])
        
        elif missing_element[0] == "MNAR_cat":
            new_attribute = ampute_MNAR_categorical(data[[missing_element[1]]], missing_element[2])
            new_attribute = new_attribute.values
        
        elif missing_element[0] == "MAR_cat":
            new_attribute = ampute_MAR_categorical(data[[missing_element[1]]], data[[missing_element[2]]], missing_element[3])
            new_attribute = new_attribute.values
    
        
        transformed_data[missing_element[1]] = new_attribute # This might not be the right type!
    
    return transformed_data