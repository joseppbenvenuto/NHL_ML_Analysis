# Project Functions
#
# Description
#
# Below are the different functions used in the analysis.

# Imported necessary packages
import pandas as pd
import numpy as np

from sklearn.metrics import *
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

from scipy import stats
import math as ma

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


###########################################################################################################################################
# NEW CODE BLOCK - Regression Test Metrics Function
###########################################################################################################################################

# Returns regression metrics
def regression_test_metrics(y, y_pred):

    r2 = round(r2_score(y, y_pred),2)
    mae = round(mean_absolute_error(y, y_pred),2)
    mse = round(mean_squared_error(y, y_pred),2)
    rmse = round(ma.sqrt(mse),2)

    results = [
        ('r-squared:',r2),
        ('mean absolute error:',mae),
        ('mean squared error:',mse),
        ('root Mean squared error:',rmse)
    ]

    print('\n' + 'model metrics:' + '\n')
    for label, value in results:
        print(f"{label:{35}} {value:.>{20}}")


# * **y** = y values
# * **y_pred** = predicted y values

###########################################################################################################################################
# NEW CODE BLOCK - K-fold Cross Validation Regression Metrics Function with Shuffle
###########################################################################################################################################

# K-fold cross validation regression metrics
def regression_cross_val_shuffle(regressor, X_train, y_train, cv, font):

    # r2
    ###############################################################################################################
    accuracies_r2 = cross_val_score(
        estimator = regressor, 
        X = X_train, 
        y = y_train, 
        cv = cv, 
        n_jobs = -1,
        scoring = 'r2'
    )
    
    accuracies_r2_mean = round(accuracies_r2.mean(),2)
    accuracies_r2_std = round(accuracies_r2.std(),2)

    # MAE
    ###############################################################################################################
    accuracies_mae = cross_val_score(
        estimator = regressor, 
        X = X_train, 
        y = y_train, 
        cv = cv, 
        n_jobs = -1,
        scoring = 'neg_mean_absolute_error'
    )

    score_mae = [score * -1 for score in accuracies_mae]
    score_mae_df = pd.DataFrame(score_mae, columns = ['col'])
    accuracies_mae_mean = round(score_mae_df['col'].mean(),2)
    accuracies_mae_std = round(score_mae_df['col'].std(),2)

    # MSE
    ###############################################################################################################
    accuracies_mse = cross_val_score(
        estimator = regressor,
        X = X_train, 
        y = y_train, 
        cv = cv, 
        n_jobs = -1,
        scoring = 'neg_mean_squared_error'
    )

    score_mse = [score * -1 for score in accuracies_mse]
    score_mse_df = pd.DataFrame(score_mse, columns = ['col'])
    accuracies_mse_mean = round(score_mse_df['col'].mean(),2)
    accuracies_mse_std = round(score_mse_df['col'].std(),2)

    # RMSE
    ###############################################################################################################
    accuracies_rmse = cross_val_score(
        estimator = regressor, 
        X = X_train, 
        y = y_train, 
        cv = 10,
        n_jobs = -1,
        scoring = 'neg_root_mean_squared_error'
    )

    score_rmse = [score * -1 for score in accuracies_rmse]
    score_rmse_df = pd.DataFrame(score_rmse, columns = ['col'])
    accuracies_rmse_mean = round(score_rmse_df['col'].mean(),2)
    accuracies_rmse_std = round(score_rmse_df['col'].std(),2)

    # Tuple unpacking
    ###############################################################################################################

    print('\n' + 'test model f-fold metrics:' + '\n')
    
    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv.get_n_splits(X_train) + 1, 1), 
        accuracies_r2, 
        ls = '-', 
        marker = 'o'
    )
    
    plt.title('r2 for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('r2')
    plt.ylim(0,1.1)
    plt.show()

    results1 = [
        ("r-squared k-fold cross validation: ", accuracies_r2_mean),
        ("r-squared std: ", accuracies_r2_std)
    ]

    for label, value in results1:
        print(f"{label:{50}} {value:.>{20}}")
    
    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv.get_n_splits(X_train) + 1, 1), 
        score_mae, 
        ls = '-', 
        marker = 'o'
    )
    
    plt.title('mae for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('mae')
    plt.ylim(0, max(score_mae) * 1.25)
    plt.show()

    results2 = [
        ("mean absolute error k-fold cross validation: ", accuracies_mae_mean),
        ("mean absolute error std: ", accuracies_mae_std)
    ]

    print("\n")
    for label, value in results2:
        print(f"{label:{50}} {value:.>{20}}")
    
    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv.get_n_splits(X_train) + 1, 1), 
        score_mse, 
        ls = '-', 
        marker = 'o'
    )
    
    plt.title('mse for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('mse')
    plt.ylim(0, max(score_mse) * 1.25)
    plt.show()

    results3 = [
        ("mean squared error k-fold cross validation: ", accuracies_mse_mean),
        ("mean squared error std: ", accuracies_mse_std)
    ]

    print("\n")
    for label, value in results3:
        print(f"{label:{50}} {value:.>{20}}")

    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv.get_n_splits(X_train) + 1, 1), 
        score_rmse, 
        ls = '-',
        marker = 'o'
    )
    
    plt.title('rmse for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('rmse')
    plt.ylim(0, max(score_rmse) * 1.25)
    plt.show()

    results4 = [
        ("root mean squared error k-fold cross validation: ", accuracies_rmse_mean),
        ("root mean squared error std: ", accuracies_rmse_std)
    ]

    print("\n")
    for label, value in results4:
        print(f"{label:{50}} {value:.>{20}}")
    print("\n")


# * **regressor** = regressor model
# * **X_train** = X train values
# * **y_train** y train values
# * **cv** = k-fold split object
# * **font** = plot font


###########################################################################################################################################
# NEW CODE BLOCK - K-fold Cross Validation Regression Metrics Function with CV Split
###########################################################################################################################################

# K-fold cross validation regression metrics
def regression_cross_val_splits(regressor, X_train, y_train, cv, font):

    # r2
    ###############################################################################################################
    accuracies_r2 = cross_val_score(
        estimator = regressor,
        X = X_train,
        y = y_train, 
        cv = cv, 
        n_jobs = -1,
        scoring = 'r2'
    )
    
    accuracies_r2_mean = round(accuracies_r2.mean(),2)
    accuracies_r2_std = round(accuracies_r2.std(),2)

    # MAE
    ###############################################################################################################
    accuracies_mae = cross_val_score(
        estimator = regressor, 
        X = X_train, 
        y = y_train, 
        cv = cv,
        n_jobs = -1,
        scoring = 'neg_mean_absolute_error'
    )

    score_mae = [score * -1 for score in accuracies_mae]
    score_mae_df = pd.DataFrame(score_mae, columns = ['col'])
    accuracies_mae_mean = round(score_mae_df['col'].mean(),2)
    accuracies_mae_std = round(score_mae_df['col'].std(),2)

    # MSE
    ###############################################################################################################
    accuracies_mse = cross_val_score(
        estimator = regressor,
        X = X_train,
        y = y_train, 
        cv = cv,
        n_jobs = -1,
        scoring = 'neg_mean_squared_error'
    )

    score_mse = [score * -1 for score in accuracies_mse]
    score_mse_df = pd.DataFrame(score_mse, columns = ['col'])
    accuracies_mse_mean = round(score_mse_df['col'].mean(),2)
    accuracies_mse_std = round(score_mse_df['col'].std(),2)

    # RMSE
    ###############################################################################################################
    accuracies_rmse = cross_val_score(
        estimator = regressor, 
        X = X_train,
        y = y_train, 
        cv = 10, 
        n_jobs = -1,
        scoring = 'neg_root_mean_squared_error'
    )

    score_rmse = [score * -1 for score in accuracies_rmse]
    score_rmse_df = pd.DataFrame(score_rmse, columns = ['col'])
    accuracies_rmse_mean = round(score_rmse_df['col'].mean(),2)
    accuracies_rmse_std = round(score_rmse_df['col'].std(),2)

    # Tuple unpacking
    ###############################################################################################################

    print('\n' + 'test model f-fold metrics:' + '\n')
    
    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv + 1, 1), 
        accuracies_r2, 
        ls = '-', 
        marker = 'o'
    )
    
    plt.title('r2 for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('r2')
    plt.ylim(0,1.1)
    plt.show()

    results1 = [
        ("r-squared k-fold cross validation: ", accuracies_r2_mean),
        ("r-squared std: ", accuracies_r2_std)
    ]

    for label, value in results1:
        print(f"{label:{50}} {value:.>{20}}")

    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv + 1, 1), 
        score_mae,
        ls = '-', 
        marker = 'o'
    )
    
    plt.title('mae for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('mae')
    plt.ylim(0, max(score_mae) * 1.25)
    plt.show()

    results2 = [
        ("mean absolute error k-fold cross validation: ", accuracies_mae_mean),
        ("mean absolute error std: ", accuracies_mae_std)
    ]

    print("\n")
    for label, value in results2:
        print(f"{label:{50}} {value:.>{20}}")

    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv + 1, 1), 
        score_mse, 
        ls = '-', 
        marker = 'o'
    )
    
    plt.title('mse for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('mse')
    plt.ylim(0, max(score_mse) * 1.25)
    plt.show()

    results3 = [
        ("mean squared error k-fold cross validation: ", accuracies_mse_mean),
        ("mean squared error std: ", accuracies_mse_std)
    ]

    print("\n")
    for label, value in results3:
        print(f"{label:{50}} {value:.>{20}}")

    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (10,3))
    
    plt.plot(
        range(1, cv + 1, 1), 
        score_rmse,
        ls = '-',
        marker = 'o'
    )
    
    plt.title('rmse for kfold')
    plt.xlabel('kfold index')
    plt.ylabel('rmse')
    plt.ylim(0, max(score_rmse) * 1.25)
    plt.show()

    results4 = [
        ("root mean squared error k-fold cross validation: ", accuracies_rmse_mean),
        ("root mean squared error std: ", accuracies_rmse_std)
    ]

    print("\n")
    for label, value in results4:
        print(f"{label:{50}} {value:.>{20}}")
    print("\n")


# * **regressor** = regressor model
# * **X_train** = X train values
# * **y_train** y train values
# * **cv** = k-fold split object
# * **font** = plot font


###########################################################################################################################################
# NEW CODE BLOCK - Feature Importance Bar Chart and Coefficients
###########################################################################################################################################

# Feature importance bar chart followed by supporting stats
def regression_feature_importance(model, X_cols, font, length, width, pos, neg):
    
    coefficients = list(model.coef_)
    intercept =  [model.intercept_]
    coefficients = coefficients + intercept
    X_col = list(X_cols.columns) + ['intercept']
    coefficients = pd.DataFrame({'features': X_col, 'coef': coefficients})
    coefficients['positive'] = coefficients['coef'] > 0 
    coefficients['coef2'] = abs(coefficients['coef'])
    coefficients = coefficients.sort_values(by = ['coef2'], ascending = True)
    coefficients = coefficients.reset_index()
    
    blue_patch = mpatches.Patch(color = '#1f77b4', label = 'positive')
    red_patch = mpatches.Patch(color = 'r', label = 'negative')

    sns.set(font_scale = font, style = 'white')  
    
    coefficients.plot(
        x = 'features', 
        y = 'coef',
        kind = 'barh',
        figsize = (width, length), 
        color = coefficients.positive.map({True: pos, False: neg})
    )

    plt.title('feature importance (features scaled)')
    plt.xlabel('coefficient units')
    plt.ylabel('features')
    
    plt.legend(
        handles = [blue_patch, red_patch], 
        bbox_to_anchor = (1.05, 1.0), 
        loc = 'upper left'
    )
    
    plt.show()

    coefficients = coefficients.sort_values(by = ['coef2'], ascending = False)
    
    coefficients = coefficients.drop(
        ['coef2'], 
        axis = 1, 
        errors = 'ignore'
    )
    
    coefficients = coefficients.drop(
        ['index'], 
        axis = 1, 
        errors = 'ignore'
    )

    display(coefficients)


# * **model** = model 
# * **X_cols** = X variables
# * **font** = plot font
# * **length** = plot length
# * **width** = plot width
# * **pos** = target feature 1 colour
# * **neg** = target feature 0 colour


###########################################################################################################################################
# NEW CODE BLOCK - Residual Means and Counts Plot
###########################################################################################################################################

# Residual means and counts plot
def residual_means_counts_plot(df, X, res, ymin1, ymax1, ymin2, ymax2, font, length, width):
    col_range = round(df[X].max() - df[X].min(),0) + 1
    bins = pd.cut(df[X], int(col_range))
    mean_res = df.groupby(bins).agg({res: "mean"})
    mean_res = mean_res.rename(columns = {res: 'mean'}, inplace = False).reset_index()
    count_res = df.groupby(bins).agg({res: "count"})
    count_res = count_res.rename(columns = {res: 'count'}, inplace = False).reset_index()
    mean_count_res = pd.merge(mean_res, count_res, on = X)
    mean_count_res[X] = mean_count_res[X].astype(str)

    # Plot the results
    sns.set(font_scale = font, style = 'white')
    plt.figure(figsize = (width,length))
    blue_patch = mpatches.Patch(color = '#1f77b4', label = 'avg residuals per bin')
    darkgreen_patch = mpatches.Patch(color = 'darkgreen', label = 'count of residuals per bin')
    red_patch = mpatches.Patch(color = 'r', label = 'avg residuals')
    
    plt.legend(
        handles = [blue_patch, darkgreen_patch, red_patch], 
        bbox_to_anchor = (1.05, 1.0), 
        loc = 'upper left'
    )
    
    ax1 = plt.axes()
    ax2 = ax1.twinx()

    ax1.plot(
        mean_count_res[X],
        mean_count_res['mean'], 
        ls = '-', 
        marker = 'o', 
        color = '#1f77b4'
    )
    
    ax1.set_xticklabels(mean_count_res[X], rotation = 'vertical')
    
    ax1.axhline(
        y = 0, 
        color = 'r', 
        linestyle = '--'
    )
    
    ax1.set(
        xlabel = X + ' bins', 
        ylabel = 'average residuals per bin', 
        title = 'average residuals per binned ' + X)
    
    ax1.set_ylim(mean_count_res['mean'].min() + ymin1, mean_count_res['mean'].max() + ymax1)

    ax2.bar(
        mean_count_res[X], 
        mean_count_res['count'], 
        color = 'darkgreen'
    )
    
    ax2.set(ylabel = 'count per bin')
    ax2.set_ylim(mean_count_res['count'].min() + ymin2, mean_count_res['count'].max() + ymax2)
    plt.show()


# * **df** = X values with prediction values
# * **X** = X column
# * **res** = residuals column name
# * **ymin1** = line plot add to ymin on the left side of plot
# * **ymax1** = line plot add to ymax on the left side of plot
# * **ymin2** = bar plot add to ymin on the right side of plot
# * **ymax2** = bar plot add to ymax on the right side of plot
# * **font** = plot font
# * **length** = plot length
# * **width** = plot width
