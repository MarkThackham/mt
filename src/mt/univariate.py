"""
Module: mt.univariate

This module contains functions for univariate analysis and metrics.

Functions:
- gini_coefficient(y_true, y_pred): 
Calculate the Gini Coefficient from the given true and predicted values.

- binned_feature(y_true, feature, num_bins=4):
Bin a feature based on the target variable.

- woe_iv_calc(y_true, feature, num_bins=4):
Calculate the Weight of Evidence (WoE) and Information Value (IV) for a given feature.
"""

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


def bin_feature(y_true, feature, num_bins=4):
    """
    Purpose:
    Bin a feature based on the target variable.
    
    Parameters:
    y_true (pd.Series): The target variable indicating the class labels.
    feature (pd.Series): The feature to be binned, either numeric or categorical.
    num_bins (int, optional): The number of bins to create for numeric features. Default is 4.
    
    Returns:
    pd.Series: The binned feature.

    Notes:
    - If the feature is numeric, it will be quatile binned.
    - If the feature is categorical, it will be used as is.
    - Table includes counts of y_true=0 and y_true=1, their proportions, 
      WoE values, and IV components.
    - The IV value is the sum of the IV components from the WoE table.
    """

    # Check if feature is numeric or categorical
    binned_feature=None
    if isinstance(feature.values[0], (int, float, np.number)):
        # Feature for survivors
        vals=feature[y_true==1]

        # Bin Feature for y_true==1
        _ , bins=pd.qcut(vals, q=num_bins, duplicates='drop', retbins=True)
        bins[0]=-999
        bins[-1]=999

        #Apply bins to Feature
        binned_feature=pd.cut(feature, bins=bins)

    elif isinstance(feature.values[0], str):
        # Feature is categorical
        binned_feature=feature

    return binned_feature


def gini(y_true, y_pred, decimals=4):
    """
    Purpose:
    Calculate the Gini coefficient based on true and predicted values.
    The Gini coefficient is a measure of inequality or discrimination, 
    commonly used in binary classification tasks. It is derived from 
    the area under the ROC curve (AUC).

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Predicted probabilities or scores.

    Returns:
    Gini coefficient (float) value between -1 and 1
    
    Notes:
    1 indicates perfect discrimination
    0 indicates no discrimination
    -1 indicates perfect inverse discrimination
    """

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)

    # Calculate Gini coefficient
    gini_value = round(float(2 * auc - 1), decimals)

    return gini_value


def woe_iv_calc(y_true, feature, num_bins=4):
    """
    Purpose:
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a given feature.
    
    Parameters:
    y_true (pd.Series): The target variable (binary) indicating the outcome.
    feature (pd.Series): The feature for which WoE and IV are to be calculated.
    num_bins (int, optional): The number of bins to use for numeric features. Default is 4.

    Returns:
    list: A list containing the WoE table (pd.DataFrame) and the IV value (float).
    """

    # Check if feature is numeric or categorical
    this_binned_feature=bin_feature(y_true, feature, num_bins=num_bins)

    # Calculate WoE
    woe_table=pd.crosstab(index=this_binned_feature,columns=y_true)
    woe_table['Total']=woe_table[0] + woe_table[1]
    woe_table['P0']=woe_table[0] / woe_table[0].sum()
    woe_table['P1']=woe_table[1] / woe_table[1].sum()
    woe_table['woe']=np.log(woe_table['P1']/woe_table['P0'])
    woe_table['iv_comp']=woe_table['woe'] * (woe_table['P1'] - woe_table['P0'])

    # Calculate IV;
    iv=woe_table['iv_comp'].sum()

    return [woe_table, iv]
