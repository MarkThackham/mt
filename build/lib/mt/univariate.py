from sklearn.metrics import roc_auc_score

def gini_coefficient(y_true, y_pred):
    """
    Calculate the Gini Coefficient from the given true and predicted values.

    The Gini Coefficient is a measure of inequality in a distribution,
    where a Gini Coefficient of 0 indicates perfect equality (everyone has the same value),
    and a Gini Coefficient of 1 indicates perfect inequality (one person has all the value, and everyone else has none).

    Parameters:
    y_true (array-like): The true values of the target variable.
    y_pred (array-like): The predicted values of the target variable.

    Returns:
    float: The calculated Gini Coefficient.
    """
    auc = roc_auc_score(y_true, y_pred)
    gini = float(2 * auc - 1)
    return gini