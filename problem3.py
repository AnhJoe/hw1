"""
Problem 3: Naive Bayes Classifiers

We use a small binary-feature email dataset stored in a CSV file.
You will implement Naive Bayes parameter estimation and prediction.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import sys


def load_email_data(csv_path: str = "data/email_data.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the email dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing columns x1, x2, x3, x4, x5, y.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, 5).
    y : np.ndarray
        Label vector of shape (n_samples,), with values +1 or -1.
    """
    # Load the CSV file using pandas
    df = pd.read_csv(csv_path)
    # Extract features and labels
    X = df[["x1", "x2", "x3", "x4", "x5"]].to_numpy()
    y = df["y"].to_numpy()
    return X, y


def estimate_nb_params(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Estimate Naive Bayes parameters from data.

    Parameters
    ----------
    X : np.ndarray
        Binary feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Labels of shape (n_samples,), values in {+1, -1}.

    Returns
    -------
    params : dict
        A dictionary containing:
          - 'class_prior': dict mapping y_value -> p(y)
          - 'feature_conditional': dict mapping
            y_value -> np.ndarray of shape (n_features,)
            with p(x_i=1 | y) for each feature i.
    """
    # TODO: replace with your implementation
    
    # Total number of samples and features
    n_samples, n_features = X.shape
    # Unique classes in y
    classes = np.unique(y)
    # Create dictionaries to hold parameters
    class_prior = {}
    feature_conditional = {}
    # Laplace smoothing parameter
    alpha = 1.0

    # Loop over each class to compute priors and conditionals
    for cls in classes:
        # Select samples belonging to class cls
        mask = (y == cls)
        Xc = X[mask]

        # Number of samples in class cls
        n_c = Xc.shape[0]

        # p(y=cls) = # of samples in class cls / total samples
        class_prior[cls] = n_c / n_samples

        # Since it's Bernoulli Naive Bayes, we only need p(x_i=1 | y=cls)
        # We can calculate p(x_i=0 | y=cls) as 1 - p(x_i=1 | y=cls)
        # Using Laplace smoothing:
        # p(x_i=1 | y=cls) = (count_ones + alpha) / (n_c + 2*alpha) for binary (2) features
        # Total count of ones for each feature in class cls
        count_ones = np.sum(Xc, axis=0)
        # Compute conditional probabilities (likelihood) with Laplace smoothing
        feature_conditional[cls] = (count_ones + alpha) / (n_c + 2.0 * alpha)

    return {
        "class_prior": class_prior,
        "feature_conditional": feature_conditional
    }


def predict_nb(x: np.ndarray, params: Dict[str, Any]) -> int:
    """
    Predict the class for a new feature vector using Naive Bayes.

    In case of a tie in posterior probabilities, predict +1.

    Parameters
    ----------
    x : np.ndarray
        1D array of shape (n_features,) with binary features.
    params : dict
        Parameters as returned by estimate_nb_params.

    Returns
    -------
    y_pred : int
        Predicted label (+1 or -1).
    """
    # TODO: replace with your implementation
    
    # Estimate_nb_params dict: cls -> p(y=cls)
    class_prior = params["class_prior"]

    # Estimate_nb_params dict: cls -> p(x_i=1|y=cls)          
    feature_cond = params["feature_conditional"]

    # Compute log posterior for each class: log p(y) + sum_i log p(x_i | y)
    log_post = {}
    for cls, py in class_prior.items():
        # Conditional probabilities (likelihood) for features given class
        p1 = np.asarray(feature_cond[cls], dtype=np.float64)  # p(x_i=1|y=cls)
        p0 = 1.0 - p1                                         # p(x_i=0|y=cls)

        # Bernoulli likelihood per feature:
        # p(x_i|y) = p1^x_i * p0^(1-x_i)
        # log p(x|y) = sum_i [x_i log p1_i + (1-x_i) log p0_i]
        log_lik = np.sum(x * np.log(p1) + (1 - x) * np.log(p0))
        log_post[cls] = np.log(py) + log_lik

    # Tie-breaking: predict +1
    log_pos = log_post.get(+1, -np.inf)
    log_neg = log_post.get(-1, -np.inf)
    return +1 if log_pos >= log_neg else -1


def posterior_nb(x: np.ndarray, params: Dict[str, Any]) -> float:
    """
    Compute the posterior probability p(y=+1 | x) under the Naive Bayes model.

    Parameters
    ----------
    x : np.ndarray
        1D array of shape (n_features,) with binary features.
    params : dict
        Parameters as returned by estimate_nb_params.

    Returns
    -------
    p_pos : float
        Posterior probability that y = +1 given x.
    """
    # TODO: replace with your implementation
    
    # Return a probability for y=+1 given x
    # Using Bayes' theorem: p(y=+1|x) = p(y=+1, x) / p(x)

    # Select class priors and conditionals from params
    class_prior = params["class_prior"]
    feature_cond = params["feature_conditional"]

    # Function to compute log joint probability log p(y, x)
    def log_joint(cls: int) -> float:
        py = class_prior[cls]
        p1 = np.asarray(feature_cond[cls], dtype=np.float64)
        p0 = 1.0 - p1
        log_lik = np.sum(x * np.log(p1) + (1 - x) * np.log(p0))
        return np.log(py) + log_lik

    # Compute log joint probabilities for both classes
    lp = log_joint(+1)
    ln = log_joint(-1)

    # Compute posterior p(y=+1|x) using log-sum-exp trick for numerical stability in the denominator
    # The demoninator is p(x) = p(y=+1,x) + p(y=-1,x)
    # The log equivalent of the numerator is lp for y=+1
    # The log equivalent of the demoninator is log p(x) = log (exp(lp) + exp(ln))
    # The log-sum-exp trick: subtract max(lp, ln) from both terms to avoid overflow
    m = max(lp, ln)
    # Normalization denominator
    denom = np.exp(lp - m) + np.exp(ln - m)
    # Final posterior probability
    return float(np.exp(lp - m) / denom)


def drop_feature_and_retrain(X: np.ndarray, y: np.ndarray):
    """
    Remove the first feature (x1), retrain the Naive Bayes model,
    and return predictions on the training set both before and after
    dropping x1.

    Parameters
    ----------
    X : np.ndarray
        Original feature matrix (n_samples, n_features).
    y : np.ndarray
        Labels (n_samples,).

    Returns
    -------
    preds_full : np.ndarray
        Predictions on X using all features.
    preds_reduced : np.ndarray
        Predictions on X_reduced (dropping first feature).
    """
    # TODO: replace with your implementation
    
    # Full model (all features)
    params_full = estimate_nb_params(X, y)
    preds_full = np.array([predict_nb(X[i], params_full) for i in range(X.shape[0])])

    # Reduced model (drop x1)
    X_reduced = X[:, 1:]
    params_reduced = estimate_nb_params(X_reduced, y)
    preds_reduced = np.array([predict_nb(X_reduced[i], params_reduced) for i in range(X_reduced.shape[0])])

    return preds_full, preds_reduced


def compare_accuracy(csv_path: str = "data/email_data.csv") -> str:
    """
    Load the dataset, train models with and without x1,
    and compare training accuracy.

    Returns a short string:
      - "improves"
      - "degrades"
      - "stays the same"

    indicating what happens to accuracy after removing x1.

    Parameters
    ----------
    csv_path : str
        Path to the email dataset CSV.

    Returns
    -------
    verdict : str
        One of {"improves", "degrades", "stays the same"}.
    """
    # TODO: replace with your implementation
    
    # Load data
    X, y = load_email_data(csv_path)

    # Get predictions before and after dropping feature (x1)
    preds_full, preds_reduced = drop_feature_and_retrain(X, y)

    # Compute accuracies for both models
    acc_full = np.mean(preds_full == y)
    acc_reduced = np.mean(preds_reduced == y)

    # Compare accuracies and return verdict
    if acc_reduced > acc_full:
        return "improves"
    elif acc_reduced < acc_full:
        return "degrades"
    else:
        return "stays the same"


def main() -> int:
    """
    Minimal demo runner for Problem 3.
    """

    # load_email_data
    try:
        X, y = load_email_data()
        print(f"load_email_data: X.shape={X.shape}, y.shape={y.shape}")
    except Exception as e:
        print(f"load_email_data: error: {e}")
        return 1

    # estimate_nb_params
    try:
        params = estimate_nb_params(X, y)
        print("estimate_nb_params: OK")
    except NotImplementedError:
        print("estimate_nb_params: NotImplemented")
        return 0
    except Exception as e:
        print(f"estimate_nb_params: error: {e}")
        return 1

    # predict_nb on first example
    try:
        y_pred0 = predict_nb(X[0], params)
        print(f"predict_nb: first example prediction={y_pred0}")
    except NotImplementedError:
        print("predict_nb: NotImplemented")
    except Exception as e:
        print(f"predict_nb: error: {e}")

    # posterior_nb on first example
    try:
        p_pos0 = posterior_nb(X[0], params)
        print(f"posterior_nb: p(y=+1|x0)={p_pos0:.4f}")
    except NotImplementedError:
        print("posterior_nb: NotImplemented")
    except Exception as e:
        print(f"posterior_nb: error: {e}")

    # drop_feature_and_retrain
    try:
        preds_full, preds_reduced = drop_feature_and_retrain(X, y)
        print(
            f"drop_feature_and_retrain: preds_full.shape={getattr(preds_full, 'shape', None)}, "
            f"preds_reduced.shape={getattr(preds_reduced, 'shape', None)}"
        )
    except NotImplementedError:
        print("drop_feature_and_retrain: NotImplemented")
    except Exception as e:
        print(f"drop_feature_and_retrain: error: {e}")

    # compare_accuracy
    try:
        verdict = compare_accuracy()
        print(f"compare_accuracy: {verdict}")
    except NotImplementedError:
        print("compare_accuracy: NotImplemented")
    except Exception as e:
        print(f"compare_accuracy: error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
