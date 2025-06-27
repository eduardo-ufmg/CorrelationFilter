import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to drop features that are highly
    correlated with others.

    Parameters:
    -----------
    threshold : float, default=0.9
        The absolute correlation threshold above which a feature will be
        considered for dropping.

    Attributes:
    -----------
    to_drop_indices_ : list of int
        A list of feature indices to be dropped. This is learned during the
        `fit` method call.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        # Using a set for faster additions during the fitting process
        self._to_drop_set = set()

    def fit(self, X, y=None):
        """
        Learns which feature indices to drop based on the correlation matrix.

        Parameters:
        -----------
            X (np.ndarray): The input data from which to identify correlated
                            features. It should have a shape of (n_samples, n_features).
            y (np.ndarray, optional): Ignored. Present for API consistency.

        Returns:
        --------
            self: The fitted transformer instance.
        """
        # Ensure the input is a NumPy array
        if not isinstance(X, np.ndarray):
            raise TypeError("CorrelationFilter requires a NumPy array as input.")

        # Calculate the absolute correlation matrix for the features (columns)
        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))

        # Get the number of features
        n_features = X.shape[1]

        # We iterate over the upper triangle of the correlation matrix,
        # excluding the diagonal.
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # If the correlation between two different features is above the threshold
                if corr_matrix[i, j] >= self.threshold:
                    # We add the index of the second feature in the pair to our drop set.
                    # This provides a deterministic way of choosing which feature to drop.
                    self._to_drop_set.add(j)

        # Store the final list of indices to drop, sorted for consistency
        self.to_drop_indices_ = sorted(list(self._to_drop_set))

        return self

    def transform(self, X):
        """
        Transforms the data by dropping the highly correlated features identified
        during the fit process.

        Parameters:
        -----------
            X (np.ndarray): The data to transform, with shape (n_samples, n_features).

        Returns:
        --------
            np.ndarray: The array with correlated features removed.
        """
        # Ensure the input is a NumPy array
        if not isinstance(X, np.ndarray):
            raise TypeError("CorrelationFilter requires a NumPy array as input.")

        # Drop the identified columns by index
        X_transformed = np.delete(X, self.to_drop_indices_, axis=1)

        return X_transformed
