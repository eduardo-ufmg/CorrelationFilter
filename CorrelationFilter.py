import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to drop features that are highly
    correlated with others.

    Parameters:
    threshold : float, default=0.9
        The absolute correlation threshold above which a feature will be
        considered for dropping.

    Attributes:
    to_drop_ : list of str
        A list of feature names to be dropped from the DataFrame. This is
        learned during the `fit` method call.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        # Using a set for faster additions during the fitting process
        self._to_drop_set = set()

    def fit(self, X, y=None):
        """
        Learns which features to drop based on the correlation matrix.

        Parameters:
            X (pd.DataFrame): The input data from which to identify correlated features.
                              The DataFrame must have feature names (columns).
            y (pd.Series, optional): Ignored. Present for API consistency.

        Returns:
            self: The fitted transformer instance.
        """
        # Ensure the input is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HighCorrelationDropper requires a pandas DataFrame as input.")

        # Calculate the absolute correlation matrix
        corr_matrix = X.corr().abs()
        
        # We iterate over the upper triangle of the correlation matrix, excluding the diagonal.
        # This prevents checking pairs twice and a feature against itself.
        columns = corr_matrix.columns
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                
                # If the correlation between the two features is above the threshold
                value = corr_matrix.loc[col1, col2]
                if isinstance(value, (int, float)) and not pd.isna(value) and value >= self.threshold:
                    # We add the second feature of the pair to our drop set.
                    # This provides a deterministic way of choosing which feature to drop.
                    self._to_drop_set.add(col2)
        
        # Store the final list of columns to drop
        self.to_drop_ = list(self._to_drop_set)
        
        print(f"[HighCorrelationDropper] Fit complete. Features to drop: {self.to_drop_}")
        
        return self

    def transform(self, X):
        """
        Transforms the data by dropping the highly correlated features identified
        during the fit process.

        Parameters:
            X (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: The DataFrame with correlated features removed.
        """
        # Ensure the input is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HighCorrelationDropper requires a pandas DataFrame as input.")
            
        # Drop the identified columns
        X_transformed = X.drop(columns=self.to_drop_)
        
        print(f"[HighCorrelationDropper] Transformed data shape: {X_transformed.shape}")
        
        return X_transformed
