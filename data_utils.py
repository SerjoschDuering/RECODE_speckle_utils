
import pandas as pd
import numpy as np
import copy


def cleanData(data, mode="drop", num_only=False):
    """
    Cleans data by handling missing or null values according to the specified mode.

    Args:
        data (numpy.ndarray, pandas.DataFrame, pandas.Series): Input data to be cleaned.
        mode (str, optional): Specifies the method to handle missing or null values.
                              "drop" drops rows with missing values (default),
                              "replace_zero" replaces missing values with zero,
                              "replace_mean" replaces missing values with the mean of the column.
        num_only (bool, optional): If True and data is a DataFrame, only numeric columns are kept. Defaults to False.

    Returns:
        numpy.ndarray, pandas.DataFrame, pandas.Series: Cleaned data with the same type as the input.

    Raises:
        ValueError: If the input data type is not supported (must be numpy.ndarray, pandas.DataFrame or pandas.Series).

    This function checks the type of the input data and applies the appropriate cleaning operation accordingly.
    It supports pandas DataFrame, pandas Series, and numpy array. For pandas DataFrame, it can optionally 
    convert and keep only numeric columns.
    """

    # check the type of input data
    if isinstance(data, pd.DataFrame):
        if num_only:
            data = data.select_dtypes(include=['int64', 'float64'])
        else:
          data_copy = data.copy()
          for col in data.columns:
              data[col] = pd.to_numeric(data[col], errors='coerce')
              data[col].fillna(data_copy[col], inplace=True)
  
        if mode == "drop":
            data = data.dropna()
        elif mode=="replace_zero":
            data = data.fillna(0)
        elif mode=="replace_mean":
            data = data.fillna(data.mean())

    elif isinstance(data, pd.Series):
        if mode == "drop":
            data = data.dropna()
        elif mode=="replace_zero":
            data = data.fillna(0)
        elif mode=="replace_mean":
            data = data.fillna(data.mean())

    elif isinstance(data, np.ndarray):
        if mode=="drop":
            data = data[~np.isnan(data)]
        elif mode=="replace_zero":
            data = np.nan_to_num(data, nan=0)
        elif mode=="replace_mean":
            data = np.where(np.isnan(data), np.nanmean(data), data)

    else:
        raise ValueError("Unsupported data type")

    return data


def transform_to_score(data, minPts, maxPts, t_low, t_high, cull_invalid=False):
    """
        Transforms data to a score based on percentiles and provided points.

        Args:
            data (numpy.array or pandas.Series): Input data to be transformed.
            minPts (float): The minimum points to be assigned.
            maxPts (float): The maximum points to be assigned.
            t_low (float): The lower percentile threshold.
            t_high (float): The upper percentile threshold.
            cull_invalid (bool, optional): If True, invalid data is removed. Defaults to False.

        Returns:
            numpy.array: The transformed data, where each element has been converted to a score based on its percentile rank.

        This function calculates the t_low and t_high percentiles of the input data, and uses linear interpolation
        to transform each data point to a score between minPts and maxPts. Any data point that falls above the t_high
        percentile is given a score of maxPts. If cull_invalid is True, any invalid data points (such as NaNs or 
        infinite values) are removed before the transformation is applied.
    """

    # If cull_invalid is True, the data is cleaned and invalid data is removed.
    if cull_invalid:
        inp_data = cleanData(inp_data, mode="drop", num_only=True)
    
    # Calculate the percentile values based on the data
    percentile_low = np.percentile(data, t_low)
    percentile_high = np.percentile(data, t_high)

    # Create a copy of the data to store the transformed points
    transformed_data = data.copy()

    # Apply linear interpolation between minPts and maxPts
    transformed_data = np.interp(transformed_data, [percentile_low, percentile_high], [minPts, maxPts])

    # Replace values above the percentile threshold with maxPts
    transformed_data[transformed_data >= percentile_high] = maxPts

    return transformed_data
