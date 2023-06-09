
import pandas as pd
import numpy as np
import copy
import os 
import csv
import io
import json
def helper():
    """
    Prints out the help message for this module.
    """
    print("This module contains a set of utility functions for data processing.")
    print("______________________________________________________________________")
    print("for detailed help call >>> help(speckle_utils.function_name) <<< ")
    print("______________________________________________________________________")
    print("available functions:")
    print("cleanData(data, mode='drop', num_only=False) -> clean dataframes, series or numpy arrays" )
    print( """ sort_and_match_df(A, B, uuid_column) -> merges two dataframes by a common uuid comon (best practice: always use this)""")
    print("transform_to_score(data, minPts, maxPts, t_low, t_high, cull_invalid=False) -> transform data to a score based on percentiles and provided points") 
    print("colab_create_directory(base_name) -> create a directory with the given name, if it already exists, add a number to the end of the name, usefull for colab")
    print("colab_zip_download_folder(dir_name) -> zips and downloads a directory from colab. will only work in google colaboratory ")

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


def sort_and_match_df(A, B, uuid_column):
    """
    Sorts and matches DataFrame B to A based on a shared uuid_column.
    
    Parameters:
    A, B (DataFrame): Input DataFrames to be sorted and matched.
    uuid_column (str): Shared column for matching rows.
    
    Returns:
    DataFrame: Resulting DataFrame after left join of A and B on uuid_column.
    """
    merged_df = pd.merge(A, B, on=uuid_column, how='left')
    return merged_df


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


def colab_create_directory(base_name):
    """ creates a directory with the given name, if it already exists, add a number to the end of the name.
    Usefull for colab to batch save e.g. images and avoid overwriting.
    Args:
        base_name (str): name of the directory to create
    Returns:
        str: name of the created directory"""
    counter = 1
    dir_name = base_name

    while os.path.exists(dir_name):
        dir_name = f"{base_name}_{counter}"
        counter += 1

    os.mkdir(dir_name)
    return dir_name

def colab_zip_download_folder(dir_name):
    """ zips and downloads a directory from colab. will only work in google colab
    Args: 
        dir_name (str): name of the directory to zip and download
    returns: 
        None, file will be downloaded to the local machine"""
    try:
        # zip the directory
        get_ipython().system('zip -r /content/{dir_name}.zip /content/{dir_name}')

        # download the zip file
        from google.colab import files
        files.download(f"/content/{dir_name}.zip")
    except:
        print("something went wrong, this function will only work in google colab, make sure to import the necessary packages. >>> from google.colab import files <<<" ) 




def generate__cluster_prompt(data_context, analysis_goal, column_descriptions, cluster_stat, complexity, exemplary_cluster_names_descriptions=None, creativity=None):
    # Define complexity levels
    complexity_levels = {
        1: "Please explain the findings in a simple way, suitable for someone with no knowledge of statistics or data science.",
        2: "Please explain the findings in moderate detail, suitable for someone with basic understanding of statistics or data science.",
        3: "Please explain the findings in great detail, suitable for someone with advanced understanding of statistics or data science."
    }

    # Start the prompt
    prompt = f"The data you are analyzing is from the following context: {data_context}. The goal of this analysis is: {analysis_goal}.\n\n"

    # Add column descriptions
    prompt += "The data consists of the following columns:\n"
    for column, description in column_descriptions.items():
        prompt += f"- {column}: {description}\n"

    # Add cluster stat and ask for generation
    prompt += "\nBased on the data, the following cluster has been identified:\n"
    prompt += f"\nCluster ID: {cluster_stat['cluster_id']}\n"
    for column, stats in cluster_stat['columns'].items():
        prompt += f"- {column}:\n"
        for stat, value in stats.items():
            prompt += f"  - {stat}: {value}\n"
    
    # Adjust the prompt based on whether examples are provided
    if exemplary_cluster_names_descriptions is not None and creativity is not None:
        prompt += f"\nPlease generate a name and description for this cluster, using a creativity level of {creativity} (where 0 is sticking closely to the examples and 1 is completely original). The examples provided are: {exemplary_cluster_names_descriptions}\n"
    else:
        prompt += "\nPlease generate a name and description for this cluster. Be creative and original in your descriptions.\n"

    prompt += "Please fill the following JSON template with the cluster name and two types of descriptions:\n"
    prompt += "{\n  \"cluster_name\": \"<generate>\",\n  \"description_narrative\": \"<generate>\",\n  \"description_statistical\": \"<generate>\"\n}\n"
    prompt += f"\nFor the narrative description, {complexity_levels[complexity]}"

    return prompt


def generate_cluster_description(cluster_df, original_df=None, stats_list=['mean', 'min', 'max', 'std', 'kurt'], cluster_id = ""):
    cluster_description = {"cluster_id": cluster_id,
                            "name":"<generate>",
                              "description_narrative":"<generate>",
                              "description_statistical":"<generate>",
                              "size": len(cluster_df),
                                "columns": {}
                                }
    if original_df is not None:
        size_relative = round(len(cluster_df)/len(original_df), 2)
    for column in cluster_df.columns:
        cluster_description["columns"][column] = {}
        for stat in stats_list:
            # Compute the statistic for the cluster
            if stat == 'mean':
                value = round(cluster_df[column].mean(),2)
            elif stat == 'min':
                value = round(cluster_df[column].min(),2)
            elif stat == 'max':
                value = round(cluster_df[column].max(),2)
            elif stat == 'std':
                value = round(cluster_df[column].std(), 2)
            elif stat == 'kurt':
                value = round(cluster_df[column].kurt(), 2)

            # Compute the relative difference if the original dataframe is provided
            if original_df is not None:
                original_value = original_df[column].mean() if stat == 'mean' else original_df[column].min() if stat == 'min' else original_df[column].max() if stat == 'max' else original_df[column].std() if stat == 'std' else original_df[column].kurt()
                relative_difference = (value - original_value) / original_value * 100
                cluster_description["columns"][column][stat] = {"value": round(value,2), "relative_difference": f"{round(relative_difference,2)}%"}
            else:
                cluster_description["columns"][column][stat] = {"value": round(value,2)}

    return cluster_description




def generate_cluster_description_mixed(cluster_df, original_df=None, stats_list=['mean', 'min', 'max', 'std', 'kurt'], cluster_id = ""):
    cluster_description = {
        "cluster_id": cluster_id,
        "name":"<generate>",
        "description_narrative":"<generate>",
        "description_statistical":"<generate>",
        "size": len(cluster_df),
        "columns": {}
    }

    if original_df is not None:
        size_relative = round(len(cluster_df)/len(original_df), 2)

    # Create CSV string in memory
    csv_io = io.StringIO()
    writer = csv.writer(csv_io)

    # CSV Headers
    writer.writerow(['Column', 'Stat', 'Value', 'Relative_Difference'])

    for column in cluster_df.columns:
        for stat in stats_list:
            if stat == 'mean':
                value = round(cluster_df[column].mean(),2)
            elif stat == 'min':
                value = round(cluster_df[column].min(),2)
            elif stat == 'max':
                value = round(cluster_df[column].max(),2)
            elif stat == 'std':
                value = round(cluster_df[column].std(), 2)
            elif stat == 'kurt':
                value = round(cluster_df[column].kurt(), 2)

            if original_df is not None:
                original_value = original_df[column].mean() if stat == 'mean' else original_df[column].min() if stat == 'min' else original_df[column].max() if stat == 'max' else original_df[column].std() if stat == 'std' else original_df[column].kurt()
                relative_difference = (value - original_value) / original_value * 100
                writer.writerow([column, stat, value, f"{round(relative_difference,2)}%"])
            else:
                writer.writerow([column, stat, value, "N/A"])

    # Store CSV data in JSON
    cluster_description["columns"] = csv_io.getvalue()

    data_description = """
        The input data is a JSON object with details about clusters. It has the following structure:

        1. 'cluster_id': An identifier for the cluster.
        2. 'name': A placeholder for the name of the cluster.
        3. 'description_narrative': A placeholder for a narrative description of the cluster.
        4. 'description_statistical': A placeholder for a statistical description of the cluster.
        5. 'size': The number of elements in the cluster.
        6. 'columns': This contains statistical data about different aspects, presented in CSV format. 

        In the 'columns' CSV:
        - 'Column' corresponds to the aspect.
        - 'Stat' corresponds to the computed statistic for that aspect in the cluster.
        - 'Value' is the value of that statistic.
        - 'Relative_Difference' is the difference of the statistic's value compared to the average value of this statistic in the entire dataset, expressed in percentages.
        """

    return cluster_description, data_description



