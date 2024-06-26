
import pandas as pd
import numpy as np
import copy
import os 
import csv
import io
import json
import requests

try:
    from fuzzywuzzy import fuzz
except:
    pass

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


def cleanData(data, mode="drop", num_only=False, print_report=True):
    """
    Cleans data by handling missing or null values according to the specified mode.

    Args:
        data (numpy.ndarray, pandas.DataFrame, pandas.Series): Input data to be cleaned.
        mode (str, optional): Specifies the method to handle missing or null values.
                              "drop" drops rows with missing values (default),
                              "replace_zero" replaces missing values with zero,
                              "replace_mean" replaces missing values with the mean of the column.
        num_only (bool, optional): If True and data is a DataFrame, only numeric columns are kept. Defaults to False.#
        print_report (bool, optional): if True the report is printed to the console. Defaults to True.

    Returns:
        numpy.ndarray, pandas.DataFrame, pandas.Series: Cleaned data with the same type as the input.


    Raises:
        ValueError: If the input data type is not supported (must be numpy.ndarray, pandas.DataFrame or pandas.Series).

    This function checks the type of the input data and applies the appropriate cleaning operation accordingly.
    It supports pandas DataFrame, pandas Series, and numpy array. For pandas DataFrame, it can optionally 
    convert and keep only numeric columns.
    """
    report = {}
    if isinstance(data, pd.DataFrame):
        initial_cols = data.columns.tolist()
        initial_rows = data.shape[0]
        if num_only:
            # attempt casting before doing this selection
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.select_dtypes(include=['int64', 'float64'])
            report['dropped_cols'] = list(set(initial_cols) - set(data.columns.tolist()))

        if mode == "drop":
            data = data.dropna()
            report['dropped_rows'] = initial_rows - data.shape[0]
        elif mode=="replace_zero":
            data = data.fillna(0)
        elif mode=="replace_mean":
            data = data.fillna(data.mean())

    elif isinstance(data, pd.Series):
        initial_length = len(data)
        if mode == "drop":
            data = data.dropna()
            report['dropped_rows'] = initial_length - len(data)
        elif mode=="replace_zero":
            data = data.fillna(0)
        elif mode=="replace_mean":
            data = data.fillna(data.mean())

    elif isinstance(data, np.ndarray):
        initial_length = data.size
        if mode=="drop":
            data = data[~np.isnan(data)]
            report['dropped_rows'] = initial_length - data.size
        elif mode=="replace_zero":
            data = np.nan_to_num(data, nan=0)
        elif mode=="replace_mean":
            data = np.where(np.isnan(data), np.nanmean(data), data)

    else:
        raise ValueError("Unsupported data type")
    if print_report:
        print(report)   
    return data



def sort_and_match_df(A, B, uuid_column):
    """
    Sorts and matches DataFrame B to A based on a shared uuid_column.
    Prioritizes uuid_column as an index if present, otherwise uses it as a column.
    
    Parameters:
    A, B (DataFrame): Input DataFrames to be sorted and matched.
    uuid_column (str): Shared column/index for matching rows.
    
    Returns:
    DataFrame: Resulting DataFrame after left join of A and B on uuid_column.
    """
    if uuid_column in A.columns:
        A = A.set_index(uuid_column, drop=False)
    if uuid_column in B.columns:
        B = B.set_index(uuid_column, drop=False)

    merged_df = pd.merge(A, B, left_index=True, right_index=True, how='left')
    return merged_df.reset_index(drop=False)


def sort_and_match_dfs(dfs, uuid_column):
    """
    Sorts and matches all DataFrames in list based on a shared uuid_column.
    Prioritizes uuid_column as an index if present, otherwise uses it as a column.
    Raises a warning if any two DataFrames have overlapping column names.
    
    Parameters:
    dfs (list): A list of DataFrames to be sorted and matched.
    uuid_column (str): Shared column/index for matching rows.
    
    Returns:
    DataFrame: Resulting DataFrame after successive left joins on uuid_column.
    """
    if not dfs:
        raise ValueError("The input list of DataFrames is empty")
    
    # Convert uuid_column to index if it's a column
    for i, df in enumerate(dfs):
        if uuid_column in df.columns:
            dfs[i] = df.set_index(uuid_column, drop=False)

    # Check for overlapping column names
    all_columns = [set(df.columns) for df in dfs]
    for i, columns_i in enumerate(all_columns):
        for j, columns_j in enumerate(all_columns[i+1:], start=i+1):
            overlapping_columns = columns_i.intersection(columns_j) - {uuid_column}
            if overlapping_columns:
                print(f"Warning: DataFrames at indices {i} and {j} have overlapping column(s): {', '.join(overlapping_columns)}")
    
    result_df = dfs[0]
    for df in dfs[1:]:
        result_df = pd.merge(result_df, df, left_index=True, right_index=True, how='left')

    return result_df.reset_index(drop=False)




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

def smart_round(x):
    if abs(x) >= 1000:
        return round(x)
    elif abs(x) >= 10:
        return round(x, 1)
    elif abs(x) >= 1:
        return round(x, 2)
    else:
        return round(x, 3)

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

# ==================================================================================================
# ========== TESTING ===============================================================================

def compare_column_names(ref_list, check_list):
    """
    Compares two lists of column names to check for inconsistencies.

    Args:
        ref_list (list): The reference list of column names.
        check_list (list): The list of column names to be checked.

    Returns:
        report_dict (dict): Report about the comparison process.

    Raises:
        ValueError: If the input types are not list.
    """
    # Check the type of input data
    if not all(isinstance(i, list) for i in [ref_list, check_list]):
        raise ValueError("Both inputs must be of type list")

    missing_cols = [col for col in ref_list if col not in check_list]
    extra_cols = [col for col in check_list if col not in ref_list]
    
    try:
      typos = {}
      for col in check_list:
          if col not in ref_list:
              similarity_scores = {ref_col: fuzz.ratio(col, ref_col) for ref_col in ref_list}
              likely_match = max(similarity_scores, key=similarity_scores.get)
              if similarity_scores[likely_match] > 70:  # you may adjust this threshold as needed
                  typos[col] = likely_match
    except:
      typos = {"error":"fuzzywuzzy is probably not installed"}

    report_dict = {
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "likely_typos": typos
    }

    print("\nREPORT:")
    print('-'*50)
    print("\n- Missing columns:")
    print('   ' + '\n   '.join(f'"{col}"' for col in missing_cols) if missing_cols else '   None')
    print("\n- Extra columns:")
    print('   ' + '\n   '.join(f'"{col}"' for col in extra_cols) if extra_cols else '   None')
    print("\n- Likely typos:")
    if typos:
        for k, v in typos.items():
            print(f'   "{k}": "{v}"')
    else:
        print('   None')

    return report_dict


def compare_dataframes(df1, df2, threshold=0.1):
    """
    Compare two pandas DataFrame and returns a report highlighting any significant differences.
    Significant differences are defined as differences that exceed the specified threshold.

    Args:
        df1, df2 (pandas.DataFrame): Input dataframes to be compared.
        threshold (float): The percentage difference to be considered significant. Defaults to 0.1 (10%).

    Returns:
        pandas.DataFrame: A report highlighting the differences between df1 and df2.
    """
    # Column comparison
    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)
    
    common_cols = cols_df1 & cols_df2
    missing_df1 = cols_df2 - cols_df1
    missing_df2 = cols_df1 - cols_df2

    print("Column Comparison:")
    print("------------------")
    print(f"Common columns ({len(common_cols)}): {sorted(list(common_cols)) if common_cols else 'None'}")
    print(f"Columns missing in df1 ({len(missing_df1)}): {sorted(list(missing_df1)) if missing_df1 else 'None'}")
    print(f"Columns missing in df2 ({len(missing_df2)}): {sorted(list(missing_df2)) if missing_df2 else 'None'}")
    print("\n")

    # Check for new null values
    print("Null Values Check:")
    print("------------------")
    inconsistent_values_cols = []
    inconsistent_ranges_cols = []
    constant_cols = []
    
    for col in common_cols:
        nulls1 = df1[col].isnull().sum()
        nulls2 = df2[col].isnull().sum()
        if nulls1 == 0 and nulls2 > 0:
            print(f"New null values detected in '{col}' of df2.")

        # Check for value consistency
        if df1[col].nunique() <= 10 and df2[col].nunique() <= 10:
            inconsistent_values_cols.append(col)

        
          # Check for range consistency
        if df1[col].dtype.kind in 'if' and df2[col].dtype.kind in 'if':
          range1 = df1[col].max() - df1[col].min()
          range2 = df2[col].max() - df2[col].min()
          diff = abs(range1 - range2)
          mean_range = (range1 + range2) / 2
          if diff / mean_range * 100 > threshold * 100:
              inconsistent_ranges_cols.append(col)

        # Check for constant columns
        if len(df1[col].unique()) == 1 or len(df2[col].unique()) == 1:
            constant_cols.append(col)

    # Print out the results of value consistency, range consistency, and constant columns check
    print("\nValue Consistency Check:")
    print("------------------------")
    print(f"Columns with inconsistent values (checks if the unique values are the same in both dataframes): {inconsistent_values_cols if inconsistent_values_cols else 'None'}")
    
    print("\nRange Consistency Check (checks if the range (max - min) of the values in both dataframes is consistent):")
    print("------------------------")
    print(f"Columns with inconsistent ranges: {inconsistent_ranges_cols if inconsistent_ranges_cols else 'None'}")
    
    print("\nConstant Columns Check (columns that have constant values in either dataframe):")
    print("-----------------------")
    print(f"Constant columns: {constant_cols if constant_cols else 'None'}")

    # Check for changes in data type
    print("\nData Type Check:")
    print("----------------")
    for col in common_cols:
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype
        if dtype1 != dtype2:
            print(f"df1 '{dtype1}' -> '{dtype2}' in df2, Data type for '{col}' has changed.")
    print("\n")
    
  

    report_dict = {"column": [], "statistic": [], "df1": [], "df2": [], "diff%": []}
    statistics = ["mean", "std", "min", "25%", "75%", "max", "nulls", "outliers"]

    for col in common_cols:
        if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']:
            desc1 = df1[col].describe()
            desc2 = df2[col].describe()
            for stat in statistics[:-2]:
                report_dict["column"].append(col)
                report_dict["statistic"].append(stat)
                report_dict["df1"].append(desc1[stat])
                report_dict["df2"].append(desc2[stat])
                diff = abs(desc1[stat] - desc2[stat])
                mean = (desc1[stat] + desc2[stat]) / 2
                report_dict["diff%"].append(diff / mean * 100 if mean != 0 else 0)  # Fix for division by zero
            nulls1 = df1[col].isnull().sum()
            nulls2 = df2[col].isnull().sum()
            outliers1 = df1[(df1[col] < desc1["25%"] - 1.5 * (desc1["75%"] - desc1["25%"])) | 
                            (df1[col] > desc1["75%"] + 1.5 * (desc1["75%"] - desc1["25%"]))][col].count()
            outliers2 = df2[(df2[col] < desc2["25%"] - 1.5 * (desc2["75%"] - desc2["25%"])) | 
                            (df2[col] > desc2["75%"] + 1.5 * (desc2["75%"] - desc2["25%"]))][col].count()
            for stat, value1, value2 in zip(statistics[-2:], [nulls1, outliers1], [nulls2, outliers2]):
                report_dict["column"].append(col)
                report_dict["statistic"].append(stat)
                report_dict["df1"].append(value1)
                report_dict["df2"].append(value2)
                diff = abs(value1 - value2)
                mean = (value1 + value2) / 2
                report_dict["diff%"].append(diff / mean * 100 if mean != 0 else 0)  # Fix for division by zero

    report_df = pd.DataFrame(report_dict)
    report_df["significant"] = report_df["diff%"] > threshold * 100
    report_df = report_df[report_df["significant"]]
    report_df = report_df.round(2)

    print(f"REPORT:\n{'-'*50}")
    for col in report_df["column"].unique():
        print(f"\n{'='*50}")
        print(f"Column: {col}\n{'='*50}")
        subset = report_df[report_df["column"]==col][["statistic", "df1", "df2", "diff%"]]
        subset.index = subset["statistic"]
        print(subset.to_string(header=True))

    return report_df


def notion_db_as_df(database_id, token):
    base_url = "https://api.notion.com/v1"

    # Headers for API requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    response = requests.post(f"{base_url}/databases/{database_id}/query", headers=headers)
    # response.raise_for_status()  # Uncomment to raise an exception for HTTP errors
    pages = response.json().get('results', [])
    print(response.json().keys())

    # Used to create df
    table_data = {}
    page_cnt = len(pages)
    for i, page in enumerate(pages):
        for cur_col, val in page["properties"].items():
            if cur_col not in table_data:
                table_data[cur_col] = [None] * page_cnt
            val_type = val["type"]
            if val_type == "title":
                value = val[val_type][0]["text"]["content"]
            elif val_type in ["number", "checkbox"]:
                value = val[val_type]
            elif val_type in ["select", "multi_select"]:
                value = ', '.join([option["name"] for option in val[val_type]])
            elif val_type == "date":
                value = val[val_type]["start"]
            elif val_type in ["people", "files"]:
                value = ', '.join([item["id"] for item in val[val_type]])
            elif val_type in ["url", "email", "phone_number"]:
                value = val[val_type]
            elif val_type == "formula":
                value = val[val_type]["string"] if "string" in val[val_type] else val[val_type]["number"]
            elif val_type == "rich_text":
              value = val[val_type][0]["text"]["content"]
            else:
                value = str(val[val_type])  # Fallback to string representation
            table_data[cur_col][i] = value

    # To DataFrame
    df = pd.DataFrame(table_data)
    return df




def send_matrices_and_create_commit_vec(matrices, client, stream_id, branch_name, commit_message, rows_per_chunk, containerMetadata):
    """
    Send matrices to the Speckle server and create a commit.

    Parameters:
    matrices (dict or pd.DataFrame): A dictionary of DataFrames, where each key is the matrix name and the value is the DataFrame.
                                      If a single DataFrame is provided, it will be wrapped into a dictionary with the key 'unnamed_matrix'.
    client (SpeckleClient): The Speckle client used for communication with the server.
    stream_id (str): The ID of the stream where the matrices will be sent.
    branch_name (str): The name of the branch to which the commit will be made.
    commit_message (str): The message for the commit.
    rows_per_chunk (int): The number of rows per chunk to be sent.
    containerMetadata (dict): Metadata to be added to the container object.

    Returns:
    str: The ID of the created commit.

    Raises:
    Exception: If there is an error during the process.
    """
    # If a single DataFrame is provided, wrap it in a dictionary
    if isinstance(matrices, pd.DataFrame):
        matrices = {"unnamed_matrix": matrices}

    transport = ServerTransport(client=client, stream_id=stream_id)
    matrix_ids = {}

    container_object = Base()
    # Add to container object
    for k, v in containerMetadata.items():
        container_object[k] = v

    # Initialize the keys
    for k, df in matrices.items():
        container_object[k] = Base()

    for k, df in matrices.items():
        matrix_object = Base(metaData="Some metadata")
        # Convert DataFrame to NumPy array for efficient processing
        matrix = df.to_numpy()
        indices = list(df.index.to_numpy().astype(str))
        cols = list(df.columns)

        # Round the matrix using a vectorized operation
        rounded_matrix = round_matrix(matrix).tolist()

        # Chunk the rows and create Speckle objects
        chunks = []
        row_lists = []
        for row in rounded_matrix:
            row_obj = Base()  # Create a new Speckle object for each row
            set_speckle_attribute(row_obj, "@row", row)
            row_lists.append(row_obj)

            # If we reach the chunk size, create a new chunk
            if len(row_lists) >= rows_per_chunk:
                chunk_obj = Base()  # Create a new Speckle object for the chunk
                set_speckle_attribute(chunk_obj, "@rows", row_lists)
                chunks.append(chunk_obj)
                row_lists = []  # Reset the row list for the next chunk

        # Add any remaining rows as a final chunk
        if row_lists:
            chunk_obj = Base()  # Create a new Speckle object for the chunk
            set_speckle_attribute(chunk_obj, "@rows", row_lists)
            chunks.append(chunk_obj)

        # Create the main Speckle object
        obj = Base()
        set_speckle_attribute(obj, "@originUUID", indices)
        set_speckle_attribute(obj, "@destinationUUID", cols)
        set_speckle_attribute(obj, "@chunks", chunks)

        # Add the matrix object to the container
        container_object[k] = obj

        print(container_object)

    try:
        container_objectid = operations.send(base=container_object, transports=[transport])
        print(f"Container Object ID: {container_objectid}")
        commit_id = client.commit.create(stream_id=stream_id, object_id=container_objectid, branch_name=branch_name, message=commit_message)
        print(f"Commit ID: {commit_id}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            logging.error(f"Response JSON: {e.response.json()}")
        raise

    return commit_id