#speckle utils
import json 
import pandas as pd
import numpy as np
import numpy as np
from numba import jit
import specklepy
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_default_account, get_local_accounts
from specklepy.transports.server import ServerTransport
from specklepy.api import operations
from specklepy.objects.geometry import Polyline, Point, Mesh
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_default_account
from specklepy.transports.server import ServerTransport
from specklepy.objects import Base
from specklepy.api import operations
import logging
from specklepy.api.wrapper import StreamWrapper
import urllib.parse
import requests
import inspect

try:
    import openai
except:
    pass

import requests
from datetime import datetime
import copy


# HELP FUNCTION ===============================================================
def helper():
    """
    Prints out the help message for this module.
    """
    print("This module contains a set of utility functions for speckle streams.")
    print("______________________________________________________________________")
    print("It requires the specklepy package to be installed -> !pip install specklepy")
    print("the following functions are available:")
    print("getSpeckleStream(stream_id, branch_name, client)")
    print("getSpeckleGlobals(stream_id, client)")
    print("get_dataframe(objects_raw, return_original_df)")
    print("updateStreamAnalysis(stream_id, new_data, branch_name, geometryGroupPath, match_by_id, openai_key, return_original)")
    print("there are some more function available not documented fully yet, including updating a notion database")
    print("______________________________________________________________________")
    print("for detailed help call >>> help(speckle_utils.function_name) <<< ")
    print("______________________________________________________________________")
    print("standard usage:")
    print("______________________________________________________________________")
    print("retreiving data")
    print("1. import speckle_utils & speckle related libaries from specklepy")
    print("2. create a speckle client -> client = SpeckleClient(host='https://speckle.xyz/')" )
    print("                              client.authenticate_with_token(token='your_token_here')")
    print("3. get a speckle stream -> stream = speckle_utils.getSpeckleStream(stream_id, branch_name, client)")
    print("4. get the stream data -> data = stream['pth']['to']['data']")
    print("5. transform data to dataframe -> df = speckle_utils.get_dataframe(data, return_original_df=False)")
    print("______________________________________________________________________")
    print("updating data")
    print("1. call updateStreamAnalysis --> updateStreamAnalysis(new_data, stream_id, branch_name, geometryGroupPath, match_by_id, openai_key, return_original)")


#==============================================================================

def updateSpeckleStream(stream_id,
                        branch_name,
                        client,
                        data_object,
                        commit_message="Updated the data object",
                        ):
    """
    Updates a speckle stream with a new data object.

    Args:
        stream_id (str): The ID of the speckle stream.
        branch_name (str): The name of the branch within the speckle stream.
        client (specklepy.api.client.Client): A speckle client.
        data_object (dict): The data object to send to the speckle stream.
        commit_message (str): The commit message. Defaults to "Updated the data object".
    """
    # set stream and branch
    branch = client.branch.get(stream_id, branch_name)
    # Get transport
    transport = ServerTransport(client=client, stream_id=stream_id)
    # Send the data object to the speckle stream
    object_id = operations.send(data_object, [transport])

    # Create a new commit with the new object
    commit_id = client.commit.create(
        stream_id,
        object_id= object_id,
        message=commit_message,
        branch_name=branch_name,
    )

    return commit_id
def getSpeckleStream(stream_id,
                     branch_name,
                     client,
                     commit_id=""
                     ):
    """
    Retrieves data from a specific branch of a speckle stream.

    Args:
        stream_id (str): The ID of the speckle stream.
        branch_name (str): The name of the branch within the speckle stream.
        client (specklepy.api.client.Client, optional): A speckle client. Defaults to a global `client`.
        commit_id (str): id of a commit, if nothing is specified, the latest commit will be fetched

    Returns:
        dict: The speckle stream data received from the specified branch.

    This function retrieves the last commit from a specific branch of a speckle stream.
    It uses the provided speckle client to get the branch and commit information, and then 
    retrieves the speckle stream data associated with the last commit.
    It prints out the branch details and the creation dates of the last three commits for debugging purposes.
    """

    print("updated A")

    # set stream and branch
    try:
        branch = client.branch.get(stream_id, branch_name, 3)
        print(branch)
    except:
        branch = client.branch.get(stream_id, branch_name, 1)
        print(branch)

    print("last three commits:")
    [print(ite.createdAt) for ite in branch.commits.items]

    if commit_id == "":
        latest_commit = branch.commits.items[0]
        choosen_commit_id = latest_commit.id
        commit = client.commit.get(stream_id, choosen_commit_id)
        print("latest commit ", branch.commits.items[0].createdAt, " was choosen")
    elif type(commit_id) == type("s"): # string, commit uuid
        choosen_commit_id = commit_id
        commit = client.commit.get(stream_id, choosen_commit_id)
        print("provided commit ", choosen_commit_id, " was choosen")
    elif type(commit_id) == type(1): #int 
        latest_commit = branch.commits.items[commit_id]
        choosen_commit_id = latest_commit.id
        commit = client.commit.get(stream_id, choosen_commit_id)


    print(commit)
    print(commit.referencedObject)
    # get transport
    transport = ServerTransport(client=client, stream_id=stream_id)
    #speckle stream
    res = operations.receive(commit.referencedObject, transport)

    return res
 
def getSpeckleGlobals(stream_id, client):
    """
    Retrieves global analysis information from the "globals" branch of a speckle stream.

    Args:
        stream_id (str): The ID of the speckle stream.
        client (specklepy.api.client.Client, optional): A speckle client. Defaults to a global `client`.

    Returns:
        analysisInfo (dict or None): The analysis information retrieved from globals. None if no globals found.
        analysisGroups (list or None): The analysis groups retrieved from globals. None if no globals found.

    This function attempts to retrieve and parse the analysis information from the "globals" 
    branch of the specified speckle stream. It accesses and parses the "analysisInfo" and "analysisGroups" 
    global attributes, extracts analysis names and UUIDs.
    If no globals are found in the speckle stream, it returns None for both analysisInfo and analysisGroups.
    """
    # get the latest commit
    try:
        # speckle stream globals
        branchGlob = client.branch.get(stream_id, "globals")
        latest_commit_Glob = branchGlob.commits.items[0]
        transport = ServerTransport(client=client, stream_id=stream_id)

        globs = operations.receive(latest_commit_Glob.referencedObject, transport)
        
        # access and parse globals
        #analysisInfo = json.loads(globs["analysisInfo"]["@{0;0;0;0}"][0].replace("'", '"'))
        #analysisGroups = [json.loads(gr.replace("'", '"')) for gr in globs["analysisGroups"]["@{0}"]]

        def get_error_context(e, context=100):
            start = max(0, e.pos - context)
            end = e.pos + context
            error_line = e.doc[start:end]
            pointer_line = ' ' * (e.pos - start - 1) + '^'
            return error_line, pointer_line

        try:
            analysisInfo = json.loads(globs["analysisInfo"]["@{0;0;0;0}"][0].replace("'", '"').replace("None", "null"))
        except json.JSONDecodeError as e:
            print(f"Error decoding analysisInfo: {e}")
            error_line, pointer_line = get_error_context(e)
            print("Error position and surrounding text:")
            print(error_line)
            print(pointer_line)
            analysisInfo = None

        try:
            analysisGroups = [json.loads(gr.replace("'", '"').replace("None", "null")) for gr in globs["analysisGroups"]["@{0}"]]
        except json.JSONDecodeError as e:
            print(f"Error decoding analysisGroups: {e}")
            error_line, pointer_line = get_error_context(e)
            print("Error position and surrounding text:")
            print(error_line)
            print(pointer_line)
            analysisGroups = None



        # extract analysis names 
        analysis_names = []
        analysis_uuid = []
        [(analysis_names.append(key.split("++")[0]),analysis_uuid.append(key.split("++")[1]) ) for key in analysisInfo.keys()]


        # print extracted results
        print("there are global dictionaries with additional information for each analysis")
        print("<analysisGroups> -> ", [list(curgrp.keys()) for curgrp in analysisGroups])
        print("<analysis_names> -> ", analysis_names)                       
        print("<analysis_uuid>  -> ", analysis_uuid)
    except Exception as e:  # catch exception as 'e'
        analysisInfo = None
        analysisGroups = None
        print("No GlOBALS FOUND")
        print(f"Error: {e}")  # print error description
  
    return analysisInfo, analysisGroups



#function to extract non geometry data from speckle 
def get_dataframe(objects_raw, return_original_df=False):
    """
    Creates a pandas DataFrame from a list of raw Speckle objects.

    Args:
        objects_raw (list): List of raw Speckle objects.
        return_original_df (bool, optional): If True, the function also returns the original DataFrame before any conversion to numeric. Defaults to False.

    Returns:
        pd.DataFrame or tuple: If return_original_df is False, returns a DataFrame where all numeric columns have been converted to their respective types, 
                               and non-numeric columns are left unchanged. 
                               If return_original_df is True, returns a tuple where the first item is the converted DataFrame, 
                               and the second item is the original DataFrame before conversion.

    This function iterates over the raw Speckle objects, creating a dictionary for each object that excludes the '@Geometry' attribute. 
    These dictionaries are then used to create a pandas DataFrame. 
    The function attempts to convert each column to a numeric type if possible, and leaves it unchanged if not. 
    Non-convertible values in numeric columns are replaced with their original values.
    """
    # dataFrame
    df_data = []
    # Iterate over speckle objects
    for obj_raw in objects_raw:
        obj = obj_raw.__dict__
        df_obj = {k: v for k, v in obj.items() if k != '@Geometry'}
        df_data.append(df_obj)

    # Create DataFrame and GeoDataFrame
    df = pd.DataFrame(df_data)
    # Convert columns to float or int if possible, preserving non-convertible values <-
    df_copy = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df_copy[col], inplace=True)

    if return_original_df:
        return df, df_copy
    else:
        return df
    

def updateStreamAnalysis(
          client,
          new_data,
          stream_id,
          branch_name,
          geometryGroupPath=None,
          match_by_id="",
          openai_key ="",
          return_original = False
      ):
  

    """
    Updates Stream Analysis by modifying object attributes based on new data.

    Args:
        new_data (pandas.DataFrame): DataFrame containing new data.
        stream_id (str): Stream ID.
        branch_name (str): Branch name.
        geometry_group_path (list, optional): Path to geometry group. Defaults to ["@Data", "@{0}"].
        match_by_id (str, optional): key for column that should be used for matching. If empty, the index is used.
        openai_key (str, optional): OpenAI key. If empty no AI commit message is generated Defaults to an empty string.
        return_original (bool, optional): Determines whether to return original speckle stream objects. Defaults to False.

    Returns:
        list:  original speckle stream objects as backup if return_original is set to True.

    This function retrieves the latest commit from a specified branch, obtains the 
    necessary geometry objects, and matches new data with existing objects using 
    an ID mapper. The OpenAI GPT model is optionally used to create a commit summary 
    message. Changes are sent back to the server and a new commit is created, with 
    the original objects returned as a backup if return_original is set to True. 
    The script requires active server connection, necessary permissions, and relies 
    on Speckle and OpenAI's GPT model libraries.
    """
    print("1")
    if geometryGroupPath == None:
        geometryGroupPath = ["@Speckle", "Geometry"]

    branch = client.branch.get(stream_id, branch_name, 2)

    latest_commit = branch.commits.items[0]
    commitID = latest_commit.id 

    commit = client.commit.get(stream_id, commitID)

    # get objects
    transport = ServerTransport(client=client, stream_id=stream_id)

    #speckle stream
    res = operations.receive(commit.referencedObject, transport)

    # get geometry objects (they carry the attributes)
    objects_raw = res[geometryGroupPath[0]][geometryGroupPath[1]]
    res_new = copy.deepcopy(res)
    print("2")
    # map ids 
    id_mapper = {}
    if match_by_id != "":
        for i, obj in enumerate(objects_raw):
            id_mapper[obj[match_by_id]] = i
    else:
        for i, obj in enumerate(objects_raw):
            id_mapper[str(i)] = i
    print("3")
    # iterate through rows (objects)
    for index, row in new_data.iterrows():
        #determin target object 
        if match_by_id != "":
            local_id = row[match_by_id]
        else:
            local_id = index
        target_id = id_mapper[local_id]     

        #iterate through columns (attributes)
        for col_name in new_data.columns:
            res_new[geometryGroupPath[0]][geometryGroupPath[1]][target_id][col_name] = row[col_name]

    print("4")
    # ======================== OPEN AI FUN ===========================
    """
    try:
        try:
            answer_summary = gptCommitMessage(objects_raw, new_data,openai_key)
            if answer_summary == None:
                _, answer_summary = compareStats(get_dataframe(objects_raw),new_data)
        except:
            _, answer_summary = compareStats(get_dataframe(objects_raw),new_data)
    except:
        answer_summary = ""
    """
    answer_summary = ""
    # ================================================================
    print("5")
    new_objects_raw_speckle_id = operations.send(base=res_new, transports=[transport])
    print("6")
    # You can now create a commit on your stream with this object
    commit_id = client.commit.create(
        stream_id=stream_id,
        branch_name=branch_name,
        object_id=new_objects_raw_speckle_id,
        message="Updated item in colab -" + answer_summary,
        )
    print("7")
    print("Commit created!")
    if return_original:
        return objects_raw #as back-up


def updateStreamAnalysisFast(client, new_data_in, stream_id, branch_name, geometryGroupPath=None, match_by_id="", return_original = False, commit_id=None):
    """
    Updates data on a Speckle stream by matching and synchronizing new data inputs with existing data objects within a specified stream and branch. 
    This function is designed to be efficient in processing and updating large datasets by leveraging data frame operations and direct data manipulation.

    Parameters:
    - client: SpeckleClient object, used to interact with the Speckle server.
    - new_data_in (DataFrame): The new data to be updated on the stream. Must include a column for matching with existing data objects if `match_by_id` is specified.
    - stream_id (str): The unique identifier of the Speckle stream to be updated.
    - branch_name (str): The name of the branch within the stream where the data update is to take place.
    - geometryGroupPath (list of str, optional): The path to the geometry group within the Speckle data structure. Defaults to ["@Speckle", "Geometry"] if not provided.
    - match_by_id (str, optional): The column name in `new_data_in` used to match data points with existing objects in the Speckle stream. Defaults to an empty string, which implies no matching by ID.
    - return_original (bool, optional): If True, returns the original data objects from the Speckle stream as a backup. Defaults to False.
    - commit_id (str, optional): The commit ID to use for fetching the existing data. If not provided, the latest commit on the specified branch is used.

    Returns:
    - If `return_original` is True, returns the original data objects from the Speckle stream as a list or DataFrame. Otherwise, returns None.

    Requires:
    - The `new_data_in` DataFrame must contain a column with the name provided in `match_by_id` (if specified) for matching data points with existing data points on the Speckle stream.
    - A valid SpeckleClient object authenticated and connected to the desired Speckle server.

    """
    
    if geometryGroupPath is None:
        geometryGroupPath = ["@Speckle", "Geometry"]
    new_data = new_data_in.copy()


    branch = client.branch.get(stream_id, branch_name, 2)
    latest_commit = branch.commits.items[0]
    print(latest_commit)
    commit = client.commit.get(stream_id, latest_commit.id)
    transport = ServerTransport(client=client, stream_id=stream_id)
    res = operations.receive(commit.referencedObject, transport)
    objects_raw = res[geometryGroupPath[0]][geometryGroupPath[1]]

    # unique columns
    uniqu_cols = list(new_data.columns)
    print("uniqu_cols", uniqu_cols)


    # Pre-create a mapping from IDs to objects for faster lookup
    id_to_object_map = {obj[match_by_id]: obj for obj in objects_raw} if match_by_id else {i: obj for i, obj in enumerate(objects_raw)}

    # Pre-process DataFrame if match_by_id is provided
    if match_by_id:
        new_data.set_index(match_by_id, inplace=True, drop=False)

     # First, update objects with available data from new_data
    for local_id, updates in new_data.iterrows():
        target_object = id_to_object_map.get(str(local_id))
        if target_object:
            for col_name in uniqu_cols:  # Iterate over all columns
                value = updates.get(col_name, "NA")  # Fetch update or default to "NA"
                target_object[col_name] = value


    # Now, ensure all objects have all columns, adding "NA" where data is missing
    for obj_id, obj in id_to_object_map.items():
        # Check for each unique column in the object

        for col_name in uniqu_cols:
          if col_name not in obj.__dict__.keys():
              obj[col_name] = "NA"  # Add "NA" if the column is missing

    toPrint = [obj.__dict__[match_by_id] for obj in objects_raw]
    print("uuids:", toPrint)
    if "NA" in obj.__dict__[match_by_id]:
      print("!!!!! UUIDS not found anymore - abort commit !!!!!")
      return "fail"
    # Send updated objects back to Speckle
    new_objects_raw_speckle_id = operations.send(base=res, transports=[transport])
    commit_id = client.commit.create(stream_id=stream_id, branch_name=branch_name, object_id=new_objects_raw_speckle_id, message="Updated item in colab")
    print("commit created")
    if return_original:
        return objects_raw  



def custom_describe(df):
    # Convert columns to numeric if possible
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    # Initial describe with 'include = all'
    desc = df.describe(include='all')

    # Desired statistics
    desired_stats = ['count', 'unique', 'mean', 'min', 'max']

    # Filter for desired statistics
    result = desc.loc[desired_stats, :].copy()
    return result

def compareStats(df_before, df_after):
  """
    Compares the descriptive statistics of two pandas DataFrames before and after some operations.

    Args:
        df_before (pd.DataFrame): DataFrame representing the state of data before operations.
        df_after (pd.DataFrame): DataFrame representing the state of data after operations.

    Returns:
        The CSV string includes column name, intervention type, and before and after statistics for each column.
        The summary string provides a count of updated and new columns.

    This function compares the descriptive statistics of two DataFrames: 'df_before' and 'df_after'. 
    It checks the columns in both DataFrames and categorizes them as either 'updated' or 'new'.
    The 'updated' columns exist in both DataFrames while the 'new' columns exist only in 'df_after'.
    For 'updated' columns, it compares the statistics before and after and notes the differences.
    For 'new' columns, it lists the 'after' statistics and marks the 'before' statistics as 'NA'.
    The function provides a summary with the number of updated and new columns, 
    and a detailed account in CSV format of changes in column statistics.
  """
   
  desc_before = custom_describe(df_before)
  desc_after = custom_describe(df_after)

  # Get union of all columns
  all_columns = set(desc_before.columns).union(set(desc_after.columns))

  # Track number of updated and new columns
  updated_cols = 0
  new_cols = 0

  # Prepare DataFrame output
  output_data = []

  for column in all_columns:
      row_data = {'column': column}
      stat_diff = False  # Track if there's a difference in stats for a column

      # Check if column exists in both dataframes
      if column in desc_before.columns and column in desc_after.columns:
          updated_cols += 1
          row_data['interventionType'] = 'updated'
          for stat in desc_before.index:
              before_val = round(desc_before.loc[stat, column], 1) if pd.api.types.is_number(desc_before.loc[stat, column]) else desc_before.loc[stat, column]
              after_val = round(desc_after.loc[stat, column], 1) if pd.api.types.is_number(desc_after.loc[stat, column]) else desc_after.loc[stat, column]
              if before_val != after_val:
                  stat_diff = True
                  row_data[stat+'_before'] = before_val
                  row_data[stat+'_after'] = after_val
      elif column in desc_after.columns:
          new_cols += 1
          stat_diff = True
          row_data['interventionType'] = 'new'
          for stat in desc_after.index:
              row_data[stat+'_before'] = 'NA'
              after_val = round(desc_after.loc[stat, column], 1) if pd.api.types.is_number(desc_after.loc[stat, column]) else desc_after.loc[stat, column]
              row_data[stat+'_after'] = after_val

      # Only add to output_data if there's actually a difference in the descriptive stats between "before" and "after".
      if stat_diff:
          output_data.append(row_data)

  output_df = pd.DataFrame(output_data)
  csv_output = output_df.to_csv(index=False)
  print (output_df)
  # Add summary to beginning of output
  summary = f"Summary:\n  Number of updated columns: {updated_cols}\n  Number of new columns: {new_cols}\n\n"
  csv_output = summary + csv_output

  return csv_output, summary



# Function to call ChatGPT API
def ask_chatgpt(prompt, model="gpt-3.5-turbo", max_tokens=300, n=1, stop=None, temperature=0.3):
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpfull assistant,."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature,
    )
    return response.choices[0].message['content']




def gptCommitMessage(objects_raw, new_data,openai_key):
    # the idea is to automatically create commit messages. Commits coming through this channel are all
    # about updating or adding a dataTable. So we can compare the descriptive stats of a before and after
    # data frame 
    #try:
    try:
        import openai
        openai.api_key = openai_key
    except NameError as ne:
        if str(ne) == "name 'openai' is not defined":
            print("No auto commit message: openai module not imported. Please import the module before setting the API key.")
        elif str(ne) == "name 'openai_key' is not defined":
            print("No auto commit message: openai_key is not defined. Please define the variable before setting the API key.")
        else:
            raise ne

    report, summary = compareStats(get_dataframe(objects_raw),new_data)

    # prompt
    prompt = f"""Given the following changes in my tabular data structure, generate a 
    precise and informative commit message. The changes involve updating or adding 
    attribute keys and values. The provided summary statistics detail the changes in 
    the data from 'before' to 'after'. 
    The CSV format below demonstrates the structure of the summary:

    Summary:
    Number of updated columns: 2
    Number of new columns: 1
    column,interventionType,count_before,count_after,unique_before,unique_after,mean_before,mean_after,min_before,min_after,max_before,max_after
    A,updated,800,800,2,3,,nan,nan,nan,nan,nan
    B,updated,800,800,3,3,,nan,nan,nan,nan,nan
    C,new,NA,800,NA,4,NA,nan,NA,nan,NA,nan

    For the commit message, your focus should be on changes in the data structure, not the interpretation of the content. Be precise, state the facts, and highlight significant differences or trends in the statistics, such as shifts in mean values or an increase in unique entries.

    Based on the above guidance, draft a commit message using the following actual summary statistics:

    {report}

    Your commit message should follow this structure:

    1. Brief description of the overall changes.
    2. Significant changes in summary statistics (count, unique, mean, min, max).
    3. Conclusion, summarizing the most important findings with the strucutre:
    # changed columns: , comment: ,
    # added Columns:  , comment: ,
    # Chaged statistic: ,  coment: ,

    Mark the beginning of the conclusion with ">>>" and ensure to emphasize hard facts and significant findings. 
    """

    try:
        answer = ask_chatgpt(prompt)
        answer_summery = answer.split(">>>")[1]
        if answer == None:
            answer_summery = summary
    except:
        answer_summery = summary

    print(answer_summery)
    return answer_summery

def specklePolyline_to_BokehPatches(speckle_objs, pth_to_geo="curves", id_key="ids"):
  """
  Takes a list of speckle objects, extracts the polyline geometry at the specified path, and returns a dataframe of x and y coordinates for each polyline.
  This format is compatible with the Bokeh Patches object for plotting.
  
  Args:
    speckle_objs (list): A list of Speckle Objects
    pth_to_geo (str): Path to the geometry in the Speckle Object
    id_key (str): The key to use for the uuid in the dataframe. Defaults to "uuid"
    
  Returns:
    pd.DataFrame: A Pandas DataFrame with columns "uuid", "patches_x" and "patches_y"
  """
  patchesDict = {"uuid":[], "patches_x":[], "patches_y":[]}
  
  for obj in speckle_objs:
    obj_geo = obj[pth_to_geo]
    obj_pts = Polyline.as_points(obj_geo)
    coorX = []
    coorY = []
    for pt in obj_pts:
      coorX.append(pt.x)
      coorY.append(pt.y)
    
    patchesDict["patches_x"].append(coorX)
    patchesDict["patches_y"].append(coorY)
    patchesDict["uuid"].append(obj[id_key])

  return pd.DataFrame(patchesDict)



def rebuildAnalysisInfoDict(analysisInfo):
    """rebuild the analysisInfo dictionary to remove the ++ from the keys

    Args:
        analysisInfo (list): a list containing the analysisInfo dictionary

    Returns:
        dict: a dictionary containing the analysisInfo dictionary with keys without the ++

    """
    analysisInfoDict = {}
    for curKey in analysisInfo[0]:
        newkey = curKey.split("++")[0]
        analysisInfoDict[newkey] = analysisInfo[0][curKey]
    return analysisInfoDict


def specklePolyline2Patches(speckle_objs, pth_to_geo="curves", id_key=None):
    """
    Converts Speckle objects' polyline information into a format suitable for Bokeh patches.

    Args:
        speckle_objs (list): A list of Speckle objects.
        pth_to_geo (str, optional): The path to the polyline geometric information in the Speckle objects. Defaults to "curves".
        id_key (str, optional): The key for object identification. Defaults to "uuid".

    Returns:
        DataFrame: A pandas DataFrame with three columns - "uuid", "patches_x", and "patches_y". Each row corresponds to a Speckle object.
                    "uuid" column contains the object's identifier.
                    "patches_x" and "patches_y" columns contain lists of x and y coordinates of the polyline points respectively.

    This function iterates over the given Speckle objects, retrieves the polyline geometric information and the object's id from each Speckle object, 
    and formats this information into a format suitable for Bokeh or matplotlib patches. The formatted information is stored in a dictionary with three lists 
    corresponding to the "uuid", "patches_x", and "patches_y", and this dictionary is then converted into a pandas DataFrame.
    """
    patchesDict = {"patches_x":[], "patches_y":[]}
    if id_key != None:
        patchesDict[id_key] = []

    for obj in speckle_objs:
        obj_geo = obj[pth_to_geo]
        
        coorX = []
        coorY = []
        
        if isinstance(obj_geo, Mesh):
            # For meshes, we'll just use the vertices for now
            for pt in obj_geo.vertices:
                coorX.append(pt.x)
                coorY.append(pt.y)
        else:
            # For polylines, we'll use the existing logic
            obj_pts = Polyline.as_points(obj_geo)
            for pt in obj_pts:
                coorX.append(pt.x)
                coorY.append(pt.y)

        patchesDict["patches_x"].append(coorX)
        patchesDict["patches_y"].append(coorY)
        if id_key != None:
            patchesDict[id_key].append(obj[id_key])

    return pd.DataFrame(patchesDict)


#================= NOTION INTEGRATION ============================
headers = {
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

def get_page_id(token, database_id, name):
    headers['Authorization'] = "Bearer " + token
    # Send a POST request to the Notion API
    response = requests.post(f"https://api.notion.com/v1/databases/{database_id}/query", headers=headers)

    # Load the response data
    data = json.loads(response.text)

    # Check each page in the results
    for page in data['results']:
        # If the name matches, return the ID
        if page['properties']['name']['title'][0]['text']['content'] == name:
            return page['id']

    # If no match was found, return None
    return None

def add_or_update_page(token, database_id, name, type, time_updated, comment, speckle_link):
    # Format time_updated as a string 'YYYY-MM-DD'
    date_string = time_updated.strftime('%Y-%m-%d')

    # Construct the data payload
    data = {
        'parent': {'database_id': database_id},
        'properties': {
            'name': {'title': [{'text': {'content': name}}]},
            'type': {'rich_text': [{'text': {'content': type}}]},
            'time_updated': {'date': {'start': date_string}},
            'comment': {'rich_text': [{'text': {'content': comment}}]},
            'speckle_link': {'rich_text': [{'text': {'content': speckle_link}}]}
        }
    }

    # Check if a page with this name already exists
    page_id = get_page_id(token, database_id, name)

    headers['Authorization'] = "Bearer " + token
    if page_id:
        # If the page exists, send a PATCH request to update it
        response = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=headers, data=json.dumps(data))
    else:
        # If the page doesn't exist, send a POST request to create it
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, data=json.dumps(data))
    
    print(response.text)

# Use the function
#add_or_update_page('your_token', 'your_database_id', 'New Title', 'New Type', datetime.now(), 'This is a comment', 'https://your-link.com')



def extractChunkedMatrices(streamObj):
    """
    Extracts chunked matrices from a Speckle base object and returns them as a dictionary of DataFrames.

    This function processes a Speckle base object, which contains nested matrices either as part of a list or
    as dynamic attributes of the object. It extracts the matrices, processes their chunks, and returns a 
    dictionary where the keys are the matrix names and the values are DataFrames representing the matrices.

    Parameters:
    streamObj (SpeckleBaseObject): The Speckle base object containing the chunked matrices. It is expected 
                                   that the matrices are nested under the "@Data" key.

    Returns:
    dict: A dictionary where the keys are matrix names (str) and the values are DataFrames containing the 
          matrix data. The DataFrame indices are origin UUIDs and columns are destination UUIDs.
    
    Raises:
    KeyError: If the specified keys are not found in the Speckle base object.
    AttributeError: If the object structure does not match the expected format.
    
    Notes:
    - The function first tries to access the nested matrices via the "@{0}" key in "@Data".
    - If the access fails, it assumes the data is in the dynamic attributes of the "@Data" object.
    - Matrices are identified by the presence of "matrix" in their attribute names.
    - Chunks of rows are processed and combined to form the final matrices.
    """

    matrices = {}
    isDict = False

    try:
        data_part = streamObj["@Data"]["@{0}"]
        for matrix in data_part:
            # Find the matrix name
            matrix_name = next((attr for attr in dir(matrix) if "matrix" in attr), None)

            if not matrix_name:
                continue

            matrix_data = getattr(matrix, matrix_name)
            originUUID = matrix_data["@originUUID"]
            destinationUUID = matrix_data["@destinationUUID"]

            processed_rows = []
            for chunk in matrix_data["@chunks"]:
                for row in chunk["@rows"]:
                    processed_rows.append(row["@row"])

            matrix_array = np.array(processed_rows)
            matrix_df = pd.DataFrame(matrix_array, index=originUUID, columns=destinationUUID)
            matrices[matrix_name] = matrix_df
    except KeyError:
        data_part = streamObj["@Data"].__dict__
        print(data_part.keys())

        for k, v in data_part.items():
            if "matrix" in k:
                matrix_name = k
                matrix_data = v
                originUUID = matrix_data["@originUUID"]
                destinationUUID = matrix_data["@destinationUUID"]

                processed_rows = []
                for chunk in matrix_data["@chunks"]:
                    for row in chunk["@rows"]:
                        processed_rows.append(row["@row"])

                matrix_array = np.array(processed_rows)
                matrix_df = pd.DataFrame(matrix_array, index=originUUID, columns=destinationUUID)
                matrices[matrix_name] = matrix_df

    return matrices

def extractChunkedMatrix(matrix):
    """
    Extracts a chunked matrix from a given Speckle matrix object and converts it into a pandas DataFrame.

    Parameters:
    matrix (dict): The Speckle matrix object containing chunked matrix data.

    Returns:
    pd.DataFrame: A pandas DataFrame constructed from the chunked matrix data in the matrix object.

    The function navigates through the matrix object to extract the rows from chunks, combines them,
    and constructs a DataFrame using origin and destination UUIDs as the index and columns, respectively.
    """
    # Get the origin and destination UUIDs
    originUUID = matrix["@originUUID"]
    destinationUUID = matrix["@destinationUUID"]
    
    # Process the rows from chunks
    processed_rows = []
    for chunk in matrix["@chunks"]:
        for row in chunk["@rows"]:
            processed_rows.append(row["@row"])
    
    # Convert the processed rows into a NumPy array
    matrix_arr = np.array(processed_rows)
    
    # Create a DataFrame with the originUUIDs as the index and destinationUUIDs as the columns
    matrix_df = pd.DataFrame(matrix_arr, index=originUUID, columns=destinationUUID)
    
    return matrix_df




@jit(nopython=True)
def round_matrix(matrix):
    return np.round(matrix, 4)

def send_matrices_and_create_commit_v(matrices, client, stream_id, branch_name, commit_message, rows_per_chunk, parameterData, MetaData={}):
    """
    Sends chunked matrices to Speckle and creates a commit.

    This function processes a dictionary of matrices, chunks their data, rounds them to 4 decimal places,
    and sends them to Speckle. It then creates a commit on the specified branch with the given commit message.

    Parameters:
    matrices (dict): A dictionary where keys are matrix names and values are DataFrames containing the matrices.
    client (SpeckleClient): The Speckle client object used to communicate with the Speckle server.
    stream_id (str): The ID of the Speckle stream where the matrices will be sent.
    branch_name (str): The name of the branch where the commit will be created.
    commit_message (str): The commit message.
    rows_per_chunk (int): The number of rows per chunk for splitting the matrix data.
    parameterData (dict): Additional parameter data to be included in the container object.
    MetaData (dict, optional): Metadata to be included in the commit. Defaults to an empty dictionary.

    Returns:
    str: The ID of the created commit.

    Raises:
    Exception: If an error occurs during the sending or committing process.
    """
    transport = ServerTransport(client=client, stream_id=stream_id)
    matrix_ids = {}

    container_object = Base()
    # Add parameter data to the container object
    for k, v in parameterData.items():
        setattr(container_object, k, v)

    # Initialize the keys in the container object
    for k in matrices.keys():
        setattr(container_object, k, Base())

    for k, df in matrices.items():
        matrix_object = Base(metaData="Some metadata")
        matrix = df.to_numpy()
        indices = list(df.index.to_numpy().astype(str))
        cols = list(df.columns)

        # Round the matrix using a vectorized operation
        rounded_matrix = round_matrix(matrix).tolist()

        # Chunk the rows and create Speckle objects
        chunks = []
        row_lists = []
        for row in rounded_matrix:
            row_obj = Base()
            setattr(row_obj, "@row", row)
            row_lists.append(row_obj)

            if len(row_lists) >= rows_per_chunk:
                chunk_obj = Base()
                setattr(chunk_obj, "@rows", row_lists)
                chunks.append(chunk_obj)
                row_lists = []

        if row_lists:
            chunk_obj = Base()
            setattr(chunk_obj, "@rows", row_lists)
            chunks.append(chunk_obj)

        obj = Base()
        setattr(obj, "@originUUID", indices)
        setattr(obj, "@destinationUUID", cols)
        setattr(obj, "@chunks", chunks)

        setattr(container_object, k, obj)

        print(container_object)

    try:
        # Recreate GH Speckle structure
        dataContainer = Base()
        setattr(dataContainer, "@Data", container_object)
        setattr(dataContainer, "@MetaData", json.dumps(MetaData))
        container_objectid = operations.send(base=dataContainer, transports=[transport])
        print(f"Container Object ID: {container_objectid}")
        commit_id = client.commit.create(stream_id=stream_id, object_id=container_objectid, branch_name=branch_name, message=commit_message)
        print(f"Commit ID: {commit_id}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            logging.error(f"Response JSON: {e.response.json()}")
        raise

    return commit_id



def generate_metadata(inputs, speckleToken, methodName=None):
    """
    Generates metadata by querying the Speckle API for given inputs.

    This function processes a list of inputs, which can be either URLs or lists, to extract stream, branch, 
    and commit information from the Speckle API. It returns metadata containing the extracted information 
    along with the method name.

    Parameters:
    inputs (list): A list of inputs where each input is either a URL (str) or a list containing stream and 
                   branch and commit(optional) information.
    speckleToken (str): The Speckle API token for authentication.
    methodName (str, optional): The name of the method generating this metadata. Defaults to the caller's file name.

    Returns:
    dict: A dictionary containing the generated metadata, including stream, branch, and commit information, 
          as well as the method name.

    Raises:
    ValueError: If the input format is incorrect or not supported.
    """
    def decode_url(input_url):
        parsed_url = urllib.parse.urlparse(input_url)
        return parsed_url.geturl()

    def send_graphql_query(speckleToken, apiUrl, query):
        headers = {
            "Authorization": f"Bearer {speckleToken}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        postData = json.dumps({"query": query})
        
        response = requests.post(apiUrl, headers=headers, data=postData)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"HTTP Error: {response.status_code}, {response.reason}")
            print(response.text)
            return None

    def parse_input(input_data, input_type='url'):
        if input_type == 'url':
            return parse_input_url(input_data)
        elif input_type == 'list':
            return parse_input_list(input_data)
        else:
            raise ValueError("Invalid input type. Must be 'url' or 'list'.")

    def parse_input_url(input_url):
        print("Original URL: ", input_url)
        decoded_url = decode_url(input_url)
        print("Decoded URL: ", decoded_url)
        parts = decoded_url.strip('/').split('/')
        if len(parts) < 7:
            raise ValueError("URL format is incorrect or not supported.")
        stream_id = parts[4]
        resource_type = parts[5]
        if resource_type.lower() == "branches":
            resource_id = '/'.join(parts[6:])
        else:
            resource_id = parts[6]

        return stream_id, resource_type, resource_id

    def parse_input_list(input_list):
        if len(input_list) < 2 or len(input_list) > 3:
            raise ValueError("List format is incorrect or not supported.")
        stream_id = input_list[0]
        resource_type = 'branches' if len(input_list) == 2 else 'commits'
        resource_id = input_list[1] if resource_type == 'branches' else input_list[2]
        return stream_id, resource_type, resource_id

    def construct_query(stream_id, resource_type, resource_id):
        if resource_type == "commits":
            return f"""
            {{
              stream(id: "{stream_id}") {{
                name
                commit(id: "{resource_id}") {{
                  id
                  message
                  createdAt
                  branch {{
                    id
                    name
                  }}
                }}
              }}
            }}
            """
        elif resource_type == "branches":
            return f"""
            {{
              stream(id: "{stream_id}") {{
                name
                branch(name: "{resource_id}") {{
                  id
                  name
                  commits {{
                    items {{
                      id
                      message
                      createdAt
                    }}
                  }}
                }}
              }}
            }}
            """

    def extract_information(data):
        if 'data' in data and 'stream' in data['data']:
            stream = data['data']['stream']
            stream_name = stream.get('name', 'Unknown Stream Name')
            stream_id = stream.get('id', 'Unknown Stream ID')
            if 'branch' in stream:
                branch_info = stream['branch']
                branch_name = branch_info.get('name', 'Unknown Branch Name')
                branch_id = branch_info.get('id', 'Unknown Branch ID')
                commits = branch_info.get('commits', {}).get('items', [])
                if commits:
                    last_commit = commits[0]
                    return {
                        "streamName": stream_name,
                        "branchName": branch_name,
                        "branchID": branch_id,
                        "commitID": last_commit['id'],
                        "commitMessage": last_commit['message'],
                        "commitCreatedAt": last_commit['createdAt']
                    }
            elif 'commit' in stream:
                commit_info = stream['commit']
                branch_info = commit_info.get('branch', {})
                return {
                    "streamName": stream_name,
                    "branchName": branch_info.get('name', 'Unknown Branch Name'),
                    "branchID": branch_info.get('id', 'Unknown Branch ID'),
                    "commitID": commit_info['id'],
                    "commitMessage": commit_info['message'],
                    "commitCreatedAt": commit_info['createdAt']
                }
        return "No valid data found"

    if methodName is None:
        methodName = inspect.getfile(sys._getframe(1))
    
    apiUrl = "https://speckle.xyz/graphql"
    speckleSources = []

    for input_data in inputs:
        input_type = 'url' if isinstance(input_data, str) else 'list'
        try:
            stream_id, resource_type, resource_id = parse_input(input_data, input_type)
            query = construct_query(stream_id, resource_type, resource_id)
            result = send_graphql_query(speckleToken, apiUrl, query)
            if result:
                extracted_info = extract_information(result)
                if isinstance(extracted_info, dict):
                    extracted_info["streamID"] = stream_id
                    speckleSources.append(extracted_info)
                else:
                    print("Failed to extract valid information.")
                    continue
            else:
                print("Failed to retrieve data.")
                continue
        except ValueError as e:
            print(str(e))
            continue

    metadata = {
        "speckleSources": speckleSources,
        "methodName": methodName
    }

    return metadata