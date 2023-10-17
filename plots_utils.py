import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import r2_score

def cleanData(data, mode="drop", num_only=False):
    """
    This function cleans the input data based on the specified mode.

    Parameters:
    data (pd.DataFrame, pd.Series, or np.ndarray): The input data to be cleaned.
    mode (str, optional): The cleaning method, one of "drop", "replace_zero", or "replace_mean". 
                          "drop" removes NaN values, 
                          "replace_zero" replaces NaN values with zeros,
                          "replace_mean" replaces NaN values with the mean of the data.
                          Defaults to "drop".
    num_only (bool, optional): If True and data is a DataFrame, only integer and float columns are kept.
                               Defaults to False.

    Returns:
    data (same type as input): The cleaned data.

    The function works with pandas DataFrame, Series, and numpy array. Depending on the 'mode' argument, 
    it either drops the NaN values, replaces them with zero, or replaces them with the mean of the data. 
    If the data is a DataFrame and num_only is set to True, the function only keeps the columns with 
    numeric data (int64 and float64 dtypes).
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

def boxPlot(inp_data, columName, cull_invalid=True):
  """
    This function generates a boxplot for a given set of data.

    Parameters:
    inp_data (array or list): Input data for which the boxplot is to be created.
    columName (str): The name of the column which the data represents, to be used as title for the boxplot.
    cull_invalid (bool, optional): If True, invalid entries in the data are dropped. Defaults to True.
    
    Returns:
    fig (matplotlib Figure object): Figure containing the boxplot.
    ax (matplotlib Axes object): Axes of the created boxplot.

    The function creates a boxplot of the provided data, marking the 25th, 50th, and 75th percentiles. 
    The style of the boxplot is custom, with specific colors and properties for different boxplot elements. 
    The figure title is set to the provided column name.
    """
  if cull_invalid == True:
    inp_data = cleanData(inp_data, mode="drop", num_only=True)

  # Create a new figure
  fig, ax = plt.subplots(figsize=(10,3), dpi=200)

  # Set the style to white background
  sns.set_style("white")

  # Calculate the min, max, Q1, and Q3 of the data
  min_val = np.min(inp_data)
  max_val = np.max(inp_data)
  Q1 = np.percentile(inp_data, 25)
  Q3 = np.percentile(inp_data, 75)
  mean_val = np.mean(inp_data)

  # Define the positions and labels for the x ticks
  x_ticks = [] #[min_val, mean_val, Q3, max_val]
  x_tick_labels =[] #[ round(v,1) for v in x_ticks]

  # Add vertical lines at mean and Q3
  ax.vlines([mean_val], ymin=-0.35, ymax=0.35, colors='black', linewidth=3)
  ax.text(mean_val, -0.35, '  mean', ha='left', va='top',  fontsize=14)

  # Define the properties for the boxplot elements
  boxprops = {'edgecolor': 'black', 'linewidth': 2, 'facecolor': 'white', 'alpha':0.5}
  medianprops = {'color': 'gray', 'linewidth': 0}
  whiskerprops = {'color': 'black', 'linewidth': 1}
  capprops = {'color': 'black', 'linewidth': 2}
  flierprops = {'marker':'o', 'markersize':3, 'color':'white',  'markerfacecolor':'lightgray'}
  meanprops = {'color': 'black', 'linewidth': 1.0}
  kwargs = {'meanline': True, 'showmeans': True}

  # Create the boxplot
  bplot = sns.boxplot(x=inp_data, 
              boxprops=boxprops, 
              medianprops=medianprops, 
              whiskerprops=whiskerprops, 
              capprops=capprops,
              flierprops=flierprops,
              meanprops=meanprops,
              width=0.3,
              ax=ax,
              **kwargs
              )

  # Set the figure title and place it on the top left corner
  ax.set_title(columName, loc='left', color="lightgrey", alpha =0.2)

  # Remove the black outline from the figure
  for spine in ax.spines.values():
      spine.set_visible(False)

  # Set the x-axis ticks and labels
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_tick_labels)

  # Remove the x-axis label
  ax.set_xlabel('')
    
  return fig, ax



def boxPlot_colorbar(inp_data, columName, cull_invalid=True, color =  ['blue', 'red']):
  """
    This function creates a boxplot with an integrated colorbar for a given set of data. 

    Parameters:
    inp_data (array or list): Input data for which the boxplot is to be created.
    columName (str): The name of the column which the data represents, to be used as title for the boxplot.
    cull_invalid (bool, optional): If True, invalid entries in the data are dropped. Defaults to True.
    color (list of str, optional): List of colors to use for the gradient colorbar. Defaults to ['blue', 'red'].
    
    Returns:
    fig (matplotlib Figure object): Figure containing the boxplot.
    ax (matplotlib Axes object): Axes of the created boxplot.

    The function creates a boxplot of the provided data, marking the 25th, 50th, and 75th percentiles. 
    It also creates a horizontal colorbar above the boxplot that serves as a gradient from the minimum 
    to the maximum values of the data, emphasizing the data distribution.
    """
  if cull_invalid == True:
    inp_data = cleanData(inp_data, mode="drop", num_only=True)

  # Create a new figure
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(10,3), dpi=75,
        gridspec_kw={'height_ratios': [0.1, 1], 'hspace': 0.02}) # Adjust hspace for less space between plots


  # Set the style to white background
  sns.set_style("white")

  # Calculate the min, max, Q1, and Q3 of the data
  min_val = np.min(inp_data)
  max_val = np.max(inp_data)
  Q1 = np.percentile(inp_data, 25)
  Q3 = np.percentile(inp_data, 75)
  mean_val = np.mean(inp_data)

  ratio = int(np.ceil((Q3 - min_val) / (max_val - min_val) * 100))

  # Create a custom colormap
  cmap1 = LinearSegmentedColormap.from_list("mycmap", color)
  colors = np.concatenate((cmap1(np.linspace(0, 1, ratio)), np.repeat([cmap1(1.)], 100 - ratio, axis=0)))
  cmap2 = ListedColormap(colors)

  norm = Normalize(vmin=min_val, vmax=max_val)
  sm = ScalarMappable(norm=norm, cmap=cmap2)

  # Draw a vertical line at Q3
  cax.axvline(Q3*0.97, color='k', linewidth=3)
  cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[])

  # Define the positions and labels for the x ticks
  x_ticks = [] #[min_val, mean_val, Q3, max_val]
  x_tick_labels =[] #[ round(v,1) for v in x_ticks]

  # Add vertical lines at mean and Q3
  ax.vlines([Q3], ymin=-0.35, ymax=0.35, colors='black', linewidth=3)
  ax.text(Q3, 0.83, '  75th percentile', ha='left', va='top', transform=ax.get_xaxis_transform(), fontsize=14)


  # Define the properties for the boxplot elements
  boxprops = {'edgecolor': 'black', 'linewidth': 2, 'facecolor': 'white', 'alpha':0.5}
  medianprops = {'color': 'gray', 'linewidth': 0}
  whiskerprops = {'color': 'black', 'linewidth': 1}
  capprops = {'color': 'black', 'linewidth': 2}
  flierprops = {'marker':'o', 'markersize':3, 'color':'white',  'markerfacecolor':'lightgray'}
  meanprops = {'color': 'black', 'linewidth': 1.0}
  kwargs = {'meanline': True, 'showmeans': True}

  # Create the boxplo
  bplot = sns.boxplot(x=inp_data, 
              boxprops=boxprops, 
              medianprops=medianprops, 
              whiskerprops=whiskerprops, 
              capprops=capprops,
              flierprops=flierprops,
              meanprops=meanprops,
              width=0.3,
              ax=ax,
              **kwargs
              )

  # Set the figure title and place it on the top left corner
  ax.set_title(columName, loc='left', color="lightgrey", alpha=0.2)

  # Remove the black outline from the figure
  for spine in ax.spines.values():
      spine.set_visible(False)

  # Set the x-axis ticks and labels
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_tick_labels)

  # Remove the x-axis label
  ax.set_xlabel('')
    
  return fig, ax






def histogramScore(inp_data,columName, cull_invalid=True):
  # Create a new figure
  if cull_invalid:
    inp_data = cleanData(inp_data, mode="drop", num_only=True)

  fig, ax = plt.subplots()

  # Set the style to white background
  sns.set_style("white")

  # Create the histogram with an automatic number of bins
  ax.hist(inp_data, edgecolor='black', facecolor=(0.99,0.99,0.99,1), bins='auto')

  # Remove the black outline from the figure
  for spine in ax.spines.values():
      spine.set_visible(False)

  # Make the y-axis visible
  ax.spines['left'].set_visible(True)
  ax.spines['left'].set_color("lightgrey")
  ax.spines['bottom'].set_visible(True)
  ax.spines['bottom'].set_color("lightgrey")

  # Calculate the min, max, Q1, and Q3 of the data
  min_val = np.min(inp_data)
  max_val = np.max(inp_data)
  Q1 = np.percentile(inp_data, 25)
  Q3 = np.percentile(inp_data, 75)
  mean_val = np.mean(inp_data)



  # Calculate two equally spaced values on either side of the mean
  step = (mean_val - min_val) / 2
  xticks = [mean_val - 2*step, mean_val - step, mean_val, max_val]
  xticks = [ round(v,1) for v in xticks]
 
  ax.set_xticks(xticks)

  # Add a dotted line at the mean value
  ax.axvline(x=mean_val, ymax=0.85, linestyle='dotted', color='black')

  # Add a text tag at the end of the line  
  ax.text(mean_val, ax.get_ylim()[1] * 0.98,"Mean", weight = "bold", size=22, ha="center", 
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))
  ax.text(mean_val, ax.get_ylim()[1] * 0.85, str(round(mean_val,1)) + " from " + str(round(max_val,1)), ha='center', va='bottom', size=22,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))

  # Set the figure title and place it on the top left corner
  ax.set_title(columName, loc='left', color="lightgrey", alpha=0.3)

  # Make the y-axis tick labels smaller
  ax.tick_params(axis='y', labelsize=8)

  # Remove the x-axis label
  ax.set_xlabel('')


  return fig, ax


# =============================================================================
#==============================================================================


def get_drawing_order(dataset, order_of_importance, sorting_direction):
    # for activity nodes 
    temp_dataset = dataset.copy()
    temp_dataset[['id1', 'id2', 'id3']] = temp_dataset['ids'].str.split(';', expand=True).astype(int)
    columns_ordered = [f'id{i}' for i in order_of_importance]
    sorting_direction_ordered = [direction == '+' for direction in sorting_direction]
    drawing_order = temp_dataset.sort_values(columns_ordered, ascending=sorting_direction_ordered).index.tolist()
    return drawing_order


def calculate_aspect_ratio(all_x_coords, all_y_coords):
    x_range = max(all_x_coords) - min(all_x_coords)
    y_range = max(all_y_coords) - min(all_y_coords)
    aspect_ratio = y_range / x_range
    size = 15
    return (size, aspect_ratio) if aspect_ratio > 1 else (size / aspect_ratio, size)


def create_colorbar(fig, ax, dataset, coloring_col, cmap, title="", cb_positioning=[0.9, 0.4, 0.02, 0.38],
                     tick_unit="", normalize_override=("min", "max")):
    
    divider = make_axes_locatable(ax)
    divider.append_axes("right", size="2%", pad=5.55)

    # Determine normalization values
    if normalize_override[0] == "min":
        vmin = dataset[coloring_col].min()
    else:
        vmin = normalize_override[0]

    if normalize_override[1] == "max":
        vmax = dataset[coloring_col].max()
    else:
        vmax = normalize_override[1]

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    colorbar_ax = fig.add_axes(cb_positioning)
    colorbar = fig.colorbar(sm, cax=colorbar_ax)

    min_tick = vmin
    max_tick = vmax
    colorbar.set_ticks([min_tick*1.05, max_tick*0.95])
    colorbar.ax.set_yticklabels([
                                 str(round(min_tick,1))+" " +tick_unit, 
                                 str(round(max_tick,1)) + " " +tick_unit
                                 ])
    colorbar.ax.tick_params(labelsize=44)
    

    colorbar.ax.annotate(title , xy=(0.55, 1.1), xycoords='axes fraction', fontsize=44,
                     xytext=(-45, 15), textcoords='offset points',
                     ha='left', va='bottom')

    for a in fig.axes:
        if a is not ax and a is not colorbar_ax:
            a.axis('off')

    return sm, colorbar



def draw_polygons(ax, dataset, x_cord_name, y_cord_name, style_dict, sm=None, drawing_order=None, cmap=None, coloring_col=None):
    """
    This function draws polygons on a given axes object based on coordinates defined in the dataset.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object on which to draw the polygons.
    dataset (pd.DataFrame): The input DataFrame containing the coordinates of the polygons.
    x_cord_name (str): The name of the column in the dataset that contains the x-coordinates.
    y_cord_name (str): The name of the column in the dataset that contains the y-coordinates.
    style_dict (dict): A dictionary defining the style parameters for the polygons.
    sm (matplotlib.cm.ScalarMappable, optional): The scalar mappable object used for mapping normalized data to RGBA.
    drawing_order (list, optional): A list of indices defining the order in which to draw the polygons.
    cmap (matplotlib.colors.Colormap, optional): The colormap to use for coloring the polygons.
    coloring_col (str, optional): The name of the column in the dataset that contains the coloring values for the polygons.

    Returns:
    None

    The function reads the x and y coordinates from the dataset and creates a polygon for each row. 
    If a scalar mappable and a colormap are provided, the polygons are colored accordingly. 
    The order in which the polygons are drawn can be specified with the drawing_order parameter. 
    If no order is specified, the polygons are drawn in the order they appear in the dataset.
    """
    if drawing_order is None:
        drawing_order = dataset.index
    for idx in drawing_order:
        row  = dataset.loc[idx]

        # If it's a string, convert to list, if list, use directly
        if isinstance(row[x_cord_name], str) and len(row[x_cord_name]) > 2:
            patch_x_list = [float(i) for i in row[x_cord_name][1:-1].split(",")]
        elif isinstance(row[x_cord_name], list):
            patch_x_list = row[x_cord_name]

        if isinstance(row[y_cord_name], str) and len(row[y_cord_name]) > 2:
            patch_y_list = [float(i) for i in row[y_cord_name][1:-1].split(",")]
        elif isinstance(row[y_cord_name], list):
            patch_y_list = row[y_cord_name]

        # Check if the row is not None and the length is greater than 0
        if patch_x_list is not None and patch_y_list is not None and len(patch_x_list) > 0 and len(patch_y_list) > 0:
            try:
                if patch_x_list[0] != patch_x_list[-1] and patch_y_list[0] != patch_y_list[-1]:
                    patch_x_list.append(patch_x_list[0])
                    patch_y_list.append(patch_y_list[0])

                if sm is not None:
                    normalized_data = sm.norm(row[coloring_col])
                    polygon = patches.Polygon(np.column_stack((patch_x_list, patch_y_list)), **style_dict, facecolor=cmap(normalized_data))

                else:
                    polygon = patches.Polygon(np.column_stack((patch_x_list, patch_y_list)), **style_dict)

                ax.add_patch(polygon)
            except Exception as e:
               pass
               #print(f"Error occurred: {e}")


def configure_plot(ax, all_x_coords, all_y_coords, buffer=0.03):
    x_range = max(all_x_coords) - min(all_x_coords)
    y_range = max(all_y_coords) - min(all_y_coords)
    
    ax.set_aspect('equal')
    ax.set_xlim([min(all_x_coords) - buffer*x_range, max(all_x_coords) + buffer*x_range])
    ax.set_ylim([min(all_y_coords) - buffer*y_range, max(all_y_coords) + buffer*y_range])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# Main script
#dataset = dataset.dropna()

# column used for heatmap and colorbar
def createActivityNodePlot(dataset, 
                           colorbar_title="", 
                           color="coolwarm", 
                           data_col=None, 
                           cb_positioning = [0.9, 0.4, 0.02, 0.38], 
                           draw_oder_instruction=['-', '-', '+'],
                           tick_unit="",
                           normalize_override=("min", "max")):
    
    """
    This function creates an activity node plot using the provided dataset, and optionally includes a colorbar. 

    Parameters:
    dataset (pd.DataFrame): The input DataFrame containing the data.
    colorbar_title (str, optional): The title for the colorbar. Default is an empty string.
    color (str or list, optional): The colormap for the plot. Can be a matplotlib colormap name or a list of colors. Default is "coolwarm".
    data_col (str, optional): The name of the column in the dataset to use for coloring the nodes. If not provided, the first column of the dataset is used.
    cb_positioning (list, optional): A list of four floats defining the position and size of the colorbar. Defaults to [0.9, 0.4, 0.02, 0.38].
    draw_oder_instruction (list, optional): A list of strings defining the order in which to draw the polygons. Defaults to ['-', '-', '+'].
    tick_unit (str, optional): The unit for the ticks on the colorbar. Default is an empty string.

    Returns:
    fig (matplotlib.figure.Figure): The created figure object.
    ax (matplotlib.axes._subplots.AxesSubplot): The created Axes object.

    The function creates an activity node plot with optional coloring based on a data column. 
    The plot includes polygons representing nodes, and optionally a colorbar. 
    The order in which the nodes are drawn can be specified. 
    The plot's aspect ratio is calculated based on the provided coordinates.
    """
    
    if data_col == None:
        coloring_col = dataset.columns[0]
    else:
        coloring_col = data_col

    # not very elegant
    all_x_coords = []
    all_y_coords = []

    for idx, row in dataset.iterrows():
        # If it's a string, convert to list, if list, use directly
        if isinstance(row["patches_x_AN"], str) and len(row["patches_x_AN"]) > 2:
            patch_x_list = [float(i) for i in row["patches_x_AN"][1:-1].split(",")]
        elif isinstance(row["patches_x_AN"], list):
            patch_x_list = row["patches_x_AN"]

        if isinstance(row["patches_y_AN"], str) and len(row["patches_y_AN"]) > 2:
            patch_y_list = [float(i) for i in row["patches_y_AN"][1:-1].split(",")]
        elif isinstance(row["patches_y_AN"], list):
            patch_y_list = row["patches_y_AN"]
        all_x_coords.extend(patch_x_list)
        all_y_coords.extend(patch_y_list)
    
    figsize = calculate_aspect_ratio(all_x_coords, all_y_coords)
    fig, ax = plt.subplots(figsize=figsize)

    # color map
    if type(color) == type([]):
        
        cmap = LinearSegmentedColormap.from_list('custom_color', color)
    else:
        cmap = plt.cm.get_cmap(color)

    # Activity Node geometry
    style_dict_an = {'linewidth': 1, 'edgecolor': "Black"} 

    color_data_exists = is_numeric_dtype(dataset[coloring_col])

    if color_data_exists:
        sm, colorbar = create_colorbar(fig, ax, dataset, coloring_col, cmap, colorbar_title, 
                                       cb_positioning = cb_positioning, tick_unit=tick_unit,
                                       normalize_override=normalize_override)
    drawing_order = get_drawing_order(dataset, [1, 3, 2], draw_oder_instruction)

    draw_polygons(ax, 
                dataset, 
                "patches_x_AN", 
                "patches_y_AN", 
                style_dict_an, 
                sm,
                drawing_order,
                cmap,
                coloring_col)

    style_dict_bridges = {'linewidth': 1, 'edgecolor': "Black", 'facecolor':"Black"} 


    draw_polygons(ax, 
                dataset, 
                "patches_x_Bridges", 
                "patches_y_Bridges", 
                style_dict_bridges,
                cmap,
                coloring_col=coloring_col,
                )

    configure_plot(ax, all_x_coords, all_y_coords)
    return fig, ax



    
def radar(values_norm,
          labels,  
          color, 
          cluster_name, 
          factor=100, 
          ax_multi = None, 
          fig_multi=None, 
          label_font_size =6,
          num_datapoints=None):
        
    """
    This function creates a radar chart (also known as a spider or star chart) from given normalized values and labels.

    Parameters:
    values_norm (list of numbers): Normalized values to plot on the radar chart, these values will be scaled within the function.
    labels (list of str): Labels for the axes of the radar chart.
    color (str): Color of the fill and outline on the radar chart.
    cluster_name (str): Title for the radar chart.
    factor (int, optional): Scaling factor for the data, defaults to 100.
    ax_multi (matplotlib Axes object, optional): Predefined matplotlib Axes. If None, a new Axes object is created.
    fig_multi (matplotlib Figure object, optional): Predefined matplotlib Figure for the plot. If None, a new Figure is created.
    label_font_size (int, optional): Font size for the axis labels, defaults to 6.
    num_datapoints (int, optional): Number of datapoints used to calculate the values, will be displayed in the plot if provided.

    Returns:
    fig (matplotlib Figure object): Figure containing the radar chart.
    ax (matplotlib Axes object): Axes of the created radar chart.

    This function plots each value from 'values_norm' as an axis on the radar chart, 
    the aesthetics of the plot such as color and font size are customizable. The chart 
    is scaled using the provided factor. 'values_norm' should be preprocessed outside 
    of this function: they should be the mean values of your original data, normalized 
    to be between 0 and 1.
    """

    # ax = plt.subplot(polar=True)
    if ax_multi == None or fig_multi == None:
      fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True), dpi=200)
    else:
      fig = fig_multi
      ax = ax_multi

    values_norm = [v*factor for v in values_norm]

    # Number of variables we're plotting.
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    values_norm += values_norm[:1]
    angles += angles[:1]

    # Draw the outline of our data.
    ax.plot(angles, values_norm, color=color, linewidth=2)

    # Fill it in.
    ax.fill(angles, values_norm, color=color, alpha=0.15)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    labels += labels[:1]
    ax.set_thetagrids(np.degrees(angles), labels)
   
    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_fontsize(label_font_size)

    # Ensure radar goes from 0 to 100.
    ax.set_ylim(0, 100)

    # of the first two axes.
    ax.set_rlabel_position(180 / num_vars)

    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')

    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=6)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#222222')
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FAFAFA')

    # Lastly, give the chart a title and give it some
    # padding above the "Acceleration" label.
    ax.set_title(cluster_name, y=1.11)

     # Add this at the end of your function
    if num_datapoints is not None:
        # plt.figtext adds text to the figure as a whole, outside individual subplots
        # The parameters are (x, y, text), where x and y are in figure coordinates
        plt.figtext(0.5, -0.05, f'datapoints: {num_datapoints}', ha='center')

    return fig, ax


def gh_color_blueRed():
    # grasshoper color scheme 
    color_list = [[15,16,115],
            [177,198,242],
            [251,244,121],
            [222,140,61],
            [183,60,34]]
    # Scale RGB values to [0,1] range
    color_list = [[c/255. for c in color] for color in color_list]
    return color_list


def linear_regression_with_residuals(
    df, x_name, y_name, buffer=5, data_range_max=None, max_residual_color=None, rescale_range=None, generateName=False
    ):

    """
    Generate a scatter plot with linear regression, residuals, and a color-coded line of equality.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    x_name (str): The name of the x-axis variable.
    y_name (str): The name of the y-axis variable.
    buffer (int, optional): Buffer as a percentage of data range for plot margins. Default is 5.
    data_range_max (float, optional): Maximum value for x and y axes. Default is None (auto-calculated).
    max_residual_color (float, optional): Maximum residual value for color normalization. Default is None (auto-calculated).
    rescale_range (tuple, optional): Rescale both x and y to the specified range. Default is None (no rescaling).
    save_png (str, optional): File path to save the plot as a PNG image. Default is None (no saving).
    date_source (str, optional): Date source identifier for the filename. Default is None.

    Returns:
    plt: Matplotlib figure for the generated plot.
    """

    # Extract x and y values from the DataFrame
    x = df[x_name].values
    y = df[y_name].values

    # Rescale x and y if rescale_range is provided
    if rescale_range:
        x_min, x_max = rescale_range
        x = (x - min(x)) / (max(x) - min(x)) * (x_max - x_min) + x_min
        y = (y - min(y)) / (max(y) - min(y)) * (x_max - x_min) + x_min

    # Calculate R2 score
    r2 = r2_score(x, y)
    print(f"R2 Score: {r2}")

    # Calculate residuals in relation to the 45-degree line
    residuals_45 = y - x.flatten()

    # Calculate the data range with a buffer
    if data_range_max:
        data_min = 0
        data_max = data_range_max
    else:
        data_min = min(min(x), min(y))
        data_max = max(max(x), max(y))
    buffer_value = (data_max - data_min) * (buffer / 100)

    # Create a square plot with the same range for both axes
    plt.figure()
    colormap = 'bwr'  # Choose a colormap
    cmap = plt.get_cmap(colormap)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Shift the midpoint of the colormap to zero
    if max_residual_color is None:
        max_residual_color = max(abs(residuals_45))
    norm = plt.Normalize(-max_residual_color, max_residual_color)

    colors = np.array(cmap(norm(residuals_45)), dtype=object)

    # Darken the edge color by making it 90% darker than the fill color
    edge_colors = [tuple(0.9 * np.array(c)) for c in colors]

    # Add a contour to scatter points with the same color as the point fill
    scatter = plt.scatter(x, y, c=colors, label='True values', edgecolors=edge_colors, linewidths=2, zorder=3)

    # Plot the line of equality (x == y)
    combined_line = plt.plot([data_min - buffer_value, data_max + buffer_value], [data_min - buffer_value, data_max + buffer_value],
             color='black', linewidth=1, zorder=5)

    # Calculate and plot residuals in relation to the line of equality
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], x[i]], color='gray', linestyle='--', linewidth=0.5, zorder=1)

    # Plot the linear regression line
    m, b = np.polyfit(x, y, 1)
    regression_line = plt.plot(x, m * x + b, color='grey', linestyle='dotted', linewidth=1, label='Linear Regression line', zorder=4)

    # Calculate the R2 score text position
    text_x = data_min + 0.01 * (data_max - data_min)
    text_y = data_max - 0.01 * (data_max - data_min)

    # Annotate the plot with the R2 score
    plt.text(text_x, text_y, f'$R^2$ Score: {r2:.2f}', fontsize=8, color='black')

    # Add colorbar for residuals (smaller and within the plot)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.2, aspect=15, pad=0.03)
    cbar.set_label('Residuals (line of Equality)', fontsize=8)

    # Create separate legend handles and labels
    legend_handles = [scatter, regression_line[0], combined_line[0]]
    legend_labels = ['True values', 'Linear Regression line', 'Line of Equality']

    # Create the combined legend
    combined_legend = plt.legend(handles=legend_handles, labels=legend_labels, loc='lower right', fontsize=8)

    # Set the same limits for both x and y axes with a buffer
    plt.xlim(data_min - buffer_value, data_max + buffer_value)
    plt.ylim(data_min - buffer_value, data_max + buffer_value)

    plt.gca().add_artist(combined_legend)  # Add the combined legend to the plot

    plt.title('Linear Regression Visualization with Residuals (line of Equality)')
    plt.xlabel(" ".join(x_name.split("+"))[0].capitalize() + " ".join(x_name.split("+"))[1:])
    plt.ylabel(" ".join(y_name.split("+"))[0].capitalize() + " ".join(y_name.split("+"))[1:]

    # Add very light grey background grid lines
    plt.grid(True, color='lightgrey', linestyle='--', alpha=0.6, zorder=0)


    if generateName:
        # Plot name
        plt_name = "linearRegr_" + "".join(word.capitalize() for word in x_name.split("+")) + "_vs_" + "".join(
            word.capitalize() for word in y_name.split("+"))
        return plt, plt_name
    else:
        return plt
