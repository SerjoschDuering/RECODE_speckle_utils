import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cleanData(data, mode="drop", num_only=False):
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
  ax.set_title(columName, loc='left', color="lightgrey")

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
  ax.text(mean_val, ax.get_ylim()[1] * 0.98,"Mean", weight = "bold", size=30, ha="center", 
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))
  ax.text(mean_val, ax.get_ylim()[1] * 0.85, str(round(mean_val,1)) + " from " + str(max_val), ha='center', va='bottom', size=30,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))

  # Set the figure title and place it on the top left corner
  ax.set_title(columName, loc='left', color="lightgrey")

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


def create_colorbar(fig, ax, dataset, coloring_col, cmap, title="", cb_positioning = [0.9, 0.4, 0.02, 0.38]):
    divider = make_axes_locatable(ax)
    divider.append_axes("right", size="2%", pad=5.55)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=dataset[coloring_col].min(), vmax=dataset[coloring_col].max()))

    colorbar_ax = fig.add_axes(cb_positioning)
    colorbar = fig.colorbar(sm, cax=colorbar_ax)

    min_tick = dataset[coloring_col].min()
    max_tick = dataset[coloring_col].max()
    colorbar.set_ticks([min_tick*1.05, max_tick*0.95])
    colorbar.ax.set_yticklabels([str(round(min_tick,1)), str(round(max_tick,1))])
    colorbar.ax.tick_params(labelsize=44)
    

    colorbar.ax.annotate(title , xy=(0.5, 1.1), xycoords='axes fraction', fontsize=44,
                     xytext=(-50, 15), textcoords='offset points',
                     ha='left', va='bottom')

    for a in fig.axes:
        if a is not ax and a is not colorbar_ax:
            a.axis('off')

    return sm, colorbar



def draw_polygons(ax, dataset, x_cord_name, y_cord_name, style_dict, sm=None, drawing_order=None, cmap=None, coloring_col=None):
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
                    polygon = patches.Polygon(np.column_stack((patch_x_list, patch_y_list)), **style_dict, facecolor=cmap(sm.norm(row[coloring_col])))
                else:
                    polygon = patches.Polygon(np.column_stack((patch_x_list, patch_y_list)), **style_dict)

                ax.add_patch(polygon)
            except Exception as e:
               print(f"Error occurred: {e}")


def configure_plot(ax, all_x_coords, all_y_coords, buffer=0.05):
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
def createActivityNodePlot(dataset, colorbar_title="", color="coolwarm", data_col=None, cb_positioning = [0.9, 0.4, 0.02, 0.38]):
    
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
        sm, colorbar = create_colorbar(fig, ax, dataset, coloring_col, cmap, colorbar_title, cb_positioning = cb_positioning)
    drawing_order = get_drawing_order(dataset, [1, 3, 2], ['-', '+', '+'])

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