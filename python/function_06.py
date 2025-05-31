import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


def load_data_with_delimiters(file_info):
    """
    Loads multiple CSV files with possibly different delimiters.

    Parameters:
        file_info (list of tuples): Each tuple is (file_path, delimiter)

    Returns:
        list: List of pandas DataFrames loaded using specified delimiters.
    """
    return [pd.read_csv(path, delimiter=delim) for path, delim in file_info]


def show_heads(dfs, n=5):
    """
    Prints the first n rows of each DataFrame in a list.
    
    Parameters:
        dfs (list): List of pandas DataFrames.
        n (int): Number of rows to show from each DataFrame.
    """
    for i, df in enumerate(dfs, start=1):
        print(f"\n--- Head of DataFrame {i} ---")
        print(df.head(n))


