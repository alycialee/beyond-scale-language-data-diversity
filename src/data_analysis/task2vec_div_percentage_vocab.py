"""
ref: https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/75c02825-c539-4258-a12a-07056e4dba78
"""

import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For loading and handling CSV data
from pathlib import Path  # For handling filesystem paths
import numpy as np  # For numerical operations
from scipy.interpolate import UnivariateSpline

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    :param file_path: The Path object pointing to the CSV file.
    :return: A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file_path)

def wandb_df_to_list(df: pd.DataFrame, column_idx: int = 1) -> list:
    """ 
    Get data from wandb df and make it into a list.
    Basically gets the column of the df gets all the rows and makes it into a list.

    We default to 1 because the 1th (2nd) column contains the data but wandb puts funny names to the csv columns. 
    """
    rows: list = list(df.iloc[:, column_idx])
    return rows

def remove_indices_from_lists(list: list, indices: list) -> None:
    """
    Removes elements at the specified indices from list.
    The removal process accounts for the shift in indices as items are removed.

    :param lists: A list containing lists from which to remove elements.
    :param indices: A list of indices of elements to remove.
    """
    for index in sorted(indices, reverse=True):
        if index < len(list):  # Check if index is within the current length of the list
            del list[index]

def extract_metric_from_path(ci_file_path: Path) -> str:
    """
    Extracts the metric name from the CI file path.

    :param ci_file_path: Path object of the CI file.
    :return: The metric name extracted from the file path.
    """
    file_name = ci_file_path.name  # Get the file name from the path
    # Split the file name by '_' and extract the metric name (assuming it's always before 'wandb')
    metric = file_name.split('_')[1]  # This works given the naming convention used in the examples
    return metric

def main():
    # Set to True to plot a smoothed curve, False to plot the original data points
    smoothed: bool = True  

    # Base directory path
    base_dir = Path('/Users/brandomiranda/Documents/Research/beyond_scale_div_coeff_for_llms/beyond_scale_acts_vs_task2vec_vs_tokens').expanduser()
    
    # File paths
    # pwcca https://wandb.ai/brando/beyond-scale/runs/ig7x0kx8?workspace=user-brando
    # File paths for PWCCA analysis
    ci_file = base_dir / 'ci_pwcca_wandb_export_2024-02-01T12_31_52.166-08_00.csv'
    div_file = base_dir / 'div_pwcca_wandb_export_2024-02-01T12_32_43.412-08_00.csv'
    percentage_file = base_dir / 'percentage_pwcca_wandb_export_2024-02-01T12_32_09.641-08_00.csv'
    std_file = base_dir / 'std_pwcca_wandb_export_2024-02-01T12_32_28.025-08_00.csv'
    metric = "pwcca"

    # lincka https://wandb.ai/brando/beyond-scale/runs/qxdl5u8d?workspace=user-brando
    # File paths for LinCKA analysis
    # ci_file = base_dir / 'ci_lincka_wandb_export_2024-02-01T12_30_49.420-08_00.csv'
    # div_file = base_dir / 'div_lincka_wandb_export_2024-02-01T12_24_39.310-08_00.csv'
    # percentage_file = base_dir / 'percentage_lincka_wandb_export_2024-02-01T12_24_08.658-08_00.csv'
    # std_file = base_dir / 'std_lincka_wandb_export_2024-02-01T12_24_22.126-08_00.csv'  # Assuming you might need it later
    # metric = "lincka"

    # task2vec https://wandb.ai/brando/beyond-scale/runs/krup16om/workspace?workspace=user-brando 
    ci_file = base_dir / 'ci_task2vec_wandb_export_2024-02-01T11_25_22.529-08_00.csv'
    div_file = base_dir / 'div_task2vec_wandb_export_2024-02-01T11_26_06.630-08_00.csv'
    percentage_file = base_dir / 'percentage_task2vec_wandb_export_2024-02-01T11_25_48.409-08_00.csv'
    std_file = base_dir / 'std_task2vec_wandb_export_2024-02-01T11_26_26.772-08_00.csv' # Not used directly in the plot
    metric = "Task2Vec"

    # Load data
    ci_data = load_data(ci_file)
    div_data = load_data(div_file)
    percentage_data = load_data(percentage_file)
    
    # Get data as list
    ci_per_data_set: list = wandb_df_to_list(ci_data)
    avg_dists_per_data_set: list = wandb_df_to_list(div_data)
    percentages: list = wandb_df_to_list(percentage_data)
    print(f'{len(ci_per_data_set)=} {len(avg_dists_per_data_set)=} {len(percentages)=}')

    # # remove 0 and 41 42 43 and 81 82 83 84 and 91 inidices for all list
    # remove_indices_from_lists(ci_per_data_set, [41, 42, 43] + [81, 82, 83, 84] + [91] ) 
    # remove_indices_from_lists(avg_dists_per_data_set,[41, 42, 43] + [81, 82, 83, 84] + [91]) 
    # remove_indices_from_lists(percentages, [41, 42, 43] + [81, 82, 83, 84] + [91]) 
    # print(f'{len(ci_per_data_set)=} {len(avg_dists_per_data_set)=} {len(percentages)=}')

    # Plotting
    if not smoothed:
        plt.figure(figsize=(10, 6))
        plt.errorbar(percentages, avg_dists_per_data_set, yerr=ci_per_data_set, ecolor='gray', fmt='-o', capsize=5)
        plt.xlabel('Percentage of Vocabulary Used')
        plt.ylabel(f'Average {metric} Distance (Diversity)')
        plt.title(f'Average {metric} Distance (Diversity) vs. Vocabulary Usage Percentage')
        plt.grid(True)
        # Save the figure to the base directory
        figure_path = base_dir / f'div_acts_vs_task2vec_metric_{metric}_0p0_0p4_100_percentages_len_{len(percentages)}.png'
        plt.savefig(figure_path)
        
        # Display the plot
        plt.show()
    elif smoothed:
        # Convert percentages to numerical format if they're not already
        percentages_num = np.array(percentages, dtype=float)
        avg_dists_num = np.array(avg_dists_per_data_set, dtype=float)
        
        # Sorting data to ensure smooth spline fitting
        sorted_indices = np.argsort(percentages_num)
        sorted_percentages = percentages_num[sorted_indices]
        sorted_avg_dists = avg_dists_num[sorted_indices]

        # Fitting a spline
        spline = UnivariateSpline(sorted_percentages, sorted_avg_dists)
        spline.set_smoothing_factor(0.5)  # Adjust the smoothing factor to suit your data

        # Generating points on the curve
        spline_percentages = np.linspace(sorted_percentages.min(), sorted_percentages.max(), 200)
        spline_avg_dists = spline(spline_percentages)

        # Plotting the smooth curve
        plt.figure(figsize=(10, 6))
        plt.plot(spline_percentages, spline_avg_dists, label='Fitted Curve', color='red')
        plt.scatter(percentages, avg_dists_per_data_set, label='Original Data', color='blue')  # Original data points
        plt.xlabel('Percentage of Vocabulary Used')
        plt.ylabel(f'Average {metric} Distance (Diversity)')
        plt.title(f'Average {metric} Distance (Diversity) vs. Vocabulary Usage Percentage')
        plt.legend()
        plt.grid(True)

        # Save the figure to the base directory
        figure_path = base_dir / f'div_acts_vs_{metric}_smoothed_0p0_0p4_100_percentages_len_{len(percentages)}.png'
        plt.savefig(figure_path)
        
        # Display the plot
        plt.show()
    else:
        raise ValueError("Invalid value for smoothed. Please set it to either True or False.")

if __name__ == "__main__":
    main()

