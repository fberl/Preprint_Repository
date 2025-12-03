import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression

def drop_nan_columns(df, how='any', verbose=True):
    """
    Drop columns containing NaN values from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    how : str, default 'any'
        Determines if a column is removed when we have at least one NaN ('any') 
        or when all values are NaN ('all')
    verbose : bool, default True
        Whether to print information about dropped columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with NaN columns removed
    """
    original_columns = df.columns.tolist()
    original_shape = df.shape
    
    if verbose:
        # First, check for any NaN values in the entire DataFrame
        total_nan_count = df.isna().sum().sum()
        print(f"Total NaN values in DataFrame: {total_nan_count}")
        
        if total_nan_count > 0:
            print(f"NaN values by column:")
            nan_counts = df.isna().sum()
            for col, count in nan_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    print(f"  {col}: {count} NaN values ({percentage:.1f}%)")
        
        # Check for non-standard missing value representations
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                non_standard_missing = 0
                for val in ['', 'NaN', 'nan', 'NA', 'null', 'NULL', 'None']:
                    count = (df[col] == val).sum()
                    non_standard_missing += count
                if non_standard_missing > 0:
                    print(f"  {col}: {non_standard_missing} non-standard missing values")
    
    # Drop columns with NaN values
    df_cleaned = df.dropna(axis=1, how=how)
    
    if verbose:
        dropped_columns = set(original_columns) - set(df_cleaned.columns)
        if dropped_columns:
            print(f"Dropped {len(dropped_columns)} columns with NaN values: {sorted(dropped_columns)}")
            print(f"DataFrame shape: {original_shape} -> {df_cleaned.shape}")
        else:
            if total_nan_count > 0:
                print("Found NaN values, but no columns were dropped (they don't meet the 'how' criteria)")
            else:
                print("No NaN values found in DataFrame")
    
    return df_cleaned

def load_behavior(path, algorithm = 2, monkey = None, drop_nan_cols=False):
    if monkey == 'E' or monkey == 18:
        monkey = 18
    # Don't set monkey to None if it's already a number!
    df = pd.read_csv(path)
    if algorithm is not None:
        df = df[df['task'] == algorithm]
    if monkey is not None:
        df = df[df['animal'] == monkey]
    df['trial'] =  pd.Series(dtype='int')
    for sess in list(sorted(set(df['id']))):
        mask = df['id'] == sess
        df.loc[mask, 'trial'] = pd.Series(np.arange(mask.sum()) + 1, index=df.index[mask])
    
    # Optionally drop columns with NaN values
    if drop_nan_cols:
        df = drop_nan_columns(df)
            
    return df


def plot_monkey_learning():
    # plot first MP algo 1 for monkey E, then N episodes of monkey E after it swaps to algo 2 and include WR. 
    # Then last N episodes of monkey E and include WR. this plot is way easier than the other one...
    n = 5
    m = 5
    algo1 = load_behavior('/Users/fmb35/Desktop/MPbehdata.csv',algorithm=1)
    monkeyE_algo2_early = load_behavior('/Users/fmb35/Desktop/MPbehdata.csv',algorithm=2, monkey= 18)
    session_list = list(monkeyE_algo2_early['id'].unique())
    monkeyE_algo2_early = monkeyE_algo2_early[(monkeyE_algo2_early['id'] < session_list[m+n]) & (monkeyE_algo2_early['id'] >= session_list[n]) ] # first 10 sessions after 5 warmup sessions
    monkeyE_algo2_late = load_behavior('/Users/fmb35/Desktop/MPbehdata.csv',algorithm=2, monkey= 18)
    monkeyE_algo2_late = monkeyE_algo2_late[monkeyE_algo2_late['id'] >= session_list[-m]] # last 10 sessions
    
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle('Strategic Behavior in Monkeys Playing Matching Pennies')
    
    # compute winrates for each set of sessions
    win_rates = [algo1['reward'].mean(), monkeyE_algo2_early['reward'].mean(),
                 monkeyE_algo2_late['reward'].mean()]
    
    
    axs[0].set_title('All Monkeys vs. MP Algorithm 1\n Win Rate: {:.2f}'.format(win_rates[0]))
    axs[1].set_title('Monkey E vs. MP Algorithm 2 (Early Sessions)\n Win Rate: {:.2f}'.format(win_rates[1]))
    axs[2].set_title('Monkey E vs. MP Algorithm 2 (Late Sessions)\n Win Rate: {:.2f}'.format(win_rates[2]))
    
    paper_logistic_regression(axs[0],False,data=algo1, legend = True)
    paper_logistic_regression(axs[1],False,data=monkeyE_algo2_early)
    paper_logistic_regression(axs[2],False,data=monkeyE_algo2_late)

    axs[0].set_ylabel('Logistic Regression Coefficient')
    axs[1].set_xlabel('Trials Back')

    fig.tight_layout()
    
    plt.show()