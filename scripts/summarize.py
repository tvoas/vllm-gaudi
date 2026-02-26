import pandas as pd
import argparse

def get_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else pd.NA
get_mode.__name__ = 'mode'

def generate_stats(log_file, ignore_req_ids=False, analysis_types=None, merge_gaps=False):
    if analysis_types is None:
        analysis_types = ['mean', 'min', 'max']
        
    # Map string to function for mode, otherwise use the string directly
    agg_funcs = []
    for a in analysis_types:
        if a == 'mode':
            agg_funcs.append(get_mode)
        else:
            agg_funcs.append(a)

    # 1. Read the log file
    df = pd.read_csv(log_file)

    if ignore_req_ids and 'req_ids' in df.columns:
        df['req_ids'] = 'ALL'
    
    required_cols = {'req_ids', 'step_type', 's_time_s', 'd_time_s'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Log file must contain at least the following columns: {required_cols}")
        
    # 2. Process Gaps
    # Shift dataframe to align current row with the next row
    df_next = df.shift(-1)
    
    # Calculate gap: s_time_s + d_time_s - s_time_s'
    gap_values = df_next['s_time_s'] - (df['s_time_s'] + df['d_time_s'])
    
    if merge_gaps:
        # Add the gap time to the current item's duration (fill the last row's NaN with 0)
        df['d_time_s'] += gap_values.fillna(0)
    else:
        # Determine gap step_type (current_to_next)
        gap_step_types = df['step_type'].astype(str) + "_to_" + df_next['step_type'].astype(str)
        
        # Determine gap req_id (current if same as next, else 'transition')
        gap_req_ids = df['req_ids'].where(df['req_ids'] == df_next['req_ids'], 'transition')
        
        # Create gaps DataFrame (drop the last row as it has no 'next' item)
        gaps_df = pd.DataFrame({
            'req_ids': gap_req_ids.iloc[:-1],
            'step_type': gap_step_types.iloc[:-1],
            'd_time_s': gap_values.iloc[:-1]
        })
    
    # 3. Process Original Data Stats
    # Select numeric columns and exclude 's_time_s'
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if 's_time_s' in numeric_cols:
        numeric_cols.remove('s_time_s')
        
    # Group and aggregate original data
    orig_grouped = df.groupby(['req_ids', 'step_type'])[numeric_cols].agg(agg_funcs)
    
    # Stack the analysis types into a column
    orig_stacked = orig_grouped.stack(level=1, dropna=False).reset_index()
    orig_stacked.rename(columns={'level_2': 'analysis_type'}, inplace=True)
    
    # 4. Process Gaps Stats & Combine
    if merge_gaps:
        final_df = orig_stacked
    else:
        gaps_grouped = gaps_df.groupby(['req_ids', 'step_type'])[['d_time_s']].agg(agg_funcs)
        gaps_stacked = gaps_grouped.stack(level=1, dropna=False).reset_index()
        gaps_stacked.rename(columns={'level_2': 'analysis_type'}, inplace=True)
        
        # 5. Combine
        final_df = pd.concat([orig_stacked, gaps_stacked], ignore_index=True)
    
    # Normalize d_time_s to d_time_u
    shortest_d_time_s = final_df['d_time_s'].min()
    final_df['d_time_u'] = final_df['d_time_s'] / shortest_d_time_s
    
    # Sort for better readability and replace NaNs with empty strings for clean printing
    final_df = final_df.sort_values(by=['req_ids', 'step_type', 'analysis_type']).fillna('')
    
    # 6. Print as a formatted table
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log and print dense stats table.")
    parser.add_argument("log_file", help="Path to the CSV log file")
    parser.add_argument("--ignore-req-ids", action="store_true", help="Ignore req_ids and combine all stats")
    parser.add_argument("--merge-gaps", action="store_true", help="Add gap time to the d_time_s of the previous item instead of creating separate gap items")
    parser.add_argument("--analysis", nargs='+', choices=['mean', 'min', 'max', 'median', 'mode', 'std'], 
                        default=['mean', 'min', 'max'], help="Statistical analysis to perform (default: mean min max)")
    args = parser.parse_args()
    
    generate_stats(args.log_file, args.ignore_req_ids, args.analysis, args.merge_gaps)
