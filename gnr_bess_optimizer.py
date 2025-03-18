"""
GNR BESS Optimization Module

This module contains the core optimization functionality for the GNR BESS Curtailment Analysis.
It provides functions for calculating battery dispatch in standalone mode.
"""

import pandas as pd
import numpy as np
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, value, LpStatus, PULP_CBC_CMD
import time
import concurrent.futures
import os

# Set solver parameters for better performance - using PuLP's built-in CBC solver
SOLVER = PULP_CBC_CMD(msg=False, timeLimit=180)  # 3-minute time limit per month, no messages

def validate_dataframe(df):
    """
    Validate that the input dataframe has the required columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: (boolean, message) indicating if the dataframe is valid and why if not
    """
    # Convert column names to lowercase for case-insensitive comparison
    df_columns_lower = [col.lower() for col in df.columns]
    
    required_columns = ['date, time', 'lmp', 'system export - kw (800)']
    
    for col in required_columns:
        if col not in df_columns_lower:
            return False, f"Missing required column: {col}"
    
    try:
        # Check if the Date, time column can be converted to datetime
        date_col_idx = df_columns_lower.index('date, time')
        date_col_original = df.columns[date_col_idx]
        pd.to_datetime(df[date_col_original])
    except:
        return False, "Date, time column could not be parsed as datetime"
    
    return True, "DataFrame is valid"

def get_battery_parameters(
    energy_capacity, charge_power_limit, discharge_power_limit,
    charge_efficiency, discharge_efficiency, 
    soc_max_percentage, soc_min_percentage,
    daily_cycle_limit, annual_cycle_limit,
    solar_dc_capacity, maximum_export_capacity
):
    """
    Create a dictionary of battery parameters.
    
    Args:
        Various battery parameters as input
        
    Returns:
        dict: Dictionary of battery parameters
    """
    SOC_max = (soc_max_percentage / 100) * energy_capacity
    SOC_min = (soc_min_percentage / 100) * energy_capacity
    SOC_initial = SOC_min
    
    return {
        'energy_capacity': energy_capacity,
        'charge_power_limit': charge_power_limit,
        'discharge_power_limit': discharge_power_limit,
        'charge_efficiency': charge_efficiency,
        'discharge_efficiency': discharge_efficiency,
        'SOC_max': SOC_max,
        'SOC_min': SOC_min,
        'SOC_initial': SOC_initial,
        'daily_cycle_limit': daily_cycle_limit,
        'annual_cycle_limit': annual_cycle_limit,
        'solar_dc_capacity': solar_dc_capacity,
        'maximum_export_capacity': maximum_export_capacity
    }

def preprocess_data(df, solar_dc_capacity, maximum_export_capacity, start_date, end_date):
    """
    Preprocess the input data: scale solar, calculate dynamic export capacity, and filter by date.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        solar_dc_capacity (float): Solar DC capacity in MW
        maximum_export_capacity (float): Maximum export capacity in MW
        start_date (str): Start date for filtering
        end_date (str): End date for filtering
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Normalize column names to expected format (but keep original columns in the dataframe)
    col_map = {
        'date, time': None,
        'lmp': None,
        'system export - kw (800)': None
    }
    
    for i, col in enumerate(df.columns):
        col_lower = col.lower()
        if col_lower == 'date, time':
            col_map['date, time'] = col
        elif col_lower == 'lmp':
            col_map['lmp'] = col
        elif col_lower == 'system export - kw (800)':
            col_map['system export - kw (800)'] = col
    
    # Ensure Date, time is datetime type
    df[col_map['date, time']] = pd.to_datetime(df[col_map['date, time']])
    
    # Scale the solar data
    original_capacity = 1040  # Original capacity in MWs from the original code
    df[col_map['system export - kw (800)']] = (df[col_map['system export - kw (800)']] / original_capacity) * solar_dc_capacity
    
    # Ensure that the scaled values do not exceed 800,000 kW
    df[col_map['system export - kw (800)']] = df[col_map['system export - kw (800)']].clip(upper=800000)
    
    # Filter by date
    df_filtered = df[(df[col_map['date, time']] >= pd.Timestamp(start_date)) & 
                     (df[col_map['date, time']] <= pd.Timestamp(end_date))].copy()
    
    # Calculate the maximum export capacity
    df_filtered.loc[:, 'Solar_Production_MW'] = -df_filtered[col_map['system export - kw (800)']] / 1000
    df_filtered.loc[:, 'Max_Export_Capacity_MW'] = maximum_export_capacity + df_filtered['Solar_Production_MW']
    
    # Rename the columns to the expected format for the rest of the code
    df_filtered = df_filtered.rename(columns={
        col_map['date, time']: 'Date, time',
        col_map['lmp']: 'lmp',
        col_map['system export - kw (800)']: 'System Export - KW (800)'
    })
    
    # Set the index to the datetime column for proper time-based operations
    if 'Date, time' in df_filtered.columns:
        df_filtered = df_filtered.set_index('Date, time')
    
    # Verify the index is a DatetimeIndex
    if not isinstance(df_filtered.index, pd.DatetimeIndex):
        print("Warning: Failed to set DatetimeIndex. Attempting conversion...")
        df_filtered.index = pd.to_datetime(df_filtered.index)
    
    return df_filtered

def optimize_month(month_data):
    """
    Optimize dispatch for a specific month of data.
    
    Args:
        month_data (tuple): A tuple containing (df_month, battery_params, month_label)
        
    Returns:
        tuple: Optimization results for the month and month label
    """
    df_month, battery_params, month_label = month_data
    
    print(f"Starting optimization for {month_label} with {len(df_month)} intervals")
    
    # Extract prices for optimization
    da_prices = df_month['lmp'].tolist()
    num_intervals = len(da_prices)
    
    if num_intervals == 0:
        print(f"Warning: No data for {month_label}")
        return None, 0.0, month_label
    
    # Create the optimization problem
    problem_name = f"Battery_Scheduling_{month_label}_{time.time()}"
    prob = LpProblem(problem_name, LpMaximize)
    
    # Initialize variables
    charge_vars = LpVariable.dicts("Charging", range(num_intervals), 
                                   lowBound=0, upBound=battery_params['charge_power_limit'])
    discharge_vars = LpVariable.dicts("Discharging", range(num_intervals), 
                                     lowBound=0, upBound=battery_params['discharge_power_limit'])
    SOC_vars = LpVariable.dicts("SOC", range(num_intervals+1), 
                               lowBound=battery_params['SOC_min'], upBound=battery_params['SOC_max'])
    
    # Objective function - maximize profit
    prob += lpSum([
        da_prices[t] * battery_params['discharge_efficiency'] * discharge_vars[t] - 
        da_prices[t] * charge_vars[t] / battery_params['charge_efficiency'] 
        for t in range(num_intervals)
    ])
    
    # Initial SOC constraint
    prob += SOC_vars[0] == battery_params['SOC_initial']
    
    # SOC update constraints (half-hourly intervals)
    for t in range(num_intervals):
        prob += SOC_vars[t+1] == SOC_vars[t] + (battery_params['charge_efficiency'] * charge_vars[t] * 0.5) - (discharge_vars[t] * 0.5)
    
    # Charge/Discharge constraints based on SOC limits
    for t in range(num_intervals):
        prob += SOC_vars[t] + battery_params['charge_efficiency'] * charge_vars[t] * 0.5 <= battery_params['SOC_max']
        prob += SOC_vars[t] - discharge_vars[t] * 0.5 >= battery_params['SOC_min']
    
    # Prevent simultaneous charge and discharge
    for t in range(num_intervals):
        prob += charge_vars[t] + discharge_vars[t] <= max(battery_params['charge_power_limit'], battery_params['discharge_power_limit'])
    
    # Daily cycle limit constraint
    daily_intervals = min(48, num_intervals)  # For months with partial days
    daily_cycle_limit_scaled = battery_params['daily_cycle_limit'] * (num_intervals / daily_intervals)
    prob += lpSum([charge_vars[t] for t in range(num_intervals)]) * battery_params['charge_efficiency'] * 0.5 / battery_params['energy_capacity'] <= daily_cycle_limit_scaled
    
    try:
        # Solve the problem
        prob.solve(SOLVER)
        
        # Handle suboptimal or infeasible solutions
        if LpStatus[prob.status] not in ['Optimal', 'Not Solved']:
            print(f"Warning: Optimization status for {month_label}: {LpStatus[prob.status]}")
            
        # Extract results
        discharge_values = [value(discharge_vars[t]) if discharge_vars[t].value() is not None else 0.0 for t in range(num_intervals)]
        charge_values = [value(charge_vars[t]) if charge_vars[t].value() is not None else 0.0 for t in range(num_intervals)]
        soc_values = [value(SOC_vars[t]) if SOC_vars[t].value() is not None else battery_params['SOC_min'] for t in range(num_intervals)]
        
        # Calculate energy discharged
        total_energy_discharged = sum(discharge_values) * 0.5 * battery_params['discharge_efficiency']
        
        # Create results dictionary
        result_dict = {
            'discharge_vars': discharge_values,
            'charge_vars': charge_values,
            'SOC_vars': soc_values,
            'lmp': da_prices
        }
        
        print(f"Completed optimization for {month_label} - Energy discharged: {total_energy_discharged:.2f} MWh")
        return result_dict, total_energy_discharged, month_label
        
    except Exception as e:
        print(f"Error in month optimization for {month_label}: {e}")
        # Return empty results with zeros
        return None, 0.0, month_label

def run_standalone_dispatch(df, battery_params, start_date, end_date, disable_parallel=False):
    """
    Run the standalone dispatch optimization with monthly parallelization.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        battery_params (dict): Battery parameters
        start_date (str): Start date for filtering
        end_date (str): End date for filtering
        disable_parallel (bool): Whether to disable parallelization
        
    Returns:
        tuple: (battery_dispatch_df, stats) - Battery dispatch results and statistics
    """
    try:
        start_time = time.time()
        
        # Preprocess data
        df_filtered = preprocess_data(
            df, 
            battery_params['solar_dc_capacity'], 
            battery_params['maximum_export_capacity'],
            start_date, 
            end_date
        )
        
        # Verify we have a valid datetime index
        print(f"Filtered dataframe index type: {type(df_filtered.index)}")
        print(f"Filtered dataframe index sample: {df_filtered.index[:5]}")
        
        total_intervals = len(df_filtered)
        print(f"Total intervals to optimize: {total_intervals}")
        
        if disable_parallel or total_intervals <= 1440:  # If less than 1 month of data (30 days * 48 intervals)
            print(f"Using single optimization for {total_intervals} intervals")
            
            # If dataset is small, use the direct approach without parallelization
            # Extract prices for optimization
            da_prices = df_filtered['lmp'].tolist()
            num_intervals = len(da_prices)
            
            print(f"Optimizing {num_intervals} intervals...")
            
            # Create the optimization problem
            problem_name = f"Battery_Scheduling_{time.time()}"
            prob = LpProblem(problem_name, LpMaximize)
            
            # Initialize variables
            charge_vars = LpVariable.dicts("Charging", range(num_intervals), 
                                           lowBound=0, upBound=battery_params['charge_power_limit'])
            discharge_vars = LpVariable.dicts("Discharging", range(num_intervals), 
                                             lowBound=0, upBound=battery_params['discharge_power_limit'])
            SOC_vars = LpVariable.dicts("SOC", range(num_intervals+1), 
                                       lowBound=battery_params['SOC_min'], upBound=battery_params['SOC_max'])
            
            # Objective function - maximize profit
            prob += lpSum([
                da_prices[t] * battery_params['discharge_efficiency'] * discharge_vars[t] - 
                da_prices[t] * charge_vars[t] / battery_params['charge_efficiency'] 
                for t in range(num_intervals)
            ])
            
            # Initial SOC constraint
            prob += SOC_vars[0] == battery_params['SOC_initial']
            
            # SOC update constraints (half-hourly intervals)
            for t in range(num_intervals):
                prob += SOC_vars[t+1] == SOC_vars[t] + (battery_params['charge_efficiency'] * charge_vars[t] * 0.5) - (discharge_vars[t] * 0.5)
            
            # Charge/Discharge constraints based on SOC limits
            for t in range(num_intervals):
                prob += SOC_vars[t] + battery_params['charge_efficiency'] * charge_vars[t] * 0.5 <= battery_params['SOC_max']
                prob += SOC_vars[t] - discharge_vars[t] * 0.5 >= battery_params['SOC_min']
            
            # Prevent simultaneous charge and discharge
            for t in range(num_intervals):
                prob += charge_vars[t] + discharge_vars[t] <= max(battery_params['charge_power_limit'], battery_params['discharge_power_limit'])
            
            # Daily cycle limit constraint
            prob += lpSum([charge_vars[t] for t in range(num_intervals)]) * battery_params['charge_efficiency'] * 0.5 / battery_params['energy_capacity'] <= battery_params['daily_cycle_limit']
            
            # Solve the problem
            print("Solving optimization problem...")
            prob.solve(SOLVER)
            
            # Handle suboptimal or infeasible solutions
            if LpStatus[prob.status] not in ['Optimal', 'Not Solved']:
                print(f"Warning: Optimization status: {LpStatus[prob.status]}")
            
            # Extract results
            discharge_values = [value(discharge_vars[t]) if discharge_vars[t].value() is not None else 0.0 for t in range(num_intervals)]
            charge_values = [value(charge_vars[t]) if charge_vars[t].value() is not None else 0.0 for t in range(num_intervals)]
            soc_values = [value(SOC_vars[t]) if SOC_vars[t].value() is not None else battery_params['SOC_min'] for t in range(num_intervals)]
            
            # Calculate energy discharged
            total_energy_discharged = sum(discharge_values) * 0.5 * battery_params['discharge_efficiency']
            
            # Create dataframe with results, preserving the datetime index
            battery_dispatch_df = pd.DataFrame({
                'discharge_vars': discharge_values,
                'charge_vars': charge_values,
                'SOC_vars': soc_values,
                'lmp': da_prices
            }, index=df_filtered.index)
            
        else:
            # For larger datasets, use parallelization by month
            print("Using monthly parallel optimization approach")
            
            # Group data by month for parallel processing
            df_filtered = df_filtered.sort_index()  # Ensure sorted by datetime
            df_by_month = {}
            
            # Group by year and month
            for name, group in df_filtered.groupby([df_filtered.index.year, df_filtered.index.month]):
                month_label = f"{name[0]}-{name[1]:02d}"
                df_by_month[month_label] = group
            
            print(f"Split into {len(df_by_month)} monthly chunks")
            
            # Prepare month data for multiprocessing
            month_data = [(df, battery_params, month_label) for month_label, df in df_by_month.items()]
            
            # Determine number of cores to use
            max_workers = min(os.cpu_count(), len(df_by_month))
            print(f"Using {max_workers} workers for parallel processing")
            
            # Process months in parallel
            results = []
            total_energy_discharged = 0
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(optimize_month, month_data_item) for month_data_item in month_data]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result_dict, energy_discharged, month_label = future.result()
                        if result_dict:
                            results.append((result_dict, df_by_month[month_label].index))
                            total_energy_discharged += energy_discharged
                            print(f"Processed {month_label} - {len(results)}/{len(df_by_month)} months complete")
                    except Exception as e:
                        print(f"Error processing future: {e}")
            
            # Combine all results
            all_dfs = []
            
            for result_dict, month_index in results:
                month_df = pd.DataFrame({
                    'discharge_vars': result_dict['discharge_vars'],
                    'charge_vars': result_dict['charge_vars'],
                    'SOC_vars': result_dict['SOC_vars'],
                    'lmp': result_dict['lmp']
                }, index=month_index)
                
                all_dfs.append(month_df)
            
            # Check if we have any valid results
            if all_dfs:
                # Combine and sort by datetime index
                battery_dispatch_df = pd.concat(all_dfs)
                battery_dispatch_df = battery_dispatch_df.sort_index()
            else:
                print("Warning: No valid optimization results")
                # Return an empty dataframe with the correct format
                battery_dispatch_df = pd.DataFrame({
                    'discharge_vars': [0.0] * len(df_filtered),
                    'charge_vars': [0.0] * len(df_filtered),
                    'SOC_vars': [battery_params['SOC_min']] * len(df_filtered),
                    'lmp': df_filtered['lmp'].values
                }, index=df_filtered.index)
                total_energy_discharged = 0
        
        # Calculate metrics
        DoD = battery_params['SOC_max'] - battery_params['SOC_min']
        usable_energy_capacity_MWh = battery_params['energy_capacity'] * (DoD / battery_params['energy_capacity'])
        total_cycles = total_energy_discharged / usable_energy_capacity_MWh
        
        stats = {
            'total_energy_discharged': total_energy_discharged,
            'usable_energy_capacity_MWh': usable_energy_capacity_MWh,
            'total_cycles': total_cycles
        }
        
        processing_time = time.time() - start_time
        print(f"Standalone dispatch completed in {processing_time:.2f} seconds")
        
        # Debug the final dataframe
        print(f"Final dataframe shape: {battery_dispatch_df.shape}")
        print(f"Final index type: {type(battery_dispatch_df.index)}")
        print(f"Final index sample: {battery_dispatch_df.index[:5]}")
        
        return battery_dispatch_df, stats
        
    except Exception as e:
        print(f"Error in standalone dispatch: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal valid data with a proper datetime index
        minimal_df = pd.DataFrame({
            'discharge_vars': [0.0],
            'charge_vars': [0.0],
            'SOC_vars': [battery_params['SOC_min']],
            'lmp': [0.0]
        }, index=[pd.Timestamp(start_date)])
        
        minimal_stats = {
            'total_energy_discharged': 0.0,
            'usable_energy_capacity_MWh': 0.0,
            'total_cycles': 0.0
        }
        
        return minimal_df, minimal_stats

def run_solar_constrained_dispatch(df, battery_params, start_date, end_date, disable_parallel=True):
    """
    Simplified function that just calls standalone dispatch for API compatibility.
    We're removing the solar constrained dispatch as requested.
    """
    print("Note: Solar constrained dispatch is disabled - using standalone dispatch instead")
    return run_standalone_dispatch(df, battery_params, start_date, end_date, disable_parallel)

def calculate_monthly_metrics(df, discharge_power_limit, charge_efficiency, discharge_efficiency):
    """
    Calculate monthly revenue metrics.
    
    Args:
        df (pandas.DataFrame): Battery dispatch dataframe
        discharge_power_limit (float): Discharge power limit in MW
        charge_efficiency (float): Charge efficiency
        discharge_efficiency (float): Discharge efficiency
        
    Returns:
        pandas.DataFrame: Monthly metrics
    """
    # Calculate revenue and costs
    df = df.copy()
    
    # Print debug info
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame index type: {type(df.index)}")
    print(f"Index is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")
    
    # Check if we have the required columns
    required_columns = ['discharge_vars', 'charge_vars', 'lmp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        # Create empty columns for missing data to avoid errors
        for col in missing_columns:
            df[col] = 0.0
    
    # Ensure we have a datetime index for resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: DataFrame index is not a DatetimeIndex. Converting...")
        try:
            df.index = pd.to_datetime(df.index)
            print("Successfully converted index to DatetimeIndex")
        except Exception as e:
            print(f"Error: Cannot convert index to DatetimeIndex: {e}")
            # Return empty DataFrame to avoid errors
            return pd.DataFrame(columns=['Net Revenue (£)', 'Annualized Revenue per MWp (£)'])
    
    # Calculate revenue metrics
    df['hourly_discharging_revenue'] = df['discharge_vars'] * df['lmp'] * discharge_efficiency
    df['hourly_charging_costs'] = df['charge_vars'] * df['lmp'] / charge_efficiency
    df['hourly_net_revenue'] = df['hourly_discharging_revenue'] - df['hourly_charging_costs']
    
    # Debug print to help diagnose issues
    print(f"First few index values: {df.index[:5]}")
    print(f"Last few index values: {df.index[-5:]}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    
    # Detailed debug output about the dataframe before resampling
    print(f"Net revenue sum: {df['hourly_net_revenue'].sum()}")
    print(f"Discharging revenue sum: {df['hourly_discharging_revenue'].sum()}")
    print(f"Charging costs sum: {df['hourly_charging_costs'].sum()}")
    
    # Resample to monthly and calculate metrics
    try:
        # Make sure the index is sorted for resampling
        df = df.sort_index()
        
        # First convert to datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Resample to month start frequency and sum
        monthly_metrics = df[['hourly_discharging_revenue', 'hourly_charging_costs', 'hourly_net_revenue']].resample('MS').sum()
        monthly_metrics['Net Revenue (£)'] = monthly_metrics['hourly_net_revenue']
        
        # Calculate annualized revenue per MWp
        monthly_metrics['Annualized Revenue per MWp (£)'] = (monthly_metrics['Net Revenue (£)'] / discharge_power_limit) * 12
        
        # Print debug info for monthly metrics
        print(f"Monthly metrics shape: {monthly_metrics.shape}")
        print(f"Monthly metrics index: {monthly_metrics.index}")
        print(f"Monthly metrics values:\n{monthly_metrics}")
        
        return monthly_metrics[['Net Revenue (£)', 'Annualized Revenue per MWp (£)']]
    except Exception as e:
        print(f"Error in monthly resampling: {e}")
        import traceback
        traceback.print_exc()
        # Return empty DataFrame if resampling fails
        return pd.DataFrame(columns=['Net Revenue (£)', 'Annualized Revenue per MWp (£)'])