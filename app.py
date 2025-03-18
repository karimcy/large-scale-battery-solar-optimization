import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile
import pathlib

# Import the GNR BESS optimization module
from gnr_bess_optimizer import (
    run_standalone_dispatch,
    calculate_monthly_metrics,
    get_battery_parameters,
    validate_dataframe
)

st.set_page_config(page_title="GNR BESS Curtailment Analysis", page_icon="🔋", layout="wide")

st.title("GNR BESS Curtailment Analysis")

# Sidebar for parameters
with st.sidebar:
    st.header("Battery and Solar Parameters")
    
    # Battery parameters
    energy_capacity = st.number_input("Energy Capacity (MWh)", value=1600, min_value=1)
    st.caption("The maximum volume of energy that can be stored in the battery system.")
    
    charge_power_limit = st.number_input("Charge Power Limit (MW)", value=800, min_value=1)
    st.caption("The maximum power rate at which the battery can charge.")
    
    discharge_power_limit = st.number_input("Discharge Power Limit (MW)", value=800, min_value=1)
    st.caption("The maximum power rate at which the battery can discharge.")
    
    charge_efficiency = st.number_input("Charge Efficiency", value=0.95, min_value=0.01, max_value=1.0, format="%.2f")
    st.caption("The efficiency at which energy can enter the battery.")
    
    discharge_efficiency = st.number_input("Discharge Efficiency", value=0.95, min_value=0.01, max_value=1.0, format="%.2f")
    st.caption("The efficiency at which energy can leave the battery.")
    
    soc_max_percentage = st.number_input("Maximum State of Charge (%)", value=100, min_value=1, max_value=100)
    st.caption("The maximum allowable percentage of energy stored in the battery.")
    
    soc_min_percentage = st.number_input("Minimum State of Charge (%)", value=20, min_value=0, max_value=99)
    st.caption("The minimum allowable percentage of energy stored in the battery.")
    
    daily_cycle_limit = st.number_input("Daily Cycle Limit", value=1.5, min_value=0.1, format="%.1f")
    st.caption("The maximum number of cycles allowed in a day.")
    
    annual_cycle_limit = st.number_input("Annual Cycle Limit", value=600, min_value=1)
    st.caption("The maximum number of cycles allowed in a year.")
    
    solar_dc_capacity = st.number_input("Solar DC Capacity (MW)", value=1120, min_value=1)
    st.caption("The maximum power generation capacity of the solar system.")
    
    maximum_export_capacity = st.number_input("Maximum Export Capacity (MW)", value=800, min_value=1)
    st.caption("The maximum power that can be exported to the grid.")

    # Add parallelization toggle
    st.header("Performance Settings")
    disable_parallel = st.checkbox("Disable parallelization", value=False)
    st.caption("Disable monthly parallel processing (slower but more stable).")

# Main content area
st.header("Data Input")

# Look for the CSV file in the Downloads directory
home_dir = str(pathlib.Path.home())
downloads_dir = os.path.join(home_dir, "Downloads")
default_csv_path = os.path.join(downloads_dir, "UK Half Hourly for merge.csv")

# Toggle for using the default file or uploading a new one
use_default_file = st.checkbox("Use file from Downloads directory", value=True)

if use_default_file:
    if os.path.exists(default_csv_path):
        st.success(f"Using file: {default_csv_path}")
        # Read the data directly
        try:
            df = pd.read_csv(default_csv_path)
            
            # Validate the dataframe structure
            validation_result, message = validate_dataframe(df)
            if not validation_result:
                st.error(f"Error in data file: {message}")
                st.info("Please fix the CSV file or upload a different one.")
                data_file = st.file_uploader("Upload your data file (CSV format)", type=["csv"])
                if data_file is not None:
                    try:
                        # Save the uploaded file to a temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(data_file.getvalue())
                            tmp_filepath = tmp_file.name
                        
                        # Read the data
                        df = pd.read_csv(tmp_filepath)
                        
                        # Validate the dataframe structure again
                        validation_result, message = validate_dataframe(df)
                        if not validation_result:
                            st.error(f"Error in uploaded data file: {message}")
                        else:
                            st.success("Data loaded successfully from uploaded file.")
                            os.unlink(tmp_filepath)  # Clean up the temporary file
                    except Exception as e:
                        st.error(f"Error processing the uploaded file: {str(e)}")
            else:
                st.success("Data loaded successfully from default file.")
        except Exception as e:
            st.error(f"Error reading the default file: {str(e)}")
            st.info("Please upload a file instead.")
            use_default_file = False
    else:
        st.error(f"Default file not found: {default_csv_path}")
        st.info("Please upload a file instead.")
        use_default_file = False

if not use_default_file:
    data_file = st.file_uploader("Upload your data file (CSV format)", type=["csv"])
    if data_file is not None:
        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(data_file.getvalue())
                tmp_filepath = tmp_file.name
            
            # Read the data
            df = pd.read_csv(tmp_filepath)
            
            # Validate the dataframe structure
            validation_result, message = validate_dataframe(df)
            if not validation_result:
                st.error(f"Error in data file: {message}")
            else:
                st.success("Data loaded successfully from uploaded file.")
            
            # Clean up the temporary file
            os.unlink(tmp_filepath)
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    else:
        st.info("Please upload a CSV file with the required data format.")
        st.markdown("""
        #### Required Data Format:
        The CSV file should contain the following columns:
        - `Date, time`: Timestamp for each interval
        - `LMP` or `lmp`: Locational Marginal Price
        - `System Export - KW (800)`: Solar export data in KW
        
        If your file has different column names, you may need to rename them before uploading.
        """)

# Continue only if we have a valid dataframe
if 'df' in locals() and validation_result:
    # Allow date range selection
    st.header("Date Range Selection")
    
    # Find the date column (case insensitive)
    date_col = None
    for col in df.columns:
        if col.lower() == 'date, time':
            date_col = col
            break
    
    df[date_col] = pd.to_datetime(df[date_col])
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    if start_date <= end_date:
        # Create a dictionary of parameters to pass to the optimizer
        battery_params = get_battery_parameters(
            energy_capacity, charge_power_limit, discharge_power_limit,
            charge_efficiency, discharge_efficiency, 
            soc_max_percentage, soc_min_percentage,
            daily_cycle_limit, annual_cycle_limit,
            solar_dc_capacity, maximum_export_capacity
        )
        
        # Show estimate of data size
        filtered_df = df[(df[date_col] >= pd.Timestamp(start_date)) & (df[date_col] <= pd.Timestamp(end_date))]
        intervals = len(filtered_df)
        
        st.info(f"Selected range contains {intervals:,} intervals ({intervals/48:.1f} days).")
        
        # Run the analysis when the user clicks the button
        if st.button("Run Analysis"):
            # Create progress placeholders
            status = st.empty()
            status.info("Starting battery dispatch optimization...")
            
            # Add time tracking
            import time
            start_time = time.time()
            
            # Run dispatch with progress updates
            with st.spinner("Running Battery Dispatch Optimization..."):
                # Run the optimization with parallelization option
                battery_dispatch_df, dispatch_stats = run_standalone_dispatch(
                    df, battery_params, start_date, end_date, disable_parallel
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                status.success(f"Battery dispatch optimization completed in {elapsed:.1f} seconds")
            
            # Calculate monthly metrics
            with st.spinner("Calculating monthly metrics..."):
                monthly_metrics = calculate_monthly_metrics(
                    battery_dispatch_df,
                    discharge_power_limit,
                    charge_efficiency,
                    discharge_efficiency
                )
            
            # Display results
            st.header("Results")
            
            # Display statistics
            st.subheader("Battery Dispatch Statistics")
            
            # Calculate additional statistics
            total_days = intervals / 48  # Half-hourly intervals mean 48 intervals per day
            avg_daily_cycles = dispatch_stats['total_cycles'] / total_days
            lifetime_revenue = monthly_metrics['Net Revenue (£)'].sum() if not monthly_metrics.empty else 0
            lifetime_revenue_per_mwp = lifetime_revenue / discharge_power_limit if discharge_power_limit > 0 else 0
            avg_annual_revenue_per_mwp = lifetime_revenue_per_mwp * (365.25 / total_days) if total_days > 0 else 0
            
            # Display in two rows of three columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Energy Discharged (MWh)", 
                          f"{dispatch_stats['total_energy_discharged']:,.2f}")
            with col2:
                st.metric("Usable Energy Capacity (MWh)", 
                          f"{dispatch_stats['usable_energy_capacity_MWh']:,.2f}")
            with col3:
                st.metric("Total Cycles", 
                          f"{dispatch_stats['total_cycles']:,.2f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Daily Cycles", 
                          f"{avg_daily_cycles:.2f}")
            with col2:
                st.metric("Total Earnings (£)", 
                          f"{lifetime_revenue:,.2f}")
            with col3:
                st.metric("Avg Annual Revenue per MW (£/MW)", 
                          f"{avg_annual_revenue_per_mwp:,.2f}")
            
            # Display monthly metrics
            st.subheader("Monthly Revenue")
            
            if not monthly_metrics.empty:
                # Calculate annual total for reference
                annual_revenue = monthly_metrics['Net Revenue (£)'].sum()
                annual_revenue_per_mwp = annual_revenue / discharge_power_limit
                
                # Create a bar chart for the monthly revenue
                fig = go.Figure()
                
                # Add bar for Net Revenue
                fig.add_trace(go.Bar(
                    x=monthly_metrics.index,
                    y=monthly_metrics['Net Revenue (£)'],
                    name='Net Revenue (£)',
                    marker_color='royalblue'
                ))
                
                # Add line for the Annualized Revenue per MWp
                fig.add_trace(go.Scatter(
                    x=monthly_metrics.index,
                    y=monthly_metrics['Annualized Revenue per MWp (£)'],
                    name='Annualized Revenue per MWp (£)',
                    mode='lines+markers',
                    line=dict(color='firebrick', width=2, dash='dot'),
                    yaxis='y2'
                ))
                
                # Update layout with a secondary y-axis
                fig.update_layout(
                    title='Monthly Revenue and Annualized Revenue per MWp',
                    xaxis_title='Month',
                    yaxis_title='Net Revenue (£)',
                    yaxis2=dict(
                        title='Annualized Revenue per MWp (£)',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Format the dataframe for display with better formatting
                display_df = monthly_metrics.copy()
                display_df['Net Revenue (£)'] = display_df['Net Revenue (£)'].map(lambda x: f"£{x:,.2f}")
                display_df['Annualized Revenue per MWp (£)'] = display_df['Annualized Revenue per MWp (£)'].map(lambda x: f"£{x:,.2f}")
                
                # Display metrics in a table with better styling
                st.dataframe(
                    display_df, 
                    column_config={
                        "Net Revenue (£)": st.column_config.TextColumn("Net Revenue (£)", width="medium"),
                        "Annualized Revenue per MWp (£)": st.column_config.TextColumn("Annualized Revenue (£/MWp)", width="medium")
                    },
                    use_container_width=True
                )
            else:
                st.warning("No monthly metrics available. This may be due to issues with the dispatch results.")
                
            # Display time series data for a sample period
            st.subheader("Battery Operation Sample")
            
            # Select a sample of data to display (first 48 hours or all if less)
            sample_size = min(96, len(battery_dispatch_df))
            sample_df = battery_dispatch_df.iloc[:sample_size]
            
            fig = go.Figure()
            
            # Add traces for charge, discharge and price
            fig.add_trace(go.Scatter(
                x=sample_df.index,
                y=sample_df['charge_vars'],
                name='Charging (MW)',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=sample_df.index,
                y=sample_df['discharge_vars'],
                name='Discharging (MW)',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=sample_df.index,
                y=sample_df['SOC_vars'],
                name='State of Charge (MWh)',
                line=dict(color='blue', dash='dash')
            ))
            
            # Create a secondary y-axis for price
            fig.add_trace(go.Scatter(
                x=sample_df.index,
                y=sample_df['lmp'],
                name='Price (£/MWh)',
                line=dict(color='purple'),
                yaxis='y2'
            ))
            
            # Update layout with a secondary y-axis
            fig.update_layout(
                title='Battery Operation - First 48 Hours (Sample)',
                xaxis_title='Time',
                yaxis_title='Power (MW) / Energy (MWh)',
                legend_title='Variables',
                height=600,
                yaxis2=dict(
                    title='Price (£/MWh)',
                    overlaying='y',
                    side='right'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow downloading the results
            st.subheader("Download Results")
            
            # Create download buttons for the dataframes
            battery_dispatch_csv = battery_dispatch_df.to_csv(index=True)
            monthly_metrics_csv = monthly_metrics.to_csv(index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Battery Dispatch Data",
                    data=battery_dispatch_csv,
                    file_name="battery_dispatch_results.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download Monthly Metrics",
                    data=monthly_metrics_csv,
                    file_name="monthly_metrics.csv",
                    mime="text/csv"
                )
                
    else:
        st.error("Error: End date must be after start date.") 