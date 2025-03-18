# Large-Scale Co-located Battery/Solar Optimization

A sophisticated web application for optimizing battery energy storage systems (BESS) dispatch strategies when co-located with solar PV facilities. This tool helps maximize revenue based on time-varying electricity prices and operational constraints.

## Overview

This application provides a powerful optimization framework for large-scale battery storage systems co-located with solar PV. The tool:

- Optimizes battery charge/discharge schedules to maximize revenue
- Handles large datasets efficiently through parallel processing
- Provides detailed monthly and annual revenue metrics
- Visualizes optimization results with interactive charts
- Supports customization of all battery and solar parameters

## Features

- **Battery Parameter Customization**: Set energy capacity, power limits, efficiencies, state of charge limits, and cycle limits.
- **Solar Integration**: Configure solar DC capacity and maximum export capacity.
- **High-Performance Optimization**: Uses PuLP with CBC solver and parallel processing by month.
- **Interactive Visualizations**: View revenue metrics, battery operation patterns, and price data.
- **Detailed Metrics**: Analyze total energy dispatched, cycles used, and revenue per MW.
- **Data Export**: Download optimization results for further analysis.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/karimcy/large-scale-battery-solar-optimization.git
cd large-scale-battery-solar-optimization
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload your electricity price and solar generation data (CSV format).

3. Configure battery and solar parameters in the sidebar.

4. Select the date range for analysis.

5. Click "Run Analysis" to optimize the battery dispatch.

6. View the results and download the data for further analysis.

## Required Data Format

The application requires a CSV file with the following columns:
- `Date, time`: Timestamp for each interval
- `LMP` or `lmp`: Locational Marginal Price (£/MWh)
- `System Export - KW (800)`: Solar export data in KW

## How It Works

The optimization algorithm:
1. Processes the input data and applies date range filtering
2. For large datasets, splits the data into monthly chunks
3. Optimizes battery dispatch for each month in parallel
4. Maximizes revenue based on price arbitrage while respecting battery constraints
5. Combines results and calculates performance metrics

## Technical Details

- Linear programming optimization using PuLP
- Parallel processing for efficient handling of large datasets
- Half-hourly interval processing
- Custom constraints for battery state of charge, power limits, and cycling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PuLP library for linear programming optimization
- Streamlit for the interactive web interface
- Plotly for data visualization 