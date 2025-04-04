import streamlit as st
import pandas as pd
import scipy.stats as stats
import numpy as np
from statsmodels.stats.weightstats import ztest
from scipy.stats import ks_2samp
import plotly.express as px

st.set_page_config(layout="wide")

# Password protection
if "authenticated" not in st.session_state:
    password = st.text_input("Enter Password:", type="password")
    if password == "Antidormancy@68C":
        st.session_state["authenticated"] = True
    else:
        st.stop()

st.title("Experiment Success Dashboard")

# Backend File Upload (Assumed to be fixed)
uploaded_file = "1MG_Test_and_control_report_transformed (2).csv"  # Replace with actual backend file path

df = pd.read_csv(uploaded_file, parse_dates=['date'])

# Ensure necessary columns exist
required_columns = {'date', 'data_set', 'audience_size', 'app_opens', 'transactors', 'orders', 'gmv', 'cohort'}
# Check for Recency column
has_recency_data = 'Recency' in df.columns

if not required_columns.issubset(df.columns):
    st.write("Missing required columns in the CSV file.")
    st.stop()

if has_recency_data:
    st.sidebar.write("✅ Recency data available")

# Sort data by date
df = df.sort_values(by='date')

# Calculate metrics
df['gmv_per_audience'] = df['gmv'] / df['audience_size']
df['app_opens_per_audience'] = df['app_opens'] / df['audience_size']
df['orders_per_audience'] = df['orders'] / df['audience_size']
df['transactors_per_audience'] = df['transactors'] / df['audience_size']

# Define control group
control_group = "Control Set"

# Test start dates
test_start_dates = {
    "resp": pd.Timestamp("2025-03-05"),
    "cardiac": pd.Timestamp("2025-03-18"),
    "diabetic": pd.Timestamp("2025-03-06"),
    "derma": pd.Timestamp("2025-03-18")
}

# Marked test dates
test_marked_dates = {
    "derma": ["2025-03-18", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-25", "2025-03-28", "2025-04-01", "2025-04-02", "2025-04-05", "2025-04-07", "2025-04-08", "2025-04-10", "2025-04-11"],
    "diabetic": ["2025-03-06", "2025-03-07", "2025-03-11", "2025-03-13", "2025-03-15", "2025-03-19", "2025-03-23", "2025-03-25", "2025-03-29", "2025-04-01", "2025-04-03"],
    "cardiac": ["2025-03-18", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-25", "2025-04-01", "2025-04-02", "2025-04-05", "2025-04-07", "2025-04-10", "2025-04-12", "2025-04-15", "2025-04-17"],
    "resp": ["2025-03-05", "2025-03-08", "2025-03-12", "2025-03-15", "2025-03-17", "2025-03-19", "2025-03-23", "2025-03-27", "2025-03-30", "2025-04-02", "2025-04-04"]
}

# Cohort selection
selected_cohort = st.sidebar.selectbox("Select Cohort", df['cohort'].unique())
st.sidebar.write(f"Test Start Date: {test_start_dates.get(selected_cohort, 'Unknown')}")

# Recency selector (new feature)
if has_recency_data:
    # Get all available recency values
    recency_values = sorted(df['Recency'].unique())
    # Add "Overview" option at the beginning
    recency_options = ["Overview"] + list(recency_values)
    selected_recency = st.sidebar.selectbox("Select Recency", recency_options)
    st.sidebar.write(f"Selected Recency: {selected_recency}")
else:
    selected_recency = "Overview"  # Default if no recency data

st.write(f"## Selected Cohort: {selected_cohort}")
if selected_recency != "Overview":
    st.write(f"## Selected Recency: {selected_recency}")

# Filter data by cohort and date
base_filtered_df = df[(df['cohort'] == selected_cohort) & (df['date'] >= test_start_dates.get(selected_cohort, df['date'].min()))]

# Now apply recency filter if needed
if selected_recency != "Overview" and has_recency_data:
    df_filtered = base_filtered_df[base_filtered_df['Recency'] == selected_recency]
else:
    df_filtered = base_filtered_df

test_groups = [g for g in df_filtered['data_set'].unique() if g != control_group]

if not test_groups:
    st.write("No test groups found for this cohort and recency combination.")
    st.stop()

# Metric Selection
metrics = ['gmv_per_audience', 'app_opens_per_audience', 'orders_per_audience', 'transactors_per_audience']

# Plot trends
st.write("### Metric Trends: Control vs Test Groups")

# Function to prepare data for plotting based on selected recency
def prepare_plot_data(base_df, metric, selected_recency):
    if selected_recency == "Overview" and has_recency_data:
        # Sum metrics for the same date across recencies and recalculate per audience
        plot_data = base_df.groupby(['date', 'data_set']).agg({
            'gmv': 'sum',
            'app_opens': 'sum',
            'orders': 'sum',
            'transactors': 'sum',
            'audience_size': 'sum'
        }).reset_index()
        
        # Recalculate the metrics after aggregation
        plot_data['gmv_per_audience'] = plot_data['gmv'] / plot_data['audience_size']
        plot_data['app_opens_per_audience'] = plot_data['app_opens'] / plot_data['audience_size']
        plot_data['orders_per_audience'] = plot_data['orders'] / plot_data['audience_size']
        plot_data['transactors_per_audience'] = plot_data['transactors'] / plot_data['audience_size']
        
        return plot_data
    else:
        # Use filtered data directly if specific recency is selected
        return base_df

for metric in metrics:
    plot_data = prepare_plot_data(base_filtered_df, metric, selected_recency)
    
    fig = px.line(plot_data, x='date', y=metric, color='data_set', title=metric.replace("_", " ").title())
    fig.update_traces(connectgaps=False)
    fig.update_xaxes(tickformat="%d/%m")

    for mark_date in test_marked_dates.get(selected_cohort, []):
        if mark_date in plot_data['date'].astype(str).values:
            fig.add_vline(x=mark_date, line_width=2, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

# Add recency analysis if data is available and Overview is selected
if has_recency_data and selected_recency == "Overview":
    st.write("### Recency Breakdown Analysis")
    
    # Define recency ranges we're interested in
    recency_ranges = ['91-120', '121-150', '151-180', '181-365']
    
    # Filter by recency ranges we're interested in
    df_recency = base_filtered_df[base_filtered_df['Recency'].isin(recency_ranges)]
    
    # Get the latest data for each data_set and recency combination
    latest_date = base_filtered_df['date'].max()
    latest_data = df_recency[df_recency['date'] == latest_date]
    
    if len(latest_data) > 0:
        # Group by data_set and Recency, and sum the audience_size
        recency_summary = latest_data.groupby(['data_set', 'Recency']).agg({
            'audience_size': 'sum'
        }).reset_index()
        
        # Calculate total audience size per data_set
        total_audience = latest_data.groupby('data_set')['audience_size'].sum().reset_index()
        total_audience.columns = ['data_set', 'total_audience']
        
        # Merge to get percentages
        recency_summary = recency_summary.merge(total_audience, on='data_set')
        recency_summary['percentage'] = (recency_summary['audience_size'] / recency_summary['total_audience'] * 100).round(2)
        
        # Plot recency distribution
        fig1 = px.bar(recency_summary, x='data_set', y='percentage', color='Recency', 
                     barmode='group', title='Recency Distribution (%) by Group')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.bar(recency_summary, x='data_set', y='audience_size', color='Recency', 
                     barmode='group', title='Recency Distribution (Count) by Group')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Create a data table for the recency metrics
        st.write("### Recency Data Table")
        
        # Pivot table for better display
        pivot_table = recency_summary.pivot(index='data_set', columns='Recency', 
                                          values=['audience_size', 'percentage'])
        
        # Flatten the MultiIndex and create a new DataFrame
        table_data = []
        for data_set in pivot_table.index:
            row = {'Data Set': data_set, 'Total Audience': total_audience[total_audience['data_set'] == data_set]['total_audience'].iloc[0]}
            
            for recency_range in recency_ranges:
                try:
                    row[f'{recency_range} Count'] = pivot_table.loc[data_set, ('audience_size', recency_range)]
                    row[f'{recency_range} %'] = pivot_table.loc[data_set, ('percentage', recency_range)]
                except:
                    row[f'{recency_range} Count'] = 0
                    row[f'{recency_range} %'] = 0.0
                    
            table_data.append(row)
            
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df)
    else:
        st.write("No recency data available for the latest date.")

# Prepare results table based on the filtered data (respecting recency selection)
st.write("### Detailed Experiment Results Table")
all_results = []

test_start_dates_actual = {tg: df_filtered[df_filtered['data_set'] == tg]['date'].min() for tg in test_groups}

for test_group in test_groups:
    first_test_date = test_start_dates_actual[test_group]

    for metric in metrics:
        # Get data after test start date for control and test groups
        if selected_recency == "Overview" and has_recency_data:
            # For overview, we need to aggregate first then calculate stats
            control_data = base_filtered_df[(base_filtered_df['data_set'] == control_group) & 
                                    (base_filtered_df['date'] >= first_test_date)]
            test_data = base_filtered_df[(base_filtered_df['data_set'] == test_group) & 
                                (base_filtered_df['date'] >= first_test_date)]
            
            # Group by date and calculate the metrics
            control_grouped = control_data.groupby('date').agg({
                'gmv': 'sum',
                'app_opens': 'sum',
                'orders': 'sum',
                'transactors': 'sum',
                'audience_size': 'sum'
            })
            
            test_grouped = test_data.groupby('date').agg({
                'gmv': 'sum',
                'app_opens': 'sum',
                'orders': 'sum',
                'transactors': 'sum',
                'audience_size': 'sum'
            })
            
            # Calculate metrics from aggregated data
            if metric == 'gmv_per_audience':
                control_values = control_grouped['gmv'] / control_grouped['audience_size']
                test_values = test_grouped['gmv'] / test_grouped['audience_size']
            elif metric == 'app_opens_per_audience':
                control_values = control_grouped['app_opens'] / control_grouped['audience_size']
                test_values = test_grouped['app_opens'] / test_grouped['audience_size']
            elif metric == 'orders_per_audience':
                control_values = control_grouped['orders'] / control_grouped['audience_size']
                test_values = test_grouped['orders'] / test_grouped['audience_size']
            elif metric == 'transactors_per_audience':
                control_values = control_grouped['transactors'] / control_grouped['audience_size']
                test_values = test_grouped['transactors'] / test_grouped['audience_size']
        else:
            # For specific recency, use filtered data directly
            df_control = df_filtered[(df_filtered['data_set'] == control_group) & 
                                    (df_filtered['date'] >= first_test_date)]
            df_test = df_filtered[(df_filtered['data_set'] == test_group) & 
                                (df_filtered['date'] >= first_test_date)]
            
            # Group by date to get daily averages
            control_values = df_control.groupby('date')[metric].mean()
            test_values = df_test.groupby('date')[metric].mean()

        # Ensure we have matching dates for paired tests
        df_combined = pd.concat([control_values, test_values], axis=1, keys=[control_group, test_group]).dropna()
        
        if len(df_combined) < 2:
            # Skip if not enough data points for statistical testing
            continue
            
        control_values = df_combined[control_group]
        test_values = df_combined[test_group]

        # T-test
        t_stat, p_value_ttest = stats.ttest_rel(control_values, test_values)
        
        # Mann-Whitney U Test
        u_stat, p_value_mw = stats.mannwhitneyu(control_values, test_values, alternative='two-sided')
        
        # Kolmogorov-Smirnov Test
        ks_stat, p_value_ks = ks_2samp(control_values, test_values)

        tests = [
            ("Paired t-test", t_stat, p_value_ttest),
            ("Mann-Whitney U Test", u_stat, p_value_mw), 
            ("Kolmogorov-Smirnov Test", ks_stat, p_value_ks)
        ]

        recency_label = selected_recency if selected_recency != "Overview" else "All"
        
        for test_name, stat, p_value in tests:
            pass_fail = "Pass" if p_value < 0.05 else "Fail"
            lift = ((test_values.mean() - control_values.mean()) / control_values.mean() * 100).round(2)
            all_results.append([
                selected_cohort, 
                test_group, 
                recency_label,
                metric, 
                control_values.mean(), 
                test_values.mean(),
                lift, 
                test_name, 
                stat, 
                p_value, 
                pass_fail
            ])

results_df = pd.DataFrame(all_results, columns=[
    "Cohort", "Test Group", "Recency", "Metric", 
    "Control Mean", "Test Mean", "Lift %", 
    "Test", "Statistic", "P-Value", "Pass/Fail"
])

# Round numeric columns for better display
numeric_cols = ["Control Mean", "Test Mean", "Lift %", "Statistic", "P-Value"]
results_df[numeric_cols] = results_df[numeric_cols].round(4)

# Add color styling for pass/fail and lift
def style_dataframe(val):
    if isinstance(val, str) and val == "Pass":
        return 'background-color: lightgreen'
    elif isinstance(val, float) and pd.Series([val]).index[0] == results_df.columns.get_loc("Lift %") and val > 0:
        return 'color: green'
    elif isinstance(val, float) and pd.Series([val]).index[0] == results_df.columns.get_loc("Lift %") and val < 0:
        return 'color: red'
    return ''

styled_df = results_df.style.applymap(style_dataframe)
st.dataframe(styled_df)
