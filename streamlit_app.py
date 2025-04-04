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
# Check for recency columns - UPDATED COLUMN NAMES
recency_columns = {'r_91_120', 'r_121_150', 'r_151_180', 'r_181_365'}
has_recency_data = recency_columns.issubset(df.columns)

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

st.write(f"## Selected Cohort: {selected_cohort}")

df_filtered = df[(df['cohort'] == selected_cohort) & (df['date'] >= test_start_dates.get(selected_cohort, df['date'].min()))]

test_groups = [g for g in df_filtered['data_set'].unique() if g != control_group]

if not test_groups:
    st.write("No test groups found for this cohort.")
    st.stop()

# Metric Selection
metrics = ['gmv_per_audience', 'app_opens_per_audience', 'orders_per_audience', 'transactors_per_audience']

# Plot trends
st.write("### Metric Trends: Control vs Test Groups")
for metric in metrics:
    fig = px.line(df_filtered, x='date', y=metric, color='data_set', title=metric.replace("_", " ").title())
    fig.update_traces(connectgaps=False)
    fig.update_xaxes(tickformat="%d/%m")

    for mark_date in test_marked_dates.get(selected_cohort, []):
        if mark_date in df_filtered['date'].astype(str).values:
            fig.add_vline(x=mark_date, line_width=2, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

# Add recency analysis if data is available
if has_recency_data:
    st.write("### Recency Breakdown Analysis")
    
    # Get the latest date for each data_set
    latest_data = df_filtered.loc[df_filtered.groupby('data_set')['date'].idxmax()]
    
    # Create recency columns with percentages - UPDATED COLUMN NAMES
    recency_metrics = ['r_91_120', 'r_121_150', 'r_151_180', 'r_181_365']
    
    # Pivot the recency data for easier comparison
    recency_pivot = latest_data.pivot(index='data_set', columns=None, 
                                     values=['audience_size'] + recency_metrics)
    
    # Calculate percentages
    for metric in recency_metrics:
        recency_pivot[f'{metric}_pct'] = (recency_pivot[metric] / recency_pivot['audience_size'] * 100).round(2)
    
    # Convert to a format suitable for plotting
    recency_plot_data = []
    for idx, row in recency_pivot.iterrows():
        for metric in recency_metrics:
            label = metric.replace('r_', '')
            recency_plot_data.append({
                'Data Set': idx,
                'Recency Range (days)': label,
                'Count': row[metric],
                'Percentage': row[f'{metric}_pct']
            })
    
    recency_df = pd.DataFrame(recency_plot_data)
    
    # Plot recency distribution
    fig1 = px.bar(recency_df, x='Data Set', y='Percentage', color='Recency Range (days)', 
                 barmode='group', title='Recency Distribution (%) by Group')
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.bar(recency_df, x='Data Set', y='Count', color='Recency Range (days)', 
                 barmode='group', title='Recency Distribution (Count) by Group')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Create a data table for the recency metrics
    st.write("### Recency Data Table")
    
    # Reshape for better display in a table
    table_data = []
    for group in recency_pivot.index:
        row_data = {'Data Set': group, 'Audience Size': recency_pivot.loc[group, 'audience_size']}
        for metric in recency_metrics:
            row_data[f'{metric} Count'] = recency_pivot.loc[group, metric]
            row_data[f'{metric} %'] = recency_pivot.loc[group, f'{metric}_pct']
        table_data.append(row_data)
        
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df)

# Prepare results table
all_results = []

test_start_dates_actual = {tg: df_filtered[df_filtered['data_set'] == tg]['date'].min() for tg in test_groups}

for test_group in test_groups:
    first_test_date = test_start_dates_actual[test_group]

    for metric in metrics:
        df_control = df_filtered[(df_filtered['data_set'] == control_group) & (df_filtered['date'] >= first_test_date)].groupby('date')[metric].mean()
        df_test = df_filtered[(df_filtered['data_set'] == test_group) & (df_filtered['date'] >= first_test_date)].groupby('date')[metric].mean()

        df_combined = pd.concat([df_control, df_test], axis=1, keys=[control_group, test_group]).dropna()
        control_values = df_combined[control_group]
        test_values = df_combined[test_group]

        # T-test (already implemented)
        t_stat, p_value_ttest = stats.ttest_rel(control_values, test_values)
        
        # Mann-Whitney U Test (already implemented)
        u_stat, p_value_mw = stats.mannwhitneyu(control_values, test_values, alternative='two-sided')
        
        # Kolmogorov-Smirnov Test (already implemented)
        ks_stat, p_value_ks = ks_2samp(control_values, test_values)
        
        # No additional tests needed

        tests = [
            ("Paired t-test", t_stat, p_value_ttest),
            ("Mann-Whitney U Test", u_stat, p_value_mw), 
            ("Kolmogorov-Smirnov Test", ks_stat, p_value_ks)
        ]

        for test_name, stat, p_value in tests:
            pass_fail = "Pass" if p_value < 0.05 else "Fail"
            all_results.append([selected_cohort, test_group, metric, control_values.mean(), test_values.mean(), test_name, stat, p_value, pass_fail])

st.write("### Detailed Experiment Results Table")
results_df = pd.DataFrame(all_results, columns=["Cohort", "Test Group", "Metric", "Control Mean", "Test Mean", "Test", "Statistic", "P-Value", "Pass/Fail"])

styled_df = results_df.style.apply(lambda s: ['background-color: lightgreen' if v == "Pass" else '' for v in s], subset=["Pass/Fail"])
st.dataframe(styled_df)
