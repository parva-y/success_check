import streamlit as st
import pandas as pd
import scipy.stats as stats
import numpy as np
from statsmodels.stats.weightstats import ztest
from scipy.stats import ks_2samp
import ruptures as rpt
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Experiment Success Dashboard")

# Backend File Upload (Assumed to be fixed)
uploaded_file = "1MG_Test_and_control_report_transformed (2).csv"  # Replace with actual backend file path

df = pd.read_csv(uploaded_file, parse_dates=['date'])

# Ensure necessary columns exist
required_columns = {'date', 'data_set', 'audience_size', 'app_opens', 'transactors', 'orders', 'gmv', 'cohort'}
if not required_columns.issubset(df.columns):
    st.write("Missing required columns in the CSV file.")
    st.stop()

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

# Cohort selection
selected_cohort = st.sidebar.selectbox("Select Cohort", df['cohort'].unique())
st.sidebar.write(f"Test Start Date: {test_start_dates.get(selected_cohort, 'Unknown')}")

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
    fig.update_traces(connectgaps=False)  # Fix line connection issue
    fig.update_xaxes(type='category', tickformat="%d/%m")  # Ensure dates are displayed correctly without timestamp
    st.plotly_chart(fig, use_container_width=True)

# Prepare results table
all_results = []
summary_results = []

test_start_dates_actual = {tg: df_filtered[df_filtered['data_set'] == tg]['date'].min() for tg in test_groups}

# Iterate over test groups
for test_group in test_groups:
    first_test_date = test_start_dates_actual[test_group]
    
    for metric in metrics:
        df_control = df_filtered[(df_filtered['data_set'] == control_group) & (df_filtered['date'] >= first_test_date)].groupby('date')[metric].mean()
        df_test = df_filtered[(df_filtered['data_set'] == test_group) & (df_filtered['date'] >= first_test_date)].groupby('date')[metric].mean()
        
        df_combined = pd.concat([df_control, df_test], axis=1, keys=[control_group, test_group]).dropna()
        control_values = df_combined[control_group]
        test_values = df_combined[test_group]
        
        # Perform statistical tests
        t_stat, p_value_ttest = stats.ttest_rel(control_values, test_values)
        u_stat, p_value_mw = stats.mannwhitneyu(control_values, test_values, alternative='two-sided')
        z_stat, p_value_ztest = ztest(control_values, test_values)
        ks_stat, p_value_ks = ks_2samp(control_values, test_values)
        
        tests = [
            ("Paired t-test", t_stat, p_value_ttest),
            ("Mann-Whitney U Test", u_stat, p_value_mw),
            ("Z-Test", z_stat, p_value_ztest),
            ("Kolmogorov-Smirnov Test", ks_stat, p_value_ks)
        ]
        
        for test_name, stat, p_value in tests:
            significance = "Pass" if (not pd.isna(p_value) and p_value < 0.05) else "Fail"
            all_results.append([selected_cohort, test_group, metric, control_values.mean(), test_values.mean(), test_name, stat, p_value])
            summary_results.append([selected_cohort, test_name, significance])

# Display detailed results
st.write("### Detailed Experiment Results Table")
results_df = pd.DataFrame(all_results, columns=["Cohort", "Test Group", "Metric", "Control Mean", "Test Mean", "Test", "Statistic", "P-Value"])

styled_df = results_df.style.apply(lambda s: ['background-color: lightgreen' if (not pd.isna(v) and v < 0.05) else '' for v in s], subset=["P-Value"])
st.dataframe(styled_df)

# Display summary results
st.write("### Summary Table")
summary_df = pd.DataFrame(summary_results, columns=["Cohort", "Test Name", "Pass/Fail"]).drop_duplicates()
st.dataframe(summary_df)
