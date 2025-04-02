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
        
        t_stat, p_value_ttest = stats.ttest_rel(control_values, test_values)
        pass_fail = "Pass" if p_value_ttest < 0.05 else "Fail"
        all_results.append([selected_cohort, test_group, metric, control_values.mean(), test_values.mean(), "Paired t-test", t_stat, p_value_ttest, pass_fail])

st.write("### Detailed Experiment Results Table")
results_df = pd.DataFrame(all_results, columns=["Cohort", "Test Group", "Metric", "Control Mean", "Test Mean", "Test", "Statistic", "P-Value", "Pass/Fail"])

styled_df = results_df.style.apply(lambda s: ['background-color: lightgreen' if v == "Pass" else '' for v in s], subset=["Pass/Fail"])
st.dataframe(styled_df)
