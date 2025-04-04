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
    
    # Define recency ranges we're interested in
    recency_ranges = ['91-120', '121-150', '151-180', '181-365']
    
    # Filter by recency ranges we're interested in
    df_recency = df_filtered[df_filtered['Recency'].isin(recency_ranges)]
    
    # Get the latest data for each data_set and recency combination
    latest_date = df_filtered['date'].max()
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
