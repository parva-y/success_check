import streamlit as st
import pandas as pd
import scipy.stats as stats

st.title("Experiment Success Checker")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    st.write("Data Preview:")
    st.write(df.head())
    
    # Ensure necessary columns exist
    required_columns = {'date', 'data_set', 'audience_size', 'app_opens', 'transactors', 'orders', 'gmv', 'cohort'}
    if not required_columns.issubset(df.columns):
        st.write("Missing required columns in the CSV file.")
    else:
        # Calculate metrics
        df['gmv_per_audience'] = df['gmv'] / df['audience_size']
        df['app_opens_per_audience'] = df['app_opens'] / df['audience_size']
        df['orders_per_audience'] = df['orders'] / df['audience_size']
        df['transactors_per_audience'] = df['transactors'] / df['audience_size']
        
        # Select cohort
        cohort_options = df['cohort'].unique()
        selected_cohort = st.selectbox("Select Cohort", cohort_options)
        df_filtered = df[df['cohort'] == selected_cohort]
        
        # Identify control and test groups
        data_sets = df_filtered['data_set'].unique()
        if len(data_sets) != 2:
            st.write("Error: Exactly two data sets (Control & Test) are required for comparison.")
        else:
            control_group = st.selectbox("Select Control Group", data_sets)
            test_group = [ds for ds in data_sets if ds != control_group][0]
            
            # Select metric to analyze
            metric_options = ['gmv_per_audience', 'app_opens_per_audience', 'orders_per_audience', 'transactors_per_audience']
            selected_metric = st.selectbox("Select Metric", metric_options)
            
            control_data = df_filtered[df_filtered['data_set'] == control_group][selected_metric]
            test_data = df_filtered[df_filtered['data_set'] == test_group][selected_metric]
            
            # Perform statistical test (t-test)
            t_stat, p_value = stats.ttest_ind(control_data, test_data, equal_var=False, nan_policy='omit')
            
            st.write(f"### {selected_metric.replace('_', ' ').title()} Comparison")
            st.write(f"Control Mean: {control_data.mean():.4f}")
            st.write(f"Test Mean: {test_data.mean():.4f}")
            st.write(f"T-Statistic: {t_stat:.4f}")
            st.write(f"P-Value: {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("Statistically Significant Difference Detected! 🚀")
            else:
                st.info("No Significant Difference Detected.")
