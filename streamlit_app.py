import streamlit as st
import pandas as pd
import scipy.stats as stats
import numpy as np

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
            
            df_control = df_filtered[df_filtered['data_set'] == control_group].groupby('date')[selected_metric].mean()
            df_test = df_filtered[df_filtered['data_set'] == test_group].groupby('date')[selected_metric].mean()
            
            # Align dates for paired testing
            df_combined = pd.concat([df_control, df_test], axis=1, keys=[control_group, test_group]).dropna()
            control_values = df_combined[control_group]
            test_values = df_combined[test_group]
            
            # Perform statistical tests
            results = []
            
            # 1. Paired t-test (if data is normally distributed)
            t_stat, p_value_ttest = stats.ttest_rel(control_values, test_values)
            results.append(["Paired t-test", t_stat, p_value_ttest])
            
            # 2. Mann-Whitney U Test (if data is non-normal)
            u_stat, p_value_mw = stats.mannwhitneyu(control_values, test_values, alternative='two-sided')
            results.append(["Mann-Whitney U Test", u_stat, p_value_mw])
            
            # Display results in a structured format
            st.write(f"### {selected_metric.replace('_', ' ').title()} Comparison")
            st.write(f"**Control Mean:** {control_values.mean():.4f}")
            st.write(f"**Test Mean:** {test_values.mean():.4f}")
            
            st.write("### Statistical Test Results")
            results_df = pd.DataFrame(results, columns=["Test", "Statistic", "P-Value"])
            st.dataframe(results_df)
            
            # Interpretation of results
            for test_name, stat, p_val in results:
                if p_val < 0.05:
                    st.success(f"{test_name}: Statistically Significant Difference Detected! 🚀")
                else:
                    st.info(f"{test_name}: No Significant Difference Detected.")
