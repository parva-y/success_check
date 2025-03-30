import streamlit as st
import pandas as pd
import scipy.stats as stats
import numpy as np
from statsmodels.stats.weightstats import ztest
from scipy.stats import ks_2samp
import ruptures as rpt

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
        
        # Define control group
        control_group = "Control Set"
        
        # Prepare results table
        all_results = []
        summary_results = []
        
        # Iterate over all cohorts and test groups
        for cohort in df['cohort'].unique():
            df_filtered = df[df['cohort'] == cohort]
            test_groups = df_filtered['data_set'].unique()
            if control_group not in test_groups:
                continue  # Skip if no control group exists
            
            for test_group in test_groups:
                if test_group == control_group:
                    continue
                
                for metric in ['gmv_per_audience', 'app_opens_per_audience', 'orders_per_audience', 'transactors_per_audience']:
                    df_control = df_filtered[df_filtered['data_set'] == control_group].groupby('date')[metric].mean()
                    df_test = df_filtered[df_filtered['data_set'] == test_group].groupby('date')[metric].mean()
                    
                    # Align dates for paired testing
                    df_combined = pd.concat([df_control, df_test], axis=1, keys=[control_group, test_group]).dropna()
                    control_values = df_combined[control_group]
                    test_values = df_combined[test_group]
                    
                    # Perform statistical tests
                    t_stat, p_value_ttest = stats.ttest_rel(control_values, test_values)
                    u_stat, p_value_mw = stats.mannwhitneyu(control_values, test_values, alternative='two-sided')
                    z_stat, p_value_ztest = ztest(control_values, test_values)
                    ks_stat, p_value_ks = ks_2samp(control_values, test_values)
                    algo = rpt.Pelt(model="l2").fit(control_values.values - test_values.values)
                    change_points = algo.predict(pen=1)
                    
                    # Store results
                    tests = [
                        ("Paired t-test", t_stat, p_value_ttest),
                        ("Mann-Whitney U Test", u_stat, p_value_mw),
                        ("Z-Test", z_stat, p_value_ztest),
                        ("Kolmogorov-Smirnov Test", ks_stat, p_value_ks),
                        ("Change Point Detection", len(change_points)-1, np.nan)  # Use np.nan for non-numeric values
                    ]
                    
                    for test_name, stat, p_value in tests:
                        significance = "Pass" if (not pd.isna(p_value) and p_value < 0.05) else "Fail"
                        all_results.append([cohort, test_group, metric, control_values.mean(), test_values.mean(), test_name, stat, p_value])
                        summary_results.append([cohort, test_name, significance])
        
        # Display detailed results
        if all_results:
            results_df = pd.DataFrame(all_results, columns=["Cohort", "Test Group", "Metric", "Control Mean", "Test Mean", "Test", "Statistic", "P-Value"])
            st.write("### Detailed Experiment Results Table")
            
            # Apply conditional formatting only to numeric p-values
            def highlight_significant(s):
                return ['background-color: lightgreen' if (not pd.isna(v) and v < 0.05) else '' for v in s]
            
            styled_df = results_df.style.apply(highlight_significant, subset=["P-Value"])
            st.dataframe(styled_df)
        else:
            st.write("No valid test-control comparisons found.")
        
        # Display summary results
        if summary_results:
            summary_df = pd.DataFrame(summary_results, columns=["Cohort", "Test Name", "Pass/Fail"]).drop_duplicates()
            st.write("### Summary Table")
            st.dataframe(summary_df)
