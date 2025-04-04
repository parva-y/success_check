# Fix 1: Update the prepare_plot_data function
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
        # For specific recency, filter by the selected recency first
        if has_recency_data and selected_recency != "Overview":
            filtered_df = base_df[base_df['Recency'] == selected_recency]
            return filtered_df
        else:
            return base_df

# Fix 2: Alternative approach - use df_filtered instead of base_filtered_df
for metric in metrics:
    # Use df_filtered which already has the recency filter applied
    plot_data = prepare_plot_data(df_filtered, metric, selected_recency)
    
    fig = px.line(plot_data, x='date', y=metric, color='data_set', title=metric.replace("_", " ").title())
    fig.update_traces(connectgaps=False)
    fig.update_xaxes(tickformat="%d/%m")

    for mark_date in test_marked_dates.get(selected_cohort, []):
        if mark_date in plot_data['date'].astype(str).values:
            fig.add_vline(x=mark_date, line_width=2, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)
