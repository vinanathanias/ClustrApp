import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta
from chart import (revenue_by_purchase_type, 
                   purchase_type_proportion, 
                   sales_over_time, 
                   top_products_by_sales, 
                   average_order_value,
                   monthly_active_customers,
                   repeat_purchase_rate,
                   top_products_by_sales1
                   )
###### Streamlit page setup #####
st.set_page_config(page_title="Clustering App", 
                   page_icon=":material/scatter_plot:", 
                   layout="wide")

#### FUNCTIONS ####

# Caching K-Means calculations to avoid rerunning on unrelated changes
@st.cache_data(show_spinner=False)
def calculate_kmeans(data_scaled, columns, k_values):
    distortions = []
    silhouette_scores = []
    kmeans_models = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data_scaled[columns])
        kmeans_models[k] = kmeans
        distortions.append(kmeans.inertia_)
        if k > 1:
            silhouette = silhouette_score(data_scaled[columns], kmeans.labels_)
            silhouette_scores.append(silhouette)

    return distortions, silhouette_scores, kmeans_models


# Caching month-specific filtering
@st.cache_data(show_spinner=False)
def filter_by_month(df_rfm, selected_month):
    return df_rfm[df_rfm['month_year_end'] == selected_month]


# Function to calculate the transition matrix
def calculate_transition_matrix(df_rfm, rfm_columns, num_clusters, monthly_kmeans_models): 
    # Get sorted unique months
    months_sorted = sorted(df_rfm['month_year_end'].unique())
    transition_matrix = np.zeros((num_clusters, num_clusters))  # Initialize transition matrix

    # Iterate through consecutive months
    for i in range(len(months_sorted) - 1):
        # Filter data for consecutive months
        month_data_current = filter_by_month(df_rfm, months_sorted[i])
        month_data_next = filter_by_month(df_rfm, months_sorted[i + 1])

        if not month_data_current.empty and not month_data_next.empty:
            # Perform clustering using pre-trained KMeans models
            kmeans = monthly_kmeans_models[num_clusters]  # Get the model for the current num_clusters
            month_data_current['Cluster'] = kmeans.predict(month_data_current[rfm_columns])
            month_data_next['Cluster'] = kmeans.predict(month_data_next[rfm_columns])

            # Calculate transitions
            for cluster_from in range(num_clusters):
                # Customers in the current cluster
                customers_in_cluster = month_data_current[month_data_current['Cluster'] == cluster_from]['CustomerID']
                count_from = len(customers_in_cluster)

                if count_from > 0:  # Avoid division by zero
                    for cluster_to in range(num_clusters):
                        # Count how many transitioned to the target cluster
                        count_to = len(month_data_next[
                            (month_data_next['CustomerID'].isin(customers_in_cluster)) &
                            (month_data_next['Cluster'] == cluster_to)
                        ])
                        transition_matrix[cluster_from, cluster_to] += count_to / count_from

    # Average the transition probabilities across all transitions (months)
    if len(months_sorted) > 1:
        transition_matrix /= (len(months_sorted) - 1)
    return transition_matrix
    

### LAYOUT ###
st.header("Clustering App", divider="blue", anchor=False)

# Data selection
existing_data_path = "filtered_data.csv"
existing_data = pd.read_csv(existing_data_path)

# Expected columns for the dataset
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", 
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]

data_option = st.selectbox("Choose Dataset", ["Select", "Use existing data", "Upload new data"])
df = None

if data_option == "Use existing data":
    df = existing_data
elif data_option == "Upload new data":
    uploaded_file = st.file_uploader("Upload Dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file else None
        # Validation

        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            st.error(f"Dataset does not match the required format."
                       f"Missing columns: {missing_columns or 'None'}")
            df = None
else:
    df = None

### DATA PREP ###
if df is not None:
    st.dataframe(df, use_container_width=True)

    # Create segmented control
    selected_section = st.segmented_control(
        "Select Section",
        ["About Dataset", "Silhouette Score & Elbow Method", "K-Means Clustering"],
        default="About Dataset"
    )

    if selected_section == "About Dataset":

        ##========= EDA ==========##
        st.subheader("About Dataset", anchor=False)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            ###### Monthly Active Customers ######
            st.write("Monthly Active Customers")
            active_customers_df = monthly_active_customers(df)
            # st.line_chart(active_customers_df.set_index('InvoiceDate')['ActiveCustomers'])
            # Create Altair line chart with bullet points
            line_chart = alt.Chart(active_customers_df).mark_line().encode(
                x=alt.X('InvoiceDate:T', title='Month'),
                y=alt.Y('ActiveCustomers:Q', title='Active Customers')
            ).properties(
                width=700,
                height=400
            )

            # Add bullet points for the data points
            points = alt.Chart(active_customers_df).mark_point(filled=True, size=100).encode(
                x='InvoiceDate:T',
                y='ActiveCustomers:Q',
                tooltip=['InvoiceDate:T', 'ActiveCustomers:Q']  # Add tooltips for interactivity
            )

            # Combine the line chart and points
            final_chart = line_chart + points

            # Render in Streamlit
            st.altair_chart(final_chart, use_container_width=True)

            ##### Sales over time ####
            st.write("Sales Over Time")
            sales_df = sales_over_time(df)
            # st.line_chart(sales_df.set_index('Month')['Revenue'])
            line_chart =alt.Chart(sales_df).mark_line(color='#FDC04D').encode(
                x=alt.X('Month:T', title='Month'),
                y=alt.Y('Revenue:Q', title='Total Revenue'),
            ).properties(
                width=700,
                height=400
            )

            #Add bullet points for the data points
            points = alt.Chart(sales_df).mark_point(filled=True, size=100, color='#FDC04D').encode(
                x='Month:T',
                y='Revenue:Q',
                tooltip=['Month:T', 'Revenue:Q']  # Add tooltips for interactivity
            )
            # Combine the line chart and points
            final_chart = line_chart + points

            # Render in Streamlit
            st.altair_chart(final_chart, use_container_width=True)

        with col2:
            # Proportion of Single Item vs Multi Item Purchases
            st.write("Proportion of Single Item vs Multi Item Purchases")
            proportion_df = purchase_type_proportion(df)
            # st.bar_chart(proportion_df.set_index('Purchase Type')['Percentage'])

            # Create Altair bar chart
            bar_chart = alt.Chart(proportion_df).mark_bar().encode(
                x=alt.X('Purchase Type:O', title='Purchase Type', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Percentage:Q', title='Percentage'),
                color=alt.Color('Purchase Type:N', scale=alt.Scale(scheme='viridis'))  # Custom bar colors  scale=alt.Scale(range=['#FACD64', '#51A2F8'])
            ).properties(
                width=600,
                height=400
            )

            st.altair_chart(bar_chart, use_container_width=True)

            # Top products by sales volume
            st.write("Top Products by Sales Volume")
            top_products_df = top_products_by_sales(df)
            # st.write(top_products_df)

            # Create Altair bar chart
            bar_chart = alt.Chart(top_products_df).mark_bar().encode(
                x=alt.X('Description:O', title='Description', axis=alt.Axis(labelAngle=0), sort="-y"),
                y=alt.Y('Sales:Q', title='Sales'),
                color=alt.Color('Description:N', scale=alt.Scale(scheme='viridis'))  # Custom bar colors
            ).properties(
                width=600,
                height=400
            )

            st.altair_chart(bar_chart, use_container_width=True)


    ### RFM (Monthly Only) ###
    def calculate_rfm(df):
        # df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
        df['month_end'] = df['InvoiceDate'] + pd.offsets.MonthEnd(0)
        df['month_year_end'] = df['month_end'].dt.strftime('%Y-%m')
        df['recency'] = (df['month_end'] - df['InvoiceDate']).dt.days
        df['monetary'] = df['UnitPrice'] * df['Quantity']
        df['frequency'] = df.groupby(['CustomerID', 'month_end'])['InvoiceNo'].transform('count')

        return df.groupby(['CustomerID', 'month_year_end']).agg(
            recency=('recency', 'mean'),
            monetary=('monetary', 'mean'),
            frequency=('frequency', 'sum')
        ).reset_index()

    # Monthly RFM Data
    monthly_data = calculate_rfm(df)

    def scale_rfm(df, columns):
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    # Scale data
    monthly_data_scaled = scale_rfm(monthly_data.copy(), ['recency', 'frequency', 'monetary'])

    ### K-MEANS ###
    k_values = range(2, 11)

    # Calculate K-Means models
    rfm_columns = ['recency', 'frequency', 'monetary']
    distortions, silhouette_scores, monthly_kmeans_models = calculate_kmeans(monthly_data_scaled, rfm_columns, k_values)

    if selected_section == "Silhouette Score & Elbow Method":
            ##========= SILHOUETTE & ELBOW ==========##
            st.subheader("SILHOUETTE & ELBOW", divider="orange", anchor=False)

            # Elbow Plot
            tab1, tab2 = st.tabs(["📈 Elbow Plot", "💯 Silhouette Scores"])
            with tab1:
                elbow_chart = alt.Chart(pd.DataFrame({
                    'K': list(k_values),
                    'Distortion': distortions
                })).mark_line(point=True).encode(
                    x=alt.X('K:O', title='Number of Clusters (K)', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Distortion:Q', title='Distortion (Inertia)')
                ).properties(title="Elbow Method")
                st.altair_chart(elbow_chart, use_container_width=True)

            # Silhouette Scores
            with tab2:
                silhouette_df = pd.DataFrame({
                    'K': list(k_values),
                    'Silhouette Score': silhouette_scores
                })
                st.dataframe(silhouette_df, width=400, hide_index=True)

    if selected_section == "K-Means Clustering":
        ##========= CLUSTERING (K-MEANS)==========##
        st.subheader("CLUSTERING", anchor=False)

        # Select number of clusters
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

        # # Summary tab
        # tab1 = st.tabs(["Summary"])

        # with tab1[0]:
        # Assign clusters to the monthly data
        kmeans = monthly_kmeans_models[num_clusters]
        monthly_data['Cluster'] = kmeans.predict(monthly_data_scaled[rfm_columns])

        # st.dataframe(monthly_data)

        # Compute the distribution of clusters
        cluster_distribution = monthly_data['Cluster'].value_counts().reset_index()
        cluster_distribution.columns = ['Cluster', 'Count']
        cluster_distribution = cluster_distribution.sort_values('Cluster')

        # Calculate the average RFM values for each cluster
        cluster_averages = monthly_data.groupby('Cluster')[rfm_columns].mean().reset_index()

        # Merge the distribution counts into the cluster averages
        cluster_averages = cluster_averages.merge(
                cluster_distribution,
                on='Cluster',
                how='left'
            )
        cluster_averages.rename(columns={'Count': 'Cluster Distribution'}, inplace=True)

        st.subheader("Cluster Averages with Distribution", divider="grey", anchor=False)
        st.dataframe(cluster_averages, hide_index=True, use_container_width=True)

        # Create columns for layout
        coll1, coll2 = st.columns(2, gap="large")

        with coll1:
            st.subheader("Clusters Mapping", divider="grey", anchor=False)

            scatter_3d = px.scatter_3d(
                    monthly_data,
                    x='recency',
                    y='frequency',
                    z='monetary',
                    color='Cluster',
                    hover_data=['CustomerID', 'recency', 'frequency', 'monetary', 'Cluster'],
                    title=f"3D Clusters with K={num_clusters} (Monthly)",
                    color_continuous_scale=px.colors.sequential.Plasma
            )
            scatter_3d.update_layout(
                    scene=dict(
                        xaxis_title='Recency',
                        yaxis_title='Frequency',
                        zaxis_title='Monetary'
                ),
                margin=dict(l=40, r=40, b=10, t=40)
            )
            st.plotly_chart(scatter_3d, use_container_width=True)

            with coll2:
                ##========= TRANSITION MATRIX ==========##
                st.subheader("Transition Matrix", divider="grey", anchor=False)

                # Calculate the transition matrix
                transition_matrix = calculate_transition_matrix(monthly_data, rfm_columns, num_clusters, monthly_kmeans_models)

                # Convert the transition matrix into a DataFrame for better readability
                transition_matrix_df = pd.DataFrame(
                    transition_matrix,
                    columns=[f'Cluster {i}' for i in range(num_clusters)],
                    index=[f'Cluster {i}' for i in range(num_clusters)]
                )

                # st.dataframe(transition_matrix_df)

                # Create the heatmap
                heatmap = px.imshow(
                    transition_matrix_df,
                    labels=dict(x="To Cluster", y="From Cluster", color="Probability"),
                    x=transition_matrix_df.columns,
                    y=transition_matrix_df.index,
                    color_continuous_scale="Inferno",
                    text_auto=".4f",
                )

                heatmap.update_layout(title="Transition Matrix Heatmap", title_x=0.5)

                # Display the heatmap in Streamlit
                st.plotly_chart(heatmap, use_container_width=True)

            # Aggregating RFM values by month and cluster
            monthly_cluster_rfm = monthly_data.groupby(['month_year_end', 'Cluster'])[rfm_columns].mean().reset_index()
            # st.dataframe(monthly_cluster_rfm)

            # Plotting Recency over time for each cluster
            recency_chart = alt.Chart(monthly_cluster_rfm).mark_line().encode(
                x=alt.X('month_year_end:T', title='Month'),
                y=alt.Y('recency:Q', title='Recency'),
                color=alt.Color('Cluster:N', title='Cluster'),
                tooltip=['month_year_end:T', 'recency:Q', 'Cluster:N']
            ).properties(
                title='Recency over Time by Cluster',
                width=700,
                height=400
            )

            # Plotting Frequency over time for each cluster
            frequency_chart = alt.Chart(monthly_cluster_rfm).mark_line().encode(
                x=alt.X('month_year_end:T', title='Month'),
                y=alt.Y('frequency:Q', title='Frequency'),
                color=alt.Color('Cluster:N', title='Cluster'),
                tooltip=['month_year_end:T', 'frequency:Q', 'Cluster:N']
            ).properties(
                title='Frequency over Time by Cluster',
                width=700,
                height=400
            )

            # Plotting Monetary over time for each cluster
            monetary_chart = alt.Chart(monthly_cluster_rfm).mark_line().encode(
                x=alt.X('month_year_end:T', title='Month'),
                y=alt.Y('monetary:Q', title='Monetary'),
                color=alt.Color('Cluster:N', title='Cluster'),
                tooltip=['month_year_end:T', 'monetary:Q', 'Cluster:N']
            ).properties(
                title='Monetary over Time by Cluster',
                width=700,
                height=400
            )
        ##### RFM Metrics over Time #####
        st.subheader("RFM Metrics over Time", divider="grey", anchor=False)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.altair_chart(recency_chart, use_container_width=True)

        with col2:
            st.altair_chart(frequency_chart, use_container_width=True)

        with col3:
            st.altair_chart(monetary_chart, use_container_width=True)
       
        # # Calculate the difference for each metric (recency, frequency, monetary) over time for each cluster
        # monthly_cluster_rfm['recency_diff'] = monthly_cluster_rfm.groupby('Cluster')['recency'].diff()
        # monthly_cluster_rfm['frequency_diff'] = monthly_cluster_rfm.groupby('Cluster')['frequency'].diff()
        # monthly_cluster_rfm['monetary_diff'] = monthly_cluster_rfm.groupby('Cluster')['monetary'].diff()

        # # Now you can analyze trends for each cluster
        # for cluster in monthly_cluster_rfm['Cluster'].unique():
        #     cluster_data = monthly_cluster_rfm[monthly_cluster_rfm['Cluster'] == cluster]
            
        #     cluster_output = []
            
        #     # Recency trend interpretation
        #     if cluster_data['recency_diff'].mean() > 0:
        #         cluster_output.append(f"Cluster {cluster} is becoming less recent over time.")
        #     elif cluster_data['recency_diff'].mean() < 0:
        #         cluster_output.append(f"Cluster {cluster} is becoming more recent over time.")
        #     else:
        #         cluster_output.append(f"Cluster {cluster} shows no significant change in recency over time.")
            
        #     # Frequency trend interpretation
        #     if cluster_data['frequency_diff'].mean() > 0:
        #         cluster_output.append(f"Cluster {cluster} is increasing its frequency over time.")
        #     elif cluster_data['frequency_diff'].mean() < 0:
        #         cluster_output.append(f"Cluster {cluster} is decreasing its frequency over time.")
        #     else:
        #         cluster_output.append(f"Cluster {cluster} shows no significant change in frequency over time.")
            
        #     # Monetary trend interpretation
        #     if cluster_data['monetary_diff'].mean() > 0:
        #         cluster_output.append(f"Cluster {cluster} is increasing its spending over time.")
        #     elif cluster_data['monetary_diff'].mean() < 0:
        #         cluster_output.append(f"Cluster {cluster} is decreasing its spending over time.")
        #     else:
        #         cluster_output.append(f"Cluster {cluster} shows no significant change in spending over time.")
            
        #     # Display output in Streamlit or print
        #     st.write("\n".join(cluster_output))



else:
    st.warning("Please select a dataset to continue.")
