import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def revenue_by_purchase_type(df):
    # Convert 'InvoiceDate' to datetime if not already
    if df['InvoiceDate'].dtype != 'datetime64[ns]':
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Identify first-time and repeat purchases
    df_sorted = df.sort_values(by=['CustomerID', 'InvoiceDate'])
    df_sorted['is_first_purchase'] = df_sorted.groupby('CustomerID')['InvoiceNo'].transform(lambda x: x.duplicated(keep='first'))

    # Calculate revenue
    df_sorted['monetary'] = df_sorted['UnitPrice'] * df_sorted['Quantity']
    first_purchase_revenue = df_sorted[df_sorted['is_first_purchase'] == False]['monetary'].sum()
    repeat_purchase_revenue = df_sorted[df_sorted['is_first_purchase'] == True]['monetary'].sum()

    # Prepare data for bar chart
    summary_df = pd.DataFrame({
        'Purchase Type': ['First-Time Purchases', 'Repeat Purchases'],
        'Revenue': [first_purchase_revenue, repeat_purchase_revenue]
    })
    
    return summary_df

def purchase_type_proportion(df):
    # Calculate total quantity per purchase (InvoiceNo)
    quantity_per_purchase = df.groupby('InvoiceNo')['Quantity'].sum().reset_index()
    quantity_per_purchase['purchase_type'] = quantity_per_purchase['Quantity'].apply(lambda x: 'Single Item' if x == 1 else 'Multi Item')

    # Calculate proportions
    purchase_counts = quantity_per_purchase['purchase_type'].value_counts()
    purchase_proportions = (purchase_counts / purchase_counts.sum()) * 100

    # Prepare data for bar chart
    proportion_df = purchase_proportions.reset_index().round(2)
    proportion_df.columns = ['Purchase Type', 'Percentage']

    return proportion_df

def sales_over_time(df):
    # Convert 'InvoiceDate' to datetime if not already
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Group by month and calculate total revenue
    df['Month'] = df['InvoiceDate'].dt.to_period("M")
    sales_df = df.groupby('Month').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(name='Revenue')
    # Convert 'Month' to a proper datetime format for Altair
    sales_df['Month'] = sales_df['Month'].dt.start_time  # Use start of the month as datetime
    
    return sales_df

def top_products_by_sales(df, top_n=5):
    # Calculate total sales for each product
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    top_products_df = df.groupby('Description')['Sales'].sum().nlargest(top_n).reset_index()
    
    return top_products_df

def top_products_by_sales1(df, top_n=1):
    # Calculate total sales for each product
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    
    # Extract the Month-Year from InvoiceDate
    df['Month_Year'] = df['InvoiceDate'].dt.to_period('M')
    
    # Group by 'Month_Year' and 'Description' and sum the sales
    product_sales = df.groupby(['Month_Year', 'Description'])['Sales'].sum().reset_index()

    # Select the top product for each month
    top_products_df = product_sales.sort_values('Sales', ascending=False).groupby('Month_Year').head(top_n)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_products_df, x='Month_Year', y='Sales', hue='Description', palette='viridis')

    # Adding labels and title
    plt.title(f'Top 1 Product by Sales Each Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)

    return top_products_df



def average_order_value(df):
    # Ensure 'InvoiceDate' is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate revenue per order (InvoiceNo)
    df['OrderRevenue'] = df['Quantity'] * df['UnitPrice']
    aov_df = df.groupby(df['InvoiceDate'].dt.to_period("M")).apply(
        lambda x: x['OrderRevenue'].sum() / x['InvoiceNo'].nunique()
    ).reset_index(name='AOV')
    
    return aov_df

def monthly_active_customers(df):
    # Ensure 'InvoiceDate' is datetime
    # df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Calculate unique customers per month
    active_customers_df = df.groupby(df['InvoiceDate'].dt.to_period("M"))['CustomerID'].nunique().reset_index(name='ActiveCustomers')
    # Convert 'InvoiceDate' to datetime for Altair
    active_customers_df['InvoiceDate'] = active_customers_df['InvoiceDate'].dt.start_time  # Use start of the month
    
    return active_customers_df

def repeat_purchase_rate(df):
    # Ensure 'InvoiceDate' is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Count purchases per customer per month
    purchases_df = df.groupby([df['InvoiceDate'].dt.to_period("M"), 'CustomerID'])['InvoiceNo'].nunique().reset_index()
    
    # Determine if each customer is a repeat purchaser (more than 1 purchase)
    repeat_df = purchases_df.groupby('InvoiceDate').apply(
        lambda x: (x['InvoiceNo'] > 1).sum() / x['CustomerID'].nunique() * 100
    ).reset_index(name='RepeatPurchaseRate')
    
    return repeat_df
