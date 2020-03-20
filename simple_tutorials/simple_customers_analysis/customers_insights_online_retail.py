# Some typical syntax picked from the tutorial 
# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot
import tensorflow as tf
tf.enable_eager_execution()

import pandas as pd
import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from collections import Counter 

import math

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

# For reference, starting header will look like this 
#['Invoice' 'StockCode' 'Description' 'Quantity' 'InvoiceDate' 'Price' 'Customer ID' 'Country']

# Load data
# Need to enforce encoding as described here https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python 
df = pd.read_csv('online_retail_II_2011.csv', engine='python')
# Here I want to clean up some information from the InvoiceDate column (don't plan to use time and year, just day and month)
df['InvoiceDate'] = df['InvoiceDate'].str.slice(3, -6)

print("Prining head of the file to see how it looks")
print(df.head())
print("Prining data types")
print(df.dtypes)

# Very first step, remove lines with incomplete data (e.g. missing Customer IDs). 
df = df.dropna()

# Now that the data is loaded, here's a set of functions needed to arrange the data as needed for further inspection 

# Group data based on a Series (result in two Series, one stays untouched, the other has content based on merging of original unmerged cells)
# Simple syntax from here https://stackoverflow.com/questions/46636080/merge-rows-within-a-group-together 
def get_grouped_data(dataframe,grouping_feature,manipulated_data):
    grouped_data = dataframe.groupby(grouping_feature, as_index=False).agg({manipulated_data : ' '.join})  
    return grouped_data

def get_grouped_average(dataframe,grouping_feature,manipulated_data):
    grouped_data = df.groupby(grouping_feature)[manipulated_data].mean().reset_index(name=manipulated_data)
    return grouped_data

def get_grouped_average_multiplied(dataframe,grouping_feature,manipulated_data,manipulated_data_second):
    dataframe['ValueM'] = manipulated_data.abs() * manipulated_data_second.abs()
    grouped_data = dataframe.groupby(grouping_feature)['ValueM'].mean().reset_index(name='ValueM')
    return grouped_data

def get_grouped_sum(dataframe,grouping_feature,manipulated_data):
    grouped_data = df.groupby(grouping_feature)[manipulated_data].sum().reset_index(name=manipulated_data)
    return grouped_data

def get_grouped_sum_multiplied(dataframe,grouping_feature,manipulated_data,manipulated_data_second):
    dataframe['ValueM'] = manipulated_data.abs() * manipulated_data_second.abs()
    grouped_data = dataframe.groupby(grouping_feature)['ValueM'].sum().reset_index(name='ValueM')
    return grouped_data

def get_counting(dataframe,grouping_feature,manipulated_feature,title='count'): 
    grouped_data = dataframe.groupby(grouping_feature)[manipulated_feature].count().reset_index(name=title)
    return grouped_data

def group_two_columns(dataframe,first_feature,second_feature,manipulated_data,title): 
    grouped_data = dataframe.groupby([first_feature,second_feature])[manipulated_data].sum().reset_index(name=title)
    return grouped_data

# Per item averages
# Item cost average vs invoice 
average_item_price_per_invoice = get_grouped_average(df,'Invoice','Price')
# Item cost average vs customer  
average_item_price_per_customer = get_grouped_average(df,'Customer ID','Price')
# Item cost average vs country  
average_item_price_per_country = get_grouped_average(df,'Country','Price')

# Per item sums 
# Per Item cost sum vs invoice 
sum_item_price_per_invoice = get_grouped_sum(df,'Invoice','Price')
# Per Item cost sum vs customer
sum_item_price_per_customer = get_grouped_sum(df,'Customer ID','Price')
# Per Item cost sum vs country
sum_item_price_per_country = get_grouped_sum(df,'Country','Price')

# Averages orders cost 
# Averages orders cost vs invoice 
average_items_price_per_invoice = get_grouped_average_multiplied(df,'Invoice',df.Price,df.Quantity)
# Averages orders cost vs customer 
average_items_price_per_customer = get_grouped_average_multiplied(df,'Customer ID',df.Price,df.Quantity)
# Averages orders cost vs country 
average_items_price_per_country = get_grouped_average_multiplied(df,'Country',df.Price,df.Quantity)

# Sums of orders cost 
# Sums orders cost vs invoice 
sum_items_price_per_invoice = get_grouped_sum_multiplied(df,'Invoice',df.Price,df.Quantity)
# Sums orders cost vs customers 
sum_items_price_per_customer = get_grouped_sum_multiplied(df,'Customer ID',df.Price,df.Quantity)
# Sums orders cost vs country 
sum_items_price_per_country = get_grouped_sum_multiplied(df,'Country',df.Price,df.Quantity)

# Counts 
counting_purchases_per_user = get_counting(df,'Customer ID','Invoice','Invoices per customers') 
# Counts of oderes per country 
counting_purchases_per_country = get_counting(df,'Country','Invoice','Invoices per country') 
# Counts of customers per country  
counting_users_per_country = get_counting(df,'Country','Customer ID','Customers per country') 
# Counts of oderes per day
counting_users_per_month = get_counting(df,'InvoiceDate','Invoice','Invoices per month') 


# Plotting (simple plots). 

# Based on what can be seen here http://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot
def make_chart_plot(dataframe,x_axis_name,do_log_y,lines_coord,plot_title,plot_kind): 
    dataframe.plot(kind=plot_kind,x=x_axis_name,logy=do_log_y,color='skyblue',lw=0.5,ec='black')
    plt.axhline(y=10,color='gray',linestyle='--',linewidth=0.5)
    for line in lines_coord : 
        plt.axhline(y=line,color='gray',linestyle='--',linewidth=0.5)
    plt.show()
    plt.tight_layout()
    plt.savefig(plot_title)
    plt.clf()
    plt.close()

def make_density_plot_hist(dataframe_feature,is_density,feature_to_plot,n_bins,plot_title,range_plot,is_log_x,is_log_y,x_axis_name,y_axis_name,is_grid):
    dataframe_feature.plot.hist(bins=n_bins,density=is_density,range=range_plot,color='skyblue',lw=0.5,ec='black')
    if is_log_x: 
        plt.xscale('log')
    if is_log_y:
        plt.yscale('log')        
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    if is_grid: 
        plt.grid(True)
    plt.show()
    plt.savefig(plot_title)
    plt.clf()
    plt.close()

# Reference https://chrisalbon.com/python/data_visualization/matplotlib_scatterplot_from_pandas/
def make_scatter_plot(dataframe,first_feature,second_feature,third_feature,z_value,plot_title,scaling_factor): 
    tick_size=8
    colors = np.random.rand(298)
    dataframe[z_value]=scaling_factor*dataframe[z_value].astype(float)
    ax = plt.axes()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_size) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_size) 
    plt.scatter(first_feature,second_feature,dataframe.ValueM,alpha=0.5,c=colors,lw=0.0)
    plt.xticks(rotation=90)
    plt.show()
    plt.tight_layout()
    plt.savefig(plot_title)
    plt.clf()
    plt.close()

print('\n Starting EDA \n _____________ \n')

# Basic info from per-item averages. Part-1
make_density_plot_hist(average_item_price_per_invoice.Price,True,average_item_price_per_invoice.Price,50,'average_item_price_per_invoice.png', 
                       [0, 150.0],True,True,'Average item price per invoice','Norm. to unity',True)
make_density_plot_hist(average_item_price_per_customer.Price,True,average_item_price_per_invoice.Price,50,'average_item_price_per_customer.png', 
                       [0, 150.0],True,True,'Average item price per invoice','Norm. to unity',True)
make_chart_plot(average_item_price_per_country,'Country',True,[10,100],'average_item_price_per_country.png','bar')  

# Basic info from per-item sums. Part-2
make_density_plot_hist(sum_item_price_per_invoice.Price,True,sum_item_price_per_invoice.Price,50,'sum_item_price_per_invoice.png', 
                       [0, 150.0],True,True,'Sum items price per invoice','Norm. to unity',True)
make_density_plot_hist(sum_item_price_per_customer.Price,True,sum_item_price_per_invoice.Price,50,'sum_item_price_per_customer.png', 
                       [0, 150.0],True,True,'Sum items price per invoice','Norm. to unity',True)
make_chart_plot(sum_item_price_per_country,'Country',True,[10,100,1000,10000],'sum_item_price_per_country.png','bar')  

# Basic info from average price. Part-3
make_density_plot_hist(average_items_price_per_invoice.ValueM,True,average_items_price_per_invoice.ValueM,50,'average_items_price_per_invoice.png',
                       [0, 150.0],True,True,'Sum items price per invoice','Norm. to unity',True)                                                             
make_density_plot_hist(average_items_price_per_customer.ValueM,True,average_items_price_per_customer.ValueM,50,'average_items_price_per_customer.png',
                       [0, 150.0],True,True,'Sum items price per invoice','Norm. to unity',True)                                                             
make_chart_plot(average_items_price_per_country,'Country',True,[10,50,100,],'average_items_price_per_country.png','bar')   

# Basic info from total price. Part-4
make_density_plot_hist(sum_items_price_per_invoice.ValueM,True,sum_items_price_per_invoice.ValueM,50,'sum_items_price_per_invoice.png',
                       [0, 150.0],True,True,'Sum items price per invoice','Norm. to unity',True)                                                             
make_density_plot_hist(sum_items_price_per_customer.ValueM,True,sum_items_price_per_customer.ValueM,50,'sum_items_price_per_customer.png',
                       [0, 150.0],True,True,'Sum items price per customer','Norm. to unity',True)                                                          
make_chart_plot(sum_items_price_per_country,'Country',True,[10,100,1000,10000],'sum_items_price_per_country.png','bar')   

# Obtain basic information out of counts data. Part-5
print('Printing info from counting_purchases_per_user')
print(counting_purchases_per_user['Invoices per customers'].describe())
make_chart_plot(counting_purchases_per_country,'Country',True,[10,100,1000,10000,100000],'purchases_per_country.png','bar')
make_chart_plot(counting_users_per_country,'Country',True,[10,100,1000,10000,100000],'users_per_country.png','bar')
make_chart_plot(counting_users_per_month,'InvoiceDate',False,[10000,20000,30000,40000,50000,60000,70000],'users_per_month.png','bar')

# Grouping by multiple columns for scatter plots  
counting_purchases_per_country_vs_time = get_counting(df,['Country','InvoiceDate'],'Invoice') 
sum_items_price_country_vs_time = get_grouped_sum_multiplied(df,['Country','InvoiceDate'],df.Price,df.Quantity)
average_items_price_country_vs_time = get_grouped_average_multiplied(df,['Country','InvoiceDate'],df.Price,df.Quantity)

counting_purchases_per_country_vs_time_grouped = group_two_columns(counting_purchases_per_country_vs_time,'Country','InvoiceDate','count','ValueM')
sum_items_price_country_vs_time_grouped = group_two_columns(sum_items_price_country_vs_time,'Country','InvoiceDate','ValueM','ValueM')
average_items_price_country_vs_time_grouped = group_two_columns(sum_items_price_country_vs_time,'Country','InvoiceDate','ValueM','ValueM')

print(counting_purchases_per_country_vs_time.head())
print(sum_items_price_country_vs_time.head())
print(average_items_price_country_vs_time.head())

print(counting_purchases_per_country_vs_time_grouped.head())
print(sum_items_price_country_vs_time_grouped.head())
print(average_items_price_country_vs_time_grouped.head())

make_scatter_plot(counting_purchases_per_country_vs_time_grouped,counting_purchases_per_country_vs_time_grouped.InvoiceDate,counting_purchases_per_country_vs_time_grouped.Country,counting_purchases_per_country_vs_time_grouped.ValueM,'ValueM','counting_purchases_per_country_vs_time_grouped.png',0.04)
make_scatter_plot(sum_items_price_country_vs_time_grouped,sum_items_price_country_vs_time_grouped.InvoiceDate,sum_items_price_country_vs_time_grouped.Country,sum_items_price_country_vs_time_grouped.ValueM,'ValueM',"sum_items_price_country_vs_time_grouped.png",0.0011)
make_scatter_plot(average_items_price_country_vs_time_grouped,average_items_price_country_vs_time_grouped.InvoiceDate,average_items_price_country_vs_time_grouped.Country,average_items_price_country_vs_time_grouped.ValueM,'ValueM',"average_items_price_country_vs_time_grouped.png",0.0011)


print('\n Ending EDA \n _____________ \n')


print('\n Starting: Split the customers into groups according to their purchase patterns and product purchases, and characterize/quantify the obtained customer personas \n _____________ \n')

# Functionalities to be used
def get_most_common_words(dataframe_series,max_words):
    splitted_string = dataframe_series.str.split() 
    most_occur = Counter(dataframe_series).most_common(max_words)  
    print(most_occur)
    return most_occur

def add_most_common_words(dataframe,max_words,name_ranked_col): 
    # Accessing the hardcoded thingy 
    dataframe_ranked = dataframe.groupby("Customer ID")["Description"].apply(lambda x: Counter(" ".join(x).split()).most_common(max_words)[0][0]).reset_index(name=name_ranked_col)
    return dataframe_ranked 

# Build dataframe with two columns, customer ID and description of purchases 
grouped_data_user_descriptions = get_grouped_data(df,'Customer ID','Description')

# Now obtain dataframe with first, second, third most frequent words
dataframe_first_ranked_words = add_most_common_words(grouped_data_user_descriptions,1,'first_ranked_words')

# Ideally here I'd try to get rid of words which are not useful (e.g., 'of', 'and', ..)
print('\nTop-15 first top ranked words across all customers')
first_list = get_most_common_words(dataframe_first_ranked_words.first_ranked_words,15)

print('\n Ending: Split the customers into groups according to their purchase patterns and product purchases, and characterize/quantify the obtained customer personas \n _____________ \n')

print('\n Starting: Customers churn \n _____________ \n')

# Here want to look for distinct customers ID before a certain date and look if any of them drops after a certain date, 
# first let's keep the interesting part of the dataset 
# Syntax for keepping other columns when doing a group by 
# https://stackoverflow.com/questions/49783178/python-keep-other-columns-when-using-sum-with-groupby

def cutstomer_churn_data(dataframe,grouping_feature):
    grouped_data = dataframe
    #grouped_data = grouped_data.drop(['Description', 'Quantity','StockCode','Price','Country','ValueM'], axis=1)
    #grouped_data['ValueM'] = manipulated_data * manipulated_data_second    
    grouped_data = grouped_data.drop(['Description', 'Quantity','StockCode','Price','Country'], axis=1)
    grouped_data = grouped_data.groupby(['Invoice'], as_index=False).agg({'InvoiceDate': 'first', 'Customer ID': 'first'})
    return grouped_data


dataframe_customers_churn = cutstomer_churn_data(df,'Invoice')
# Convert InvoiceDate to numeric and perform selection on dataset to split it in first and second half year invoices. 
# Alternative solution .apply(pd.to_numeric) 
# Customers in first six months 
# Select 2011 and 2010
dataframe_2011_first = dataframe_customers_churn[dataframe_customers_churn['InvoiceDate'].apply(lambda x: int(x.split('/')[0])<6 and int(x.split('/')[1])==2011  or int(x.split('/')[1])==2010 )]
# Customers in second six months 
dataframe_2011_second = dataframe_customers_churn[dataframe_customers_churn['InvoiceDate'].apply(lambda x: int(x.split('/')[0])>=6 and int(x.split('/')[1])==2011 )]

# Get customaers ids in form of lists
cust_2011_first = dataframe_2011_first['Customer ID'].to_numpy()
cust_2011_second = dataframe_2011_second['Customer ID'].to_numpy()
#print(cust_2011_first)
#print(cust_2011_second)

# Solution from here https://www.geeksforgeeks.org/python-get-unique-values-list/
def unique(list1): 
    # intilize a null list 
    unique_list = [] 
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not         
        if x not in unique_list and not math.isnan(x): 
            unique_list.append(x) 
    return unique_list

cust_2011_first_unq=unique(cust_2011_first)
print('Numer of customers in first half od 2011')
print(len(cust_2011_first_unq))
cust_2011_second_unq=unique(cust_2011_second)
print('Numer of customers in second half od 2011')
print(len(cust_2011_second_unq))

# Return elements missing in list 2 from list 1
def missing_elements(list1,list2):
    missing_element_list = []
    for element1 in list1: 
        found_element = False 
        for element2 in list2: 
            if found_element:
                continue
            if element2 == element1: 
                found_element = True 
        if not found_element:
            missing_element_list.append(element1)
    #for missing in  missing_element_list:
    #    print(missing)
    return missing_element_list 

churned_customers = missing_elements(cust_2011_first_unq,cust_2011_second_unq)
churned_customers_unique = unique(churned_customers)
print('Churned customers')
print(len(churned_customers_unique))
print(churned_customers_unique)

# Now that we have the missing elements we need add in the dataframe a customers churn data value in order to characterise them. 
# Use again a loop solution, maybe there is a better way
def add_churn_value_loop(dataframe,customers): 
    dataframe_out = dataframe
    for customer in customers: 
        #print(customer)
        for index, row in dataframe_out.iterrows():
            #print(index)
            if customer == dataframe_out['Customer ID'].iloc[index]:
                dataframe_out['Churned'] = 1.
            else : 
                dataframe_out['Churned'] = 0.


def add_churn_value(x,churned_customers):
    for index, row in df.iterrows():
        #print(index)        
        for customer in churned_customers:
            if customer == df['Customer ID'].iloc[index]:
                df['Churned'] = 1.
            else :
                df['Churned'] = 0.
    return df

#dataframe_customers_churn.apply(add_churn_value(dataframe_customers_churn,churned_customers))
#churned_customers.to_csv('out_final_dataset.csv', index=False) 
#dataframe_customers_churn_valued = add_churn_value(dataframe_customers_churn,churned_customers )
#print(dataframe_customers_churn_valued.head())            
