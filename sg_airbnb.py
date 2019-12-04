import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster
sns.set(style="darkgrid")

os.chdir('C:/Users/bmuljono/Desktop/Personal Stuff/Singapore Airbnb Data')

raw_df = pd.read_csv('listings.csv')

# Data preprocessing
## Remove unwanted columns
drop = ["last_review", "reviews_per_month", "id", "host_id", "host_name", "calculated_host_listings_count", "availability_365"]
raw_df.drop(drop, inplace=True, axis=1)
## Remove price outliers
q1 = raw_df[['price']].quantile(0.25)
q3 = raw_df[['price']].quantile(0.75)
IQR = q3 - q1
df = raw_df[~((raw_df < (q1 - 1.5 * IQR)) |(raw_df > (q3 + 1.5 * IQR))).any(axis=1)]
## Remove minimum nights > 365
df = df.drop(df[df.minimum_nights > 365].index)
## Assign risky column
q90 = float(df[['number_of_reviews']].quantile(0.9))
df['risk_rating'] = np.where((df['minimum_nights'] < 90) & (df['number_of_reviews'] < q90),
  'High risk', np.where((df['minimum_nights'] < 90) & (df['number_of_reviews'] >= q90),
                        "Medium risk", "Low risk"))
## Assign minimum nights >= 90 column
df['min_night_90_days'] = np.where(df['minimum_nights'] >= 90, '90 nights and above', 'Below 90 nights')

# EDA
sns.distplot(df["price"])
stats.kruskal(*[group["price"].values for name, group in df.groupby("neighbourhood_group")])
## Distribution of price over regions
plt.figure(figsize=(10,6))
sns.boxplot(y="price",x ='neighbourhood_group' ,data = df)
plt.title("Price distribution by region")
plt.show()

## Finding the distribution of listings by region
plt.figure(figsize=(10,6))
sns.scatterplot("longitude","latitude",hue="neighbourhood_group", data = df)
plt.title("Distribution of listings by region")
plt.show()

## Listing distribution through folium
m=folium.Map([1.38255,103.83580],zoom_start=11)
m.add_child(FastMarkerCluster(df[['latitude','longitude']].values.tolist()))
display(m)

## Review distribution by minimum nights
plt.figure(figsize=(10,6))
sns.scatterplot(x='minimum_nights',y='number_of_reviews',data=df)
plt.title("Distribution of reviews by minimum nights")

## Listing barplot distribution by region
plt.figure(figsize=(10,6))
sns.countplot(x = 'neighbourhood_group',hue = "risk_rating",data = df)
plt.title("Distribution of risky listings by region")
plt.show()

## Listing barplot distribution by room type
plt.figure(figsize=(10,6))
sns.countplot(x = 'risk_rating',hue = "room_type",data = df)
plt.title("Distribution of risky listings by room type")
plt.show()

## Top 5 neighbourhoods with most listings
df_plot = df.groupby(['neighbourhood', 'risk_rating']).size().reset_index().pivot(columns='risk_rating',index = 'neighbourhood', values=0)
df_plot['Total'] = df_plot.sum(axis = 1)
df_plot = df_plot.sort_values(by = ['Total'], ascending = False).head(5)
df_plot.drop(['Total'], inplace=True, axis=1)
df_plot.plot(kind = 'bar', stacked = True)
plt.show()

## Pie chart for listings above 90 nights
labels = list(df['min_night_90_days'].unique())
sizes = list(df.groupby('min_night_90_days').count()['neighbourhood'])
sizes.sort()
#colors
colors = ['#66b3ff','#ff9999'] 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.69,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()