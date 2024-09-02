import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from shapely import MultiPolygon, Polygon
from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Point
import seaborn as sns

def getSpeedCamera(base_url_Speed, chunk_size=1000):
    offset = 0
    dataframes = []
    
    # First, let's determine the total number of rows
    url = f"{base_url_Speed}?$select=count(*)"
    response = requests.get(url)
    if response.status_code == 200:
        total_rows_speed = int(response.json()[0]['count'])
        print(f"Total rows in the speed camera dataset: {total_rows_speed}")
    else:
        print(f"Error fetching row count for speed camera data: {response.status_code}")
        return pd.DataFrame()

    with tqdm(total=total_rows_speed, desc="Fetching data speed camera data") as pbar:
        while offset < total_rows_speed:
            url = f"{base_url_Speed}?$limit={chunk_size}&$offset={offset}"
            response = requests.get(url)
            
            if response.status_code == 200:
                chunk_data = response.json()
                chunk_df = pd.DataFrame(chunk_data)
                dataframes.append(chunk_df)
                offset += len(chunk_data)
                pbar.update(len(chunk_data))
            else:
                print(f"Error fetching data: {response.status_code}")
                break

            if len(chunk_data) < chunk_size:
                break  # We've reached the end of the dataset

    # Combine all dataframes
    final_df = pd.concat(dataframes, ignore_index=True)
    return final_df

def getStopCamera(base_url_Stop, chunk_size=1000):
    offset = 0
    dataframes = []

    url = f"{base_url_Stop}?$select=count(*)"
    response = requests.get(url)
    if response.status_code == 200:
        total_rows_stop = int(response.json()[0]['count'])
        print(f"Total rows in the red light camera dataset: {total_rows_stop}")
    else:
        print(f"Error fetching row count for red light camera data: {response.status_code}")
        return pd.DataFrame()
    with tqdm(total=total_rows_stop, desc="Fetching data red light camera data") as pbar:
        while offset < total_rows_stop:
            url = f"{base_url_Stop}?$limit={chunk_size}&$offset={offset}"
            response = requests.get(url)
            
            if response.status_code == 200:
                chunk_data = response.json()
                chunk_df = pd.DataFrame(chunk_data)
                dataframes.append(chunk_df)
                offset += len(chunk_data)
                pbar.update(len(chunk_data))
            else:
                print(f"Error fetching red light camera data: {response.status_code}")
                break

            if len(chunk_data) < chunk_size:
                break  # We've reached the end of the dataset

    # Combine all dataframes
    final_df = pd.concat(dataframes, ignore_index=True)
    return final_df

def process_violation_data(df):
    # Try to find the date column
    date_column = None
    possible_date_columns = ['VIOLATION DATE', 'VIOLATION_DATE', 'DATE', 'VIOLATION_DT', 'ViolationDate', 'violation_date']
    for col in possible_date_columns:
        if col in df.columns:
            date_column = col
            break

    if date_column is None:
        raise ValueError("Could not find a date column. Please check your DataFrame.")

    # Now let's process the date column
    df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)

    # Extract month, year, day, and day of week
    df['MONTH'] = df[date_column].dt.strftime('%B')  # Full month name
    df['YEAR'] = df[date_column].dt.year
    df['DAY'] = df[date_column].dt.day  # Day of month as number
    df['DAY_OF_WEEK'] = df[date_column].dt.strftime('%A')  # Full day name (Monday, Tuesday, etc.)
    df['MONTH_NUM'] = df[date_column].dt.month
    df['MONTH_YEAR'] = df[date_column].dt.strftime('%Y-%m')
    df['MONTH_DAY_YEAR'] = df[date_column].dt.strftime('%Y-%m-%d')  # Fixed format

    return df

def getCameraLocations(base_url_location, limit = 1000):
    params = {
        "$limit": limit,
    }

    response = requests.get(base_url_location, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"Error fetching the Camera Location Data")
        return pd.DataFrame()

def process_and_plot_data(df, title, color):
    # Ensure 'violations' is numeric
    df['violations'] = pd.to_numeric(df['violations'], errors='coerce')
    
    # Convert violation_date to datetime
    df['violation_date'] = pd.to_datetime(df['violation_date'])
    
    # Group by month and sum violations
    monthly_data = df.groupby(df['violation_date'].dt.to_period('M')).agg({
        'violations': 'sum'
    }).reset_index()
    
    # Convert period to timestamp for plotting
    monthly_data['violation_date'] = monthly_data['violation_date'].dt.to_timestamp()
    
    # Sort and filter data
    monthly_data = monthly_data.sort_values('violation_date')
    start_date = pd.to_datetime('2014-08-01')
    monthly_data = monthly_data[monthly_data['violation_date'] >= start_date]
    
    # Plot the data
    plt.figure(figsize=(15, 8))
    plt.plot(monthly_data['violation_date'], monthly_data['violations'], 
             marker='o', linestyle='-', linewidth=2, markersize=4, color=color)
    
    # Customize the plot
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xlim(start_date, monthly_data['violation_date'].max())
    plt.gcf().autofmt_xdate()
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Violations', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()


# API endpoints for Speed and Redlight cameras
base_url_Speed = "https://data.cityofchicago.org/resource/hhkd-xvj4.json"
base_url_Stop = "https://data.cityofchicago.org/resource/spqx-js37.json"
base_url_location_speed = "https://data.cityofchicago.org/resource/4i42-qv3h.json"
base_url_location_stop = "https://data.cityofchicago.org/resource/thvf-6diy.json"

# Fetch the data into their respective dataframes
dfSpeedCamera = getSpeedCamera(base_url_Speed)
dfStopCamera = getStopCamera(base_url_Stop)
dfLocationSpeed = getCameraLocations(base_url_location_speed)
dfLocationStop = getCameraLocations(base_url_location_stop)


#print(f"Total rows fetched: {len(dfSpeedCamera)}")
#print(f"Total rows fetched: {len(dfStopCamera)}")
#print(f"Total rows fetched: {len(dfLocationSpeed)}")
#print(f"Total rows fetched: {len(dfLocationStop)}")


# Converting the Dates of the violations to dateTime for futher analysis
dfSpeedCamera = process_violation_data(dfSpeedCamera)
dfStopCamera = process_violation_data(dfStopCamera)

"""
# Process and plot Speed Camera data
process_and_plot_data(dfSpeedCamera, 'Monthly Traffic Violations by Speed Camera', 'blue')

# Process and plot Red Light Camera data
process_and_plot_data(dfStopCamera, 'Monthly Traffic Violations by Red Light Camera', 'red')





# Creating a map with the locations of the speed cameras 
# Print the column names to verify the structure
print("Columns in the dataset:", dfSpeedCamera.columns.tolist())

# Check if 'latitude' and 'longitude' columns exist
if 'latitude' in dfSpeedCamera.columns and 'longitude' in dfSpeedCamera.columns:
    # Convert latitude and longitude to float
    dfSpeedCamera['latitude'] = dfSpeedCamera['latitude'].astype(float)
    dfSpeedCamera['longitude'] = dfSpeedCamera['longitude'].astype(float)
else:
    print("Error: 'latitude' and 'longitude' columns not found in the dataset")
    exit()

def fetch_chicago_boundary():
    url = "https://data.cityofchicago.org/resource/qqq8-j68g.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if not data:
            print("Error: Empty data received for Chicago boundary")
            return gpd.GeoDataFrame()
        
        geometries = []
        for feature in data:
            if 'the_geom' in feature and 'coordinates' in feature['the_geom']:
                if feature['the_geom']['type'] == 'MultiPolygon':
                    multi_poly = MultiPolygon([Polygon(poly[0]) for poly in feature['the_geom']['coordinates']])
                    geometries.append(multi_poly)
                elif feature['the_geom']['type'] == 'Polygon':
                    poly = Polygon(feature['the_geom']['coordinates'][0])
                    geometries.append(poly)
        
        if geometries:
            return gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        else:
            print("Error: No valid geometries found in the data")
            return gpd.GeoDataFrame()
    else:
        print(f"Error fetching Chicago boundary: {response.status_code}")
        return gpd.GeoDataFrame()

# Create a GeoDataFrame for speed cameras
geometry = [Point(xy) for xy in zip(dfSpeedCamera['longitude'], dfSpeedCamera['latitude'])]
cameras = gpd.GeoDataFrame(dfSpeedCamera, geometry=geometry, crs="EPSG:4326")

# Fetch and create a GeoDataFrame for Chicago boundary
chicago = fetch_chicago_boundary()
if chicago.empty:
    print("Error: Failed to fetch Chicago boundary data")
    exit()

# Ensure both datasets are in the same CRS
chicago = chicago.to_crs("EPSG:4326")
cameras = cameras.to_crs(chicago.crs)

# Create the map
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the Chicago boundary
chicago.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plot the camera locations
cameras.plot(ax=ax, color='blue', markersize=5)

# Customize the plot
ax.set_title('Speed Camera Locations in Chicago', fontsize=16)
ax.axis('off')  # Turn off the axis

# Add a north arrow
x, y, arrow_length = 0.05, 0.95, 0.1
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)

# Add a scale bar
ax.add_artist(ScaleBar(dx=1, units="km", location="lower right"))

plt.tight_layout()



# Creating a map for the red light camera locations in Chicago
# will follow the same format as above
print("Columns in the dataset:", dfStopCamera.columns.tolist())

# Check if 'latitude' and 'longitude' columns exist
if 'latitude' in dfStopCamera.columns and 'longitude' in dfStopCamera.columns:
    # Convert latitude and longitude to float
    dfStopCamera['latitude'] = dfStopCamera['latitude'].astype(float)
    dfStopCamera['longitude'] = dfStopCamera['longitude'].astype(float)
else:
    print("Error: 'latitude' and 'longitude' columns not found in the dataset")
    exit()

# Create a GeoDataFrame for speed cameras
geometry = [Point(xy) for xy in zip(dfStopCamera['longitude'], dfStopCamera['latitude'])]
cameras = gpd.GeoDataFrame(dfStopCamera, geometry=geometry, crs="EPSG:4326")

# Fetch and create a GeoDataFrame for Chicago boundary
chicago = fetch_chicago_boundary()
if chicago.empty:
    print("Error: Failed to fetch Chicago boundary data")
    exit()

# Ensure both datasets are in the same CRS
chicago = chicago.to_crs("EPSG:4326")
cameras = cameras.to_crs(chicago.crs)

# Create the map
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the Chicago boundary
chicago.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plot the camera locations
cameras.plot(ax=ax, color='red', markersize=5)

# Customize the plot
ax.set_title('Red Light Camera Locations in Chicago', fontsize=16)
ax.axis('off')  # Turn off the axis

# Add a north arrow
x, y, arrow_length = 0.05, 0.95, 0.1
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)

# Add a scale bar
ax.add_artist(ScaleBar(dx=1, units="km", location="lower right"))

plt.tight_layout()

"""
# Creating a chart that maps the total amount of infractions that happen during a given week

print(dfSpeedCamera.head, dfStopCamera.head)

def prepare_data(df):
    # Ensure 'violations' is numeric
    df['violations'] = pd.to_numeric(df['violations'], errors='coerce')
    return df.groupby('DAY_OF_WEEK')['violations'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# Process both datasets
speeding_by_day = prepare_data(dfSpeedCamera)
red_light_by_day = prepare_data(dfStopCamera)

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(speeding_by_day.index))
width = 0.35

rects1 = ax.bar(x - width/2, speeding_by_day.values / 1e6, width, label='Speed Camera Infractions', color='skyblue')
rects2 = ax.bar(x + width/2, red_light_by_day.values / 1e6, width, label='Red Light Camera Infractions', color='lightcoral')

ax.set_ylabel('Number of Infractions (Millions)', fontsize=12)
ax.set_xlabel('Day of the Week', fontsize=12)
ax.set_title('Comparison of Speed and Red Light Camera Infractions by Day of the Week', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(speeding_by_day.index, rotation=0)
ax.legend(fontsize=10)

# Format y-axis in millions
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}M',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

print(speeding_by_day.head)
print(red_light_by_day.head)
"""

# Check for any non-numeric data in the 'violations' column
print("\nNon-numeric data in Speed Camera violations:")
print(dfSpeedCamera[pd.to_numeric(dfSpeedCamera['violations'], errors='coerce').isna()])

print("\nNon-numeric data in Red Light Camera violations:")
print(dfStopCamera[pd.to_numeric(dfStopCamera['violations'], errors='coerce').isna()])


def create_infraction_heatmap(df, title, cmap):
    # Ensure 'violations' column is numeric
    df['violations'] = pd.to_numeric(df['violations'], errors='coerce')
    
    # Create a pivot table
    pivot = pd.pivot_table(df, values='violations', index='MONTH', columns='DAY_OF_WEEK', aggfunc='sum', fill_value=0)
    
    # Reorder the columns to start with Monday
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=days_order)
    
    # Reorder the index (months) to start with January
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    pivot = pivot.reindex(months_order)

    # Ensure all data is numeric
    pivot = pivot.astype(float)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=cmap, cbar_kws={'label': 'Number of Infractions'})
    
    plt.title(title)
    plt.xlabel('Day of Week')
    plt.ylabel('Month')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()


# Create heatmap for speed camera infractions (blue color scheme)
create_infraction_heatmap(dfSpeedCamera, 'Speed Camera Infractions by Month and Day of Week', 'Blues')

# Create heatmap for red light camera infractions (red color scheme)
create_infraction_heatmap(dfStopCamera, 'Red Light Camera Infractions by Month and Day of Week', 'Reds')

# Optional: Calculate correlation between speed and red light infractions
speed_grouped = dfSpeedCamera.groupby(['MONTH', 'DAY_OF_WEEK'])['violations'].sum()
redlight_grouped = dfStopCamera.groupby(['MONTH', 'DAY_OF_WEEK'])['violations'].sum()

# Ensure both Series have the same index
common_index = speed_grouped.index.intersection(redlight_grouped.index)
correlation = speed_grouped[common_index].corr(redlight_grouped[common_index])
print(f"Correlation between speed and red light infractions: {correlation:.2f}")

def create_top_15_charts(df, title, location_col, chart_type='bar'):
    # Ensure 'violations' column is numeric
    df['violations'] = pd.to_numeric(df['violations'], errors='coerce')
    
    # Group by Camera ID and sum the violations
    grouped = df.groupby(['camera_id', location_col])['violations'].sum().sort_values(ascending=False)
    
    # Get top 15
    top_15 = grouped.head(15)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(24, 12))  # Increased figure size
    
    if chart_type == 'bar':
        if title == "Speed Camera Violations":
            top_15.plot(kind='bar', ax=ax, color='blue')
        else:
            top_15.plot(kind='bar', ax=ax, color='red')
        ax.set_ylabel('Number of Violations')
        ax.set_title(f'Top 15 Locations for {title} - Bar Chart')
        ax.set_xticklabels([f"{loc}\n({cam_id})" for (cam_id, loc), _ in top_15.items()], rotation=45, ha='right')
    elif chart_type == 'pie':
        wedges, texts, autotexts = ax.pie(top_15.values, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        ax.set_title(f'Top 15 Locations for {title} - Pie Chart')
        
        # Create legend
        legend_labels = [f"{loc} ({cam_id}): {val:,.0f}" for (cam_id, loc), val in top_15.items()]
        ax.legend(wedges, legend_labels, title="Locations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    
    # Adjust bottom margin
    plt.subplots_adjust(bottom=0.2)

    # Print the data
    print(f"\nTop 15 Locations for {title}:")
    print(top_15)

# Create bar charts
create_top_15_charts(dfSpeedCamera, 'Speed Camera Violations', 'address', 'bar')
create_top_15_charts(dfStopCamera, 'Red Light Camera Violations', 'intersection', 'bar')

# Create pie charts
create_top_15_charts(dfSpeedCamera, 'Speed Camera Violations', 'address', 'pie')
create_top_15_charts(dfStopCamera, 'Red Light Camera Violations', 'intersection', 'pie')
"""

plt.show()