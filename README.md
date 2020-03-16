# SQLAlchemy Challenge - Surfs Up!

This project integrates knowledge of Python, and SQL queries through SQL Alchemy.

A SQLite document with historical weather data is used to query and analyze for vacation weather forecasting.
```python
%matplotlib inline
from matplotlib import style

import matplotlib.pyplot as plt

# dependencies for styling
# Can also use ggplot2 reference https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
plt.style.use('fivethirtyeight')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina', quality=100)

import seaborn as sns
```


```python
import numpy as np
import pandas as pd
```


```python
import datetime as dt
```

# Reflect Tables into SQLAlchemy ORM


```python
# Python SQL toolkit and Object Relational Mapper
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

# existing functions within sqlalchemy 
from sqlalchemy import create_engine, func, inspect, asc, desc
```


```python
engine = create_engine("sqlite:///Resources/hawaii.sqlite")
```


```python
# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)
```


```python
# We can view all of the classes that automap found
Base.classes.keys()
```




    ['measurement', 'station']




```python
# Save references to each table
Measurement = Base.classes.measurement
Station = Base.classes.station
```


```python
# Create our session (link) from Python to the DB
session = Session(engine)

# Create inspector to view .sqlite file - or any other connected engine
inspector = inspect(engine)
```


```python
# method of exploring the data within an engine
# call inspector and use get_columns() to view columns
# must be called pythonically

# measurement data inspection - create callable list of column names to refer back to and inspect datatypes
measurement_columns = []

measurement_ins = inspector.get_columns('measurement')
for column in measurement_ins:
    measurement_columns.append(column['name'])
    print(column['name'], column['type'])
    
measurement_columns
```

    id INTEGER
    station TEXT
    date TEXT
    prcp FLOAT
    tobs FLOAT
    




    ['id', 'station', 'date', 'prcp', 'tobs']




```python
# station data inspection - create a callable list of column names to refer back to and inspect datatypes
station_columns = []

station_ins = inspector.get_columns('station')
for column in station_ins:
    station_columns.append(column['name'])
    print(column['name'], column['type'])

station_columns
```

    id INTEGER
    station TEXT
    name TEXT
    latitude FLOAT
    longitude FLOAT
    elevation FLOAT
    




    ['id', 'station', 'name', 'latitude', 'longitude', 'elevation']




```python
# Observe station data
# Using the engine method to query data in SQL language 

engine.execute('SELECT * FROM station LIMIT 10').fetchall()
```




    [(1, 'USC00519397', 'WAIKIKI 717.2, HI US', 21.2716, -157.8168, 3.0),
     (2, 'USC00513117', 'KANEOHE 838.1, HI US', 21.4234, -157.8015, 14.6),
     (3, 'USC00514830', 'KUALOA RANCH HEADQUARTERS 886.9, HI US', 21.5213, -157.8374, 7.0),
     (4, 'USC00517948', 'PEARL CITY, HI US', 21.3934, -157.9751, 11.9),
     (5, 'USC00518838', 'UPPER WAHIAWA 874.3, HI US', 21.4992, -158.0111, 306.6),
     (6, 'USC00519523', 'WAIMANALO EXPERIMENTAL FARM, HI US', 21.33556, -157.71139, 19.5),
     (7, 'USC00519281', 'WAIHEE 837.5, HI US', 21.45167, -157.84888999999998, 32.9),
     (8, 'USC00511918', 'HONOLULU OBSERVATORY 702.2, HI US', 21.3152, -157.9992, 0.9),
     (9, 'USC00516128', 'MANOA LYON ARBO 785.2, HI US', 21.3331, -157.8025, 152.4)]




```python
#  Observe measurement data
engine.execute('SELECT * FROM measurement LIMIT 10').fetchall()
```




    [(1, 'USC00519397', '2010-01-01', 0.08, 65.0),
     (2, 'USC00519397', '2010-01-02', 0.0, 63.0),
     (3, 'USC00519397', '2010-01-03', 0.0, 74.0),
     (4, 'USC00519397', '2010-01-04', 0.0, 76.0),
     (5, 'USC00519397', '2010-01-06', None, 73.0),
     (6, 'USC00519397', '2010-01-07', 0.06, 70.0),
     (7, 'USC00519397', '2010-01-08', 0.0, 64.0),
     (8, 'USC00519397', '2010-01-09', 0.0, 68.0),
     (9, 'USC00519397', '2010-01-10', 0.0, 73.0),
     (10, 'USC00519397', '2010-01-11', 0.01, 64.0)]



# Exploratory Climate Analysis


```python

```


```python

```


```python
# Design a query to retrieve the last 12 months of precipitation data and plot the results

# Find most recent date - data is in the form of a tuple until position is called [0]
latest_date = session.query(Measurement.date).order_by(Measurement.date.desc()).first()[0]
print(f'Latest Date: {latest_date}')

# Find date 12 months prior to latest_date yr-m-d
query_date = dt.datetime.strptime(latest_date, "%Y-%m-%d") - dt.timedelta(days=365)
print("Query Date: ", query_date)

# Query to find all dates between 8/23/16 and 8/23/17
lastyear_dates_query = session.query(Measurement.date, Measurement.prcp).filter(Measurement.date >= query_date).all()
lastyear_data = pd.DataFrame(lastyear_dates_query)


# # Manipulate lastyear_data to be plotted
lastyear_data = lastyear_data.set_index('date').sort_values(by='date', ascending=True)
lastyear_data.rename(columns = {'prcp':'Precipitation'}, inplace = True) 


# Use Pandas Plotting with Matplotlib to plot the data
plt.rcParams['figure.figsize'] = (20, 10)

lastyear_data.plot(title="2016-2017 Rainfall in Hawaii")
plt.xticks(rotation=45)
plt.ylabel("Precipitation (in.)")
plt.xlabel("Date")
plt.legend(loc='upper center')
plt.show()

plt.savefig("prcp_hi.png")
```

    Latest Date: 2017-08-23
    Query Date:  2016-08-23 00:00:00
    


![png](climate_analysis_files/climate_analysis_17_1.png)



    <Figure size 1440x720 with 0 Axes>


![precipitation](Images/precipitation.png)


```python
# Use Pandas to calcualte the summary statistics for the precipitation data
lastyear_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precipitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.176462</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.460288</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.130000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>6.700000</td>
    </tr>
  </tbody>
</table>
</div>



![describe](Images/describe.png)


```python
# Design a query to show how many stations are available in this dataset?
print(station_columns)
session.query(func.count(Station.id)).all()
```

    ['id', 'station', 'name', 'latitude', 'longitude', 'elevation']
    




    [(9)]




```python
# What are the most active stations? (i.e. what stations have the most rows)?
# List the stations and the counts in descending order.
stations_df = engine.execute("SELECT * FROM station").fetchall()
stations_df = pd.DataFrame(stations_df, columns=station_columns)
stations_df = stations_df.set_index("id")
stations_df

measurement_df = engine.execute("SELECT * FROM measurement").fetchall()
measurement_df = pd.DataFrame(measurement_df, columns=measurement_columns)
measurement_df = measurement_df.set_index("id")
measurement_df

# Count stations
station_count = measurement_df.groupby("station").count().sort_values(by="date", ascending=False)
station_count = pd.DataFrame(station_count)
station_count 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>prcp</th>
      <th>tobs</th>
    </tr>
    <tr>
      <th>station</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>USC00519281</td>
      <td>2772</td>
      <td>2772</td>
      <td>2772</td>
    </tr>
    <tr>
      <td>USC00519397</td>
      <td>2724</td>
      <td>2685</td>
      <td>2724</td>
    </tr>
    <tr>
      <td>USC00513117</td>
      <td>2709</td>
      <td>2696</td>
      <td>2709</td>
    </tr>
    <tr>
      <td>USC00519523</td>
      <td>2669</td>
      <td>2572</td>
      <td>2669</td>
    </tr>
    <tr>
      <td>USC00516128</td>
      <td>2612</td>
      <td>2484</td>
      <td>2612</td>
    </tr>
    <tr>
      <td>USC00514830</td>
      <td>2202</td>
      <td>1937</td>
      <td>2202</td>
    </tr>
    <tr>
      <td>USC00511918</td>
      <td>1979</td>
      <td>1932</td>
      <td>1979</td>
    </tr>
    <tr>
      <td>USC00517948</td>
      <td>1372</td>
      <td>683</td>
      <td>1372</td>
    </tr>
    <tr>
      <td>USC00518838</td>
      <td>511</td>
      <td>342</td>
      <td>511</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Using the station id from the previous query, calculate the lowest temperature recorded, 
# highest temperature recorded, and average temperature of the most active station?
# measurement_df = measurement_df.set_index("station")
USC00519281 = measurement_df.loc["USC00519281", :]
USC00519281

print(f"Highest temperature at USC00519281: {USC00519281.tobs.max()}F")
print(f"Lowest temperature at USC00519281: {USC00519281.tobs.min()}F")
print(f"Mean temperature at USC00519281: {USC00519281.tobs.mean()}F")

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~\Anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2896             try:
    -> 2897                 return self._engine.get_loc(key)
       2898             except KeyError:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index_class_helper.pxi in pandas._libs.index.Int64Engine._check_type()
    

    KeyError: 'USC00519281'

    
    During handling of the above exception, another exception occurred:
    

    KeyError                                  Traceback (most recent call last)

    <ipython-input-19-8d8ef14f4f01> in <module>
          2 # highest temperature recorded, and average temperature of the most active station?
          3 # measurement_df = measurement_df.set_index("station")
    ----> 4 USC00519281 = measurement_df.loc["USC00519281", :]
          5 USC00519281
          6 
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
       1416                 except (KeyError, IndexError, AttributeError):
       1417                     pass
    -> 1418             return self._getitem_tuple(key)
       1419         else:
       1420             # we by definition only have the 0th axis
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_tuple(self, tup)
        803     def _getitem_tuple(self, tup):
        804         try:
    --> 805             return self._getitem_lowerdim(tup)
        806         except IndexingError:
        807             pass
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_lowerdim(self, tup)
        927         for i, key in enumerate(tup):
        928             if is_label_like(key) or isinstance(key, tuple):
    --> 929                 section = self._getitem_axis(key, axis=i)
        930 
        931                 # we have yielded a scalar ?
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
       1848         # fall thru to straight lookup
       1849         self._validate_key(key, axis)
    -> 1850         return self._get_label(key, axis=axis)
       1851 
       1852 
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexing.py in _get_label(self, label, axis)
        158             raise IndexingError("no slices here, handle elsewhere")
        159 
    --> 160         return self.obj._xs(label, axis=axis)
        161 
        162     def _get_loc(self, key: int, axis: int):
    

    ~\Anaconda3\lib\site-packages\pandas\core\generic.py in xs(self, key, axis, level, drop_level)
       3735             loc, new_index = self.index.get_loc_level(key, drop_level=drop_level)
       3736         else:
    -> 3737             loc = self.index.get_loc(key)
       3738 
       3739             if isinstance(loc, np.ndarray):
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2897                 return self._engine.get_loc(key)
       2898             except KeyError:
    -> 2899                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2900         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2901         if indexer.ndim > 1 or indexer.size > 1:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index_class_helper.pxi in pandas._libs.index.Int64Engine._check_type()
    

    KeyError: 'USC00519281'



```python

USC00519281 = USC00519281.sort_values("date", ascending=False)
USC00519281.head()

year_USC00519281 = USC00519281.iloc[0:365, :]



```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-20-3243037c3b25> in <module>
    ----> 1 USC00519281 = USC00519281.sort_values("date", ascending=False)
          2 USC00519281.head()
          3 
          4 year_USC00519281 = USC00519281.iloc[0:365, :]
          5 
    

    NameError: name 'USC00519281' is not defined



```python

# Choose the station with the highest number of temperature observations.
USC00519281 = session.query(Measurement.station, func.count(Measurement.tobs))\
.group_by(Measurement.station).order_by(func.count(Measurement.station).desc()).first()
USC00519281 = USC00519281[0]
print(f"Station with most temperature data : {USC00519281}")

# Query the last 12 months of temperature observation data for this station and plot the results as a histogram
USC00519281_hist = session.query( Measurement.tobs).filter(Measurement.date >= query_date)\
.filter(Measurement.station == USC00519281).all()
USC00519281_hist = pd.DataFrame(USC00519281_hist, columns=['temperature'])

# plot
USC00519281_hist.plot.hist(bins=12, title="Frequency of Temperatures at Station USC00519281 in Hawaii")
plt.ylabel("Frequency")
plt.xlabel("Temperature (F)")
plt.legend(loc='upper center')
plt.tight_layout()
plt.savefig("temperaturehist.png")
plt.show()
```

    Station with most temperature data : USC00519281
    


![png](climate_analysis_files/climate_analysis_25_1.png)


![precipitation](Images/station-histogram.png)


```python
# This function called `calc_temps` will accept start date and end date in the format '%Y-%m-%d' 
# and return the minimum, average, and maximum temperatures for that range of dates
def calc_temps(start_date, end_date):
    """TMIN, TAVG, and TMAX for a list of dates.
    
    Args:
        start_date (string): A date string in the format %Y-%m-%d
        end_date (string): A date string in the format %Y-%m-%d
        
    Returns:
        TMIN, TAVE, and TMAX
    """
    
    return session.query(func.min(Measurement.tobs), func.avg(Measurement.tobs), func.max(Measurement.tobs)).\
        filter(Measurement.date >= start_date).filter(Measurement.date <= end_date).all()

# function usage example
print(calc_temps('2012-02-28', '2012-03-05'))
```

    [(62.0, 69.57142857142857, 74.0)]
    


```python
# Use your previous function `calc_temps` to calculate the tmin, tavg, and tmax 
# for your trip using the previous year's data for those same dates.
print(calc_temps('2017-02-28', '2017-03-05'))
trip_avg_temp = calc_temps('2017-02-28', '2017-03-05')
```

    [(64.0, 72.02777777777777, 78.0)]
    


```python
# Plot the results from your previous query as a bar chart. 
# Use "Trip Avg Temp" as your Title
# Use the average temperature for the y value
# Use the peak-to-peak (tmax-tmin) value as the y error bar (yerr)

# Create a dataframe with the calculated tmin, tavg, and tmax values
trip_avg_temp = pd.DataFrame(trip_avg_temp, columns=['TMIN', 'TAV', 'TMAX'])


# Plot the results from your previous query as a bar chart. 
# Use "Trip Avg Temp" as your Title
# Use the average temperature for the y value
# Use the peak-to-peak (tmax-tmin) value as the y error bar (yerr)
trip_avg_temp.plot.bar(y='TAV', yerr=(trip_avg_temp['TMAX'] - trip_avg_temp['TMIN']), color='teal', figsize=(6,10))
plt.suptitle('Average Daily Temperature', y = 1.03, fontsize = 18)
plt.title('Hawaii (Feb. 28th - May 3rd 2017)', fontsize = 12)
plt.ylabel("Temp (F)")
plt.tight_layout()
plt.gca().legend_.remove()
plt.xticks([])
plt.savefig("avtripbar.png")
plt.show()
```


![png](climate_analysis_files/climate_analysis_29_0.png)



```python
# Predict the total amount of rainfall per weather station for your trip dates using the previous year's matching dates.
# Sort this in descending order by precipitation amount and list the station, name, latitude, longitude, and elevation
# Total Rainfall per weather station

# Data grouped by weather station

# 

def prcp(start_date, end_date):
    
        # Docstring for the function `calc_temps`
    """Precipitation information per weather station
    
    Args:
        start_date (string): A date string in the format %Y-%m-%d
        end_date (string): A date string in the format %Y-%m-%d
        
    Returns:
        A list of tuples containing precipitation amount, station, name, latitude, longitude, and elevation in descending order.
    """
    
    sel = [Station.name,
           Measurement.prcp,
           Measurement.station, 
           Station.latitude, 
           Station.longitude, 
           Station.elevation]
    
    return session.query(*sel).\
            filter(Measurement.station == Station.station).filter(Measurement.date >= start_date).filter(Measurement.date <= end_date).group_by(Measurement.station).order_by(Measurement.prcp.desc()).all()

rainfall_query = prcp('2017-02-28', '2017-03-05')

rainfall_query = pd.DataFrame(rainfall_query)
rainfall_query
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>prcp</th>
      <th>station</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>MANOA LYON ARBO 785.2, HI US</td>
      <td>0.58</td>
      <td>USC00516128</td>
      <td>21.33310</td>
      <td>-157.80250</td>
      <td>152.4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>KANEOHE 838.1, HI US</td>
      <td>0.16</td>
      <td>USC00513117</td>
      <td>21.42340</td>
      <td>-157.80150</td>
      <td>14.6</td>
    </tr>
    <tr>
      <td>2</td>
      <td>WAIHEE 837.5, HI US</td>
      <td>0.04</td>
      <td>USC00519281</td>
      <td>21.45167</td>
      <td>-157.84889</td>
      <td>32.9</td>
    </tr>
    <tr>
      <td>3</td>
      <td>KUALOA RANCH HEADQUARTERS 886.9, HI US</td>
      <td>0.04</td>
      <td>USC00514830</td>
      <td>21.52130</td>
      <td>-157.83740</td>
      <td>7.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>WAIMANALO EXPERIMENTAL FARM, HI US</td>
      <td>0.00</td>
      <td>USC00519523</td>
      <td>21.33556</td>
      <td>-157.71139</td>
      <td>19.5</td>
    </tr>
    <tr>
      <td>5</td>
      <td>WAIKIKI 717.2, HI US</td>
      <td>0.00</td>
      <td>USC00519397</td>
      <td>21.27160</td>
      <td>-157.81680</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>PEARL CITY, HI US</td>
      <td>NaN</td>
      <td>USC00517948</td>
      <td>21.39340</td>
      <td>-157.97510</td>
      <td>11.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
jupyter nbconvert --to markdown climate_analysis.ipynb  
```


      File "<ipython-input-7-980421729f3a>", line 1
        jupyter nbconvert --to markdown climate_analysis.ipynb
                        ^
    SyntaxError: invalid syntax
    



```python

```
