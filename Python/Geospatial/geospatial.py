###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: October 06, 2015
# summary: exploration of Python's geospatial analysis tools
###############################################################################

###############################################################################
###############################################################################
###############################################################################
## pyshp library docusmentation:
# https://github.com/GeospatialPython/pyshp

## geospatialpython website:
# http://geospatialpython.com


###############################################################################
############################### DEPENDENCIES ##################################
###############################################################################
### STANDARD MODULES

import numpy as np
import pandas as pd

import urllib
import urllib2
import json
import re

# to get ggplot-like style for plots
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot

import functools
import itertools
import os, sys

# Get Python environment parameters
print 'Python version ' + sys.version
print 'Pandas version: ' + pd.__version__


### USER VARS
localPath = '/Users/amirkavousian/Documents/Py_Codes/geospatial'


###############################################################################
############################### FORMATTING ####################################
###############################################################################
### CONVERTING A CSV FILE INTO A SHAPE FILE
# Source: https://github.com/GeospatialPython/Learn/blob/master/csv2shp.py

import csv
import shapefile

# Create a polygon shapefile writer
w = shapefile.Writer(shapefile.POLYGON)

# Add our fields
w.field("NAME", "C", "40")
w.field("AREA", "C", "40")

# Download the file from: https://github.com/GeospatialPython/Learn/blob/master/sample.csv
# Open the csv file and set up a reader
with open(localPath+"/Data/sample.csv") as p:
    reader = csv.DictReader(p)
    for row in reader:
        # Add records for each polygon for name and area
        w.record(row["Name"], row["Area"])
        # parse the coordinate string
        wkt = row["geometry"][9:-2]
        # break the coordinate string in to x,y values
        coords = wkt.split(",")
        # set up a list to contain the coordinates
        part = []
        # convert the x,y values to floats
        for c in coords:
            x,y = c.split(" ")
            part.append([float(x),float(y)])
        # create a polygon record with the list of coordinates.
        w.poly(parts=[part])

# save the shapefile. This will save three files in these formats: .dbf, .shp, .shx
w.save(localPath+"/polys.shp")

###############################################################################
################################ PLOTTING #####################################
###############################################################################
### PLOT A MAP OF NYC USING SHAPE FILE
import shapely
import geopandas

# Example plot using shapely
p1 = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1)])
p2 = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
p3 = shapely.geometry.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
g = geopandas.GeoSeries([p1,p2,p3])
g.area
g.plot()
plt.show()
plt.close()

# Plot using geopandas
boros = geopandas.GeoDataFrame.from_file(localPath+'/Data/nycb2010_15c/nycb2010.shp')
boros.ix[0]
# plt.figure((6,8))
boros.plot()
plt.draw()
plt.savefig(localPath+'/Results/nyc_geopandas.png')
plt.close()

# See the real one here: https://data.cityofnewyork.us/City-Government/2010-Census-Blocks/v2h8-6mxf
###############################################################################


###############################################################################
### PLOT NYC USING JSON FILE
import simplejson as json

bx = file(localPath+"/Data/nyc.json")
bx = json.loads(bx.read())

polygons = bx['features']

censusTracts = np.array([bx['features'][k]['geometry']['coordinates'][0] for k in range(len(bx['features']))])

for x in range(len(censusTracts)):
	plt.plot(np.array(censusTracts[x])[:, 0] , np.array(censusTracts[x])[:, 1]  , '-')

plt.xlim(xmin=900000, xmax=1100000)
plt.ylim(ymin=120000, ymax=280000)
plt.axes().set_aspect('equal', 'datalim')
plt.show()
###############################################################################


###############################################################################
### PLOT NYC TAXI DATA
# http://www.danielforsyth.me/mapping-nyc-taxi-data/
import googleapiclient as gac

#*#TODO: I still cannot get the BigQuery to return anything. I get authentication error.
df = pd.io.gbq.read_gbq("""
        SELECT ROUND(pickup_latitude, 4) as lat, ROUND(pickup_longitude, 4) as long, COUNT(*) as num_pickups
        FROM [nyc-tlc:yellow.trips_2015]
        WHERE (pickup_latitude BETWEEN 40.61 AND 40.91) AND (pickup_longitude BETWEEN -74.06 AND -73.77 )
        GROUP BY lat, long
        """, project_id='taxi-1029')

import matplotlib
import matplotlib.pyplot as plt

pd.options.display.mpl_style = 'default' #Better Styling
new_style = {'grid': False} #Remove grid
matplotlib.rc('axes', **new_style)
from matplotlib import rcParams
rcParams['figure.figsize'] = (17.5, 17) #Size of figure
rcParams['figure.dpi'] = 250

P.set_axis_bgcolor('black') #Background Color

P=df.plot(kind='scatter', x='long', y='lat',color='white',xlim=(-74.06,-73.77),ylim=(40.61, 40.91),s=.02,alpha=.6)
###############################################################################

###############################################################################
### PLOT DATA POINTS ON A MAP
# import pylab as plt
from mpl_toolkits.basemap import Basemap
plt.close('all')

# Data of city location (logitude,latitude) and population
# dictionary of the populations of each city
pop={'New York':8244910,
    'Los Angeles':3819702,
    'Chicago':2707120,
    'Houston':2145146,
    'Philadelphia':1536471,
    'Pheonix':1469471,
    'San Antonio':1359758,
    'San Diego':1326179,
    'Dallas':1223229,
    'San Jose':967487,
    'Jacksonville':827908,
    'Indianapolis':827908,
    'Austin':820611,
    'San Francisco':812826,
    'Columbus':797434}

# dictionary of the latitudes of each city
lats={'New York':40.6643,
    'Los Angeles':34.0194,
    'Chicago':41.8376,
    'Houston':29.7805,
    'Philadelphia':40.0094,
    'Pheonix':33.5722,
    'San Antonio':29.4724,
    'San Diego':32.8153,
    'Dallas':32.7942,
    'San Jose':37.2969,
    'Jacksonville':30.3370,
    'Indianapolis':39.7767,
    'Austin':30.3072,
    'San Francisco':37.7750,
    'Columbus':39.9848}

# dictionary of the longitudes of each city
lngs={'New York':73.9385,
    'Los Angeles':118.4108,
    'Chicago':87.6818,
    'Houston':95.3863,
    'Philadelphia':75.1333,
    'Pheonix':112.0880,
    'San Antonio':98.5251,
    'San Diego':117.1350,
    'Dallas':96.7655,
    'San Jose':121.8193,
    'Jacksonville':81.6613,
    'Indianapolis':86.1459,
    'Austin':97.7560,
    'San Francisco':122.4183,
    'Columbus':82.9850}

# PLot geographical boundaries (maps)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='c')
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Plot points of interest
max_size=80
for city in lngs.keys():
        x, y = m(-lngs[city],lats[city])
        m.scatter(x,y,max_size*pop[city]/pop['New York'],marker='o',color='r')

plt.show()
plt.draw()
plt.close()
###############################################################################

###############################################################################
### PLOT A US MAP WITH CONTOURS, GRID, POINTS
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import cm as mcm
from matplotlib.backends.backend_pdf import PdfPages
# import gbapi as gbp
import numpy as np
import matplotlib.pyplot as plt
import pCommon as com
# import filters as fil
from datetime_tut import datetime_tut as dt
import os

destRoot='map-differences/'

validPoligons=com.geoPoly()


# initialize variables for tracking
glats, glngs, diffVals = [],[],[]
iCounter = 0


### for each area, scan coordinates
''' hawaii '''
minLat, maxLat = 18.05, 22.95
minLng, maxLng = -160.95, -154.15
minMaxCoords=[minLat,maxLat,minLng,maxLng]

''' continental US '''
minLat, maxLat = 30, 50
minLng, maxLng = -127, -115
minMaxCoords=[minLat,maxLat,minLng,maxLng]


### create figure and axes instances
fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.add_axes([0.1,0.1,0.8,0.8])

# create polar stereographic Basemap instance.
m = Basemap(lon_0=100,lat_0=90.,lat_ts=37,\
            llcrnrlat=min([min(x) for x in glats]),urcrnrlat=max([max(x) for x in glats]),\
            llcrnrlon=min([min(x) for x in glngs]),urcrnrlon=max([max(x) for x in glngs]),\
            rsphere=6371200.,resolution='l',area_thresh=10000)

# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# draw parallels.
parallels = np.arange(0.,90,1.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

# draw meridians
meridians = np.arange(180.,360.,1.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
l=np.array(diffVals)
lons, lats = m.makegrid(l.shape[1],l.shape[0]) # get lat/lons of ny by nx evenly space grid.
x, y = m(lons, lats) # compute map proj coordinates.

# draw filled contours.
#clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
clevs = [-0.0002,0,0.0002,0.0004,0.0006,0.0008,0.001,0.0012,0.0014,0.0016,0.0018,0.002,0.0022,0.0024,0.0026,0.0028,0.003,0.0032,0.0034,0.0036,0.0038,0.004,0.01,0.02,0.03]
cs = m.contourf(x,y,diffVals,clevs,cmap=cm.s3pcpn)

# draw points of interest
m.scatter(x,y,max_size*pop[city]/pop['New York'],marker='o',color='r')

# add colorbar.
cbar = m.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('kWh')

# add title
plt.title('kWh Diff. Between PVWatts v4 and Sunsim')

# save to file
pp = PdfPages(localPath+'/Results/US_grid.pdf')
pp.savefig(fig)
pp.close()

# plot on screen
plt.draw()
plt.close()
###############################################################################

###############################################################################
### DRAWING A MAP BACKGROUND
# http://matplotlib.org/basemap/users/geography.html
# This is a feature in basemap module, so you can use it with the maps drawn above.
###############################################################################


###############################################################################
################################# GEOCODING ###################################
###############################################################################

###############################################################################
def get_hack_coords(address):
    # get geocoder.us coordinates of address, does not have intelligent address correction like google

    queryUrl="http://geocoder.us/demo.cgi?address="+urllib.quote_plus(address)

    req = urllib2.Request(queryUrl)
    usock = urllib2.urlopen(req)
    html = usock.read()
    o = re.search('<td><h3>Latitude<\/h3><\/td>\s+<td>(\d+(\.\d+)?) &deg;<br>',html)
    lat = float(o.group(1))
    o = re.search('<td><h3>Longitude<\/h3><\/td>\s+<td>((\-)?\d+(\.\d+)?) &deg;<br>',html)
    lng = float(o.group(1))
    return {'lat':lat, 'lng':lng}
###############################################################################


###############################################################################
### GET LAT/LNG OF AN ADDRESS
def get_google_coords(address):
    # get google maps coordinates
    address = '747 Anderson St San Francisco CA 94110'
    queryUrl = 'https://maps.googleapis.com/maps/api/geocode/json?address='+urllib.quote_plus(address)
    req = urllib2.Request(queryUrl)
    usock = urllib2.urlopen(req)
    html = usock.read()
    resp = json.loads(html)
    if(resp['status']=='OK'):
        return resp['results'][0]['geometry']['location']
    else:
        queryUrl='https://www.google.com/maps?t=m&q='+urllib.quote_plus(address)
        req = urllib2.Request(queryUrl)
        usock = urllib2.urlopen(req)
        html = usock.read()
        o = re.search('viewport_center_lat=((\-)?\d+(\.\d+)?)',html)
        lat = float(o.group(1))
        o = re.search('viewport_center_lng=((\-)?\d+(\.\d+)?)',html)
        lng = float(o.group(1))
        return {'lat':lat, 'lng':lng}
###############################################################################
