###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: scripts, functions, and modules to handle date time data
# other resources:
# https://docs.python.org/2/library/datetime.html
###############################################################################

### MODULES
from datetime import datetime as dt
from datetime import timedelta
import datetime as dat
import datetime
import pytz

###

### FORMATS
unixFormat='%Y-%m-%d %H:%M:%S'
oracleFormat='%Y%m%d'
sqlUnixFormat = 'YYYY-MM-DD HH24:MI:SS'
repgenTimezone =  'America/Los_Angeles' # Config.get("Settings", "Repgen_Timestamp_Timezone")

###############################################################################
### USAGE EXAMPLES
# used in pMysql
# to get the time tuple for current time
dt.utcnow()
# to convert to a string
dt.utcnow().strftime(unixFormat)

# used in pOrcl
# to get the time tuple for current time
dt.now(pytz.timezone(repgenTimezone))
# to convert to string
dt.now(pytz.timezone(repgenTimezone)).strftime(unixFormat)


###############################################################################
### GENERAL INFO
# There are two kinds of date and time objects: naive and aware.
# An aware object has sufficient knowledge of applicable algorithmic and political time adjustments,
# such as time zone and daylight saving time information, to locate itself relative to other aware objects.
# A naive object does not contain enough information to unambiguously locate itself relative to other date/time objects.
# Whether a naive object represents Coordinated Universal Time (UTC), local time, or time in some other timezone is purely up to the program.

# Objects of the date type are always naive.
# An object of type time or datetime may be naive or aware.
# A datetime object d is aware if d.tzinfo is not None and d.tzinfo.utcoffset(d) does not return None.
# If d.tzinfo is None, or if d.tzinfo is not None but d.tzinfo.utcoffset(d) returns None, d is naive.
# A time object t is aware if t.tzinfo is not None and t.tzinfo.utcoffset(None) does not return None. Otherwise, t is naive.

# For applications requiring aware objects, datetime and time objects have an optional time zone information attribute, tzinfo,
# that can be set to an instance of a subclass of the abstract tzinfo class.
dt.utcnow().tzinfo
print dt.now(pytz.timezone(repgenTimezone)).tzinfo

# Note that no concrete tzinfo classes are supplied by the datetime module.
# Supporting timezones at whatever level of detail is required is up to the application.
# This implies that, unless specifically specified by the application, tzinfo will not be populated.




###############################################################################
### datetime module constants:
datetime.MINYEAR
datetime.MAXYEAR


###############################################################################
######################### datetime module classes #############################
### Three main object types in the datetime module are: date, time, and datetime.

#######################################
### datetime.date
# attributes: year, month, and day. These three attributes are mandatory.
# create a datetime.date object from a tuple of numeric values
my_birthday = datetime.date(dt.now().year, 6, 24)
abs(my_birthday - dt.utcnow().date())

# Note that the exact same objects are also available for datetime.datetime objects,
# with the difference that the latter has both date and time info, while the former only includes date info.
datetime.date(2015, 10, 01)
datetime.date.today()

datetime.date.fromtimestamp(28600)
datetime.date.fromordinal(1000)  # 1000th day after 1. 1. 0001

d = datetime.date(2002, 12, 31)
d.replace(day=26) == datetime.date(2002, 12, 26)

d.toordinal()  # you can also call datetime.date.toordinal(d) . The same applies to other functions below
d.timetuple()
d.weekday()  # Monday is 0 and Sunday is 6
d.isoweekday()  # Monday is 1 and Sunday is 7
d.isocalendar() # (ISO year, ISO week number, ISO weekday)
d.isoformat()
d.__str__()  # same as isoformat() for most applications
d.ctime()
d.strftime(unixFormat)
d.__format__(unixFormat)



#######################################
### datetime.time
# attributes: hour, minute, second, microsecond, and tzinfo.
# These attributes are not mandatory. If not supplied, they are all assumed to be zero.
# create a datetime.time object from a tuple of values
datetime.time(12,8,59)


#######################################
### datetime.datetime
# attributes: year, month, day, hour, minute, second, microsecond, and tzinfo.
# The year, month, day attributes are mandatory. But other attributes are not mandatory and are assumed to be zero if not supplied.
# In our work, since datetime is the most general object, we import datetime.datetime as its own separate object (call it dt) to reduce typing.
import datetime.datetime as dt

# to create a datetime object from a tuple of values
# Note that we used 'import datetime.datetime as dt' . Modify the call if you use 'import datetime as dt'
dt(2015,01,01)

# the last element of the tuple is milliseconds
dt.today()
dt.now().date()
dt.now().time()
now = dt.now()

# use combine to make a datetime object from separate date and time objects.
dt.combine(dt.now().date(), dt.now().time())


dt.fromtimestamp(28600)
dt.fromtimestamp(286000)
dt.utcfromtimestamp(0)  # epoch
dt.fromtimestamp(0)  # uses local timezone

dt.fromordinal(1000)
dt.fromordinal(730920) # 730920th day after 1. 1. 0001

now = dt.utcnow()
delta = now - dt.utcfromtimestamp(0)
delta.days  # days have passed since the epoch
seconds = delta.seconds  # returns the seconds attribute of the time (i.e., it is not equal to the time in seconds)
total_seconds = delta.total_seconds()  # adds up the year, months, days, seconds, and microseconds into one single seconds unit
# recover now using total number of seconds passed since the epoch
now == dt.utcfromtimestamp(total_seconds)

# convert to / from string
string_date = str(now)
string_date = str(datetime.date.today())
my_date = datetime.date(*[int(i) for i in string_date.split("-")])
# customize the string format
string_date =  my_date.strftime('%m/%d/%Y')  # This writes "06/24/1984"
string_date2 =  my_date.strftime(unixFormat)
string_date3 =  my_date.strftime(oracleFormat)
string_date4 =  my_date.strftime(sqlUnixFormat)
(string_date, string_date2, string_date3, string_date4)

dt.now().timetuple()
dt.now().toordinal()
dt.now().weekday()  # Monday is 0 and Sunday is 6
dt.now().isoweekday()  # Monday is 1 and Sunday is 7
dt.now().isocalendar()  # (ISO year, ISO week number, ISO weekday)
dt.now().isoformat()
dt.now().__str__()  # same as isoformat() for most applications
dt.now().ctime()
dt.now().ctime(dt.now().mktime(dt.now().timetuple()))
dt.now().strftime(unixFormat)  # Return a string representing the date, controlled by an explicit format string.
dt.now().__format__(format)
dt.fromtimestamp(dt.now().time())

# use replace to partially modify a datetime object
now = dt.now()
now.replace(day=now.day-1)

### class attributes
dt.min
dt.max
dt.resolution
dt.year
dt.month
dt.day


#######################################
### datetime.timedelta
# A duration expressing the difference between two date, time, or datetime instances
# Internally, it only stores days, seconds, and microseconds values
d = timedelta(microseconds=-1)
(d.days, d.seconds, d.microseconds)

# String representations of timedelta objects are normalized similarly to their internal representation.
# This leads to somewhat unusual results for negative timedeltas.
# For example:
timedelta(hours=-5)
timedelta(days=-5)
timedelta(seconds=-5)

# The most negative timedelta object, timedelta(-999999999).
timedelta.min

# The most positive timedelta object, timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999).
timedelta.max

# The smallest possible difference between non-equal timedelta objects, timedelta(microseconds=1).
timedelta.resolution

# you can move in time using timedelta objects or represent the difference between two time objects.
yesterday = datetime.date.today() - datetime.timedelta(1)
delta = yesterday - datetime.date.today()

# timedelta objects are hashable (usable as dictionary keys), support efficient pickling, and
# in Boolean contexts, a timedelta object is considered to be true if and only if it is not equal to timedelta(0).


# timedelta stores time data in absolute form, so there is no rounding error.
year = timedelta(days=365)
another_year = timedelta(weeks=40, days=84, hours=23,
                          minutes=50, seconds=600)  # adds up to 365 days

# timedelta.total_seconds(): Return the total number of seconds contained in the duration.
year.total_seconds()
another_year.total_seconds()

year == another_year


#######################################
### Create a specific time in a specific timezone
dt.now(tz=pytz.utc).astimezone(pytz.timezone('America/Los_Angeles')).replace(hour=3, minute=15, second=0, microsecond=0)

#######################################
### datetime.tzinfo
# An abstract base class for time zone information objects.


###############################################################################
### USAGE EXAMPLES
# convert from string to datetime object
dt_object = dt.strptime('20151025', '%Y%m%d')

# Convert back to string with desired format
string_date =  dt_object.strftime('%Y-%m-%d')

# date meta-data
dt_object.weekday()

# Either of the two formats below will work
dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)
dt.now(pytz.timezone(timestampTimezone)).strftime(unixFormat)
# or if you want to use UTC
dt.utcnow().strftime(unixFormat)
## or if you want SQL itself to get the current time in MySQL
# CONVERT_TZ(UTC_TIMESTAMP(),'UTC','{timezone}')
# timezone = timestampTimezone
## Get current time in Oracle query
# SELECT CURRENT_TIMESTAMP FROM DUAL
## Conver time to number
# TO_NUMBER(TO_CHAR(TO_TIMESTAMP('{max_date_limit}', 'YYYY-MM-DD HH24:MI:SS'), 'YYYYMMDD'))