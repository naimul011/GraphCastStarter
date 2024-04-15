import math
import datetime

# The fields to be fetched from the single-level source.
singlelevelfields = [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'geopotential',
                        'land_sea_mask',
                        'mean_sea_level_pressure',
                        'toa_incident_solar_radiation',
                        'total_precipitation'
                    ]

# The fields to be fetched from the pressure-level source.
pressurelevelfields = [
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                        'specific_humidity',
                        'temperature',
                        'vertical_velocity'
                    ]

# The 13 pressure levels.
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# The 37 pressure levels.

#pressure_levels = [1, 2, 3, 5, 7, 10, 20, 30,50,70,100, 125,150, 175,200, 225,250,300, 350,400, 450,500,550,600, 650,700, 750, 775,800, 825,850, 875, 900,925,950, 975, 1000]
# Initializing other required constants.
pi = math.pi
gap = 6 # There is a gap of 6 hours between each graphcast prediction.
predictions_steps = 4 # Predicting for 4 timestamps.
watts_to_joules = 3600
first_prediction = datetime.datetime(2024, 1, 2, 0, 0) # Timestamp of the first prediction.
lat_range = range(-180, 181, 1) # Latitude range.
lon_range = range(0, 360, 1) # Longitude range.

# A utility function used for ease of coding.
# Converting the variable to a datetime object.
def toDatetime(dt) -> datetime.datetime:
    if isinstance(dt, datetime.date) and isinstance(dt, datetime.datetime):
        return dt
    
    elif isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        return datetime.datetime.combine(dt, datetime.datetime.min.time())
    
    elif isinstance(dt, str):
        if 'T' in dt:
            return isodate.parse_datetime(dt)
        else:
            return datetime.datetime.combine(isodate.parse_date(dt), datetime.datetime.min.time())