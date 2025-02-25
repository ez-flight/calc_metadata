import math
from datetime import date, datetime, timedelta

import numpy as np
import pyorbital
from pyorbital.orbital import XKMPER, F, astronomy

#from sgp4.earth_gravity import wgs84

#pi =math.pi

## Ellipsoid Parameters as tuples (semi major axis, inverse flattening)
grs80 = (6378137, 298.257222100882711)
#wgs84 = (6378137., 1./298.257223563)
wgs84 = (6378137, 298.257223563)


A = 6378.137  # WGS84 Equatorial radius (km)
B = 6356.752314245 # km, WGS84
MFACTOR = 7.292115E-5

def get_xyzv_from_latlon(time, lon, lat, alt):
    """Calculate observer ECI position.
        http://celestrak.com/columns/v02n03/
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (pyorbital.astronomy.gmst(time) + lon) % (2 * np.pi)
    c = 1 / np.sqrt(1 + F * (F - 2) * np.sin(lat)**2)
    sq = c * (1 - F)**2

    achcp = (A * c + alt) * np.cos(lat)
    x = achcp * np.cos(theta)  # kilometers
    y = achcp * np.sin(theta)
    z = (A * sq + alt) * np.sin(lat)

    vx = -MFACTOR * y  # kilometers/second
    vy = MFACTOR * x
    vz = 0

    return (x, y, z), (vx, vy, vz)

def get_lonlatalt(pos, utc_time):
    """Calculate sublon, sublat and altitude of satellite, considering the earth an ellipsoid.
    http://celestrak.com/columns/v02n03/
    """
    (pos_x, pos_y, pos_z) = pos / XKMPER
    lon = ((np.arctan2(pos_y * XKMPER, pos_x * XKMPER) - astronomy.gmst(utc_time)) % (2 * np.pi))
    lon = np.where(lon > np.pi, lon - np.pi * 2, lon)
    lon = np.where(lon <= -np.pi, lon + np.pi * 2, lon)

    r = np.sqrt(pos_x ** 2 + pos_y ** 2)
    lat = np.arctan2(pos_z, r)
    e2 = F * (2 - F)
    while True:
        lat2 = lat
        c = 1 / (np.sqrt(1 - e2 * (np.sin(lat2) ** 2)))
        lat = np.arctan2(pos_z + c * e2 * np.sin(lat2), r)
        if np.all(abs(lat - lat2) < 1e-10):
            break
    alt = r / np.cos(lat) - c
    alt *= A
    return np.rad2deg(lon), np.rad2deg(lat), alt


def _test():
    lat = 55.75583
    lon = 37.6173
    h = 155
    X_msk = 2849.897965
    Y_msk = 2195.949753
    Z_msk = 5249.076832

    delta = timedelta(days=0, seconds=0.5, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    dt_start = datetime.utcnow()
    dt_end = dt_start + timedelta(
        days=1,
        seconds=0,
        microseconds=0,
        milliseconds=0,
        minutes=0,
        hours=0,
        weeks=0
    )
    dt = dt_start
 #   X_isk, Y_isk, Z_isk = geodetic_to_ISK(lat, lon, h, wgs84, dt_start)
  #  print(f"X = {X_isk}, Y = {Y_isk}, Z = {Z_isk} ")

    while dt<dt_end:
#        if calc_gmst(dt) < 0.0001:
#            print(f"1 -> {calc_gmst(dt)} на {dt}")
#        if astronomy.gmst(dt) < 0.0001:
#            print(f"2 -> {astronomy.gmst(dt)} на {dt}")
#        if GMST(dt) < 0.0001:
#            print(f"3 -> {GMST(dt)} на {dt}")      
#        if gmsts(dt) < 0.0001:
#            print(f"4 -> {gmsts(dt)} на {dt}")

      
#        print (p)
#        print (p_0)
        dt += delta



if __name__ == "__main__":
    _test()
