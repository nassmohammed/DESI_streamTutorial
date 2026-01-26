import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None
import healpy as hp
import scipy as sp
import scipy.stats as stats
from astropy.io import fits
from astropy import table
import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
import matplotlib
import importlib

import emcee
import corner

from collections import OrderedDict
import time

from scipy import optimize, stats
from scipy.stats import expon
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from scipy.interpolate import interp1d
def plot_form(ax):
    ax.grid(ls='-.', alpha=0.2, zorder=0)
    ax.tick_params(direction='in')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.minorticks_on()

def plot_orbit(model_ra, model_dec, stream_ra, stream_dec, ra_prog):
    dec_prog = interp1d(model_ra, model_dec, kind='linear', fill_value='extrapolate')(ra_prog)
    plt.scatter(model_ra, model_dec, c='orange', s=1, label='Orbit Model')
    plt.scatter(stream_ra, stream_dec, c='tab:blue', s=1, label='Stream Data')
    plt.scatter(ra_prog, dec_prog, marker='*', label = 'Progenitor Location', s=100, c='black')
    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$\delta$ [deg]')
    plt.grid(ls='--', alpha=0.3)
    plt.tick_params(axis='both', which='both', direction='in')
    plt.legend()


def orbit_model(theta, ra_prog, ts_rw, ts_ff, values=True, return_o=False):
    ra = ra_prog
    dec, pmra, pmdec, vrad, dist = theta[0:5]

    ra = ra*u.deg
    dec = dec*u.deg
    dist = dist*u.kpc
    pmra_cosdec = pmra*u.mas/u.yr*np.cos(dec)
    pmdec = pmdec*u.mas/u.yr
    vrad = vrad*u.km/u.s

    o_rw = Orbit(vxvv=[ra, dec, dist, pmra_cosdec, pmdec, vrad], radec=True)
    o_rw.integrate(ts_rw, MWPotential2014)
    model_ra_rw, model_dec_rw, model_pmra_rw, model_pmdec_rw, model_vlos_rw, model_dist_rw = o_rw.ra(ts_rw), o_rw.dec(ts_rw), o_rw.pmra(ts_rw), o_rw.pmdec(ts_rw), o_rw.vlos(ts_rw), o_rw.dist(ts_rw) ##!!!! take array of distance

    o_ff = Orbit(vxvv=[ra, dec, dist, pmra_cosdec, pmdec, vrad], radec=True)
    o_ff.integrate(ts_ff, MWPotential2014)
    model_ra_ff, model_dec_ff, model_pmra_ff, model_pmdec_ff, model_vlos_ff, model_dist_ff = o_ff.ra(ts_ff), o_ff.dec(ts_ff), o_ff.pmra(ts_ff), o_ff.pmdec(ts_ff), o_ff.vlos(ts_ff), o_ff.dist(ts_ff) ##!!!! take array of distance

    model_ra = np.concatenate([model_ra_rw, model_ra_ff])
    model_dec = np.concatenate([model_dec_rw, model_dec_ff])
    model_pmra = np.concatenate([model_pmra_rw, model_pmra_ff])
    model_pmdec = np.concatenate([model_pmdec_rw, model_pmdec_ff])
    model_vlos = np.concatenate([model_vlos_rw, model_vlos_ff])
    model_dist = np.concatenate([model_dist_rw, model_dist_ff])
    if return_o:
        return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist, o_rw
    else:
        return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist

def zero_360_to_180(ra):
    ra_copy = np.copy(ra)
    where_180 = np.where(ra_copy > 180)
    ra_copy[where_180] = ra_copy[where_180] - 360
        
    return ra_copy

def change_ra_range(ra):
    ra_copy = np.copy(ra)
    if np.any(ra_copy > 350):
        where_50 = np.where(ra_copy < 50)
        ra_copy[where_50] = ra_copy[where_50] + 360
        
    return ra_copy

def mas_yr_to_km_s(pm, dist):
    '''
    Input:
        pm: proper motion in mas/yr
        dist: distance in kpc
        
    Output:
        pm in km/s
    '''
    return pm * dist * 4.74

def betw(y_data, fit_data, delta):
    return (y_data > fit_data - delta) & (y_data < fit_data + delta)

# def isochrone_btw(x, y, x_wiggle, y_wiggle, isochrone):
#     x_left, x_right = isochrone - x_wiggle, isochrone + x_wiggle
#     y_up, y_down = isochrone + y_wiggle, isochrone - y_wiggle
    
#     return (((x > x_left) | (y < y_up)) & ((x < x_right) | (y > y_down)))

def isochrone_btw(data_color, data_mag, x_wiggle, y_wiggle, iso_color, iso_mag):

    # Create upper and lower shifted curves
    upper_curve = np.column_stack((iso_color, iso_mag + y_wiggle))
    lower_curve = np.column_stack((iso_color, iso_mag - y_wiggle))
    
    left_curve = np.column_stack((iso_color - x_wiggle, iso_mag))
    right_curve = np.column_stack((iso_color + x_wiggle, iso_mag))

    # Form a closed polygon region
    # y_polygon = np.vstack([upper_curve, lower_curve[::-1]])
    y_polygon = np.vstack([lower_curve, right_curve[::-1]])
    y_poly_path = Path(y_polygon)
    
    x_polygon = np.vstack([left_curve, upper_curve[::-1]])
    x_poly_path = Path(x_polygon)

    # Stack data points into (x, y)
    data_points = np.column_stack((data_color, data_mag))

    # Return mask of points inside polygon
    # return (y_poly_path.contains_points(data_points))
    return ((x_poly_path.contains_points(data_points)) | (y_poly_path.contains_points(data_points)))

def load_fits_columns(file_name, columns=None, hdu_indices=None):
    final_data = {}
    added_columns = set()  # To track added columns
    with fits.open(file_name, memmap=True) as hdulist:
        if hdu_indices is None:
            hdu_indices = range(len(hdulist))
        for hdu_index in hdu_indices:
            hdu = hdulist[hdu_index]
            if isinstance(hdu.data, fits.FITS_rec):
                data = hdu.data
                if columns:
                    available_columns = set(columns) & set(data.names)  # Find common columns
                else:
                    available_columns = set(data.names)  # Use all columns if none specified
                for col in available_columns:
                    if col not in added_columns:  # Check if column is already added
                        final_data[col] = data[col]
                        added_columns.add(col)  # Mark column as added
                        
    return table.Table(final_data)

def get_rot_matrix(ra1, dec1, ra2, dec2):
    [phi, theta, psi] = euler_angles(ra1, dec1, ra2, dec2)
    rot_matrix = np.diag([1.0, 1.0, 1.0])@create_matrix(phi,theta,psi)
    return rot_matrix

def dist_to_dist_mod(dist):
    return 5 * np.log10(dist) - 5
    
    
def find_pole(lon1,lat1,lon2,lat2):
    """ Find the pole of a great circle orbit between two points.

    Parameters:
    -----------
    lon1 : longitude of the first point (deg)
    lat1 : latitude of the first point (deg)
    lon2 : longitude of the second point (deg)
    lat2 : latitude of the second point (deg)

    Returns:
    --------
    lon,lat : longitude and latitude of the pole
    """
    vec = np.cross(hp.ang2vec(lon1,lat1,lonlat=True),
                   hp.ang2vec(lon2,lat2,lonlat=True))
    lon,lat = hp.vec2ang(vec,lonlat=True)
    return [np.ndarray.item(lon),np.ndarray.item(lat)]

def create_matrix(phi,theta,psi):
    """ Create the transformation matrix.
    """
    # Generate the rotation matrix using the x-convention (see Goldstein)
    D = rotation_matrix(np.radians(phi),   "z", unit=u.radian)
    C = rotation_matrix(np.radians(theta), "x", unit=u.radian)
    B = rotation_matrix(np.radians(psi),   "z", unit=u.radian)
    return  np.array(B.dot(C).dot(D))
    
def euler_angles(lon1,lat1,lon2,lat2):
    """ Calculate the Euler angles for spherical rotation using the x-convention
    (see Goldstein).

    Parameters:
    -----------
    lon1 : longitude of the first point (deg)
    lat1 : latitude of the first point (deg)
    lon2 : longitude of the second point (deg)
    lat2 : latitude of the second point (deg)

    Returns:
    --------
    phi,theta,psi : rotation angles around Z,X,Z
    """
    pole = find_pole(lon1,lat1,lon2,lat2)
     
    # Initial rotation
    phi   = pole[0] - 90.
    theta = pole[1] + 90.
    psi   = 0.
     
    matrix = create_matrix(phi,theta,psi)
    ## Generate the rotation matrix using the x-convention (see Goldstein)
    #D = rotation_matrix(np.radians(phi),   "z", unit=u.radian)
    #C = rotation_matrix(np.radians(theta), "x", unit=u.radian)
    #B = rotation_matrix(np.radians(psi),   "z", unit=u.radian)
    #MATRIX = np.array(B.dot(C).dot(D))

    lon = np.radians([lon1,lon2])
    lat = np.radians([lat1,lat2])

    X = np.cos(lat)*np.cos(lon)
    Y = np.cos(lat)*np.sin(lon)
    Z = np.sin(lat)

    # Calculate X,Y,Z,distance in the stream system
    Xs, Ys, Zs = matrix.dot(np.array([X, Y, Z]))
    Zs = -Zs

    # Calculate the transformed longitude
    Lambda = np.arctan2(Ys,Xs)
    Lambda[Lambda < 0] = Lambda[Lambda < 0] + 2.*np.pi
    psi = float(np.mean(np.degrees(Lambda)))

    return [phi, theta, psi]


# Function to drop rows with NaNs or None values in specified columns
def dropna_Table(table, columns=None):
    if columns is None:
        columns = table.colnames
    mask = np.ones(len(table), dtype=bool)
    for col in columns:
        column_data = np.array(table[col].value)
        # Check if the column is of a numeric type
        if np.issubdtype(column_data.dtype, np.number):
            mask &= ~np.isnan(column_data)
        # Check if the column is a string type
        elif np.issubdtype(column_data.dtype, np.str_):
            mask &= (column_data != None) & (column_data != '') & (column_data != 'nan')
        else:
            mask &= column_data != None
    return table[mask]

def drop_duplicates_Table(table, columns):
    # Create a structured array based on the specified columns
    structured_array = table[columns].as_array()
    # Find unique rows and their indices
    _, unique_indices = np.unique(structured_array, axis=0, return_index=True)
    # Sort the unique indices to maintain order
    unique_indices.sort()
    # Select rows corresponding to unique indices
    unique_table = table[unique_indices]
    return unique_table

def zero_to_360(ra, min_ra_to_consider = 20):
    ra_copy = np.copy(ra)
    where_below_min_ra = np.where(ra_copy < min_ra_to_consider)
    ra_copy[where_below_min_ra] = ra_copy[where_below_min_ra] + 360
    
    return ra_copy

def threeD_max_min_mask(ra_data, dec_data, plx, plx_err, ra_max_min_data, dec_max_min_data, min_dist, ra_wiggle = 5, dec_wiggle = 5):
    '''
    Add a mask to the data by cutting out the stars outside of the distance you want to look at. Note the default wiggle room is 5 degrees for RA and DEC, and pc for distance.
    
    Input:
        ra_data: right ascension data
        dec_data: declination data
        dist_data: distance data
        ra_max_min_data: the maximum and minimum ra data you want to keep
        dec_max_min_data: the maximum and minimum declination data you want to keep
        dist_max_min_data: the maximum and minimum distance data you want to keep
        ra_wiggle: the amount of extra ra data you want to keep
        dec_wiggle: the amount of extra declination data you want to keep
        dist_wiggle: the amount of extra distance data you want to keep
        
    Output:
        mask: the mask that will be applied to the data
    '''
    # Obtain the maximum and minumum values with a bit of wiggle room to compare the new data to
    ra_max = np.max(ra_max_min_data) + ra_wiggle
    ra_min = np.min(ra_max_min_data) - ra_wiggle
    dec_max = np.max(dec_max_min_data) + dec_wiggle
    dec_min = np.min(dec_max_min_data) - dec_wiggle
    if (ra_max > 360):
        ra_max = ra_max - 360
        mask = (((ra_data > ra_min) | (ra_data < ra_max)) & ((dec_data > dec_min) &  (dec_data < dec_max)) & ((plx - 2 * plx_err) < (1/min_dist))) # & (dist_data < dist_max)
    elif ra_min < 0:
        ra_min = ra_min + 360
        mask = (((ra_data > ra_min) | (ra_data < ra_max)) & ((dec_data > dec_min) &  (dec_data < dec_max)) & ((plx - 2 * plx_err) < (1/min_dist))) # & (dist_data < dist_max)
    else:
        # Check whether the data is larger or smaller than the maximum and minimum values
        mask = (((ra_data > ra_min) & (ra_data < ra_max)) & ((dec_data > dec_min) &  (dec_data < dec_max)) & ((plx - 2 * plx_err) < (1/min_dist))) # & (dist_data < dist_max)
    
    return mask


def dist_mod_to_dist(dist_mod):
    return 10 ** ((dist_mod + 5) / 5)

def phi12_rotmat(alpha,delta,R_phi12_radec):
    '''
    Converts coordinates (alpha,delta) to ones defined by a rotation matrix R_phi12_radec, applied on the original coordinates

    Critical: All angles must be in degrees
    
    Parameters:
    alpha - float. Right ascension
    delta - float. Declination
    R_phi12_radec - 3x3 array. Rotation matrix
    
    Returns:
    [phi1,phi2] - the transformed stream coordinates as a size 2 array.
    '''
    
    vec_radec = np.array([np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),
                          np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),
                          np.sin(delta*np.pi/180.)])

    vec_phi12 = np.zeros(np.shape(vec_radec))
    
    vec_phi12[0] = np.sum(R_phi12_radec[0][i]*vec_radec[i] for i in range(3))
    vec_phi12[1] = np.sum(R_phi12_radec[1][i]*vec_radec[i] for i in range(3))
    vec_phi12[2] = np.sum(R_phi12_radec[2][i]*vec_radec[i] for i in range(3))
    
    vec_phi12 = vec_phi12.T

    vec_phi12 = np.dot(R_phi12_radec,vec_radec).T

    phi1 = np.arctan2(vec_phi12[:,1],vec_phi12[:,0])*180./np.pi
    phi2 = np.arcsin(vec_phi12[:,2])*180./np.pi

    return [phi1,phi2]

def pmphi12(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec):
    '''
    Converts proper motions (mu_alpha_cos_delta,mu_delta) to those in coordinates defined by the rotation matrix, R_phi12_radec, applied to the original coordinates

    Critical: All angles must be in degrees
    '''
    
    k_mu = 4.74047

    phi1,phi2 = phi12_rotmat(alpha,delta,R_phi12_radec)


    r = np.ones(len(alpha))

    vec_v_radec = np.array([np.zeros(len(alpha)),k_mu*mu_alpha_cos_delta*r,k_mu*mu_delta*r]).T

    worker = np.zeros((len(alpha),3))

    worker[:,0] = ( np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.)*vec_v_radec[:,0]
                   -np.sin(alpha*np.pi/180.)*vec_v_radec[:,1]
                   -np.cos(alpha*np.pi/180.)*np.sin(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker[:,1] = ( np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.)*vec_v_radec[:,0]
                   +np.cos(alpha*np.pi/180.)*vec_v_radec[:,1]
                   -np.sin(alpha*np.pi/180.)*np.sin(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker[:,2] = ( np.sin(delta*np.pi/180.)*vec_v_radec[:,0]
                   +np.cos(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker2 = np.zeros((len(alpha),3))

    worker2[:,0] = np.sum(R_phi12_radec[0][axis]*worker[:,axis] for axis in range(3))
    worker2[:,1] = np.sum(R_phi12_radec[1][axis]*worker[:,axis] for axis in range(3))
    worker2[:,2] = np.sum(R_phi12_radec[2][axis]*worker[:,axis] for axis in range(3))

    worker[:,0] = ( np.cos(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*worker2[:,0]
                   +np.sin(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*worker2[:,1]
                   +np.sin(phi2*np.pi/180.)*worker2[:,2] )

    worker[:,1] = (-np.sin(phi1*np.pi/180.)*worker2[:,0]
                   +np.cos(phi1*np.pi/180.)*worker2[:,1] )
                   

    worker[:,2] = (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*worker2[:,0]
                   -np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*worker2[:,1]
                   +np.cos(phi2*np.pi/180.)*worker2[:,2] )

    mu_phi1_cos_delta = worker[:,1]/(k_mu*r)
    mu_phi2 = worker[:,2]/(k_mu*r)

    return mu_phi1_cos_delta, mu_phi2

def pmphi12_reflex(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec,dist,vlsr = np.array([ 12.9, 245.6 , 7.78])):
    
    ''' 
    returns proper motions in coordinates defined by R_phi12_radec transformation corrected by the Sun's reflex motion
    all angles must be in degrees
     vlsr = np.array([11.1,240.,7.3]) 
    '''

    k_mu = 4.74047

    a_g = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                    [-0.8734370902, -0.4448296300, -0.1980763734], 
                    [-0.4838350155, 0.7469822445, +0.4559837762]])

    nvlsr = -vlsr

    phi1, phi2 = phi12_rotmat(alpha,delta,R_phi12_radec)

    phi1 = phi1*np.pi/180.
    phi2 = phi2*np.pi/180.

    pmphi1, pmphi2 = pmphi12(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec)

    M_UVW_phi12 = np.array([[np.cos(phi1)*np.cos(phi2),-np.sin(phi1),-np.cos(phi1)*np.sin(phi2)],
                            [np.sin(phi1)*np.cos(phi2), np.cos(phi1),-np.sin(phi1)*np.sin(phi2)],
                            [     np.sin(phi2)        ,      0. *np.ones_like(phi2)    , np.cos(phi2)]])

    vec_nvlsr_phi12 = np.dot(M_UVW_phi12.T,np.dot(R_phi12_radec,np.dot(a_g,nvlsr)))

    return pmphi1 - vec_nvlsr_phi12[:,1]/(k_mu*dist), pmphi2 - vec_nvlsr_phi12[:,2]/(k_mu*dist)

def ra_dec_to_phi1_phi2(frame, ra, dec):
    '''
    Given a frame, convert ra and dec to phi1 and phi2
    
    Input:
        frame: astropy.coordinates frame
        ra: right ascension in degrees
        dec: declination in degrees
        
    Output:
        phi1: stream phi1 coordinates in degrees
        phi2: stream phi2 coordinates in degrees
    '''
    skycoord_data = coord.SkyCoord(ra=ra, dec=dec, frame='icrs')
    transformed_skycoord = skycoord_data.transform_to(frame)
    phi1, phi2 = transformed_skycoord.phi1.deg, transformed_skycoord.phi2.deg
    return phi1, phi2   

def pmra_pmdec_to_pmphi12(frame, ra, dec, pmra, pmdec):
    '''
    Given a frame, convert ra, dec, pmra, and pmdec to pmphi1 and pmphi2
    
    Input:
        frame: astropy.coordinates frame
        ra: right ascension in degrees
        dec: declination in degrees
        pmra: proper motion in right ascension in mas/yr
        pmdec: proper motion in declination in mas/yr
        
    Output:
        pmphi1: stream pmphi1 in mas/yr
        pmphi2: stream pmphi2 in mas/yr
    '''
    skycoord_data = coord.SkyCoord(ra=ra, dec=dec, pm_ra_cosdec=pmra, pm_dec=pmdec, frame='icrs')
    transformed_skycoord = skycoord_data.transform_to(frame)
    pmphi1, pmphi2 = transformed_skycoord.pm_phi1_cosphi2.to(u.mas/u.yr).value, transformed_skycoord.pm_phi2.to(u.mas/u.yr).value
    return pmphi1, pmphi2


def stream_data_cutting(stream_ra, stream_dec, data_ra, data_dec, radius = 0.5):
    '''
    Cut down a big dataset to only include data points within a certain radius of a stellar stream

    input:
        stream_ra: float, ra of the stream in degrees
        stream_dec: float, dec of the stream in degrees
        data_ra: array, ra of the data in degrees
        data_dec: array, dec of the data in degrees
        radius: float, radius in degrees

    output:
        data_mask: boolean array, True if the data point is within the radius of the stream (size of data_ra/data_dec)
    '''
    
    # Expanding dimensions for broadcasting
    expanded_stream_ra = stream_ra[np.newaxis, :]
    expanded_stream_dec = stream_dec[np.newaxis, :]
    expanded_data_ra = data_ra[:, np.newaxis]
    expanded_data_dec = data_dec[:, np.newaxis]
    
    # Calculate squared distances
    distance_squared = (expanded_stream_ra - expanded_data_ra)**2 + (expanded_stream_dec - expanded_data_dec)**2
    radius_squared = radius**2
    
    # Check within radius without using sqrt
    mask = distance_squared < radius_squared
    
    # Any point within the radius
    data_mask = np.any(mask, axis=1)
    
    return data_mask

def quality_cut_mask(data, vgsr, max_feh, min_feh, max_vgsr, min_vgsr, max_pmra, min_pmra, max_pmdec, min_pmdec, max_vrad_err, max_feh_err, max_pmra_err, max_pmdec_err):
    '''
    Function to apply a mask to a dataset based on the metallicities, radial velocities, proper motions, and their errors.
    
    Input:
        data (DataFrame): Dataset with all of the metallicity, radial velocity, proper motion, and error values.
        max_feh (float): Maximum metallicity value to keep in the dataset.
        max_vgsr (float): Maximum radial velocity value to keep in the dataset.
        min_vgsr (float): Minimum radial velocity value to keep in the dataset.
        max_pmra (float): Maximum proper motion in RA value to keep in the dataset.
        min_pmra (float): Minimum proper motion in RA value to keep in the dataset.
        max_pmdec (float): Maximum proper motion in DEC value to keep in the dataset.
        min_pmdec (float): Minimum proper motion in DEC value to keep in the dataset.
        max_vrad_err (float): Maximum radial velocity error value to keep in the dataset.
        max_feh_err (float): Maximum metallicity error value to keep in the dataset.
        max_pmra_err (float): Maximum proper motion in RA error value to keep in the dataset.
        max_pmdec_err (float): Maximum proper motion in DEC error value to keep in the dataset.
        
    Output:
        quality_cuts (array): Boolean array to mask the dataset with the quality cuts. 
    '''
    quality_cuts = (data['FEH'] < max_feh) & (data['FEH'] > min_feh) \
               & (vgsr < max_vgsr) & (vgsr > min_vgsr) \
               & (data['PMRA'] < max_pmra) & (data['PMRA'] > min_pmra) \
               & (data['PMDEC'] < max_pmdec) & (data['PMDEC'] > min_pmdec) \
               & (data['VRAD_ERR'] < max_vrad_err) & (data['FEH_ERR'] < max_feh_err) \
                & (data['PMRA_ERROR'] < max_pmra_err) & (data['PMDEC_ERROR'] < max_pmdec_err)
                
    return quality_cuts

def rv_to_gsr(c, v_sun=None):
    """Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

    The input radial velocity must be passed in as a

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()

    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj

def vhel_to_vgsr(data_ra, data_dec, data_vhel):
    '''
    Convert heliocentric velocities to Galactocentric velocities.
    
    Input:
        vhel (array): Array of heliocentric radial velocities. [km/s]
        ra (array): Array of right ascension values. [deg]
        dec (array): Array of declination values. [deg]
        
    Output:
        vgsr (array): Array of Galactocentric radial velocities
    '''
    if not isinstance(data_vhel, u.quantity.Quantity):
        data_vhel = data_vhel * u.km/u.s
        
    if not isinstance(data_ra, u.quantity.Quantity):
        data_ra = data_ra * u.deg
        
    if not isinstance(data_dec, u.quantity.Quantity):
        data_dec = data_dec * u.deg
        
    icrs = coord.SkyCoord(ra=data_ra, dec=data_dec, radial_velocity=data_vhel, frame='icrs')
    data_vgsr = rv_to_gsr(icrs)
        
    return data_vgsr

def get_colour_index_and_abs_mag(ebv, g_flux, r_flux, distances):
    '''
    Obtain a mask of the dataset based on a CMD isochrone cut.
    
    Input:
        ebv (array): Array of E(B-V) values
        g_flux (array): Array of g-band flux values
        r_flux (array): Array of r-band flux values
        z_flux (array): Array of z-band flux values
        distances (array): Array of distances in kpc
        
    Output:
        color_index (array): Array of g-r 
        G_mag (array): Array of absolute G magnitudes
    '''
    # Create copies to avoid SettingWithCopyWarning
    ebv = np.array(ebv.copy()) if hasattr(ebv, 'copy') else np.array(ebv)
    g_flux = np.array(g_flux.copy()) if hasattr(g_flux, 'copy') else np.array(g_flux)
    r_flux = np.array(r_flux.copy()) if hasattr(r_flux, 'copy') else np.array(r_flux)
    
    flux_mask = (g_flux > 0) & (r_flux > 0)
    ebv[~flux_mask] = np.nan # extinction comes from colour excess between b and v bands -> name of paper?, also Planck
    g_flux[~flux_mask] = np.nan
    r_flux[~flux_mask] = np.nan
    
    # The extinction coefficients for the g, and r band in DESI data
    ext_c = {'g': 3.237, 'r': 2.176} # comes from DESI filter <--- double check, Ting did them differently ! they need to go into the paper
    eg, er = [ext_c[_] * ebv for _ in 'gr']

    ext = {'g': eg, 'r': er}
    
    g_mag = 22.5 - 2.5 * np.log10(g_flux) - np.array(ext['g'])
    r_mag = 22.5 - 2.5 * np.log10(r_flux) - np.array(ext['r'])

    colour_index =  g_mag - r_mag

    if distances is None:
        # No distance information -> absolute magnitude undefined
        R_mag = np.full_like(r_mag, np.nan, dtype=float)
    else:
        # np.asarray allows a copy when needed (compatible with numpy>=2.0)
        distances = np.asarray(distances)
        R_mag = r_mag - 5 * np.log10(distances) + 5  

    return colour_index, R_mag, r_mag

    
def quad_f(phi,c1,c2,c3):
    '''
    Quadratic function
    
    Parameters are all floats, returns a float
    '''
    x = phi/10
    return c1 + c2*x + c3*x**2

def logpdf_2dnorm(x1,x2,m1,m2,u1,u2,s1,s2,c):
    v11 = s1**2 + u1**2
    v22 = s2**2 + u2**2
    v12 = c*u1*u2
    detcov = v11*v22 - v12**2
    d1 = x1 - m1
    d2 = x2 - m2
    
    a = ((2*np.pi)**(-1)) * (detcov**(-1/2))
    b = (-1/2)*(1/detcov)*(d2*(d2*v11-d1*v12)+d1*(d1*v22-d2*v12))
    return np.log(a) + b

param_labels = ["lpstream",
                "v1","v2","v3","lsigv",
                "feh1","lsigfeh",
                "pmra1","pmra2","pmra3","lsigpmra",
                "pmdec1","pmdec2","pmdec3","lsigpmdec",
                "bv", "lsigbv", "bfeh", "lsigbfeh", "bpmra", "lsigbpmra", "bpmdec", "lsigbpmdec"]


def get_paramdict(theta, labels = param_labels):
    '''Make an ordered dictionary of the parameters as keys and inputted theta as values'''
    return OrderedDict(zip(labels, theta))

def lnprob(theta, prior, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1, pmcorr, trunc_fit = False, assert_prior = False, feh_fit=True):
    """ Likelihood and Prior """
    
    # params
    #pstream, \
    lpstream, \
    v1, v2, v3, lsigv, \
    feh1, lsigfeh, \
    pmra1, pmra2, pmra3, lsigpmra, \
    pmdec1, pmdec2, pmdec3, lsigpmdec, \
    bv, lsigbv, bfeh, lsigbfeh, bpmra, lsigbpmra, bpmdec, lsigbpmdec = theta
    
    v1_min, v1_max, v2_min, v2_max, v3_min, v3_max, lsigv_min, lsigv_max,\
    feh1_min, feh1_max, lsigfeh_min, lsigfeh_max,\
    pmra1_min, pmra1_max, pmra2_min, pmra2_max, pmra3_min, pmra3_max, lsigpmra_min, lsigpmra_max,\
    pmdec1_min, pmdec1_max, pmdec2_min, pmdec2_max, pmdec3_min, pmdec3_max, lsigpmdec_min, lsigpmdec_max,\
    bv_min, bv_max, bfeh_min, bfeh_max, bpmra_min, bpmra_max, bpmdec_min, bpmdec_max,\
    lsigbv_min, lsigbv_max, lsigbfeh_min, lsigbfeh_max, lsigbpmra_min, lsigbpmra_max, lsigbpmdec_min, lsigbpmdec_max = prior
    
    if feh_fit == False:
        feh1_min, feh1_max, lsigfeh_min, lsigfeh_max, bfeh_min, bfeh_max = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
    
    # The prior
    if (
        (lpstream > 0) or 
        # (lsigv > 4) or (lsigfeh > 4) or (lsigpmra > 1) or (lsigpmdec > 0.5) or 
        # (lsigv < -1) or (lsigfeh < -2) or (lsigpmra < -2) or (lsigpmdec < -3) or 
        # (lsigbv > 4) or (lsigbfeh > 3) or (lsigbpmra > 3) or (lsigbpmdec > 3) or 
        # (lsigbv < -1) or (lsigbfeh < -1) or (lsigbpmra < -2) or (lsigbpmdec < -2) or 
        (lsigv > lsigv_max) or (lsigfeh > lsigfeh_max) or (lsigpmra > lsigpmra_max) or (lsigpmdec > lsigpmdec_max) or 
        (lsigv < lsigv_min) or (lsigfeh < lsigfeh_min) or (lsigpmra < lsigpmra_min) or (lsigpmdec < lsigpmdec_min) or 
        (lsigbv > lsigbv_max) or (lsigbfeh > lsigbfeh_max) or (lsigbpmra > lsigbpmra_max) or (lsigbpmdec > lsigbpmdec_max) or 
        (lsigbv < lsigbv_min) or (lsigbfeh < lsigbfeh_min) or (lsigbpmra < lsigbpmra_min) or (lsigbpmdec < lsigbpmdec_min) or 
        (v1 > v1_max) or (v1 < v1_min) or (v2 > v2_max) or (v2 < v2_min) or (v3 > v3_max) or (v3 < v3_min) or 
        (feh1 > feh1_max) or (feh1 < feh1_min) or 
        (pmra1 > pmra1_max) or (pmra1 < pmra1_min) or (pmra2 > pmra2_max) or (pmra2 < pmra2_min) or (pmra3 > pmra3_max) or (pmra3 < pmra3_min) or 
        (pmdec1 > pmdec1_max) or (pmdec1 < pmdec1_min) or (pmdec2 > pmdec2_max) or (pmdec2 < pmdec2_min) or (pmdec3 > pmdec3_max) or (pmdec3 < pmdec3_min) or 
        (bv > bv_max) or (bv < bv_min) or (bfeh > bfeh_max) or (bfeh < bfeh_min) or (bpmra > bpmra_max) or (bpmra < bpmra_min) or (bpmdec > bpmdec_max) or (bpmdec < bpmdec_min)
    ): 
        if assert_prior:
            index = [
            (lpstream > 0), 
            (lsigv > lsigv_max) or (lsigv < lsigv_min) , (lsigfeh > lsigfeh_max) or (lsigfeh < lsigfeh_min), (lsigpmra > lsigpmra_max) or (lsigpmra < lsigpmra_min), (lsigpmdec > lsigpmdec_max) or (lsigpmdec < lsigpmdec_min),
            (lsigbv > lsigbv_max) or (lsigbv < lsigbv_min), (lsigbfeh > lsigbfeh_max) or (lsigbfeh < lsigbfeh_min), (lsigbpmra > lsigbpmra_max) or (lsigbpmra < lsigbpmra_min), (lsigbpmdec > lsigbpmdec_max) or (lsigbpmdec < lsigbpmdec_min),
            (v1 > v1_max) or (v1 < v1_min), (v2 > v2_max) or (v2 < v2_min), (v3 > v3_max) or (v3 < v3_min), 
            (feh1 > feh1_max) or (feh1 < feh1_min), 
            (pmra1 > pmra1_max) or (pmra1 < pmra1_min), (pmra2 > pmra2_max) or (pmra2 < pmra2_min), (pmra3 > pmra3_max) or (pmra3 < pmra3_min),
            (pmdec1 > pmdec1_max) or (pmdec1 < pmdec1_min), (pmdec2 > pmdec2_max) or (pmdec2 < pmdec2_min), (pmdec3 > pmdec3_max) or (pmdec3 < pmdec3_min),
            (bv > bv_max) or (bv < bv_min), (bfeh > bfeh_max) or (bfeh < bfeh_min), (bpmra > bpmra_max) or (bpmra < bpmra_min), (bpmdec > bpmdec_max) or (bpmdec < bpmdec_min)
            ]
            
            print(np.array([f'pstream: {10**lpstream}', 'lsigv', 'lsigfeh', 'lsigpmra', 'lsigpmdec', 'lsigbv', 'lsigbfeh', 'lsigbpmra', 'lsigbpmdec', 'v1', 'v2', 'v3', 'feh1',\
                    'pmra1', 'pmra2', 'pmra3', 'pmdec1', 'pmdec2', 'pmdec3', 'bv', 'bfeh', 'bpmra', 'bpmdec'])[index])

        return -1e10  # outside of prior, return a tiny number
    
    if feh_fit:
        scale_stream_feh = np.sqrt(feh_err**2 + (10**lsigfeh)**2)
        scale_bg_feh = np.sqrt(feh_err**2 + (10**lsigbfeh)**2)
        
    scale_stream_vgsr = np.sqrt(vgsr_err**2 + (10**lsigv)**2)
    scale_bg_vgsr = np.sqrt(vgsr_err**2 + (10**lsigbv)**2)
    
    ## Compute log likelihood in feh
    if trunc_fit == False:
        ## Compute log likelihood in v_gsr
        lstream_v = stats.norm.logpdf(vgsr, loc=quad_f(phi1,v1,v2,v3), scale=scale_stream_vgsr)
        lbg_v = stats.norm.logpdf(vgsr, loc=bv, scale=scale_bg_vgsr)
        
        if feh_fit:
            lstream_feh = stats.norm.logpdf(feh, loc=feh1, scale=scale_stream_feh)
            lbg_feh = stats.norm.logpdf(feh, loc=bfeh, scale=scale_bg_feh)
        
    elif trunc_fit:
        # Compute standardized bounds for truncnorm
        min_trunc_vgsr, max_trunc_vgsr = np.min(vgsr), np.max(vgsr)
        lvgsr_cdf_dif = np.log(stats.norm.cdf(max_trunc_vgsr, loc=bv, scale=scale_bg_vgsr) - stats.norm.cdf(min_trunc_vgsr, loc=bv, scale=scale_bg_vgsr))
        lstream_v = stats.norm.logpdf(vgsr, loc=quad_f(phi1,v1,v2,v3), scale=scale_stream_vgsr)
        lbg_v = stats.norm.logpdf(vgsr, loc=bv, scale=scale_bg_vgsr) - lvgsr_cdf_dif
                
        if feh_fit:
            min_trunc_feh, max_trunc_feh = np.min(feh), np.max(feh)
            lfeh_cdf_dif = np.log(stats.norm.cdf(max_trunc_feh, loc=bfeh, scale=scale_bg_feh) - stats.norm.cdf(min_trunc_feh, loc=bfeh, scale=scale_bg_feh))
            lstream_feh = stats.norm.logpdf(feh, loc=feh1, scale=scale_stream_feh)
            lbg_feh = stats.norm.logpdf(feh, loc=bfeh, scale=scale_bg_feh) - lfeh_cdf_dif
    
    ## Compute log likelihood in pm
    lstream_pm = logpdf_2dnorm(pmra,pmdec,
                               quad_f(phi1,pmra1,pmra2,pmra3),quad_f(phi1,pmdec1,pmdec2,pmdec3),
                               pmra_err,pmdec_err,
                               10**lsigpmra,10**lsigpmdec,
                               pmcorr)
    lbg_pm = logpdf_2dnorm(pmra,pmdec,
                            bpmra,bpmdec,
                            pmra_err, pmdec_err,
                            10**lsigbpmra,10**lsigbpmdec,
                            pmcorr)

    if feh_fit:
        ## Combine the components
        lstream = np.log(10**lpstream) + lstream_v + lstream_feh + lstream_pm
        lbg = np.log(1-(10**lpstream)) + lbg_v + lbg_feh + lbg_pm

    else:
        lstream = np.log(10**lpstream) + lstream_v + lstream_pm
        lbg = np.log(1-(10**lpstream)) + lbg_v + lbg_pm
        
    ltot = np.logaddexp(lstream, lbg)

    return np.sum(ltot)

def project_model(theta, vgsr_min, vgsr_max, feh_min, feh_max, pmra_min, pmra_max, pmdec_min, pmdec_max, trunc_fit=True, feh_fit=True):
    """ Turn parameters into vgsr, pmra, pmdec distributions. An aproximation since no phi dependence"""
    params = get_paramdict(theta)
    
    pstream = 10**params['lpstream']
    
    vgsr_arr = np.linspace(vgsr_min-50, vgsr_max+50, 1000)
    pvgsr0 = pstream*stats.norm.pdf(vgsr_arr, loc=params['v1'], scale=10**params['lsigv'])
    
    feh_arr = np.linspace(feh_min-0.5, feh_max+0.5, 1000)
    if feh_fit:
        pfeh0 = pstream*stats.norm.pdf(feh_arr, loc=params['feh1'], scale=10**params['lsigfeh'])
    elif feh_fit == False:
        pfeh0 = 0
    
    pmra_arr = np.linspace(pmra_min-10, pmra_max+10, 1000)
    ppmra0 = pstream*stats.norm.pdf(pmra_arr, loc=params['pmra1'], scale=10**params['lsigpmra'])
    
    pmdec_arr = np.linspace(pmdec_min-10, pmdec_max+10, 1000)
    ppmdec0 = pstream*stats.norm.pdf(pmdec_arr, loc=params['pmdec1'], scale=10**params['lsigpmdec'])
    
    if trunc_fit:
        pvgsr1 = (1-pstream)*stats.truncnorm.pdf(vgsr_arr, a=(vgsr_min-params['bv'])/10**params['lsigbv'], b=(vgsr_max-params['bv'])/10**params['lsigbv'], loc=params['bv'], scale=10**params['lsigbv'])
        if feh_fit:
            pfeh1 = (1-pstream)*stats.truncnorm.pdf(feh_arr, a=(feh_min-params['bfeh'])/10**params['lsigbfeh'], b=(feh_max-params['bfeh'])/10**params['lsigbfeh'], loc=params['bfeh'], scale=10**params['lsigbfeh'])
        elif feh_fit == False:
            pfeh1 = 0
        
    else:
        pvgsr1 = (1-pstream)*stats.norm.pdf(vgsr_arr, loc=params['bv'], scale=10**params['lsigbv'])
        if feh_fit:
            pfeh1 = (1-pstream)*stats.norm.pdf(feh_arr, loc=params['bfeh'], scale=10**params['lsigbfeh'])
        elif feh_fit == False:
            pfeh1 = 0
    
    ppmra1 = (1-pstream)*stats.norm.pdf(pmra_arr, loc=params['bpmra'], scale=10**params['lsigbpmra'])
    ppmdec1 = (1-pstream)*stats.norm.pdf(pmdec_arr, loc=params['bpmdec'], scale=10**params['lsigbpmdec'])
    
    return vgsr_arr, pvgsr0, pvgsr1, feh_arr, pfeh0, pfeh1, pmra_arr, ppmra0, ppmra1, pmdec_arr, ppmdec0, ppmdec1

def plot_1d_distrs(theta, vgsr, feh, pmra, pmdec, trunc_fit=True, feh_fit=True, streamfinder_table=None, orbsub=False):
    '''
    Plots the distributions of vgsr, feh, pmra, pmdec
    '''
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    if trunc_fit:
        model_output = project_model(theta, np.min(vgsr), np.max(vgsr), np.min(feh),np.max(feh), np.min(pmra),np.max(pmra),np.min(pmdec),np.max(pmdec), trunc_fit=trunc_fit, feh_fit=feh_fit)
        
    else:
        model_output = project_model(theta, np.min(vgsr),np.max(vgsr),np.min(feh),np.max(feh),np.min(pmra),np.max(pmra),np.min(pmdec),np.max(pmdec), feh_fit=feh_fit)
    
    scaled_down_sf3_density = 10
    
    
    fig, axes = plt.subplots(2,2,figsize=(12,12))
    
    #vgsr
    ax = axes[0][0]
    ax.hist(vgsr, density=True, color='lightgrey', bins=50)
    if streamfinder_table is not None:
        if orbsub == True:
            sf3_vgsr_hist, sf3_vgsr_bins = np.histogram(streamfinder_table['oVGSR'], density=True, bins=50, range=(np.min(vgsr),np.max(vgsr)))
        else:
            sf3_vgsr_hist, sf3_vgsr_bins = np.histogram(streamfinder_table['VGSR'], density=True, bins=50, range=(np.min(vgsr),np.max(vgsr)))
        bar_width = (np.max(vgsr) - np.min(vgsr)) / 50
        ax.bar(sf3_vgsr_bins[1:], sf3_vgsr_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
    xp, p0, p1 = model_output[0:3]
    ax.plot(xp, p0 + p1, 'k-', label='total', lw=3)
    ax.plot(xp, p1, ':', color=colors[1], label='bg', lw=3)
    ax.plot(xp, p0, ':', color=colors[0], label='stream', lw=3)
    ax.set_xlabel('vgsr')
    ax.set_xlim(np.min(vgsr)- 50,np.max(vgsr)+50)
    ax.legend(fontsize='small')

    if feh_fit:
        # feh
        ax = axes[0][1]
        ax.hist(feh, density=True, color='lightgrey', bins=50)
        if streamfinder_table is not None:
            sf3_feh_hist, sf3_feh_bins = np.histogram(streamfinder_table['FEH'], density=True, bins=50,range=(np.min(feh),np.max(feh)))
            bar_width = (np.max(feh) - np.min(feh)) / 50
            ax.bar(sf3_feh_bins[1:], sf3_feh_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
        xp, p0, p1 = model_output[3:6]
        
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[1], lw=3)
        ax.plot(xp, p0, ':', color=colors[0], lw=3)
        ax.set_xlim(np.min(feh)-0.5,np.max(feh)+0.5)
        ax.set_xlabel('feh')
    
    # pmra
    ax = axes[1][0]
    ax.hist(pmra, density=True, color='lightgrey', bins=50)
    if streamfinder_table is not None:
        if orbsub == True:
            sf3_pmra_hist, sf3_pmra_bins = np.histogram(streamfinder_table['oPMRA'], density=True, bins=50, range=(np.min(pmra),np.max(pmra)))
        else:
            sf3_pmra_hist, sf3_pmra_bins = np.histogram(streamfinder_table['PMRA'], density=True, bins=50, range=(np.min(pmra),np.max(pmra)))
        bar_width = (np.max(pmra) - np.min(pmra)) / 50
        ax.bar(sf3_pmra_bins[1:], sf3_pmra_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
    xp, p0, p1 = model_output[6:9]
    ax.plot(xp, p0 + p1, 'k-', lw=3)
    ax.plot(xp, p1, ':', color=colors[1], lw=3)
    ax.plot(xp, p0, ':', color=colors[0], lw=3)
    ax.set_xlabel('pmra')
    ax.set_xlim(np.min(pmra)-15,np.max(pmra)+15)
    
    # pmdec
    ax = axes[1][1]
    ax.hist(pmdec, density=True, color='lightgrey', bins=50)
    if streamfinder_table is not None:
        if orbsub == True:
            sf3_pmdec_hist, sf3_pmdec_bins = np.histogram(streamfinder_table['oPMDEC'], density=True, bins=50, range=(np.min(pmdec),np.max(pmdec)))
        else:
            sf3_pmdec_hist, sf3_pmdec_bins = np.histogram(streamfinder_table['PMDEC'], density=True, bins=50, range=(np.min(pmdec),np.max(pmdec)))
        bar_width = (np.max(pmdec) - np.min(pmdec)) / 50
        ax.bar(sf3_pmdec_bins[1:], sf3_pmdec_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
    xp, p0, p1 = model_output[9:12]
    ax.plot(xp, p0 + p1, 'k-', lw=3)
    ax.plot(xp, p1, ':', color=colors[1], lw=3)
    ax.plot(xp, p0, ':', color=colors[0], lw=3)
    ax.set_xlabel('pmdec')
    ax.set_xlim(np.min(pmdec)-15,np.max(pmdec)+15)
    return fig


def process_chain(chain, avg_error=True, labels=param_labels):
    ''' Returns the means and errors of teh parameters
    
    Parameters:
    chain - array. The chain
    avg_error - bool. Will average the + and - errors if True
    
    Return:
    2 or 3 OrderedDict as a tuple. means, errors or means, errors+, errors-
    '''
    chain = np.asarray(chain)

    if chain.ndim != 2:
        raise ValueError(f"chain must be 2D (nsamples, nparams); got shape {chain.shape}")

    if labels is None:
        raise ValueError("labels must be provided when processing a chain")

    n_dim = chain.shape[1]
    if len(labels) != n_dim:
        raise ValueError(
            f"Dimension mismatch: chain has {n_dim} parameters but labels supplies {len(labels)} entries. "
            "Ensure MCMeta configuration matches the saved chain before calling process_chain."
        )

    pctl = np.percentile(chain, [16, 50, 84], axis=0)
    meds = pctl[1]
    ep = pctl[2]-pctl[1]
    em = pctl[0]-pctl[1]

    if avg_error: # just for simplicity, assuming no asymmetry
        err = (ep-em)/2
        return OrderedDict(zip(labels, meds)), OrderedDict(zip(labels, err))
    else:
        return OrderedDict(zip(labels, meds)), OrderedDict(zip(labels, ep)), OrderedDict(zip(labels, em))

def memprob(theta, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1, pmcorr, return_lik=False):
    """ Calculates membership probability based on inputs
    
    Parameters:
    theta - array. model parameters
    vgsr, pmra, pmdec, phi1 - floats. Respective values
    
    Return:
    float. The membership probability
    """
    # params
    lpstream, \
    v1, v2, v3, lsigv, \
    feh1, lsigfeh, \
    pmra1, pmra2, pmra3, lsigpmra, \
    pmdec1, pmdec2, pmdec3, lsigpmdec, \
    bv, lsigbv, bfeh, lsigbfeh, bpmra, lsigbpmra, bpmdec, lsigbpmdec = theta
    
    pstream = 10**(lpstream)
    
    ## Compute log likelihood in v_gsr
    lstream_v = stats.norm.logpdf(vgsr, loc=quad_f(phi1,v1,v2,v3), scale=np.sqrt(vgsr_err**2+(10**lsigv)**2))
    lbg_v = stats.norm.logpdf(vgsr, loc=bv, scale=np.sqrt(vgsr_err**2+(10**lsigbv)**2))
    
    ## Compute log likelihood in feh
    lstream_feh = stats.norm.logpdf(feh, loc=feh1, scale=np.sqrt(feh_err**2+(10**lsigfeh)**2))
    lbg_feh = stats.norm.logpdf(feh, loc=bfeh, scale=np.sqrt(feh_err**2+(10**lsigbfeh)**2))
    
    ## Compute log likelihood in pm
    lstream_pm = logpdf_2dnorm(pmra,pmdec,
                               quad_f(phi1,pmra1,pmra2,pmra3),quad_f(phi1,pmdec1,pmdec2,pmdec3),
                               pmra_err,pmdec_err,
                               10**lsigpmra,10**lsigpmdec,
                               pmcorr)
    lbg_pm = logpdf_2dnorm(pmra,pmdec,
                           bpmra,bpmdec,
                           pmra_err,pmdec_err,
                           10**lsigbpmra,10**lsigbpmdec,
                           pmcorr)

    ## Combine the components
    lstream = np.log(pstream) + lstream_v + lstream_feh + lstream_pm
    lbg = np.log(1-pstream) + lbg_v + lbg_feh + lbg_pm
    
    stream = np.exp(lstream)
    bg = np.exp(lbg)
    
    p = stream/(stream+bg)
    if return_lik == False:
        return p
    else:
        return p, stream, bg


def generate_data(n, dataframe, flatchain, meds):
    n_stream = np.round(n*(10**meds[list(meds.keys())[0]]))
    n_bg = np.round(n*(1-(10**meds[list(meds.keys())[0]])))
    
    print(n_stream,n_bg,n_stream+n_bg,n)
    
    np.random.seed(24)
    
    vgsr_stream  = []
    feh_stream  = []
    pmra_stream  = []
    pmdec_stream = []

    vgsr_bg = []
    feh_bg = []
    pmra_bg = []
    pmdec_bg = []
    
    for i in range(int(n_stream)):
        random_params = flatchain[np.random.randint(0,flatchain.shape[0])]
        random_meds = get_paramdict(random_params)
        random_phi1 = np.random.choice(dataframe['phi1'])
        random_vgsrerr = np.random.choice(dataframe['VRAD_ERR'])
        random_feherr = np.random.choice(dataframe['FEH_ERR'])
        random_pmraerr = np.random.choice(dataframe['PMRA_ERROR'])
        random_pmdecerr = np.random.choice(dataframe['PMDEC_ERROR'])
        vgsr_stream.append(np.random.normal(quad_f(random_phi1,random_meds['v1'],random_meds['v2'],random_meds['v3']),
                                           np.sqrt((10**random_meds['lsigv'])**2+random_vgsrerr**2)))
        feh_stream.append(np.random.normal(random_meds['feh1'],
                                           np.sqrt((10**random_meds['lsigfeh'])**2+random_feherr**2)))
        pmra_stream.append(np.random.normal(quad_f(random_phi1,random_meds['pmra1'],random_meds['pmra2'],random_meds['pmra3']),
                                            np.sqrt((10**random_meds['lsigpmra'])**2+random_pmraerr**2)))
        pmdec_stream.append(np.random.normal(quad_f(random_phi1,random_meds['pmdec1'],random_meds['pmdec2'],random_meds['pmdec3']),
                                             np.sqrt((10**random_meds['lsigpmdec'])**2+random_pmdecerr**2)))
        

    for i in range(int(n_bg)):
        random_params = flatchain[np.random.randint(0,flatchain.shape[0])]
        random_meds = get_paramdict(random_params)
        random_phi1 = np.random.choice(dataframe['phi1'])
        random_vgsrerr = np.random.choice(dataframe['VRAD_ERR'])
        random_feherr = np.random.choice(dataframe['FEH_ERR'])
        random_pmraerr = np.random.choice(dataframe['PMRA_ERROR'])
        random_pmdecerr = np.random.choice(dataframe['PMDEC_ERROR'])
        vgsr_bg.append(np.random.normal(random_meds['bv'],
                                       np.sqrt((10**random_meds['lsigbv'])**2+random_vgsrerr**2)))
        feh_bg.append(np.random.normal(random_meds['bfeh'],
                                       np.sqrt((10**random_meds['lsigbfeh'])**2+random_feherr**2)))
        pmra_bg.append(np.random.normal(random_meds['bpmra'],
                                        np.sqrt((10**random_meds['lsigbpmra'])**2+random_pmraerr**2)))
        pmdec_bg.append(np.random.normal(random_meds['bpmdec'],
                                         np.sqrt((10**random_meds['lsigbpmdec'])**2+random_pmdecerr**2)))

        
    stream_samples = np.vstack([vgsr_stream,feh_stream,pmra_stream,pmdec_stream]).T
    bg_samples = np.vstack([vgsr_bg,feh_bg,pmra_bg,pmdec_bg]).T
    
    vgsr_tot = vgsr_stream + vgsr_bg
    feh_tot = feh_stream + feh_bg
    pmra_tot = pmra_stream + pmra_bg
    pmdec_tot = pmdec_stream + pmdec_bg
    tot_samples = np.vstack([vgsr_tot,feh_tot,pmra_tot,pmdec_tot]).T
    
    return stream_samples, bg_samples, tot_samples


def distance_modulus_to_kpc(distance_modulus):
    """
    Convert distance modulus to distance in kiloparsecs (kpc).

    Parameters:
    distance_modulus (float): The distance modulus value.

    Returns:
    float: The distance in kiloparsecs.
    """
    distance_pc = 10 ** ((distance_modulus + 5) / 5)
    distance_kpc = distance_pc * 10**-3
    return distance_kpc


def are_points_within_isochrone(iso_r, iso_g_r, points_r, points_g_r, threshold=1.0):
    """
    Determine if each point in the arrays is within ±threshold of the isochrone curve using numpy broadcasting.

    Parameters:
    iso_r (array-like): Array of absolute R magnitudes representing the isochrone.
    iso_g_r (array-like): Array of g-r color indices representing the isochrone.
    points_r (array-like): Array of absolute R magnitudes of the points to check.
    points_g_r (array-like): Array of g-r color indices of the points to check.
    threshold (float): Distance threshold to determine if the points are within the isochrone curve. Default is 1.0.

    Returns:
    array: Array of booleans indicating if each point is within ±threshold of the isochrone.
    """
    # Ensure points_r and points_g_r are numpy arrays
    points_r = np.array(points_r)
    points_g_r = np.array(points_g_r)
    
    # Expand dimensions of points arrays to broadcast with isochrone arrays
    points_r_expanded = points_r[:, np.newaxis]
    points_g_r_expanded = points_g_r[:, np.newaxis]
    
    # Calculate the squared distances from each point to each point on the isochrone
    distances_squared = (iso_r - points_r_expanded)**2 + (iso_g_r - points_g_r_expanded)**2
    
    # Check if any distance is within the threshold
    within_threshold = np.any(distances_squared <= threshold**2, axis=1)
    
    return within_threshold






################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################

spline_param_labels = [
    "pstream_spline_points",
    "vgsr_spline_points", "lsigv_spline_points",
    "feh1", "lsigfeh",
    "pmra_spline_points", "lsigpmra",
    "pmdec_spline_points", "lsigpmdec",
    "bv", "lsigbv", "bfeh", "lsigbfeh", "bpmra", "lsigbpmra", "bpmdec", "lsigbpmdec"
]


def apply_spline(phi1, x_points, val_arr, k = 2):
    if len([val_arr]) == 1:
        val_arr = np.full_like(x_points, val_arr, dtype=float)
    spline_fnc = sp.interpolate.InterpolatedUnivariateSpline(x_points, np.array(val_arr), k=k)
    # spline_fnc = sp.interpolate.UnivariateSpline(x_points, np.array(val_arr), k=k)
    
    return spline_fnc(phi1)

def evaluate_probability_spline(eval_points, x_points, probability_knots, k=None, bounds=(1e-6, 1 - 1e-6)):
    """
    Evaluate a spline defined by probability knots while enforcing values remain in (0, 1).

    Parameters
    ----------
    eval_points : array-like or scalar
        Points (typically phi1) where the spline should be evaluated.
    x_points : array-like or None
        Knot locations associated with the probability knots. If None or length mismatch,
        a uniform grid spanning the knot range is constructed.
    probability_knots : array-like or scalar
        Probability values defined at the knot locations. Values are clipped to `bounds`
        before transforming to logit space.
    k : int or None, optional
        Spline order. When None, defaults to linear for <=2 knots and cubic otherwise,
        capped at degree len(knots) - 1.
    bounds : tuple, optional
        Lower and upper bounds (exclusive) used to clip probabilities to avoid singular logits.

    Returns
    -------
    numpy.ndarray or float
        Evaluated spline values at `eval_points`, clipped to `bounds`.
    """
    eval_points = np.asarray(eval_points, dtype=float)
    was_scalar = eval_points.ndim == 0
    if was_scalar:
        eval_points = eval_points.reshape(1)

    probs = np.asarray(probability_knots, dtype=float)
    if probs.size == 0:
        raise ValueError('Empty probability knot array encountered in evaluate_probability_spline')

    bounds = tuple(bounds)
    if len(bounds) != 2 or not (0.0 < bounds[0] < bounds[1] < 1.0):
        raise ValueError(f"bounds must satisfy 0 < low < high < 1; got {bounds}")
    low, high = bounds

    if probs.size == 1:
        clipped = float(np.clip(probs.reshape(-1)[0], low, high))
        result = np.full_like(eval_points, clipped, dtype=float)
        return float(result[0]) if was_scalar else result

    knots_arr = np.asarray(x_points, dtype=float) if x_points is not None else np.asarray([], dtype=float)
    if knots_arr.size != probs.size:
        if knots_arr.size > 0:
            knots_arr = np.linspace(knots_arr[0], knots_arr[-1], probs.size)
        elif eval_points.size > 0:
            knots_arr = np.linspace(eval_points.min(), eval_points.max(), probs.size)
        else:
            knots_arr = np.linspace(0.0, 1.0, probs.size)

    if k is None:
        eff_order = 1 if probs.size <= 2 else min(3, probs.size - 1)
    else:
        eff_order = int(k)
        eff_order = max(1, min(eff_order, probs.size - 1))

    logits = sp.special.logit(np.clip(probs, low, high))
    interpolated = apply_spline(eval_points, knots_arr, logits, k=eff_order)
    result = np.clip(sp.special.expit(interpolated), low, high)

    if was_scalar:
        return float(result[0])
    return result

def reshape_arr(theta, reshape_arr_shape):
    reshaped_theta = []
    idx = 0
    
    # Loop through each length in reshape_arr_shape to reshape appropriately
    for length in reshape_arr_shape:
        if idx == sum(reshape_arr_shape):
            break
        elif length == 1:
            # If it's a scalar, just append the value
            reshaped_theta.append(theta[idx])
        else:
            # If it's an array, reshape it based on the recorded length
            reshaped_theta.append(theta[idx:idx + length])
        idx += length
        # print(idx)
    
    return reshaped_theta

theta_labels = [ 'lpstream1', 'lpstream2', 'lpstream3', 'lpstream4', 'lpstream5',
                'lsigv',
                'feh1', 'lsigfeh',
                'lsigpmra',
                'lsigpmdec',
                'bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigpmdec']

lpstream_labels = ['lpstream1', 'lpstream2', 'lpstream3', 'lpstream4', 'lpstream5']
lsigv_label = ['lsigv']
feh_label = ['feh1', 'lsigfeh']
lsigpmra_label = ['lsigpmra']
lsigpmdec_label = ['lsigpmdec']

def spline_lnprob(theta, prior, spline_x_points, sf_spline_x_points, lsigmav_spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1, pmcorr, trunc_fit = False, assert_prior = False, feh_fit=True, k=2, reshape_arr_shape=None):
    """ Likelihood and Prior """
    
    reshaped_theta = reshape_arr(theta, reshape_arr_shape)
    
    # params
    #pstream, \
    lpstream_spline_points, \
    vgsr_spline_points, lsigv_spline_points, \
    feh1, lsigfeh, \
    pmra_spline_points, lsigpmra, \
    pmdec_spline_points, lsigpmdec, \
    bv, lsigbv, bfeh, lsigbfeh, bpmra, lsigbpmra, bpmdec, lsigbpmdec = reshaped_theta
    
    if feh_fit == False:
        feh1_min, feh1_max, lsigfeh_min, lsigfeh_max, bfeh_min, bfeh_max = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
        
    lpstream_indices = np.arange(1, len(sf_spline_x_points) + 1).astype(str)
    indices = np.arange(1, len(spline_x_points) + 1).astype(str)
    lsigv_indices = np.arange(1, len(lsigmav_spline_x_points) + 1).astype(str)
    # Generate labels
    lpstream_labels = np.char.add('lpstream', lpstream_indices)
    velocity_labels = np.char.add('v', indices)
    pmra_labels = np.char.add('pmra', indices)
    pmdec_labels = np.char.add('pmdec', indices)
    lsigv_labels = np.char.add('lsigv', lsigv_indices)
    
    # Insert labels at the correct positions
    theta_labels = (
        lpstream_labels.tolist() +                           # Start with lpstream labels
        velocity_labels.tolist() +                           # Insert velocity labels
        lsigv_labels.tolist() +     
        ['feh1', 'lsigfeh'] +
        pmra_labels.tolist() +                               # Insert pmra labels
        ['lsigpmra'] +       # Existing labels between 'lsigpmra' and 'lsigpmdec'
        pmdec_labels.tolist() +                              # Insert pmdec labels
        ['lsigpmdec'] +                       # Remaining labels after 'lsigpmdec'
        ['bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigbpmdec']
    )

    for i in range(len(theta)):
        if (theta[i] < prior[i][0]) or (theta[i] > prior[i][1]):
            
            if assert_prior:
                print(theta[i])
                print(theta_labels[i])

            return -1e10  # outside of prior, return a tiny number   
    if isinstance(lpstream_spline_points, np.ndarray): 
        if len(lpstream_spline_points) > 3:
            lpstream = np.log(10**(apply_spline(phi1,sf_spline_x_points,lpstream_spline_points, k=3)))
        elif len(lpstream_spline_points) <= 3:
            lpstream = np.log(10**(apply_spline(phi1,sf_spline_x_points,lpstream_spline_points, k=len(lpstream_spline_points)-1)))
    elif isinstance(lpstream_spline_points, np.float64):
        lpstream = np.log(10**lpstream_spline_points)

    if isinstance(lsigv_spline_points, np.ndarray):
        if len(lsigv_spline_points) > 3:
            lsigv = np.log(10**apply_spline(phi1, lsigmav_spline_x_points, lsigv_spline_points, k=3))
        elif len(lsigv_spline_points) <= 3:
            lsigv = np.log(10**apply_spline(phi1, lsigmav_spline_x_points, lsigv_spline_points, k=len(lsigv_spline_points)-1))
    elif isinstance(lsigv_spline_points, np.float64):
        lsigv = np.log(10**lsigv_spline_points)
    
    if np.any(1-(np.e**lpstream) <= 0):
        print('bad lpstream')
        return -1e10 # bad lpstream spline extrapolation. May have a dip that goes below 0

    if feh_fit:
        scale_stream_feh = np.sqrt(feh_err**2 + (10**lsigfeh)**2)
        scale_bg_feh = np.sqrt(feh_err**2 + (10**lsigbfeh)**2)
        
    lsigv_vals = apply_spline(phi1, lsigmav_spline_x_points, lsigv_spline_points, k=k)
    scale_stream_vgsr = np.sqrt(vgsr_err**2 + (10**lsigv_vals)**2)
    scale_bg_vgsr = np.sqrt(vgsr_err**2 + (10**lsigbv)**2)
    
    ## Compute log likelihood in feh
    if trunc_fit == False:
        ## Compute log likelihood in v_gsr
        lstream_v = stats.norm.logpdf(vgsr, loc=apply_spline(phi1, spline_x_points, vgsr_spline_points, k=k), scale=scale_stream_vgsr)
        lbg_v = stats.norm.logpdf(vgsr, loc=bv, scale=scale_bg_vgsr)
        
        if feh_fit:
            lstream_feh = stats.norm.logpdf(feh, loc=feh1, scale=scale_stream_feh)
            lbg_feh = stats.norm.logpdf(feh, loc=bfeh, scale=scale_bg_feh)
        
    elif trunc_fit:
        # Compute standardized bounds for truncnorm
        min_trunc_vgsr, max_trunc_vgsr = np.min(vgsr), np.max(vgsr)
        lvgsr_cdf_dif = np.log(stats.norm.cdf(max_trunc_vgsr, loc=bv, scale=scale_bg_vgsr) - stats.norm.cdf(min_trunc_vgsr, loc=bv, scale=scale_bg_vgsr))
        
        lstream_v = stats.norm.logpdf(vgsr, loc=apply_spline(phi1, spline_x_points, vgsr_spline_points, k=k), scale=scale_stream_vgsr)
        lbg_v = stats.norm.logpdf(vgsr, loc=bv, scale=scale_bg_vgsr) - lvgsr_cdf_dif
                
        if feh_fit:
            min_trunc_feh, max_trunc_feh = np.min(feh), np.max(feh)
            lfeh_cdf_dif = np.log(stats.norm.cdf(max_trunc_feh, loc=bfeh, scale=scale_bg_feh) - stats.norm.cdf(min_trunc_feh, loc=bfeh, scale=scale_bg_feh))
            lstream_feh = stats.norm.logpdf(feh, loc=feh1, scale=scale_stream_feh)
            lbg_feh = stats.norm.logpdf(feh, loc=bfeh, scale=scale_bg_feh) - lfeh_cdf_dif
    
    ## Compute log likelihood in pm
    lstream_pm = logpdf_2dnorm(pmra,pmdec,
                              apply_spline(phi1, spline_x_points, pmra_spline_points, k=k),apply_spline(phi1, spline_x_points, pmdec_spline_points, k=k),
                               pmra_err,pmdec_err,
                               10**lsigpmra,10**lsigpmdec,
                               pmcorr)
    lbg_pm = logpdf_2dnorm(pmra,pmdec,
                            bpmra,bpmdec,
                            pmra_err, pmdec_err,
                            10**lsigbpmra,10**lsigbpmdec,
                            pmcorr)
    
    if feh_fit:
        ## Combine the components
        lstream = lpstream + lstream_v + lstream_feh + lstream_pm
        lbg = np.log(1-(np.e**lpstream)) + lbg_v + lbg_feh + lbg_pm

    else:
        lstream = lpstream + lstream_v + lstream_pm
        lbg = np.log(1-(np.e**lpstream)) + lbg_v + lbg_pm
        
    ltot = np.logaddexp(lstream, lbg)

    return np.sum(ltot)

def spline_project_model(theta, spline_x_points, pstream_spline_x_points, lsig_vgsr_spline_x_points, vgsr_min, vgsr_max, feh_min, feh_max, pmra_min, pmra_max, pmdec_min, pmdec_max, trunc_fit=True, feh_fit=True, param_labels=spline_param_labels,k=2):
    """ Turn parameters into vgsr, pmra, pmdec distributions. An aproximation since no phi dependence"""
    params = get_paramdict(theta, param_labels)
    # print(len(params['pstream_spline_points']))
    pstream = evaluate_probability_spline(
        0.0,
        pstream_spline_x_points,
        params['pstream_spline_points'],
        k=None
    )
        
    if isinstance(params['lsigv_spline_points'], (int, np.int64, float, np.float64)):
        lsigv = 10**params['lsigv_spline_points']
    elif len(params['lsigv_spline_points']) == 1:
        lsigv = 10**params['lsigv_spline_points'][0]
    elif len(params['lsigv_spline_points']) > 3:
        lsigv = 10**apply_spline(0,lsig_vgsr_spline_x_points,params['lsigv_spline_points'], k=3)
    elif len(params['lsigv_spline_points']) <= 3:
        lsigv = 10**apply_spline(0,lsig_vgsr_spline_x_points,params['lsigv_spline_points'], k=len(params['lsigv_spline_points'])-1)
    
    vgsr_arr = np.linspace(vgsr_min-50, vgsr_max+50, 1000)
    
    # Handle vgsr_spline_points for location and lsigv_spline_points for scale
    if isinstance(params['vgsr_spline_points'], (float, np.float64)):
        vgsr_loc = params['vgsr_spline_points']
    elif len(params['vgsr_spline_points']) == 1:
        vgsr_loc = params['vgsr_spline_points'][0]
    else:
        effective_k = min(k, len(spline_x_points)-1)
        vgsr_loc = apply_spline(0, spline_x_points, params['vgsr_spline_points'], k=effective_k)
    
    if isinstance(params['lsigv_spline_points'], (int, np.int64, float, np.float64)):
        lsigv_scale = 10**params['lsigv_spline_points']
    elif len(params['lsigv_spline_points']) == 1:
        lsigv_scale = 10**params['lsigv_spline_points'][0]
    else:
        effective_k_sigv = min(3, len(lsig_vgsr_spline_x_points)-1)
        lsigv_scale = 10**apply_spline(0, lsig_vgsr_spline_x_points, params['lsigv_spline_points'], k=effective_k_sigv)
        
    pvgsr0 = pstream*stats.norm.pdf(vgsr_arr, loc=vgsr_loc, scale=lsigv_scale)
    
    feh_arr = np.linspace(feh_min-0.5, feh_max+0.5, 1000)
    if feh_fit:
        pfeh0 = pstream*stats.norm.pdf(feh_arr, loc=params['feh1'], scale=10**params['lsigfeh'])
    elif feh_fit == False:
        pfeh0 = 0
    
    pmra_arr = np.linspace(pmra_min-10, pmra_max+10, 1000)
    ppmra0 = pstream*stats.norm.pdf(pmra_arr, loc=apply_spline(0,spline_x_points,params['pmra_spline_points'],k=k), scale=10**params['lsigpmra'])
    
    pmdec_arr = np.linspace(pmdec_min-10, pmdec_max+10, 1000)
    ppmdec0 = pstream*stats.norm.pdf(pmdec_arr, loc=apply_spline(0,spline_x_points,params['pmdec_spline_points'],k=k), scale=10**params['lsigpmdec'])
    
    if trunc_fit:
        pvgsr1 = (1-pstream)*stats.truncnorm.pdf(vgsr_arr, a=(vgsr_min-params['bv'])/10**params['lsigbv'], b=(vgsr_max-params['bv'])/10**params['lsigbv'], loc=params['bv'], scale=10**params['lsigbv'])
        if feh_fit:
            pfeh1 = (1-pstream)*stats.truncnorm.pdf(feh_arr, a=(feh_min-params['bfeh'])/10**params['lsigbfeh'], b=(feh_max-params['bfeh'])/10**params['lsigbfeh'], loc=params['bfeh'], scale=10**params['lsigbfeh'])
        elif feh_fit == False:
            pfeh1 = 0
        
    else:
        pvgsr1 = (1-pstream)*stats.norm.pdf(vgsr_arr, loc=params['bv'], scale=10**params['lsigbv'])
        if feh_fit:
            pfeh1 = (1-pstream)*stats.norm.pdf(feh_arr, loc=params['bfeh'], scale=10**params['lsigbfeh'])
        elif feh_fit == False:
            pfeh1 = 0
    
    ppmra1 = (1-pstream)*stats.norm.pdf(pmra_arr, loc=params['bpmra'], scale=10**params['lsigbpmra'])
    ppmdec1 = (1-pstream)*stats.norm.pdf(pmdec_arr, loc=params['bpmdec'], scale=10**params['lsigbpmdec'])
    
    return vgsr_arr, pvgsr0, pvgsr1, feh_arr, pfeh0, pfeh1, pmra_arr, ppmra0, ppmra1, pmdec_arr, ppmdec0, ppmdec1

def spline_plot_1d_distrs(theta, spline_x_points, pstream_spline_x_points, lsig_vgsr_spline_points, vgsr, feh, pmra, pmdec, trunc_fit=True, feh_fit=True, streamfinder_table=None, param_labels=spline_param_labels, k=2, lsigpmra=None, lsigpmdec=None):
    '''
    Plots the distributions of vgsr, feh, pmra, pmdec
    '''
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    
    theta_copy = theta.copy()
    if lsigpmra != None:
        theta_copy.insert(6, lsigpmra)
    if lsigpmdec != None:
        theta_copy.insert(8, lsigpmra)

    if trunc_fit:
        model_output = spline_project_model(theta_copy, spline_x_points, pstream_spline_x_points,lsig_vgsr_spline_points,np.min(vgsr), np.max(vgsr), np.min(feh),np.max(feh), np.min(pmra),np.max(pmra),np.min(pmdec),np.max(pmdec),\
            trunc_fit=trunc_fit, feh_fit=feh_fit, param_labels=param_labels, k=k)
        
    else:
        model_output = spline_project_model(theta_copy, spline_x_points, pstream_spline_x_points,lsig_vgsr_spline_points, np.min(vgsr),np.max(vgsr),np.min(feh),np.max(feh),np.min(pmra),np.max(pmra),np.min(pmdec),np.max(pmdec),\
            feh_fit=feh_fit, param_labels=param_labels, k=k)
    
    scaled_down_sf3_density = 10
    
    
    fig, axes = plt.subplots(2,2,figsize=(12,12))
    
    #vgsr
    ax = axes[0][0]
    ax.hist(vgsr, density=True, color='lightgrey', bins=50)
    if streamfinder_table is not None:
        sf3_vgsr_hist, sf3_vgsr_bins = np.histogram(streamfinder_table['VGSR'], density=True, bins=50, range=(np.min(vgsr),np.max(vgsr)))
        bar_width = (np.max(vgsr) - np.min(vgsr)) / 50
        ax.bar(sf3_vgsr_bins[1:], sf3_vgsr_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
    xp, p0, p1 = model_output[0:3]
    ax.plot(xp, p0 + p1, 'k-', label='total', lw=3)
    ax.plot(xp, p1, ':', color=colors[1], label='bg', lw=3)
    ax.plot(xp, p0, ':', color=colors[0], label='stream', lw=3)
    ax.set_xlabel(r'$v_{GSR}$ km/s', fontsize=12)
    ax.set_xlim(np.min(vgsr)- 50,np.max(vgsr)+50)
    ax.legend(fontsize='large')
    ax.tick_params(axis='both', labelsize=14)
    plot_form(ax)
    if feh_fit:
        # feh
        ax = axes[0][1]
        ax.hist(feh, density=True, color='lightgrey', bins=50)
        if streamfinder_table is not None:
            sf3_feh_hist, sf3_feh_bins = np.histogram(streamfinder_table['FEH'], density=True, bins=50,range=(np.min(feh),np.max(feh)))
            bar_width = (np.max(feh) - np.min(feh)) / 50
            ax.bar(sf3_feh_bins[1:], sf3_feh_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
        xp, p0, p1 = model_output[3:6]
        
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[1], lw=3)
        ax.plot(xp, p0, ':', color=colors[0], lw=3)
        ax.set_xlim(np.min(feh)-0.5,np.max(feh)+0.5)
        ax.set_xlabel('[Fe/H]',fontsize=12)
        ax.tick_params(axis='both', labelsize=14)
        plot_form(ax)
    # pmra
    ax = axes[1][0]
    ax.hist(pmra, density=True, color='lightgrey', bins=50)
    if streamfinder_table is not None:
        sf3_pmra_hist, sf3_pmra_bins = np.histogram(streamfinder_table['PMRA'], density=True, bins=50, range=(np.min(pmra),np.max(pmra)))
        bar_width = (np.max(pmra) - np.min(pmra)) / 50
        ax.bar(sf3_pmra_bins[1:], sf3_pmra_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
    xp, p0, p1 = model_output[6:9]
    ax.plot(xp, p0 + p1, 'k-', lw=3)
    ax.plot(xp, p1, ':', color=colors[1], lw=3)
    ax.plot(xp, p0, ':', color=colors[0], lw=3)
    ax.set_xlabel(r'$\mu_{RA}$ mas/yr',fontsize=12)
    ax.set_xlim(np.min(pmra)-15,np.max(pmra)+15)
    ax.tick_params(axis='both', labelsize=14)
    plot_form(ax)
    # pmdec
    ax = axes[1][1]
    ax.hist(pmdec, density=True, color='lightgrey', bins=50)
    if streamfinder_table is not None:
        sf3_pmdec_hist, sf3_pmdec_bins = np.histogram(streamfinder_table['PMDEC'], density=True, bins=50, range=(np.min(pmdec),np.max(pmdec)))
        bar_width = (np.max(pmdec) - np.min(pmdec)) / 50
        ax.bar(sf3_pmdec_bins[1:], sf3_pmdec_hist/scaled_down_sf3_density, color='lightblue', alpha=0.8, width=bar_width)
    xp, p0, p1 = model_output[9:12]
    ax.plot(xp, p0 + p1, 'k-', lw=3)
    ax.plot(xp, p1, ':', color=colors[1], lw=3)
    ax.plot(xp, p0, ':', color=colors[0], lw=3)
    ax.set_xlabel(r'$\mu_{DEC}$ mas/yr',fontsize=12)
    ax.set_xlim(np.min(pmdec)-15,np.max(pmdec)+15)
    ax.tick_params(axis='both', labelsize=14)
    plot_form(ax)
    return fig

def tan_transform(x):
    """Symmetric mapping from (0,1) to (-inf, inf) using tangent."""
    x = np.asarray(x)
    if np.any((x <= 0) | (x >= 1)):
        raise ValueError("Input must be in the open interval (0, 1)")
    return np.tan(np.pi * (x - 0.5))

def atan_inverse(y):
    """Inverse: maps (-inf, inf) back to (0, 1) using arctangent."""
    return (np.arctan(y) / np.pi) + 0.5

blind_panels = ['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2']
blind_meds_ind = [1, 3, 5, 7]

def does_it_touch(desi_df, phi1_spline_points, nested_list_meds, spline_k, pad, blind_panels=blind_panels, blind_meds_ind=blind_meds_ind, pad_only=False, sigma_pad=2, upper_bound=False):
    masks_dict = {}
    for panel in blind_panels:
        masks_dict[panel] = {}
        for i, other_panel in enumerate(blind_panels):
            if other_panel != panel:
                error_col = other_panel + '_ERR' if other_panel + '_ERR' in desi_df.columns else other_panel + '_ERROR'
                
                # Special case for FEH - use meds['feh1'] instead of apply_spline
                if other_panel == 'FEH':
                    if upper_bound:
                        masks_dict[panel][other_panel] = (desi_df[other_panel] < pad[i])
                    else:
                        reference_value = nested_list_meds[3]
                        masks_dict[panel][other_panel] = (np.abs(desi_df[other_panel] - reference_value) < pad[i])
                else:
                    if other_panel == 'phi2':
                        reference_value = 0
                    else:
                        reference_value = apply_spline(desi_df['phi1'], phi1_spline_points, nested_list_meds[blind_meds_ind[i]], spline_k)
                    if pad_only:
                        if other_panel == 'phi2':
                            masks_dict[panel][other_panel] = (np.abs(desi_df[other_panel] - reference_value) < pad[i])
                        masks_dict[panel][other_panel] = (np.abs(desi_df[other_panel] - reference_value) < pad[i])
                    else:
                        if other_panel == 'phi2':
                            masks_dict[panel][other_panel] = (np.abs(desi_df[other_panel] - reference_value) < pad[i])
                        else:
                            masks_dict[panel][other_panel] = (np.abs(desi_df[other_panel] - reference_value) < np.sqrt((desi_df[error_col]*sigma_pad)**2 + pad[i]**2))
    return masks_dict

def box_cuts(desi_df, phi1_spline_points, nested_list_meds, spline_k, pad, blind_panels=blind_panels, blind_meds_ind=blind_meds_ind, sigma_pad=2, upper_bound=False):
    masks_dict = {}
    for i, panel in enumerate(blind_panels):
        if panel == 'FEH':
            if upper_bound:
                mask = (desi_df[panel] < pad[i])
            else:
                reference_value = nested_list_meds[3]
                mask = (np.abs(desi_df[panel] - reference_value) < pad[i])
        else:
            if panel == 'phi2':
                reference_value = 0
            else:
                print(nested_list_meds[blind_meds_ind[i]])
                print(i)
                reference_value = reference_value = apply_spline(desi_df['phi1'], phi1_spline_points, nested_list_meds[blind_meds_ind[i]], spline_k)
            mask = (np.abs(desi_df[panel] - reference_value) < pad[i])
        masks_dict[panel] = mask
    return masks_dict



def print_meds(stream_dir):
    mcmc_dict = np.load(stream_dir + 'mcmc_dict.npy', allow_pickle=True).item()

    flatchain = mcmc_dict['flatchain']
    meds, errs = process_chain(flatchain, labels = mcmc_dict['extended_param_labels'])
    exp_flatchain = np.copy(flatchain)
    for i, label in enumerate(meds.keys()):
        if label[0] == 'l':
            exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
    exp_meds, exp_errs = process_chain(exp_flatchain, mcmc_dict['extended_param_labels'])

    _, ep, em = process_chain(mcmc_dict['flatchain'], avg_error=False, labels = mcmc_dict['extended_param_labels'])

    exp_flatchain = np.copy(flatchain)
    for i, label in enumerate(meds.keys()):
        if label[0] == 'l':
            exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
    exp_meds, exp_ep, exp_em = process_chain(exp_flatchain, avg_error=False, labels = mcmc_dict['extended_param_labels'])

    i = 0
    # print("{:<10} {:>10} {:>10} {:>10} {:>10}".format('param','med','err','exp(med)','exp(err)'))
    print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format('param','med', 'em','ep','exp(med)', 'exp(em)','exp(ep)'))
    print('--------------------------------------------------------------------------------------')
    for label,v in meds.items():
        # if label[:8] == 'lpstream':
        #     print("{:<10} {:>10.3f} {:>10.3f} {:>10.5f} {:>10.5f}".format(label,v,errs[label], np.e**v, np.log(10)*(np.e**v)*errs[label]))
        if label[0] == 'l':
            # print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(label,v,errs[label], exp_meds[label], exp_errs[label]))
            print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(label,v,em[label],ep[label], exp_meds[label], exp_em[label], exp_ep[label]))
        else:
            print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f}".format(label, v, em[label], ep[label]))
        i += 1


# theta_labels = (
#     lpstream_labels.tolist() +                           # Start with lpstream labels
#     velocity_labels.tolist() +                           # Insert velocity labels
#     lsigv_labels.tolist() +     
#     ['feh1', 'lsigfeh'] +
#     pmra_labels.tolist() +                               # Insert pmra labels
#     ['lsigpmra'] +       # Existing labels between 'lsigpmra' and 'lsigpmdec'
#     pmdec_labels.tolist() +                              # Insert pmdec labels
#     ['lsigpmdec'] +                       # Remaining labels after 'lsigpmdec'
#     ['bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigbpmdec']
# )

def call_likelihood(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                     trunc_fit=False, assert_prior=False, feh_fit=True, k=2, reshape_arr_shape=None,
                     vgsr_trunc=[-np.inf, np.inf], feh_trunc=[-np.inf, np.inf], pmra_trunc=[-np.inf, np.inf],
                     pmdec_trunc=[-np.inf, np.inf], **kwargs):
    lsigpm_ = kwargs.get('lsigpm_set', None)
    reshaped_theta = reshape_arr(theta, reshape_arr_shape)

    pstream, vgsr_spline_points, lsigv, feh1, lsigfeh, \
    pmra_spline_points, *rest = reshaped_theta

    if lsigpm_ is None:
        lsigpmra, pmdec_spline_points, lsigpmdec, bv, lsigbv, bfeh, lsigbfeh, bpmra, lsigbpmra, bpmdec, lsigbpmdec = rest
    else:
        pmdec_spline_points, bv, lsigbv, bfeh, lsigbfeh, bpmra, lsigbpmra, bpmdec, lsigbpmdec = rest
        lsigpmra = lsigpmdec = lsigpm_

    phi1 = np.asarray(phi1, dtype=float)
    pstream_bounds = kwargs.get('pstream_bounds', (1e-6, 1 - 1e-6))

    def _evaluate_knotted(values, knots_candidate, order_candidate, default_knots):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError('Empty parameter array encountered in spline evaluation')
        if arr.size == 1:
            val = float(arr.reshape(-1)[0])
            return np.full_like(phi1, val, dtype=float)

        knots_arr = knots_candidate if knots_candidate is not None else default_knots
        knots_arr = np.asarray(knots_arr, dtype=float)
        if knots_arr.size != arr.size:
            if knots_arr.size > 0:
                knots_arr = np.linspace(knots_arr[0], knots_arr[-1], arr.size)
            elif phi1.size > 0:
                knots_arr = np.linspace(phi1.min(), phi1.max(), arr.size)
            else:
                knots_arr = np.linspace(0.0, 1.0, arr.size)

        if order_candidate is None:
            eff_order = 1 if arr.size <= 2 else min(3, arr.size - 1)
        else:
            eff_order = int(order_candidate)
            eff_order = max(1, min(eff_order, arr.size - 1))

        return apply_spline(phi1, knots_arr, arr, k=eff_order)

    pstream_vals = evaluate_probability_spline(
        phi1,
        kwargs.get('pstream_phi1_spline_points'),
        pstream,
        k=kwargs.get('pstream_spline_k'),
        bounds=pstream_bounds
    )

    if np.any(~np.isfinite(pstream_vals)):
        return -1e10

    lsigv_vals = _evaluate_knotted(
        lsigv,
        kwargs.get('lsigv_phi1_spline_points'),
        kwargs.get('lsigv_spline_k'),
        spline_x_points
    )

    if np.any(~np.isfinite(lsigv_vals)):
        return -1e10

    scale_stream_vgsr = np.sqrt(vgsr_err**2 + (10**lsigv_vals)**2)
    scale_bg_vgsr = np.sqrt(vgsr_err**2 + (10**lsigbv)**2)

    scale_stream_pmra = np.sqrt(pmra_err**2 + (10**lsigpmra)**2)
    scale_bg_pmra = np.sqrt(pmra_err**2 + (10**lsigbpmra)**2)

    scale_stream_pmdec = np.sqrt(pmdec_err**2 + (10**lsigpmdec)**2)
    scale_bg_pmdec = np.sqrt(pmdec_err**2 + (10**lsigbpmdec)**2)

    scale_stream_feh = np.sqrt(feh_err**2 + (10**lsigfeh)**2)
    scale_bg_feh = np.sqrt(feh_err**2 + (10**lsigbfeh)**2)

    lvgsr_cdf_dif = np.log(
        stats.norm.cdf(vgsr_trunc[1], loc=bv, scale=scale_bg_vgsr)
        - stats.norm.cdf(vgsr_trunc[0], loc=bv, scale=scale_bg_vgsr)
    )
    lstream_v = stats.norm.logpdf(
        vgsr, loc=apply_spline(phi1, spline_x_points, vgsr_spline_points, k=k), scale=scale_stream_vgsr
    )
    lbg_v = stats.norm.logpdf(vgsr, loc=bv, scale=scale_bg_vgsr) - lvgsr_cdf_dif

    lfeh_cdf_dif = np.log(
        stats.norm.cdf(feh_trunc[1], loc=bfeh, scale=scale_bg_feh)
        - stats.norm.cdf(feh_trunc[0], loc=bfeh, scale=scale_bg_feh)
    )
    lstream_feh = stats.norm.logpdf(feh, loc=feh1, scale=scale_stream_feh)
    lbg_feh = stats.norm.logpdf(feh, loc=bfeh, scale=scale_bg_feh) - lfeh_cdf_dif

    lpmra_cdf_dif = np.log(
        stats.norm.cdf(pmra_trunc[1], loc=bpmra, scale=scale_bg_pmra)
        - stats.norm.cdf(pmra_trunc[0], loc=bpmra, scale=scale_bg_pmra)
    )
    lstream_pmra = stats.norm.logpdf(
        pmra, loc=apply_spline(phi1, spline_x_points, pmra_spline_points, k=k), scale=scale_stream_pmra
    )
    lbg_pmra = stats.norm.logpdf(pmra, loc=bpmra, scale=scale_bg_pmra) - lpmra_cdf_dif

    lpmdec_cdf_dif = np.log(
        stats.norm.cdf(pmdec_trunc[1], loc=bpmdec, scale=scale_bg_pmdec)
        - stats.norm.cdf(pmdec_trunc[0], loc=bpmdec, scale=scale_bg_pmdec)
    )
    lstream_pmdec = stats.norm.logpdf(
        pmdec, loc=apply_spline(phi1, spline_x_points, pmdec_spline_points, k=k), scale=scale_stream_pmdec
    )
    lbg_pmdec = stats.norm.logpdf(pmdec, loc=bpmdec, scale=scale_bg_pmdec) - lpmdec_cdf_dif

    lpstream = np.log(pstream_vals)
    lbg_weight = np.log1p(-pstream_vals)

    if np.any(~np.isfinite(lpstream)) or np.any(~np.isfinite(lbg_weight)):
        return -1e10

    lstream = lpstream + lstream_v + lstream_feh + lstream_pmra + lstream_pmdec
    lbg = lbg_weight + lbg_v + lbg_feh + lbg_pmra + lbg_pmdec

    return lstream, lbg
def lnlikelihood(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                     trunc_fit = False, assert_prior = False, feh_fit=True, k=2, reshape_arr_shape=None, vgsr_trunc=[-np.inf, np.inf], feh_trunc=[-np.inf, np.inf], pmra_trunc=[-np.inf, np.inf], pmdec_trunc=[-np.inf, np.inf], lsigpm_set=None, **kwargs):
    lstream, lbg = call_likelihood(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                     trunc_fit, assert_prior, feh_fit, k, reshape_arr_shape, vgsr_trunc, feh_trunc, pmra_trunc, pmdec_trunc, lsigpm_set=lsigpm_set, **kwargs)
    ltot = np.logaddexp(lstream, lbg)
    return np.sum(ltot)

def lnprior(theta, prior, spline_x_points, assert_prior=False, reshape_arr_shape=None):
    reshaped_theta = reshape_arr(theta, reshape_arr_shape)

    # flatten back out to a single list of numbers
    # Safe flattener that handles scalars + variable-length lists/arrays
    def _flatten(obj):
        if isinstance(obj, (list, tuple, np.ndarray)):
            for x in obj:
                yield from _flatten(x)
        else:
            yield obj
    
    flat_params = list(_flatten(reshaped_theta))

    # check same length
    if len(flat_params) != len(prior):
        raise ValueError(f"Length mismatch: {len(flat_params)} parameters vs {len(prior)} prior bounds")

    # bounds check
    for val, (low, high) in zip(flat_params, prior):
        if not (low <= val <= high):
            if assert_prior:
                raise AssertionError(f"Parameter {val} outside {(low, high)}")
            return -np.inf

    return 0.

def lnprob(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                     trunc_fit = False, assert_prior = False, feh_fit=True, k=2, reshape_arr_shape=None, vgsr_trunc=[-np.inf, np.inf], feh_trunc=[-np.inf, np.inf], pmra_trunc=[-np.inf, np.inf], pmdec_trunc=[-np.inf, np.inf], lsigpm_set=None, **kwargs):
    """ Likelihood and Prior """
    
    lp = lnprior(theta, prior, spline_x_points, assert_prior=assert_prior, reshape_arr_shape=reshape_arr_shape)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = lnlikelihood(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                     trunc_fit=trunc_fit, assert_prior=assert_prior, feh_fit=feh_fit, k=k, reshape_arr_shape=reshape_arr_shape,
                     vgsr_trunc=vgsr_trunc, feh_trunc=feh_trunc, pmra_trunc=pmra_trunc, pmdec_trunc=pmdec_trunc, lsigpm_set=lsigpm_set, **kwargs)
    
    return lp + ll

def memprob(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                      trunc_fit = False, assert_prior = False, feh_fit=True, k=2, reshape_arr_shape=None, vgsr_trunc=[-np.inf, np.inf], feh_trunc=[-np.inf, np.inf], pmra_trunc=[-np.inf, np.inf], pmdec_trunc=[-np.inf, np.inf], lsigpm_set=None, **kwargs):  
    lstream, lbg = call_likelihood(theta, prior, spline_x_points, vgsr, vgsr_err, feh, feh_err, pmra, pmra_err, pmdec, pmdec_err, phi1,
                     trunc_fit, assert_prior, feh_fit, k, reshape_arr_shape, vgsr_trunc, feh_trunc, pmra_trunc, pmdec_trunc, lsigpm_set=lsigpm_set, **kwargs)
    stream = np.exp(lstream)
    bg = np.exp(lbg)
    p = stream/(stream+bg)
    return p
