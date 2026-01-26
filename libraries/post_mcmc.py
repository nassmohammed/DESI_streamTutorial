from arrow import get
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import table
import stream_functions as stream_funcs
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import galstreams
import astropy.units as u
import astropy.coordinates as coord
from gala.coordinates import GreatCircleICRSFrame
import importlib
from astropy.table import Table
import scipy as sp
from collections import OrderedDict
import copy
from types import SimpleNamespace
import gallery_functions as gallery_funcs
importlib.reload(gallery_funcs)
from stream_functions import apply_spline
import scripts.vdisp as vdisp
import os
import corner
import polars as pl
from scripts.streamTutorial import StreamPlotter

SPLINE_ZORDER = -5
ERROR_BAR_ZORDER = -2
BACKGROUND_SCATTER_ZORDER = 1
SELECTED_SCATTER_ZORDER = 4


def _label_style(label):
    """Return plotting style (marker, color) for a given data label."""
    styles = {
        'MS+RG': ('o', '0.4'),
        'BHB': ('^', 'blue'),
        'RRL': ('v', "#ff0022"),
    }
    return styles.get(label, ('o', '0.3'))


def _ensure_numeric(values):
    """Coerce a pandas Series / array to a numeric numpy array."""
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    arr = np.asarray(values)
    if arr.dtype.kind in 'biufc':
        return arr.astype(float)
    return pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)


def _pick_numeric_column(data, candidates):
    """Return the first matching numeric column from candidates or None."""
    if data is None:
        return None
    for name in candidates:
        if isinstance(data, (pd.DataFrame, pd.Series)) and name in data:
            return _ensure_numeric(data[name])
        if hasattr(data, 'dtype') and data.dtype.names and name in data.dtype.names:
            return _ensure_numeric(data[name])
    return None


COLUMN_ALIASES = {
    'VGSR': ['VGSR', 'VGSR_desi', 'VGSR_DESI', 'vgsr', 'V_GSR', 'VGSR_MS', 'VHELIO_AVG'],
    'VGSR_ERR': ['VGSR_ERR', 'VRAD_ERR', 'VGSR_ERR_desi', 'VRAD_ERR_desi', 'VRAD_ERROR', 'vgsr_err', 'v0_std'],
    'FEH': ['FEH', 'FEH_desi', 'feh', 'METALLICITY', 'FEH_MS'],
    'PMRA': ['PMRA', 'PMRA_desi', 'pmra'],
    'PMRA_ERROR': ['PMRA_ERROR', 'PMRA_ERROR_desi', 'pmra_error', 'PMRA_ERR'],
    'PMDEC': ['PMDEC', 'PMDEC_desi', 'pmdec'],
    'PMDEC_ERROR': ['PMDEC_ERROR', 'PMDEC_ERROR_desi', 'pmdec_error', 'PMDEC_ERR'],
    'PARALLAX': ['PARALLAX', 'parallax', 'PARALLAX_desi'],
    'PARALLAX_ERROR': ['PARALLAX_ERROR', 'parallax_error', 'PARALLAX_ERROR_desi'],
    'FEH_ERROR': ['FEH_ERR', 'FEH_ERROR', 'FEH_ERR_desi'],
    'PHI2_ERROR': ['phi2_err', 'PHI2_ERR', 'phi2_error'],
    'DISTANCE': ['dist_kpc', 'distance_kpc', 'dist', 'distance', 'DIST_KPC'],
    'DISTANCE_ERR': ['dist_kpc_err', 'distance_kpc_err', 'dist_err', 'distance_err', 'DIST_KPC_ERR'],
    'DIST_MOD': ['dist_mod', 'DIST_MOD'],
    'DIST_MOD_ERR': ['dist_mod_err', 'DIST_MOD_ERR'],
    'phi1': ['phi1', 'PHI1'],
    'phi2': ['phi2', 'PHI2'],
}


def _get_series_by_alias(data, key):
    """Return a pandas Series / array for the first matching alias."""
    if data is None:
        return None
    candidates = COLUMN_ALIASES.get(key, [key])
    if isinstance(data, pd.DataFrame):
        for cand in candidates:
            if cand in data.columns:
                return data[cand]
    elif hasattr(data, 'dtype') and data.dtype.names:
        for cand in candidates:
            if cand in data.dtype.names:
                return data[cand]
    elif isinstance(data, pd.Series):
        if data.name in candidates:
            return data
    return None


def _get_numeric_array(data, key):
    """Return numeric numpy array for the alias, or NaNs if not available."""
    series = _get_series_by_alias(data, key)
    if series is None:
        if data is None:
            return None
        return np.full(len(data), np.nan)
    return _ensure_numeric(series)


def _ensure_dataframe_columns(df, fallbacks, constant_overrides=None):
    """Fill missing columns in a DataFrame using fallback columns or constants."""
    if not isinstance(df, pd.DataFrame):
        return df
    for target, candidates in fallbacks.items():
        needs_fill = target not in df or df[target].isna().all()
        if needs_fill:
            for cand in candidates:
                if cand in df and not df[cand].isna().all():
                    df[target] = df[cand]
                    needs_fill = False
                    break
    if constant_overrides:
        for target, value in constant_overrides.items():
            if target not in df or df[target].isna().all():
                df[target] = value
    return df


def _dist_from_dm(dm_values, dm_err_values=None):
    """Convert distance modulus arrays to distances (kpc) and optional errors."""
    if dm_values is None:
        return None, None
    dm_arr = _ensure_numeric(dm_values)
    if dm_arr is None:
        return None, None
    dist_pc = _ensure_numeric(stream_funcs.dist_mod_to_dist(dm_arr))
    if dist_pc is None:
        return None, None
    dist_kpc = dist_pc / 1000.0
    dist_err = None
    if dm_err_values is not None:
        dm_err_arr = _ensure_numeric(dm_err_values)
        if dm_err_arr is not None:
            dist_err = (np.log(10) / 5.0) * dist_kpc * dm_err_arr
    return dist_kpc, dist_err


def _extract_distance_series(data, label):
    """
    Attempt to pull distance (and optional uncertainty) arrays for plotting.
    """
    if data is None or len(data) == 0:
        return None, None

    base_candidates = COLUMN_ALIASES['DISTANCE']
    base_errors = COLUMN_ALIASES['DISTANCE_ERR']
    distance_candidates = {
        'MS+RG': base_candidates,
        'BHB': base_candidates + ['DISTANCE_KPC_BHB'],
        'RRL': ['dist_FEH'] + base_candidates,
    }
    error_candidates = {
        'MS+RG': base_errors,
        'BHB': base_errors,
        'RRL': ['dist_FEH_err'] + base_errors,
    }

    dist = _pick_numeric_column(data, distance_candidates.get(label, base_candidates))
    dist_err = _pick_numeric_column(data, error_candidates.get(label, base_errors))

    if dist is None:
        dm = _pick_numeric_column(data, COLUMN_ALIASES['DIST_MOD'])
        dm_err = _pick_numeric_column(data, COLUMN_ALIASES['DIST_MOD_ERR'])
        dist, dist_err = _dist_from_dm(dm, dm_err)

    return dist, dist_err

def zero_360_to_180(ra):
    ra_copy = np.copy(ra)
    where_180 = np.where(ra_copy > 180)
    ra_copy[where_180] = ra_copy[where_180] - 360
        
    return ra_copy
def get_mem_path(directory):
    import os
    files = os.listdir(directory)
    for file in files:
        if '0.5%_mem' in file and file.endswith('.fits'):
            return os.path.join(directory, file)

param_labels = ["lpstream",
                "v1","v2","v3","lsigv",
                "feh1","lsigfeh",
                "pmra1","pmra2","pmra3","lsigpmra",
                "pmdec1","pmdec2","pmdec3","lsigpmdec",
                "bv", "lsigbv", "bfeh", "lsigbfeh", "bpmra", "lsigbpmra", "bpmdec", "lsigbpmdec"]


def get_paramdict(theta, labels = param_labels):
    '''Make an ordered dictionary of the parameters as keys and inputted theta as values'''
    return OrderedDict(zip(labels, theta))

import orbit_functions as ofuncs
def plot_form(ax):
    ax.grid(ls='-.', alpha=0.2, zorder=0)
    ax.tick_params(direction='in')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.minorticks_on()

def process_chain(chain, avg_error=True, labels=param_labels):
    ''' Returns the means and errors of teh parameters
    
    Parameters:
    chain - array. The chain
    avg_error - bool. Will average the + and - errors if True
    
    Return:
    2 or 3 OrderedDict as a tuple. means, errors or means, errors+, errors-
    '''
    pctl = np.percentile(chain, [16, 50, 84], axis=0)
    meds = pctl[1]
    ep = pctl[2]-pctl[1]
    em = pctl[0]-pctl[1]

    if avg_error: # just for simplicity, assuming no asymmetry
        err = (ep-em)/2
        return OrderedDict(zip(labels, meds)), OrderedDict(zip(labels, err))
    else:
        return OrderedDict(zip(labels, meds)), OrderedDict(zip(labels, ep)), OrderedDict(zip(labels, em))
def plx_mask(min_dist, plx, plx_err):
    return (plx - 2*plx_err) < 1/min_dist

def d2dm(d):
    '''d in kpc'''
    return 5*np.log10(d*1000)-5

def load_stream_mems(mem_path = '/home/jupyter-nassermoha/raid_nassermoha/data/runs/C-19-I21_250529_final_const/C-19-I21_phi2_spline_0.5%_mem.fits'):

    stream_run_directory =  os.path.dirname(mem_path)
    stream_data = table.Table.read(mem_path)
    return stream_data, stream_run_directory

def load_desi_data(desi_path= '/raid/DESI/catalogs/loa/rv_output/241119/rvpix-loa.fits',
                  distance_path='/raid/DESI/catalogs/loa/rv_output/241119/rvsdistnn-loa-241126.fits',
                  decals_path='/raid/DESI/catalogs/loa/rv_output/241119/legacyphot-loa-241126.fits',
                  desired_columns=None, fr=None, local=False):
    """
    Joseph's code to load DESI data
    """
    if not local:
        desired_columns = [
        'VRAD', 'VRAD_ERR', 'RVS_WARN', 'PARALLAX', 'PARALLAX_ERROR', 
        'RR_SPECTYPE', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 'SOURCE_ID', 
        'TARGETID', 'PMRA_PMDEC_CORR', 'PRIMARY'
    ]
        desi_hdu_indices = [1, 4]
        desi_vrad_data = stream_funcs.load_fits_columns(desi_path, desired_columns, desi_hdu_indices)

        # Load Sergey's distance data
        dist_columns = ['TARGETID', 'dist_mod', 'dist_mod_err']
        distance_data  = stream_funcs.load_fits_columns(distance_path, dist_columns)

        # Load the DECaLS data
        decal_columns = ['EBV', 'FLUX_G', 'FLUX_R']
        decal_data = stream_funcs.load_fits_columns(decals_path, decal_columns)

        # Combine the data
        desi_data = table.hstack([desi_vrad_data, distance_data, decal_data])
        del desi_vrad_data, distance_data, decal_data

        # Delete the repeated TargetID column
        if len(np.where(desi_data['TARGETID_1'].value == desi_data['TARGETID_2'].value)[0]) == len(desi_data):
            desi_data.remove_columns(['TARGETID_2'])
            desi_data.rename_column('TARGETID_1', 'TARGETID')
            
        elif len(np.where(desi_data['TARGETID_1'].value == desi_data['TARGETID_2'].value)[0]) != len(desi_data):
            print('The TargetID columns do not match')


        # Drop the rows with NaN values in all columns
        print(f"Length of DESI Data before Cuts: {len(desi_data)}")
        drop_nan_columns = np.concatenate((desired_columns, decal_columns))
        desi_dropped_nan_df = stream_funcs.dropna_Table(desi_data, columns = drop_nan_columns) # Custom function to drop rows with NaN values
        print(f"Length of DESI Data after NaN cut: {len(desi_dropped_nan_df)}")

        # Drop the rows with 'RVS_WARN' != 0 and 'RR_SPECTYPE' != 'STAR', are not duplicates, and with low enough radial velocity and metallicity errors
        desi_dropped_vals = desi_dropped_nan_df[(desi_dropped_nan_df['RVS_WARN'] == 0) & (desi_dropped_nan_df['RR_SPECTYPE'] == 'STAR') & (desi_dropped_nan_df['PRIMARY']) &\
            (desi_dropped_nan_df['VRAD_ERR'] < 10) & (desi_dropped_nan_df['FEH_ERR'] < 0.5)]
        
        # Drop the rows with 'RVS_WARN' != 0 and 'RR_SPECTYPE' != 'STAR', are not duplicates, and with low enough radial velocity and metallicity errors
        sel_qual = (desi_dropped_nan_df['RVS_WARN'] == 0) & (desi_dropped_nan_df['RR_SPECTYPE'] == 'STAR') & (desi_dropped_nan_df['PRIMARY']) &\
            (desi_dropped_nan_df['VRAD_ERR'] < 10) & (desi_dropped_nan_df['FEH_ERR'] < 0.5)


        print(f"Length of DESI data after RVS_WARN, RR_SPECTYPE, PRIMARY, VRAD_ERR, and FEH_ERR: {len(desi_dropped_vals)}")

        # Drop the columns 'RVS_WARN' and 'RR_SPECTYPE' and convert to pandas DataFrame
        desi_dropped_vals.remove_columns(['RVS_WARN', 'RR_SPECTYPE'])
        desi_dropped_vals = desi_dropped_vals.to_pandas()

        # Add a floor to the uncertainties since they are underestimated
        desi_dropped_vals['VRAD_ERR'] = np.sqrt(desi_dropped_vals['VRAD_ERR']**2 + 0.9**2) ### Turn into its own column
        desi_dropped_vals['PMRA_ERROR'] = np.sqrt(desi_dropped_vals['PMRA_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
        desi_dropped_vals['PMDEC_ERROR'] = np.sqrt(desi_dropped_vals['PMDEC_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
        desi_dropped_vals['FEH_ERR'] = np.sqrt(desi_dropped_vals['FEH_ERR']**2 + 0.01**2) ### Turn into its own column

        # Delete some old variables
        del desi_dropped_nan_df, desi_data

        desi_data = desi_dropped_vals

        del desi_dropped_vals
        print('converting to phi1, phi2...')
        desi_data.loc[:,'phi1'], desi_data.loc[:,'phi2']  = stream_funcs.ra_dec_to_phi1_phi2(fr, np.array(desi_data['TARGET_RA'])*u.deg, np.array(desi_data['TARGET_DEC'])*u.deg)
        desi_data['VGSR'] =  np.array(stream_funcs.vhel_to_vgsr(np.array(desi_data['TARGET_RA'])*u.deg, np.array(desi_data['TARGET_DEC'])*u.deg, np.array(desi_data['VRAD'])*u.km/u.s).value)
        desi_data['VGSR_ERR'] = desi_data['VRAD_ERR']

        if distance_path:
            distance_data = stream_funcs.load_fits_columns(distance_path, ['TARGETID', 'dist_mod', 'dist_mod_err'])
            distance_data = pl.from_pandas(distance_data.to_pandas())
            desi_data_pl = pl.from_pandas(desi_data)

            # Ensure same dtype
            distance_data = distance_data.with_columns(
                distance_data['TARGETID'].cast(desi_data_pl['TARGETID'].dtype)
            )

            # Fast hash join
            desi_data_pl = desi_data_pl.join(distance_data, on='TARGETID', how='left')

            # Back to pandas if needed downstream
            desi_data = desi_data_pl.to_pandas()

            print("RVS distances added")

        return desi_data
    else:
         desi_data_tbl = table.Table.read(desi_path, format='fits')
         desi_data = desi_data_tbl.to_pandas()

         if distance_path:
            distance_data = stream_funcs.load_fits_columns(distance_path, ['TARGETID', 'dist_mod', 'dist_mod_err'])
            distance_data = pl.from_pandas(distance_data.to_pandas())
            desi_data_pl = pl.from_pandas(desi_data)

            # Ensure same dtype
            distance_data = distance_data.with_columns(
                distance_data['TARGETID'].cast(desi_data_pl['TARGETID'].dtype)
            )

            # Fast hash join
            desi_data_pl = desi_data_pl.join(distance_data, on='TARGETID', how='left')

            # Back to pandas if needed downstream
            desi_data = desi_data_pl.to_pandas()

            print("RVS distances added")
         
         return desi_data


def get_sel_qual_mask(
    desi_path='/raid/DESI/catalogs/loa/rv_output/241119/rvpix-loa.fits',
    distance_path='/raid/DESI/catalogs/loa/rv_output/241119/rvsdistnn-loa-241126.fits',
    decals_path='/raid/DESI/catalogs/loa/rv_output/241119/legacyphot-loa-241126.fits'
):
    desired_columns = [
        'VRAD', 'VRAD_ERR', 'RVS_WARN', 'PARALLAX', 'PARALLAX_ERROR', 
        'RR_SPECTYPE', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 'SOURCE_ID', 
        'TARGETID', 'PMRA_PMDEC_CORR', 'PRIMARY'
    ]
    decal_columns = ['EBV', 'FLUX_G', 'FLUX_R']
    dist_columns = ['TARGETID', 'dist_mod', 'dist_mod_err']

    # Load data
    desi = stream_funcs.load_fits_columns(desi_path, desired_columns, [1, 4])
    dist = stream_funcs.load_fits_columns(distance_path, dist_columns)
    decals = stream_funcs.load_fits_columns(decals_path, decal_columns)
    data = table.hstack([desi, dist, decals])

    # Fix duplicate TARGETID
    if 'TARGETID_2' in data.colnames:
        if np.all(data['TARGETID_1'] == data['TARGETID_2']):
            data.remove_columns(['TARGETID_2'])
            data.rename_column('TARGETID_1', 'TARGETID')

    # Drop NaNs
    drop_columns = desired_columns + decal_columns
    data = stream_funcs.dropna_Table(data, columns=drop_columns)

    # Return boolean selection mask
    return (data['RVS_WARN'] == 0) & \
           (data['RR_SPECTYPE'] == 'STAR') & \
           data['PRIMARY'] & \
           (data['VRAD_ERR'] < 10) & \
           (data['FEH_ERR'] < 0.5)

def import_mcmc_results(stream_run_directory):
    mcmc_dict = np.load(stream_run_directory + '/mcmc_dict.npy', allow_pickle=True).item()
    nested_dict = np.load(stream_run_directory + '/nested_dict.npy', allow_pickle=True).item()
    spline_points_dict = np.load(stream_run_directory + '/spline_points_dict.npy', allow_pickle=True).item()
    return mcmc_dict, nested_dict, spline_points_dict

def mag_select(data, mag_min, mag_max, mag_col='rmag0'):
    return (data[mag_col] < mag_min) & (data[mag_col] > mag_max)

def isochrone_cut(color_indx_wiggle = 0.10, isochrone_path= '/Users/nasserm/Documents/vscode/research/streamTut/DESI-DR1_streamTutorial/data/dotter/iso_a13.5_z0.00010.dat', 
                  desi_data=[], desi_distance=18e3, withAss=True):
    dotter_mp = np.loadtxt(isochrone_path)
    # Obtain the M_g and M_r color band data
    dotter_g_mp = dotter_mp[:,6]
    dotter_r_mp = dotter_mp[:,7]

    desi_ebv = np.array(desi_data['EBV'].values)
    desi_g_flux, desi_r_flux = np.array(desi_data['FLUX_G'].values), np.array(desi_data['FLUX_R'].values)

    # Custom function to calculate the color index (g-r), absolute R magnitude, and apparent r magnitude after EBV correction
    desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(desi_ebv, desi_g_flux, desi_r_flux, desi_distance)
    # Fit a line to the isochrone data. To use scipy's interpolate properly, the absolute magnitude values must be increasing, which is why we sort the values first

    g_r_color_dif = dotter_g_mp - dotter_r_mp
    sorted_indices = np.argsort(dotter_r_mp)
    sorted_dotter_r_mp = dotter_r_mp[sorted_indices]
    g_r_color_dif = g_r_color_dif[sorted_indices]

    # Fit for the isochrone line
    isochrone_fit = sp.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0)

    # Cut around the isochrone by the amount specified in color_indx_wiggle
    isochrone_cut = stream_funcs.betw(desi_colour_index, isochrone_fit(desi_abs_mag), color_indx_wiggle) 

    bhb_color_wiggle = 0.4
    bhb_abs_mag_wiggle = 0.1

    # build the BHB using empirical data from M92
    dm_m92_harris = 14.59 #dm of M92
    m92ebv = 0.023
    m92ag = m92ebv * 3.184
    m92ar = m92ebv * 2.130
    m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05])
    m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1])
    m92_hb_g = m92_hb_r + m92_hb_col
    des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
    des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
    des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
    des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris

    dm = 5 * np.log10(desi_distance) - 5


    bhb_cut = stream_funcs.isochrone_btw(desi_colour_index, desi_abs_mag, bhb_abs_mag_wiggle, bhb_color_wiggle, des_m92_hb_g - des_m92_hb_r, des_m92_hb_r)

    if withAss:
        return isochrone_cut | bhb_cut
    else:
        print('Not applying BHB cut')
        return isochrone_cut

class DataHandler:
    def __init__(self, frame, spline_points_dict, nested_dict):
        self.frame = frame
        self.spline_points_dict = spline_points_dict
        self.nested_dict = nested_dict
        self.data = None
        self.mask = None
        self.isochrone_path = None
        self.isochrone_distance_pc = None
        self.iso_mask = None
        self.iso_applied = False
        self.iso_withAss = True

    @staticmethod
    def _normalize_iso_distance(distance):
        """
        Accept either parsec or kpc inputs and store everything in parsec.
        Distances <= 0 are treated as undefined.
        """
        if distance is None:
            return None
        try:
            dist_val = float(distance)
        except (TypeError, ValueError):
            return None
        if dist_val <= 0:
            return None
        return dist_val if dist_val > 1000 else dist_val * 1000.0

    def configure_isochrone(self, path=None, distance_pc=None):
        """
        Store defaults for isochrone selections so downstream calls do not need
        to thread the information through every time.
        """
        if path:
            self.isochrone_path = path
        if distance_pc is not None:
            norm = self._normalize_iso_distance(distance_pc)
            if norm is not None:
                self.isochrone_distance_pc = norm
    # MASS ESTIMATION
    def apply_box_cut(self, pad=np.array([30. , -2. ,  0.5,  0.5, 10. ]), upper_bound_feh=True, MaskList=['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2']):
        self.touch_masks = {}  # initialize before the loop

        if self.data is not None:
            for i, MaskReturn in enumerate(MaskList):
                    k=i
                    if i==4:
                        k=3
                    touch_mask = get_touch_mask(
                        data=self.data, 
                        MaskReturn=MaskReturn,
                        pad=pad[i],
                        nested_list_meds=self.nested_dict['meds'], 
                        phi1_spline_points=self.spline_points_dict['phi1_spline_points'],
                        spline_k=self.spline_points_dict['spline_k'], 
                        upper_bound=upper_bound_feh,
                        meds_ind=[1, 3, 5, 6][k]
                    )
                    self.touch_masks[MaskReturn] = touch_mask

    def apply_plx_cut(self, min_dist):
        if self.data is not None:
            plx = _get_numeric_array(self.data, 'PARALLAX')
            plx_err = _get_numeric_array(self.data, 'PARALLAX_ERROR')
            if plx is None or len(plx) == 0:
                self.plx_mask = np.array([], dtype=bool)
                return
            if not np.any(np.isfinite(plx)):
                self.plx_mask = np.ones(len(plx), dtype=bool)
                return
            mask = plx_mask(min_dist, plx, plx_err)
            mask[~np.isfinite(plx)] = True
            mask[~np.isfinite(plx_err)] = True
            self.plx_mask = mask
        
    def apply_phi1_mask(self, phi1_min=-20, phi1_max=50):
        if self.data is not None:
            phi1 = self.data['phi1']
            self.phi1_mask = (phi1 > phi1_min) & (phi1 < phi1_max)

    def notmask(self,MaskList=['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2']):
        """
        Returns a combined mask (logical AND) of all masks in handler except the ones in exclude_keys.

        Parameters:
        - handler: object with touch_masks dict and other masks as attributes
        - exclude_keys: list of mask names to exclude, e.g., ["VGSR", "FEH"]
        """
        # Assuming self.touch_masks, self.plx_mask, self.phi1_mask are already defined
        # and MaskList is an iterable of iterables (e.g., list of lists of strings)

        self.not_masks = {}
        for i, exclude_keys_list in enumerate(MaskList): # Renamed to avoid confusion
            masks = None

            # Mapping of keys to their source
            key_to_mask = {
                "VGSR": self.touch_masks['VGSR'],
                "FEH": self.touch_masks['FEH'],
                "PMRA": self.touch_masks['PMRA'],
                "PMDEC": self.touch_masks['PMDEC'],
                "phi2": self.touch_masks['phi2'],
                "plx": self.plx_mask,
                "phi1": self.phi1_mask
            }

            # Ensure exclude_keys_list is in a consistent format for "not in" check, e.g., a set
            # If exclude_keys_list could be a single string, you'd need to handle that:
            if isinstance(exclude_keys_list, str):
                current_exclude_set = {exclude_keys_list}
                # For the key, we'll still use a tuple of one item
                dict_key_tuple = (exclude_keys_list,)
            else:
                # Convert to set for efficient "not in" lookup
                current_exclude_set = set(exclude_keys_list)
                # Create a sorted tuple for the dictionary key for consistency
                # (e.g., ['FEH', 'PMRA'] and ['PMRA', 'FEH'] become the same key)
                dict_key_tuple = tuple(sorted(list(current_exclude_set)))


            for key_name in key_to_mask.keys():
                if key_name not in current_exclude_set:
                    if masks is None:
                        masks = key_to_mask[key_name]
                    else:
                        masks = masks & key_to_mask[key_name]

            if masks is None:
                # This means current_exclude_set contained all keys from key_to_mask
                raise ValueError(f"All masks were excluded for exclusion set: {current_exclude_set}, no mask to combine.")

            self.not_masks[dict_key_tuple] = masks

    def all_box_cuts_DESI(self, withIso=True, withAss=True, isochrone_path=None, desi_distance=None):
        """
        Plotting function for applying masks excluding the current panel. Used for box_plot figures.

        More robust to cases where `touch_masks` isn't populated yet or `data` is None.
        """
        n = 0 if getattr(self, 'data', None) is None else len(self.data)

        # If no data available, nothing to select
        if n == 0:
            self.sel = np.array([], dtype=bool)
            return

        tm = getattr(self, 'touch_masks', {}) or {}

        # Safely pull masks; default to True if missing
        sel_vgsr = tm.get('VGSR', np.ones(n, dtype=bool))
        sel_feh  = tm.get('FEH', np.ones(n, dtype=bool))
        sel_pmra = tm.get('PMRA', np.ones(n, dtype=bool))
        sel_pmdec= tm.get('PMDEC', np.ones(n, dtype=bool))
        sel_phi2 = tm.get('phi2', np.ones(n, dtype=bool))

        # Other masks may also be missing; default to True
        sel_plx  = getattr(self, 'plx_mask', np.ones(n, dtype=bool))
        sel_phi1 = getattr(self, 'phi1_mask', np.ones(n, dtype=bool))

        # Determine the configuration for isochrone selection
        iso_path = isochrone_path or self.isochrone_path
        iso_distance = self._normalize_iso_distance(desi_distance) if desi_distance is not None else self.isochrone_distance_pc

        apply_iso = bool(withIso) and iso_path and iso_distance is not None
        sel_iso = np.ones(n, dtype=bool)

        if apply_iso:
            desi_data = self.data
            if not isinstance(desi_data, pd.DataFrame):
                desi_data = pd.DataFrame(desi_data)
            try:
                sel_iso = np.asarray(
                    isochrone_cut(
                        color_indx_wiggle=0.10,
                        isochrone_path=iso_path,
                        desi_data=desi_data,
                        desi_distance=iso_distance,
                        withAss=withAss,
                    ),
                    dtype=bool,
                )
            except KeyError as exc:
                print(f'Isochrone cut skipped due to missing columns ({exc}).')
                sel_iso = np.ones(n, dtype=bool)
        elif withIso:
            print('Not applying isochrone cut (disabled or missing configuration).')

        self.iso_mask = np.array(sel_iso, dtype=bool, copy=True)
        self.iso_applied = apply_iso
        self.iso_withAss = bool(withAss)

        sel = sel_vgsr & sel_feh & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_phi1 & sel_iso

        self.sel = sel

class GaiaDeCALSHandler:
    def __init__(self, stream_obj, GaiaDeCALS_data=None, stream_data=None, isochrone_path=None, min_dist=None):
        self.frame = stream_obj.frame
        self.spline_points_dict = stream_obj.spline_points_dict
        self.spline_k = stream_obj.spline_points_dict['spline_k']
        self.nested_dict = stream_obj.nested_dict
        self.data = GaiaDeCALS_data
        self.stream_data = stream_data
        self.isochrone_path = isochrone_path
        self.min_dist = min_dist
        self.pmra_idx = stream_obj.pmra_idx
        self.pmdec_idx = stream_obj.pmdec_idx
        print(f"Duplicates: {sum(self.data.duplicated(subset=['source_id']))}")
        self.data = self.data[self.data['release'] != 9011]
        print(f"Duplicates remaining: {sum(self.data.duplicated(subset=['source_id']))}")

        # Dealing with duplicates, keeping the instance with the smallest delta_mag
        self.data['delta_mag'] = np.abs(self.data['rmag']-self.data['phot_g_mean_mag'])
        self.data.sort_values('delta_mag', inplace=True)
        self.data = self.data[~self.data.duplicated(subset=['source_id'], keep='first')]
        # check if there are any duplicates remaining
        print(f"Duplicates remaining: {sum(self.data.duplicated(subset=['source_id']))}")

        print('Converting to phi1, phi2...')
        self.data['phi1'], self.data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(self.data['ra'])*u.deg, np.array(self.data['dec'])*u.deg)

        print('showing GaiaDeCALS data in stream frame...')
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(self.data['phi1'], self.data['phi2'], s=1, color='k', alpha=0.5, label='GaiaDeCALS')
        ax.scatter(self.stream_data['phi1'], self.stream_data['phi2'], s=5, color='orange', alpha=1, label='Stream')
        ax.set_xlabel(r'$\phi_1$ (deg)')
        ax.set_ylabel(r'$\phi_2$ (deg)')
        #show legend
        ax.legend(loc='upper right')
        plot_form(ax)

    def apply_box_cut(
        self,
        pad=np.array([0.3, 0.3, 10]), # MASS ESTIMATION
        phi1_spline_points=None,
        nested_list_meds=None,
        blind_panels=['pmra', 'pmdec', 'phi2'],
        blind_meds_ind=[5, 6],
        spline_k=None
    ):
        """ Apply box cuts to the GaiaDeCALS data based on the given pad values."""

        # Assign defaults from self if not provided
        if phi1_spline_points is None:
            phi1_spline_points = self.spline_points_dict['phi1_spline_points']
        if nested_list_meds is None:
            nested_list_meds = self.nested_dict['meds']
        if spline_k is None:
            spline_k = self.spline_k

        masks_dict = {}
        for i, panel in enumerate(blind_panels):
            if panel == 'phi2':
                reference_value = 0
            else:
                reference_value = reference_value = apply_spline(self.data['phi1'], phi1_spline_points, nested_list_meds[blind_meds_ind[i]], spline_k)
            mask = (np.abs(self.data[panel] - reference_value) < pad[i])
            masks_dict[panel] = mask
        
        sel_pmra, sel_pmdec, sel_phi2 = (masks_dict['pmra'], masks_dict['pmdec'], masks_dict['phi2'])
        sel_plx = plx_mask(self.min_dist/1000, self.data['parallax'], self.data['parallax_error'])

        x_arr = np.linspace(-15, 45, 1000)
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        ax[0].plot(x_arr, np.zeros_like(x_arr), color='cyan', ls='--', zorder=0)
        ax[1].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k), color='cyan', ls='--', zorder=0)
        ax[2].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k), color='cyan', ls='--', zorder=0)

        ax[1].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k) - pad[0], color='blue', lw=0.5)
        ax[1].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k) + pad[0], color='blue', lw=0.5)
        ax[2].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k) - pad[1], color='blue', lw=0.5)
        ax[2].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k) + pad[1], color='blue', lw=0.5)

        ax[0].scatter(self.data['phi1'][sel_pmra & sel_pmdec & sel_plx & sel_phi2], self.data['phi2'][sel_pmra & sel_pmdec & sel_plx & sel_phi2], s=0.5, alpha=0.1, c='b')
        ax[1].scatter(self.data['phi1'][sel_phi2 & sel_pmdec & sel_plx & sel_pmra], self.data['pmra'][sel_phi2 & sel_pmdec & sel_plx & sel_pmra], s=0.5, alpha=0.1, c='b')
        ax[2].scatter(self.data['phi1'][sel_phi2 & sel_pmra & sel_plx & sel_pmdec], self.data['pmdec'][sel_phi2 & sel_pmra & sel_plx & sel_pmdec], s=0.5, alpha=0.1, c='b')

        ax[0].scatter(self.data['phi1'][sel_pmra & sel_pmdec & sel_plx], self.data['phi2'][sel_pmra & sel_pmdec & sel_plx], s=0.1, alpha=0.1, c='0.5')
        ax[1].scatter(self.data['phi1'][sel_phi2 & sel_pmdec & sel_plx], self.data['pmra'][sel_phi2 & sel_pmdec & sel_plx], s=0.1, alpha=0.1, c='0.5')
        ax[2].scatter(self.data['phi1'][sel_phi2 & sel_pmra & sel_plx], self.data['pmdec'][sel_phi2 & sel_pmra & sel_plx], s=0.1, alpha=0.1, c='0.5')

        # plot stream data as orange
        if self.stream_data is not None:
            ax[0].scatter(self.stream_data['phi1'], self.stream_data['phi2'], s=1, color='orange', alpha=1, label='Stream')
            ax[1].scatter(self.stream_data['phi1'], self.stream_data['PMRA'], s=1, color='orange', alpha=1)
            ax[2].scatter(self.stream_data['phi1'], self.stream_data['PMDEC'], s=1, color='orange', alpha=1)

        ax[0].set_ylabel(r'$\phi_2$ (deg)')
        ax[1].set_ylabel(r'$\mu_{\alpha}$ (mas/yr)')
        ax[2].set_ylabel(r'$\mu_{\delta}$ (mas/yr)')

        ax[0].set_ylim(pad[-1] * -1.5, pad[-1] * 1.5)
        ax[1].set_ylim(apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k).min() - pad[0] * 1.4,
                    apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k).max() + pad[0] * 1.4)
        ax[2].set_ylim(apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k).min() - pad[1] * 1.4,
                    apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k).max() + pad[1] * 1.4)


        for a in ax:
            plot_form(a)

        print(f'Applied cuts: {pad} for panels: {blind_panels}, with a parallax cut')
        self.sel_pmra = sel_pmra
        self.sel_pmdec = sel_pmdec
        self.sel_plx = sel_plx
        self.sel_phi2 = sel_phi2
        self.sel_box = sel_pmra & sel_pmdec & sel_plx & sel_phi2

    def apply_iso_cut(self, color_indx_wiggle=None):

        sel_pmra = self.sel_pmra
        sel_pmdec = self.sel_pmdec
        sel_phi2 = self.sel_phi2
        sel_plx = self.sel_plx
        stream_data = self.stream_data
        desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(stream_data['EBV'], stream_data['FLUX_G'], stream_data['FLUX_R'], self.min_dist*1000)

        color_indx_wiggle = color_indx_wiggle if color_indx_wiggle is not None else 0.075 # MASS ESTIMATION

        dotter_mp = np.loadtxt(self.isochrone_path)
        self.dotter_mp = dotter_mp
        dotter_g_mp = dotter_mp[:,6] # Absolute magnitude gband, M_g
        dotter_r_mp = dotter_mp[:,7] # Absolute magnitude rband, M_r
        dotter_z_mp = dotter_mp[:,9] # Absolute magnitude zband, M_z

        g_r_color_dif = dotter_g_mp - dotter_r_mp # color
        sorted_indices = np.argsort(dotter_r_mp)  # sorting to go from most green to most red
        sorted_dotter_r_mp = dotter_r_mp[sorted_indices] 
        g_r_color_dif = g_r_color_dif[sorted_indices] 

        isochrone_fit = sp.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0) # function of colour as a function of absolute magnitude
        gaia_colour_index, gaia_abs_mag, gaia_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data['ebv'], self.data['flux_g'], self.data['flux_r'], self.min_dist*1000)
        sel_iso = abs(self.data['gmag0']-self.data['rmag0']-apply_spline(self.data['rmag0']-d2dm(self.min_dist),dotter_r_mp[::-1], dotter_g_mp[::-1]-dotter_r_mp[::-1]-0.01, k=1)) < color_indx_wiggle
        # MASS ESTIMATION
        mag_min = 20.5
        mag_max = 16.0
        sel_mag = mag_select(self.data, mag_min=mag_min, mag_max=mag_max, mag_col='rmag0')
        "CHANGE ABOVE TO GET DIFFERENT MAG CUTS, NOT GONNA MAKE THIS PART USER FRIENDLY SORRY"

        self.sel_iso = sel_iso 
        self.sel_mag = sel_mag

        fig, ax = plt.subplots(1, 1, figsize=(4, 6))

        sel2 = self.sel_mag & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_iso
        ax.scatter(gaia_colour_index[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & ~self.sel_iso], gaia_abs_mag[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & ~self.sel_iso], color='0.2', s=0.5, alpha=0.05)
        ax.scatter(gaia_colour_index[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & self.sel_iso], gaia_abs_mag[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & self.sel_iso], color='blue', s=1, alpha=0.1)
        ax.scatter(desi_colour_index, desi_abs_mag, color='orange', s=10, alpha=1, marker='s', label=r'Stream Stars')
        ax.plot(isochrone_fit(sorted_dotter_r_mp), sorted_dotter_r_mp, color='red', lw=2, label='Isochrone Fit')
        ax.plot(isochrone_fit(sorted_dotter_r_mp)+color_indx_wiggle, sorted_dotter_r_mp, color='red', lw=1, ls='-.')
        ax.plot(isochrone_fit(sorted_dotter_r_mp)-color_indx_wiggle, sorted_dotter_r_mp, color='red', lw=1, ls='-.')
        ax.axhline(mag_max-5*np.log10(18e3)+5, ls='dotted', c='k', label=rf'$r={mag_max}$')
        ax.axhline(mag_min-5*np.log10(18e3)+5, ls='dotted', c='k', label=rf'$r={mag_min}$')

        ax.invert_yaxis()
        ax.set_ylim(5, -3)
        ax.set_xlim(-0.5, 1)
        ax.set_xlabel('Colour Index (g - r)')
        ax.set_ylabel('Absolute Magnitude (M_r)')
        ax.set_title(r'Gaia x DECaLS Data: Isochrone Cut from $r \in$'+ f'({mag_min}, {mag_max})')
        ax.legend()
        plot_form(ax)

        self.sel_all = sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_iso & sel_mag
    
    def apply_phot_metallicity(self):
        from scipy.interpolate import interp1d
        mpsel = self.data['rmag0']-self.data['zmag0'] - (0.24/0.25)*(self.data['gmag0']-self.data['rmag0']-0.5) - 0.24 > 0 # Empirical, from Ting
        sel_all_old = self.sel_all
        self.sel_all = sel_all_old & mpsel
        print(np.sum(self.sel_all))

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.scatter(self.data['gmag0'][sel_all_old]-self.data['rmag0'][sel_all_old], self.data['rmag0'][sel_all_old]-self.data['zmag0'][sel_all_old], marker='.', s=10, c='0.2')
        ax.scatter(self.data['gmag0'][self.sel_all]-self.data['rmag0'][self.sel_all], self.data['rmag0'][self.sel_all]-self.data['zmag0'][self.sel_all], marker='.', s=10, c='blue')

        #ax.scatter(self.stream_data['gmag0']-self.stream_data['rmag0'], self.stream_data['rmag0']-self.stream_data['zmag0'], color='tab:orange', marker='s')
        #ax.scatter(stream_data_joined['gmag0']-stream_data_joined['rmag0'], stream_data_joined['rmag0']-stream_data_joined['zmag0'], color='tab:orange', marker='x')

        ax.set_xlabel('g-r',fontsize=15)
        ax.set_ylabel('r-z',fontsize=15)
        ax.set_aspect('equal')
        #ax.plot(dotter_g_mp[s:r] - dotter_r_mp[s:r], dotter_r_mp[s:r] - dotter_z_mp[s:r], color='0.2', lw=1,label='Isochrone Fit')
        #ax.plot(dotter_g_mp[s:r] - dotter_r_mp[s:r], dotter_r_mp[s:r] - dotter_z_mp[s:r], color='0.5', lw=1,label='Isochrone Fit')

        #ax.plot(dotter_g_mp - dotter_r_mp, dotter_r_mp - dotter_z_mp, color='0.2', lw=1, ls = 'dotted', label='Isochrone Fit')

        ax.set_xlim(0.1,0.8)
        ax.set_ylim(-0.1,0.5)

        x = np.linspace(0.1,0.8,100)
        y = (0.24/0.25)*(x-0.5) + 0.24

        ax.plot(x,y, c='red')

        plot_form(ax)
def dm_to_distance(dm, dmerr=None, dmerr_plus=None, dmerr_minus=None, to_kpc=False):
    """
    Convert distance modulus (dm) to linear distance, returning asymmetric
    or symmetric uncertainties as appropriate.

    Parameters
    ----------
    dm : float or array-like
        Distance modulus.
    dmerr : float or array-like, optional
        1-σ *symmetric* uncertainty on dm.  Ignored if dmerr_plus/minus are given.
    dmerr_plus : float or array-like, optional
        Positive (upper) uncertainty on dm.
    dmerr_minus : float or array-like, optional
        Negative (lower) uncertainty on dm.
    to_kpc : bool, default False
        If True, distances are returned in kiloparsecs instead of parsecs.

    Returns
    -------
    d : float or ndarray
        Central distance (pc or kpc).
    derr_plus : float or ndarray
        Upper uncertainty on distance (same units as d).
    derr_minus : float or ndarray
        Lower uncertainty on distance (same units as d).
        For symmetric-error input, derr_plus == derr_minus.
    """
    # central distance
    d = 10.0**((np.asarray(dm) + 5.0) / 5.0)

    # choose error treatment
    if dmerr_plus is not None and dmerr_minus is not None:
        # asymmetric case
        d_hi = 10.0**(((dm + dmerr_plus) + 5.0) / 5.0)
        d_lo = 10.0**(((dm - dmerr_minus) + 5.0) / 5.0)
        derr_plus  = d_hi - d
        derr_minus = d - d_lo
    elif dmerr is not None:
        # symmetric case – linear propagation
        factor = np.log(10.0) / 5.0           # ∂d/∂dm divided by d
        derr_plus = derr_minus = d * factor * dmerr
    else:
        raise ValueError("Provide either dmerr (symmetric) or both dmerr_plus and dmerr_minus.")

    if to_kpc:
        d          = d / 1e3
        derr_plus  = derr_plus  / 1e3
        derr_minus = derr_minus / 1e3

    return d, [derr_minus, derr_plus]
def get_touch_mask(data, MaskReturn=[], pad=[], nested_list_meds=[], phi1_spline_points=[], meds_ind = 1, spline_k=1, upper_bound=True):
    """
    Create a mask for called value
    """
    mask = None
    values = None
    if len(data) == 0:
        return np.array([], dtype=bool)
    if MaskReturn == 'phi2':
        values = _ensure_numeric(data['phi2'])
    else:
        values = _get_numeric_array(data, MaskReturn)

    if MaskReturn=='FEH':
        if not np.any(np.isfinite(values)):
            mask = np.ones(len(data), dtype=bool)
        elif upper_bound:
            mask = values < pad
        else:
            reference_value = nested_list_meds[3]
            mask = np.abs(values - reference_value) < pad
    else:
        if MaskReturn == 'phi2':
            reference_value = 0
        else:
            reference_value = apply_spline(data['phi1'], phi1_spline_points, nested_list_meds[meds_ind], spline_k)
        mask = np.abs(values - reference_value) < pad
        mask[~np.isfinite(values)] = False
    return mask

    


class StreamMembers:
    """
    Class to handle post-mcmc processing of stream
    """
    def __init__(self, withBHB=True, withRRL=True, min_dist=18e3,
                stream_data=[],
                stream_run_directory=None,
                isochrone_path='',
                desi_data=None, fr=None):

        self.frame=fr
        # Resolve isochrone path: use provided path or discover in repo
        def _resolve_iso_path(path_hint:str|None):
            import glob, os
            # If a valid hint is passed, use it
            if path_hint and os.path.exists(path_hint):
                return path_hint
            # Try repository data location
            here = os.path.dirname(__file__)
            cand_dir = os.path.join(here, 'data', 'dotter')
            patterns = [
                os.path.join(cand_dir, 'iso_a13.5_z0.00010.dat'),
                os.path.join(cand_dir, 'iso_a13.5_z0.00006.dat'),
                os.path.join(cand_dir, 'iso_a*.dat'),
            ]
            for pat in patterns:
                cands = sorted(glob.glob(pat))
                if cands:
                    return cands[0]
            # Fall back to legacy hardcoded location if it exists
            legacy = '/Users/nasserm/Documents/vscode/research/streamTut/DESI-DR1_streamTutorial/data/dotter/iso_a13.5_z0.00010.dat'
            if os.path.exists(legacy):
                return legacy
            return path_hint or legacy

        self.isochrone_path = _resolve_iso_path(isochrone_path)
        if not os.path.exists(self.isochrone_path):
            print(f"Warning: could not find isochrone at '{self.isochrone_path}'. Set 'isochrone_path' when constructing StreamMembers or place a Dotter file under data/dotter/.")
        self.stream_data, self.stream_run_directory = stream_data, stream_run_directory
        self.min_dist = min_dist
        # go to stream_run_directory and get path to fits file that contains C-19-I21_phi2_spline_all%_mem.fits
        # self.stream_run_directory = os.path.dirname(self.stream_run_directory)
        if self.stream_run_directory:
            import glob
            pattern = os.path.join(self.stream_run_directory, "*all%_mem.fits")
            files = glob.glob(pattern)
            if files:
                all_mems_path = files[0]  # take first match
                self.all_memberships = table.Table.read(all_mems_path)
                self.all_memberships = self.all_memberships.to_pandas()
            else:
                raise FileNotFoundError(f"No file ending with 'all%_mem.fits' found in {self.stream_run_directory}.")
        else:
            print('no stream directory given, moving on...')
        

        # Ensure optional handlers exist even if we skip loading
        self.bhb_handler = None
        self.rrl_handler = None

        print('Loading desi_data...')
        # if desi_data is not an array or a pandas DataFrame
        if not isinstance(desi_data, (np.ndarray, pd.DataFrame)):
            self.desi_data = None
            print('No desi_data provided, will not use DESI data')
        else:
            self.desi_data = desi_data
            print('Desi data loaded')
        if stream_run_directory is not None:
            self.mcmc_dict, self.nested_dict, self.spline_points_dict = import_mcmc_results(self.stream_run_directory)

            # Create data handlers
            self.ms_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
            self.ms_handler.configure_isochrone(self.isochrone_path, self.min_dist)
            self.ms_handler.data = self.desi_data
        
        if not isinstance(desi_data, (np.ndarray, pd.DataFrame)):
            print('Skipping BHB and RRL Data')
        else:
            print('getting BHB and RRL data...')
            if withBHB:
                bhb_path = '/Users/nasserm/Documents/vscode/research/streamTut/DESI-DR1_streamTutorial/data/loa_bhb_250116.fits' #NOTE
                bhb_data = table.Table.read(bhb_path)
                bhb_data['phi1'], bhb_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(bhb_data['TARGET_RA'])*u.deg, np.array(bhb_data['TARGET_DEC'])*u.deg)
                bhb_desi = bhb_data.to_pandas().merge(self.ms_handler.data, how='left', on=['TARGETID'], suffixes=('', '_desi'))
                bhb_fallbacks = {
                    'VGSR': ['VGSR_desi', 'VHELIO_AVG', 'RV_GSR'],
                    'VGSR_ERR': ['VGSR_ERR_desi', 'VRAD_ERR', 'SIGMA_RV'],
                    'FEH': ['FEH_desi'],
                    'PMRA': ['PMRA_desi'],
                    'PMRA_ERROR': ['PMRA_ERROR_desi'],
                    'PMDEC': ['PMDEC_desi'],
                    'PMDEC_ERROR': ['PMDEC_ERROR_desi'],
                    'PARALLAX': ['PARALLAX_desi'],
                    'PARALLAX_ERROR': ['PARALLAX_ERROR_desi'],
                    'dist_mod': ['dist_mod_desi'],
                    'dist_mod_err': ['dist_mod_err_desi'],
                }
                feh_default = self.nested_dict['meds'][3] if getattr(self, 'nested_dict', None) else np.nan
                bhb_desi = _ensure_dataframe_columns(
                    bhb_desi,
                    bhb_fallbacks,
                    constant_overrides={'FEH': feh_default}
                )
                self.bhb_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
                self.bhb_handler.data = bhb_desi
            else:
                self.bhb_handler = None
            
            if withRRL:
                rrl_path = '/Users/nasserm/Documents/vscode/research/streamTut/DESI-DR1_streamTutorial/data/DESI_loa_VAC_v0.2.csv' #NOTE
                rrl_data = pd.read_csv(rrl_path)
                rrl_data['phi1'], rrl_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(rrl_data['TARGET_RA'])*u.deg, np.array(rrl_data['TARGET_DEC'])*u.deg)
                rrl_data['PMRA'], rrl_data['PMDEC'] = rrl_data['pmra'], rrl_data['pmdec']
                rrl_data['PMRA_ERROR'], rrl_data['PMDEC_ERROR'] = rrl_data['pmra_error'], rrl_data['pmdec_error']
                rrl_data['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(rrl_data['TARGET_RA'])*u.deg, np.array(rrl_data['TARGET_DEC'])*u.deg, np.array(rrl_data['v0_mean'])*u.km/u.s).value)
                rrl_data['VGSR_ERR'] = rrl_data['v0_std']
                self.rrl_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
                self.rrl_handler.data = rrl_data
                self.rrl_handler.data['PARALLAX'] = self.rrl_handler.data['parallax']
                self.rrl_handler.data['PARALLAX_ERROR'] = self.rrl_handler.data['parallax_error']
                rrl_fallbacks = {
                    'FEH': ['feh', 'FEH_MS'],
                    'dist_mod': ['dist_mod_rrl'],
                    'dist_mod_err': ['dist_mod_err_rrl'],
                }
                feh_default = self.nested_dict['meds'][3] if getattr(self, 'nested_dict', None) else np.nan
                self.rrl_handler.data = _ensure_dataframe_columns(
                    self.rrl_handler.data,
                    rrl_fallbacks,
                    constant_overrides={'FEH': feh_default}
                )
            else:
                self.rrl_handler = None
        self.vgsr_idx = 1
        self.pmra_idx = 5
        labels = self.mcmc_dict.get('extended_param_labels', []) if getattr(self, 'mcmc_dict', None) is not None else []
        self.pmdec_idx = 7 if 'lsigpmra' in labels else 6

        print('Done')
    
    def _compute_axis_limits(self, active_handlers, pad):
        """
        Compute dynamic axis limits based on data and spline tracks.
        
        Parameters
        ----------
        active_handlers : list of tuples
            List of (label, handler, skip_msg) tuples for active data handlers
        pad : array-like
            Padding values for each coordinate: [vgsr, feh, pmra, pmdec, phi2]
            
        Returns
        -------
        phi1_range : tuple
            (min, max) range for phi1 axis
        data_ranges : dict
            Dictionary mapping coordinate names to (min, max) ylim tuples
        """
        pad = np.asarray(pad, dtype=float).reshape(-1)
        
        # Collect all phi1 values and spline points to determine x-range
        all_phi1 = []
        spline_points_all = []
        
        # Collect data ranges for each coordinate
        all_phi2 = []
        all_vgsr = []
        all_pmra = []
        all_pmdec = []
        all_feh = []
        all_dist = []
        
        for label, handler, _ in active_handlers:
            if handler is None or handler.data is None or len(handler.data) == 0:
                continue
            
            data = handler.data
            
            # Collect phi1 values
            phi1_vals = _get_numeric_array(data, 'phi1')
            if phi1_vals is not None:
                finite_phi1 = phi1_vals[np.isfinite(phi1_vals)]
                if len(finite_phi1) > 0:
                    all_phi1.extend(finite_phi1)
            
            # Collect spline points
            if hasattr(handler, 'spline_points_dict') and handler.spline_points_dict is not None:
                sp = handler.spline_points_dict.get('phi1_spline_points')
                if sp is not None:
                    spline_points_all.extend(np.asarray(sp).flatten())
            
            # Collect coordinate values
            for vals, col_list in [
                (all_phi2, ['phi2', 'PHI2']),
                (all_vgsr, COLUMN_ALIASES['VGSR']),
                (all_pmra, COLUMN_ALIASES['PMRA']),
                (all_pmdec, COLUMN_ALIASES['PMDEC']),
                (all_feh, COLUMN_ALIASES['FEH']),
            ]:
                col_vals = _pick_numeric_column(data, col_list)
                if col_vals is not None:
                    finite_vals = col_vals[np.isfinite(col_vals)]
                    if len(finite_vals) > 0:
                        vals.extend(finite_vals)
            
            # Distance requires special handling
            dist_vals, _ = _extract_distance_series(data, label)
            if dist_vals is not None:
                dist_arr = np.asarray(dist_vals, dtype=float)
                finite_dist = dist_arr[np.isfinite(dist_arr) & (dist_arr > 0)]
                if len(finite_dist) > 0:
                    all_dist.extend(finite_dist)
        
        # Compute phi1 range from spline points with padding
        if len(spline_points_all) > 0:
            phi1_min = np.min(spline_points_all) - 5
            phi1_max = np.max(spline_points_all) + 5
        elif len(all_phi1) > 0:
            phi1_min = np.min(all_phi1) - 5
            phi1_max = np.max(all_phi1) + 5
        else:
            # Fallback
            phi1_min, phi1_max = -10, 40
        
        phi1_range = (phi1_min, phi1_max)
        
        # Compute y-axis ranges with padding factor
        def _compute_ylim(values, margin_factor=0.15):
            """Compute ylim with margin around data range."""
            if len(values) == 0:
                return None
            vmin, vmax = np.min(values), np.max(values)
            margin = (vmax - vmin) * margin_factor
            if margin < 1e-6:
                margin = abs(vmin) * 0.1 if vmin != 0 else 1.0
            return (vmin - margin, vmax + margin)
        
        # Compute ranges for spline-tracked quantities using spline values
        # For vgsr, pmra, pmdec: use spline track range plus padding tolerance
        data_ranges = {}
        
        # phi2: symmetric around zero based on pad[4] or data range
        if len(all_phi2) > 0:
            phi2_extent = max(abs(np.min(all_phi2)), abs(np.max(all_phi2)), pad[4])
            data_ranges['phi2'] = (-phi2_extent * 1.3, phi2_extent * 1.3)
        else:
            data_ranges['phi2'] = (-pad[4] * 2, pad[4] * 2)
        
        # For tracked quantities, use data range plus some margin
        if len(all_vgsr) > 0:
            data_ranges['vgsr'] = _compute_ylim(all_vgsr, margin_factor=0.2)
        
        if len(all_pmra) > 0:
            data_ranges['pmra'] = _compute_ylim(all_pmra, margin_factor=0.3)
        
        if len(all_pmdec) > 0:
            data_ranges['pmdec'] = _compute_ylim(all_pmdec, margin_factor=0.3)
        
        if len(all_feh) > 0:
            data_ranges['feh'] = _compute_ylim(all_feh, margin_factor=0.2)
        
        if len(all_dist) > 0:
            # For distance on log scale, use data range with some margin
            dist_min = max(0.5, np.min(all_dist) * 0.5)
            dist_max = np.max(all_dist) * 2.0
            data_ranges['dist'] = (dist_min, dist_max)
        else:
            data_ranges['dist'] = (1, 100)  # Default fallback
        
        return phi1_range, data_ranges
        
    def box_cut(self, pad, with_plot=True, save_fig=False, fig_path=None, residual=True, show_panels=None, **kwargs):
        """
        Apply box cuts to select stream members across multiple phase-space coordinates.
        
        For each panel, showing one phase-space coordinate as a function of phi1, stars 
        remaining after all other selection cuts are shown (i.e. the vgsr panel shows stars 
        selected with phi2, metallicity, and proper motions). The dashed lines show the 
        tolerance about the spline track fit used to select likely member stars.
        
        Parameters
        ----------
        pad : array-like
            Tolerance values for each coordinate: [vgsr, feh, pmra, pmdec, phi2]
        with_plot : bool, optional
            Whether to generate the diagnostic plot (default: True)
        save_fig : bool, optional
            Whether to save the figure (default: False)
        fig_path : str, optional
            Path to save the figure if save_fig is True
        residual : bool, optional
            Whether to show residual panels for vgsr, pmra, pmdec (default: True)
        show_panels : list of int, optional
            Which panels to display. Panel indices:
            0: phi2, 1: vgsr, 2: pmra, 3: pmdec, 4: feh, 5: dist
            If None, all panels are shown.
        **kwargs : dict
            Additional keyword arguments:
            - highlight : array-like, optional - Indices to highlight
            - withIso : bool - Apply isochrone filter (default: True)
            - withAss : bool - Include BHB allowance in isochrone (default: True)
            - isochrone_path : str - Override isochrone file path
            - iso_distance : float - Override isochrone distance (pc)
            - xlim : tuple, optional - Override x-axis (phi1) limits
            - ylim_phi2 : tuple, optional - Override phi2 y-axis limits
            - ylim_vgsr : tuple, optional - Override vgsr y-axis limits  
            - ylim_pmra : tuple, optional - Override pmra y-axis limits
            - ylim_pmdec : tuple, optional - Override pmdec y-axis limits
            - ylim_feh : tuple, optional - Override feh y-axis limits
            - ylim_dist : tuple, optional - Override distance y-axis limits
            - phi1_exclude : tuple, optional - (min, max) phi1 range to exclude from plot
              e.g., phi1_exclude=(-0.75, 0.75) masks out stars in that range
        """
        print('Applying box cuts...')
        highlight = kwargs.get('highlight', None)
        residual = kwargs.pop('residual', residual)
        show_panels = kwargs.pop('show_panels', show_panels)
        self.pad = pad
        with_iso = kwargs.get('withIso', True)
        with_ass = kwargs.get('withAss', True)
        iso_path_override = kwargs.get('isochrone_path', None)
        iso_distance_override = kwargs.get('iso_distance', None)
        
        # Extract optional axis limit overrides
        xlim_override = kwargs.get('xlim', None)
        ylim_overrides = {
            'phi2': kwargs.get('ylim_phi2', None),
            'vgsr': kwargs.get('ylim_vgsr', None),
            'pmra': kwargs.get('ylim_pmra', None),
            'pmdec': kwargs.get('ylim_pmdec', None),
            'feh': kwargs.get('ylim_feh', None),
            'dist': kwargs.get('ylim_dist', None),
        }
        
        # Extract phi1 exclusion range (e.g., to mask out a region like -0.75 to 0.75)
        phi1_exclude = kwargs.get('phi1_exclude', None)
        
        ms = getattr(self, 'ms_handler', None)
        bhb = getattr(self, 'bhb_handler', None)
        rrl = getattr(self, 'rrl_handler', None)

        handler_entries = []
        if ms is not None:
            handler_entries.append(('MS+RG', ms, 'Skipping MS+RG plot: no DESI data available.'))
        if bhb is not None:
            handler_entries.append(('BHB', bhb, 'Skipping BHB plot: no BHB data available.'))
        if rrl is not None:
            handler_entries.append(('RRL', rrl, 'Skipping RRL plot: no RRL data available.'))

        active_handlers = []
        for label, handler, skip_msg in handler_entries:
            if handler is None:
                continue
            if handler.data is not None:
                if label == 'MS+RG':
                    handler.apply_box_cut(pad=pad)
                    handler.apply_plx_cut(min_dist=self.min_dist)
                    handler.apply_phi1_mask()
                    if iso_path_override or iso_distance_override is not None:
                        handler.configure_isochrone(
                            iso_path_override,
                            iso_distance_override if iso_distance_override is not None else None,
                        )
                    handler.all_box_cuts_DESI(
                        withIso=with_iso,
                        withAss=with_ass,
                        isochrone_path=iso_path_override,
                        desi_distance=(
                            iso_distance_override
                            if iso_distance_override is not None
                            else handler.isochrone_distance_pc
                        ),
                    )
                else:
                    box_kwargs = {'pad': pad} if label != 'RRL' else {}
                    handler.apply_box_cut(**box_kwargs)
                    handler.apply_phi1_mask()
                    handler.apply_plx_cut(min_dist=self.min_dist)
                    handler.all_box_cuts_DESI(withIso=False)
            else:
                handler.sel = np.array([], dtype=bool)
            active_handlers.append((label, handler, skip_msg))

        concat_frames = [
            handler.data[getattr(handler, 'sel')]
            for _, handler, _ in active_handlers
            if handler is not None and handler.data is not None and getattr(handler, 'sel', None) is not None
        ]

        if len(concat_frames) > 0:
            self.box_data = pd.concat(concat_frames, ignore_index=True)
        else:
            # create empty DataFrame with at least phi1 column to avoid downstream issues
            self.box_data = pd.DataFrame()
        print('Box cuts applied. Access total masks using StreamMembers.<handler>.sel')
        if with_plot:
            print('Plotting box cuts...')

            import matplotlib.gridspec as gridspec
            
            # Compute dynamic axis limits from data and spline tracks
            phi1_range, data_ranges = self._compute_axis_limits(active_handlers, pad)
            
            # Apply user overrides if provided
            if xlim_override is not None:
                phi1_range = xlim_override
            for key, override_val in ylim_overrides.items():
                if override_val is not None:
                    data_ranges[key] = override_val
            
            panel_definitions = [
                (0, 'phi2', 1.5, {'ylabel': r'$\phi_2$ (deg)', 'ylim': data_ranges.get('phi2')}, None),
                (1, 'vgsr', 1.5, {'ylabel': r'$V_{GSR}$ (km/s)', 'ylim': data_ranges.get('vgsr')},
                 ('dvgsr', 1.0, {'ylabel': r'$\Delta V_{GSR}$ (km/s)', 'ylim': (-pad[0]*1.5, pad[0]*1.5)})),
                (2, 'pmra', 1.5, {'ylabel': r'$\mu_{\alpha}$ (mas/yr)', 'ylim': data_ranges.get('pmra')},
                 ('dpmra', 1.0, {'ylabel': r'$\Delta \mu_{\alpha}$ (mas/yr)', 'ylim': (-pad[2]*1.5, pad[2]*1.5)})),
                (3, 'pmdec', 1.5, {'ylabel': r'$\mu_{\delta}$ (mas/yr)', 'ylim': data_ranges.get('pmdec')},
                 ('dpmdec', 1.0, {'ylabel': r'$\Delta \mu_{\delta}$ (mas/yr)', 'ylim': (-pad[3]*1.5, pad[3]*1.5)})),
                (4, 'feh', 1.5, {'ylabel': r'[Fe/H]', 'ylim': data_ranges.get('feh')}, None),
                (5, 'dist', 1.5, {'ylabel': 'Distance (kpc)', 'yscale': 'log', 'ylim': data_ranges.get('dist', (1, 100))}, None),
            ]
            if show_panels is None:
                show_set = {panel_id for panel_id, *_ in panel_definitions}
            else:
                try:
                    show_set = set(int(idx) for idx in show_panels)
                except Exception as exc:
                    raise TypeError("show_panels must be an iterable of integers.") from exc
                if not show_set:
                    raise ValueError("show_panels must contain at least one panel index.")
            panel_groups = []
            for panel_id, name, height, props, residual_def in panel_definitions:
                if panel_id not in show_set:
                    continue
                main_height = 1.6 if name == 'phi2' else 1.2
                group_entry = {
                    'panel_id': panel_id,
                    'height': main_height,
                    'main': {
                        'name': name,
                        'props': props,
                        'height': main_height,
                    },
                    'residual': None,
                }
                if residual and residual_def is not None:
                    res_name, _, res_props = residual_def
                    group_entry['residual'] = {
                        'name': res_name,
                        'props': res_props,
                        'height': 0.6,
                    }
                    group_entry['height'] += group_entry['residual']['height']
                panel_groups.append(group_entry)
            if not panel_groups:
                raise ValueError("Requested show_panels produced no panels to display.")

            heights = [grp['height'] for grp in panel_groups]
            panel_count = len(show_panels) if show_panels is not None else len(panel_groups)
            fig_height = 1.5 * panel_count
            fig = plt.figure(figsize=(8*1.2*0.5*1.1, fig_height*1.3*1.1))
            outer_gs = gridspec.GridSpec(len(panel_groups), 1, height_ratios=heights, hspace=0.18)

            ax_order = []
            ax_map = {}
            shared_axis = None
            for idx, group in enumerate(panel_groups):
                main_info = group['main']
                residual_info = group['residual']

                if residual_info is not None:
                    sub_gs = gridspec.GridSpecFromSubplotSpec(
                        2,
                        1,
                        subplot_spec=outer_gs[idx],
                        height_ratios=[main_info['height'], residual_info['height']],
                        hspace=0.02,
                    )
                    main_spec = sub_gs[0]
                    residual_spec = sub_gs[1]
                else:
                    main_spec = outer_gs[idx]
                    residual_spec = None

                if shared_axis is None:
                    main_ax = fig.add_subplot(main_spec)
                    shared_axis = main_ax
                else:
                    main_ax = fig.add_subplot(main_spec, sharex=shared_axis)

                main_props = main_info['props']
                if 'ylabel' in main_props:
                    main_ax.set_ylabel(main_props['ylabel'], fontsize=14)
                if main_props.get('ylim') is not None:
                    main_ax.set_ylim(*main_props['ylim'])
                if main_props.get('yscale'):
                    main_ax.set_yscale(main_props['yscale'])
                main_ax.set_xlim(*phi1_range)

                ax_order.append(main_ax)
                ax_map[main_info['name']] = main_ax

                if residual_info is not None:
                    if shared_axis is None:
                        residual_ax = fig.add_subplot(residual_spec)
                    else:
                        residual_ax = fig.add_subplot(residual_spec, sharex=shared_axis)
                    res_props = residual_info['props']
                    if res_props.get('ylim') is not None:
                        residual_ax.set_ylim(*res_props['ylim'])
                    if res_props.get('yscale'):
                        residual_ax.set_yscale(res_props['yscale'])
                    ax_order.append(residual_ax)
                    ax_map[residual_info['name']] = residual_ax

            # ensure all axes share same x-limits
            for axis in ax_order:
                axis.set_xlim(*phi1_range)

            dist_ax = ax_map.get('dist')
            bottom_axis = dist_ax if dist_ax is not None else ax_order[-1]
            bottom_axis.set_xlabel(r'$\phi_1$ (deg)', fontsize=14)

            for label, handler, skip_msg in active_handlers:
                if handler is not None and handler.data is not None and len(handler.data) > 0:
                    self._plot_handler_panels(
                        ax_map,
                        handler,
                        label=label,
                        pad=pad,
                        highlight=highlight,
                        vgsr_idx=self.vgsr_idx,
                        pmra_idx=self.pmra_idx,
                        pmdec_idx=self.pmdec_idx,
                        phi1_range=phi1_range,
                        phi1_exclude=phi1_exclude,
                    )
                else:
                    print(skip_msg)
            for axis in ax_order:
                axis.grid(ls='-.', alpha=0.2, zorder=-10)
                axis.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                axis.spines['top'].set_linewidth(1)
                axis.spines['right'].set_linewidth(1)
                axis.spines['left'].set_linewidth(1)
                axis.spines['bottom'].set_linewidth(1)
                axis.minorticks_on()
                if axis is not bottom_axis:
                    axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                else:
                    axis.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

            legend_handles = {
                'MS+RGB': plt.Line2D([0], [0], marker='o', color='w', label='MS+RGB',
                                    markeredgecolor='none', markerfacecolor='0.4', markersize=6,
                                    linestyle='None', alpha=0.8),
                'BHB': plt.Line2D([0], [0], marker='^', color='w', label='BHB',
                                   markerfacecolor='none', markeredgecolor='blue', markersize=6,
                                   linestyle='None', alpha=1.0),
                'RRL': plt.Line2D([0], [0], marker='v', color='w', label='RRL',
                                   markerfacecolor='none', markeredgecolor='#ff0022', markersize=6,
                                   linestyle='None', alpha=1.0)
            }
            ax_order[0].legend(
                handles=legend_handles.values(),
                loc='lower left',
                # bbox_to_anchor=(0.12, 0.878),
                ncol=3,
                fontsize=9,
                frameon=True,
                handletextpad=0.5,
                labelspacing=0.5,
            )
            # fig.tight_layout(rect=[0, 0, 1, 0.97])

            return fig, ax_order

    def _plot_handler_panels(
        self,
        axes,
        handler,
        *,
        label,
        pad,
        highlight=None,
        vgsr_idx=1,
        pmra_idx=5,
        pmdec_idx=6,
        phi1_range=None,
        phi1_exclude=None,
    ):
        """
        Plot main and residual panels for a single handler selection.
        
        Parameters
        ----------
        phi1_exclude : tuple, optional
            (min, max) phi1 range to exclude from plotting. Stars with phi1
            values in this range will be masked out.
        """
        if handler is None or handler.data is None or len(handler.data) == 0:
            return

        pad = np.asarray(pad, dtype=float).reshape(-1)
        ax_phi2 = axes.get('phi2')
        ax_vgsr = axes.get('vgsr')
        ax_pmra = axes.get('pmra')
        ax_pmdec = axes.get('pmdec')
        ax_feh = axes.get('feh')
        ax_dist = axes.get('dist')
        ax_dvgsr = axes.get('dvgsr')
        ax_dpmra = axes.get('dpmra')
        ax_dpmdec = axes.get('dpmdec')

        data_length = len(handler.data)

        def _ensure_mask(mask):
            if mask is None:
                return np.ones(data_length, dtype=bool)
            arr = np.asarray(mask, dtype=bool).reshape(-1)
            if arr.shape[0] != data_length:
                return np.ones(data_length, dtype=bool)
            return arr

        mask_components = OrderedDict([
            ('VGSR', _ensure_mask(handler.touch_masks['VGSR'])),
            ('FEH', _ensure_mask(handler.touch_masks['FEH'])),
            ('PMRA', _ensure_mask(handler.touch_masks['PMRA'])),
            ('PMDEC', _ensure_mask(handler.touch_masks['PMDEC'])),
            ('phi2', _ensure_mask(handler.touch_masks['phi2'])),
            ('plx', _ensure_mask(handler.plx_mask)),
            ('phi1', _ensure_mask(handler.phi1_mask)),
        ])
        
        # Create phi1 exclusion mask if specified
        phi1_all_for_mask = _get_numeric_array(handler.data, 'phi1')
        if phi1_exclude is not None and phi1_all_for_mask is not None:
            phi1_min_ex, phi1_max_ex = phi1_exclude
            phi1_exclude_mask = ~((phi1_all_for_mask >= phi1_min_ex) & (phi1_all_for_mask <= phi1_max_ex))
        else:
            phi1_exclude_mask = np.ones(data_length, dtype=bool)

        iso_mask = np.ones(data_length, dtype=bool)
        if label == 'MS+RG':
            iso_mask = _ensure_mask(getattr(handler, 'iso_mask', None))

        def _combine(exclude=()):
            keys = [name for name in mask_components if name not in exclude]
            if not keys:
                return iso_mask.copy() & phi1_exclude_mask
            combined = mask_components[keys[0]].copy()
            for key in keys[1:]:
                combined &= mask_components[key]
            return combined & iso_mask & phi1_exclude_mask

        sel = _combine()
        sel_variants = {
            'phi2': _combine(('phi2',)),
            'vgsr': _combine(('VGSR',)),
            'pmra': _combine(('PMRA',)),
            'pmdec': _combine(('PMDEC',)),
            'feh': _combine(('FEH',)),
        }
        handler.set_variants = sel_variants
        handler.sel = sel

        phi1_all = _get_numeric_array(handler.data, 'phi1')
        phi2_all = _get_numeric_array(handler.data, 'phi2')
        vgsr_all = _get_numeric_array(handler.data, 'VGSR')
        pmra_all = _get_numeric_array(handler.data, 'PMRA')
        pmdec_all = _get_numeric_array(handler.data, 'PMDEC')
        feh_all = _get_numeric_array(handler.data, 'FEH')

        def _as_float(values):
            if values is None:
                return None
            arr = np.asarray(values, dtype=float).reshape(-1)
            return arr

        phi2_err = _as_float(_pick_numeric_column(handler.data, COLUMN_ALIASES.get('PHI2_ERROR', [])))
        vgsr_err = _as_float(_pick_numeric_column(handler.data, COLUMN_ALIASES['VGSR_ERR']))
        pmra_err = _as_float(_pick_numeric_column(handler.data, COLUMN_ALIASES['PMRA_ERROR'] + ['SIGMA_PMRA']))
        pmdec_err = _as_float(_pick_numeric_column(handler.data, COLUMN_ALIASES['PMDEC_ERROR'] + ['SIGMA_PMDEC']))
        feh_err = _as_float(_pick_numeric_column(handler.data, COLUMN_ALIASES['FEH_ERROR']))

        marker, color = _label_style(label)
        marker_size = 30
        marker_alpha = 0.5 if label == 'MS+RG' else 1.0
        background_alpha = marker_alpha  # Use same alpha for background and selected points
        facecolor_main = 'none' if label in ('BHB', 'RRL') else color
        edgecolor_main = color if label in ('BHB', 'RRL') else 'none'

        # Use phi1_range if provided, otherwise compute from spline points
        spline_points = handler.spline_points_dict['phi1_spline_points']
        spline_k = handler.spline_points_dict['spline_k']
        if phi1_range is not None:
            x_arr = np.arange(phi1_range[0], phi1_range[1], 0.1)
        else:
            # Fall back to spline points range with some padding
            x_arr = np.arange(np.min(spline_points) - 5, np.max(spline_points) + 5, 0.1)

        vgsr_track_dense = apply_spline(x_arr, spline_points, handler.nested_dict['meds'][vgsr_idx], spline_k)
        pmra_track_dense = apply_spline(x_arr, spline_points, handler.nested_dict['meds'][pmra_idx], spline_k)
        pmdec_track_dense = apply_spline(x_arr, spline_points, handler.nested_dict['meds'][pmdec_idx], spline_k)

        vgsr_track_data = apply_spline(phi1_all, spline_points, handler.nested_dict['meds'][vgsr_idx], spline_k)
        pmra_track_data = apply_spline(phi1_all, spline_points, handler.nested_dict['meds'][pmra_idx], spline_k)
        pmdec_track_data = apply_spline(phi1_all, spline_points, handler.nested_dict['meds'][pmdec_idx], spline_k)

        vgsr_residual = vgsr_all - vgsr_track_data
        pmra_residual = pmra_all - pmra_track_data
        pmdec_residual = pmdec_all - pmdec_track_data

        def _plot_with_errorbar(axis, mask, values, errors, *, alpha, zorder, label_enabled=False):
            if axis is None or values is None:
                return
            mask = np.asarray(mask, dtype=bool)
            if not np.any(mask):
                return
            value_arr = np.asarray(values, dtype=float)
            base_mask = mask & np.isfinite(phi1_all) & np.isfinite(value_arr)
            if not np.any(base_mask):
                return
            base_kwargs = {
                'fmt': marker,
                'ms': max(np.sqrt(marker_size), 1.0),
                'mfc': facecolor_main,
                'mec': edgecolor_main,
                'ecolor': color,
                'elinewidth': 0.75,
                'capsize': 2,
                'alpha': alpha,
                'zorder': zorder,
                'label': label if label_enabled else '_nolegend_',
            }
            if errors is None:
                axis.errorbar(
                    phi1_all[base_mask],
                    value_arr[base_mask],
                    yerr=None,
                    **base_kwargs,
                )
                return
            err_arr = np.asarray(errors, dtype=float)
            finite_err_mask = base_mask & np.isfinite(err_arr)
            no_err_mask = base_mask & ~np.isfinite(err_arr)
            if np.any(finite_err_mask):
                axis.errorbar(
                    phi1_all[finite_err_mask],
                    value_arr[finite_err_mask],
                    yerr=err_arr[finite_err_mask],
                    **base_kwargs,
                )
            if np.any(no_err_mask):
                axis.errorbar(
                    phi1_all[no_err_mask],
                    value_arr[no_err_mask],
                    yerr=None,
                    **base_kwargs,
                )

        if ax_phi2 is not None and phi2_all is not None:
            finite_phi_mask = np.isfinite(phi1_all) & np.isfinite(phi2_all)
            background_phi2 = sel_variants['phi2'] & finite_phi_mask & ~sel
            selected_phi2 = sel & finite_phi_mask
            _plot_with_errorbar(
                ax_phi2,
                background_phi2,
                phi2_all,
                phi2_err,
                alpha=background_alpha,
                zorder=BACKGROUND_SCATTER_ZORDER,
            )
            _plot_with_errorbar(
                ax_phi2,
                selected_phi2,
                phi2_all,
                phi2_err,
                alpha=marker_alpha,
                zorder=SELECTED_SCATTER_ZORDER,
            )
            # ax_phi2.axhline(0, color='g', lw=1, zorder=2)  # Removed central green line
            ax_phi2.axhline(pad[4], color='k', lw=0.5, ls='-.', zorder=0)
            ax_phi2.axhline(-pad[4], color='k', lw=0.5, ls='-.', zorder=0)

        series_configs = [
            {
                'name': 'VGSR',
                'axis': ax_vgsr,
                'res_axis': ax_dvgsr,
                'data': vgsr_all,
                'err': vgsr_err,
                'residual': vgsr_residual,
                'sel_mask': sel_variants['vgsr'],
                'track_dense': vgsr_track_dense,
                'pad_value': pad[0],
                'warn': 'VGSR',
            },
            {
                'name': 'PMRA',
                'axis': ax_pmra,
                'res_axis': ax_dpmra,
                'data': pmra_all,
                'err': pmra_err,
                'residual': pmra_residual,
                'sel_mask': sel_variants['pmra'],
                'track_dense': pmra_track_dense,
                'pad_value': pad[2],
                'warn': 'PMRA',
            },
            {
                'name': 'PMDEC',
                'axis': ax_pmdec,
                'res_axis': ax_dpmdec,
                'data': pmdec_all,
                'err': pmdec_err,
                'residual': pmdec_residual,
                'sel_mask': sel_variants['pmdec'],
                'track_dense': pmdec_track_dense,
                'pad_value': pad[3],
                'warn': 'PMDEC',
            },
        ]

        for series in series_configs:
            axis = series['axis']
            if axis is None or series['data'] is None:
                continue
            valid = np.isfinite(series['data'])
            background_mask = series['sel_mask'] & valid & ~sel
            selected_mask = sel & valid
            _plot_with_errorbar(
                axis,
                background_mask,
                series['data'],
                series['err'],
                alpha=background_alpha,
                zorder=BACKGROUND_SCATTER_ZORDER,
            )
            _plot_with_errorbar(
                axis,
                selected_mask,
                series['data'],
                series['err'],
                alpha=marker_alpha,
                zorder=SELECTED_SCATTER_ZORDER,
            )
            if series['track_dense'] is not None:
                # axis.plot(x_arr, series['track_dense'], color='g', lw=1, zorder=SPLINE_ZORDER)  # Removed central green line
                pad_val = series['pad_value']
                axis.plot(x_arr, series['track_dense'] + pad_val, color='k', lw=0.5, ls='-.', zorder=SPLINE_ZORDER)
                axis.plot(x_arr, series['track_dense'] - pad_val, color='k', lw=0.5, ls='-.', zorder=SPLINE_ZORDER)
            res_axis = series['res_axis']
            if res_axis is None or series['residual'] is None:
                continue
            finite_residual = np.isfinite(series['residual'])
            background_res_mask = background_mask & finite_residual
            selected_res_mask = selected_mask & finite_residual
            if series['err'] is None:
                print(f"Warning: no {series['warn']} uncertainty column found for {label}, skipping {series['warn']} residual error bars.")
            _plot_with_errorbar(
                res_axis,
                background_res_mask,
                series['residual'],
                series['err'],
                alpha=background_alpha,
                zorder=BACKGROUND_SCATTER_ZORDER,
            )
            _plot_with_errorbar(
                res_axis,
                selected_res_mask,
                series['residual'],
                series['err'],
                alpha=marker_alpha,
                zorder=SELECTED_SCATTER_ZORDER,
            )
            pad_val = series['pad_value']
            # res_axis.axhline(0, color='g', lw=1, zorder=2)  # Removed central green line
            res_axis.axhline(pad_val, color='k', lw=0.5, ls='-.', zorder=0)
            res_axis.axhline(-pad_val, color='k', lw=0.5, ls='-.', zorder=0)

        if ax_feh is not None and feh_all is not None:
            valid_feh = np.isfinite(feh_all)
            background_feh = sel_variants['feh'] & valid_feh & ~sel
            selected_feh = sel & valid_feh
            _plot_with_errorbar(
                ax_feh,
                background_feh,
                feh_all,
                feh_err,
                alpha=background_alpha,
                zorder=BACKGROUND_SCATTER_ZORDER,
            )
            _plot_with_errorbar(
                ax_feh,
                selected_feh,
                feh_all,
                feh_err,
                alpha=marker_alpha,
                zorder=SELECTED_SCATTER_ZORDER,
            )
            # ax_feh.axhline(handler.nested_dict['meds'][3], color='g', lw=1, zorder=0)  # Removed central green line
            ax_feh.axhline(pad[1], color='k', lw=0.5, ls='-.', zorder=0)

        dist_vals, dist_err = _extract_distance_series(handler.data, label)
        if ax_dist is not None and dist_vals is not None:
            dist_vals = np.asarray(dist_vals, dtype=float).reshape(-1)
            dist_err = _as_float(dist_err)
            finite_dist = np.isfinite(dist_vals) & np.isfinite(phi1_all)
            selected_dist = sel & finite_dist
            _plot_with_errorbar(
                ax_dist,
                selected_dist,
                dist_vals,
                dist_err,
                alpha=marker_alpha,
                zorder=SELECTED_SCATTER_ZORDER,
            )

        def _track_eval(idx, values):
            arr = np.atleast_1d(_ensure_numeric(values))
            if arr is None:
                return None
            return apply_spline(
                arr,
                handler.spline_points_dict['phi1_spline_points'],
                handler.nested_dict['meds'][idx],
                handler.spline_points_dict['spline_k'],
            )

        if highlight is not None:
            if ax_phi2 is not None and 'phi2' in highlight:
                ax_phi2.scatter(highlight['phi1'], highlight['phi2'], s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_vgsr is not None and 'VGSR' in highlight:
                ax_vgsr.scatter(highlight['phi1'], highlight['VGSR'], s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_dvgsr is not None and 'VGSR' in highlight:
                track_vals = _track_eval(vgsr_idx, highlight['phi1'])
                if track_vals is not None:
                    dvgsr = _ensure_numeric(highlight['VGSR']) - track_vals
                    ax_dvgsr.scatter(highlight['phi1'], dvgsr, s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_pmra is not None and 'PMRA' in highlight:
                ax_pmra.scatter(highlight['phi1'], highlight['PMRA'], s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_dpmra is not None and 'PMRA' in highlight:
                track_vals = _track_eval(pmra_idx, highlight['phi1'])
                if track_vals is not None:
                    dpmra = _ensure_numeric(highlight['PMRA']) - track_vals
                    ax_dpmra.scatter(highlight['phi1'], dpmra, s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_pmdec is not None and 'PMDEC' in highlight:
                ax_pmdec.scatter(highlight['phi1'], highlight['PMDEC'], s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_dpmdec is not None and 'PMDEC' in highlight:
                track_vals = _track_eval(pmdec_idx, highlight['phi1'])
                if track_vals is not None:
                    dpmdec = _ensure_numeric(highlight['PMDEC']) - track_vals
                    ax_dpmdec.scatter(highlight['phi1'], dpmdec, s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            if ax_feh is not None and 'FEH' in highlight:
                ax_feh.scatter(highlight['phi1'], highlight['FEH'], s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)
            highlight_dist = None
            if hasattr(highlight, 'get'):
                highlight_dist = highlight.get('dist_FEH')
                if highlight_dist is None:
                    highlight_dist = highlight.get('dist_kpc')
                if highlight_dist is None:
                    highlight_dist = highlight.get('distance')
                if highlight_dist is None:
                    highlight_dm = highlight.get('dist_mod')
                    if highlight_dm is not None:
                        highlight_dist, _ = _dist_from_dm(highlight_dm)
            else:
                if 'dist_FEH' in highlight:
                    highlight_dist = highlight['dist_FEH']
                elif 'dist_kpc' in highlight:
                    highlight_dist = highlight['dist_kpc']
                elif 'distance' in highlight:
                    highlight_dist = highlight['distance']
                elif 'dist_mod' in highlight:
                    highlight_dist, _ = _dist_from_dm(highlight['dist_mod'])
            if highlight_dist is not None and ax_dist is not None:
                highlight_dist = np.atleast_1d(_ensure_numeric(highlight_dist))
                ax_dist.scatter(highlight['phi1'], highlight_dist, s=80, c='m', alpha=0.5, label='highlight', zorder=SELECTED_SCATTER_ZORDER + 1)

    def do_orbit(self, progenitor_RA, theta_init=None, with_plot=True, use_mcmc=True, plot_chains=False, plot_corner=False, **kwargs):
        """
        Run the orbit for the stream and plot the results.
        """
        # if orbit_kwargs is None:
        #     orbit_kwargs = {
        #         'fw' : np.linspace(0., 0.2, 2001) * u.Gyr,
        #         'bw' : np.linspace(0, -0.2, 2001) * u.Gyr,
        #         'progenitor_distance': self.min_dist/1000, # in kpc
        #     }
        orbit_kwargs = {
            'fw' :  kwargs.get('fw',  np.linspace(0., 0.2, 2001) * u.Gyr),
            'bw' : kwargs.get('bw',  np.linspace(0., 0.2, 2001) * u.Gyr),
            'progenitor_distance': self.min_dist/1000, # in kpc
        }

        if theta_init is None:
            if orbit_kwargs is None:
                progenitor_dist = self.min_dist # in kpc
            else:
                progenitor_dist = orbit_kwargs['progenitor_distance']  # in kpc
            self.stream_data['RA'] = np.array(self.stream_data['TARGET_RA'])
            self.stream_data['DEC'] = np.array(self.stream_data['TARGET_DEC'])
            progenitor_RA = np.mean(self.stream_data['RA'])
            distances = np.abs(self.stream_data['RA'] - progenitor_RA) #finding the stars closest to the progenitor RA

            k = np.argsort(distances)
            guess = {
                    'DEC': np.nanmean(self.stream_data[k]['DEC'][:10]),
                    'PMRA': np.nanmean(self.stream_data[k]['PMRA'][:10]),
                    'PMDEC': np.nanmean(self.stream_data[k]['PMDEC'][:10]),
                    'VRAD': np.nanmean(self.stream_data[k]['VRAD'][:10]),
                    'DIST': progenitor_dist
                }
            lsig_dec_init = np.log10(0.1)  # Example: log10(0.1 deg)
            lsig_pmra_init = np.log10(0.1) # Example: log10(0.1 mas/yr)
            lsig_pmdec_init = np.log10(0.1)# Example: log10(0.1 mas/yr)
            lsig_vrad_init = np.log10(1.0) # Example: log10(1 km/s)

            theta_init = [
                guess['DEC'],
                guess['PMRA'],
                guess['PMDEC'],
                guess['VRAD'],
                guess['DIST'],
                lsig_dec_init,
                lsig_pmra_init,
                lsig_pmdec_init,
                lsig_vrad_init
            ]
        
        self.fw = orbit_kwargs['fw']
        self.bw = orbit_kwargs['bw']

        fit_kwargs = {
            'nwalkers': kwargs.get('nwalkers', 50),
            'nsteps': kwargs.get('nsteps', 1100),
            'nburn': kwargs.get('nburn', 100),
            'pre_optimize': kwargs.get('pre_optimize', True),
            'make_corner': use_mcmc and plot_corner,
        }
        if 'burnin' in kwargs and kwargs['burnin'] is not None:
            fit_kwargs['burnin'] = kwargs['burnin']
        for opt_key in ('progress', 'seed', 'walker_spread'):
            if kwargs.get(opt_key) is not None:
                fit_kwargs[opt_key] = kwargs[opt_key]
        if plot_corner and kwargs.get('corner_kwargs') is not None:
            fit_kwargs['corner_kwargs'] = kwargs['corner_kwargs']
        if plot_corner and kwargs.get('corner_thin') is not None:
            fit_kwargs['corner_thin'] = kwargs['corner_thin']

        results_o, orbit = ofuncs.fit_orbit(
            self.stream_data,
            self.frame,
            progenitor_RA,
            orbit_kwargs["fw"],
            orbit_kwargs["bw"],
            theta_init,
            use_position=True,
            use_mcmc=use_mcmc,
            **fit_kwargs
        )
        self.orbit_ran = True
        fig = None
        ax = None
        if with_plot:
            o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_dist = ofuncs.orbit_model(
                results_o.x[0:5],
                progenitor_RA,
                orbit_kwargs["fw"],
                orbit_kwargs["bw"],
                return_o=False,
            )
            fig, ax = ofuncs.plot_orbit(
                o_ra,
                o_dec,
                self.stream_data['RA'],
                self.stream_data['DEC'],
                progenitor_RA,
                model_dist=o_dist,
                cmap='brg',
                add_colorbar=True,
            )
        self.results_o = results_o
        if use_mcmc and plot_chains:
            chain_fig, chain_axes = ofuncs.plot_mcmc_traces(
                results_o,
                labels=results_o.param_labels,
                burnin=results_o.burnin,
            )
            results_o.chain_fig = chain_fig
            results_o.chain_axes = chain_axes
        elif getattr(results_o, 'chain_fig', None) is None:
            results_o.chain_fig = None
            results_o.chain_axes = None
        if not plot_corner:
            results_o.corner = getattr(results_o, 'corner', None)
            results_o.corner_axes = getattr(results_o, 'corner_axes', None)
        return results_o, orbit, fig, ax
    
    def add_orbit_track(self, ax, results_o, track='', residual=False):
        """
        Add the orbit track to the given axis.
        """
        o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_dist = ofuncs.orbit_model(
            results_o.x[0:5],
            np.mean(self.stream_data['RA']),
            self.fw,
            self.bw,
            return_o=False,
        )
        o_phi1, o_phi2 = ofuncs.ra_dec_to_phi1_phi2(self.frame, o_ra*u.deg, o_dec*u.deg)
        o_vgsr = np.array(ofuncs.vhel_to_vgsr(o_ra, o_dec, o_vrad).value)

        importlib.reload(ofuncs)
        ointerps = ofuncs.orbit_interpolations([o_phi1, o_phi2, o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_vgsr, o_dist])

        orbit_phi1 = np.linspace(np.min(self.stream_data['phi1'] - 7), np.max(self.stream_data['phi1'] + 7), 1000)
        y_vals = ointerps[track](orbit_phi1)
        if residual:
            ref_track = self._vis6_reference_track(orbit_phi1, track)
            if ref_track is not None:
                y_vals = y_vals - ref_track
        ax.plot(orbit_phi1, y_vals, color='red', label='', zorder=0, lw=1.3)

    def add_spline_track(self, ax, med_ind=0, color='blue', label='', **kwargs):
        """Add the spline track to the given axis. made to work with vis_6_panel"""
        importlib.reload(stream_funcs)

        phi1_span = np.linspace(
            np.min(self.stream_data['phi1'] - 5),
            np.max(self.stream_data['phi1'] + 5),
            1000
        )

        labels = self.nested_dict.get('expanded_param_labels', [])
        param_label = labels[med_ind] if med_ind < len(labels) else None

        def _resolve_knots(label):
            base_knots = np.asarray(self.spline_points_dict['phi1_spline_points'], dtype=float)
            base_k = self.spline_points_dict['spline_k']
            if label and label.startswith('pstream'):
                knots = np.asarray(
                    self.spline_points_dict.get('pstream_phi1_spline_points', base_knots),
                    dtype=float
                )
                order = self.spline_points_dict.get('spline_k_pstream', base_k)
            elif label and label.startswith('lsigvgsr'):
                knots = np.asarray(
                    self.spline_points_dict.get('lsigv_phi1_spline_points', base_knots),
                    dtype=float
                )
                order = self.spline_points_dict.get('spline_k_lsigv', base_k)
            else:
                knots = base_knots
                order = base_k
            return knots, order

        pstream_bounds = self.spline_points_dict.get('pstream_bounds', (1e-6, 1 - 1e-6))

        def _evaluate(values, knots, order, label):
            arr = np.asarray(values, dtype=float)
            if label and label.startswith('pstream'):
                return stream_funcs.evaluate_probability_spline(
                    phi1_span,
                    knots,
                    arr,
                    k=order,
                    bounds=pstream_bounds
                )
            if arr.size <= 1:
                val = float(arr.reshape(-1)[0])
                return np.full_like(phi1_span, val, dtype=float)
            if knots.size != arr.size:
                if knots.size > 0:
                    knots = np.linspace(knots[0], knots[-1], arr.size)
                else:
                    knots = np.linspace(phi1_span.min(), phi1_span.max(), arr.size)
            eff_order = max(1, min(int(order), arr.size - 1))
            return stream_funcs.apply_spline(phi1_span, knots, arr, k=eff_order)

        knots, order = _resolve_knots(param_label)
        raw_curve = _evaluate(self.nested_dict['meds'][med_ind], knots, order, param_label)

        def _transform(label, values):
            if label and label.startswith('lsig'):
                return 10**values
            return values

        display_curve = _transform(param_label, raw_curve)

        ax.plot(phi1_span, display_curve, color=color, ls='-.', label=label, zorder=0)

        nested_list_exp_meds = self.nested_dict['exp_meds'][med_ind]
        nested_list_exp_ep = self.nested_dict['exp_ep'][med_ind]
        nested_list_exp_em = self.nested_dict['exp_em'][med_ind]
        ax.errorbar(
            self.spline_points_dict['phi1_spline_points'],
            np.array(nested_list_exp_meds),
            yerr=(nested_list_exp_em, nested_list_exp_ep),
            capsize=3, elinewidth=0.75, ecolor=color, fmt='none', zorder=0
        )

    def return_spline_track(self, med_ind=0):
        """Return a spline track in linear units for flexible plotting."""
        importlib.reload(stream_funcs)

        phi1_span = np.linspace(
            np.min(self.stream_data['phi1'] - 5),
            np.max(self.stream_data['phi1'] + 5),
            1000
        )

        labels = self.nested_dict.get('expanded_param_labels', [])
        param_label = labels[med_ind] if med_ind < len(labels) else None

        def _resolve_knots(label):
            base_knots = np.asarray(self.spline_points_dict['phi1_spline_points'], dtype=float)
            base_k = self.spline_points_dict['spline_k']
            if label and label.startswith('pstream'):
                knots = np.asarray(
                    self.spline_points_dict.get('pstream_phi1_spline_points', base_knots),
                    dtype=float
                )
                order = self.spline_points_dict.get('spline_k_pstream', base_k)
            elif label and label.startswith('lsigvgsr'):
                knots = np.asarray(
                    self.spline_points_dict.get('lsigv_phi1_spline_points', base_knots),
                    dtype=float
                )
                order = self.spline_points_dict.get('spline_k_lsigv', base_k)
            else:
                knots = base_knots
                order = base_k
            return knots, order

        pstream_bounds = self.spline_points_dict.get('pstream_bounds', (1e-6, 1 - 1e-6))

        def _evaluate(values, knots, order, label):
            arr = np.asarray(values, dtype=float)
            if label and label.startswith('pstream'):
                return stream_funcs.evaluate_probability_spline(
                    phi1_span,
                    knots,
                    arr,
                    k=order,
                    bounds=pstream_bounds
                )
            if arr.size <= 1:
                val = float(arr.reshape(-1)[0])
                return np.full_like(phi1_span, val, dtype=float)
            if knots.size != arr.size:
                if knots.size > 0:
                    knots = np.linspace(knots[0], knots[-1], arr.size)
                else:
                    knots = np.linspace(phi1_span.min(), phi1_span.max(), arr.size)
            eff_order = max(1, min(int(order), arr.size - 1))
            return stream_funcs.apply_spline(phi1_span, knots, arr, k=eff_order)

        knots, order = _resolve_knots(param_label)
        raw_curve = _evaluate(self.nested_dict['meds'][med_ind], knots, order, param_label)

        def _transform(label, values):
            if label and label.startswith('lsig'):
                return 10**values
            return values

        display_curve = _transform(param_label, raw_curve)

        nested_list_exp_meds = self.nested_dict['exp_meds'][med_ind]
        nested_list_exp_ep = self.nested_dict['exp_ep'][med_ind]
        nested_list_exp_em = self.nested_dict['exp_em'][med_ind]

        return {
            'phi1': phi1_span,
            'spline': display_curve,
            'phi1_knots': knots,
            'lsigv_phi1_spline_points': self.spline_points_dict.get('lsigv_phi1_spline_points'),
            'meds': nested_list_exp_meds,
            'ep': nested_list_exp_ep,
            'em': nested_list_exp_em
        }
    

    def vis_6_panel(self, addBackground=True, save_fig=False, fig_path=None, dist_mod_panel=False, residual=False, **kwargs):
        """
        Visualize the 6 panel plot for the stream data.
        Set residual=True (or show_residuals=True) to display Δ values relative to the spline track
        for the kinematic panels.
        """
        show_residuals = bool(kwargs.get('show_residuals', residual))
        highlight = kwargs.get('highlight', None)
        pad = kwargs.get('pad', None)  # Default padding values
        # Dispersion overlays
        show_disp = bool(kwargs.get('show_dispersion', False))
        show_vgsr_disp = bool(kwargs.get('show_vgsr_dispersion', show_disp))
        show_pm_disp = bool(kwargs.get('show_pm_dispersion', show_disp))
        show_feh_disp = bool(kwargs.get('show_feh_dispersion', show_disp))
        nsig = float(kwargs.get('nsigma_dispersion', 1.0))
        self._vis6_context = {
            'show_residuals': show_residuals
        }

        x_arr = np.linspace(self.spline_points_dict['phi1_spline_points'][0], self.spline_points_dict['phi1_spline_points'][-1], 200)
        stream_data = self.stream_data
        phi1_vals = np.asarray(stream_data['phi1'], dtype=float)
        phi1_knots = np.asarray(self.spline_points_dict['phi1_spline_points'], dtype=float)

        def _track_at(phi1_values, med_idx):
            return apply_spline(
                phi1_values,
                phi1_knots,
                self.nested_dict['meds'][med_idx],
                self.spline_points_dict['spline_k']
            )

        vgsr_track = _track_at(x_arr, self.vgsr_idx)
        pmra_track = _track_at(x_arr, self.pmra_idx)
        pmdec_track = _track_at(x_arr, self.pmdec_idx)
        feh_track = apply_spline(
            x_arr,
            phi1_knots,
            self.nested_dict['meds'][3],
            self.spline_points_dict['spline_k']
        )
        vgsr_track_data = _track_at(phi1_vals, self.vgsr_idx)
        pmra_track_data = _track_at(phi1_vals, self.pmra_idx)
        pmdec_track_data = _track_at(phi1_vals, self.pmdec_idx)

        def _residualize(values, track_values):
            arr = np.asarray(values, dtype=float)
            track_arr = np.asarray(track_values, dtype=float)
            return arr - track_arr if show_residuals else arr

        vgsr_plot_vals = _residualize(stream_data['VGSR'], vgsr_track_data)
        pmra_plot_vals = _residualize(stream_data['PMRA'], pmra_track_data)
        pmdec_plot_vals = _residualize(stream_data['PMDEC'], pmdec_track_data)

        def _display_track(track_values):
            return np.zeros_like(track_values, dtype=float) if show_residuals else track_values

        vgsr_display_track = _display_track(vgsr_track)
        pmra_display_track = _display_track(pmra_track)
        pmdec_display_track = _display_track(pmdec_track)
        
        # Decide panel layout: include distance-modulus panel or not
        requested_dm = bool(dist_mod_panel)
        # Support both Astropy Table (colnames) and pandas DataFrame (columns)
        has_dm = (
            (hasattr(stream_data, 'colnames') and ('dist_mod' in stream_data.colnames)) or
            (hasattr(stream_data, 'columns') and ('dist_mod' in getattr(stream_data, 'columns')))
        )
        include_dm_panel = requested_dm and has_dm
        if requested_dm and not has_dm:
            print('dist_mod not found in stream_data; skipping DM panel.')
        n_pan = 6 if include_dm_panel else 5
        fig, ax = plt.subplots(n_pan, 1, figsize=(15, 2.5 * n_pan), sharex=True)
        # axis indices (place distance panel last if included)
        phi2_ax_i, vgsr_ax_i, pmra_ax_i, pmdec_ax_i = 0, 1, 2, 3
        if include_dm_panel:
            feh_ax_i = 4
            dm_ax_i = 5
        else:
            feh_ax_i = 4
            dm_ax_i = None
        cmap = 'magma_r'
        from matplotlib.colors import PowerNorm
        min_prob = float(kwargs.get('min_prob', 0.5))
        max_prob = float(kwargs.get('max_prob', 1.0))
        norm = PowerNorm(gamma=5, vmin=min_prob, vmax=max_prob)
        cm = ax[phi2_ax_i].scatter(
            phi1_vals, stream_data['phi2'],
            s=45, edgecolors='k', linewidth=0.75,
            cmap=cmap, norm=norm, c=stream_data['stream_prob'], alpha=1, zorder=1
        )

        # Precompute model tracks and optional dispersion bands before plotting data points
        if show_vgsr_disp:
            groups = self.nested_dict.get('param_labels', [])
            if 'lsigvgsr' in groups:
                lsig_idx = groups.index('lsigvgsr')
                lsig_vals = np.asarray(self.nested_dict['meds'][lsig_idx], dtype=float)
                if lsig_vals.size <= 1:
                    sigma_v = np.full_like(x_arr, 10**float(lsig_vals.reshape(-1)[0]))
                else:
                    knots = np.asarray(self.spline_points_dict.get('lsigv_phi1_spline_points', self.spline_points_dict['phi1_spline_points']), dtype=float)
                    k = int(self.spline_points_dict.get('spline_k_lsigv', self.spline_points_dict['spline_k']))
                    sigma_v = 10**apply_spline(x_arr, knots, lsig_vals, k)
                # 2-sigma (black, alpha=0.1), 1-sigma (black, alpha=0.5)
                ax[vgsr_ax_i].fill_between(
                    x_arr,
                    vgsr_display_track - 2 * nsig * sigma_v,
                    vgsr_display_track + 2 * nsig * sigma_v,
                    color='steelblue', alpha=0.15, zorder=0
                )
                ax[vgsr_ax_i].fill_between(
                    x_arr,
                    vgsr_display_track - 1 * nsig * sigma_v,
                    vgsr_display_track + 1 * nsig * sigma_v,
                    color='steelblue', alpha=0.20, zorder=1
                )

        if show_pm_disp:
            lsig_pm_fixed_log10 = self.nested_dict.get('lsig_pm_fixed_log10', np.log10(0.09))
            if lsig_pm_fixed_log10 is not None and not (isinstance(lsig_pm_fixed_log10, float) and np.isnan(lsig_pm_fixed_log10)):
                sigma_pmra = np.full_like(x_arr, 10**float(lsig_pm_fixed_log10))
                sigma_pmdec = np.full_like(x_arr, 10**float(lsig_pm_fixed_log10))
            else:
                groups = self.nested_dict.get('param_labels', [])
                if 'lsigpmra' in groups:
                    sigma_pmra = np.full_like(x_arr, 10**float(np.asarray(self.nested_dict['meds'][groups.index('lsigpmra')]).reshape(-1)[0]))
                else:
                    sigma_pmra = np.zeros_like(x_arr)
                if 'lsigpmdec' in groups:
                    sigma_pmdec = np.full_like(x_arr, 10**float(np.asarray(self.nested_dict['meds'][groups.index('lsigpmdec')]).reshape(-1)[0]))
                else:
                    sigma_pmdec = np.zeros_like(x_arr)
            if np.any(sigma_pmra > 0):
                ax[pmra_ax_i].fill_between(
                    x_arr,
                    pmra_display_track - 2 * nsig * sigma_pmra,
                    pmra_display_track + 2 * nsig * sigma_pmra,
                    color='steelblue', alpha=0.15, zorder=0
                )
                ax[pmra_ax_i].fill_between(
                    x_arr,
                    pmra_display_track - 1 * nsig * sigma_pmra,
                    pmra_display_track + 1 * nsig * sigma_pmra,
                    color='steelblue', alpha=0.20, zorder=1
                )
            if np.any(sigma_pmdec > 0):
                ax[pmdec_ax_i].fill_between(
                    x_arr,
                    pmdec_display_track - 2 * nsig * sigma_pmdec,
                    pmdec_display_track + 2 * nsig * sigma_pmdec,
                    color='steelblue', alpha=0.15, zorder=0
                )
                ax[pmdec_ax_i].fill_between(
                    x_arr,
                    pmdec_display_track - 1 * nsig * sigma_pmdec,
                    pmdec_display_track + 1 * nsig * sigma_pmdec,
                    color='steelblue', alpha=0.20, zorder=1
                )
        if show_feh_disp:
            groups = self.nested_dict.get('param_labels', [])
            sigma_feh = None
            if 'lsigfeh' in groups:
                lsig_idx = groups.index('lsigfeh')
                lsig_vals = np.asarray(self.nested_dict['meds'][lsig_idx], dtype=float)
                if lsig_vals.size <= 1:
                    sigma_feh = np.full_like(x_arr, 10**float(lsig_vals.reshape(-1)[0]))
                else:
                    knots = np.asarray(self.spline_points_dict.get('lsigfeh_phi1_spline_points', phi1_knots), dtype=float)
                    k = int(self.spline_points_dict.get('spline_k_lsigfeh', self.spline_points_dict['spline_k']))
                    sigma_feh = 10**apply_spline(x_arr, knots, lsig_vals, k)
            if sigma_feh is not None and np.any(sigma_feh > 0):
                ax[feh_ax_i].fill_between(
                    x_arr,
                    feh_track - 2 * nsig * sigma_feh,
                    feh_track + 2 * nsig * sigma_feh,
                    color='steelblue', alpha=0.15, zorder=0
                )
                ax[feh_ax_i].fill_between(
                    x_arr,
                    feh_track - 1 * nsig * sigma_feh,
                    feh_track + 1 * nsig * sigma_feh,
                    color='steelblue', alpha=0.20, zorder=1
                )

        # Now plot data points on top
        ax[vgsr_ax_i].scatter(
            phi1_vals, vgsr_plot_vals,
            s=45, edgecolors='k', linewidth=0.75,
            cmap=cmap, norm=norm, c=stream_data['stream_prob'], alpha=1, zorder=1
        )
        ax[pmra_ax_i].scatter(
            phi1_vals, pmra_plot_vals,
            s=45, edgecolors='k', linewidth=0.75,
            cmap=cmap, norm=norm, c=stream_data['stream_prob'], alpha=1, zorder=1
        )
        ax[pmdec_ax_i].scatter(
            phi1_vals, pmdec_plot_vals,
            s=45, edgecolors='k', linewidth=0.75,
            cmap=cmap, norm=norm, c=stream_data['stream_prob'], alpha=1, zorder=1
        )
        if include_dm_panel and dm_ax_i is not None:
            # Distance (kpc) panel derived from distance modulus, last panel
            dm_vals = np.asarray(stream_data['dist_mod'], dtype=float)
            dist_kpc = (10**((dm_vals + 5.0) / 5.0)) / 1000.0
            ax[dm_ax_i].scatter(
                phi1_vals, dist_kpc,
                s=45, edgecolors='k', linewidth=0.75,
                cmap=cmap, norm=norm, c=stream_data['stream_prob'], alpha=1, zorder=1
            )
        ax[feh_ax_i].scatter(
            phi1_vals, stream_data['FEH'],
            s=45, edgecolors='k', linewidth=0.75,
            cmap=cmap, norm=norm, c=stream_data['stream_prob'], alpha=1, zorder=1
        )
        if highlight is not None:
            h_phi1 = np.asarray(highlight['phi1'], dtype=float)
            ax[phi2_ax_i].scatter(h_phi1, highlight['phi2'], s=80, c='m', alpha=0.5, label='highlight')
            ax[vgsr_ax_i].scatter(
                h_phi1,
                _residualize(highlight['VGSR'], _track_at(h_phi1, self.vgsr_idx)),
                s=80, c='m', alpha=0.5, label='highlight'
            )
            ax[pmra_ax_i].scatter(
                h_phi1,
                _residualize(highlight['PMRA'], _track_at(h_phi1, self.pmra_idx)),
                s=80, c='m', alpha=0.5, label='highlight'
            )
            ax[pmdec_ax_i].scatter(
                h_phi1,
                _residualize(highlight['PMDEC'], _track_at(h_phi1, self.pmdec_idx)),
                s=80, c='m', alpha=0.5, label='highlight'
            )
            if include_dm_panel and dm_ax_i is not None and 'dist_mod' in highlight:
                dm_h = np.asarray(highlight['dist_mod'], dtype=float)
                dist_h_kpc = (10**((dm_h + 5.0) / 5.0)) / 1000.0
                ax[dm_ax_i].scatter(h_phi1, dist_h_kpc, s=80, c='m', alpha=0.5, label='highlight')
            ax[feh_ax_i].scatter(h_phi1, highlight['FEH'], s=80, c='m', alpha=0.5, label='highlight')

        if pad is not None:
            ax[phi2_ax_i].axhline(0, color='g', lw=1, zorder=2)
            ax[phi2_ax_i].axhline(pad[4], color='k', lw=0.5, ls='-.', zorder=0)
            ax[phi2_ax_i].axhline(-pad[4], color='k', lw=0.5, ls='-.', zorder=0)

            ax[vgsr_ax_i].plot(x_arr, vgsr_display_track, color='g', lw=1, zorder=0)
            ax[vgsr_ax_i].plot(x_arr, vgsr_display_track + pad[0], color='k', lw=0.5, ls='-.', zorder=0)
            ax[vgsr_ax_i].plot(x_arr, vgsr_display_track - pad[0], color='k', lw=0.5, ls='-.', zorder=0)

            ax[pmra_ax_i].plot(x_arr, pmra_display_track, color='g', lw=1, zorder=0)
            ax[pmra_ax_i].plot(x_arr, pmra_display_track + pad[2], color='k', lw=0.5, ls='-.', zorder=0)
            ax[pmra_ax_i].plot(x_arr, pmra_display_track - pad[2], color='k', lw=0.5, ls='-.', zorder=0)

            ax[pmdec_ax_i].plot(x_arr, pmdec_display_track, color='g', lw=1, zorder=0)
            ax[pmdec_ax_i].plot(x_arr, pmdec_display_track + pad[3], color='k', lw=0.5, ls='-.', zorder=0)
            ax[pmdec_ax_i].plot(x_arr, pmdec_display_track - pad[3], color='k', lw=0.5, ls='-.', zorder=0)

            ax[feh_ax_i].axhline(self.nested_dict['meds'][3], color='g', lw=1, zorder=0)
            ax[feh_ax_i].axhline(pad[1], color='k', lw=0.5, ls='-.', zorder=0)

        ax[vgsr_ax_i].errorbar(
            phi1_vals, vgsr_plot_vals,
            yerr=stream_data['VRAD_ERR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        ax[pmra_ax_i].errorbar(
            phi1_vals, pmra_plot_vals,
            yerr=stream_data['PMRA_ERROR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        ax[pmdec_ax_i].errorbar(
            phi1_vals, pmdec_plot_vals,
            yerr=stream_data['PMDEC_ERROR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        if include_dm_panel and dm_ax_i is not None:
            # Convert symmetric dist_mod errors into asymmetric distance errors and show
            try:
                cols = list(getattr(stream_data, 'columns', [])) or list(getattr(stream_data, 'colnames', []))
                dm = np.asarray(stream_data['dist_mod'], dtype=float)
                dist = (10**((dm + 5.0) / 5.0)) / 1000.0
                if 'dist_mod_err_plus' in cols and 'dist_mod_err_minus' in cols:
                    dm_hi = np.asarray(stream_data['dist_mod_err_plus'], dtype=float)
                    dm_lo = np.asarray(stream_data['dist_mod_err_minus'], dtype=float)
                elif 'dist_mod_err' in cols:
                    dm_hi = dm_lo = np.asarray(stream_data['dist_mod_err'], dtype=float)
                else:
                    dm_hi = dm_lo = None
                if dm_hi is not None:
                    d_hi = (10**(((dm + dm_hi) + 5.0) / 5.0)) / 1000.0 - dist
                    d_lo = dist - (10**(((dm - dm_lo) + 5.0) / 5.0)) / 1000.0
                    ax[dm_ax_i].errorbar(
                        phi1_vals, dist,
                        yerr=(d_lo, d_hi),
                        capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
                    )
            except Exception:
                pass
        ax[feh_ax_i].errorbar(
            stream_data['phi1'], stream_data['FEH'],
            yerr=stream_data['FEH_ERR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        ax[phi2_ax_i].set_ylabel(r'$\phi_2$ (deg)', fontsize=14)
        vgsr_ylabel = r'$\Delta V_{GSR}$ (km/s)' if show_residuals else r'$V_{GSR}$ (km/s)'
        pmra_ylabel = r'$\Delta \mu_{\alpha}$ (mas/yr)' if show_residuals else r'$\mu_{\alpha}$ (mas/yr)'
        pmdec_ylabel = r'$\Delta \mu_{\delta}$ (mas/yr)' if show_residuals else r'$\mu_{\delta}$ (mas/yr)'
        ax[vgsr_ax_i].set_ylabel(vgsr_ylabel, fontsize=14)
        ax[pmra_ax_i].set_ylabel(pmra_ylabel, fontsize=14)
        ax[pmdec_ax_i].set_ylabel(pmdec_ylabel, fontsize=14)
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].set_ylabel('Distance (kpc)', fontsize=14)
        ax[feh_ax_i].set_ylabel(r'[Fe/H]', fontsize=14)
        # Place phi1 label under distance panel if present; otherwise under FeH
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].set_xlabel(r'$\phi_1$ (deg)', fontsize=14)
        else:
            ax[feh_ax_i].set_xlabel(r'$\phi_1$ (deg)', fontsize=14)

        if addBackground:
            bg_phi1 = np.asarray(self.all_memberships['phi1'], dtype=float)
            ax[phi2_ax_i].scatter(bg_phi1, self.all_memberships['phi2'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            ax[vgsr_ax_i].scatter(
                bg_phi1,
                _residualize(self.all_memberships['VGSR'], _track_at(bg_phi1, self.vgsr_idx)),
                s=10, color='k', edgecolors='none', alpha=0.05, zorder=0
            )
            ax[pmra_ax_i].scatter(
                bg_phi1,
                _residualize(self.all_memberships['PMRA'], _track_at(bg_phi1, self.pmra_idx)),
                s=10, color='k', edgecolors='none', alpha=0.05, zorder=0
            )
            ax[pmdec_ax_i].scatter(
                bg_phi1,
                _residualize(self.all_memberships['PMDEC'], _track_at(bg_phi1, self.pmdec_idx)),
                s=10, color='k', edgecolors='none', alpha=0.05, zorder=0
            )
            if include_dm_panel and dm_ax_i is not None:
                if 'dist_mod' in self.all_memberships.columns:
                    dm_bg = np.asarray(self.all_memberships['dist_mod'], dtype=float)
                    dist_bg_kpc = (10**((dm_bg + 5.0) / 5.0)) / 1000.0
                    ax[dm_ax_i].scatter(bg_phi1, dist_bg_kpc, s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            ax[feh_ax_i].scatter(bg_phi1, self.all_memberships['FEH'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)

        # xlim based on the phi1 values
        ax[phi2_ax_i].set_xlim(np.min(phi1_vals) - 2, np.max(phi1_vals) + 2)

        # set ylims based on stream data y values
        ax[phi2_ax_i].set_ylim(np.min(stream_data['phi2']) - 2, np.max(stream_data['phi2']) + 2)

        def _set_axis_limits(axis, values, pad):
            data = np.asarray(values, dtype=float)
            data = data[np.isfinite(data)]
            if data.size == 0:
                return
            axis.set_ylim(data.min() - pad, data.max() + pad)

        _set_axis_limits(ax[vgsr_ax_i], vgsr_plot_vals, 10)
        _set_axis_limits(ax[pmra_ax_i], pmra_plot_vals, 1)
        _set_axis_limits(ax[pmdec_ax_i], pmdec_plot_vals, 1)
        if include_dm_panel and dm_ax_i is not None:
            try:
                dm_vals = np.asarray(stream_data['dist_mod'], dtype=float)
                dist_vals = (10**((dm_vals + 5.0) / 5.0)) / 1000.0
                finite = np.isfinite(dist_vals) & (dist_vals > 0)
                if np.any(finite):
                    vmin = np.nanmin(dist_vals[finite])
                    vmax = np.nanmax(dist_vals[finite])
                    ax[dm_ax_i].set_yscale('log')
                    ax[dm_ax_i].set_ylim(vmin * 0.85, vmax * 1.15)
                ax[dm_ax_i].set_ylabel('Distance (kpc)', fontsize=14)
            except Exception:
                pass

        if hasattr(self, 'results_o'):
            self.add_orbit_track(ax[phi2_ax_i], self.results_o, track='phi2')
            self.add_orbit_track(ax[vgsr_ax_i], self.results_o, track='vgsr', residual=show_residuals)
            self.add_orbit_track(ax[pmra_ax_i], self.results_o, track='pmra', residual=show_residuals)
            self.add_orbit_track(ax[pmdec_ax_i], self.results_o, track='pmdec', residual=show_residuals)
            if include_dm_panel and dm_ax_i is not None:
                self.add_orbit_track(ax[dm_ax_i], self.results_o, track='dist')
            # For distance modulus panel we skip orbit distance overlay (units mismatch). If you prefer, we can convert dist->dm.
        else:
            print('No orbit results found, skipping orbit track plotting')
        
        spline_lw = 1.8
        ax[vgsr_ax_i].plot(x_arr, vgsr_display_track, color='b', lw=spline_lw, zorder=0, ls='-')
        xsp = phi1_knots
        y_v = np.asarray(self.nested_dict['exp_meds'][1])
        ym_v = np.asarray(self.nested_dict['exp_em'][1])
        yp_v = np.asarray(self.nested_dict['exp_ep'][1])
        if y_v.ndim == 0: y_v = y_v[None]
        if ym_v.ndim == 0: ym_v = ym_v[None]
        if yp_v.ndim == 0: yp_v = yp_v[None]
        nv = int(min(len(xsp), len(y_v), len(ym_v), len(yp_v)))
        if nv > 0:
            y_v_display = y_v[:nv]
            if show_residuals:
                y_v_display = y_v_display - _track_at(xsp[:nv], self.vgsr_idx)
            ax[vgsr_ax_i].errorbar(
                xsp[:nv], y_v_display, yerr=(ym_v[:nv], yp_v[:nv]),
                capsize=3, elinewidth=0.75, ecolor='b', fmt='none', zorder=0
            )
        ax[pmra_ax_i].plot(x_arr, pmra_display_track, color='b', lw=spline_lw, zorder=0, ls='-')
        y_r = np.asarray(self.nested_dict['exp_meds'][5])
        ym_r = np.asarray(self.nested_dict['exp_em'][5])
        yp_r = np.asarray(self.nested_dict['exp_ep'][5])
        if y_r.ndim == 0: y_r = y_r[None]
        if ym_r.ndim == 0: ym_r = ym_r[None]
        if yp_r.ndim == 0: yp_r = yp_r[None]
        nr = int(min(len(xsp), len(y_r), len(ym_r), len(yp_r)))
        if nr > 0:
            y_r_display = y_r[:nr]
            if show_residuals:
                y_r_display = y_r_display - _track_at(xsp[:nr], self.pmra_idx)
            ax[pmra_ax_i].errorbar(
                xsp[:nr], y_r_display, yerr=(ym_r[:nr], yp_r[:nr]),
                capsize=3, elinewidth=0.75, ecolor='b', fmt='none', zorder=0
            )
        ax[pmdec_ax_i].plot(x_arr, pmdec_display_track, color='b', lw=spline_lw, zorder=0, ls='-')
        y_d = np.asarray(self.nested_dict['exp_meds'][6])
        ym_d = np.asarray(self.nested_dict['exp_em'][6])
        yp_d = np.asarray(self.nested_dict['exp_ep'][6])
        if y_d.ndim == 0: y_d = y_d[None]
        if ym_d.ndim == 0: ym_d = ym_d[None]
        if yp_d.ndim == 0: yp_d = yp_d[None]
        nd = int(min(len(xsp), len(y_d), len(ym_d), len(yp_d)))
        if nd > 0:
            y_d_display = y_d[:nd]
            if show_residuals:
                y_d_display = y_d_display - _track_at(xsp[:nd], self.pmdec_idx)
            ax[pmdec_ax_i].errorbar(
                xsp[:nd], y_d_display, yerr=(ym_d[:nd], yp_d[:nd]),
                capsize=3, elinewidth=0.75, ecolor='b', fmt='none', zorder=0
            )
        ax[feh_ax_i].plot(x_arr, feh_track, color='b', lw=spline_lw, zorder=0, ls='-')
        cbar = fig.colorbar(cm, ax=ax, orientation='vertical', pad=0.01, aspect=50)
        cbar.set_label('Membership Probability', fontsize=16)
        # Show discrete probability levels starting from current min_prob
        try:
            ticks = [min_prob]
            for t in (0.9, 0.95, 1.0):
                if t > min_prob:
                    ticks.append(t)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])
        except Exception:
            pass
        cbar.ax.tick_params(labelsize=12)
        for a in ax:
            plot_form(a)

#
        for axs in plt.gcf().get_axes():
            for artist in axs.get_children():
                artist.set_rasterized(True)
        if save_fig:
            if fig_path is None:
                fig_path = 'figures_draft/postmcmc_6panel.pdf'
            plt.savefig(fig_path, bbox_inches='tight', dpi=600)
        return fig, ax
        
        #self.add_spline_track(ax[5], med_ind=3, color='blue')

    def vis6_transform(self, quantity, phi1_values, values, residual=None):
        """
        Transform the supplied series into the same space used by vis_6_panel.
        When the latest vis_6_panel call was executed with residual=True (or residual=False),
        this method mirrors that behavior so subsequent annotations align with the plotted data.
        """
        quantity_key = str(quantity).lower()
        phi1_arr = np.asarray(phi1_values, dtype=float)
        data_arr = np.asarray(values, dtype=float)
        if residual is None:
            ctx = getattr(self, '_vis6_context', None)
            residual = bool(ctx.get('show_residuals')) if ctx else False
        if not residual:
            return data_arr
        track = self._vis6_reference_track(phi1_arr, quantity_key)
        if track is None:
            return data_arr
        return data_arr - track

    def _vis6_reference_track(self, phi1_values, quantity_key):
        """
        Evaluate the spline track corresponding to the requested quantity.
        Returns None when the quantity is not handled by the 6-panel residual logic.
        """
        phi1_arr = np.asarray(phi1_values, dtype=float)
        quantity = str(quantity_key).lower()
        idx_map = {
            'vgsr': self.vgsr_idx,
            'vg': self.vgsr_idx,
            'pmra': self.pmra_idx,
            'mu_alpha': self.pmra_idx,
            'pmdec': self.pmdec_idx,
            'mu_delta': self.pmdec_idx
        }
        med_idx = idx_map.get(quantity)
        if med_idx is None:
            return None
        return apply_spline(
            phi1_arr,
            self.spline_points_dict['phi1_spline_points'],
            self.nested_dict['meds'][med_idx],
            self.spline_points_dict['spline_k']
        )

    def GaiaDECALS_cut(self, pad=None, GaiaDeCALS_path = '', upper_bound_feh=True, MaskList=['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2'], save_fig=False, fig_path=None, addOrbit=False, useBox=False, isoCut=True):
        if pad is None or len(pad) == 0:
            if hasattr(self, 'pad'):
                pad = self.pad
            else:
                pad = np.array([0.3, 0.3, 10])

        self.GaiaDeCALS = GaiaDeCALSHandler(self, GaiaDeCALS_data=table.Table.read(GaiaDeCALS_path).to_pandas(), stream_data=self.stream_data, isochrone_path=self.isochrone_path, min_dist=self.min_dist)
        self.GaiaDeCALS.apply_box_cut(pad=pad)
        if isoCut:
            self.GaiaDeCALS.apply_iso_cut(color_indx_wiggle=0.05)
        else:
            self.GaiaDeCALS.apply_iso_cut(color_indx_wiggle=2)
        self.GaiaDeCALS.apply_phot_metallicity()

        gaia_decals = self.GaiaDeCALS.data[self.GaiaDeCALS.sel_all]
        #sel_qual = get_sel_qual_mask()

        #desi_match_all = gaia_decals[np.isin(gaia_decals['source_id'],self.desi_data['SOURCE_ID'])]
        #desi_match_qual = gaia_decals[np.isin(gaia_decals['source_id'],self.desi_data['SOURCE_ID'][sel_qual])]
        if useBox:
            stream_data = self.box_data
        else:
            stream_data = self.stream_data
        desi_match_prob = gaia_decals[np.isin(gaia_decals['source_id'],stream_data['SOURCE_ID'])]
        self.gaia_decals = gaia_decals

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax1.scatter(gaia_decals['phi1'], gaia_decals['phi2'], marker='s',facecolor='0.2',  s=35, linewidth=0.75, zorder=0, edgecolor='tab:blue', label='Gaia+DeCaLs', alpha=0.25)
        #ax.scatter(desi_match_all['phi1'], desi_match_all['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75, zorder=0, edgecolor='k', label='In DESI, no quality cut', alpha=0.9)
        #ax1.scatter(desi_match_qual['phi1'], desi_match_qual['phi2'],marker='s',facecolor='none', s=35, linewidth=0.75, zorder=0, edgecolor='orange', alpha=0.9)
        ax1.scatter(desi_match_prob['phi1'], desi_match_prob['phi2'],marker='s',facecolor='none', s=35, linewidth=0.75, zorder=0, edgecolor='blue',alpha=0.9)

        ax1.scatter(stream_data['phi1'], stream_data['phi2'],marker='s',facecolor='none', edgecolor='blue', s=40, linewidth=0.75, ls=(6, (2, 2)), zorder=0,  alpha=1)
        # put legend above plot
        ax1.legend(prop={'size': 10}, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
        #ax.axhline(-5, c='tab:orange')
        ax1.set_xlim(-10, 39)
        ax1.set_ylim(-10,10)
        # ax1.set_xlabel(r'$\phi_1$ [deg]', fontsize=14)
        ax1.set_ylabel(r'$\phi_2$ [deg]', fontsize=14)

        if addOrbit:
            #check if results_o param exists
            if hasattr(self, 'results_o'):
                self.add_orbit_track(ax1, self.results_o, track='phi2')
                self.add_orbit_track(ax2, self.results_o, track='phi2')
            else:
                print('No orbit has been run yet, please run do_orbit first')
        

        legend_handles = [
            plt.Line2D([0], [0], marker='s', linestyle='none', mec='none', mfc='0.8', label='Gaia+DeCaLs', alpha=1, markersize=7),
            # plt.Line2D([0], [0], marker='s', linestyle='none', mec='orange', mfc='none', label='In DESI, pass quality cuts', markersize=7),
            plt.Line2D([0], [0], marker='s', linestyle='none', mec='blue', mfc='none', label=r'Box-Cut Members', markersize=7),
        ]
        ax1.legend(handles=legend_handles, ncol=3, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False)

        plot_form(ax1)


        ax2.scatter(gaia_decals['phi1'], gaia_decals['phi2'], marker='s',facecolor='0.2', s=35, linewidth=0.75, zorder=0, edgecolor='tab:blue', label='Gaia+DeCaLs', alpha=0.25)
        #ax.scatter(desi_match_all['phi1'], desi_match_all['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75, zorder=0, edgecolor='k', label='In DESI, no quality cut', alpha=0.9)
        #ax.scatter(desi_match_qual['phi1'], desi_match_qual['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75, zorder=0, edgecolor='m', label='In DESI, quality cut', alpha=0.9)
        #ax.scatter(desi_match_prob['phi1'], desi_match_prob['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75.5, zorder=0, edgecolor='tab:orange',label=r'p > 0.5 $\in$ DESI',alpha=0.9)
        #ax.scatter(stream_data['phi1'], stream_data['phi2'],marker='s',facecolor='none', s=60, linewidth=0.75, ls=(6, (2, 2)), zorder=0, edgecolor='tab:orange',  alpha=1)
        # put legend above plot
        #ax2.legend(prop={'size': 8}, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.07), frameon=False)
        ax2.set_xlim(-10, 39)
        ax2.set_ylim(-10,10);
        ax2.set_xlabel(r'$\phi_1$ [deg]', fontsize=14)
        ax2.set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        plot_form(ax2)
        plt.tight_layout()
        for ax in plt.gcf().get_axes():
            for artist in ax.get_children():
                artist.set_rasterized(True)
        if save_fig:
            if fig_path is None:
                fig_path = 'figures_draft/gaia_decals_desi_stream.pdf'
            plt.savefig(fig_path, bbox_inches='tight', dpi=600)
        plt.show()

    def vis_isochrone(
        self,
        color_index_wiggle=0.18,
        isochrone_path='/Users/nasserm/Documents/vscode/research/streamTut/DESI-DR1_streamTutorial/data/dotter/iso_a13.5_z0.00010.dat',
        return_axes=False,
        absolute=True,
        horizontal_branch=False,
        min_prob=0.5,
        box_cut=False,
        track=None,
    ):
        """
        Visualize the isochrone on a CMD with stream members.
        
        Parameters
        ----------
        color_index_wiggle : float
            Tolerance for isochrone color matching
        isochrone_path : str
            Path to the Dotter isochrone file
        return_axes : bool
            If True, return (fig, ax) tuple
        absolute : bool
            If True, plot absolute magnitudes. Requires either `track` parameter
            or will fall back to using self.min_dist.
        horizontal_branch : bool
            If True, include horizontal branch region
        min_prob : float
            Minimum probability for membership coloring
        box_cut : bool
            If True, use box-cut selected members
        track : galstreams Track6D object, optional
            A galstreams track object (e.g., mwsts.get('M92-I21').track) that 
            provides distance as a function of position along the stream. When
            absolute=True, the track distance will be interpolated at each star's
            phi1 position to compute proper absolute magnitudes.
            Example: track=mwsts.get('M92-I21').track
        """
        def flux_to_mag(flux):
            mag = np.full_like(flux, np.nan, dtype=float)
            mask = flux > 0
            mag[mask] = 22.5 - 2.5 * np.log10(flux[mask])
            return mag
        
        def _get_track_distances(track, phi1_values):
            """Interpolate track distances at given phi1 values (in kpc)."""
            if track is None:
                return None
            try:
                # Get track coordinates in stream frame
                track_phi1, _ = stream_funcs.ra_dec_to_phi1_phi2(
                    self.frame, 
                    track.ra.value * u.deg, 
                    track.dec.value * u.deg
                )
                track_dist = track.distance.value  # in kpc
                
                # Sort by phi1 for interpolation
                sort_idx = np.argsort(track_phi1)
                track_phi1_sorted = track_phi1[sort_idx]
                track_dist_sorted = track_dist[sort_idx]
                
                # Interpolate distance at each star's phi1
                interp_dist = np.interp(
                    phi1_values, 
                    track_phi1_sorted, 
                    track_dist_sorted,
                    left=track_dist_sorted[0],
                    right=track_dist_sorted[-1]
                )
                return interp_dist  # in kpc
            except Exception as e:
                print(f"Warning: Could not interpolate track distances: {e}")
                return None

        dotter_mp = np.loadtxt(isochrone_path)
        dotter_g_mp = dotter_mp[:, 6]
        dotter_r_mp = dotter_mp[:, 7]

        dm = None
        dm_for_axis = None  # Separate dm for right axis when using track
        min_dist_attr = getattr(self, 'min_dist', None)
        try:
            _min_dist = float(min_dist_attr)
        except (TypeError, ValueError, AttributeError):
            _min_dist = None
        if _min_dist is not None and np.isfinite(_min_dist) and _min_dist > 0:
            distance_kpc = _min_dist / 1000.0 if _min_dist > 1000.0 else _min_dist
            dm = d2dm(distance_kpc)
            dm_for_axis = dm  # Default to same as dm
        
        # If track is provided, compute dm_for_axis from median track distance
        if track is not None:
            try:
                track_dist_kpc = track.distance.value  # in kpc
                median_dist_kpc = np.median(track_dist_kpc[np.isfinite(track_dist_kpc)])
                if np.isfinite(median_dist_kpc) and median_dist_kpc > 0:
                    dm_for_axis = d2dm(median_dist_kpc)
            except Exception as e:
                print(f"Warning: Could not compute dm from track, using min_dist: {e}")

        if not absolute:
            if dm is None:
                raise ValueError("Minimum distance is required to plot apparent magnitudes.")
            dotter_g_mp += dm
            dotter_r_mp += dm

        if horizontal_branch:
            bhb_color_wiggle = 0.4
            bhb_abs_mag_wiggle = 0.1
            dm_m92_harris = 14.59
            m92ebv = 0.023
            m92ag = m92ebv * 3.184
            m92ar = m92ebv * 2.130
            m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05])
            m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1])
            m92_hb_g = m92_hb_r + m92_hb_col
            des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
            des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
            des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
            des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris

        if hasattr(self, 'desi_data'):
            if self.desi_data is not None:
                self.desi_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
                self.desi_data['phi1'], self.desi_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(
                    self.frame,
                    np.array(self.desi_data['TARGET_RA']) * u.deg,
                    np.array(self.desi_data['TARGET_DEC']) * u.deg,
                )
                self.desi_handler.data = self.desi_data
                self.desi_handler.apply_box_cut(pad=np.array([400., 0., 100, 100, 10]))
                self.desi_handler.apply_plx_cut(min_dist=self.min_dist)
                self.desi_handler.apply_phi1_mask()
                self.desi_handler.all_box_cuts_DESI(withIso=False)
                desi_data = self.desi_handler.data[self.desi_handler.sel]

                desi_ebv = np.array(desi_data['EBV'])
                desi_g_flux, desi_r_flux = np.array(desi_data['FLUX_G']), np.array(desi_data['FLUX_R'])
                desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(
                    desi_ebv, desi_g_flux, desi_r_flux, self.min_dist * 1000
                )
                big_iso = isochrone_cut(
                    color_indx_wiggle=color_index_wiggle,
                    isochrone_path=isochrone_path,
                    desi_data=desi_data,
                    desi_distance=self.min_dist * 1000, withAss=False
                )
        else:
            desi_data = None
            big_iso = None

        def _get_component(handler, label):
            data = getattr(handler, 'data', None)
            sel = getattr(handler, 'sel', None)
            if data is None or sel is None:
                return None
            sel = np.asarray(sel, dtype=bool)
            if sel.size == 0 or not np.any(sel):
                return None
            if isinstance(data, pd.DataFrame):
                if sel.size != len(data):
                    print(f"Skipping {label}: selection mask length mismatch.")
                    return None
                subset = data.loc[sel].copy()
            else:
                try:
                    subset = data[sel]
                except Exception:
                    try:
                        subset = data[np.where(sel)]
                    except Exception:
                        print(f"Skipping {label}: unable to subset data with selection mask.")
                        return None
                subset = subset if isinstance(subset, pd.DataFrame) else pd.DataFrame(subset)
            required_cols = {'EBV', 'FLUX_G', 'FLUX_R'}
            if not required_cols.issubset(subset.columns):
                missing = sorted(required_cols.difference(subset.columns))
                print(f"Skipping {label}: missing columns {missing} for isochrone plotting.")
                return None
            return subset

        def _prepare_component(handler, label):
            subset = _get_component(handler, label)
            if subset is None or len(subset) == 0:
                return None
            ebv = np.asarray(subset['EBV'])
            g_flux = np.asarray(subset['FLUX_G'])
            r_flux = np.asarray(subset['FLUX_R'])
            
            # Get phi1 values for track distance interpolation
            phi1_vals = None
            if 'phi1' in subset.columns:
                phi1_vals = np.asarray(subset['phi1'])
            
            # Compute distances: use track if provided and absolute=True, else fall back
            if absolute and track is not None and phi1_vals is not None:
                interp_distances = _get_track_distances(track, phi1_vals)
                if interp_distances is not None:
                    # Convert kpc to pc for get_colour_index_and_abs_mag
                    distances_pc = interp_distances * 1000.0
                else:
                    distances_pc = self.min_dist * 1000
            else:
                distances_pc = self.min_dist * 1000
            
            colour_idx, abs_mag, r_mag = stream_funcs.get_colour_index_and_abs_mag(
                ebv, g_flux, r_flux, distances_pc
            )
            if not absolute:
                _, _, r_mag = stream_funcs.get_colour_index_and_abs_mag(
                    ebv, g_flux, r_flux, None
                )
            return {
                'label': label,
                'colour_idx': np.asarray(colour_idx, dtype=float),
                'abs_mag': np.asarray(abs_mag, dtype=float),
                'r_mag': np.asarray(r_mag, dtype=float),
            }

        component_points = []
        if box_cut:
            # Gather box-cut selections from available handlers
            ms = getattr(self, 'ms_handler', None)
            bhb = getattr(self, 'bhb_handler', None)
            rrl = getattr(self, 'rrl_handler', None)

            for handler, label in ((ms, 'MS+RG'), (bhb, 'BHB'), (rrl, 'RRL')):
                component = _prepare_component(handler, label)
                if component is None:
                    continue
                component_points.append(component)

            if not component_points:
                raise ValueError("No box-cut members available. Run 'box_cut' to populate selections before plotting.")

            x = np.concatenate([comp['colour_idx'] for comp in component_points]) if component_points else np.array([])
            y = np.concatenate([(comp['abs_mag'] if absolute else comp['r_mag']) for comp in component_points]) if component_points else np.array([])
        else:
            stream_data = self.stream_data
            stream_ebv = np.array(stream_data['EBV'])
            stream_g_flux, stream_r_flux = np.array(stream_data['FLUX_G']), np.array(stream_data['FLUX_R'])
            
            # Compute distances for stream_data
            if absolute and track is not None and 'phi1' in stream_data.colnames:
                stream_phi1 = np.array(stream_data['phi1'])
                interp_distances = _get_track_distances(track, stream_phi1)
                if interp_distances is not None:
                    stream_distances_pc = interp_distances * 1000.0
                else:
                    stream_distances_pc = self.min_dist * 1000
            else:
                stream_distances_pc = self.min_dist * 1000
            
            stream_colour_index, stream_abs_mag, stream_r_mag = stream_funcs.get_colour_index_and_abs_mag(
                stream_ebv, stream_g_flux, stream_r_flux, stream_distances_pc
            )

            if not absolute:
                stream_colour_index, _, stream_r_mag = stream_funcs.get_colour_index_and_abs_mag(
                    stream_ebv, stream_g_flux, stream_r_flux, None
                )
                if desi_data is not None:
                    desi_ebv = np.array(desi_data['EBV'])
                    desi_g_flux, desi_r_flux = np.array(desi_data['FLUX_G']), np.array(desi_data['FLUX_R'])
                    desi_colour_index, _, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(
                        desi_ebv, desi_g_flux, desi_r_flux, None
                    )

            x = np.asarray(stream_colour_index, dtype=float)
            y = np.asarray(stream_abs_mag if absolute else stream_r_mag, dtype=float)

            highlight_components = []
            if horizontal_branch:
                bhb_handler = getattr(self, 'bhb_handler', None)
                rrl_handler = getattr(self, 'rrl_handler', None)
                for handler, label in ((bhb_handler, 'BHB'), (rrl_handler, 'RRL')):
                    component = _prepare_component(handler, label)
                    if component is not None:
                        highlight_components.append(component)
            else:
                highlight_components = []

        from matplotlib.colors import PowerNorm
        norm = PowerNorm(gamma=5, vmin=min_prob, vmax=1) if not box_cut else None
        fig_size = (5, 5) if box_cut else (7, 5)
        fig, ax = plt.subplots(figsize=fig_size)
        cm = None

        if box_cut:
            for comp in component_points:
                label = comp['label']
                display_label = {
                    'BHB': 'BHB Members',
                    'RRL': 'RRL Members'
                }.get(label, label)
                marker, color = _label_style(label)
                if label in ('BHB', 'RRL'):
                    facecolor = 'none'
                    edgecolor = color
                else:
                    facecolor = 'none' if label == 'MS+RG' else color
                    edgecolor = color
                size = 30 if label == 'MS+RG' else 40
                y_vals = comp['abs_mag'] if absolute else comp['r_mag']
                ax.scatter(
                    comp['colour_idx'],
                    y_vals,
                    marker=marker,
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidths=0.8,
                    s=size,
                    alpha=0.85,
                    label=display_label,
                    zorder=2,
                )
        else:
            cm = ax.scatter(
                x,
                y,
                c=stream_data['stream_prob'],
                cmap='magma_r',
                norm=norm,
                edgecolors='none',
                linewidth=0.8,
                s=30,
                label='MS+RG Members',
                alpha=1,
                zorder=1,
            )

            for comp in highlight_components:
                label = comp['label']
                display_label = {
                    'BHB': 'BHB Members',
                    'RRL': 'RRL Members'
                }.get(label, label)
                marker, color = _label_style(label)
                if label in ('BHB', 'RRL'):
                    facecolor = 'none'
                    edgecolor = color
                else:
                    facecolor = 'none' if label == 'MS+RG' else color
                    edgecolor = color
                size = 40 if label in ('BHB', 'RRL') else 30
                y_vals = comp['abs_mag'] if absolute else comp['r_mag']
                ax.scatter(
                    comp['colour_idx'],
                    y_vals,
                    marker=marker,
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidths=1.0,
                    s=size,
                    alpha=0.95,
                    label=display_label,
                    zorder=3,
                )

        if self.desi_data is not None and not box_cut:
                ax.scatter(
                    desi_colour_index[big_iso], desi_abs_mag[big_iso],
                    c='lightblue', alpha=0.1, s=2, zorder=0
                )
                ax.scatter(
                    desi_colour_index[~big_iso], desi_abs_mag[~big_iso],
                    c='0.3', alpha=0.01, s=1, zorder=0
                )
        if absolute:
            ax.plot(dotter_g_mp - dotter_r_mp, dotter_r_mp, c='b', alpha=0.3)
        # ax.plot(dotter_g_mp - dotter_r_mp - color_index_wiggle, dotter_r_mp, c='b', alpha=0.3)
        # ax.plot(dotter_g_mp - dotter_r_mp + color_index_wiggle, dotter_r_mp, c='b', alpha=0.3)

        if horizontal_branch and absolute:
            ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r, c='b', alpha=0.3)

        
        if cm is not None:
            cbar = plt.colorbar(cm, ax=ax, pad=0.15)
            cbar.set_label('Membership Probability', fontsize=16)
            try:
                ticks = [min_prob]
                for t in (0.9, 0.95, 1.0):
                    if t > min_prob:
                        ticks.append(t)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])
            except Exception:
                pass
        # ax.scatter([], [], label=)
        ax.legend(loc='upper left', fontsize=10)

        if absolute:
            if box_cut:
                ax.set_xlim(-0.3, 0.8)
            else:
                ax.set_xlim(-0.1, 0.8)
            ax.invert_yaxis()
            ax.set_ylim(5, -3)
            ax.set_xlabel(r'$(g-r)$ [mag]', fontsize=14)
            ax.set_ylabel(r'$M_r$ [mag]', fontsize=14)
        else:
            finite_x = x[np.isfinite(x)]
            finite_y = y[np.isfinite(y)]
            if finite_x.size > 0:
                ax.set_xlim(finite_x.min() - 0.3, finite_x.max() + 0.3)
            else:
                ax.set_xlim(-0.5, 1.0)
            ax.invert_yaxis()
            ax.set_xlabel(r'$(g-r)$ [mag]', fontsize=14)
            ax.set_ylabel(r'$r$ [mag]', fontsize=14)
            if finite_y.size > 0:
                ax.set_ylim(finite_y.max() + 0.3, finite_y.min() - 1)
            else:
                ax.set_ylim(25, 10)

        plot_form(ax)
        if absolute and dm_for_axis is not None:
            ax.tick_params(axis='y', which='both', right=False)
            right_ax = ax.twinx()
            right_ax.set_ylim(np.array(ax.get_ylim()) + dm_for_axis)
            right_ax.set_ylabel(r'$m_r$ [mag]', fontsize=14)
            right_ax.tick_params(axis='y', which='both', direction='in', left=False, right=True)
            right_ax.minorticks_on()
            right_ax.spines['left'].set_visible(False)
            right_ax.spines['right'].set_linewidth(1)

        if return_axes:
            return fig, ax

    def segment_vdisp(self, segments, green_min=np.inf, green_max=np.inf, prob_cutoff=0.5, useBox=False, withPlot=True, save_fig=False, fig_path=None, externalSpline=False, show_right_panel=True):
        if not useBox:
            stream_data = self.stream_data
            high_prob = stream_data['stream_prob'] > prob_cutoff
            label='GMM'
        else:
            stream_data = self.box_data
            label='Box Cut'
            #high_prob is just a mask that is true for everything
            high_prob = np.ones(len(stream_data), dtype=bool)

        phi1 = stream_data['phi1'][high_prob]
        phi2 = stream_data['phi2'][high_prob]
        vgsr = stream_data['VGSR'][high_prob]
        vrad_err = stream_data['VRAD_ERR'][high_prob]

        rv_samples = {}
        rverr_samples = {}
        mcmc_results = {}
        sigmas = {}
        phi1_spline_points = self.spline_points_dict['phi1_spline_points']
        spline_k = self.spline_points_dict['spline_k']
        nested_list_meds = self.nested_dict['meds']
        # MCMC for each segment
        for i, (pmin, pmax) in enumerate(segments):
            print(f'running segment {pmin} to {pmax} deg')
            mask = (phi1 > pmin) & (phi1 < pmax)
            phi1_seg = phi1[mask]
            rv = vgsr[mask] - apply_spline(phi1_seg, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k)
            err = vrad_err[mask]

            rv_samples[i] = rv
            rverr_samples[i] = err
            mcmc_results[i] = vdisp.mcmc(rv, err, nsteps=500)
            sigmas[i] = 10**mcmc_results[i][:,1]

        # Print summary stats
        for i, (pmin, pmax) in enumerate(segments):
            mu = mcmc_results[i][:,0]
            sigma = sigmas[i]
            print(f"Segment {pmin} to {pmax} deg:")
            print(f"  mean RV: {np.median(mu):.2f} (+{np.percentile(mu, 84)-np.median(mu):.2f}/-{np.median(mu)-np.percentile(mu, 16):.2f})")
            print(f"  σ_v:     {np.median(sigma):.2f} (+{np.percentile(sigma, 84)-np.median(sigma):.2f}/-{np.median(sigma)-np.percentile(sigma, 16):.2f})")

        # MCMC for full
        rv = vgsr - apply_spline(phi1, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k)
        err = vrad_err

        rv_sample = rv
        rverr_sample= err
        mcmc_result = vdisp.mcmc(rv, err, nsteps=500)
        sigma = 10**mcmc_result[:,1]

        mu = mcmc_result[:,0]

        print(" ")
        print(f"OVERALL")
        print(f"  mean RV: {np.median(mu):.2f} (+{np.percentile(mu, 84)-np.median(mu):.2f}/-{np.median(mu)-np.percentile(mu, 16):.2f})")
        print(f"  σ_v:     {np.median(sigma):.2f} (+{np.percentile(sigma, 84)-np.median(sigma):.2f}/-{np.median(sigma)-np.percentile(sigma, 16):.2f})")
    

        if withPlot:
            if not hasattr(self, 'results_o'):
                print('No orbit has been run yet, plots will not include orbit track')
            from matplotlib.colors import Normalize

            # Create a new figure with three vertically-stacked subplots sharing the same x-axis
            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            fig.subplots_adjust(hspace=0.1, wspace=0.3)  # Reduced vertical spacing
            ax1 = ax[0]  # Top panel
            ax2 = ax[1]  # Middle panel
            ax3 = ax[2]  # Bottom panel
            ax4 = None
            axes_to_style = [ax1, ax2, ax3]


            # Color map and normalization
            color_map = 'seismic'
            min_prob = -40
            max_prob = 40   

            norm = Normalize(vmin=min_prob, vmax=max_prob)

            # Top left panel: phi2 vs phi1 color-coded by residual VGSR
            cm = ax1.scatter(phi1, phi2, 
                            c=vgsr - apply_spline(phi1, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                            cmap=color_map, s=50, linewidth=0.5, edgecolors='k',
                            zorder=1, norm=norm)
            ax1.axhline(-5, color='k', lw=1, ls='dotted', zorder=0, alpha=0.5)
            ax1.axhline(5, color='k', lw=1, ls='dotted', zorder=0, alpha=0.5)
            #ax1.scatter(desi_data['phi1'], desi_data['phi2'], c='0.9', s=10, linewidth=0.5,
            #            edgecolors='k', zorder=0, alpha=0.01, label='Background')


            #ax1.plot(orbit_phi1, ointerps['phi2'](orbit_phi1), color='r', ls='--', label='Orbit', zorder=0)

            ax1.set_ylabel('$\phi_2$ [deg]')
            ax1.set_xlim(-13, 43)
            ax1.set_ylim(-7, 7)
            ax1.tick_params(direction='in')
            ax1.tick_params(labelbottom=False)  # Hide x-axis tick labels

            # Middle left panel: VGSR vs phi1
            ax2.scatter(phi1, vgsr, 
                        c=vgsr - apply_spline(phi1, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                        cmap=color_map, s=50, linewidth=0.5, edgecolors='k', zorder=1, norm=norm)
            x_arr = np.linspace(-9, 38, 100)
            ax2.plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                    c='b', ls='-', lw=1.8, zorder=0)
            #ax2.plot(orbit_phi1, ointerps['vgsr'](orbit_phi1), color='r', ls='--', label='Orbit', zorder=0)

            ax2.set_ylim(-60, 35)
            ax2.set_ylabel(r'$v_{GSR}$ [km/s]')
            ax2.tick_params(direction='in')
            ax2.tick_params(labelbottom=False)  # Hide x-axis tick labels
            # Error bars for ax2
            for i, (pmin, pmax) in enumerate(segments):
                mask = (phi1 > pmin) & (phi1 < pmax)
                phi1_seg = phi1[mask]
                vgsr_seg = vgsr[mask]
                err = vrad_err[mask]
                
                # Plot error bars
                ax2.errorbar(phi1_seg, vgsr_seg, yerr=err, fmt='None', color='0', alpha=0.9, zorder=0)

            # Bottom left panel: velocity dispersion vs phi1
            for i, (pmin, pmax) in enumerate(segments):
                x_fill = np.linspace(pmin + 0.2, pmax - 0.2, 100)
                sigma = sigmas[i]
                ax3.plot(x_fill, np.ones_like(x_fill) * np.median(sigma), c='k', lw=1, zorder=1)
                ax3.fill_between(x_fill, np.percentile(sigma, 16), np.percentile(sigma, 84), 
                                color='0.7', alpha=0.8, zorder=0)

            ax3.set_ylabel(r'$\sigma_{v}$ [km/s]')
            ax3.set_ylim(0, 25)
            ax3.tick_params(direction='in')


            #----------------------Decide your phi1 green range----------------------------#
            green_min = green_min
            green_max = green_max
            if green_min == np.inf or green_max == np.inf:
                green_min = np.min(phi1) + 0.2
                green_max = np.max(phi1) - 0.2
            #------------------------------------------------------------------------------#

            mask = (phi1 > green_min) & (phi1 < green_max)
            if show_right_panel:
                ax4 = fig.add_axes([1.06, 0.11, 0.21, 0.77])  # corresponding to x position, y position, width, height
                ax4.scatter(phi2[mask], vgsr[mask] - apply_spline(phi1[mask], phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k), cmap=color_map, 
                            c=vgsr[mask] - apply_spline(phi1[mask], phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                            s=50, linewidth=0.5, edgecolors='k', zorder=1, norm=norm)
                ax4.errorbar(phi2[mask], vgsr[mask] - apply_spline(phi1[mask], phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k), yerr=vrad_err[mask], fmt='None', color='0', 
                            alpha=0.9, zorder=0)

                ax4.set_xlim(-7, 7) # hard coded
                ax4.set_ylim(-30, 25) # hard coded
                ax4.set_ylabel(r'$\Delta v_{GSR}$ [km/s]')
                ax4.tick_params(axis='x', colors='k')
                ax4.set_xlabel(r'$\phi_2$ [deg]')
                ax4.tick_params(direction='in')
                axes_to_style.append(ax4)




            # Colorbar
            cbar_targets = [ax1.get_position(), ax2.get_position()]
            y0 = min(pos.y0 for pos in cbar_targets)
            y1 = max(pos.y1 for pos in cbar_targets)
            cbar_height = y1 - y0

            if show_right_panel and ax4 is not None:
                rightmost = ax4.get_position().x1
                cbar_x0 = rightmost + 0.04
            else:
                rightmost = max(pos.x1 for pos in cbar_targets)
                cbar_x0 = rightmost + 0.01

            cbar_ax = fig.add_axes([cbar_x0, y0, 0.02, cbar_height])
            cbar = fig.colorbar(cm, cax=cbar_ax)
            cbar.set_label(r'$\Delta v_{GSR}$ [km/s]')

            # Grid and styling for all subplots
            for a in axes_to_style:
                a.grid(ls='-.', alpha=0.2, zorder=0)
                a.tick_params(direction='in')
                a.spines['top'].set_linewidth(1)
                a.spines['right'].set_linewidth(1)
                a.spines['left'].set_linewidth(1)
                a.spines['bottom'].set_linewidth(1)
                a.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                a.minorticks_on()


            for a in [ax2]:
                # Draw at the bottom (y=0 in axis coordinates)
                a.plot([green_min, green_max], [0, 0], transform=a.get_xaxis_transform(), color='green', linewidth=6, alpha=0.8)
                a.plot([green_min, green_max], [1, 1], transform=a.get_xaxis_transform(), color='green', linewidth=6, alpha=0.8)


            if show_right_panel and ax4 is not None:
                ax4.plot([0, 1], [0, 0], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # bottom edge
                ax4.plot([0, 1], [1, 1], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # top edge
                ax4.plot([0, 0], [0, 1], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # left edge
                ax4.plot([1, 1], [0, 1], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # right edge

            ax3.set_xlabel(r'$\phi_1$ [deg]', fontsize=12)

            if show_right_panel:
                panel_labels = ['(a)', '(b)', '(c)', '(d)']
                for label_ax, label in zip([ax1, ax2, ax3, ax4], panel_labels):
                    label_ax.text(
                        0.03,
                        0.97,
                        label,
                        transform=label_ax.transAxes,
                        ha='left',
                        va='top',
                        fontweight='bold',
                        bbox=dict(
                            boxstyle='square,pad=0.15',
                            facecolor='0.8',
                            edgecolor='none',
                            alpha=0.3,
                        ),
                    )

            # Create legend
            legend_handles = {
                'C-19': plt.Line2D([0], [0], marker='o', color='w', label='C-19 Members', 
                                    markerfacecolor='b', markersize=8, linestyle='None'),
                'Orbit': plt.Line2D([0], [0], color='r', lw=1, ls='solid',label='Orbit')
            }
            ax1.legend(handles=legend_handles.values(), loc='upper right', fontsize=10, ncol=2, frameon=True)
            legend_handles = {
                'Spline' : plt.Line2D([0], [0], color='b',ls='solid', lw=1, label='Spline Track')}
            ax2.legend(handles=legend_handles.values(), loc='upper right', fontsize=10, frameon=False)

            if hasattr(self, 'results_o'):
                self.add_orbit_track(ax1, self.results_o, track='phi2')
                self.add_orbit_track(ax2, self.results_o, track='vgsr')

            # Title for the top-left plot
            #ax1.set_title(f'C-19, p> {p}' , fontsize=12)

            if externalSpline:
                return fig, ax

            else: 
                if save_fig:
                    for ax in plt.gcf().get_axes():
                        for artist in ax.get_children():
                            artist.set_rasterized(True)
                    if fig_path is None:
                        fig_path = 'figures_draft/vdispbox.pdf'
                    plt.savefig(fig_path, bbox_inches='tight', dpi=600)


                plt.show()


    def show_completness(
        self,
        gaia_data_path,
        member_type='boxcut',
        empirical_cuts=None,
        verbose=False,
        min_prob=0.5,
        return_data=False,
    ):
        """Visualise Gaia completeness with optional DESI member overlays.

        Parameters
        ----------
        gaia_data_path : str or pathlib.Path
            Path to the Gaia x DECaLS catalogue used to assess completeness.
        member_type : {'boxcut', 'mcmc'}, optional
            Which DESI members to highlight on top of the Gaia selection.
        empirical_cuts : dict, optional
            Dictionary overriding default empirical selection thresholds.
        verbose : bool, optional
            If True, display diagnostic plots for each selection stage.
        min_prob : float, optional
            Membership probability threshold when ``member_type='mcmc'``.
        return_data : bool, optional
            If True, return a dictionary with the selection DataFrames and masks.

        Returns
        -------
        dict or None
            Selection summary when ``return_data`` is True, otherwise None.
        """

        if gaia_data_path is None or gaia_data_path == '':
            raise ValueError("A Gaia catalogue path must be provided.")

        if self.frame is None:
            raise ValueError("Stream frame (self.frame) must be defined before calling show_completness.")

        if not hasattr(self, 'spline_points_dict') or self.spline_points_dict is None:
            raise ValueError("Spline information is unavailable. Initialise StreamMembers with a valid run directory first.")

        if empirical_cuts is not None and not isinstance(empirical_cuts, dict):
            raise TypeError("empirical_cuts must be a dictionary if provided.")

        default_cuts = {
            'sigma_pad': 1.0,
            'pmra_pad': 0.3,
            'pmdec_pad': 0.3,
            'phi2_pad': 10.0,
            'mag_min': 20.5,
            'mag_max': 16.0,
            'color_wiggle': 0.05,
            'mp_slope': 0.24 / 0.25,
            'mp_offset': 0.24,
            'mp_pivot': 0.5,
            'exclude_release': 9011,
            'distance_kpc': None,
        }

        cuts = default_cuts.copy()
        if empirical_cuts:
            cuts.update(empirical_cuts)

        if cuts['distance_kpc'] is None:
            if self.min_dist is None:
                cuts['distance_kpc'] = 18.0
            else:
                dist_val = float(self.min_dist)
                # Accept either parsec or kiloparsec input
                cuts['distance_kpc'] = dist_val / 1e3 if dist_val > 1e3 else dist_val

        member_type = (member_type or 'boxcut').lower()
        if member_type not in {'boxcut', 'mcmc'}:
            raise ValueError("member_type must be 'boxcut' or 'mcmc'.")

        required_aliases = {
            'source_id': ['source_id', 'SOURCE_ID'],
            'ra': ['ra', 'RA', 'TARGET_RA'],
            'dec': ['dec', 'DEC', 'TARGET_DEC'],
            'pmra': ['pmra', 'PMRA'],
            'pmdec': ['pmdec', 'PMDEC'],
            'parallax': ['parallax', 'PARALLAX'],
            'parallax_error': ['parallax_error', 'PARALLAX_ERROR'],
            'flux_g': ['flux_g', 'FLUX_G'],
            'flux_r': ['flux_r', 'FLUX_R'],
            'flux_z': ['flux_z', 'FLUX_Z'],
            'ebv': ['ebv', 'EBV'],
        }

        optional_aliases = {
            'phot_g_mean_mag': ['phot_g_mean_mag', 'PHOT_G_MEAN_MAG'],
            'release': ['release', 'RELEASE'],
            'stream_prob': ['stream_prob', 'STREAM_PROB'],
            'phi1': ['phi1'],
            'phi2': ['phi2'],
            'dist_mod': ['dist_mod', 'DIST_MOD'],
        }

        phi1_spline_points = np.asarray(self.spline_points_dict['phi1_spline_points'])
        spline_k = self.spline_points_dict['spline_k']
        pmra_idx = getattr(self, 'pmra_idx', 5)
        pmdec_idx = getattr(self, 'pmdec_idx', 7)

        # Re-resolve isochrone in case user moved files after init
        import os
        iso_path = self.isochrone_path
        if not os.path.exists(iso_path):
            iso_path = os.path.join(os.path.dirname(__file__), 'data', 'dotter', 'iso_a13.5_z0.00010.dat')
        try:
            dotter_mp = np.loadtxt(iso_path)
        except OSError as exc:
            raise OSError(f"Unable to load isochrone file: {iso_path}") from exc

        dotter_g = dotter_mp[:, 6]
        dotter_r = dotter_mp[:, 7]
        colour_track_x = dotter_r[::-1]
        colour_track_y = (dotter_g - dotter_r)[::-1] - 0.01
        distance_kpc = cuts['distance_kpc']
        distance_mod = d2dm(distance_kpc)

        def _rename_with_aliases(df, alias_map, dataset_name, errors='raise'):
            rename_map = {}
            for canonical, aliases in alias_map.items():
                if canonical in df.columns:
                    continue
                match = next((col for col in aliases if col in df.columns), None)
                if match is None:
                    if errors == 'raise':
                        raise KeyError(f"{dataset_name} is missing required column '{canonical}'. Expected one of {aliases}.")
                    continue
                rename_map[match] = canonical
            if rename_map:
                df = df.rename(columns=rename_map)
            return df

        def _prepare_catalog(data, dataset_name):
            if isinstance(data, table.Table):
                df = data.to_pandas()
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                raise TypeError(f"Unsupported data type for {dataset_name}: {type(data)}")
            df = _rename_with_aliases(df, required_aliases, dataset_name, errors='raise')
            df = _rename_with_aliases(df, optional_aliases, dataset_name, errors='ignore')
            return df

        def _compute_magnitudes(df):
            ebv = np.asarray(df['ebv'], dtype=float)
            for band, coeff in (('g', 3.186), ('r', 2.140), ('z', 1.196)):
                flux_col = f'flux_{band}'
                flux = np.asarray(df[flux_col], dtype=float)
                mag = np.full_like(flux, np.nan, dtype=float)
                positive = flux > 0
                with np.errstate(divide='ignore'):
                    mag[positive] = 22.5 - 2.5 * np.log10(flux[positive])
                df[f'{band}mag'] = mag
                df[f'{band}mag0'] = mag - ebv * coeff
            return df

        def _ensure_stream_coords(df, dataset_name):
            if 'phi1' not in df.columns or 'phi2' not in df.columns or df['phi1'].isnull().any() or df['phi2'].isnull().any():
                ra = np.asarray(df['ra'], dtype=float)
                dec = np.asarray(df['dec'], dtype=float)
                phi1, phi2 = stream_funcs.ra_dec_to_phi1_phi2(self.frame, ra * u.deg, dec * u.deg)
                df['phi1'] = phi1
                df['phi2'] = phi2
            return df

        def _deduplicate_gaia(df):
            result = df
            if 'release' in result.columns:
                result = result[result['release'] != cuts['exclude_release']]
            if 'source_id' in result.columns:
                if 'phot_g_mean_mag' in result.columns:
                    delta = np.abs(result['rmag'] - result['phot_g_mean_mag'])
                    result = (
                        result.assign(_delta_mag=delta)
                        .sort_values('_delta_mag')
                        .drop_duplicates('source_id', keep='first')
                        .drop(columns=['_delta_mag'])
                    )
                else:
                    result = result.drop_duplicates('source_id', keep='first')
            return result.reset_index(drop=True)

        def _build_selection_masks(df, dataset_name):
            pad = np.array([
                cuts['pmra_pad'] * cuts['sigma_pad'],
                cuts['pmdec_pad'] * cuts['sigma_pad'],
                cuts['phi2_pad'],
            ])

            box_masks = stream_funcs.box_cuts(
                df,
                phi1_spline_points,
                self.nested_dict['meds'],
                spline_k,
                pad,
                blind_panels=['pmra', 'pmdec', 'phi2'],
                blind_meds_ind=[pmra_idx, pmdec_idx],
            )

            sel_pmra = np.array(box_masks['pmra'], dtype=bool)
            sel_pmdec = np.array(box_masks['pmdec'], dtype=bool)
            sel_phi2 = np.array(box_masks['phi2'], dtype=bool)
            sel_plx = plx_mask(distance_kpc, np.asarray(df['parallax']), np.asarray(df['parallax_error']))

            # get_colour_index_and_abs_mag expects distance in parsec
            # Use flux-based magnitudes for a clean CMD diagnostic, but
            # for consistency with the selection we also compute CMD axes
            # directly from the extinction-corrected mags used in cuts.
            _, abs_mag_flux, _ = stream_funcs.get_colour_index_and_abs_mag(
                df['ebv'], df['flux_g'], df['flux_r'], np.full(len(df), distance_kpc * 1000.0)
            )
            colour_index = df['gmag0'] - df['rmag0']
            abs_mag = df['rmag0'] - distance_mod

            # Isochrone test in apparent magnitude space converted to colour_index
            colour_track = apply_spline(df['rmag0'] - distance_mod, colour_track_x, colour_track_y, k=1)
            sel_iso = np.abs(colour_index - colour_track) < cuts['color_wiggle']

            mag_sel = (df['rmag0'] < cuts['mag_min']) & (df['rmag0'] > cuts['mag_max'])
            mp_val = (
                df['rmag0']
                - df['zmag0']
                - cuts['mp_slope'] * (df['gmag0'] - df['rmag0'] - cuts['mp_pivot'])
                - cuts['mp_offset']
            )
            sel_mp = mp_val > 0

            core_mask = sel_pmra & sel_pmdec & sel_phi2 & sel_plx
            sel_all = core_mask & sel_iso & mag_sel & sel_mp

            masks = {
                'pmra': sel_pmra,
                'pmdec': sel_pmdec,
                'phi2': sel_phi2,
                'plx': sel_plx,
                'iso': sel_iso,
                'mag': mag_sel,
                'mp': sel_mp,
                'core': core_mask,
                'all': sel_all,
            }

            diagnostics = {
                'colour_index': colour_index,
                'abs_mag': abs_mag,
                'abs_mag_flux': abs_mag_flux,
                'mp_metric': mp_val,
            }

            return masks, diagnostics

        def _prepare_boxcut_members():
            if self.desi_data is None:
                return pd.DataFrame()
            desi_df = _prepare_catalog(self.desi_data, 'DESI')
            desi_df = _compute_magnitudes(desi_df)
            desi_df = _ensure_stream_coords(desi_df, 'DESI')
            masks, _ = _build_selection_masks(desi_df, 'DESI')
            return desi_df.loc[masks['all']].reset_index(drop=True)

        def _prepare_mcmc_members():
            if self.stream_data is None or len(self.stream_data) == 0:
                return pd.DataFrame()
            stream_df = _prepare_catalog(self.stream_data, 'Stream Members')
            if 'stream_prob' in stream_df.columns and min_prob is not None:
                stream_df = stream_df[stream_df['stream_prob'] >= min_prob]
            stream_df = _ensure_stream_coords(stream_df, 'Stream Members')
            return stream_df.reset_index(drop=True)

        def _plot_kinematic_diagnostics(df, masks):
            x_grid = np.linspace(df['phi1'].min() - 2, df['phi1'].max() + 2, 300)
            pmra_track = apply_spline(x_grid, phi1_spline_points, self.nested_dict['meds'][pmra_idx], spline_k)
            pmdec_track = apply_spline(x_grid, phi1_spline_points, self.nested_dict['meds'][pmdec_idx], spline_k)

            fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

            ax[0].plot(x_grid, np.zeros_like(x_grid), color='cyan', ls='--', zorder=0)
            ax[0].axhline(cuts['phi2_pad'], color='blue', lw=0.6)
            ax[0].axhline(-cuts['phi2_pad'], color='blue', lw=0.6)
            ax[0].scatter(df['phi1'][masks['core']], df['phi2'][masks['core']], s=2, alpha=0.1, c='k')

            ax[1].plot(x_grid, pmra_track, color='cyan', ls='--', zorder=0)
            ax[1].plot(x_grid, pmra_track + cuts['pmra_pad'] * cuts['sigma_pad'], color='blue', lw=0.6)
            ax[1].plot(x_grid, pmra_track - cuts['pmra_pad'] * cuts['sigma_pad'], color='blue', lw=0.6)
            ax[1].scatter(df['phi1'][masks['core']], df['pmra'][masks['core']], s=2, alpha=0.1, c='k')

            ax[2].plot(x_grid, pmdec_track, color='cyan', ls='--', zorder=0)
            ax[2].plot(x_grid, pmdec_track + cuts['pmdec_pad'] * cuts['sigma_pad'], color='blue', lw=0.6)
            ax[2].plot(x_grid, pmdec_track - cuts['pmdec_pad'] * cuts['sigma_pad'], color='blue', lw=0.6)
            ax[2].scatter(df['phi1'][masks['core']], df['pmdec'][masks['core']], s=2, alpha=0.1, c='k')

            ax[0].set_ylabel(r'$\phi_2$ [deg]')
            ax[1].set_ylabel(r'$\mu_{\alpha}\cos\delta$ [mas yr$^{-1}$]')
            ax[2].set_ylabel(r'$\mu_\delta$ [mas yr$^{-1}$]')
            ax[2].set_xlabel(r'$\phi_1$ [deg]')

            for axis in ax:
                plot_form(axis)

            plt.tight_layout()

        def _plot_isochrone_diagnostics(df, masks, diagnostics):
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ci = diagnostics['colour_index']
            abs_mag = diagnostics['abs_mag']
            selected = masks['all']
            core = masks['core']
            ax.scatter(ci[core & ~masks['iso']], abs_mag[core & ~masks['iso']], s=2, alpha=0.1, c='0.5', label='Rejected')
            ax.scatter(ci[selected], abs_mag[selected], s=4, alpha=0.4, c='tab:orange', label='Selected')
            # Isochrone in CMD space: x=(g-r), y=M_r
            ax.plot(colour_track_y, colour_track_x, color='red', lw=1.2, label='Isochrone')
            ax.axhline(cuts['mag_min'] - distance_mod, color='k', ls=':', label=rf'$r={cuts["mag_min"]}$')
            ax.axhline(cuts['mag_max'] - distance_mod, color='k', ls='--', label=rf'$r={cuts["mag_max"]}$')
            # Dynamic y-range: tighten to where stars exist, with small padding
            try:
                sel_mask = selected & np.isfinite(abs_mag) & np.isfinite(ci)
                if np.any(sel_mask):
                    bright = float(np.nanpercentile(abs_mag[sel_mask], 5))
                    bright = min(bright, float(np.nanmin(colour_track_x)))
                    faint_line = cuts['mag_min'] - distance_mod
                    y_lo = bright - 0.4
                    y_hi = faint_line + 0.2
                    ax.set_ylim(y_lo, y_hi)

                    ci_vals = np.asarray(ci[sel_mask], dtype=float)
                    ci_vals = ci_vals[np.isfinite(ci_vals)]
                    if ci_vals.size == 0:
                        ci_vals = np.array([0.0])
                    x_lo = float(np.nanpercentile(ci_vals, 5))
                    x_hi = float(np.nanpercentile(ci_vals, 95))
                    x_lo = min(x_lo, float(np.nanmin(colour_track_y))) - 0.1
                    x_hi = max(x_hi, float(np.nanmax(colour_track_y))) + 0.1
                    ax.set_xlim(x_lo, x_hi)
            except Exception:
                pass
            ax.invert_yaxis()
            ax.set_xlabel('Colour index (g-r)')
            ax.set_ylabel(r'$M_r$ [mag]')
            ax.legend(loc='best')
            plot_form(ax)
            # Annotate diagnostics to help sanity-check units
            try:
                med_rmag0 = np.nanmedian(df['rmag0'])
                ax.text(0.02, 0.06, f"DM={distance_mod:.2f}\nmed r0={med_rmag0:.2f}\nmed M_r={np.nanmedian(abs_mag):.2f}",
                        transform=ax.transAxes, fontsize=9, va='bottom', ha='left')
            except Exception:
                pass
            plt.tight_layout()

        def _plot_metallicity_diagnostics(df, masks):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            x_vals = df['gmag0'][masks['core']] - df['rmag0'][masks['core']]
            y_vals = df['rmag0'][masks['core']] - df['zmag0'][masks['core']]
            ax.scatter(x_vals, y_vals, s=4, alpha=0.1, c='0.5', label='Core')
            ax.scatter(df['gmag0'][masks['all']] - df['rmag0'][masks['all']], df['rmag0'][masks['all']] - df['zmag0'][masks['all']],
                       s=6, alpha=0.6, c='tab:orange', label='Selected')
            x_line = np.linspace(0.1, 0.8, 200)
            y_line = cuts['mp_slope'] * (x_line - cuts['mp_pivot']) + cuts['mp_offset']
            ax.plot(x_line, y_line, color='k', lw=1.2, label='Ting relation')
            ax.set_xlabel('g - r')
            ax.set_ylabel('r - z')
            ax.set_xlim(0.1, 0.8)
            ax.set_ylim(-0.1, 0.5)
            ax.legend(loc='best')
            plot_form(ax)
            plt.tight_layout()

        # Prepare Gaia catalogue and compute selection masks
        gaia_raw = table.Table.read(gaia_data_path)
        gaia_df = _prepare_catalog(gaia_raw, 'Gaia+DECaLS')
        gaia_df = _compute_magnitudes(gaia_df)
        gaia_df = _ensure_stream_coords(gaia_df, 'Gaia+DECaLS')
        gaia_df = _deduplicate_gaia(gaia_df)
        gaia_masks, gaia_diag = _build_selection_masks(gaia_df, 'Gaia+DECaLS')
        gaia_selected = gaia_df.loc[gaia_masks['all']].reset_index(drop=True)

        # Prepare DESI members depending on requested overlay
        desi_boxcut = _prepare_boxcut_members() if member_type == 'boxcut' else pd.DataFrame()
        mcmc_members = _prepare_mcmc_members() if member_type == 'mcmc' else pd.DataFrame()

        # Always highlight the DESI stream members (MCMC) in orange
        highlight_df = mcmc_members
        highlight_label = f'C-19 Members (p > {min_prob})'
        highlight_color = 'tab:orange'
        if highlight_df.empty:
            highlight_label = 'Members unavailable'

        # Split members by overlap with the Gaia selection
        mem_in = highlight_df.iloc[0:0].copy()
        mem_out = highlight_df.iloc[0:0].copy()
        members_in_gaia = 0
        members_not_in_gaia = 0
        if not highlight_df.empty:
            try:
                gaia_ids = set(gaia_selected['source_id']) if 'source_id' in gaia_selected.columns else set()
                if 'source_id' in highlight_df.columns and len(gaia_ids) > 0:
                    in_gaia_mask = highlight_df['source_id'].isin(gaia_ids)
                else:
                    in_gaia_mask = np.ones(len(highlight_df), dtype=bool)
            except Exception:
                in_gaia_mask = np.ones(len(highlight_df), dtype=bool)

            mem_in = highlight_df.loc[in_gaia_mask]
            mem_out = highlight_df.loc[~in_gaia_mask]
            members_in_gaia = int(len(mem_in))
            members_not_in_gaia = int(len(mem_out))

        if verbose:
            _plot_kinematic_diagnostics(gaia_df, gaia_masks)
            _plot_isochrone_diagnostics(gaia_df, gaia_masks, gaia_diag)
            _plot_metallicity_diagnostics(gaia_df, gaia_masks)

        # Completeness metrics
        metrics = {
            'gaia_total': int(len(gaia_df)),
            'gaia_core': int(np.sum(gaia_masks['core'])),
            'gaia_selected': int(len(gaia_selected)),
            'highlight_count': int(len(highlight_df)),
            'distance_kpc': float(distance_kpc),
            'distance_modulus': float(distance_mod),
            'members_in_gaia': members_in_gaia,
            'members_not_in_gaia': members_not_in_gaia,
        }

        if 'source_id' in gaia_selected.columns and 'source_id' in highlight_df.columns and len(gaia_selected) > 0:
            overlap = gaia_selected['source_id'].isin(highlight_df['source_id']).sum()
            metrics['highlight_overlap'] = int(overlap)
            metrics['highlight_fraction'] = overlap / len(gaia_selected)

        # Final on-sky figure
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        marker_size_bg = 70
        marker_size_hi = 80

        def _add_dotted_squares(ax, df, size_pts, color):
            if df is None or len(df) == 0:
                return
            x = np.asarray(df['phi1'], dtype=float)
            y = np.asarray(df['phi2'], dtype=float)
            if x.size == 0:
                return
            half = np.sqrt(size_pts) / 2.0
            offsets = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
            trans = ax.transData
            inv = trans.inverted()
            centers = trans.transform(np.column_stack([x, y]))
            squares = centers[:, None, :] + offsets[None, :, :]
            verts = inv.transform(squares.reshape(-1, 2)).reshape(len(x), 4, 2)
            segments = []
            for v in verts:
                segments.extend([[v[0], v[1]], [v[1], v[2]], [v[2], v[3]], [v[3], v[0]]])
            lc = LineCollection(segments, colors=color, linewidths=1.2, linestyles=(0, (3.0, 2.4)), zorder=6)
            ax.add_collection(lc)

        axes[0].scatter(
            gaia_selected['phi1'],
            gaia_selected['phi2'],
            marker='s',
            s=marker_size_bg,
            linewidths=0.0,
            facecolors='0.1',
            alpha=0.25,
            label='Gaia+DECaLS',
        )

        if not highlight_df.empty:
            if len(mem_in) > 0:
                axes[0].scatter(
                    mem_in['phi1'], mem_in['phi2'],
                    marker='s', facecolors='none', edgecolors=highlight_color,
                    linewidths=1.2, s=marker_size_hi, label=highlight_label, zorder=6,
                )
            if len(mem_out) > 0:
                _add_dotted_squares(axes[0], mem_out, marker_size_hi, highlight_color)

        # If an external orbit fit is available, overlay it; otherwise skip

        axes[1].scatter(
            gaia_selected['phi1'],
            gaia_selected['phi2'],
            marker='s',
            s=marker_size_bg,
            linewidths=0.0,
            facecolors='0.1',
            alpha=0.25,
        )

        # Note: no member overlays on the bottom panel by request

        axes[0].set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        axes[1].set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        axes[1].set_xlabel(r'$\phi_1$ [deg]', fontsize=14)

        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            plot_form(ax)
            ax.set_ylim(-cuts['phi2_pad'], cuts['phi2_pad'])
        axes[0].legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        plt.show()

        if return_data:
            return {
                'gaia_all': gaia_df,
                'gaia_selected': gaia_selected,
                'highlight': highlight_df,
                'metrics': metrics,
                'masks': gaia_masks,
            }


    def print_meds(self):
        """
        Print the median and error values for the MCMC parameters.

        From Joseph's stream_funtions.py
        """
        stream_dir = self.stream_run_directory
        mcmc_dict = np.load(stream_dir + '/mcmc_dict.npy', allow_pickle=True).item()

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

class streamCompare:
    """
    Class to handle comparison between two streams
    """
    def __init__(self, stream1, stream2, fr=None):
        self.fr = fr

        # rewrite their phi1s to be on the same frame
        if fr is not None:
            print('Putting streams onto same stream frame')
            stream1.stream_data['phi1'], stream1.stream_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(fr, np.array(stream1.stream_data['TARGET_RA'])*u.deg, np.array(stream1.stream_data['TARGET_DEC'])*u.deg)
            stream2.stream_data['phi1'], stream2.stream_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(fr, np.array(stream2.stream_data['TARGET_RA'])*u.deg, np.array(stream2.stream_data['TARGET_DEC'])*u.deg)
        self.stream1 = stream1
        self.stream2 = stream2
        self.stream1_data = stream1.stream_data
        self.stream2_data = stream2.stream_data
        self.vgsr_idx = stream1.vgsr_idx
        self.pmra_idx = stream1.pmra_idx
        self.pmdec_idx = stream1.pmdec_idx

    def on_sky(self, return_axes=False, **kwargs):
        stream1_label = kwargs.get('stream1_name', 'Stream 1')
        stream2_label = kwargs.get('stream2_name', 'Stream 2')
        title = kwargs.get('title', '')
        
        if self.fr is not None:
            print('original stream frames being used...')
        stream1_data = self.stream1_data
        stream2_data = self.stream2_data
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(stream1_data['phi1'], stream1_data['phi2'], marker=10, s=40, label=stream1_label + f' ({len(stream1_data)})', color='tab:blue')
        ax.scatter(stream2_data['phi1'], stream2_data['phi2'], marker=11, s=40, label=stream2_label + f' ({len(stream2_data)})', color='tab:orange')

        # phi1 min and max based off the minimum and maximum from combined streams
        phi1_min = min(np.min(stream1_data['phi1']), np.min(stream2_data['phi1']))
        phi1_max = max(np.max(stream1_data['phi1']), np.max(stream2_data['phi1']))
        phi2_min = min(np.min(stream1_data['phi2']), np.min(stream2_data['phi2']))
        phi2_max = max(np.max(stream1_data['phi2']), np.max(stream2_data['phi2']))
        ax.set_xlim(phi1_min - 1, phi1_max + 1)
        ax.set_ylim(phi2_min - 1, phi2_max + 1)
        ax.set_title(title)

        # axes labels
        ax.set_xlabel(r'$\phi_1$ [deg]', fontsize=14)
        ax.set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        ax.legend()

        plot_form(ax)

        if return_axes:
            return fig, ax
        else:
            plt.show()

    def sixD(self, return_axes=False, **kwargs):
        stream1_label = kwargs.get('stream1_name', 'Stream 1')
        stream2_label = kwargs.get('stream2_name', 'Stream 2')
        title = kwargs.get('title', '')
        if self.fr is not None:
            print('original stream frames being used...')
        stream1_data = self.stream1_data
        stream2_data = self.stream2_data
        fig, ax = plt.subplots(1, 6, figsize=(15, 3), sharey=True, sharex=True)
        ax[0].scatter(stream1_data['phi1'], stream1_data['phi2'], marker=10, s=40, label=stream1_label + f' ({len(stream1_data)})', color='tab:blue')
        ax[0].scatter(stream2_data['phi1'], stream2_data['phi2'], marker=11, s=40, label=stream2_label + f' ({len(stream2_data)})', color='tab:orange')
        ax[0].set_title(title)
        #plot respective spline   ax[1].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.vgsr_idx], self.spline_points_dict['spline_k']), color='b', lw=1, zorder=0, ls='--')
        ax[0].set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        ax[0].legend()

        
        ax[1].scatter(stream1_data['phi1'], stream1_data['VRAD'], marker=10, s=40, label=stream1_label + f' ({len(stream1_data)})', color='tab:blue')
        ax[1].scatter(stream2_data['phi1'], stream2_data['VRAD'], marker=11, s=40, label=stream2_label + f' ({len(stream2_data)})', color='tab:orange')
        ax[1].set_ylabel(r'$V_{rad}$ [km/s]', fontsize=14)

        for a in ax:
            plot_form(a)

        if return_axes:
            return fig, ax
        else:
            plt.show()

    def six6(self, addBackground=False, save_fig=False, fig_path=None, return_axes=False, residual=0, dist_mod_panel=False, cmap='magma_r', **kwargs):
        """
        Visualize the 6 panel plot comparing two streams similar to vis_6_panel but for stream comparison.

        New argument:
        - residual : int {0,1,2}
            0 -> plot raw quantities (default)
            1 -> plot residuals relative to stream1 spline (VGSR, PMRA, PMDEC)
            2 -> plot residuals relative to stream2 spline (VGSR, PMRA, PMDEC)
        - dist_mod_panel : bool, default False
            If True, include a panel showing distance modulus (mag) vs phi1.
            If False (default), omit the distance/distance-modulus panel entirely (reduces panel count by one).
            This is useful when distance modulus is not available for one or both streams.
        - cmap : str or Colormap, default 'magma_r'
            Colormap to use when coloring points by membership probability.
        """
        stream1_label = kwargs.get('stream1_name', 'Stream 1')
        stream2_label = kwargs.get('stream2_name', 'Stream 2')
        title = kwargs.get('title', '')
        
        stream1_data = self.stream1_data
        stream2_data = self.stream2_data
        
        # Decide subplot layout depending on whether the distance-modulus panel is requested
        include_dm_panel = bool(dist_mod_panel)
        n_pan = 6 if include_dm_panel else 5
        fig, ax = plt.subplots(n_pan, 1, figsize=(15, 2.5 * n_pan), sharex=True)
        # Index helpers
        phi2_ax_i, vgsr_ax_i, pmra_ax_i, pmdec_ax_i = 0, 1, 2, 3
        dm_ax_i = 4 if include_dm_panel else None
        feh_ax_i = 5 if include_dm_panel else 4
        
        # Prepare distance-modulus data only if we are going to plot that panel
        if include_dm_panel:
            # Guard against missing columns; if missing, gracefully hide the panel by reducing layout
            if ('dist_mod' not in stream1_data.colnames) or ('dist_mod' not in stream2_data.colnames):
                # Rebuild figure without the DM panel to avoid errors
                plt.close(fig)
                include_dm_panel = False
                n_pan = 5
                fig, ax = plt.subplots(n_pan, 1, figsize=(15, 2.5 * n_pan), sharex=True)
                phi2_ax_i, vgsr_ax_i, pmra_ax_i, pmdec_ax_i = 0, 1, 2, 3
                dm_ax_i = None
                feh_ax_i = 4
            else:
                # Distance modulus values and (symmetric) uncertainties (mag)
                dm1 = stream1_data['dist_mod']
                dm2 = stream2_data['dist_mod']
                # Prefer explicit plus/minus if present, else fall back to single error column
                if 'dist_mod_err_plus' in stream1_data.colnames and 'dist_mod_err_minus' in stream1_data.colnames:
                    dm1_err = (stream1_data['dist_mod_err_minus'], stream1_data['dist_mod_err_plus'])
                else:
                    dm1_err = stream1_data['dist_mod_err'] if 'dist_mod_err' in stream1_data.colnames else None
                if 'dist_mod_err_plus' in stream2_data.colnames and 'dist_mod_err_minus' in stream2_data.colnames:
                    dm2_err = (stream2_data['dist_mod_err_minus'], stream2_data['dist_mod_err_plus'])
                else:
                    dm2_err = stream2_data['dist_mod_err'] if 'dist_mod_err' in stream2_data.colnames else None

        # Decide if residuals should be computed and from which stream
        use_residual = residual in (1, 2)
        spline_ref = None
        nd_ref = None

        # initialize residual arrays to avoid UnboundLocalError when some branches don't assign them
        s1_vgsr = s1_pmra = s1_pmdec = s2_vgsr = s2_pmra = s2_pmdec = None

        if residual == 1:
            if hasattr(self.stream1, 'spline_points_dict') and hasattr(self.stream1, 'nested_dict'):
                spline_ref = self.stream1.spline_points_dict
                nd_ref = self.stream1.nested_dict
            else:
                print("Requested residual=1 but stream1 has no spline; falling back to residual=0")
                use_residual = False
        elif residual == 2:
            if hasattr(self.stream2, 'spline_points_dict') and hasattr(self.stream2, 'nested_dict'):
                spline_ref = self.stream2.spline_points_dict
                nd_ref = self.stream2.nested_dict
            else:
                print("Requested residual=2 but stream2 has no spline; falling back to residual=0")
                use_residual = False

        # helper to safely fetch exp_meds and errors
        def fetch_exp(nd, idx):
            # prefer exp_meds/exp_em/exp_ep if present, otherwise fallback to meds and safe defaults
            if nd is None:
                return None, None, None
            meds_source = nd.get('exp_meds', nd.get('meds'))
            if meds_source is None:
                return None, None, None
            med = meds_source[idx]
            em_list = nd.get('exp_em', [None] * len(nd.get('meds', meds_source)))
            ep_list = nd.get('exp_ep', [None] * len(nd.get('meds', meds_source)))
            em = em_list[idx] if em_list is not None else None
            ep = ep_list[idx] if ep_list is not None else None
            return med, em, ep

        # If using residuals and we have a valid reference spline, compute residual arrays for VGSR, PMRA, PMDEC for both streams
        if use_residual and (spline_ref is not None) and (nd_ref is not None):
            spline_points_ref = spline_ref['phi1_spline_points']
            kref = spline_ref['spline_k']

            def ref_val(phi_arr, meds_ind):
                return apply_spline(phi_arr, spline_points_ref, nd_ref['meds'][meds_ind], kref)

            # residuals for stream1
            s1_vgsr = stream1_data['VGSR'] - ref_val(stream1_data['phi1'], self.vgsr_idx)
            s1_pmra = stream1_data['PMRA'] - ref_val(stream1_data['phi1'], self.pmra_idx)
            s1_pmdec = stream1_data['PMDEC'] - ref_val(stream1_data['phi1'], self.pmdec_idx)
            # residuals for stream2
            s2_vgsr = stream2_data['VGSR'] - ref_val(stream2_data['phi1'], self.vgsr_idx)
            s2_pmra = stream2_data['PMRA'] - ref_val(stream2_data['phi1'], self.pmra_idx)
            s2_pmdec = stream2_data['PMDEC'] - ref_val(stream2_data['phi1'], self.pmdec_idx)
        else:
            # disable residual plotting if we couldn't prepare reference values
            use_residual = False

        # Panel 0: phi2 vs phi1 (unchanged)
        from matplotlib.colors import PowerNorm
        min_prob = float(kwargs.get('min_prob', 0.5))
        norm = PowerNorm(gamma=5, vmin=min_prob, vmax=1)
        cm = ax[phi2_ax_i].scatter(
            stream1_data['phi1'], stream1_data['phi2'], marker=10, s=40,
            c=stream1_data['stream_prob'], norm=norm, cmap=cmap, alpha=0.8, zorder=1,
            label=stream1_label + f' ({len(stream1_data)})'
        )
        ax[phi2_ax_i].scatter(
            stream2_data['phi1'], stream2_data['phi2'], marker=11, s=40,
            c=stream2_data['stream_prob'], norm=norm, cmap=cmap, alpha=0.8, zorder=1,
            label=stream2_label + f' ({len(stream2_data)})'
        )

        # Panel 1: VGSR vs phi1 (or residual)
        if use_residual:
            ax[vgsr_ax_i].scatter(
                stream1_data['phi1'], s1_vgsr, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[vgsr_ax_i].scatter(
                stream2_data['phi1'], s2_vgsr, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            # errorbars remain the same vertical scale (use original VRAD_ERR)
            ax[vgsr_ax_i].errorbar(
                stream1_data['phi1'], s1_vgsr,
                yerr=stream1_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[vgsr_ax_i].errorbar(
                stream2_data['phi1'], s2_vgsr,
                yerr=stream2_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        else:
            ax[vgsr_ax_i].scatter(
                stream1_data['phi1'], stream1_data['VGSR'], marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[vgsr_ax_i].scatter(
                stream2_data['phi1'], stream2_data['VGSR'], marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[vgsr_ax_i].errorbar(
                stream1_data['phi1'], stream1_data['VGSR'],
                yerr=stream1_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[vgsr_ax_i].errorbar(
                stream2_data['phi1'], stream2_data['VGSR'],
                yerr=stream2_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )

        # Panel 2: PMRA vs phi1 (or residual)
        if use_residual:
            ax[pmra_ax_i].scatter(
                stream1_data['phi1'], s1_pmra, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].scatter(
                stream2_data['phi1'], s2_pmra, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].errorbar(
                stream1_data['phi1'], s1_pmra,
                yerr=stream1_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmra_ax_i].errorbar(
                stream2_data['phi1'], s2_pmra,
                yerr=stream2_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        else:
            ax[pmra_ax_i].scatter(
                stream1_data['phi1'], stream1_data['PMRA'], marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].scatter(
                stream2_data['phi1'], stream2_data['PMRA'], marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].errorbar(
                stream1_data['phi1'], stream1_data['PMRA'],
                yerr=stream1_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmra_ax_i].errorbar(
                stream2_data['phi1'], stream2_data['PMRA'],
                yerr=stream2_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )

        # Panel 3: PMDEC vs phi1 (or residual)
        if use_residual:
            ax[pmdec_ax_i].scatter(
                stream1_data['phi1'], s1_pmdec, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].scatter(
                stream2_data['phi1'], s2_pmdec, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].errorbar(
                stream1_data['phi1'], s1_pmdec,
                yerr=stream1_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmdec_ax_i].errorbar(
                stream2_data['phi1'], s2_pmdec,
                yerr=stream2_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        else:
            ax[pmdec_ax_i].scatter(
                stream1_data['phi1'], stream1_data['PMDEC'], marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].scatter(
                stream2_data['phi1'], stream2_data['PMDEC'], marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].errorbar(
                stream1_data['phi1'], stream1_data['PMDEC'],
                yerr=stream1_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmdec_ax_i].errorbar(
                stream2_data['phi1'], stream2_data['PMDEC'],
                yerr=stream2_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        # Optional Panel: Distance modulus vs phi1
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].scatter(
                stream1_data['phi1'], dm1, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[dm_ax_i].scatter(
                stream2_data['phi1'], dm2, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            # Error bars can be symmetric value or tuple (minus, plus)
            try:
                ax[dm_ax_i].errorbar(
                    stream1_data['phi1'], dm1, yerr=dm1_err,
                    capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
                )
            except Exception:
                pass
            try:
                ax[dm_ax_i].errorbar(
                    stream2_data['phi1'], dm2, yerr=dm2_err,
                    capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
                )
            except Exception:
                pass
        
        # Final Panel: FEH vs phi1
        ax[feh_ax_i].scatter(
            stream1_data['phi1'], stream1_data['FEH'], marker=10, s=40,
            c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
        )
        ax[feh_ax_i].scatter(
            stream2_data['phi1'], stream2_data['FEH'], marker=11, s=40,
            c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
        )
        ax[feh_ax_i].errorbar(
            stream1_data['phi1'], stream1_data['FEH'],
            yerr=stream1_data['FEH_ERR'],
            capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
        )
        ax[feh_ax_i].errorbar(
            stream2_data['phi1'], stream2_data['FEH'],
            yerr=stream2_data['FEH_ERR'],
            capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
        )
        
        # Add spline tracks for both streams if available (for visual reference)
        phi1_min = min(np.min(stream1_data['phi1']), np.min(stream2_data['phi1']))
        phi1_max = max(np.max(stream1_data['phi1']), np.max(stream2_data['phi1']))
        x_arr = np.linspace(phi1_min - 1, phi1_max + 1, 100)

        # show cbar with unified normalization/style
        from matplotlib.colors import PowerNorm
        norm = PowerNorm(gamma=5, vmin=0.5, vmax=1)
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # Adjust the position of the colorbar
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar.set_label('Membership Probability', fontsize=16)
        try:
            ticks = [min_prob]
            for t in (0.9, 0.95, 1.0):
                if t > min_prob:
                    ticks.append(t)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])
        except Exception:
            pass
        cbar.ax.tick_params(labelsize=10)

        # Stream 1 splines (blue)
        spd1 = self.stream1.spline_points_dict
        nd1 = self.stream1.nested_dict
        spd2 = self.stream2.spline_points_dict
        nd2 = self.stream2.nested_dict

        # collect legend handles for panel 2 (index 1)
        spline_leg_handles = []

        if hasattr(self.stream1, 'spline_points_dict') and hasattr(self.stream1, 'nested_dict'):
            if use_residual and residual == 1:
                # reference is stream1: show spline points with errorbars centered at 0
                for panel_idx, med_idx in zip([1, 2, 3], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    _, em, ep = fetch_exp(nd1, med_idx)
                    try:
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(
                            spd1['phi1_spline_points'], np.zeros_like(spd1['phi1_spline_points']),
                            yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9
                        )
                    except Exception:
                        ax[panel_idx].plot(
                            spd1['phi1_spline_points'], np.zeros_like(spd1['phi1_spline_points']), 'o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9
                        )
            elif not use_residual:
                # normal plotting of stream1 spline when not reference residual (or residual==0)
                h1, = ax[vgsr_ax_i].plot(
                    x_arr,
                    apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.vgsr_idx], spd1['spline_k']),
                    color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.8, label=f"{stream1_label} spline"
                )
                spline_leg_handles.append(h1)
                ax[pmra_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmra_idx], spd1['spline_k']),
                    color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.8
                )
                ax[pmdec_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmdec_idx], spd1['spline_k']),
                    color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.8
                )

            # spline points & errorbars for stream1 (if not reference plotted as zeros)
            for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                med_vals, em, ep = fetch_exp(nd1, med_idx)
                try:
                    if residual == 1 and use_residual:
                        # when ref is stream1 we already plotted zeros+errors; skip plotting med-centered points to avoid confusion
                        continue
                    if med_vals is not None:
                        ax[panel_idx].errorbar(
                            spd1['phi1_spline_points'], med_vals,
                            yerr=(em, ep) if (em is not None or ep is not None) else None,
                            capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.8
                        )
                except Exception:
                    pass

        # Stream 2 splines (orange)
        if hasattr(self.stream2, 'spline_points_dict') and hasattr(self.stream2, 'nested_dict'):
            spd2 = self.stream2.spline_points_dict
            nd2 = self.stream2.nested_dict

            if residual == 2 and use_residual:
                # reference is stream2: show spline points with errorbars centered at 0 for panels 1-3
                for panel_idx, med_idx in zip([1, 2, 3], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    _, em, ep = fetch_exp(nd2, med_idx)
                    try:
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(
                            spd2['phi1_spline_points'], np.zeros_like(spd2['phi1_spline_points']),
                            yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9
                        )
                    except Exception:
                        ax[panel_idx].plot(
                            spd2['phi1_spline_points'], np.zeros_like(spd2['phi1_spline_points']), 'o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9
                        )
            elif not use_residual:
                # normal plotting of stream2 spline when not reference residual (or residual==0)
                h2, = ax[vgsr_ax_i].plot(
                    x_arr,
                    apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.vgsr_idx], spd2['spline_k']),
                    color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.8, label=f"{stream2_label} spline"
                )
                spline_leg_handles.append(h2)
                ax[pmra_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmra_idx], spd2['spline_k']),
                    color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.8
                )
                ax[pmdec_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmdec_idx], spd2['spline_k']),
                    color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.8
                )
                # also plot stream2 spline points with error bars (medians) in non-residual mode
                for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    med_vals, em, ep = fetch_exp(nd2, med_idx)
                    try:
                        if med_vals is not None:
                            ax[panel_idx].errorbar(
                                spd2['phi1_spline_points'], med_vals,
                                yerr=(em, ep) if (em is not None or ep is not None) else None,
                                capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9
                            )
                    except Exception:
                        pass

            # If reference is stream1 and we're showing residuals, plot stream2 spline as residual to reference
            if residual == 1 and use_residual:
                try:
                    y_vgsr = apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.vgsr_idx], spd2['spline_k']) - apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.vgsr_idx], spd1['spline_k'])
                    y_pmra = apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmra_idx], spd2['spline_k']) - apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmra_idx], spd1['spline_k'])
                    y_pmdec = apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmdec_idx], spd2['spline_k']) - apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmdec_idx], spd1['spline_k'])
                    h_diff21, = ax[vgsr_ax_i].plot(x_arr, y_vgsr, color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9, label=f"{stream2_label} - {stream1_label}")
                    spline_leg_handles.append(h_diff21)
                    ax[pmra_ax_i].plot(x_arr, y_pmra, color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9)
                    ax[pmdec_ax_i].plot(x_arr, y_pmdec, color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9)
                except Exception:
                    pass
                # plot the spline points for stream2 as residuals (meds - ref_evaluated_at_spd2_phi)
                for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    med_vals, em, ep = fetch_exp(nd2, med_idx)
                    try:
                        if med_vals is None:
                            continue
                        ref_at_points = apply_spline(spd2['phi1_spline_points'], spd1['phi1_spline_points'], nd1['meds'][med_idx], spd1['spline_k'])
                        ypts = med_vals - ref_at_points
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(spd2['phi1_spline_points'], ypts, yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9)
                    except Exception:
                        pass

            # If reference is stream2 and we're showing residuals, plot stream1 spline as residual to stream2
            if residual == 2 and use_residual:
                try:
                    y_vgsr = apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.vgsr_idx], spd1['spline_k']) - apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.vgsr_idx], spd2['spline_k'])
                    y_pmra = apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmra_idx], spd1['spline_k']) - apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmra_idx], spd2['spline_k'])
                    y_pmdec = apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmdec_idx], spd1['spline_k']) - apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmdec_idx], spd2['spline_k'])
                    h_diff12, = ax[vgsr_ax_i].plot(x_arr, y_vgsr, color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9, label=f"{stream1_label} - {stream2_label}")
                    spline_leg_handles.append(h_diff12)
                    ax[pmra_ax_i].plot(x_arr, y_pmra, color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9)
                    ax[pmdec_ax_i].plot(x_arr, y_pmdec, color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9)
                except Exception:
                    pass
                # plot the spline points for stream1 as residuals (meds - ref_evaluated_at_spd1_phi)
                for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    med_vals, em, ep = fetch_exp(nd1, med_idx)
                    try:
                        if med_vals is None:
                            continue
                        ref_at_points = apply_spline(spd1['phi1_spline_points'], spd2['phi1_spline_points'], nd2['meds'][med_idx], spd2['spline_k'])
                        ypts = med_vals - ref_at_points
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(spd1['phi1_spline_points'], ypts, yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9)
                    except Exception:
                        pass

        # Always show FEH spline (non-residual) for both streams if available
        # Stream1 FEH spline + spline points
        if hasattr(self.stream1, 'spline_points_dict') and hasattr(self.stream1, 'nested_dict'):
            spd1 = self.stream1.spline_points_dict
            nd1 = self.stream1.nested_dict
            try:
                ax[feh_ax_i].plot(x_arr, apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][3], spd1['spline_k']),
                       color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9)
                med_vals, em, ep = fetch_exp(nd1, 3)
                if med_vals is not None:
                    ax[feh_ax_i].errorbar(spd1['phi1_spline_points'], med_vals,
                           yerr=(em, ep) if (em is not None or ep is not None) else None,
                           capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9)
            except Exception:
                pass

        # Stream2 FEH spline + spline points
        if hasattr(self.stream2, 'spline_points_dict') and hasattr(self.stream2, 'nested_dict'):
            spd2 = self.stream2.spline_points_dict
            nd2 = self.stream2.nested_dict
            try:
                ax[feh_ax_i].plot(x_arr, apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][3], spd2['spline_k']),
                       color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9)
                med_vals, em, ep = fetch_exp(nd2, 3)
                if med_vals is not None:
                    ax[feh_ax_i].errorbar(spd2['phi1_spline_points'], med_vals,
                           yerr=(em, ep) if (em is not None or ep is not None) else None,
                           capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9)
            except Exception:
                pass

        # Draw a dashed zero line for residual plots (label only once on panel 2 for legend)
        if use_residual:
            zero_color = 'tab:blue' if residual == 1 else 'tab:orange'
            zero_label = (f"Ref: {stream1_label} spline" if residual == 1 else f"Ref: {stream2_label} spline")
            # label on panel 2
            h0 = ax[vgsr_ax_i].axhline(0, color=zero_color, lw=1, zorder=2, alpha=0.9, ls='--', label=zero_label)
            spline_leg_handles.append(h0)
            # repeat on other residual panels (no label)
            for panel_idx in [pmra_ax_i, pmdec_ax_i]:
                ax[panel_idx].axhline(0, color=zero_color, lw=1, zorder=2, alpha=0.9, ls='--')

        # Set labels (change to Delta labels when residuals applied)
        ax[phi2_ax_i].set_ylabel(r'$\phi_2$ (deg)', fontsize=14)
        if use_residual:
            ax[vgsr_ax_i].set_ylabel(r'$\Delta V_{GSR}$ (km/s)', fontsize=14)
            ax[pmra_ax_i].set_ylabel(r'$\Delta \mu_{\alpha}$ (mas/yr)', fontsize=14)
            ax[pmdec_ax_i].set_ylabel(r'$\Delta \mu_{\delta}$ (mas/yr)', fontsize=14)
        else:
            ax[vgsr_ax_i].set_ylabel(r'$V_{GSR}$ (km/s)', fontsize=14)
            ax[pmra_ax_i].set_ylabel(r'$\mu_{\alpha}$ (mas/yr)', fontsize=14)
            ax[pmdec_ax_i].set_ylabel(r'$\mu_{\delta}$ (mas/yr)', fontsize=14)
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].set_ylabel('Distance modulus (mag)', fontsize=14)
        ax[feh_ax_i].set_ylabel(r'[Fe/H]', fontsize=14)
        ax[feh_ax_i].set_xlabel(r'$\phi_1$ (deg)', fontsize=14)
        
        # Set title and legends
        if title:
            ax[phi2_ax_i].set_title(title, fontsize=16)
        ax[phi2_ax_i].legend()
        # add legend for splines on the second panel if any handles were collected
        if len(spline_leg_handles) > 0:
            ax[vgsr_ax_i].legend(handles=spline_leg_handles, fontsize=9, frameon=False)
        
        # Set xlim based on combined phi1 values
        phi1_min = min(np.min(stream1_data['phi1']), np.min(stream2_data['phi1']))
        phi1_max = max(np.max(stream1_data['phi1']), np.max(stream2_data['phi1']))
        ax[phi2_ax_i].set_xlim(phi1_min - 2, phi1_max + 2)
        
        # Set ylims based on combined stream data y values (for residuals use combined residual arrays if available)
        phi2_combined = np.concatenate([stream1_data['phi2'], stream2_data['phi2']])
        if use_residual:
            vgsr_combined = np.concatenate([s1_vgsr, s2_vgsr])
            pmra_combined = np.concatenate([s1_pmra, s2_pmra])
            pmdec_combined = np.concatenate([s1_pmdec, s2_pmdec])
        else:
            vgsr_combined = np.concatenate([stream1_data['VGSR'], stream2_data['VGSR']])
            pmra_combined = np.concatenate([stream1_data['PMRA'], stream2_data['PMRA']])
            pmdec_combined = np.concatenate([stream1_data['PMDEC'], stream2_data['PMDEC']])
        feh_combined = np.concatenate([stream1_data['FEH'], stream2_data['FEH']])
        
        ax[phi2_ax_i].set_ylim(np.min(phi2_combined) - 2, np.max(phi2_combined) + 2)
        ax[vgsr_ax_i].set_ylim(np.min(vgsr_combined) - 10, np.max(vgsr_combined) + 10)
        ax[pmra_ax_i].set_ylim(np.min(pmra_combined) - 1, np.max(pmra_combined) + 1)
        ax[pmdec_ax_i].set_ylim(np.min(pmdec_combined) - 1, np.max(pmdec_combined) + 1)
        ax[feh_ax_i].set_ylim(np.min(feh_combined) - 0.2, np.max(feh_combined) + 0.2)

        # Set scale and nice limits for the DM panel
        if include_dm_panel and dm_ax_i is not None:
            # No log scale for magnitudes; set margins based on data ranges if available
            try:
                dm_combined = np.concatenate([np.atleast_1d(dm1), np.atleast_1d(dm2)])
                pad = 0.2 if np.isfinite(dm_combined).all() else 0.0
                ymin, ymax = np.nanmin(dm_combined) - pad, np.nanmax(dm_combined) + pad
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ax[dm_ax_i].set_ylim(ymin, ymax)
            except Exception:
                pass
        
        # Apply formatting to all axes
        for a in ax:
            plot_form(a)
        
        # Add background if requested (would need all_memberships data)
        if addBackground:
            print("Background plotting not implemented for stream comparison - requires all_memberships data")
        
        # Set rasterization for better PDF output
        for ax_curr in plt.gcf().get_axes():
            for artist in ax_curr.get_children():
                artist.set_rasterized(True)
        
        # Save figure if requested
        if save_fig:
            if fig_path is None:
                fig_path = 'figures_draft/postmcmc_6panel_comparison.pdf'
            plt.savefig(fig_path, bbox_inches='tight', dpi=600)
        
        if return_axes:
            return fig, ax
        else:
            plt.show()
