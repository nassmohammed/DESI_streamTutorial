import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stream_functions as stream_funcs
import scipy
from scipy.interpolate import interp1d
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import astropy.units as u
from scipy.optimize import minimize
from astropy.table import Table
import pandas as pd

def sf_in_desi(SoI_streamfinder, desi_dropped_vals):
        gaia_source_ids = SoI_streamfinder.keys()[0]
        confirmed_sf_and_desi = pd.merge(SoI_streamfinder.drop_duplicates(subset=[gaia_source_ids]), desi_dropped_vals.drop_duplicates(subset=['SOURCE_ID']), left_on = gaia_source_ids, right_on = 'SOURCE_ID', how = 'inner')
        confirmed_sf_and_desi.dropna(inplace = True)
        print(f"Number of stars in SF: {len(SoI_streamfinder[gaia_source_ids])}, Number of DESI and SF stars: {len(confirmed_sf_and_desi['SOURCE_ID'])}")
        return confirmed_sf_and_desi

def ra_dec_dist_cut(SoI_galstream, SoI_streamfinder, desi_dropped_vals):
        min_dist = np.min(SoI_galstream.track.distance.value)
        sf_ra_name = SoI_streamfinder.keys()[1]
        sf_dec_name = SoI_streamfinder.keys()[2]
        catalogue_ra = SoI_streamfinder[sf_ra_name]
        catalogue_dec = SoI_streamfinder[sf_dec_name]


        desi_SoI_mask = stream_funcs.threeD_max_min_mask(desi_dropped_vals['TARGET_RA'], desi_dropped_vals['TARGET_DEC'], desi_dropped_vals['PARALLAX'], desi_dropped_vals['PARALLAX_ERROR'],\
        catalogue_ra, catalogue_dec, min_dist, 10, 10)
        return desi_dropped_vals[desi_SoI_mask]

def betw(y_data, fit_data, delta):
    return (y_data > fit_data - delta) & (y_data < fit_data + delta)

def kin_cut(SoI_galstream, SoI_streamfinder, sf_desi, desi_data, feh, gal_phi1, gal_phi2, plot=False):
        phi1_array = np.linspace(desi_data['phi1'].min(), desi_data['phi1'].max(), 1000)

        pmra_fit = interp1d(gal_phi1, SoI_galstream.track.pm_ra_cosdec, kind='linear', fill_value='extrapolate')
        pmdec_fit = interp1d(gal_phi1, SoI_galstream.track.pm_dec, kind='linear', fill_value='extrapolate')
        phi_fit = scipy.interpolate.UnivariateSpline(gal_phi1, gal_phi2)


        pmra_cut_width = 2*np.std(sf_desi['PMRA'])
        pmdec_cut_width = 2*np.std(sf_desi['PMDEC'])
        feh_cut_width = 2*np.std(sf_desi['FEH'])
        phi2_cut_width = 2*np.std(sf_desi['phi2'])

        pmra_mask = betw(desi_data['PMRA'], pmra_fit(desi_data['phi1']), pmra_cut_width)
        pmdec_mask = betw(desi_data['PMDEC'], pmdec_fit(desi_data['phi1']), pmdec_cut_width)
        feh_mask = betw(desi_data['FEH'], np.ones_like(pmdec_fit)*feh, feh_cut_width)

        
        phi2_mask = stream_funcs.betw(desi_data['phi2'], phi_fit(desi_data['phi1']), phi2_cut_width) &\
        (desi_data['phi1'] < np.max(gal_phi1) + phi2_cut_width) & (desi_data['phi1'] > np.min(gal_phi1) - phi2_cut_width)
        if plot==True:
            fig, ax = plt.subplots(4, 1, figsize=(10, 10))
            for a in ax:
                a.grid(ls='-.', alpha=0.2, zorder=0)
                a.tick_params(direction='in')

            ax[0].scatter(desi_data['phi1'],desi_data['PMRA'], s=1, c='k', alpha=0.005, label='DESI data')
            ax[0].scatter(desi_data['phi1'][pmra_mask],desi_data['PMRA'][pmra_mask], s=10, c='b', alpha=0.5, label='This Cut')
            ax[0].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['PMRA'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10, label='Kin Cut')
            ax[0].scatter(gal_phi1, SoI_galstream.track.pm_ra_cosdec.value, c='y', s=5, alpha=0.2, label='galstream')
            ax[0].set_ylim(np.nanmin(SoI_galstream.track.pm_ra_cosdec.value)-10, np.nanmax(SoI_galstream.track.pm_ra_cosdec.value)+20)
            ax[0].legend(loc=1, prop={'size': 8}, ncol=4)
            ax[0].set_ylabel(r'PMRA')

            ax[1].scatter(desi_data['phi1'],desi_data['PMDEC'], s=1, c='k', alpha=0.005, label='DESI data')
            ax[1].scatter(desi_data['phi1'][pmdec_mask],desi_data['PMDEC'][pmdec_mask], s=10, c='b', alpha=0.5, label='This Cut')
            ax[1].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['PMDEC'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10, label='Kin Cut')
            ax[1].scatter(gal_phi1, SoI_galstream.track.pm_dec.value, c='y', s=5, alpha=0.2, label='galstream')
            ax[1].set_ylim(np.nanmin(SoI_galstream.track.pm_dec.value)-10, np.nanmax(SoI_galstream.track.pm_dec.value)+20)
            ax[1].set_ylabel(r'PMDEC')

            ax[2].scatter(desi_data['phi1'],desi_data['phi2'], s=1, c='k', alpha=0.005, label='DESI data')
            ax[2].scatter(desi_data['phi1'][phi2_mask],desi_data['phi2'][phi2_mask], s=10, c='b', alpha=0.5, label='This Cut')
            ax[2].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['phi2'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10, label='Kin Cut')
            ax[2].scatter(gal_phi1, gal_phi2, c='y', s=5, alpha=0.2, label='galstream')
            ax[2].set_ylim(np.nanmin(gal_phi2)-10, np.nanmax(gal_phi2)+20)
            ax[2].set_ylabel(r'$\phi_2$')

            ax[3].scatter(desi_data['phi1'],desi_data['FEH'], s=1, c='k', alpha=0.005)
            ax[3].scatter(desi_data['phi1'][feh_mask],desi_data['FEH'][feh_mask], s=10, c='b', alpha=0.5)
            ax[3].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['FEH'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10)
            ax[3].scatter(desi_data['phi1'], np.ones_like(desi_data['phi1'])*feh, c='pink', s=1, alpha=0.2, label='SF3 FeH')
            #ax[3].set_ylim(np.nanmin(gal_phi2)-10, np.nanmax(gal_phi2)+20)
            ax[3].set_ylabel(r'[Fe/H]')
            ax[3].legend(loc=1, prop={'size': 8}, ncol=4)
            ax[-1].set_xlabel(r'$\phi_1$')

        sf_left = sf_in_desi(SoI_streamfinder, desi_data[pmra_mask & pmdec_mask & feh_mask & phi2_mask])

        return desi_data[pmra_mask & pmdec_mask & feh_mask & phi2_mask], sf_left

def vgsr_cut(ovrad, sf_left, plot=False):
        phi1_array = np.linspace(sf_left['phi1'].min(), sf_left['phi1'].max(), 1000)
        vrad_cut_width = 1*np.std(sf_left['VRAD'])
        vrad_mask = betw(sf_left['VRAD'], ovrad(sf_left['phi1']), vrad_cut_width)
        if plot==True:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.scatter(sf_left['phi1'],sf_left['VRAD'], s=1, c='k', alpha=1, label='Velocity Cut')
            ax.scatter(sf_left['phi1'][vrad_mask],sf_left['VRAD'][vrad_mask], s=10, c='b', alpha=1, label='Remaining', marker='d')
            ax.plot(phi1_array, ovrad(phi1_array), c='r', ls=":", lw=2, alpha=1, label='orbit')
            #ax.set_ylim(np.nanmin(ovrad(phi1_array))-10, np.nanmax(ovrad(phi1_array))+20)
            ax.legend(loc=1, prop={'size': 8}, ncol=4)
            ax.set_ylabel(r'VRAD')
            ax.set_xlabel(r'$\phi_1$')
            ax.grid(ls='-.', alpha=0.2, zorder=0)
            ax.tick_params(direction='in')
        return sf_left[vrad_mask]

def kin_cut2(ointerps, SoI_streamfinder, sf_desi, desi_data, feh, plot=False):
        phi1_array = np.linspace(desi_data['phi1'].min(), desi_data['phi1'].max(), 1000)

        pmra_fit = ointerps['pmra']
        pmdec_fit = ointerps['pmdec']
        phi_fit = ointerps['phi2']


        pmra_cut_width = 2*np.std(sf_desi['PMRA'])
        pmdec_cut_width = 2*np.std(sf_desi['PMDEC'])
        feh_cut_width = 2*np.std(sf_desi['FEH'])
        phi2_cut_width = 2*np.std(sf_desi['phi2'])

        pmra_mask = betw(desi_data['PMRA'], pmra_fit(desi_data['phi1']), pmra_cut_width)
        pmdec_mask = betw(desi_data['PMDEC'], pmdec_fit(desi_data['phi1']), pmdec_cut_width)
        feh_mask = betw(desi_data['FEH'], np.ones_like(pmdec_fit)*feh, feh_cut_width)

        
        phi2_mask = stream_funcs.betw(desi_data['phi2'], phi_fit(desi_data['phi1']), phi2_cut_width) 

        if plot==True:
            fig, ax = plt.subplots(4, 1, figsize=(10, 10))
            for a in ax:
                a.grid(ls='-.', alpha=0.2, zorder=0)
                a.tick_params(direction='in')

            ax[0].scatter(desi_data['phi1'],desi_data['PMRA'], s=1, c='k', alpha=0.005, label='DESI data')
            ax[0].scatter(sf_desi['phi1'],sf_desi['PMRA'], s=10, c='r', alpha=1, label='SF data', zorder=10)
            ax[0].scatter(desi_data['phi1'][pmra_mask],desi_data['PMRA'][pmra_mask], s=10, c='b', alpha=0.5, label='This Cut')
            ax[0].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['PMRA'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10, label='Kin Cut')
            ax[0].plot(phi1_array, pmra_fit(phi1_array), c='r', ls=":", lw=2, alpha=1, label='orbit')
            ax[0].set_ylim(np.nanmin(pmra_fit(phi1_array))-10, np.nanmax(pmra_fit(phi1_array))+20)
            ax[0].legend(loc=1, prop={'size': 8}, ncol=4)
            ax[0].set_ylabel(r'PMRA')

            ax[1].scatter(desi_data['phi1'],desi_data['PMDEC'], s=1, c='k', alpha=0.005, label='DESI data')
            ax[1].scatter(sf_desi['phi1'],sf_desi['PMDEC'], s=10, c='r', alpha=1, label='SF data', zorder=10)
            ax[1].scatter(desi_data['phi1'][pmdec_mask],desi_data['PMDEC'][pmdec_mask], s=10, c='b', alpha=0.5, label='This Cut')
            ax[1].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['PMDEC'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10, label='Kin Cut')
            ax[1].plot(phi1_array, pmdec_fit(phi1_array), c='r', ls=":", lw=2, alpha=1, label='orbit')
            ax[1].set_ylim(np.nanmin(pmdec_fit(phi1_array))-10, np.nanmax(pmdec_fit(phi1_array))+20)
            ax[1].set_ylabel(r'PMDEC')

            ax[2].scatter(desi_data['phi1'],desi_data['phi2'], s=1, c='k', alpha=0.005, label='DESI data')
            ax[2].scatter(sf_desi['phi1'],sf_desi['phi2'], s=10, c='r', alpha=1, label='SF data', zorder=10)
            ax[2].scatter(desi_data['phi1'][phi2_mask],desi_data['phi2'][phi2_mask], s=10, c='b', alpha=0.5, label='This Cut')
            ax[2].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['phi2'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10, label='Kin Cut')
            ax[2].plot(phi1_array, phi_fit(phi1_array), c='r', ls=":", lw=2, alpha=1, label='orbit')
            ax[2].set_ylim(np.nanmin(phi_fit(phi1_array))-10, np.nanmax(phi_fit(phi1_array))+20)
            ax[2].set_ylabel(r'$\phi_2$')

            ax[3].scatter(desi_data['phi1'],desi_data['FEH'], s=1, c='k', alpha=0.005)
            ax[3].scatter(sf_desi['phi1'],sf_desi['FEH'], s=10, c='r', alpha=1, label='SF data', zorder=10)
            ax[3].scatter(desi_data['phi1'][feh_mask],desi_data['FEH'][feh_mask], s=10, c='b', alpha=0.5)
            ax[3].scatter(desi_data['phi1'][pmra_mask & pmdec_mask & feh_mask & phi2_mask],desi_data['FEH'][pmra_mask & pmdec_mask & feh_mask & phi2_mask], c='g', s=10)
            ax[3].scatter(desi_data['phi1'], np.ones_like(desi_data['phi1'])*feh, c='pink', s=1, alpha=1, label='SF3 FeH')
            #ax[3].set_ylim(np.nanmin(gal_phi2)-10, np.nanmax(gal_phi2)+20)
            ax[3].set_ylabel(r'[Fe/H]')
            ax[3].legend(loc=1, prop={'size': 8}, ncol=4)
            ax[-1].set_xlabel(r'$\phi_1$')

        sf_left = sf_in_desi(SoI_streamfinder, desi_data[pmra_mask & pmdec_mask & feh_mask & phi2_mask])

        return desi_data[pmra_mask & pmdec_mask & feh_mask & phi2_mask], sf_left

def iso_mask(o1interps, SoI_galstream, SoI_streamfinder, desi_data, feh, gal_phi1, colour_wiggle, plot=False, sf_only=False):
    mass_fraction_guess = 0.0181 * 10 ** feh
    dotter_mfs = ["0.00006", "0.00007", "0.00009", "0.00010", "0.00011", "0.00013", "0.00014", "0.00016", "0.00017", "0.00019",
                   "0.00021", "0.00024", "0.00028", "0.00032", "0.00037", "0.00042", "0.00049", "0.00057", "0.00063", "0.00072",
                   "0.00082", "0.00093", "0.00108", "0.00124", "0.00144", "0.00166", "0.00189", "0.00213", "0.00242", "0.00276",
                   "0.00316", "0.00363", "0.00417"]
    # Convert list of strings to a numpy array of floats for calculations
    dotter_mfs_float = np.array([float(val) for val in dotter_mfs])
    idx = np.argmin(np.abs(dotter_mfs_float - mass_fraction_guess))
    dotter_num = dotter_mfs_float[idx]
    # Format dotter_num fully and remove trailing zeros after the decimal point
    formatted_dotter = format(dotter_num, '.10f').rstrip('0').rstrip('.')
    isochrone_path = f'../data/dotter/iso_a12.5_z{formatted_dotter}.dat'
    dotter_mp = np.loadtxt(isochrone_path)
    dotter_g_mp = dotter_mp[:,6]
    dotter_r_mp = dotter_mp[:,7]

    desi_distances = np.array(o1interps['dist'](desi_data.loc[:,'phi1']))
    desi_ebv = np.array(desi_data['EBV'].values)
    desi_g_flux, desi_r_flux = np.array(desi_data['FLUX_G'].values), np.array(desi_data['FLUX_R'].values)
    desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(desi_ebv, desi_g_flux, desi_r_flux, desi_distances)

    g_r_color_dif = dotter_g_mp - dotter_r_mp
    sorted_indices = np.argsort(dotter_r_mp)
    sorted_dotter_r_mp = dotter_r_mp[sorted_indices]
    g_r_color_dif = g_r_color_dif[sorted_indices]

    isochrone_fit = scipy.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0)

    # Cut around the isochrone by the amount specified in colour_wiggle
    isochrone_cut = stream_funcs.betw(desi_colour_index, isochrone_fit(desi_abs_mag), colour_wiggle) 

    #streamfinder same
    if sf_only == False:
        confirmed_sf_and_desi = sf_in_desi(SoI_streamfinder, desi_data)
        sf_desi_distances = stream_funcs.dist_mod_to_dist(confirmed_sf_and_desi['dist_mod'])
        sf_desi_ebv = np.array(confirmed_sf_and_desi['EBV'].values)
        sf_desi_g_flux, sf_desi_r_flux = np.array(confirmed_sf_and_desi['FLUX_G'].values), np.array(confirmed_sf_and_desi['FLUX_R'].values)
        sf_desi_colour_index, sf_desi_abs_mag, sf_desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(sf_desi_ebv, sf_desi_g_flux, sf_desi_r_flux, sf_desi_distances)
        sf_desi_isochrone_cut = stream_funcs.betw(sf_desi_colour_index, isochrone_fit(sf_desi_abs_mag), colour_wiggle)
        s1 = 10; alpha1=1
    else:
        s1=10; alpha1=1
    




    if plot == True:

        fig, ax = plt.subplots(1,1, figsize=(6,6))

        ax.scatter(desi_colour_index[~isochrone_cut], desi_abs_mag[~isochrone_cut], s=s1, alpha=alpha1, c='k')
        ax.scatter(desi_colour_index[isochrone_cut], desi_abs_mag[isochrone_cut], s=s1, alpha=alpha1, color='lightblue')


        legend_handles = [
        plt.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Cut DESI', alpha=1),
        plt.Line2D([], [], color='lightblue', marker='.', linestyle='None', markersize=10, label='DESI', alpha=1)
        ]
        # plt.scatter(color_index_br, abs_r_mag, s=1, alpha=0.5, c='r')
        ax.plot(dotter_g_mp - dotter_r_mp, dotter_r_mp, c='b')
        ax.plot(dotter_g_mp - dotter_r_mp - colour_wiggle, dotter_r_mp, c='b')
        ax.plot(dotter_g_mp - dotter_r_mp + colour_wiggle, dotter_r_mp, c='b')
        ax.plot(isochrone_fit(dotter_r_mp), dotter_r_mp, c='r')
        if sf_only==False:
            ax.scatter(sf_desi_colour_index[sf_desi_isochrone_cut], sf_desi_abs_mag[sf_desi_isochrone_cut], s=50, alpha=1, facecolor='g', edgecolor = 'green', marker='*')
            ax.scatter(sf_desi_colour_index[~sf_desi_isochrone_cut], sf_desi_abs_mag[~sf_desi_isochrone_cut], s=50, alpha=1, facecolor='None', edgecolor = 'green', marker='*')
            #ax[0].scatter(sf_, sf_desi_abs_mag[sf_and_desi_trunc_cuts & sf_desi_3D_mask & sf_desi_phi2_cut], s=50, alpha=1, facecolor='green', edgecolor = 'green',marker='*')
        # Set these custom handles in the legend
        ax.legend(handles=legend_handles, ncol=2, loc=3, prop={'size': 10})

        ax.set_xlabel('g-r',fontsize=15)
        ax.set_ylabel('$M_r$',fontsize=15)
        #ax.set_xlim(-0.2, 1.2)
        #ax.set_ylim(-1, 9)
        # plt.xlim(np.min(sf_color_index) - 0.5, np.max(sf_color_index) + 0.5)
        ax.invert_yaxis()  # Magnitudes are plotted with brighter stars at the top
        plt.tight_layout()
        plt.show()
    

    return isochrone_cut

def fit_orbit(stream_array, distance_guess, fr, progenitor_ra, fw, bw, use_position=True):
    """
    Fit an orbit to the observed stream data.

    This function estimates the best-fit orbit parameters for a stellar stream by using
    the provided observational data and distance guess. 
    Parameters:
        stream_array (pandas.DataFrame): DataFrame containing observational data for the stream.
            Expected columns include:
                - 'TARGET_RA': Right Ascension values of the stream targets.
                - 'RAdeg': RA in degrees.
                - 'DEdeg': Declination in degrees.
                - 'pmRA': Proper motion in RA.
                - 'pmDE': Proper motion in Dec.
                - 'VRAD': Radial velocities.
                - 'PMRA_ERROR': Error in proper motion in RA.
                - 'PMDEC_ERROR': Error in proper motion in Dec.
                - 'VRAD_ERR': Error in radial velocities.
        distance_guess (array-like): Estimate of the distance(s) to the stream; used to set an
            initial guess for the distance parameter.
        fr (object): The rotation matrix required to rotate from RA DEC to phi1, phi2.
        progenitor_ra (float): Right Ascension of the progenitor; used to select relevant stream data
            and compute angular differences.
        fw (numpy.array): Timesteps to integrate the orbit forwards [Gyrs]
        bw (numpy.array): Timesteps to integrate the orbit backwards [Gyrs]
    Returns:
        OptimizeResult: The output from the scipy.optimize.minimize function containing:
            - x: Best-fit orbit parameters.
            - success: Boolean indicating whether the optimizer exited successfully.
            - status: Termination status of the optimizer.
            - message: Description of the cause of termination.
            - fun: Final value of the negative log-likelihood.
            - nit: Number of iterations performed.
            - nfev: Number of function evaluations made.
    Notes:
        - The initial guess for the orbit parameters is constructed by averaging the values of several
          observed quantities (DEdeg, pmRA, pmDE, VRAD) around the selected progenitor RA.


    Example:


        progenitor_ra = np.nanmean(stream_array)
        fw = np.linspace(0., 0.05, 2001) * u.Gyr
        bw = np.linspace(0., -0.05, 2001) * u.Gyr
        distance_guess = np.nanmean(SoI_galstream.track.distance.value)
        frame = rot_matric # e.g. (SoI_galstream.stream_frame)
        results_o = gallery_funcs.fit_orbit(stream_array, distance_guess, frame, progenitor_ra, fw, bw)
        
    """

    distances = np.abs(stream_array['TARGET_RA']- progenitor_ra)
    k = np.argsort(distances)
    guess = np.append(np.nanmean(stream_array.iloc[k][['DEdeg', 'pmRA', 'pmDE', 'VRAD']].values, axis=0),
                    (distance_guess)) #use galstream track for distance gues
    lsigs = np.asarray([-1, -1, -1, -1])
    theta = np.append(guess, lsigs)
    stream_data = [ np.asarray(stream_array['RAdeg']),
                np.asarray(stream_array['DEdeg']),
                np.asarray(stream_array['pmRA']),
                np.asarray(stream_array['pmDE']),
                np.asarray(stream_array['VRAD'])]

    errs = [np.zeros_like(stream_array['RAdeg']), np.zeros_like(stream_array['DEdeg']), stream_array['PMRA_ERROR'], stream_array['PMDEC_ERROR'], stream_array['VRAD_ERR']]
    if use_position:        
        optfunc = lambda theta: negloglike(theta, fr, stream_data, progenitor_ra, errs, bw, fw)
    else:
        optfunc = lambda theta: negloglike2(theta, fr, stream_data, progenitor_ra, errs, bw, fw)
    results_o = minimize(optfunc, theta,  method="Powell")

    print("Optimization results_o:")
    print(f"Success: {results_o.success}")
    print(f"Status: {results_o.status}")
    print(f"Message: {results_o.message}")
    print(f"Function Value: {results_o.fun}")
    print(f"Number of Iterations: {results_o.nit}")
    print(f"Number of Function Evaluations: {results_o.nfev}")

    # Print each parameter with its label
    orbit_param_label = ["dec", "pmra", "pmdec", "vrad", "dist", "lsig_dec", "lsig_pmra", "lsig_pmdec", "lsig_vrad"]

    prog_distance = results_o.x[4] * u.kpc
    print(f'Progenitor')
    print(f'phi2 {results_o.x[0]*u.deg:.2f}')
    print(f'{orbit_param_label[1]}: {(2*(results_o.x[1]*u.mas/u.yr)*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'{orbit_param_label[2]}: {(2*(results_o.x[2]*u.mas/u.yr)*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'{orbit_param_label[3]}: {results_o.x[3]*u.km/u.s:.2f}')
    print(f'{orbit_param_label[4]}: {prog_distance:.2f}')
    print(f'sig_phi2: {10**results_o.x[5]*u.deg:.2f}')
    print(f'sig_pmra: {(10**results_o.x[6]*u.mas/u.yr*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'sig_pmdec: {(10**results_o.x[7]*u.mas/u.yr*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'sig_vrad: {10**results_o.x[8]*u.km/u.s:.2f}')
    #print peri and apo
    o = stream_funcs.orbit_model(results_o.x[0:5], progenitor_ra, bw, fw, return_o=True)[6]
    print(f'Pericenter: {o.rap()*u.kpc:.2f} ')
    print(f'Apocenter: {o.rperi()*u.kpc:.2f}')
    

    return results_o, o

def orbit_model(theta, ra_prog, ts_rw, ts_ff, values=True):
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

    return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist


def negloglike(theta, fr, stream_params, ra_prog, param_errs, ts_rw, ts_ff):
    stream_ra, stream_dec, stream_pmra, stream_pmdec, stream_vrad = stream_params #stream observed
    ra_err, dec_err, pmra_err, pmdec_err, vrad_err = param_errs #observed error

    o1_model_ra, o1_model_dec, o1_model_pmra, o1_model_pmdec, o1_model_vlos, o1_model_dist = np.asarray(stream_funcs.orbit_model(theta[0:5], ra_prog, ts_rw, ts_ff))
    lsig_phi2, lsig_pmra, lsig_pmdec, lsig_vrad = theta[5:] #log sigmas

    stream_phi1, stream_phi2 = stream_funcs.ra_dec_to_phi1_phi2(fr, stream_ra*u.deg, stream_dec*u.deg)
    o1_model_phi1, o1_model_phi2 = stream_funcs.ra_dec_to_phi1_phi2(fr, o1_model_ra*u.deg, o1_model_dec*u.deg)

    phi2_y = interp1d(o1_model_phi1, o1_model_phi2, kind='linear', fill_value='extrapolate')
    pmra_y = interp1d(o1_model_phi1, o1_model_pmra, kind='linear', fill_value='extrapolate')
    pmdec_y = interp1d(o1_model_phi1, o1_model_pmdec, kind='linear', fill_value='extrapolate')
    vlos_y = interp1d(o1_model_phi1, o1_model_vlos, kind='linear', fill_value='extrapolate')


    resid_phi2 = residuals(phi2_y, stream_phi2, stream_phi1)
    resid_pmra = residuals(pmra_y, stream_pmra, stream_phi1)
    resid_pmdec = residuals(pmdec_y, stream_pmdec, stream_phi1)
    resid_vlos = residuals(vlos_y, stream_vrad, stream_phi1)

    phi2_err = 0
    chi2_phi2 = resid_phi2**2/(phi2_err**2 + (10**lsig_phi2)**2)
    const_phi2 = np.log(2 * np.pi * (phi2_err**2+(10**lsig_phi2)**2))
    logl_phi2 = -0.5*np.sum(chi2_phi2 + const_phi2)

    chi2_pmra = resid_pmra**2/(pmra_err**2+ (10**lsig_pmra)**2)
    const_pmra = np.log(2 * np.pi * (pmra_err**2+(10**lsig_pmra)**2))
    logl_pmra = -0.5*np.sum(chi2_pmra + const_pmra)

    chi2_pmdec = resid_pmdec**2/(pmdec_err**2+(10**lsig_pmdec)**2)
    const_pmdec = np.log(2 * np.pi * (pmdec_err**2+(10**lsig_pmdec)**2))
    logl_pmdec = -0.5*np.sum(chi2_pmdec + const_pmdec)

    chi2_vlos = resid_vlos**2/(vrad_err**2+(10**lsig_vrad)**2)
    const_vlos = np.log(2 * np.pi * (vrad_err**2+(10**lsig_vrad)**2))
    logl_vlos = -0.5*np.sum(chi2_vlos + const_vlos)

    neg_logl_tot = -logl_phi2 - logl_pmra - logl_pmdec - logl_vlos
    return neg_logl_tot


def negloglike2(theta, fr, stream_params, ra_prog, param_errs, ts_rw, ts_ff):
    stream_ra, stream_dec, stream_pmra, stream_pmdec, stream_vrad = stream_params #stream observed
    ra_err, dec_err, pmra_err, pmdec_err, vrad_err = param_errs #observed error

    o1_model_ra, o1_model_dec, o1_model_pmra, o1_model_pmdec, o1_model_vlos, o1_model_dist = np.asarray(stream_funcs.orbit_model(theta[0:5], ra_prog, ts_rw, ts_ff))
    lsig_phi2, lsig_pmra, lsig_pmdec, lsig_vrad = theta[5:] #log sigmas

    stream_phi1, stream_phi2 = stream_funcs.ra_dec_to_phi1_phi2(fr, stream_ra*u.deg, stream_dec*u.deg)
    o1_model_phi1, o1_model_phi2 = stream_funcs.ra_dec_to_phi1_phi2(fr, o1_model_ra*u.deg, o1_model_dec*u.deg)

    pmra_y = interp1d(o1_model_phi1, o1_model_pmra, kind='linear', fill_value='extrapolate')
    pmdec_y = interp1d(o1_model_phi1, o1_model_pmdec, kind='linear', fill_value='extrapolate')
    vlos_y = interp1d(o1_model_phi1, o1_model_vlos, kind='linear', fill_value='extrapolate')


    resid_pmra = residuals(pmra_y, stream_pmra, stream_phi1)
    resid_pmdec = residuals(pmdec_y, stream_pmdec, stream_phi1)
    resid_vlos = residuals(vlos_y, stream_vrad, stream_phi1)

    chi2_pmra = resid_pmra**2/(pmra_err**2+ (10**lsig_pmra)**2)
    const_pmra = np.log(2 * np.pi * (pmra_err**2+(10**lsig_pmra)**2))
    logl_pmra = -0.5*np.sum(chi2_pmra + const_pmra)

    chi2_pmdec = resid_pmdec**2/(pmdec_err**2+(10**lsig_pmdec)**2)
    const_pmdec = np.log(2 * np.pi * (pmdec_err**2+(10**lsig_pmdec)**2))
    logl_pmdec = -0.5*np.sum(chi2_pmdec + const_pmdec)

    chi2_vlos = resid_vlos**2/(vrad_err**2+(10**lsig_vrad)**2)
    const_vlos = np.log(2 * np.pi * (vrad_err**2+(10**lsig_vrad)**2))
    logl_vlos = -0.5*np.sum(chi2_vlos + const_vlos)

    neg_logl_tot = - logl_pmra - logl_pmdec - logl_vlos
    return neg_logl_tot

def residuals(o1_interp_func, stream_val, stream_phi1):
    return np.abs(o1_interp_func(stream_phi1) - stream_val)


def orbit_interpolations(o_s):
    o_phi1, o_phi2, o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_vgsr, o_dist = o_s
    ointerp_phi1 = interp1d(o_phi1, o_phi1, kind='linear', fill_value='extrapolate')
    ointerp_phi2 = interp1d(o_phi1, o_phi2, kind='linear', fill_value='extrapolate')
    ointerp_ra = interp1d(o_phi1, o_ra, kind='linear', fill_value='extrapolate')
    ointerp_dec = interp1d(o_phi1, o_dec, kind='linear', fill_value='extrapolate')
    ointerp_pmra = interp1d(o_phi1, o_pmra, kind='linear', fill_value='extrapolate')
    ointerp_pmdec = interp1d(o_phi1, o_pmdec, kind='linear', fill_value='extrapolate')
    ointerp_vrad = interp1d(o_phi1, o_vrad, kind='linear', fill_value='extrapolate')
    ointerp_vgsr = interp1d(o_phi1, o_vgsr, kind='linear', fill_value='extrapolate')
    ointerp_dist = interp1d(o_phi1, o_dist, kind='linear', fill_value='extrapolate')
    df = {
        "phi1": ointerp_phi1,
        "phi2": ointerp_phi2,
        "ra": ointerp_ra,
        "dec": ointerp_dec,
        "pmra": ointerp_pmra,
        "pmdec": ointerp_pmdec,
        "vrad": ointerp_vrad,
        "vgsr": ointerp_vgsr,
        "dist": ointerp_dist
    }
    return df

import astropy.coordinates as coord
from astropy import table
import galstreams
from gala.coordinates import GreatCircleICRSFrame

def get_cropped_DESI(desi_path, high_prob_members, rotation_matrix):
    desi_data = table.Table.read(desi_path, format='fits')
    print(f"DESI data Column Names: {desi_data.colnames}")

    desired_columns = [
        'VRAD', 'VRAD_ERR', 'PARALLAX', 'PARALLAX_ERROR', 
        'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 'SOURCE_ID', 
        'TARGETID', 'PMRA_PMDEC_CORR', 'PRIMARY'
    ]

    dist_columns = ['TARGETID', 'dist_mod', 'dist_mod_err']

    # Load the DECaLS data
    decal_columns = ['EBV', 'FLUX_G', 'FLUX_R']
    # Drop the rows with NaN values in all columns
    print(f"Length of desi Data before Cuts: {len(desi_data)}")
    drop_nan_columns = np.concatenate((desired_columns, decal_columns))
    desi_dropped_nan_df = stream_funcs.dropna_Table(desi_data, columns = drop_nan_columns) # Custom function to drop rows with NaN values
    print(f"Length of desi Data after NaN cut: {len(desi_dropped_nan_df)}")

    # Drop the rows with 'RVS_WARN' != 0 and 'RR_SPECTYPE' != 'STAR', and are not duplicates
    desi_dropped_vals = desi_dropped_nan_df[(desi_dropped_nan_df['PRIMARY'])]
    print(f"Length of desi data after RVS_WARN, RR_SPECTYPE, and PRIMARY: {len(desi_dropped_vals)}")

    desi_dropped_vals = desi_dropped_vals.to_pandas()
    # add floor to the radial velocity error
    desi_dropped_vals['VRAD_ERR'] = np.sqrt(desi_dropped_vals['VRAD_ERR']**2 + 0.9**2) ### Turn into its own column
    desi_dropped_vals['PMRA_ERROR'] = np.sqrt(desi_dropped_vals['PMRA_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
    desi_dropped_vals['PMDEC_ERROR'] = np.sqrt(desi_dropped_vals['PMDEC_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
    desi_dropped_vals['FEH_ERR'] = np.sqrt(desi_dropped_vals['FEH_ERR']**2 + 0.01**2) ### Turn into its own column
    # Delete some old variables
    del desi_dropped_nan_df, desi_data

    desi_SoI_mask = stream_funcs.threeD_max_min_mask(desi_dropped_vals['TARGET_RA'], desi_dropped_vals['TARGET_DEC'], desi_dropped_vals['PARALLAX'], desi_dropped_vals['PARALLAX_ERROR'],\
    high_prob_members['TARGET_RA'], high_prob_members['TARGET_DEC'], np.nanmin(stream_funcs.dist_mod_to_dist(high_prob_members['dist_mod'])/1000), 360, 360)
    desi_data_cropped = desi_dropped_vals[desi_SoI_mask]

    desi_phi1, desi_phi2 = stream_funcs.ra_dec_to_phi1_phi2(rotation_matrix, np.array(desi_data_cropped['TARGET_RA'])*u.deg, np.array(desi_data_cropped['TARGET_DEC'])*u.deg)
    desi_data_cropped.loc[:,'phi1'], desi_data_cropped.loc[:,'phi2'] = desi_phi1, desi_phi2
    del desi_phi1, desi_phi2, desi_dropped_vals
    return desi_data_cropped