from scipy.interpolate import interp1d
from scipy.optimize import minimize
from types import SimpleNamespace
from typing import cast
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import healpy as hp
import astropy.coordinates as coord
import pandas as pd
import corner
import emcee

## every non-orbit function was written by Joseph Tang

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

def fit_orbit(stream_array, fr, progenitor_ra, fw, bw, theta, use_position=True, use_mcmc=False, **kwargs):
    """Fit an orbit to the observed stream data."""

    make_corner = bool(kwargs.pop('make_corner', False))
    corner_kwargs = kwargs.pop('corner_kwargs', None)
    corner_thin = kwargs.pop('corner_thin', None)

    orbit_param_label = ["dec", "pmra", "pmdec", "vrad", "dist", "lsig_dec", "lsig_pmra", "lsig_pmdec", "lsig_vrad"]

    print('orbit parameters', orbit_param_label)
    print('intital guess', theta)

    theta0 = np.asarray(theta, dtype=float)
    errs = [stream_array['PMRA_ERROR'], stream_array['PMDEC_ERROR'], stream_array['VRAD_ERR']]
    optfunc = lambda pars: negloglike(pars, fr, stream_array, progenitor_ra, errs, bw, fw, use_position=use_position)

    if use_mcmc:
        pre_optimize = kwargs.get('pre_optimize', True)
        seed_theta = theta0.copy()
        seed_result = None
        if pre_optimize:
            print('Running preliminary Powell optimisation for walker seeding...')
            seed_result = minimize(optfunc, seed_theta, method="Powell")
            if getattr(seed_result, 'success', False):
                seed_theta = np.asarray(seed_result.x, dtype=float)
                print('Pre-optimisation succeeded; updating walker centre.')
            else:
                print('Pre-optimisation did not converge; using provided initial guess.')

        theta0 = seed_theta
        ndim = theta0.size
        nburn = int(kwargs.get('nburn', 500))
        if nburn < 0:
            raise ValueError("nburn must be a non-negative integer for MCMC sampling.")
        nwalkers = int(kwargs.get('nwalkers', max(32, 2 * ndim)))
        nsteps = int(kwargs.get('nsteps', 2000))
        if nsteps <= 0:
            raise ValueError("nsteps must be a positive integer for MCMC sampling.")
        burnin = kwargs.get('burnin', nsteps // 2)
        burnin = int(np.clip(burnin, 0, max(nsteps - 1, 0))) if nsteps > 0 else 0
        progress = kwargs.get('progress', True)
        seed = kwargs.get('seed', None)
        walker_spread = np.asarray(
            kwargs.get('walker_spread', [0.2, 0.1, 0.1, 10.0, 0.5, 0.3, 0.3, 0.3, 0.3]),
            dtype=float,
        )
        if walker_spread.size == 1:
            walker_spread = np.repeat(walker_spread, ndim)
        walker_spread = np.broadcast_to(walker_spread, (ndim,))

        rng = np.random.default_rng(seed)

        def _draw_initial():
            return theta0 + rng.normal(scale=walker_spread)

        p0 = []
        attempts = 0
        max_attempts = nwalkers * 1000
        while len(p0) < nwalkers and attempts < max_attempts:
            trial = _draw_initial()
            attempts += 1
            if np.isfinite(lnprior(trial)):
                p0.append(trial)
        if len(p0) < nwalkers:
            raise RuntimeError("Unable to initialize enough walkers inside the prior bounds.")
        p0 = np.asarray(p0)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(fr, stream_array, progenitor_ra, errs, bw, fw, use_position),
        )

        if nburn > 0:
            print("Running burn-in...")
            state = sampler.run_mcmc(p0, nburn, progress=progress)
            sampler.reset()
        else:
            state = p0

        print("Running production...")
        sampler.run_mcmc(state, nsteps, progress=progress)

        chain_steps = sampler.get_chain()
        log_prob_steps = sampler.get_log_prob()
        if burnin >= chain_steps.shape[0]:
            raise ValueError("burnin exceeds available production steps; reduce burnin or increase nsteps.")

        flat_chain = chain_steps[burnin:].reshape(-1, ndim)
        flat_log_prob = log_prob_steps[burnin:].reshape(-1)
        if flat_chain.size == 0:
            raise RuntimeError("No MCMC samples available after burn-in; adjust burnin or nsteps.")

        max_prob_index = np.argmax(flat_log_prob)
        best_fit_params = flat_chain[max_prob_index]
        best_fun = negloglike(best_fit_params, fr, stream_array, progenitor_ra, errs, bw, fw, use_position=use_position)

        results_o = SimpleNamespace(
            x=best_fit_params,
            success=True,
            status=0,
            message="MCMC completed",
            fun=best_fun,
            nit=nsteps,
            nfev=nsteps * nwalkers,
            chain=flat_chain,
            chain_steps=chain_steps,
            log_prob=flat_log_prob,
            log_prob_trace=log_prob_steps,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            sampler=sampler,
            burnin=burnin,
            nburn=nburn,
            nsteps=nsteps,
            nwalkers=nwalkers,
            method="mcmc",
            seed_result=seed_result,
        )

        if make_corner:
            corner_samples = flat_chain
            if corner_thin is not None and corner_thin > 1:
                corner_samples = corner_samples[::int(corner_thin)]
            corner_kwargs = corner_kwargs or {}
            results_o.corner = corner.corner(corner_samples, labels=orbit_param_label, **corner_kwargs)
        else:
            results_o.corner = None
        results_o.corner_axes = getattr(results_o.corner, 'axes', None) if results_o.corner is not None else None
    else:
        results_o = minimize(optfunc, theta0, method="Powell")
        results_o.method = "powell"
        results_o.acceptance_fraction = None
        results_o.chain = None
        results_o.chain_steps = None
        results_o.log_prob = None
        results_o.log_prob_trace = None
        results_o.burnin = 0
        results_o.nburn = 0
        results_o.nsteps = getattr(results_o, 'nit', None)
        results_o.nwalkers = None
        results_o.seed_result = None
        results_o.corner = None
        results_o.corner_axes = None

    results_o.param_labels = orbit_param_label
    results_o.chain_fig = getattr(results_o, 'chain_fig', None)
    results_o.chain_axes = getattr(results_o, 'chain_axes', None)

    print("Optimization results_o:")
    print(f"Success: {results_o.success}")
    print(f"Status: {results_o.status}")
    print(f"Message: {results_o.message}")
    print(f"Function Value: {getattr(results_o, 'fun', np.nan)}")
    print(f"Number of Iterations: {getattr(results_o, 'nit', 'N/A')}")
    print(f"Number of Function Evaluations: {getattr(results_o, 'nfev', 'N/A')}")
    if use_mcmc:
        print(f"Mean acceptance fraction: {results_o.acceptance_fraction:.3f}")

    prog_distance = results_o.x[4] * u.kpc
    print('Progenitor')
    print(f'{orbit_param_label[0]}: {results_o.x[0]*u.deg:.2f}')
    print(f'{orbit_param_label[1]}: {results_o.x[1]*u.mas/u.yr:.2f}')
    print(f'{orbit_param_label[2]}: {results_o.x[2]*u.mas/u.yr:.2f}')
    print(f'{orbit_param_label[3]}: {results_o.x[3]*u.km/u.s:.2f}')
    print(f'{orbit_param_label[4]}: {results_o.x[4]*u.kpc:.2f}')
    print(f'{orbit_param_label[5]}: {10**results_o.x[5]*u.deg:.2f}')
    print(f'{orbit_param_label[6]}: {10**results_o.x[6]*u.mas/u.yr:.2f} or {(10**results_o.x[6]*u.mas/u.yr*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'{orbit_param_label[7]}: {10**results_o.x[7]*u.mas/u.yr:.2f} or {(10**results_o.x[7]*u.mas/u.yr*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'{orbit_param_label[8]}: {10**results_o.x[8]*u.km/u.s:.2f}')

    orbit_outputs = orbit_model(results_o.x[0:5], progenitor_ra, bw, fw, return_o=True)
    if len(orbit_outputs) < 7:
        raise RuntimeError("orbit_model did not return an Orbit object; ensure return_o=True.")
    orbit_obj = cast(Orbit, orbit_outputs[-1])
    print(f'Pericenter: {orbit_obj.rperi()*u.kpc:.2f} ')
    print(f'Apocenter: {orbit_obj.rap()*u.kpc:.2f}')
    period = orbit_obj.Tr()
    print(f"Orbital period: {period:.2f}")

    return results_o, orbit_obj


def negloglike(theta, fr, stream_array, ra_prog, param_errs, ts_rw, ts_ff, use_position=True):
    stream_ra, stream_dec, stream_pmra, stream_pmdec, stream_vrad = stream_array['RA'], stream_array['DEC'], stream_array['PMRA'], stream_array['PMDEC'], stream_array['VRAD']
    pmra_err, pmdec_err, vrad_err = param_errs #observed error
    if np.abs(theta[0]) > 90:
        return np.inf
    else:
        o1_model_ra, o1_model_dec, o1_model_pmra, o1_model_pmdec, o1_model_vlos, o1_model_dist = np.asarray(orbit_model(theta[0:5], ra_prog, ts_rw, ts_ff))
        lsig_phi2, lsig_pmra, lsig_pmdec, lsig_vrad = theta[5:] #log sigmas

        stream_phi1, stream_phi2 = ra_dec_to_phi1_phi2(fr, np.asarray(stream_ra)*u.deg, np.asarray(stream_dec)*u.deg)
        o1_model_phi1, o1_model_phi2 = ra_dec_to_phi1_phi2(fr, o1_model_ra*u.deg, o1_model_dec*u.deg)

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

        if use_position:
            neg_logl_tot = -logl_phi2 - logl_pmra - logl_pmdec - logl_vlos
        else:
            neg_logl_tot = -logl_pmra - logl_pmdec - logl_vlos
        return neg_logl_tot

def lnprior(theta, priors=None):
    dec, pmra, pmdec, vrad, dist, lsig_phi2, lsig_pmra, lsig_pmdec, lsig_vrad = theta
    if -90 < dec < 90 and -20 < pmra < 20 and -20 < pmdec < 20 and -500 < vrad < 500 and 0.1 < dist < 100 and -5 < lsig_phi2 < 3 and -5 < lsig_pmra < 3 and -5 < lsig_pmdec < 3 and -5 < lsig_vrad < 3:
        return 0.0
    return -np.inf

def lnprob(theta, fr, stream_array, ra_prog, param_errs, ts_rw, ts_ff, use_position=True):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - negloglike(theta, fr, stream_array, ra_prog, param_errs, ts_rw, ts_ff, use_position=use_position)

def residuals(o1_interp_func, stream_val, stream_phi1):
    return np.abs(o1_interp_func(stream_phi1) - stream_val)

def plot_mcmc_traces(results, labels=None, burnin=None):
    if getattr(results, 'chain_steps', None) is None:
        raise ValueError("No MCMC chain available; run with use_mcmc=True and keep sampler outputs.")

    chain_steps = results.chain_steps
    if chain_steps.ndim != 3:
        raise ValueError("Expected chain_steps to have shape (nsteps, nwalkers, ndim).")

    if burnin is None:
        burnin = getattr(results, 'burnin', 0)
    burnin = int(max(burnin, 0))
    if burnin >= chain_steps.shape[0]:
        raise ValueError("Burn-in exceeds available samples for trace plotting.")

    trace = chain_steps[burnin:]
    nsteps, nwalkers, ndim = trace.shape

    if labels is None:
        labels = getattr(results, 'param_labels', None)
        if labels is None:
            labels = [f"param_{i}" for i in range(ndim)]

    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(10, 2.2 * ndim))
    axes = np.atleast_1d(axes)

    for idx in range(ndim):
        ax = axes[idx]
        walkers = trace[:, :, idx]
        ax.plot(walkers, alpha=0.3)
        if idx < len(labels):
            ax.set_ylabel(labels[idx])
    axes[-1].set_xlabel('step')
    fig.tight_layout()

    return fig, axes

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

    model_ra = np.concatenate([model_ra_rw[::-1], model_ra_ff])
    model_dec = np.concatenate([model_dec_rw[::-1], model_dec_ff])
    model_pmra = np.concatenate([model_pmra_rw[::-1], model_pmra_ff])
    model_pmdec = np.concatenate([model_pmdec_rw[::-1], model_pmdec_ff])
    model_vlos = np.concatenate([model_vlos_rw[::-1], model_vlos_ff])
    model_dist = np.concatenate([model_dist_rw[::-1], model_dist_ff])
    if return_o:
        return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist, o_rw
    else:
        return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist
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

def plot_orbit(
    model_ra,
    model_dec,
    stream_ra,
    stream_dec,
    ra_prog,
    model_dist=None,
    cmap='brg',
    add_colorbar=True,
    cbar_label='Distance'
):
    fig = plt.figure(figsize=(12*0.8, 10*0.8))
    ax = fig.add_subplot(111, projection="mollweide")

    # Convert RA to radians and shift to [-180, 180] for Mollweide projection
    model_ra_rad = np.radians(zero_360_to_180(model_ra))
    stream_ra_rad = np.radians(zero_360_to_180(stream_ra))
    ra_prog_rad = np.radians(zero_360_to_180(np.array([ra_prog])))

    # Convert Dec to radians
    model_dec_rad = np.radians(model_dec)
    stream_dec_rad = np.radians(stream_dec)
    dec_prog_rad = np.radians(interp1d(model_ra, model_dec, kind='linear', fill_value='extrapolate')(ra_prog))

    # Plot stream points
    ax.scatter(stream_ra_rad, stream_dec_rad, c='k', s=10, zorder=2)

    # If distances provided, color the orbit by that array's extent
    if model_dist is not None:
        # Prepare color values from provided distances (Quantity or array)
        if isinstance(model_dist, u.quantity.Quantity):
            carray = model_dist.to_value()
        else:
            carray = np.asarray(model_dist, dtype=float)

        # Build line segments for coloring, avoid RA wrap jumps
        x = model_ra_rad
        y = model_dec_rad
        dra = np.abs(np.diff(x))
        good = dra < (np.pi / 1.5)
        seg_x = np.column_stack([x[:-1], x[1:]])[good]
        seg_y = np.column_stack([y[:-1], y[1:]])[good]
        segments = np.stack([seg_x, seg_y], axis=-1)

        # Segment color values (midpoint of adjacent distances)
        cvals = 0.5 * (carray[:-1] + carray[1:])
        cvals = cvals[good]

        lc = LineCollection(segments, cmap=cmap, array=cvals, linewidths=3.0, zorder=1)
        lc.set_clim(vmin=np.nanmin(carray), vmax=np.nanmax(carray))
        ax.add_collection(lc)

        if add_colorbar:
            cbar = fig.colorbar(lc, ax=ax, orientation='vertical', pad=0.03, fraction=0.02,)
            cbar.set_label(cbar_label, fontsize=11)
    else:
        # Fallback: single-color model orbit when distances are not provided
        ax.scatter(model_ra_rad, model_dec_rad, c='r', s=0.5, zorder=1)

    ax.set_xlabel(r'$\alpha$ [deg]')
    ax.set_ylabel(r'$\delta$ [deg]')
    # ax.legend(handles = legend_handles, loc='upper right', fontsize=10)
    return fig, ax

def zero_360_to_180(ra):
    ra_copy = np.copy(ra)
    where_180 = np.where(ra_copy > 180)
    ra_copy[where_180] = ra_copy[where_180] - 360
        
    return ra_copy

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

    
