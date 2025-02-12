import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
import glob, os, shutil
import pdb

from astropy import units as u
from astropy import time
import emcee
import corner
from matplotlib import colors, cm
import scipy.stats as stats
from scipy import linalg
import multiprocessing as mp
from scipy import optimize as op
from alicesaur.gaia import gaia_utils, gaia_plot, fit_psf

from astroquery.gaia import Gaia
from astropy.wcs import WCS

from matplotlib import rc
rc('text', usetex=False)


def lnlike(p, sky_pos, sky_cov, px_pos, px_cov, include_indx):
    
    chi2 = 0
    ps_inv = np.array([1.0/p[2], 1.0/p[3]])
    target_pos = np.array((p[0], p[1]))

    rot_mat = np.array([[-np.cos(p[4]), np.sin(p[4])], [np.sin(p[4]), np.cos(p[4])]])
    ps_mat = np.array([[ps_inv[0]**2, ps_inv[0]*ps_inv[1]], [ps_inv[0]*ps_inv[1], ps_inv[1]**2]])

    sky_pos2 = []
    sky_cov2 = []
    sky_chi2 = []

    for i in include_indx:
        new_pos = ((rot_mat.dot(sky_pos[i]))*ps_inv) + target_pos
        new_cov = (rot_mat.dot(sky_cov[i]).dot(rot_mat.T)) * ps_mat

        resid = new_pos - px_pos[i]
        cov = px_cov[i] + new_cov
        cov_inv = np.linalg.inv(cov)

        new_chi2 = resid.T.dot(cov_inv).dot(resid)

        chi2 += new_chi2

        sky_pos2.append(new_pos)
        sky_cov2.append(new_cov)
        sky_chi2.append(new_chi2)

    blob = np.concatenate((sky_chi2, np.array(sky_pos2).flatten(), np.array(sky_cov2).flatten()))

    return -0.5*chi2, blob
  
def lnprior(p):

    if (p[2] <= 5) | (p[2] > 200.0):
        return -np.inf
    if (p[3] <= 5) | (p[3] > 200.0):
        return -np.inf
    if (p[4] < 0.0) | (p[4] > (2.0*np.pi)):
        return -np.inf

    return 0

def lnprob(p, sky_pos, sky_cov, px_pos, px_cov, include_indx):
    
    lnp = lnprior(p)

    lnl, blob = lnlike(p, sky_pos, sky_cov, px_pos, px_cov, include_indx)

    return lnp + lnl, blob

def mcmc(sky_pos, sky_cov, px_pos, px_cov, include_indx, guess_x, guess_y, guess_ps, guess_tn, nsteps=1000, nwalkers=256):

    ndim = 5
    p0 = [guess_x, guess_y, guess_ps, guess_ps, guess_tn % (2.0*np.pi)]
    dp = [1.0, 1.0, 0.1, 0.1, 0.1]
    pos0 = np.zeros((nwalkers, ndim))
    for k in range(0, ndim):
        pos0[..., k] = np.random.normal(p0[k], dp[k], nwalkers)

    args = (sky_pos, sky_cov, px_pos, px_cov, include_indx)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    _ = sampler.run_mcmc(pos0, nsteps)

    samples = sampler.get_chain() # nsteps, nwalkers, ndim
    lnp = sampler.get_log_prob() 
    #blobs = sampler.get_blobs() # steps, walkers, blob#, star#

    n_stars = len(include_indx)
    blobs_chi = sampler.get_blobs()[:,:,0:n_stars]
    blobs_pos = sampler.get_blobs()[:,:,1*n_stars:3*n_stars].reshape((nsteps, nwalkers, n_stars, 2))
    blobs_cov = sampler.get_blobs()[:,:,3*n_stars:7*n_stars].reshape((nsteps, nwalkers, n_stars, 2, 2))

    sampler.reset()

    return samples, lnp, [blobs_chi, blobs_pos, blobs_cov]

def calculate_max_radius(hdr, im):
    """
    Output: Maximum search radius in [arcsec].
    """

    # Calculates search radius required to encompass image using the WCS reference position as an origin

    ra0, de0 = hdr['CRVAL1'], hdr['CRVAL2']
    x, y = np.meshgrid(np.arange(hdr['NAXIS1']), np.arange(hdr['NAXIS2']))
    w = WCS(hdr)
    sky = w.pixel_to_world(x, y)#.flatten()
    sep = sky.separation(w.pixel_to_world(hdr['CRPIX1'], hdr['CRPIX2'])).arcsec

    # Only consider valid pixels
    # TODO - use mask array instead of searching for NaNs/zeros?
    indx = np.where((im == 0.0) | np.isnan(im))
    sep[indx] = 0.0

    max_sep = np.nanmax(sep) * 1.05

    return max_sep

def main(im_path, inst, target_id, target_rv, target_xy, gaia_catalogue='DR3',
         exclude_extra=[], im=None, hdr=None, out_dir=None, max_ruwe=1e10):

    '''
    # Function name TBD

    im_path - path to reduced FITS file
    target_id - Gaia source identifier for the occulted target
    target_rv - Tuple containing target radial velocity and uncertainty (km/s)
    target_xy - Pixel position of target in the FITS file
    gaia_catalogue - Which Gaia data release to use
    exclude_extra - Gaia source IDs of stars to exclude

    '''

    if im is None:
        filename = im_path.replace('.fits', '')

        '''
        Open FITS file
        '''

        with fits.open(im_path) as hdu:
            ## TODO - extension may vary by instrument - not an issue if reading image/header from object
            im = hdu[0].data
            hdr = hdu[0].header
            s = im.shape
    else:
        filename = im_path
        s = im.shape

    if out_dir is None:
        out_dir = os.path.abspath(os.path.curdir)

    '''
    Get useful quantities from header
    '''

    w = WCS(hdr)
    ref_pos = w.pixel_to_world(hdr['CRPIX1'], hdr['CRPIX2'])
    guess_ps = np.mean([x.to(u.mas).value for x in w.proj_plane_pixel_scales()])
    guess_tn = hdr.get('ORIENTAT', 0.)*np.pi/180.0

    # Exposure mid-point in decimal years
    if ("DATE-MID" in hdr.keys()) and ("TIME-MID" in hdr.keys()):
        if "T" in hdr['DATE-MID']:
            t_hst = time.Time(f"{hdr['DATE-MID']}", format='isot', scale='utc').decimalyear
        else:
            t_hst = time.Time(f"{hdr['DATE-MID']}", format='iso', scale='utc').decimalyear
    else:
        try:
            t_hst = time.Time(hdr['EXPSTART'] + (hdr['EXPTIME']/86400.0)/2., format='mjd').decimalyear
        except:
            if "T" in hdr['DATE-OBS']:
                t_hst = (time.Time(hdr['DATE-OBS'], format='isot', scale='utc') + (time.Time(['DATE-END'], format='isot', scale='utc') - time.Time(hdr['DATE-OBS'], format='isot', scale='utc'))).decimalyear
            else:
                t_hst = (time.Time(hdr['DATE-OBS'], format='iso', scale='utc') + (time.Time(['DATE-END'], format='iso', scale='utc') - time.Time(hdr['DATE-OBS'], format='iso', scale='utc'))).decimalyear



    '''
    Perform Gaia query
    ## TODO - add caching of results
    '''

    if gaia_catalogue == 'DR2':
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        t_gaia = 2015.5
    elif gaia_catalogue == 'DR3':
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        t_gaia = 2016.0
    else:
        raise NotImplementedError

    Gaia.ROW_LIMIT = 10000 # If we have more than this, something has gone wrong...

    # TODO - add inner radius for search query!

    search_radius = calculate_max_radius(hdr, im) # [arcsec]

    # Limit the search radius in case of padded images that are mostly empty.
    # Maybe move this limiting bit inside calculate_max_radius eventually.
    if inst.lower() == 'stis':
        radius_limit = 55. # [arcsec]
    else:
        radius_limit = np.inf
    if search_radius > radius_limit:
        search_radius = radius_limit

    print("Retrieving Gaia stars from online database...")
    query = Gaia.cone_search_async(ref_pos, radius=search_radius*u.arcsec)
    data = query.get_results()

    # Sanitize the results, remove entries without proper motion and parallax measurements
    indx_good = ~data['parallax'].mask | ~data['pmra'].mask | ~data['pmdec'].mask
    data = data[indx_good]

    # Remove entries with large RUWE
    indx_good = np.where(data['ruwe'].data <= max_ruwe)[0]
    data = data[indx_good]

    n_stars = len(data)
    try:
        source_id = data['SOURCE_ID'].data
    except:
        source_id = data['source_id'].data

    '''
    Set various parameters for the image
    '''
    # List of source IDs to exclude from analysis
    exclude_id = []
    exclude_id += [target_id, ] # Always exclude the target

    for i in exclude_extra:
        if i in source_id:
            exclude_id += [i, ]

    '''
    Propagate gaia astrometry to the HST epoch
    '''
    # Propagate astrometry and calculate tangent plane offsets relative to target
    dt = t_hst - t_gaia
    sky_pos, sky_cov = gaia_utils.tangent_plane_offsets(data, t_hst, dt, np.where(source_id == target_id)[0][0], target_rv, n_mc=int(1e4))

    # Compute pixel offsets using guess plate scale and true north angle
    x = (1.0/guess_ps) * (-sky_pos[:, 0]*np.cos(guess_tn) + sky_pos[:, 1]*np.sin(guess_tn)) + target_xy[0]
    y = (1.0/guess_ps) * (sky_pos[:, 0]*np.sin(guess_tn) + sky_pos[:, 1]*np.cos(guess_tn)) + target_xy[1]

    '''
    TODO: filter the list here
    there's a boolean array for this in the data structure (align_masks, and spike_masks)
    also carry these through to the PSF fitting step, give these flagged areas a weighting of zero

    '''

    for i in range(0, n_stars):
        x_int, y_int = int(np.round(x[i])), int(np.round(y[i]))
        # Exclude sources near or outside the edges of the image array.
        if not ((20 <= x[i] <= s[1] - 20) and (20 <= y[i] <= s[0] - 20)):
            exclude_id += [source_id[i]]
        # Exclude sources centered on masked pixels of the image array.
        elif (np.isnan(im[y_int, x_int]) | (im[y_int, x_int] == 0.0)):
            # TODO - check mask array
            exclude_id += [source_id[i]]
        else:
            pass

    '''
    Create an overview figure showing the image and the Gaia sources, symbols indicating which are used
    '''
    _ = gaia_plot.plot_overview(im, x, y, source_id, exclude_id,
                                outname=os.path.join(out_dir, f'gaia_overview-{filename}'))

    '''
    Fit the pixel position of each source
    TODO: Save stamps showing the data, model, and residual, as well as FWHM and amplitude
    '''
    star_errors = None
    if inst.lower() == 'stis':
        xoff, yoff = -0.054, -0.047
        xinf, yinf = 0.05, 0.05
    elif inst.lower() == 'acs':
        xoff, yoff = 0., 0.
        xinf, yinf = 0., 0.
        print("\n***WARNING: No offsets set for ACS yet")
    else:
        xoff, yoff = 0., 0.
        xinf, yinf = 0., 0.
    px_pos, px_cov, data_stamps, model_stamps, model_fits = fit_psf.fit(im,
                                x, y, source_id, exclude_id, star_errors,
                                [xoff, yoff], [xinf, yinf], method='gaussian')

    '''
    TODO: additional filtering here based on peak flux, FWHM
    '''

    '''
    Fit plate scale, true north, and (x,y) position of target
    '''
    include_id = [k for k in source_id if k not in exclude_id] # A list of the Gaia IDs to use in the analysis
    include_indx = [list(source_id).index(k) for k in include_id] # A list of the indicies for those stars in the `source_id` list
    #_ = mcmc(sky_pos, sky_cov, px_pos, px_cov, include_indx, target_xy[0], target_xy[1], guess_ps, guess_tn, nsteps=1000, nburn=250)

    print("Running Gaia MCMC...")
    nsteps = 100
    nburn = 25

    samples, lnp, blobs = mcmc(sky_pos, sky_cov, px_pos, px_cov, include_indx, target_xy[0], target_xy[1], guess_ps, guess_tn, nsteps=nsteps)

    # Generate list of chi2, to remove outliers in a second run
    # Updated star center could be used as input for the second run

    # Plots
    labels = [r'$x_0 (px)$', r'$y_0 (px)$', r'ps$_x$ (mas/px)', r'ps$_y$ (mas/px)', r'$\theta$ (deg)']
    samples[:,:,4] *= 180.0/np.pi # This is so lazy
    _ = gaia_plot.plot_mcmc_chains(samples, nburn, labels, outname=os.path.join(out_dir, f'gaia_mcmc-chains-{filename}'))
    _ = gaia_plot.plot_mcmc_corner(samples, nburn, labels, outname=os.path.join(out_dir, f'gaia_mcmc-corner-{filename}'))
    samples[:,:,4] /= 180.0/np.pi

    '''
    Posterior distributions for star position, plate scales, and north angle
    '''
    final_target_x = samples[nburn:,:,0].flatten()
    final_target_y = samples[nburn:,:,1].flatten()
    final_ps_x = samples[nburn:,:,2].flatten()
    final_ps_y = samples[nburn:,:,3].flatten()
    final_tn = samples[nburn:,:,4].flatten()

# FIX ME!!! Organize the results output better.
    final_x_median = np.nanmedian(final_target_x)
    final_y_median = np.nanmedian(final_target_y)
    final_x_std = np.nanstd(final_target_x)
    final_y_std = np.nanstd(final_target_y)
    final_ps_x_median = np.nanmedian(final_ps_x)
    final_ps_y_median = np.nanmedian(final_ps_y)
    final_ps_x_std = np.nanstd(final_ps_x)
    final_ps_y_std = np.nanstd(final_ps_y)
    final_tn_median = np.nanmedian(final_tn)
    final_tn_std = np.nanstd(final_tn)

    '''
    Create gallery plot of each star
    '''
    print("Plotting PSF stamps...")
    _ = gaia_plot.plot_fits(data, include_indx, sky_pos, sky_cov, px_pos, px_cov, data_stamps,
                            model_stamps, model_fits, samples, lnp, blobs,
                            final_target_x, final_target_y, final_ps_x, final_ps_y, final_tn,
                            xoff, yoff, outname=os.path.join(out_dir, f'gaia_psffits-{filename}'))

    return final_x_median, final_y_median, final_ps_x_median, final_ps_y_median, final_tn_median, final_x_std, final_y_std, final_ps_x_std, final_ps_y_std, final_tn_std


if __name__ == '__main__':
    # stars to test exclude_extra:
    # 6072903200423956608 - star 18 behind the wedge
    #main('vi01_1.fits', 'STIS', 6072902994276659200, [12.18, 0.15], [530, 304], gaia_catalogue='DR3', exclude_extra=[])
    main('vi01_1.fits', 'STIS', 6072902994276659200, [12.18, 0.15], [530, 304], gaia_catalogue='DR3',
        exclude_extra=[6072902994266748800, 6072903200423956608, 6072902994265478400], max_ruwe=1.6)


