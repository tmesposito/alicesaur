import numpy as np

from astropy import time
from astropy.coordinates import get_body_barycentric
from astropy.io import fits
from astroquery.simbad import Simbad


def gaia_correlated_variates(astrometry, error, correlation, n=int(1e6)):

    # astrometry - (ra[rad], de[rad], plx[mas], pmra[mas/yr], pmde[mas/yr]
    # error - (mas, mas, mas, mas/yr, mas/yr)
    # correlation - coefficients in order from VizieR table. 
    # returns array size (n, 5) with the same units as the input
 
    # mas_to_rad = (1.0*u.mas).to(u.rad).value

    mr = (1.0/(1000.0*60*60))*(np.pi/180.0)

    s1, s2, s3, s4, s5 = error
    c12, c13, c14, c15, c23, c24, c25, c34, c35, c45 = correlation

    c = [[s1*s1, c12*s1*s2, c13*s1*s3, c14*s1*s4, c15*s1*s5],
         [c12*s2*s1, s2*s2, c23*s2*s3, c24*s2*s4, c25*s2*s5], 
         [c13*s3*s1, c23*s3*s2, s3*s3, c34*s3*s4, c35*s3*s5],
         [c14*s4*s1, c24*s4*s2, c34*s4*s3, s4*s4, c45*s4*s5],
         [c15*s5*s1, c25*s5*s2, c35*s5*s3, c45*s5*s4, s5*s5]]

    np.random.seed(0)
    rand = np.random.multivariate_normal(np.zeros(5), c, n)

    rand[:, 1] = astrometry[1] + (rand[:, 1] * mr)
    rand[:, 0] = astrometry[0] + ((rand[:, 0]/np.cos(rand[:, 1])) * mr)
    rand[:, 2] += astrometry[2]
    rand[:, 3] += astrometry[3]
    rand[:, 4] += astrometry[4]

    return rand

def propagate_epoch_vector(ra0, de0, plx0, mu_ra0, mu_de0, rv0, dt):
    
    # Propagate astrometry by dt years (dt = t-t0)
    # Input and output in radians, km/s, years.
    # All arrays. dt can be a scalar
    
    n = len(ra0)
    zeta0 = rv0*plx0/4.740470446

    ca0 = np.cos(ra0)
    sa0 = np.sin(ra0)
    cd0 = np.cos(de0)
    sd0 = np.sin(de0)

    p0 = [-sa0, ca0, np.zeros(n)]
    q0 = [-sd0 * ca0, -sd0 * sa0, cd0]
    r0 = [cd0 * ca0, cd0 * sa0, sd0]

    #pmv0 = [p0[0] * mu_ra0 + q0[0] * mu_de0, p0[1] * mu_ra0 + q0[1] * mu_de0, p0[2] * mu_ra0 + q0[2] * mu_de0]
    # p0[2] * mu_ra0 = 0.0
    pmv0 = [p0[0] * mu_ra0 + q0[0] * mu_de0, p0[1] * mu_ra0 + q0[1] * mu_de0, q0[2] * mu_de0]

    dt2 = dt * dt
    pm02 = mu_ra0 * mu_ra0 + mu_de0 * mu_de0
    w = 1.0 + zeta0 * dt
    f2 = 1.0 / (1.0 + 2.0 * zeta0 * dt + (pm02 + zeta0*zeta0) * dt2)
    f = np.sqrt(f2)
    f3 = f2 * f
    #f4 = f2 * f2

    plx = plx0 * f
    r = [(r0[0] * w + pmv0[0] * dt) * f, (r0[1] * w + pmv0[1] * dt) * f, (r0[2] * w + pmv0[2] * dt) * f]
    pmv = [(pmv0[0] * w - r0[0] * pm02 * dt) * f3, (pmv0[1] * w - r0[1] * pm02 * dt) * f3, (pmv0[2] * w - r0[2] * pm02 * dt) * f3]
    zeta = ( zeta0 + (pm02 + zeta0 * zeta0) * dt ) * f2

    xy = np.sqrt(r[0]*r[0] + r[1]*r[1])
    p = np.array([-r[1]/xy, r[0]/xy, np.zeros(n)])
    ind = np.where(xy < 1e-9)
    p[0][ind] = 0.0
    p[1][ind] = 1.0
    p[2][ind] = 0.0

    #q = [r[1] * p[2] - r[2] * p[1], r[2] * p[0] - r[0] * p[2], r[0] * p[1] - r[1] * p[0]]
    #p[2] = 0
    q = [-r[2] * p[1], r[2] * p[0], r[0] * p[1] - r[1] * p[0]]


    ra = np.arctan2(-p[0], p[1])
    ra[ra<0.0] += 2.0*np.pi
    de = np.arctan2(r[2], xy)
    rv = zeta*4.740470446/plx
    mu_ra = np.einsum('ji,ji->i', p, pmv)
    mu_de = np.einsum('ji,ji->i', q, pmv)

    return ra, de, plx, mu_ra, mu_de, rv

def tangent_plane_offsets(data, t_hst, dt, ind0, rv_target, n_mc=int(1e6)):
    """

    OUTPUTS:

        sky_pos

        sky_cov

        ra_target: float
          RA of the target at the observation epoch, in [degrees].

        de_target: float
          Declination of the target at the observation epoch, in [degrees].

        plx_target: float
          Parallax of the target at the observation epoch (unitless).
    """

    dr = np.pi/180.0
    mr = (1.0/(1000.0*60*60))*(np.pi/180.0)
    rm = 1.0/mr

    # Calculate position of earth at the HST observation
    earth_position = get_body_barycentric('earth', time.Time(t_hst, format='decimalyear'))
    earth_x = earth_position.x.value
    earth_y = earth_position.y.value
    earth_z = earth_position.z.value

    # Compute reference RA/Dec
    astro = [data['ra'][ind0]*dr, data['dec'][ind0]*dr, data['parallax'][ind0], data['pmra'][ind0], data['pmdec'][ind0]]
    error = [data['ra_error'][ind0], data['dec_error'][ind0], data['parallax_error'][ind0], data['pmra_error'][ind0], data['pmdec_error'][ind0]]
    corrs = [data['ra_dec_corr'][ind0], data['ra_parallax_corr'][ind0], data['ra_pmra_corr'][ind0], data['ra_pmdec_corr'][ind0], data['dec_parallax_corr'][ind0], data['dec_pmra_corr'][ind0], data['dec_pmdec_corr'][ind0], data['parallax_pmra_corr'][ind0], data['parallax_pmdec_corr'][ind0], data['pmra_pmdec_corr'][ind0]]

    np.random.seed(0)
    astro_target = np.transpose(gaia_correlated_variates(astro, error, corrs, n=n_mc))
    rv_target = np.random.normal(rv_target[0], rv_target[1], n_mc)

    result = propagate_epoch_vector(astro_target[0], astro_target[1], astro_target[2]*mr,
                                  astro_target[3]*mr, astro_target[4]*mr, rv_target, dt)

    ra_target, de_target, plx_target = result[0], result[1], result[2]*rm ## result[2] is in radians, so convert to mas

    plx_factor_ra_target = (earth_x*np.sin(ra_target) - earth_y*np.cos(ra_target))
    plx_factor_de_target = (earth_x*np.cos(ra_target)*np.sin(de_target) + \
                     earth_y*np.sin(ra_target)*np.sin(de_target) - earth_z*np.cos(de_target))

    # Lists to store the relative astrometry and covariance matricies
    sky_pos = []
    sky_cov = []

    # Propagate all of the stars in the list to the HST epoch

    for j in range(0, len(data)):
        astro = [data['ra'][j]*dr, data['dec'][j]*dr, data['parallax'][j], data['pmra'][j], data['pmdec'][j]]
        error = [data['ra_error'][j], data['dec_error'][j], data['parallax_error'][j], data['pmra_error'][j], data['pmdec_error'][j]]
        corrs = [data['ra_dec_corr'][j], data['ra_parallax_corr'][j], data['ra_pmra_corr'][j], data['ra_pmdec_corr'][j], data['dec_parallax_corr'][j], data['dec_pmra_corr'][j], data['dec_pmdec_corr'][j], data['parallax_pmra_corr'][j], data['parallax_pmdec_corr'][j], data['pmra_pmdec_corr'][j]]

        mc_draws = np.transpose(gaia_correlated_variates(astro, error, corrs, n_mc))
        # TODO: Use RV of background stars where available
        rv = np.zeros(n_mc)

        # Propagate the background star from 2016.0 to the epoch
        result = propagate_epoch_vector(mc_draws[0], mc_draws[1], mc_draws[2]*mr,
                                      mc_draws[3]*mr, mc_draws[4]*mr, rv, dt)
        
        # as before, save only ra, dec and parallax (radians->mas)
        ra, de, plx = result[0], result[1], result[2]*rm

        
        plx_factor_ra = (earth_x*np.sin(ra) - earth_y*np.cos(ra))
        plx_factor_de = (earth_x*np.cos(ra)*np.sin(de) + \
                         earth_y*np.sin(ra)*np.sin(de) - earth_z*np.cos(de))
        
        xi = (np.cos(de)*np.sin(ra - ra_target))/ (np.sin(de_target)*np.sin(de) + 
                                                      np.cos(de_target)*np.cos(de)*np.cos(ra-ra_target))
        eta = (np.cos(de_target)*np.sin(de) - np.sin(de_target)*np.cos(de)*np.cos(ra-ra_target)) / \
              (np.sin(de_target)*np.sin(de) + np.cos(de_target)*np.cos(de)*np.cos(ra-ra_target))

        xi = (xi*rm + (plx*plx_factor_ra - plx_target*plx_factor_ra_target))
        eta = (eta*rm + (plx*plx_factor_de - plx_target*plx_factor_de_target))  

        sky_pos.append(np.nanmean((xi, eta), axis=1))
        sky_cov.append(np.cov((xi, eta)))

    sky_pos = np.array(sky_pos)
    sky_cov = np.array(sky_cov)

    return sky_pos, sky_cov, np.mean(ra_target)/dr, np.mean(de_target)/dr, np.mean(plx_target)


def confidence_levels(data):

    data /= np.nansum(data)
    cl = [0.9973, 0.9545, 0.6827]
    cl_val = [3.0, 2.0, 1.0]
    confidence = np.copy(data)
    confidence.fill(4.0)

    old_shape = np.shape(confidence)
    confidence = np.ravel(confidence)
    ravel_like = np.ravel(data)
    ind = np.argsort((-1.0) * ravel_like)
    ind2 = np.argsort(ind)
    sorted_like = np.copy(ravel_like)
    sorted_like = np.cumsum(sorted_like[ind])
    confidence = confidence[ind]
    
    # Order like array from most likely to least likely, integrate down from maximum value until sum equals cl_xsig
    for j in range(0, 3):
        ind_bound = np.argmax(sorted_like >= cl[j])
        confidence[0:ind_bound] = cl_val[j]

    confidence = confidence[ind2]
    confidence = np.reshape(confidence, old_shape) 

    return confidence


def get_gaia_id(simbadName):

    result_table = Simbad.query_objectids(simbadName)

    # If the first attempt at querying for the target name in SIMBAD fails,
    # try a couple of other name variations.
    if result_table is None:
        result_table = Simbad.query_objectids(simbadName.replace('-', ' '))

    if result_table is None:
        if simbadName[:2] == 'V-':
            result_table = Simbad.query_objectids(simbadName[2:].replace('-', ' '))

    if result_table is None:
        print('No results found from SIMBAD query for target '\
              f'name {simbadName}. Gaia astrometry cannot be '\
              'performed.')
        return None

    if 'id' in result_table.keys():
        gaiaIDs = [gid for gid in result_table['id'] if 'Gaia' in gid]
    elif 'ID' in result_table.keys():
        gaiaIDs = [gid for gid in result_table['ID'] if 'Gaia' in gid]
    else:
        gaiaIDs = []

    # Prefer Gaia DR3 ID if exists.
    if len(gaiaIDs) > 0:
        for dr in ['DR3', 'DR2']:
            wh = [dr in gid for gid in gaiaIDs]
            if np.any(wh):
                gaiaID = np.array(gaiaIDs)[wh][0]
                return gaiaID
    else:
        print('No Gaia ID found from SIMBAD query for target '\
              f'name {simbadName}. Gaia astrometry cannot be '\
              'performed.')
        return None


def add_header(filepath, ext, ra_target, de_target, plx_target,
               x_gaia, y_gaia, x_err_gaia, y_err_gaia, tn_gaia, tn_err_gaia,
               ps_x_gaia, ps_y_gaia, ps_x_err_gaia, ps_y_err_gaia):

    with fits.open(filepath, mode='update') as ff:
        hdr = ff[ext].header

        hdr['GAIATRA'] = (ra_target, 'Target RA at obs. epoch as Gaia ref (deg)')
        hdr['GAIATDEC'] = (de_target, 'Target Dec at obs. epoch as Gaia ref (deg)')
        hdr['GAIATPLX'] = (plx_target, 'Target parallax at obs. epoch as Gaia ref (mas)')
        hdr['GAIACENX'] = (x_gaia, 'Gaia astrometry star X pixel coordinate')
        hdr['GAIACENY'] = (y_gaia, 'Gaia astrometry star Y pixel coordinate')
        hdr['GAIAERRX'] = (x_err_gaia, 'Gaia astrometry star X pixel 1-sigma error')
        hdr['GAIAERRY'] = (y_err_gaia, 'Gaia astrometry star Y pixel 1-sigma error')
        hdr['GAIATRN'] = (tn_gaia, 'Gaia astrometry true north angle (deg)')
        hdr['GAIATRNE'] = (tn_err_gaia, 'Gaia true north angle 1-sigma error (deg)')
        hdr['GAIAPSX'] = (ps_x_gaia, 'Gaia astrometry X pixel scale (arcsec/pix)')
        hdr['GAIAPSY'] = (ps_y_gaia, 'Gaia astrometry Y pixel scale (arcsec/pix)')
        hdr['GAIAPSEX'] = (ps_x_err_gaia, 'Gaia X pl. scale 1-sigma error (arcsec/pix)')
        hdr['GAIAPSEY'] = (ps_y_err_gaia, 'Gaia Y pl. scale 1-sigma error (arcsec/pix)')

        ff.flush()

    print("Updated header with Gaia center values")

    return
