#!/usr/bin/env python

import os
import shutil
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import json
from glob import glob
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter, interpolation, filters, map_coordinates
from emcee import EnsembleSampler
from emcee.interruptible_pool import InterruptiblePool
import hickle
from corner import corner


def make_radii(arr, cen):
    """
    Make array of radial distances from centerpoint cen in 2-D array arr.
    """
    # Make array of indices matching arr indices.
    grid = np.indices(arr.shape, dtype=float)
    # Shift indices so that origin is located at cen coordinates.
    grid[0] -= cen[0]
    grid[1] -= cen[1]
    
    return np.sqrt((grid[0])**2 + (grid[1])**2) # [pix]


def make_phi(arr, cen, zeroAxis='+x'):
    """
    Make array of phi angles [radians] from phi=0 at +x or +y axis increasing
    counterclockwise to 2pi, around centerpoint cen in 2-D array arr.
    
    zeroAxis: axis along which to place phi=0; '+x' or '+y'
    """
    # Make array of indices matching arr indices.
    grid = np.indices(arr.shape, dtype=float)
    # Shift indices so that origin is located at cen coordinates.
    grid[0] -= cen[0]
    grid[1] -= cen[1]
    
    phi = np.arctan2(grid[0], grid[1]) # [rad]
    # Place phi=0 along +x axis and range from 0 to 2*pi.
    if zeroAxis == '+x':
        phi[phi < 0] += 2*np.pi # [rad]
    elif zeroAxis == '+y':
        phi -= np.pi/2
        phi[phi < 0] += 2*np.pi # [rad]

    return phi


def make_inclined_ring(R=50, inc=0., dR=2, dims=(1000, 1000),
                       cen=np.array([500, 500])):
    """
    """
    
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    yy = yy - cen[0]
    xx = xx - cen[1]
    
    # theta = np.tan(yy/xx)
    radii = make_radii(np.ones(dims), cen)
    theta = make_phi(np.ones(dims), cen, zeroAxis='+x')
    
    cond = (np.sqrt(xx**2 + yy**2/np.cos(np.radians(inc))**2) >= R - dR) & \
            (np.sqrt(xx**2 + yy**2/np.cos(np.radians(inc))**2) < R + dR)
    
    # yRing = np.sqrt(np.cos(np.radians(inc))**2*(R**2 - xx**2))
    
    # R**2 = xx**2 + yy**2/np.cos(theta)**2
    
    im = np.zeros(dims)
    im[cond] = 1
    
    return im


def shift_im_center(im, old_cen, new_cen, fillval=0., size_out=None):
    """
    Shift a 2D numpy array image to put a specific pixel at its center. Uses
    interpolation via map_coordinates to do the shift.
    
    Inputs:                                                                                         
        im: image 2D numpy array you want to shift.                                                 
        old_cen: [y,x] center of im before the shift [pix].                                         
        new_cen: [y,x] where you want center of im after the shift [pix].
        fillval: constant float value that you want to fill interpolation-created pixels.
        size_out: Not implemented yet.                                                              
                                                                                                    
    Outputs:                                                                                        
        im_sh: the shifted imaged as 2D numpy array.
    
    """
    
 # # FIX ME!!! Not implemented correctly. Supposed to handle trimming/padding of array
 # # simultaneously with the shift.
 #    # Center coords in coord_list at new_cen; default is ref_coords. #middle of output array.
 #    if new_cen is None and size_out is not None:
 #        new_cen = numpy.array(size_out)/2 #ref_coords #numpy.array([size_out[0]/2., size_out[1]/2])
 #    else:
 #        print("No new_cen position or size_out given, so will not shift image. Returning original.")
 #        return im
    
    # Convert all NaNs to 0 (interpolation.shift below won't handle NaNs well).
    im_clean = np.nan_to_num(im)
    
    # Size of shift in [y,x] [pixels].
    sh = np.array(new_cen) - np.array(old_cen)
    
    # Order=1 (linear interpolation?) gives most reliable (if not most accurate)
    # interpolation, particularly for sub-pixel shifts.
    im_sh = interpolation.shift(im, sh, order=1, mode='constant', cval=fillval)
    
#    plt.figure(2)
#    plt.imshow(im_sh)
#    plt.clim(0,1)
    
    return im_sh


def rotate_array(data_raw, cen, theta=0., fill_glob_med=True, fill_loc_med=False,
              preserve_nan=True, cval=0.):
    """
    Function to rotate an array around a given origin by angle theta (degrees).
    Positive theta gives a counterclockwise rotation. If preserve_nan arg is True,
    it will try to preserve NaN's through the rotation.
    
    WARNING: Rotation of NaN elements is approximate and may vary in location
    by +/- 1 pixel.
    
    Input:
        data_raw: 2-d numpy array to be rotated.
        cen: (y,x) coordinates of center around which to rotate arr.
        theta: angle to rotate counterclockwise in [degrees].
        preserve_nan: boolean to propagate NaNs through rotation (True) or replace them
        with median of nearest neighbors (False).
        cval: constant value with which to fill new pixels created by rotation.
    """
    # Set up coordinate map for rotation.
    # Array of indices matching those of data array.
    grid = np.indices((data_raw.shape), dtype=np.float32)
    # Shift indices so that origin is located at cen coordinates.
    # grid[0] = y coordinate, grid[1] = x coordinate in numpy.
    grid[0] -= cen[0]
    grid[1] -= cen[1]

    newgrid = np.empty_like(grid)
    # Rotate y and x coordinates counter-clockwise by angle theta.
    newgrid[0] = np.cos(np.radians(theta))*grid[0] - np.sin(np.radians(theta))*grid[1]
    newgrid[1] = np.sin(np.radians(theta))*grid[0] + np.cos(np.radians(theta))*grid[1]

    # Shift origin back to original coordinates.
    newgrid[0] += cen[0]
    newgrid[1] += cen[1]

    # Mask NaNs in data_raw.
    mdata = np.ma.masked_invalid(data_raw)
    mask = mdata.mask

    # If flag set, then fill NaNs with global finite median.
    if fill_glob_med:
        mdata = mdata.filled(np.median(data_raw[np.isfinite(data_raw)]))

    # If flag set, then fill NaNs with local finite median.
    if fill_loc_med:
        data_medfilt = filters.median_filter(mdata, size=9)
        mdata[mask] = data_medfilt[mask]

    # Rotate the input array according to grid coordinates.
    data_rot = map_coordinates(mdata, newgrid, order=3, cval=cval)

    if preserve_nan & np.any(mask==True):
        # Rotate NaN mask in same way as data.
        # Convert mask from boolean to 1's and 0's for rotation.
        mask_rot = map_coordinates(mask.astype(float), newgrid, order=3)
        # Convert mask back to boolean.
        mask_rot[mask_rot >= 0.5] = True
        mask_rot[mask_rot < 0.5] = False

        # Apply mask to rotated data and fill masked elements with NaN.
        mdata_rot = np.ma.masked_array(data_rot, mask_rot)
        fdata_rot = mdata_rot.filled(np.nan)

        return fdata_rot
    else:
        return data_rot


def rotate_yx(yx, cen, theta=0.):
    """
    Function to rotate a y,x point around a given origin by angle theta (degrees).
    Positive theta gives a counterclockwise rotation.
    
    Input:
        yx: ([y,x]) numpy array of point coordinates to rotate.
        cen: (y,x) coordinates of center around which to rotate.
        theta: angle to rotate counterclockwise in [degrees].
    """
    # Shift indices so that origin is located at cen coordinates.
    Y = yx[0] - cen[0]
    X = yx[1] - cen[1]

    # Rotate y and x coordinates counter-clockwise by angle theta.
    rotY = np.cos(np.radians(theta))*Y - np.sin(np.radians(theta))*X
    rotX = np.sin(np.radians(theta))*Y + np.cos(np.radians(theta))*X

    # Shift origin back to original coordinates.
    rotY += cen[0]
    rotX += cen[1]

    return np.array([rotY, rotX])


def make_1d_gauss(xx, mu, sig, C=1):
    """
    Compute a 1-d Gaussian function.
    xx= range for variable.
    mu= mean for variable.
    sig= sigma for variable.
    C= scalar multiplicative amplitude for variable.
    """
    N = 1./(sig*np.sqrt(2*np.pi))
    
    return C*N*np.exp(-(xx - mu)**2/(2*sig**2))


def make_double_1d_gauss(xx, mu1, sig1, mu2, sig2, C1=1, C2=1):
    """
    Compute two 1-d Gaussian functions and add them together.
    xx= range for variable.
    mu1= Gaussian 1 mean for variable.
    sig1= Gaussian 1 sigma for variable.
    mu2= Gaussian 2 mean for variable.
    sig2= Gaussian 2 sigma for variable.
    C1= Gaussian 1 scalar multiplicative amplitude for variable.
    C2= Gaussian 2 scalar multiplicative amplitude for variable.
    """
    N1 = 1./(sig1*np.sqrt(2*np.pi))
    N2 = 1./(sig2*np.sqrt(2*np.pi))
    
    return C1*N1*np.exp(-(xx - mu1)**2/(2*sig1**2)) + C2*N2*np.exp(-(xx - mu2)**2/(2*sig2**2))


def weighted_mean_1d(data, err, n=1):
    # Pad out data to allow equal-sized bins; last bin might contain fewer measurements.
    dataPad = np.pad(data.astype(float), (0, ((n - data.size%n) % n)), mode='constant',
                     constant_values=np.NaN).reshape(-1, n)
    errPad = np.pad(err.astype(float), (0, ((n - err.size%n) % n)), mode='constant',
                     constant_values=np.NaN).reshape(-1, n)
    n_data = np.array([np.sum(~np.isnan(ii)) for ii in dataPad], dtype=float)
    # Weighted mean and associated uncertainty.
    meanWeighted = np.nansum(dataPad*(1./errPad**2), axis=1)/np.nansum(1./errPad**2, axis=1)
    meanWeightedErr = 1./np.sqrt(np.nansum(1./errPad**2, axis=1))
    # # Trim off last binned point if it only contained one measurement.
    # if n_data[-1] <= 1:
    #     meanWeighted = meanWeighted[:-1]
    #     meanWeightedErr = meanWeightedErr[:-1]

    return meanWeighted, meanWeightedErr


def make_spikemask_stis(img, ctr, angles, width=10.):
    """
    Construct diffraction spike mask from fits header information.

    img=
    ctr= center coordinates of diffraction spike pattern (usually star loc).
    angles= angles of spikes in [degrees].
    width= width of spike mask; 0 or no mask; int or flt [pix].
    """

    y,x = np.mgrid[:img.shape[0], :img.shape[1]]

    if width == 0.:
        mask = np.zeros((y.shape[0], x.shape[1]), dtype=bool)
    else:
        yy = y - ctr[0]
        xx = x - ctr[1]
        mask = np.zeros((yy.shape[0], yy.shape[1]), dtype=bool)

        for ang in np.radians(angles):
            # Slope times x for each spike.
            mx = np.tan(ang)*xx
            # Mask with span 0.5*width above and below spike.
            band = np.abs(0.5*width/np.cos(ang))
            # Change spikemask to True anywhere inside mask.
            spikemask = (yy <= (mx + band)) & (yy >= (mx - band))
            # | is logical "or" operator; replaces False with True in mask wherever spikemask or mask is True.
            mask = mask | spikemask
    
    return mask


def median_patch(im, cen, size):
    """
    Generic median measurement of pixels within a given patch.

    cen: int array of y,x coords for center of background patch measurement.
    size: single int for a circular radius, or two ints for the full height and
        and width of a rectangle.
    """

    # Circular patch.
    if type(size) in [int, float]:
        radii = make_radii(im, cen)
        patch = im[radii <= size]
    # Rectangular patch.
    else:
        patch = im[cen[0]-size[0]//2:cen[0]+size[0]//2+1, cen[1]-size[1]//2:cen[1]+size[1]//2+1]

    med = np.nanmedian(patch)

    return med


def subtract_bg(im, cen, size):
    """
    Generic background/sky subtraction using a manually selected patch.
    
    cen: int array of y,x coords for center of background patch measurement.
    size: single int for a circular radius, or two ints for the full height and
        and width of a rectangle.
    """

    bg = median_patch(im=im, cen=cen, size=size)

    try:
        assert not np.isnan(bg)
    except:
        print("\nWARNING! Background estimate is NaN, so no background subtraction will be performed.")
        return im, bg

    imSub = im - bg

    # test = im.copy()
    # test[radii >= size] = 1e8
    # plt.figure(4)
    # plt.clf()
    # plt.imshow(test, vmin=-1, vmax=1)
    # plt.draw()
    # pdb.set_trace()
    return imSub, bg


def randomly_sample_bg(im, excludeYX=[], bgRadius=30, exclusionRadius=200,
                       mask=None):
    """

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.
    exclude_yx : TYPE, optional
        DESCRIPTION. The default is [].
    bgRadius : int, optional
        DESCRIPTION. The default is 30.
    exclusionRadius : TYPE, optional
        DESCRIPTION. The default is 200.
    mask : bool ndarray, optional
        Boolean array that is True whereever the image should be masked and
        False everywhere else. The default of None will apply no masking.

    Returns
    -------
    None.

    """

    # Space out a bunch of sample locations, then exclude any that are masked
    # or too close to the exclusion points.
    imMasked = im.copy()
    imMasked[mask] = np.nan
    # Add masking of the exclusion region.
    for exYX in excludeYX:
        radii = make_radii(im, exYX)
        imMasked[radii <= exclusionRadius] = np.nan
    whGood = np.where(~np.isnan(imMasked))
    yRange = (int(min(whGood[0]) + bgRadius), int(max(whGood[0]) - bgRadius))
    xRange = (int(min(whGood[1]) + bgRadius), int(max(whGood[1]) - bgRadius))
    sampleYs = np.arange(yRange[0], yRange[1], int(2*bgRadius))
    sampleYXs = zip(sampleYs, np.linspace(xRange[0], xRange[1], len(sampleYs),
                                          dtype=int))
    goodYXs = []
    for yx in sampleYXs:
        if np.isnan(imMasked[yx]):
            continue
        radii = make_radii(im, yx)
        if np.sum(np.isnan(imMasked[radii <= bgRadius])) > 0.5*(np.pi*bgRadius**2):
            continue
        else:
            goodYXs.append(yx)

    # Measure the median inside each sample region and return the median
    # of those medians as the background estimate.
    bgs = []
    for yx in goodYXs:
        bgs.append(median_patch(im=imMasked, cen=yx, size=bgRadius))

    bg_med = np.nanmedian(bgs)

    return bg_med


def unsharp(ds, dataHDU, fl, B=30., ident='_999', output=True, silent=True,
            save=True, parOK=True, gauss=False):
    """
    Unsharp mask all individual images in data using median boxcar filter.
    Tries to run parallelized version (# processes = # processors available) first
    and runs a non-parallel version if that fails (much slower).
    SLOW when doing large median filter boxes (>30 pix).

    Inputs:
        ds= dataset name; str
        dataHDU= HDU for data to be unsharp masked OR str filename of data.
        B = width of median boxcar or sigma parameter of gaussian.
        output= if True, return unsharp masked cube as output; if False, return nothing.
        parOK= if True, allows parallelization; if False, forces single process.
        gauss= if True, use a Gaussian smooth instead of median boxcar.

    Output:
        unsharp_cube: the unsharp-masked data cube with same shape as input data.
        Also saves the unsharp cube to disk (but will not overwrite existing).
    """

    path = os.path.expanduser('~/Research/data/%s/processed/' % ds)

    if gauss:
        filt_func = gaussian_filter_img
    else:
        filt_func = median_filter_img

    # Handle filepath input.
    if type(dataHDU)==str:
        dataHDU = fits.open(path + '%s.fits' % dataHDU)

    # Handle simple numpy array input.
    if type(dataHDU)==np.ndarray:
        data = dataHDU
    else:
        dataHdr = dataHDU[0].header
        data = dataHDU[0].data

    print("\nApplying filter (size=%.2f pixels) to each image..." % B)

    if data.ndim >= 3:
        quarters = np.linspace(0, len(data), 5, dtype=int)
    else:
        quarters = [-1] # skip progress updates for single arrays

    if parOK:
        try:
            # Parallelized filtering. Prefer this version.
            # Default is to set number of processes to number of processors available.
            # No speed gain past the point where ~1 worker per processor (with more workers than
            # processors, the median filter just takes longer for each worker).
            ncpus = np.min((8, mp.cpu_count))
            # pool = mp.Pool(ncpus)
            # pool = mp.Pool(ncpus, init_worker)
            
 # Experimental pool that likes KeyboardInterrupts better.
            pool = InterruptiblePool(processes=ncpus, initializer=init_worker) #, initargs=(), **kwargs)
    
            # Put the filtering jobs into the queue.
            result_queue = [pool.apply_async(filt_func, args=(data[ii], B, ii, quarters)) for ii in range(data.shape[0])]
            # result_queue = [pool.amap(filter_data, args=(data[ii], B, ii)) for ii in range(data.shape[0])]
            
            # Run the jobs in the queue and put the results into an array.
            # Output images in data_filt have same order as input images in data.
            data_filt = np.array([result.get() for result in result_queue])
            
            # Close the multiprocessing pool.
            pool.close()

        except:
            # Non-Parallel version. SLOWER but more stable.
            print("Unsharp NOT parallelized- this may take some time...")
            data_filt = np.array([filt_func(data[ii], B, ii, quarters) for ii in range(data.shape[0])])

        try:
            # Close multiprocessing pool if it was opened but interrupted before closing.
            if pool:
                pool.close()
                pool.join()
                print("Closed the interrupted multiprocessing pool.")
        except:
            pass

    else:
        data_filt = np.array([filt_func(data[ii], B, ii, quarters) for ii in range(data.shape[0])])

    # Unsharp mask the data by subtracting the filtered images from the originals.
    print("Performing unsharp mask...\n")
    unsharp_cube = data - data_filt

    if save:
        try:
            if type(save) == str:
                save_name = save
                path = ''
            else:
                data_fn = dataHdr['FILENAME'][:-5]
                if gauss:
                    save_name = 'unsharp_%s%s_g%.1f.fits' % (data_fn, ident, B)
                else:
                    save_name = 'unsharp_%s%s_%.1f.fits' % (data_fn, ident, B)
        
            if os.path.isfile(path + save_name): raise IOError('File already exists')
            temp = fits.PrimaryHDU(data=unsharp_cube.astype('float32'), header=dataHdr)
            # temp = headerWrite(temp, ds, None)
            temp.header['FILETYPE'] = 'unsharp mask cube'
            temp.header['FILENAME'] = save_name
            # temp.header['OUTDIR'] = '%s%s.fits' % (data_fn, ident)
            temp.header['FRAMENO'] = '%18s' % 'multiple'
            if gauss:
                temp.header['HPTYPE'] = ('Gaussian', 'Type of highpass filter')
            else:
                temp.header['HPTYPE'] = ('Median', 'Type of highpass filter')
            del temp.header['COMMENT']
            try:
                temp.header.add_comment('= unsharp mask (ie, high-pass filtered), size=%.2f' % B, after='HISTORY')
            except:
                temp.header.add_comment('= unsharp mask (ie, high-pass filtered), size=%.2f' % B)
            temp.writeto(path + save_name)
            print("Unsharp masked cube saved as", path + save_name)
        except IOError:
            print("WARNING: File already exists at", path + save_name, ", unsharp masked cube NOT SAVED to disk.")

    if output:
        return unsharp_cube


def median_filter_img(im, B, ii=None, quarters=None):
    """
    Filter a 2D image with a median boxcar filter.
    
    Inputs:
        im: 2D image (NaN's and inf's OK)
        B: int, size of the boxcar box [pix]
        ii: (optional) index of the image in a sequence (for progress only)
        quarters: (optional)
        # silent: if False, will print progress to standard out
        
    Output:
        im_filt: the filtered image with NaN's put back in to original locations
    """
    # Filter each image in data and store in a cube.
    # if not silent: print "Filtering image %d..." % ii
    if ii in quarters[1:]:
        print("~%d images filtered (%.1f%%)" % (ii, 100*ii/quarters[-1]))

    wh = np.where(np.isnan(im))
    im[wh] = np.median(im[np.isfinite(im)])
    # #boxcar = filters.median_filter(image, size=(b,b), mode='reflect')
    
    # NOTE: scipy.signal.medfilt is way slower than scipy.ndimage.filters.median_filter.
    # im_filt = filters.gaussian_filter(im, sigma=B)
    im_filt = median_filter(im, int(B)) # B must be integer
    # Replace NaN into filtered image.
    im_filt[wh] = np.nan
    
    return im_filt


def gaussian_filter_img(im, B, ii=None, quarters=None):
    """
    Filter a 2D image with a median boxcar filter.
    
    Inputs:
        im: 2D image (NaN's and inf's OK)
        B: int, size of the boxcar box [pix]
        ii: (optional) index of the image in a sequence (for progress only)
        quarters: (optional)
        # silent: if False, will print progress to standard out
        
    Output:
        im_filt: the filtered image with NaN's put back in to original locations
    """
    # Filter each image in data and store in a cube.
    # if not silent: print "Filtering image %d..." % ii
    if ii in quarters[1:]:
        print("~%d images filtered (%.1f%%)" % (ii, 100*ii/quarters[-1]))
    
    wh = np.where(np.isnan(im))
    im[wh] = np.median(im[np.isfinite(im)])
    
    im_filt = gaussian_filter(im, sigma=B)
    # Replace NaN into filtered image.
    im_filt[wh] = np.nan
    
    return im_filt


def combine_final_fits(fl):

    data = []
    hdrs = []

    for ff in fl:
        with fits.open(ff) as hdul:
            data.append(hdul[0].data)
            hdrs.append(hdul[0].header)

    dataMean = np.nanmean(data, axis=0)

    dataFilled = data[0].copy()
    for im in data:
        dataFilled[(np.isnan(dataFilled)  & ~np.isnan(im))] = im[(np.isnan(dataFilled) & ~np.isnan(im))]

    plt.figure(30)
    plt.clf()
    plt.imshow(dataFilled[924:1124, 924:1124],
               norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=0, vmax=80))
    plt.draw()

    fits.writeto('HD-146897_2020-05-20_stis_BAR10_filled.fits', dataFilled)

    return


def make_tmp_copy(filePath, suffix='tmp', verbose=True):

    origBase, origExt = os.path.splitext(filePath)
    tmpPath = os.path.join(origBase + suffix, origExt)

    shutil.copyfile(filePath, tmpPath)
    if verbose:
        print(f"Copied tmp file to {tmpPath}")

    return


def check_mkdir(newDir):
    """Check for and make directory.

    Function to check for a directory and make it if it doesn't exist.

    Parameters
    ----------
    newDir : str
        Path of directory to check for/make.

    """

    # Check for directory first.
    if os.path.isdir(newDir):
        print(f'Directory already exists: {newDir}')
    elif os.path.exists(newDir):
        print(f'File already exists at: {newDir}')
    else:
        # Make directory.
        os.makedirs(newDir, 0o774)
        print(f'Created directory: {newDir}')



def generic_mcmc(func_model, pkeys, data, dataErr, xx=None, p0=None, priors=None,
                 nwalkers=100, niter=1000, nburn=0, nthreads=1, plot=True,
                 log_path='', sIdent='999'):
    """
    Inputs:

      func_lnlike: function
        Function used by the sampler to compute the natural log likelihood.

      pkeys: list or array
        String names of parameters varied by the sampler.

      data: array
        The data to be compared with a model by the sampler.

      dataErr: array
        The uncertainties on the data.

      p0: arrays with dimensions of nwalkers x number of free parameters
        Array with initial positions for each parameter for each walker.

      priors: dict
        Dict containing bounds for a flat (uniform) prior, with pkeys strings
        for keys. Each value should be a tuple or list of prior lower bound
        followed by prior upper bound. If priors=None, no prior will be used.

      nwalkers: int
        Number of walkers for the MCMC sampler.

      niter: int
        Number of iterations per walker to go into the final posterior
        distribution sampling. This does not included nburn.

      nburn: int
        Number of "burn-in" iterations per walker to run at the start of the
        sampling and then discard. These iterations will not be included in
        the final posterior distribution sampling.

      nthreads: int
        Number of parallel processes to run during sampling. Default is 1.

      log_path: str
        Relative path to a directory inside which MCMC logs and files will be
        saved. Default (empty string) is the current working directory.

      plot: bool
        True (default) to output trace and corner plots.

      sIdent: str
        String identifier for the current MCMC run, to be appended to filenames
        for tracking results.
    """

    def mc_lnlike(pl, pkeys, priors, data, err, xx):
        
        # For affine-invariant ensemble sampler, run the prior test here.
        if not np.isfinite(mc_lnprior(pl, pkeys, priors)):
            return -np.inf
        
        # Basic chi2 residuals using the log10 power-law function.
        res = (data - func_model(xx, pl))/err
        
        return -0.5*(np.nansum(res**2))

    def mc_lnprior(pl, pkeys, priors):
        """
        Define the flat prior boundaries.
        Takes parameter list pl and parameter keys pkeys as inputs.
        
        Inputs:
            pl: array of parameter values (must be in same order as pkeys).
            pkeys: array of sorted str pkeys (must be in same order as pl).
            priors: dict with same keys as pkeys (but order doesn't matter).
                If None, all models will pass.
        
        Returns 0 if successful, or -infinity if failure.
        """
        
        if priors is not None:
            for ii, key in enumerate(pkeys):
                if priors[key][0] < pl[pkeys==key] < priors[key][1]:
                    continue
                else:
                    return -np.inf
        
        # If get to here, all parameters pass the prior and returns 0.
        return 0.


    if type(pkeys) == list:
        pkeys = np.array(pkeys)

    ndim = len(pkeys)

    # Create MCMC sampler object.
    sampler = EnsembleSampler(nwalkers, ndim, mc_lnlike,
                              args=[pkeys, priors, data, dataErr, xx],
                              threads=nthreads)

    if nburn > 0:
        print("\nBURN-IN START...\n")
        for bb, (pburn, lnprob_burn, lnlike_burn) in enumerate(sampler.sample(
                                         p0, iterations=nburn, progress=True)):
            # Print progress every 25%.
            if bb in [nburn/4, nburn/2, 3*nburn/4]:
                print("PROCESSING ITERATION %d; BURN-IN %.1f%% COMPLETE..." % (bb, 100*float(bb)/nburn))
            pass
    
        # Print burn-in autocorrelation time and acceptance fraction.
        try:
            max_acl_burn = np.nanmax(sampler.acor) # fails if too few iterations
        except:
            max_acl_burn = -1.
        print("Largest Burn-in Autocorrelation Time = %.1f" % max_acl_burn)
        print("Mean, Median Burn-in Acceptance Fractions: %.2f, %.2f" % (np.mean(sampler.acceptance_fraction), np.median(sampler.acceptance_fraction)))
        
        # Reset the sampler chains and lnprobability after burn-in.
        sampler.reset()
    
        print("BURN-IN COMPLETE!")
    
    # elif (nburn==0) & (init_samples_fn is not None):
    #   print("Walkers initialized from file and no burn-in samples requested.")
    #   sampler.reset()
    #   pburn = p0
    #   lnprob_burn = None #lnprob_init 
    #   lnlike_burn = None #lnlike_init
    else:
        print("No burn-in samples requested.")
        pburn = p0
        lnprob_burn = None
        lnlike_burn = None
    
    
    ############################
    # ------ MAIN PHASE ------ #
    
    print("\nMAIN-PHASE MCMC START...\n")

    for nn, (pp, lnprob, lnlike) in enumerate(sampler.sample(pburn,
                            log_prob0=lnprob_burn, iterations=niter, tune=True,
                            progress=True, skip_initial_state_check=True)):
        # Print progress every 25%.
        if nn in [niter/4, niter/2, 3*niter/4]:
            # print("PROCESSING ITERATION %d; MCMC %.1f%% COMPLETE..." % (nn, 100*float(nn)/niter))
            # Log the full sampler or chain (all temperatures) every so often.
            try:
                # Delete some items from the sampler that don't hickle well.
                sampler_dict = sampler.__dict__.copy()
                for item in ['pool', 'lnprobfn', 'runtime_sortingfn', '_random', 'kwargs']:
                    try:
                        sampler_dict.__delitem__(item)
                    except: #KeyError
                        continue
                hickle.dump(sampler_dict, os.path.join(log_path, '%s_mcmc_full_sampler.hkl' % sIdent), mode='w')
                print("Sampler logged at iteration %d." % nn)
            except:
                hickle.dump(sampler.chain, os.path.join(log_path, '%s_mcmc_full_chain.hkl' % sIdent), mode='w')
    
    
    print('\nMCMC RUN COMPLETE!\n')
    
    ch = sampler.chain
    samples = ch[:, :, :].reshape((-1, ndim))
    lnprob_out = sampler.lnprobability # all chi-squareds
    
    # Max likelihood params values.
    ind_lk_max = np.where(lnprob_out==lnprob_out.max())
    lk_max = np.e**lnprob_out.max()
    params_ml_mcmc = dict(zip(pkeys, ch[ind_lk_max][0]))
    params_ml_mcmc_sorted = [val for (key, val) in sorted(params_ml_mcmc.items())]
    
    # Get median values (50th percentile) and 1-sigma (68%) confidence intervals
    # for each parameter (in order +, -).
    params_med_mcmc = list(map(lambda vv: (vv[1], vv[2]-vv[1], vv[1]-vv[0]),
                        np.percentile(samples, [16, 50, 84], axis=0).T))
    
    print("\nMax-Likelihood Param Values:")
    for kk, key in enumerate(pkeys):
        print(key + ' = %.3e' % params_ml_mcmc[key])

    print("\n50%-Likelihood Param Values (50th percentile +/- 1 sigma (i.e., 34%):")
    for kk, key in enumerate(pkeys):
        print(key + ' = %.3f +/- %.3f/%.3f' % (params_med_mcmc[kk][0], params_med_mcmc[kk][1], params_med_mcmc[kk][2]))
    
    if plot:
        fontSize = 12
        
        fig = plt.figure(50)
        fig.clf()
        for aa, ff in enumerate(range(0, min(6, ndim), 1)):
            sbpl = "32%d" % (aa+1)
            ax = fig.add_subplot(int(sbpl))
            for ww in range(nwalkers):
                ax.plot(range(0,niter), ch[ww,:,ff], 'k-', alpha=10./nwalkers)
            if pkeys[ff] != 'None':
                ax.set_ylabel(r'%s (%d)' % (pkeys[ff], ff), fontsize=fontSize+2)
            if aa%2==1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
            ax.tick_params(labelsize=fontSize+1)

        # Remove xtick labels from all but bottom panels.
        for ax in fig.get_axes()[:-2]:
            ax.set_xticklabels(['']*ax.get_xticklabels().__len__())
        fig.suptitle('Walkers, by iteration', fontsize=fontSize+3)
        fig.subplots_adjust(0.16, 0.06, 0.84, 0.92, wspace=0.05, hspace=0.1)
        plt.draw()
        
        nthin = 1 # ignore thinning for now
        fontsize_tri = 10.
        contour_colors = 'k'
        # Sorting indices for keys.
        tri_sort = []
        tri_incl = []
        pkeys_tri = pkeys
        for ii, key in enumerate(pkeys_tri):
            if key in pkeys:
                tri_sort.append(np.where(pkeys==key)[0][0])
                tri_incl.append(key)
        
        labels_tri = pkeys
        range_tri = None
        
        # New array of triangle plot pkeys.
        tri_incl = np.array(tri_incl)
        # Thin out samples for triangle plot by only plotting every nthin sample.
        samples_tri = samples[::nthin, tri_sort]
        fig_tri = corner(samples_tri, labels=labels_tri, quantiles=[0.16, 0.5, 0.84],
                            label_kwargs={"size":fontsize_tri},
                            show_titles=False, verbose=True, max_n_ticks=3,
                            plot_datapoints=True, plot_contours=True, fill_contours=True,
                            range=range_tri, plot_density=False,
                            data_kwargs={'color':'0.6', 'alpha':0.2, 'ms':1.},
                            contour_kwargs={'colors':contour_colors})
        if plot:
            try:
                fig.savefig(os.path.join(log_path, '%s_mcmc_walkers.png' % sIdent),
                                format='png', dpi=300)
            except:
                print("Failed to save walker plot to file")
            try:
                fig_tri.savefig(os.path.join(log_path, '%s_mcmc_corner.png' % sIdent),
                                format='png', dpi=300)
            except:
                print("Failed to save corner plot to file")
        
    return params_med_mcmc, params_ml_mcmc, sampler


def get_ann_stdmap(im, cen, radii, r_max=None, mask_edges=False, use_mean=False, use_median=False, rprof_out=False):
    """
    Get standard deviation map from image im, measured in concentric annuli 
    around cen. NaN's in im will be ignored (use for masking).
    
    Inputs:
        im: image from which to calculate standard deviation map.
        radii: array of radial distances for pixels in im from cen.
        r_max: maximum radius to measure std dev.
        mask_edges: False or int N; mask out N pixels on all edges to mitigate edge effects
            biasing the standard devation at large radii.
        use_mean: if True, use mean instead of standard deviation.
        use_median: if True, use median instead of standard deviation.
        rprof_out: if True, also output the radial profile.
    """
    
    if r_max==None:
        r_max = radii.max()
    
    if mask_edges:
        cen = np.array(im.shape)//2
        mask = np.ma.masked_invalid(gaussian_filter(im, mask_edges)).mask
        mask[cen[0]-mask_edges*5:cen[0]+mask_edges*5, cen[1]-mask_edges*5:cen[1]+mask_edges*5] = False
        im = np.ma.masked_array(im, mask=mask).filled(np.nan)
    
    stdmap = np.zeros(im.shape, dtype=float)
    rprof = []
    for rr in np.arange(0, r_max, 1):
        # Central pixel often has std=0 b/c single element. Avoid this by giving
        # it same std dev as 2 pix-wide annulus from r=0 to 2.
        if rr==0:
            wr = np.nonzero((radii >= 0) & (radii < 2))
            if use_mean:
                val = np.nanmean(im[wr])
                stdmap[cen[0], cen[1]] = val
            elif use_median:
                val = np.nanmedian(im[wr])
                stdmap[cen[0], cen[1]] = val
            else:
                val = np.nanstd(im[wr])
                stdmap[cen[0], cen[1]] = val
        else:
            wr = np.nonzero((radii >= rr-0.5) & (radii < rr+0.5))
            #stdmap[wr] = np.std(im[wr])
            if use_mean:
                val = np.nanmean(im[wr])
                stdmap[wr] = val
            elif use_median:
                val = np.nanmedian(im[wr])
                stdmap[wr] = val
            else:
                val = np.nanstd(im[wr])
                stdmap[wr] = val
        rprof.append(val)
    
    if rprof_out:
        return stdmap, [np.arange(0, r_max, 1), rprof]
    else:
        return stdmap


def get_partialann_stdmap(im, cen, radii, phi, phi_range, r_max=None, rdelta=1,
                          ignoreNan=True):
    """
    Get standard deviation map from image im, measured in concentric annuli 
    around cen and within regions of phi. No NaN's allowed in im or radii.
    
    radii= array of radial distances for pixels in im from cen.
    phi=
    phi_range= [degrees]; 
    r_max= maximum radius to measure std dev.
    rdelta: radial width of each annulus, in [pixels].
    """
    if r_max==None:
        r_max = radii.max()
    
    dtor = np.pi/180
    if phi_range is not None:
        if len(phi_range)==2:
            phi_cond = ((phi > phi_range[0]*dtor) & (phi < phi_range[1]*dtor))
        else:
            phi_cond = ((phi > phi_range[0]*dtor) & (phi < phi_range[1]*dtor)) | ((phi > phi_range[2]*dtor) & (phi < phi_range[3]*dtor))
    else:
        phi_cond = np.ones(im.shape, dtype=bool)
    
    stdmap = np.zeros(im.shape, dtype=float)
    # for rr in np.arange(1, r_max, rdelta):
    for rr in np.arange(1, r_max, 1):
        # Region based on radius criteria only.
        wr = np.nonzero((radii >= rr-0.5*rdelta) & (radii < rr+0.5*rdelta))
        # Region with phi sections excluded.
        wp = np.nonzero((radii >= rr-0.5*rdelta) & (radii < rr+0.5*rdelta) & phi_cond)
        # stdmap[wr] = np.std(im[wp])
        stdmap[wr] = np.nanstd(im[wp])
    
    return stdmap


def stmag_to_flux(stmag):
    
    return 10**((stmag + 21.1)/-2.5)


def vmag_to_stmag(vmag, lambda_angs=None, stmag_lambda=None):
    """
    Convert from V magnitude to STMAG based on wavelength and temp for source.
    Based on https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-2-using-the-information-in-this-chapter#id-13.2UsingtheInformationinthisChapter-Table13.1

    Parameters
    ----------
    vmag : TYPE
        DESCRIPTION.
    lambda_angs : TYPE
        DESCRIPTION.
    stmag_lambda : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    stmag : TYPE
        DESCRIPTION.

    """
    
    if stmag_lambda is None:
        print("HELP!!! stmag_lambda lookup based on lambda_angs is not implemented yet. Supply an stmag_lambda as input instead.")

    stmag = vmag + stmag_lambda
    
    return stmag


def load_info_json(infoDir):
    """
    Load dataset info and reduction parameters from an info.json file.
    """
    try:
        infos = glob(os.path.join(infoDir, "info.json"))
        if len(infos) < 1:
            infos = glob(os.path.join(infoDir, "../info.json"))
        with open(infos[0]) as ff:
            info = json.load(ff)
        infoPath = os.path.normpath(infos[0])
        print("\nLoaded info from {}".format(infoPath))
    except:
        info = {}
        infoPath = ''
        print(f"\nFailed to load info.json from directory {infoDir}. "
              + "Returning an empty dict.")

def format_target_name(targetName):

    if targetName[:2].lower() == 'hd':
        formattedName = 'HD' + targetName[2:].replace('-', ' ')
    else:
        print(f"WARNING: target name formatting not set up for {targetName}")
        formattedName = targetName

    return formattedName
