#!/usr/bin/env python

import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from astropy.io import fits
from astropy.wcs import WCS
from ..utils import make_radii


def stack_images(imPaths, errorPaths=[], mode='weightedMean', negToNan=False,
                 downweightNeg=False, maskRadii=None, dataInds=None):
    """
    Combinate images by stacking using one of multiple methods given with
    the 'mode' argument.
    
    negToNan: float
        Pixels in individual images below this value will befilled with NaN.
        False (default) will not fill anything.
    mode: str
        Options: ['weightMean' (default), 'simpleMean'].
    maskRadii: list
        Radii within which to mask images listed in imPaths during the
        combination. Use 0 (zero) to indicate no masking.
    """

    data = []
    hdrs = []
    err = []
    hdrs_err = []

    if dataInds is None:
        dataInds = len(imPaths)*[0]

    for ii, ff in enumerate(imPaths):
        with fits.open(ff) as hdul:
            if hdul[0].data.ndim > 2:
                data.append(hdul[0].data[dataInds[ii]])
            else:
                data.append(hdul[0].data)
            hdrs.append(hdul[0].header)

    for ii, ff in enumerate(errorPaths):
        if ff is not None:
            if ff == 'auto':
                ff = os.path.join(os.path.split(imPaths[ii])[0],
                                  'error_' + os.path.split(imPaths[ii])[1])
            with fits.open(ff) as hdul:
                err.append(hdul[0].data)
                hdrs_err.append(hdul[0].header)
        else:
            try:
                exp_total_first = hdrs[0]['TEXPTIME'] # [s]
                exp_total_ii = hdrs[ii]['TEXPTIME'] # [s]
                exp_ratio = np.sqrt(exp_total_first/exp_total_ii)
                errScaled = exp_ratio*err[0]
                errScaled[np.isnan(err[0])] = 1e-12
                err.append(errScaled)
            except:
                err.append(1e-12*np.ones(data[ii].shape))
            hdrs_err.append({})

    data = np.array(data)
    err = np.array(err)

    # dataFilled = data[0].copy()
    # for im in data:
    #     dataFilled[(np.isnan(dataFilled)  & ~np.isnan(im))] = im[(np.isnan(dataFilled) & ~np.isnan(im))]
    # Fill negative values below the provided negToNan level with NaN.
    if negToNan:
        for ii in range(len(data)):
            data[ii][data[ii] < negToNan] = np.nan
        print(f"Filled pixels valued below {negToNan} with NaN")

    # Give minimal weight to negative-valued pixels.
    if downweightNeg:
        err[data < 0] *= 1e3

    simpleMean = np.nanmean(data, axis=0)

# FIX ME!!! Need to implement error-weighted average here.
    if mode == 'simpleMean':
        stack = simpleMean
    else:
        if maskRadii is not None:
            for ii, rad in enumerate(maskRadii):
                if rad > 0:
                    cen = np.array([hdrs[ii].get('PSFCENTY', data[ii].shape[0]//2),
                                    hdrs[ii].get('PSFCENTX', data[ii].shape[1]//2)])
                    radii = make_radii(data[ii], cen)
                    data[ii][radii <= rad] = np.nan
                    # Smooth the edges of the weighting so there isn't a sharp
                    # discontinuity at the mask boundary.
                    errReweight = 10*(((rad+3) - radii[(radii >= rad) & (radii <= rad+3)])/3.)
                    errReweight[errReweight <= 0] = 1e-6
                    err[ii][(radii >= rad) & (radii <= rad+3)] *= errReweight
                    err[ii][radii < rad] = np.nan

        pixAveraged = len(data) - np.sum(np.array([np.isnan(im).astype(float) for im in data]), axis=0)
        inverse_var_sum = np.nansum(1./(err**2), axis=0)
        avg_img = np.nansum(data/(err**2), axis=0) / inverse_var_sum
        # Put NaN back in where everything was NaN before.
        avg_img[np.sum(np.isnan(data), axis=0) == data.shape[0]] = np.nan

        stack = avg_img

        # Work through the image weighting math.
        err_map1 = err[0]
        err_map2 = err[1]
        # img1 / err_map1**2 / (1 / err_map1**2 + 1 / err_map2**2)
        # img1 / err_map1**2 / (err_map2**2 / (err_map1**2 * err_map2**2) + err_map1**2 / (err_map1**2 * err_map2**2))
        # img1 / err_map1**2 / ((err_map1**2 + err_map2**2) / (err_map1**2 * err_map2**2))
        # img1_weighted = img1 * (err_map1**2 * err_map2**2) / (err_map1**2 * (err_map1**2 + err_map2**2)) # correct units of [intensity]

        # err_nonan = np.nan_to_num(err, 1.)

# TEMP!!!
        # # weight1 = (err_map1**2 * err_map2**2) / (err_map1**2 * (err_map1**2 + err_map2**2))
        # # weight2 = (err_map1**2 * err_map2**2) / (err_map2**2 * (err_map1**2 + err_map2**2))

        # Propagate NaNs properly in weight maps.
        # # weight1 should == 1 where wedge image is NaN (close in)
        # # weight1 = np.sqrt(np.nanprod([err_map1**2, err_map2**2], axis=0) / np.nanprod([err_map1**2, np.nansum([err_map1**2, err_map2**2], axis=0)], axis=0)) # [unitless]
        # # weight2 = np.sqrt(np.nanprod([err_map1**2, err_map2**2], axis=0) / np.nanprod([err_map2**2, np.nansum([err_map1**2, err_map2**2], axis=0)], axis=0)) # [unitless]
        # weight1 = np.sqrt(np.nanprod([err_map1**2, err_map2**2], axis=0) / np.nanprod([err_map1**2, (err_map1**2 + err_map2**2)], axis=0)) # [unitless]
        # weight2 = np.sqrt(np.nanprod([err_map1**2, err_map2**2], axis=0) / np.nanprod([err_map2**2, (err_map1**2 + err_map2**2)], axis=0)) # [unitless]

        # # err_avg = np.sqrt(np.nansum([err_map1**2, err_map2**2], axis=0))/(pixAveraged**0.5)
        # err_avg_unweighted = np.sqrt(np.nansum([err_map1**2, err_map2**2], axis=0))/(pixAveraged**0.5)
        # err_avg = np.sqrt(np.nansum([(err_map1*weight1)**2, (err_map2*weight2)**2], axis=0))/(pixAveraged**0.5)

    plt.figure(30)
    plt.clf()
    plt.imshow(stack[924:1124, 924:1124],
               norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=0, vmax=80))
    plt.title(f"Stacked image {mode}")
    plt.draw()
    
    plt.figure(31)
    plt.clf()
    plt.imshow(simpleMean[924:1124, 924:1124],
               norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=0, vmax=80))
    plt.title("Simple mean")
    plt.draw()

    stackPath = os.path.splitext(imPaths[0])[0] + f'_{mode}_stack{len(imPaths)}.fits'
    outHdr = hdrs[0].copy()
    outHdr['FILETYPE'] = 'Averaged image'
    outHdr['PROPAPER'] = 'MULTIPLE'
    del outHdr['COMMENT']
    outHdr['COMBOTYP'] = (mode, "Type of image combination used")
    for ii, ff in enumerate(imPaths):
        outHdr.add_comment(f"Constituent image {ii} = {os.path.split(ff)[-1]}")
    outHdr.add_comment(f"Average type = {mode}")
    outHdr.add_comment(f"Outermost radius masked around star (by image) = {maskRadii}")
    fits.writeto(stackPath, stack, header=outHdr, overwrite=True)

    print(f"Stacked image saved to {stackPath}")

    # plt.show()

    # pdb.set_trace()

    # plt.close()

    return


def zero_pad(data, outsize, method='simple'):

    if (data.shape[0] > outsize[0]) or (data.shape[1] > outsize[1]):
        print(f"***HELP!!! One or more output dimension {outsize} is" \
              f" smaller than the input image dimension {data.shape}. " \
              "Cannot zero-pad this image.")
        return data

    pad_y = (outsize[0] - data.shape[0])//2
    pad_x = (outsize[1] - data.shape[1])//2

    padded = np.zeros(outsize)

    padded[pad_y:pad_y+data.shape[0], pad_x:pad_x+data.shape[1]] = data.copy()

    return padded


def rotate_wcs(header, theta, center_yx):

    # Create a WCS object from the header.
    wcs_orig = WCS(header)
    # Get the original CD matrix. (If only PC and CDELT are present, combine them.)
    if wcs_orig.wcs.has_cd():
        old_cd = wcs_orig.wcs.cd
    else:
        old_cd = np.dot(np.diag(wcs_orig.wcs.cdelt), wcs_orig.wcs.pc)

    # Build the 2x2 rotation matrix R(-theta) in (x, y) coordinates.
    # Convert theta from degrees to radians.
    theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    # Note: R(-theta) = [[cos(theta), sin(theta)],
    #                     [-sin(theta), cos(theta)]]
    # R_wcs = np.array([[cos_theta, sin_theta],
    #                   [-sin_theta, cos_theta]])

    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta,  cos_theta]])
    new_cd = np.dot(old_cd, rotation_matrix)

    # Set the new reference pixel to the chosen center.
    new_crpix = center_yx[::-1]
    # Adjust the reference value so that the world coordinate at the center remains the same:
    new_crval = wcs_orig.wcs.crval + np.dot(old_cd, (new_crpix - wcs_orig.wcs.crpix))

    # Update the original WCS object.
    wcs_orig.wcs.crpix = new_crpix
    wcs_orig.wcs.crval = new_crval
    wcs_orig.wcs.cd = new_cd

    # Generate a new header from the updated WCS.
    new_wcs_header = wcs_orig.to_header()
    # # Update (or replace) the original header keywords with the new WCS keywords.
    # new_header = header.copy()
    # new_header.update(new_wcs_header)

    return new_wcs_header
