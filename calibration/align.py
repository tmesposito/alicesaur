#!/usr/bin/env python

import pdb
import numpy as np
from scipy.ndimage import interpolation
# FIX ME!!! Bring radonCenter into this package independently.
from pyklip.instruments.utils import radonCenter

# Internal imports
from alicesaur import utils


def find_star_radon(img, cen, spikeAngles, IWA=25., sp_width=30, r_mask=21.,
                    radon_wdw=500):
    """
    Quick way to do a radon transform search.

    NOTE: radon_wdw=500 may not be the optimal value and might be improved.

    Inputs:
        img: image array
        cen: [pix]
        spikeAngles: [degrees]
        r_mask: radius of central circular mask [pix]
        radon_wdw: size of window for Radon function [pix]
    """

    # Clean NaN's from data- radon search doesn't like them.
    data = np.nan_to_num(img, 0.)

    # Unsharp mask image- almost always a good idea.
    data = utils.unsharp(None, data.copy(), None, B=40., ident=None, output=True,
                   silent=True, save=False, parOK=True, gauss=False)

    # # Smooth image (odd box size probably best)- usually a good idea.
    # data = gaussian_filter(data, 1)

    # DEPRECATED kinda by IWA. Should never set r_mask > IWA.
    # Mask center if wanted.
    if r_mask is not None:
        radii = utils.make_radii(data, cen)
        data[radii < r_mask] = 0.

    # Construct spike mask from fits header information.
    if sp_width != 0:
        spikeMask = ~ utils.make_spikemask_stis(data, cen, spikeAngles, sp_width) # ~ is inverse boolean
        data[spikeMask] = 0.
    else:
        spikeMask = np.zeros(data.shape, dtype=bool)

    # m is multiplicative inner limit and M is multiplicative outer limit.
    # e.g., radon performed over range (-M*radon_wdw, -m*radon_wdw)U(m*radon_wdw, M*radon_wdw),
    # ignoring x range -m*radon_wdw:m*radon_wdw around center and similar for y.
    m = float(IWA)/radon_wdw # set inner limit of sampling region.
    print("m = {:.2f}".format(m))

    # Do radon search.
    x_cen, y_cen = radonCenter.searchCenter(data, cen[1], cen[0], radon_wdw, m=m, M=1.0,
                                            size_cost=4, theta=spikeAngles, smooth=0)

    print("y, x = {}, {}".format(y_cen, x_cen))

    return np.array([y_cen, x_cen])


def shift_pix_to_pix(img, refYX, finalYX=None, outputSize=None, order=3,
                     fill=0.):
    """
    Shift interpolate an image so reference pixel ends up at new coordinates.

    Inputs:
        img: image to shift
        refYX: array of y,x reference coordinates you want to shift [pix].
        finalYX: array of y,x coordinates you want to shift refYX to [pix].
            Default is center of img.
        outputSize: tuple of y,x dimensions for output array to have [pix].
            Default (None) will output the same dimensions as img.
        fill: float
            Value with which to fill empty new pixels.
    """

    # Default final coordinates are center of array.
    if finalYX is None:
        finalYX = np.array(img.shape)//2

    # # Convert all NaNs to 0 (interpolate.shift below won't handle NaNs well).
    # imgClean = np.nan_to_num(img)

    if outputSize is not None:
        mat = np.zeros(outputSize) + fill
        mat[:img.shape[0], :img.shape[1]] = img.copy()
    else:
        mat = img

    sh_y = finalYX[0] - refYX[0]
    sh_x = finalYX[1] - refYX[1]

    # Shift the array to put coords at new_cen.
    imgShift = interpolation.shift(mat, [sh_y, sh_x], order=order,
                                   mode='constant', cval=fill)

    # # If trimming image size, put the star at the center of the trimmed array
    # # so trimmed size is (2*size_out + 1, 2*size_out + 1).
    # if size_out is not None:
    #   im_aligned = im_aligned[new_cen[0]-size_out[0]:new_cen[0]+size_out[0]+1,
    #                           new_cen[1]-size_out[1]:new_cen[1]+size_out[1]+1]
    #   new_cen = size_out

    return imgShift

