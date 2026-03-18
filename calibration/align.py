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
        radon_wdw: maximum size of window for Radon function [pix]. In this
            framework, this sets the outer radius of the considered region
            because we force M=1.0.
    """

    # Clean NaN's from data- radon search doesn't like them.
    data = np.nan_to_num(img, 0.)

    # Unsharp mask image- almost always a good idea.
    data = utils.unsharp(None, data.copy(), None, B=40., ident=None, output=True,
                   silent=True, save=False, parOK=True, gauss=False)

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
    x_cen, y_cen = radonCenter.searchCenter(data, cen[1], cen[0], radon_wdw,
                                            m=m, M=1.0, size_cost=4,
                                            theta=spikeAngles, smooth=0)
    refYX = np.array([y_cen, x_cen])

    return refYX


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

    # --- CROSS-PLATFORM FIX ---
    # Validate refYX before casting to int. If NaN or Inf from a failed 
    # star detection, perform a zero-shift to preserve the image data safely.
    if np.any(np.isnan(refYX)) or np.any(np.isinf(refYX)):
        print(f"WARNING: Invalid refYX {refYX} detected. Defaulting to zero-shift.")
        refYX = np.copy(finalYX)

    refYX_round = np.floor(refYX).astype(int)
    finalYX_round = np.floor(finalYX).astype(int)

    if not np.all(finalYX_round == finalYX):
        raise ValueError("HELP!!! Image alignment with align.shift_pix_to_pix "\
                         "ONLY works with final center coordinates (finalYX) "\
                         "that are whole integer pixels, e.g., "\
                         "[1024.0, 1024.0]. Sorry!")

    if outputSize is not None:
        # Do the sub-pixel shift first.
        sh_y, sh_x = refYX_round - refYX
        imgShift_sp = interpolation.shift(img.copy(), [sh_y, sh_x], order=order,
                                         mode='constant', cval=fill)
        imgShift = np.zeros(outputSize) + fill
        # Finally, do the integer-pixel bulk shift by inserting the shifted
        # image into the padded output array.
# FIX ME!!! ONLY WORKS for whole integer finalYX coordinates right now.
        # scipy.ndimage.interpolation.shift uses a spline interpolation of the
        # requested order.
        # Handle cases where the input image will go off the edge of the output
        # array's dimensions.
        if finalYX_round[0]-refYX_round[0]+imgShift_sp.shape[0] > imgShift.shape[0]:
            upper_index_0 = imgShift.shape[0]
            trim_0 = upper_index_0 - (finalYX_round[0]-refYX_round[0]+imgShift_sp.shape[0])
            imgShift_sp = imgShift_sp[:trim_0, :]
        else:
            upper_index_0 = finalYX_round[0]-refYX_round[0]+imgShift_sp.shape[0]

        if finalYX_round[1]-refYX_round[1]+imgShift_sp.shape[1] > imgShift.shape[1]:
            upper_index_1 = imgShift.shape[1]
            trim_1 = upper_index_1 - (finalYX_round[1]-refYX_round[1]+imgShift_sp.shape[1])
            imgShift_sp = imgShift_sp[:, :trim_1]
        else:
            upper_index_1 = finalYX_round[1]-refYX_round[1]+imgShift_sp.shape[1]

        imgShift[finalYX_round[0]-refYX_round[0]:upper_index_0,
                 finalYX_round[1]-refYX_round[1]:upper_index_1] = imgShift_sp
    else:
        sh_y = finalYX[0] - refYX[0]
        sh_x = finalYX[1] - refYX[1]

        # Shift the array to put coords at new_cen.
        # scipy.ndimage.interpolation.shift uses a spline interpolation of the
        # requested order.
        imgShift = interpolation.shift(img, [sh_y, sh_x], order=order,
                                       mode='constant', cval=fill)

    return imgShift

