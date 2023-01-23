#!/usr/bin/env python

import pdb
import numpy as np

# Internal imports
from alicesaur import utils


def mask_exclusions(im=None, mask=None, exclusions={}, cen=None,
                    cenOffset=None, paOffset=0.,
                    spikeAngles=np.array([45., 135.])):
    """
    Combine a series of masks based on coordinates and sizes of circles around
    point sources, PA wedges, rectangles, and diffraction spike shapes.
    """

    if mask is not None:
        newMask = mask.copy()
    else:
        newMask = np.zeros(im.shape)
    if cen is None:
        cen = np.array(newMask.shape)//2

    for excl in exclusions.setdefault('pa_deg', []):
        newMask = mask_pa(newMask.copy(), excl, cen, cenOffset, paOffset)
    for excl in exclusions.setdefault('rect_cenYX_widthYX_angleDeg', []):
        newMask = mask_rect(newMask.copy(), excl, cen, cenOffset, paOffset)
    for excl in exclusions.setdefault('point_yxr', []):
        newMask = mask_point(newMask.copy(), excl, cen, cenOffset, paOffset)
    # plt.figure(4)
    # plt.imshow(alignImgs[ind]*newMask, vmin=0, vmax=1)
    # plt.draw()
    # pdb.set_trace()
    # for excl in exclusions.setdefault('spikes_yxr', []):
    #     newMask = mask_spikes_offaxis(newMask.copy(), excl, cen, cenOffset, paOffset,
    #                                     spikeAngles)

    return newMask


def mask_pa(mask, paRange, cen, cenOffset=None, paOffset=0.):
    """
    paRangeList: list of [theta_min, theta_max) PA pairs between which will be masked.
        Should be measured east of north (PA=0 is at +y axis) in [degrees].
    cenOffset: 1x2 array, Y,X [pix] offset between cen and reference center of
        the exclusion Y,X coordinate system. E.g., cenOffset is needed if
        masked image is padded compared to the image the exclusions were
        located from.
    """

    addMask = np.ones(mask.shape)

    try:
        theta = utils.make_phi(mask, cen, zeroAxis='+y')
        addMask[(theta >= np.radians(paRange[0])) & (theta < np.radians(paRange[1]))] = np.nan
        addMask = utils.rotate_array(addMask, cen, paOffset,
                               preserve_nan=True, cval=1)
        addMask *= mask
        addMask = np.nan_to_num(addMask, nan=1, copy=True)
    except Exception as ee:
        print(ee)
        addMask = mask

    return addMask


def mask_rect(mask, rectExclude, cen, cenOffset=None, paOffset=0.):
    """
    cenOffset: 1x2 array, Y,X [pix] offset between cen and reference center of
        the exclusion Y,X coordinate system. E.g., cenOffset is needed if
        masked image is padded compared to the image the exclusions were
        located from.
    """

    addMask = np.ones(mask.shape)

    try:
        cenYX = np.array(rectExclude[0])
        if cenOffset is not None:
            cenYX += np.round(cenOffset).astype(int)
        hwY = rectExclude[1][0]//2 # half width in Y [pix]
        hwX = rectExclude[1][1]//2 # half width in X [pix]
        angle = rectExclude[2] # [deg]

        addMask[cenYX[0] - hwY:cenYX[0] + hwY, cenYX[1] - hwX:cenYX[1] + hwX] = np.nan
        addMask = utils.rotate_array(addMask, cen, angle + paOffset,
                               preserve_nan=True, cval=1)
        addMask *= mask
        addMask = np.nan_to_num(addMask, nan=1, copy=True)
    except Exception as ee:
        print(ee)
        addMask = mask

    return addMask


def mask_point(mask, excl, cen, cenOffset=None, paOffset=0.):
    """
    cenOffset: 1x2 array, Y,X [pix] offset between cen and reference center of
        the exclusion Y,X coordinate system. E.g., cenOffset is needed if
        masked image is padded compared to the image the exclusions were
        located from.
    """

    addMask = np.ones(mask.shape)
    try:
        cenYX = np.array(excl[:2])
        if cenOffset is not None:
            cenYX += np.round(cenOffset).astype(int)
        radius = excl[2] # [pix]

        # Rotate the point's center coordinates based on paOffset angle.
        # NOTE: The rotation matrix below returns X,Y, so I reverse it to get Y,X
        # and match our standard center coordinates array convention.
        cenYXRel = cenYX - cen
        cenYXRot = cen + np.array([cenYXRel[1]*np.cos(np.radians(paOffset)) - cenYXRel[0]*np.sin(np.radians(paOffset)),
                                 cenYXRel[1]*np.sin(np.radians(paOffset)) + cenYXRel[0]*np.cos(np.radians(paOffset))])[::-1]

        radii = utils.make_radii(mask, cenYXRot)
        addMask[radii <= radius] = np.nan
# DEPRECATED. Now rotate point coordinates above instead of whole array here.
        # addMask = rotate_array(addMask, cen, paOffset,
        #                        preserve_nan=True, cval=1)
        addMask *= mask
        addMask = np.nan_to_num(addMask, nan=1, copy=True)
    except Exception as ee:
        print(ee)
        addMask = mask

    return addMask


def mask_spikes_offaxis(mask, excl, cen, cenOffset=None, paOffset=0.,
                        spikeAngles=np.array([45., 135.])):
    """
    cenOffset: 1x2 array, Y,X [pix] offset between cen and reference center of
        the exclusion Y,X coordinate system. E.g., cenOffset is needed if
        masked image is padded compared to the image the exclusions were
        located from.
    """

    addMask = np.ones(mask.shape)
    try:
        cenYX = np.array(excl[:2])
        if cenOffset is not None:
            cenYX += np.round(cenOffset).astype(int)
        spWidth = excl[2] # [pix]

        cenYXRel = cenYX - cen
        cenYXRot = cen + np.array([cenYXRel[1]*np.cos(np.radians(paOffset)) - cenYXRel[0]*np.sin(np.radians(paOffset)),
                                 cenYXRel[1]*np.sin(np.radians(paOffset)) + cenYXRel[0]*np.cos(np.radians(paOffset))])[::-1]

        # Actually construct the spike mask.
        addMask = (~ utils.make_spikemask_stis(addMask, cenYXRot, spikeAngles,
                                        spWidth)).astype(float)

        # Do some value converting to put the mask into the right format.
        addMask[addMask == 0] = np.nan
        addMask *= mask
        addMask = np.nan_to_num(addMask, nan=1, copy=True)
    except Exception as ee:
        print(ee)
        addMask = mask

    # oldMask = mask.copy()
    # test = addMask.copy()
    # 
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(test)
    # plt.draw()
    # 
    # plt.figure(2)
    # plt.clf()
    # plt.imshow(addMask)
    # plt.draw()
    # 
    # plt.figure(3)
    # plt.clf()
    # plt.imshow(oldMask)
    # plt.title('Input mask')
    # plt.draw()

    return addMask


def add_mask_bg_star(maskIn, cen, radius=5):

    radii = utils.make_radii(maskIn, cen)
    maskOut = maskIn.copy()
    maskOut[radii <= radius]

    return maskOut


def mask_charge_bleed(mask, data, cen, paOffset=0.):
    """
    Auto-detect and mask charge bleed by finding adjacent bright pixels as go down
    columns from a given source X,Y. E.g., check for pixel in next row that is
    within +/-10% of pixel in previous row. Not perfect, especially if the bleed
    intersects another star or the target.
    """

    pass

    return
