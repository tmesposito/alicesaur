#!/usr/bin/env python
#
# Tom Esposito

import os
import pdb
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
from matplotlib.colors import SymLogNorm
from astropy.io import ascii, fits
from astropy import table
from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter, shift, zoom
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lmfit import Model
import GPy

# Internal imports
from alicesaur.utils import make_radii, make_phi, make_1d_gauss, make_double_1d_gauss, unsharp, load_info_json, weighted_mean_1d
from .plot_utils import *
from alicesaur.calibration.flux import convert_intensity


#---- DEFINE CONSTANTS ----#
pscale_stis = 0.0507 # [arcsec/pixel]


def plot_stis_gpi_rows(targList=[], zoomGpi=True, gpiOverlay=False, sbcal=False,
                       vmax_stis=None, vmax_gpi=None, makeColorbar=False):
# FIX ME!!! Remove this dependence on a private external package.
    from gpi_python.gpidisks1 import axMaker

    # targList = ['HD 117214', 'HD 129590']

    nCol = 4
    nRow = len(targList)
    #fig = plt.figure(99)
    if makeColorbar:
        spEdge = [0.25, 0.45, 0.14, 0.53] # left, bottom, right, top
    else:
        spEdge = [0.6, 0.5, 0, 0] # left, bottom, right, top
    fig, axAll = axMaker(nCol*nRow, axRC=[nRow, nCol], axSize=[1.5, 1.5],
                          axDim=None, wdw=None, spR=0,
                          spC=0, spEdge=spEdge, hold=False, figNum=999)

    vmax_list = []
    for ii, targName in enumerate(targList):
        axList = axAll[ii]
        outputAxes, vmax_stis_out = plot_stis_gpi_side(targName, stisCombo=True, roi=[(-7, 7), (-7, 7)],
                                    maskEdges=0.5, labelPanels=False, smoothContours=True,
                                    gpiOverlay=gpiOverlay, zoomGpi=zoomGpi, figAxes=(fig, axList),
                                    sbcal=sbcal, vmax_stis=vmax_stis, vmax_gpi=vmax_gpi,
                                    outputAxes=True, makeColorbar=False, save=False)
        vmax_list.append(vmax_stis_out)

    vmax_list = np.array(vmax_list)
    vmax_ratios = vmax_list/np.max(vmax_list)
    
    print(f"SB scale max values by row: {vmax_list}")

    # Label columns.
    axAll[0][0].text(0.5, 0.96, 'STIS: Average', c='w', fontsize=12,
        horizontalalignment='center', verticalalignment='top',
        transform=axAll[0][0].transAxes)
    axAll[0][1].text(0.5, 0.96, 'STIS: Wide', c='w', fontsize=12,
        horizontalalignment='center', verticalalignment='top',
        transform=axAll[0][1].transAxes)
    axAll[0][2].text(0.5, 0.96, 'STIS: Narrow', c='w', fontsize=12,
        horizontalalignment='center', verticalalignment='top',
        transform=axAll[0][2].transAxes)
    axAll[0][3].text(0.5, 0.96, 'GPI: H Pol', c='w', fontsize=12,
        horizontalalignment='center', verticalalignment='top',
        transform=axAll[0][3].transAxes)
    
    for ii in range(0, nRow):
        if np.round(vmax_ratios[ii], 1) != 1.0:
            axAll[ii][0].text(0.05, 0.15, 'x{:.1f}'.format(1./vmax_ratios[ii]),
                    c='w', fontsize=10,
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=axAll[ii][0].transAxes)
        targName = targList[ii]
        if (rescale_Qr_fluxcal[targName] != rescale_Qr_fluxcal['HD 114082']) and \
            (rescale_Qr_fluxcal[targName] != 0):
            if rescale_Qr_fluxcal[targName] < 1:
                scaleStr = 'x{:.2f}'.format(rescale_Qr_fluxcal[targName])
                xStr = 0.7
            else:
                scaleStr = 'x{:d}'.format(rescale_Qr_fluxcal[targName])
                xStr = 0.8
            axAll[ii][-1].text(xStr, 0.03, scaleStr,
                c='w', fontsize=10,
                horizontalalignment='left', verticalalignment='bottom',
                transform=axAll[ii][-1].transAxes)

    # Hide all but one Y axis label.
    for ii, ax in enumerate(axAll.flatten()):
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.tick_params(which='major', length=3.5)
        # Left Y labels.
        if ax in axAll[:, 0]:
            ax.yaxis.set_visible(True)
            ax.tick_params(labelleft=False)   
            ax.set_ylabel('')
            # pdb.set_trace()
            # if ax != axAll[nRow//2-1][0]:
            #     ax.set_ylabel('')
            # else:
            #     ax.set_ylabel('TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST', y=0., x=0., transform=ax.transAxes, c='k',
            #                   verticalalignment='center')
        # Right Y labels.
        if ax in axAll[:, -1]:
            ax.yaxis.set_visible(True)
            ax.tick_params(right=True, left=False)
            ax.set_xlabel('')
        if ax in axAll[0]:
            ax.xaxis.set_visible(True)
            ax.tick_params(top=True, bottom=False)
        if ax in axAll[-1]:
            ax.xaxis.set_visible(True)
            ax.tick_params(bottom=True)
    # Manually set Y axis label.
    axAll[1][0].set_ylabel(r'$\Delta$Dec (arcsec)', y=0., labelpad=10, c='k',
                      verticalalignment='center', horizontalalignment='center',
                      fontsize=12)
    axAll[-1][1].set_xlabel(r'$\Delta$RA (arcsec)', x=1., labelpad=10, c='k',
                      verticalalignment='center', horizontalalignment='center',
                      fontsize=12)

    # Add colorbars.
    if makeColorbar:
        cax = easy_colorbar(axAll[0][1].images[0], axAll[0][1], fig, Vmin=0., Vmax=vmax_stis,
                            step=10, widfrac=0.05, label=r'STIS', labelPad=-12,
                            fontSize=12, cticks=None, orientation='horizontal', spine_color=None,
                            side='top', output=True)
        # figure relative [left, top, width, height]
        # Set the colorbar position relative to the image axes.
        cax.set_position(pos=[np.array(axAll[0][0].get_position())[0][0],
                              np.array(axAll[0][1].get_position())[1][1],
                              np.array(axAll[0][2].get_position())[1][0] - np.array(axAll[0][0].get_position())[0][0],
                              0.015])
        cax.tick_params(direction='in', pad=1)
        
        caxGPI = easy_colorbar(axAll[0][-1].images[0], axAll[0][-1], fig,
                               Vmin=axAll[0][-1].images[0].get_clim()[0],
                               Vmax=vmax_gpi,
                               step=10, widfrac=0.05, label=r'GPI', labelPad=-12,
                               fontSize=12, cticks=None, orientation='horizontal', spine_color=None,
                               side='top', output=True)
        # figure relative [left, top, width, height]
        # Set the colorbar position relative to the image axes.
        caxGPI.set_position(pos=[np.array(axAll[0][-1].get_position())[0][0],
                              np.array(axAll[0][-1].get_position())[1][1],
                              np.array(axAll[0][-1].get_position())[1][0] - np.array(axAll[0][-1].get_position())[0][0],
                              0.015])
        caxGPI.tick_params(direction='in', pad=1)
        axAll[0][1].text(1., 1.2, r'Surface Brightness (mJy arcsec$^{-2}$)',
                         c='k', fontsize=12, horizontalalignment='center',
                         verticalalignment='bottom',
                         transform=axAll[0][1].transAxes)

    plt.draw()
    pdb.set_trace()

    return


def plot_stis_gpi_side(targName=None, stis_paths=None, gpi_path=None,
                       roi=[(-6, 6),(-6, 6)], cen_stis=None, cen_gpi=None,
                       stisCombo=False, gpiOverlay=False, smoothContours=True,
                       maskEdges=False, labelPanels=False, zoomGpi=True,
                       smoothGpi=False, figAxes=None, sbcal=False,
                       vmax_stis=None, vmax_gpi=None,
                       outputAxes=False, makeColorbar=False, save=False):
    """
    targName: str name of target to automatically get image paths from the
        dict in stis_disk_gallery_plot.
    roi: list of tuples like [(-2, 2), (-1, 1.5)] to plot rectangular window
        with 2 arcsec of view in +/-Y directions and 1.5/1.0 arcsec view in +/-X
        directions from the cen of each image.
    stisCombo: bool, True to also plot the averaged STIS bar and wedge image.
    gpiOverlay: bool, True to overlay GPI contours on a STIS image.
    smoothContours: bool, True to lightly smooth the GPI image before measuring
        the contours, reducing noise in the contours.
    maskEdges: float or False, float to mask out noise around GPI image edges
        via Gaussian smoothing of NaNs (larger value is more heavily masked).
    labelPanels: bool, True to label each column with the image type.
    zoomGpi: bool, True to zoom into the GPI image panel.
    smoothGpi: False or float; if float, will Gaussian smooth with that as sigma
        in pixels.
    sbcal: bool, True to calibrate STIS images into mJy arcsec-2 units.
    makeColorbar: bool, True to draw colorbars for all columns.
    save: bool, True to save the figure to file.
    """

# FIX ME!!! Remove this dependence on a private external package.
    from stis_disk_gallery_plot import det_fns

    pscale_gpi = 0.014166 # [arcsec/pix]

    if type(stis_paths) == str:
        stis_paths = [stis_paths]

    if targName is not None:
        if stis_paths is None:
            stis_paths = [det_fns[targName]['wedge'],
                          det_fns[targName]['bar']]
        if gpi_path is None:
            gpi_path = det_fns[targName]['rstokes']

    stis_hdus = []
    stis = []
    for ii, pth in enumerate(stis_paths):
        try:
            stis_hdus.append(fits.open(os.path.expanduser(pth)))
            if sbcal:
                img = stis_hdus[ii][0].data.copy()
                hdr = stis_hdus[ii][0].header
                # Assuming starting unit is COUNTS S^-1, can get away with
                # forcing exptime = 1 because it is only used to convert to
                # COUNTS and then back to COUNTS S^-1.
                imgCal = convert_intensity([img], [[None, hdr]],
                                unitStart=hdr.get('BUNIT'), unitEnd='mjy arcsec-2',
                                gain=4.016, exptime=1, nCombine=hdr.get('NCOMBINE'))
                stis.append(imgCal[0])
            else:
                stis.append(stis_hdus[ii][0].data)
        except:
            stis_hdus.append(None)
            stis.append(np.nan*np.ones((2048, 2048)))

    if cen_stis is None:
        cen_stis = []
        for hdu in stis_hdus:
            try:
                cen_stis.append(np.round(np.array([hdu[0].header['PSFCENTY'],
                                        hdu[0].header['PSFCENTX']])).astype(int))
            except:
                cen_stis.append(np.array([1025, 1025]))

    stisCombo_list = []
    if stisCombo:
        cen_stis.append(cen_stis[0])
        if targName not in []:
            if det_fns[targName].get('stisAverage', None) not in [None, '']:
                comboPath = os.path.expanduser(det_fns[targName].get('stisAverage'))
                with fits.open(comboPath) as hdu:
                    stisCombo_im = hdu[0].data.copy()
                    stisCombo_hdr = hdu[0].header
                stisCombo_im = convert_intensity([stisCombo_im], [[None, stisCombo_hdr]],
                                    unitStart='counts s-1', unitEnd='mjy arcsec-2',
                                    gain=4.016, exptime=1, nCombine=1)[0]
                print(f"Retrieved pre-combined STIS image for {targName} at {comboPath}")
            else:
                stisCombo_im = np.nanmean(np.array(stis), axis=0)
            stis.append(stisCombo_im)
        else:
            stis.append(np.nan*np.ones((2048,2048)))

    try:
        gpi_hdu = fits.open(os.path.expanduser(gpi_path))
        gpi_raw = gpi_hdu[1].data[1]
        gpi_hdr = gpi_hdu[1].header
        if cen_gpi is None:
            cen_gpi = np.round(np.array([gpi_hdu[1].header['PSFCENTY'],
                                         gpi_hdu[1].header['PSFCENTX']])).astype(int)
    except:
        gpi_raw = np.zeros((281, 281))
        gpi_hdr = {}
        cen_gpi = np.array([140, 140])

    if maskEdges:
        e_mask = np.ma.masked_invalid(gaussian_filter(gpi_raw, maskEdges)).mask
        gpi_raw = np.ma.masked_array(gpi_raw.copy(), mask=e_mask).filled(np.nan)

    if smoothGpi is not False:
        gpi_raw = gaussian_filter(gpi_raw, smoothGpi)

    # Interpolate the GPI image to be on same pixel scale as STIS.
    gpi_zoom = zoom(np.nan_to_num(gpi_raw, 0.), pscale_gpi/pscale_stis)
    cen_gpi_zoom = cen_gpi*pscale_gpi/pscale_stis
    cen_gpi_contours = np.floor(cen_gpi_zoom).astype(int)
    # Re-center zoomed image onto center of an integer pixel.
    gpi_zoom = shift(np.nan_to_num(gpi_zoom.copy()), cen_gpi_contours - cen_gpi_zoom, order=1, prefilter=False)

    # Interpolate the GPI image to be on same pixel scale as STIS.
    gpi = gpi_raw
    # cen_gpi_zoom = cen_gpi*pscale_gpi/pscale_stis
    cen_gpi_int = np.floor(cen_gpi).astype(int)
    # Re-center zoomed image onto center of an integer pixel.
    gpi = shift(np.nan_to_num(gpi), cen_gpi - cen_gpi_int, order=1, prefilter=False)

    # Pad the gpi array to be plotted on a larger FOV.
    newCen_stis = []
    matDim = 2200
    mat = np.nan*np.ones((matDim, matDim))
    for ii, st in enumerate(stis):
        mat_stis = mat.copy()
        mat_stis[matDim//2-cen_stis[ii][0]:matDim//2-cen_stis[ii][0]+st.shape[0],
                matDim//2-cen_stis[ii][1]:matDim//2-cen_stis[ii][1]+st.shape[1]] = st.copy()
        stis[ii] = mat_stis
        cen_stis[ii] = np.array([matDim//2, matDim//2])

    roi_stis = np.round(np.array(roi)/pscale_stis).astype(int)
    roi_gpi_contours = roi_stis #np.round(np.array(roi)/pscale_gpi).astype(int)
    roi_gpi = np.round(np.array(roi)/pscale_gpi).astype(int)

    patch_stis = []
    newCen_stis = []
    stis_mins = []
    stis_maxs = []
    for ii, st in enumerate(stis):
        patch_stis.append(st[cen_stis[ii][0]+roi_stis[0][0]:cen_stis[ii][0]+roi_stis[0][1],
                             cen_stis[ii][1]+roi_stis[1][0]:cen_stis[ii][1]+roi_stis[1][1]])
        stis_mins.append(np.percentile(np.nan_to_num(patch_stis[ii].copy(), 1e6), 1.))
        stis_maxs.append(np.percentile(np.nan_to_num(patch_stis[ii].copy(), 0), 99.99))
        newCen_stis.append(np.array([roi_stis[0][1], roi_stis[1][1]]))
    # patch_gpi = gpi[cen_gpi[0]+roi_gpi[0][0]:cen_gpi[0]+roi_gpi[0][1],
    #                 cen_gpi[1]+roi_gpi[1][0]:cen_gpi[1]+roi_gpi[1][1]]
    # newCen_gpi = np.array([roi_gpi[0][0], roi_gpi[0][1]])
    # patch_gpi = mat_gpi[0:patch_stis[0].shape[0], 0:patch_stis[0].shape[1]]

    mat_gpi = mat.copy()
    mat_gpi_contours = mat.copy()
    newCen_gpi = np.array([matDim//2, matDim//2])
    # breakpoint()
    # Get the patch for the GPI contours.
    mat_gpi_contours[newCen_gpi[0]-cen_gpi_contours[0]:newCen_gpi[0]+(gpi_zoom.shape[0]-cen_gpi_contours[0]),
                     newCen_gpi[1]-cen_gpi_contours[1]:newCen_gpi[1]+(gpi_zoom.shape[1]-cen_gpi_contours[1])] = gpi_zoom
    patch_gpi_contours = mat_gpi_contours[newCen_gpi[0]+roi_gpi_contours[0][0]:newCen_gpi[0]+roi_gpi_contours[0][1],
                        newCen_gpi[1]+roi_gpi_contours[1][0]:newCen_gpi[1]+roi_gpi_contours[1][1]]
    # mat_gpi_contours[matDim//2-cen_gpi_contours[0]:matDim//2-cen_gpi_contours[0]+(gpi_zoom.shape[0]-cen_gpi_contours[0]),
    #     matDim//2-cen_gpi_contours[1]:matDim//2-cen_gpi_contours[1]+(gpi_zoom.shape[1]-cen_gpi_contours[1])] = gpi_zoom
    # Get the patch for the GPI image.
    mat_gpi[newCen_gpi[0]-cen_gpi[0]:newCen_gpi[0]+(gpi.shape[0]-cen_gpi[0]),
        newCen_gpi[1]-cen_gpi[1]:newCen_gpi[1]+(gpi.shape[1]-cen_gpi[1])] = gpi
# TESTING!!! Bin mat_gpi and then interpolate it back onto original grid.
    binSize = 1
    if binSize != 1:
        # pscale_gpi_bin = pscale_gpi*binSize
        # newCen_gpi = (newCen_gpi/binSize).astype(int)
        # roi_gpi = np.round(roi_gpi/binSize).astype(int)
        mat_gpi_bin = zoom(np.nan_to_num(mat_gpi, 0), 1/binSize)
        # mat_gpi_bin[np.abs(mat_gpi_bin) < 1e-10] = 0.
        mat_gpi = zoom(np.nan_to_num(mat_gpi_bin, 0), binSize)
        mat_gpi[np.abs(mat_gpi) < 1e-10] = 0.
    # else:
    #     mat_gpi_bin = mat_gpi
    patch_gpi = mat_gpi[newCen_gpi[0]+roi_gpi[0][0]:newCen_gpi[0]+roi_gpi[0][1],
                            newCen_gpi[1]+roi_gpi[1][0]:newCen_gpi[1]+roi_gpi[1][1]]
    # gpi = mat_gpi
    # patch_gpi = gpi[cen_gpi[0]+roi_gpi[0][0]:cen_gpi[0]+roi_gpi[0][1],
    #                 cen_gpi[1]+roi_gpi[1][0]:cen_gpi[1]+roi_gpi[1][1]]
    # newCen_gpi = np.array([roi_gpi[0][0], roi_gpi[0][1]])
    # patch_gpi = mat_gpi[0:patch_stis[0].shape[0], 0:patch_stis[0].shape[1]]

    # Get a GPI array with no NaNs, zeros, or negatives to estimate values from.
    patch_gpi_posdef = np.nan_to_num(patch_gpi, 0)[patch_gpi > 0]
    # Get a GPI array with no NaNs or zeros.
    patch_gpi_nonzero = np.nan_to_num(patch_gpi, 0)[patch_gpi != 0]
    # Estimate standard deviation of GPI image background (non-disk regions).
    try:
        gpi_std = np.std(patch_gpi_nonzero[patch_gpi_nonzero < np.percentile(patch_gpi_posdef, 84.)])
    except:
        gpi_std = 0.1

    vmin_stis = 0. #np.nanmin(stis_mins)
    if vmax_stis is None:
        if sbcal:
            vmax_stis = 1.2*stis_maxs[2]
        else:
            vmax_stis = 1.2*np.nanmax(stis_maxs)
    vmin_gpi = 0. #np.percentile(np.nan_to_num(patch_gpi.copy(), 0), 1.)
    if vmax_gpi is None:
        try:
            vmax_gpi = 1.2*np.percentile(patch_gpi_posdef, 99.99)
        except:
            vmax_gpi = 100.
        if targName == 'HD 145560':
            vmax_gpi = 30.

#    contour_percentiles = np.array([50., 84., 95.])
    contour_percentiles = np.array([50., 95.])

    # Prep for GPI overlays, if desired.
    if gpiOverlay:
        xx, yy = np.meshgrid(range(patch_gpi_contours.shape[1]), range(patch_gpi_contours.shape[0]))
        if targName == 'GSC 07396':
            levels = [0, 1]
        elif targName == 'HD 111161':
            levels = np.linspace(1, np.round(vmax_gpi).astype(int), 4)
        # elif targName == 'HD 114082':
        #     levels = np.linspace(, np.round(vmax_gpi).astype(int), 4)
        elif np.round(vmax_gpi/binSize/20) >= 3:
            levels = [np.percentile(patch_gpi_posdef, ii) for ii in contour_percentiles]
            # levels = np.linspace(np.round(vmax_gpi/20).astype(int), np.round(vmax_gpi).astype(int), 4)
        else:
            levels = [np.percentile(patch_gpi_posdef, ii) for ii in contour_percentiles]
            # levels = np.linspace(3, np.round(vmax_gpi).astype(int), 4)

    # Clean up empty region of patch_gpi.
    patch_gpi[patch_gpi == 0] = np.nan

    fs = 12
    nCol = len(stis) + 1
    # if stisCombo:
    #     nCol = len(stis) + 1
    # else:
    #     nCol = len(stis) + 1
    # Set colormap.
    cmap = copy.copy(cm.get_cmap('viridis'))
    cmap.set_bad(color='0.4')
    starColor = 'w'
    lt = 0.005 # SymLogNorm linear threshold
    
    # contourWdwHWs = {'GSC 07396':26, 'HD 106906':38, 'HD 114082':26, 'HD 115600':26,
    #                  'HD 117214':32, 'HD 129590':26, 'HD 145560':26}
    contourWdwHWs = {'AK Sco': 32, 'GSC 07396':32, 'HD 106906':32, 'HD 111161':32, 'HD 114082':32, 'HD 115600':32,
                     'HD 117214':32, 'HD 129590':32, 'HD 143675':32, 'HD 145560':32, 'HD 146897':32}

    gpiZoomWdwHWs = {}
    for kk in contourWdwHWs.keys():
        gpiZoomWdwHWs[kk] = contourWdwHWs[kk]*pscale_stis

# ====== BASE FIGURE & AXES CREATION =======
    figsizeX = 3.5*nCol
    if makeColorbar:
        figsizeY = 4. + 0.2
        subplotAdjTop = 0.95
    else:
        figsizeY = 4.
        subplotAdjTop = 0.99

    if figAxes is None:
        fig = plt.figure(10, figsize=(figsizeX, figsizeY))
        fig.clf()
        ax0 = plt.subplot(101 + 10*nCol)
        ax1 = plt.subplot(102 + 10*nCol)
        axList = [ax0, ax1]
        if nCol == 3:
            ax2 = plt.subplot(103 + 10*nCol)
            axList.append(ax2)
        elif nCol == 4:
            ax2 = plt.subplot(103 + 10*nCol)
            ax3 = plt.subplot(104 + 10*nCol)
            axList.append(ax2)
            axList.append(ax3)
        plt.subplots_adjust(top=subplotAdjTop, bottom=0.07, left=0.053, right=0.99,
                            wspace=0)
    else:
        fig = figAxes[0]
        axList = figAxes[1]
        ax0, ax1 = axList[0:2]
        nCol = len(axList)
        if nCol >= 3:
            ax2 = axList[2]
        if nCol == 4:
            ax3 = axList[3]
# ====== ONE COLUMN LAYOUT =======
    ax0.imshow(patch_stis[0], cmap=cmap,
                norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=vmin_stis, vmax=vmax_stis, base=10),
                extent=[roi[1][0], roi[1][1],
                        roi[0][0], roi[0][1]])
    if not np.all(np.isnan(patch_stis[0])):
        ax0.scatter(0, 0, marker='+', s=40, color=starColor)
# ====== TWO COLUMN LAYOUT =======
    if nCol == 2:
        ax1.imshow(patch_gpi, cmap=cmap,
                    norm=SymLogNorm(linthresh=1., linscale=1.,
                                    vmin=0., vmax=vmax_gpi, base=10),
                    extent=[roi[1][0], roi[1][1],
                            roi[0][0], roi[0][1]])
        ax1.scatter(0, 0, marker='+', s=40, color=starColor)
# ====== THREE COLUMN LAYOUT =======
    elif nCol == 3:
        ax1.imshow(patch_stis[1], cmap=cmap,
                    norm=SymLogNorm(linthresh=0.1, linscale=1.,
                                    vmin=vmin_stis, vmax=vmax_stis, base=10),
                    extent=[roi[1][0], roi[1][1],
                            roi[0][0], roi[0][1]])
        ax1.scatter(0, 0, marker='+', s=40, color=starColor)
        ax2.imshow(patch_gpi, cmap=cmap,
                    norm=SymLogNorm(linthresh=1., linscale=1.,
                                    vmin=0., vmax=vmax_gpi, base=10),
                    extent=[roi[1][0], roi[1][1],
                            roi[0][0], roi[0][1]])
        ax2.scatter(0, 0, marker='+', s=40, color=starColor)
        ax2.yaxis.set_visible(False)
# ====== FOUR COLUMN LAYOUT =======
    elif nCol == 4:
        axAvg = ax0
        axWedge = ax1
        axBar = ax2
        axGpi = ax3

        imWedge = axWedge.imshow(patch_stis[0], cmap=cmap,
                                 norm=SymLogNorm(linthresh=lt, linscale=1.,
                                 vmin=vmin_stis, vmax=vmax_stis, base=10),
                                 extent=[roi[1][0], roi[1][1],
                                         roi[0][0], roi[0][1]])
        if not np.all(np.isnan(patch_stis[0])):
            axWedge.scatter(0, 0, marker='+', s=40, color=starColor)
        axBar.imshow(patch_stis[1], cmap=cmap,
                    norm=SymLogNorm(linthresh=lt, linscale=1.,
                                    vmin=vmin_stis, vmax=vmax_stis, base=10),
                    extent=[roi[1][0], roi[1][1],
                            roi[0][0], roi[0][1]])
        # ax1.set_yticklabels([])
        if not np.all(np.isnan(patch_stis[1])):
            axBar.scatter(0, 0, marker='+', s=40, color=starColor)
        # Overlay GPI contours if desired.
        if zoomGpi:
            wdwHW = contourWdwHWs.get(targName, 80)
            if wdwHW is not None:
                wdw = [newCen_stis[2][0]-wdwHW, newCen_stis[2][0]+wdwHW+1, newCen_stis[2][1]-wdwHW, newCen_stis[2][1]+wdwHW+1]
                panCen = np.array([wdwHW, wdwHW])
                panRoi = pscale_stis*np.array([[-wdwHW, wdwHW], [-wdwHW, wdwHW]])
            else:
                wdw = [None, None, None, None]
                panCen = newCen_stis[2].copy()
                panRoi = roi
            # ax2.imshow(patch_stis[2][wdw[0]:wdw[1], wdw[2]:wdw[3]], cmap=cmap,
            #             norm=SymLogNorm(linthresh=0.1, linscale=1.,
            #                             vmin=vmin_stis, vmax=vmax_stis*1.8, base=10),
            #             extent=[panRoi[1][0], panRoi[1][1],
            #                     panRoi[0][0], panRoi[0][1]],
            #             zorder=0)
            axAvg.imshow(patch_stis[2], cmap=cmap,
                        norm=SymLogNorm(linthresh=lt, linscale=1.,
                                        vmin=vmin_stis, vmax=vmax_stis, base=10),
                        extent=[roi[1][0], roi[1][1],
                                roi[0][0], roi[0][1]],
                        zorder=0)
            if gpiOverlay and targName not in ['GSC 07396']:
                if smoothContours:
                    # CS = ax2.contour(xx[:wdw[3]-wdw[2], :wdw[3]-wdw[2]],
                    #                  yy[:wdw[1]-wdw[0], :wdw[1]-wdw[0]],
                    #                  gaussian_filter(patch_gpi_contours, 1.)[wdw[0]:wdw[1], wdw[2]:wdw[3]],
                    #                  levels=levels, cmap='magma_r', #'magma',
                    #             )
                    CS = axAvg.contour((xx[:wdw[3]-wdw[2], :wdw[3]-wdw[2]] - wdwHW)*pscale_stis,
                                       (yy[:wdw[1]-wdw[0], :wdw[1]-wdw[0]] - wdwHW)*pscale_stis,
                                       gaussian_filter(patch_gpi_contours, 1.)[wdw[0]:wdw[1], wdw[2]:wdw[3]],
                                       levels=levels, cmap='magma_r', #'magma',
                                       extent=[panRoi[1][0], panRoi[1][1],
                                               panRoi[0][0], panRoi[0][1]])
                else:
                    CS = axAvg.contour((xx[:wdw[3]-wdw[2], :wdw[3]-wdw[2]] - wdwHW)*pscale_stis,
                                       (yy[:wdw[1]-wdw[0], :wdw[1]-wdw[0]] - wdwHW)*pscale_stis,
                                       patch_gpi_contours[wdw[0]:wdw[1], wdw[2]:wdw[3]], levels=levels,
                                       cmap='magma_r',
                                       extent=[panRoi[1][0], panRoi[1][1],
                                               panRoi[0][0], panRoi[0][1]], zorder=1)
                # gpiFPM_stisscale = patches.Circle(panCen[::-1], radius=0.123/pscale_stis,
                #                     fill=True, color='k', zorder=999)
                # ax2.add_patch(gpiFPM_stisscale)
                # ax2.scatter(panCen[1], panCen[0], marker='+', s=40,
                #             color=starColor, zorder=1000)
                gpiFPM_stisscale = patches.Circle((0,0), radius=0.123,
                                    fill=True, color='k', zorder=999)
            else:
                gpiFPM_stisscale = patches.Circle((0,0), radius=0.123,
                                    linestyle='--',
                                    fill=False, color='k', zorder=999)
            axAvg.add_patch(gpiFPM_stisscale)
            axAvg.scatter(0, 0, marker='+', s=40,
                        color=starColor, zorder=1000)
        else:
            axAvg.imshow(patch_stis[2], cmap=cmap,
                        norm=SymLogNorm(linthresh=lt, linscale=1.,
                                        vmin=vmin_stis, vmax=vmax_stis, base=10),
                        extent=[roi[1][0], roi[1][1],
                                roi[0][0], roi[0][1]])
            axAvg.scatter(0, 0, marker='+', s=40, color=starColor)
        ax2.yaxis.set_visible(False)
        # ax2.set_yticklabels([])
        if zoomGpi and targName not in ['GSC 07396']:
            # Plot a zoomed in GPI image with side widths in arcsec of 2*gpiZoomWdwHWs[targName].
            roi_zoom_gpi = np.array([[-gpiZoomWdwHWs[targName], gpiZoomWdwHWs[targName]],
                                     [-gpiZoomWdwHWs[targName], gpiZoomWdwHWs[targName]]])
            wdw_zoom_gpi = (cen_gpi_int + (roi_zoom_gpi/pscale_gpi)).astype(int)

            # GPI flux calibration to mJy arcsec^-2 with a scaling factor.
            try:
                itime = gpi_hdr['ITIME']
            except:
                itime = 59.64639 # [s]
            # Convert units from ADU/s to mJy arcsec^-2.
            fluxcal = True
            if fluxcal:
                fluxcal_fac = (1e3/pscale_gpi**2)*gpi_Qr_fconv_Jy_ADU_s[targName]
                if targName in rescale_Qr_fluxcal.keys():
                    scale_Qr = rescale_Qr_fluxcal[targName]
                else:
                    scale_Qr = 1.
                linthresh_gpi = 0.5
            # Scale the brightness for each image if previously specified.
            elif (targName in rescale_Qr.keys()) and not fluxcal:
                scale_Qr = rescale_Qr[targName]
                fluxcal_fac = 1.
                linthresh_gpi = 1.
            else:
                scale_Qr = 1.
                fluxcal_fac = 1.
                linthresh_gpi = gpi_std

            axGpi.imshow((fluxcal_fac*scale_Qr/itime)*gpi[wdw_zoom_gpi[0][0]:wdw_zoom_gpi[0][1], wdw_zoom_gpi[1][0]:wdw_zoom_gpi[1][1]],
                        cmap=cmap,
                        norm=SymLogNorm(linthresh=linthresh_gpi, linscale=1., vmin=0., vmax=vmax_gpi, base=10),
                        extent=[(wdw_zoom_gpi[1][0]-cen_gpi_int[1])*pscale_gpi,
                                (wdw_zoom_gpi[1][1]-cen_gpi_int[1])*pscale_gpi,
                                (wdw_zoom_gpi[0][0]-cen_gpi_int[0])*pscale_gpi,
                                (wdw_zoom_gpi[0][1]-cen_gpi_int[0])*pscale_gpi])
                        # extent=[panRoi[1][0], panRoi[1][1],
                        #                  panRoi[0][0], panRoi[0][1]], zorder=1)
            # gpiFPM = patches.Circle((roi_gpi[1][1], roi_gpi[0][1]), radius=0.123/pscale_gpi,
            #                         fill=True, color='k')
            gpiFPM = patches.Circle((0, 0), radius=0.123,
                                    fill=True, color='k')
            axGpi.add_patch(gpiFPM)
            axGpi.scatter(0, 0, marker='+', s=40, color=starColor, zorder=1000)
            axGpi.yaxis.set_visible(True)
            axGpi.tick_params(right=True)
            axGpi.set_yticklabels([])
        else:
            axGpi.imshow(patch_gpi, cmap=cmap,
                         norm=SymLogNorm(linthresh=1., linscale=1., vmin=0., vmax=vmax_gpi, base=10),
                         extent=[roi[1][0], roi[1][1],
                                 roi[0][0], roi[0][1]])
            # gpiFPM = patches.Circle((roi_gpi[1][1], roi_gpi[0][1]), radius=0.123/pscale_gpi,
            #                         fill=True, color='k')
            if not np.all(np.isnan(patch_gpi)):
                gpiFPM = patches.Circle((0, 0), radius=0.123,
                                        fill=True, color='k')
                axGpi.add_patch(gpiFPM)
                axGpi.scatter(0, 0, marker='+', s=40, color=starColor, zorder=1000)
            ax3.yaxis.set_visible(True)
            ax3.tick_params(right=True)
            ax3.set_yticklabels([])
        ax1.set_xlabel('[arcsec]', fontsize=fs) #, transform=ax1.transAxes)

# FIX ME!!! Only implemented at basic level.
        if makeColorbar:
            cax = easy_colorbar(imWedge, axWedge, fig, Vmin=0., Vmax=vmax_stis,
                                step=1, widfrac=0.05, label='Colorbar label', labelPad=5,
                                fontSize=fs, cticks=None, orientation='horizontal', spine_color=None,
                                side='top', output=True)
        
    # ax1.yaxis.set_visible(False)
    ax0.set_ylabel('[arcsec]', fontsize=fs)
    plt.draw()

    # # Label only disks not in the list anonymous.
    #         if nm not in anonymous:
    #             if label_names:
    #                 nm_txt = ax.text(targ_xy[0], targ_xy[1], nm, color=tc, fontsize=name_fontsize, weight='semibold',
    #                         horizontalalignment='left', verticalalignment='bottom')

    tc = 'w'
    # Label the target name.
    labelName = True
    if labelName:
        name_txt = ax0.text(0.05, 0.02, targName, color=tc, fontsize=fs, weight='normal',
                            horizontalalignment='left', verticalalignment='bottom', transform=ax0.transAxes)
    # Label the panels.
    for ii, ax in enumerate(axList):
        panelLabels = ['STIS WedgeB1.0', 'STIS Bar10', 'STIS with GPI contours', 'GPI Polarized H']
        if labelPanels:
            sideways = True
            if sideways:
                # ax_title = fig.add_axes([0.021, 0.9, 0.2, 0.1])
                panel_txt = ax.text(0.5, 0.91, panelLabels[ii], color=tc, fontsize=fs, weight='normal',
                                    horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
            # else:
            #     ax_title = fig.add_axes([0.008, 0.925, 0.2, 0.1])
            #     ax_title.text(0., 0.5, 'Stokes', color='w', fontsize=16, #weight='normal',
            #                     horizontalalignment='left', verticalalignment='center', transform=ax_title.transAxes)
            #     ax_title.text(0.65, 0.5, '$\mathcal{Q}_\mathcal{\phi}$', color='w', fontsize=18, weight='heavy',
            #                     horizontalalignment='center', verticalalignment='center', transform=ax_title.transAxes)
            # # ax_title = fig.add_axes([0.008, 0.925, 0.2, 0.1])
            # # ax_title.text(0.05, 0.5, '$\mathcal{Q}_r$', color='w', fontsize=18, weight='heavy',
            # #                   horizontalalignment='left', verticalalignment='top', transform=ax_title.transAxes)

            # ax_title.axis('off')
            # ax_arr[0].text(mode_xy[0], mode_xy[1], r'$\mathcal{Q}_r$', color='w', fontsize=18, weight='heavy',
            #               horizontalalignment='left', verticalalignment='bottom')
        # if label_scalebar and scale_xy is not None:
        #     ax_arr[0].add_patch(make_scalebar(bar_xy[0], bar_xy[1]))
        #     ax_arr[0].text(scale_xy[0], scale_xy[1], '1"', color='w', fontsize=10, weight='heavy',
        #                     horizontalalignment='left', verticalalignment='bottom')
        # if label_compass:
        #     compass(ax_arr[4], wf=0.7, hf=1.1, scale=0.5, fontSize=7, rot=0, cc=tc,
        #         N_yxadj=[0,-1], E_yxadj=[-1,0], head_len=3., head_wid=2.)
        ax.tick_params(labelsize=fs, direction='in', color='0.9', length=3)

    plt.draw()

    if save:
        if zoomGpi:
            fig.savefig('/Users/Tom/Research/data/hst/figures/stis_gpi_zoom_panels_{}.png'.format(targName.replace(' ', '')),
                        format='png', dpi=300)
        else:
            fig.savefig('/Users/Tom/Research/data/hst/figures/stis_gpi_panels_{}.png'.format(targName.replace(' ', '')),
                    format='png', dpi=300)

    if outputAxes:
        return axList, vmax_stis
    else:
        return


def plot_stis_gpi_overlay(targName=None, stis_paths=None, gpi_path=None,
                       roi=[(-6, 6),(-6, 6)], cen_stis=None, cen_gpi=None,
                       smooth=True, savePath=None):
    """
    targName: str name of target to automatically get image paths from the
        dict in stis_disk_gallery_plot.
    roi: list of tuples like [(-2, 2), (-1, 1.5)] to plot rectangular window
        with 2 arcsec of view in +/-Y directions and 1.5/1.0 arcsec view in +/-X
        directions from the cen of each image.
    """
# FIX ME!!! Remove this dependence on a private external package.
    from stis_disk_gallery_plot import det_fns

    pscale_gpi = 0.014166 # [arcsec/pix]

    if type(stis_paths) == str:
        stis_paths = [stis_paths]

    if targName is not None:
        if stis_paths is None:
            # stis_paths = [det_fns[targName]['wedge'],
            #               det_fns[targName]['bar']]
            stis_paths = [det_fns[targName]['bar'],
                          det_fns[targName]['wedge']]
        if gpi_path is None:
            gpi_path = det_fns[targName]['rstokes']

    stis_hdus = []
    stis = []
    for ii, pth in enumerate(stis_paths):
        stis_hdus.append(fits.open(os.path.expanduser(pth)))
        stis.append(stis_hdus[ii][0].data)
    if cen_stis is None:
        cen_stis = []
        for hdu in stis_hdus:
            try:
                cen_stis.append(np.round(np.array([hdu[0].header['PSFCENTY'],
                                hdu[0].header['PSFCENTX']])).astype(int))
            except:
                cen_stis.append(np.round(np.array([hdu[0].header['NAXIS1']//2,
                                hdu[0].header['NAXIS2']//2])).astype(int))

    gpi_hdu = fits.open(os.path.expanduser(gpi_path))
    gpi_raw = gpi_hdu[1].data[1]

    if cen_gpi is None:
        cen_gpi = np.round(np.array([gpi_hdu[1].header['PSFCENTY'],
                                     gpi_hdu[1].header['PSFCENTX']])).astype(int)

    # Clean the raw GPI edges.
    edgeMask = gaussian_filter(gpi_raw, 2)
    radii = make_radii(gpi_raw, cen_gpi)
    gpi_raw[np.isnan(edgeMask) & (radii > 80)] = np.nan

    # Interpolate the GPI image to be on same pixel scale as STIS.
    gpi = zoom(np.nan_to_num(gpi_raw, 0.), pscale_gpi/pscale_stis)
    cen_gpi_zoom = cen_gpi*pscale_gpi/pscale_stis
    # cen_gpi = np.round(cen_gpi_zoom).astype(int)
    cen_gpi = np.floor(cen_gpi_zoom).astype(int)
    gpi = shift(np.nan_to_num(gpi), cen_gpi - cen_gpi_zoom, order=1, prefilter=False)
    # Pad the gpi array to be plotted on a larger FOV.
    matDim = stis[0].shape
    mat = np.nan*np.ones(matDim)

    roi_stis = np.round(np.array(roi)/pscale_stis).astype(int)
    roi_gpi = roi_stis #np.round(np.array(roi)/pscale_gpi).astype(int)
    
    patch_stis = []
    newCen_stis = []
    stis_mins = []
    stis_maxs = []
    for ii, st in enumerate(stis):
        patch_stis.append(st[cen_stis[ii][0]+roi_stis[0][0]:cen_stis[ii][0]+roi_stis[0][1],
                             cen_stis[ii][1]+roi_stis[1][0]:cen_stis[ii][1]+roi_stis[1][1]])
        stis_mins.append(np.percentile(np.nan_to_num(patch_stis[ii].copy(), 1e6), 1.))
        stis_maxs.append(np.percentile(np.nan_to_num(patch_stis[ii].copy(), 0), 99.99))
        newCen_stis.append(np.array([roi_stis[0][1], roi_stis[1][1]]))
    mat_gpi = mat.copy()
    mat_gpi[newCen_stis[0][0]-cen_gpi[0]:newCen_stis[0][0]+(gpi.shape[0]-cen_gpi[0]), newCen_stis[0][1]-cen_gpi[1]:newCen_stis[0][1]+(gpi.shape[1]-cen_gpi[1])] = gpi
    patch_gpi = mat_gpi[0:patch_stis[0].shape[0], 0:patch_stis[0].shape[1]]

    if smooth:
        patch_gpi = gaussian_filter(patch_gpi, 0.5)

    vmin_stis = 0. #np.nanmin(stis_mins)
    vmax_stis = 1.2*np.nanmax(stis_maxs)
    vmin_gpi = 0. #np.percentile(np.nan_to_num(patch_gpi.copy(), 0), 1.)
    vmax_gpi = 1.2*np.percentile(np.nan_to_num(patch_gpi.copy(), 0), 99.99)

    xx, yy = np.meshgrid(range(patch_gpi.shape[1]), range(patch_gpi.shape[0]))

    fs = 16
    fig = plt.figure(11, figsize=(6, 6))
    fig.clf()
    ax0 = plt.subplot(111)
    ax0.imshow(patch_stis[0],
                norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=vmin_stis, vmax=vmax_stis))
                # extent=[roi[1][0], roi[1][1],
                #       roi[0][0], roi[0][1]])
    # norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
    # cmap = cm.PRGn
    
#   cset1 = ax0.contourf(X, Y, Z, levels) #, norm=norm,
#                           cmap=cm.get_cmap(cmap, len(levels) - 1))
#   cset2 = ax0.contour(X, Y, Z, cset1.levels, colors='k')
    if targName == 'HD 111161':
        levels = np.linspace(1, np.round(vmax_gpi).astype(int), 4)
    elif np.round(vmax_gpi/20) >= 3:
        levels = np.linspace(np.round(vmax_gpi/20).astype(int), np.round(vmax_gpi).astype(int), 4)
    else:
        levels = np.linspace(3, np.round(vmax_gpi).astype(int), 4)
    CS = ax0.contour(xx, yy, patch_gpi, levels=levels, cmap='magma') #,
#                    extent=[roi[1][0], roi[1][1], roi[0][0], roi[0][1]])
    # CS = ax0.contour(X=xx, Y=yy, Z=patch_gpi)
                # extent=[roi[1][0], roi[1][1],
                #       roi[0][0], roi[0][1]])
    # ax0.clabel(CS, inline=True, fontsize=8)
    ax0.scatter(0, 0, marker='+', s=40, color='0.5')

    try:
        plt.suptitle(gpi_hdu[0].header['OBJECT'], fontsize=fs)
    except:
        plt.suptitle(stis_hdus[0][0].header['TARGNAME'], fontsize=fs)
    ax0.set_ylabel('[arcsec]')
    plt.draw()
    
    if savePath is not None:
        fig.savefig(savePath, format='png')
    breakpoint()
    return


def plot_all_overlays(stis_type='bar', smooth=True):
    """
    stis_type: 'bar' or 'wedge'
    """
# FIX ME!!! Remove this dependence on private external packages.
    from gpi_python.gpidisks1 import det_fns
    from stis_disk_gallery_plot import det_fns as stis_fns
    
    targNames = ['AK Sco', 'HD 111161', 'HD 114082', 
                 'HD 115600', 'HD 117214', 'HD 129590', 'HD 145560', 'HD 146897']
    # targNames = ['HD 145560']
    
    # plot_stis_gpi_overlay('HD 106906', stis_paths=stis_fns['HD 106906']['wedge'],
    #                         gpi_path=det_fns['HD 106906']['rstokes'],
    #                         roi=[(-6, 6),(-6, 6)], smooth=smooth)
    
    for targ in targNames:
        savePath = '/Users/Tom/Research/data/hst/figures/{}_{}_gpi_contours.png'.format(targ, stis_type)
        plot_stis_gpi_overlay(targ, stis_paths=stis_fns[targ][stis_type],
                              gpi_path=det_fns[targ]['rstokes'],
                              roi=[(-6, 6),(-6, 6)], smooth=smooth,
                              savePath=savePath)
        # print(targ, det_fns[targ]['rstokes'])
        # print(targ, stis_fns[targ][stis_type])
    #   # breakpoint()
    
    return


def plot_PA(rads, pas, diskPA=None, label='', c='k', figNum=12):
    """
    Inputs:
      pas: position angles in [degrees].
    """
    
    # Calculate the diameter of one pixel in degrees at each separation.
    # Use half that diameter as the PA error bar.
    pixWidths = 360./(2*np.pi*np.array(rads)) # [deg]

    fig = plt.figure(figNum)
    fig.clf()
    ax0 = plt.subplot(111)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.98)
    ax0.errorbar(rads, pas, yerr=0.5*pixWidths, marker='o', c=c, label=label)
    try:
        ax0.axhline(diskPA, c='k', linestyle='--')
    except:
        pass
    ax0.set_ylabel('PA (degrees)')
    ax0.set_xlabel('Radius (pixels)')
    ax0.legend(numpoints=1)
    plt.draw()

    return


def vertical_profile(fp, rad=60, highpass=None, diskPA=None, star=None,
                     maxDiskWidth_pix=35, data=None, hdr=None, hdu=None,
                     bounds1=None, bounds2=None):
    """
    rad: radius of profile in [pix]
    diskPA: disk major axis PA in [degrees]
    maxDiskWidth_pix: maximum "vertical" width of the disk in [pixels]
    """
    
    if (data is None) | (hdr is None) | (hdu is None):
        with fits.open(os.path.expanduser(fp)) as hduf:
            hdu = hduf
            data = hdu[0].data
            hdr = hdu[0].header

# TESTING!!! Smooth image first.
            data = median_filter(data, 5)

    if highpass is not None:
        data = unsharp('', hdu, '', B=highpass, ident='_999', output=True, silent=True,
                        save=False, parOK=True, gauss=False)

    if diskPA is not None:
        diskPA_rad = np.radians(diskPA)

    if star is None:
        star = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])
    radii = make_radii(data, star)
    dr = 1.5 # half-width of each slice in [pix]
    phi = make_phi(data, star)
    phi -= np.pi/2
    phi[phi < 0] += 2*np.pi

    phis = phi[(radii < rad + dr) & (radii > rad - dr)]
    cut = data[(radii < rad + dr) & (radii > rad - dr)]

    sortInds = np.argsort(phis)
    phi_sort = phis[sortInds]
    cut_sort = cut[sortInds]

    phi_double = np.degrees(np.append(phi_sort, phi_sort + 2*np.pi))
    cut_double = np.append(cut_sort, cut_sort)
# # # TESTING!!! Smoothing the slice.
#     cut_double = gaussian_filter1d(cut_double.copy(), sigma=2, mode='reflect')

    # Width of a single pixel in phi at radius rr.
    pixel_width_phi = np.degrees(np.arcsin(1./rad)) # [deg/pixel]
    # Estimate disk FWHM to get initial guess of Gaussian sigma for fit.
    fwhm_guess_phi = 0.5*maxDiskWidth_pix*pixel_width_phi/2.355 # [deg]

    # Set all NaN values to zero to avoid fitter failing.
    cut_double_clean = np.nan_to_num(cut_double.copy(), 0.)
    # cut_double_clean[cut_double_clean < 0] = 0.

    # If the slice is flat zeros, skip the fit and move on.
    if (np.all(cut_double_clean == 0) or np.all(np.isnan(cut_double_clean))):
        print("r={}; No disk: empty data slice".format(rad))
        return np.nan, np.nan, (np.array([None]), np.array([None])), (np.array([None]), np.array([None])), None, None


    # plt.figure(1)
    # plt.clf()
    # plt.plot(range(len(cut_double)), cut_double, linestyle='-') #, markersize=8)
    # plt.draw()

    # xcorr = crosscorr_gauss_1d(np.nan_to_num(cut_double, 0.), phi_double)

    # Set bounds for the double Gaussian model fit parameters.
    # Param order: mu1, sig1, mu2, sig2, C1, C2
    side1_inds = np.where((phi_double < diskPA + 60) & (phi_double > diskPA - 60))
    side2_inds = np.where((phi_double < diskPA + 180 + 70) & (phi_double > diskPA + 180 - 70))
    # Locate slice's brightest pixel on both sides.
    max1 = np.max(cut_double_clean[side1_inds])
    max1_ind = np.argmax(cut_double_clean[side1_inds])
    max1_phi = phi_double[side1_inds][np.argmax(cut_double_clean[side1_inds])]
    max2 = np.max(cut_double_clean[side2_inds])
    max2_ind = np.argmax(cut_double_clean[side2_inds])
    max2_phi = phi_double[side2_inds][np.argmax(cut_double_clean[side2_inds])]
    # Guess that disk midplane is centered on slice's brightest pixel, to start.
    if (max1_phi <= diskPA + 30) and (max1_phi >= diskPA - 30):
        p0_1 = (max1_phi, 0.5*fwhm_guess_phi, max1_phi+5, 0.5*fwhm_guess_phi, 30., 30.)
    else:
        p0_1 = (diskPA, 0.5*fwhm_guess_phi, diskPA+5, 0.5*fwhm_guess_phi, 30., 30.)
    if (max2_phi <= diskPA + 180 + 30) and (max2_phi >= diskPA + 180 - 30):
        p0_2 = (max2_phi, 0.5*fwhm_guess_phi, max2_phi-5, 0.5*fwhm_guess_phi, 30., 30.)
    else:
        p0_2 = (diskPA+180, 0.5*fwhm_guess_phi, diskPA+180-5, 0.5*fwhm_guess_phi, 30., 30.)
    if bounds1 is None:
        bounds1 = (np.array([p0_1[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, p0_1[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, 0, 0]),
                   np.array([p0_1[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_1[2]+2*fwhm_guess_phi, fwhm_guess_phi*2, 1e5, 1e5]))
    if bounds2 is None:
        bounds2 = (np.array([p0_2[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, p0_2[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, 0, 0]),
                   np.array([p0_2[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_2[2]+2*fwhm_guess_phi, fwhm_guess_phi*4, 1e5, 1e5]))

 #--- SIDE 1 FITTING ---#
    # Double Gaussian fit to side 1.
    try:
        double_gauss_fit1 = fit_double_gauss_1d(cut_double_clean, phi_double,
                                  indMinMax=(side1_inds[0][0], side1_inds[0][-1]),
                                  p0=p0_1,
                                  bounds=bounds1)
        fitvals1 = double_gauss_fit1.best_values
        double_gauss_bf1 = make_double_1d_gauss(phi_double, fitvals1['mu1'], fitvals1['sig1'],
                                        fitvals1['mu2'], fitvals1['sig2'],
                                        C1=fitvals1['C1'], C2=fitvals1['C2'])
        # double_gauss_bf1 = make_double_1d_gauss(phi_double, double_gauss_fit1[0][0], double_gauss_fit1[0][1],
        #                                 double_gauss_fit1[0][2], double_gauss_fit1[0][3],
        #                                 C1=double_gauss_fit1[0][4], C2=double_gauss_fit1[0][5])
        # inds_1 = np.where((phi_double > gauss_fit1[0][0] - 2.355*gauss_fit1[0][1]) &
        #                 (phi_double < gauss_fit1[0][0] + 2.355*gauss_fit1[0][1]))
        # fwhm_1 = measure_fwhm(cut_double[inds_1], phi_double[inds_1])
        # print("1: FWHM measured = {:.3f}, FWHM fit = {:.3f}".format(fwhm_1, gauss_fit1[0][1]*2.355))
        # Pick the correct peak in the double-Gaussian model to be the disk
        # (meaning the brightest peak) and get the actual phi value for it
        # (not just the phi of the nearest pixel center).
        peakInd_1 = np.argmax(double_gauss_bf1)
        peakPhi_1_pix = phi_double[peakInd_1]
        pickPeak = np.argmin(np.abs(np.array([fitvals1['mu1'], fitvals1['mu2']]) - peakPhi_1_pix))
        if pickPeak == 0:
            peakPhi_1 = fitvals1['mu1']
        elif pickPeak == 1:
            peakPhi_1 = fitvals1['mu2']
        else:
            peakPhi_1 = fitvals1['mu1']
        # if np.argmin(np.abs(phi_double[peakInd_1] - np.array([fitvals1['mu1'], fitvals1['mu2']]))) == 0:
        #     peakPhi_1 = fitvals1['mu1']
        # else:
        #     peakPhi_1 = fitvals1['mu2']
        peakYX_1 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_1_pix)) == np.min(np.abs(phi - np.radians(peakPhi_1_pix)))))
        if len(peakYX_1[0]) == 0:
            peakYX_1 = (np.array([None]), np.array([None]))
        elif np.array(peakYX_1).size > 2:
            nearestRadInd = np.argmin(np.abs(rad - radii[peakYX_1]))
            peakYX_1 = (np.array([peakYX_1[0][nearestRadInd]]), np.array([peakYX_1[1][nearestRadInd]]))
        # print("1: r={}; Peak PA = {:.2f} ; Y,X = {}".format(rad, peakPhi_1, peakYX_1))
    except Exception as ee:
        print(ee)
        print("Failed to double fit side 1")
        double_gauss_bf1 = np.nan*np.ones(phi_double.shape)
        peakPhi_1 = np.nan
        peakYX_1 = (np.array([None]), np.array([None]))
        fitvals1 = p0_1

    # # Single Gaussian fit to side 1.
    # if np.isnan(peakPhi_1):
    #     try:
    #         side1_inds = np.where((phi_double < diskPA + 60) & (phi_double > diskPA - 60))
    #         gauss_fit1 = fit_gauss_1d(cut_double_clean, phi_double,
    #                                   indMinMax=(side1_inds[0][0], side1_inds[0][-1]),
    #                                   p0=(diskPA, fwhm_guess_phi, 10.),
    #                                   bounds=(np.array([diskPA-2*fwhm_guess_phi, fwhm_guess_phi*0.1, 1e-1]),
    #                                           np.array([diskPA+2*fwhm_guess_phi, fwhm_guess_phi*2, np.inf])))
    #         gauss_bf1 = make_1d_gauss(phi_double, gauss_fit1[0][0], gauss_fit1[0][1], C=gauss_fit1[0][2])
    #         # fwhm_1 = gauss_fit1[0][1]*2.355
    #         inds_1 = np.where((phi_double > gauss_fit1[0][0] - 2.355*gauss_fit1[0][1]) &
    #                         (phi_double < gauss_fit1[0][0] + 2.355*gauss_fit1[0][1]))
    #         fwhm_1 = measure_fwhm(cut_double[inds_1], phi_double[inds_1])
    #         print("1: FWHM measured = {:.3f}, FWHM fit = {:.3f}".format(fwhm_1, gauss_fit1[0][1]*2.355))
    #         peakInd_1 = np.argmax(gauss_bf1)
    #         peakPhi_1 = phi_double[peakInd_1]
    #         peakYX_1 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_1)) == np.min(np.abs(phi - np.radians(peakPhi_1)))))
    #         if len(peakYX_1[0]) == 0:
    #             peakYX_1 = (np.array([None]), np.array([None]))
    #         elif np.array(peakYX_1).size > 2:
    #             peakYX_1 = (peakYX_1[0])
    #         print("1: Peak PA = {:.2f} ; Y,X = {}".format(peakPhi_1, peakYX_1))
    #     except Exception as ee:
    #         print(ee)
    #         print("Failed to single fit side 1")
    #         gauss_bf1 = np.nan*np.ones(phi_double.shape)
    #         peakPhi_1 = np.nan
    #         peakYX_1 = (np.array([None]), np.array([None]))

 #--- SIDE 2 FITTING ---#
    # Double Gaussian fit to side 2.
    try:
        double_gauss_fit2 = fit_double_gauss_1d(cut_double_clean, phi_double,
                                  indMinMax=(side2_inds[0][0], side2_inds[0][-1]),
                                  p0=p0_2, bounds=bounds2)
        fitvals2 = double_gauss_fit2.best_values
        double_gauss_bf2 = make_double_1d_gauss(phi_double, fitvals2['mu1'], fitvals2['sig1'],
                                        fitvals2['mu2'], fitvals2['sig2'],
                                        C1=fitvals2['C1'], C2=fitvals2['C2'])
        # inds_2 = np.where((phi_double > gauss_fit2[0][0] - 2.355*gauss_fit2[0][1]) &
        #             (phi_double < gauss_fit2[0][0] + 2.355*gauss_fit2[0][1]))
        # fwhm_2 = measure_fwhm(cut_double[inds_2], phi_double[inds_2])
        # print("2: FWHM measured = {:.3f}, FWHM fit = {:.3f}".format(fwhm_2, gauss_fit2[0][1]*2.355))
        # Pick the correct peak in the double-Gaussian model to be the disk
        # (meaning the brightest peak) and get the actual phi value for it
        # (not just the phi of the nearest pixel center).
        peakInd_2 = np.argmax(double_gauss_bf2)
        peakPhi_2_pix = phi_double[peakInd_2]
        pickPeak = np.argmin(np.abs(np.array([fitvals2['mu1'], fitvals2['mu2']]) - peakPhi_2_pix))
        if pickPeak == 0:
            peakPhi_2 = fitvals2['mu1']
        elif pickPeak == 1:
            peakPhi_2 = fitvals2['mu2']
        else:
            peakPhi_2 = fitvals2['mu1']
        # if np.argmin(np.abs(phi_double[peakInd_2] - np.array([fitvals2['mu1'], fitvals2['mu2']]))) == 0:
        #     peakPhi_2 = fitvals2['mu1']
        # else:
        #     peakPhi_2 = fitvals2['mu2']
        if peakPhi_2 >= 360:
            peakYX_2 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_2_pix - 360)) == np.min(np.abs(phi - np.radians(peakPhi_2_pix - 360)))))
        else:
            peakYX_2 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_2_pix)) == np.min(np.abs(phi - np.radians(peakPhi_2_pix)))))
        if len(peakYX_2[0]) == 0:
            peakYX_2 = (np.array([None]), np.array([None]))
        elif np.array(peakYX_2).size > 2:
            nearestRadInd = np.argmin(np.abs(rad - radii[peakYX_2]))
            peakYX_2 = (np.array([peakYX_2[0][nearestRadInd]]), np.array([peakYX_2[1][nearestRadInd]]))
        # print("2: r={}, Peak PA = {:.2f} ; Y,X = {}".format(rad, peakPhi_2, peakYX_2))
    except Exception as ee:
        print(ee)
        print("Failed to double fit side 2")
        double_gauss_bf2 = np.nan*np.ones(phi_double.shape)
        peakPhi_2 = np.nan
        peakYX_2 = (np.array([None]), np.array([None]))
        fitvals2 = p0_2

    # if np.isnan(peakPhi_2):
    #     # Single Gaussian fit to side 2.
    #     try:
    #         side2_inds = np.where((phi_double < diskPA + 180 + 60) & (phi_double > diskPA + 180 - 60))
    #         gauss_fit2 = fit_gauss_1d(cut_double_clean, phi_double,
    #                                   indMinMax=(side2_inds[0][0], side2_inds[0][-1]),
    #                                   p0=(diskPA+180, fwhm_guess_phi, 10.),
    #                                   bounds=(np.array([diskPA+180-2*fwhm_guess_phi, fwhm_guess_phi*0.5, 1e-1]),
    #                                           np.array([diskPA+180+2*fwhm_guess_phi, fwhm_guess_phi*2, np.inf])))
    #         gauss_bf2 = make_1d_gauss(phi_double, gauss_fit2[0][0], gauss_fit2[0][1], C=gauss_fit2[0][2])
    #         # inds_2 = np.where((phi_double > gauss_fit2[0][0] - 2.355*gauss_fit2[0][1]) &
    #         #             (phi_double < gauss_fit2[0][0] + 2.355*gauss_fit2[0][1]))
    #         # fwhm_2 = measure_fwhm(cut_double[inds_2], phi_double[inds_2])
    #         # print("2: FWHM measured = {:.3f}, FWHM fit = {:.3f}".format(fwhm_2, gauss_fit2[0][1]*2.355))
    #         peakInd_2 = np.argmax(gauss_bf2)
    #         peakPhi_2 = phi_double[peakInd_2]
    #         if peakPhi_2 >= 360:
    #             peakYX_2 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_2 - 360)) == np.min(np.abs(phi - np.radians(peakPhi_2 - 360)))))
    #         else:
    #             peakYX_2 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_2)) == np.min(np.abs(phi - np.radians(peakPhi_2)))))
    #         if len(peakYX_2[0]) == 0:
    #             peakYX_2 = (np.array([None]), np.array([None]))
    #         elif np.array(peakYX_2).size > 2:
    #             peakYX_2 = (peakYX_2[0])
    #         print("2: Peak PA = {:.2f} ; Y,X = {}".format(peakPhi_2, peakYX_2))
    #     except:
    #         print("Failed to single fit side 2")
    #         gauss_bf2 = np.nan*np.ones(phi_double.shape)
    #         peakPhi_2 = np.nan
    #         peakYX_2 = (np.array([None]), np.array([None]))

    # # Clean clearly bad points that are far from the disk.
    # if np.abs(peakPhi_1 - diskPA) > 25:
    #     peakPhi_1 = np.nan
    #     peakYX_1 = (np.array([None]), np.array([None]))
    # if np.abs(peakPhi_2 - (diskPA + 180)) > 25:
    #     peakPhi_2 = np.nan
    #     peakYX_2 = (np.array([None]), np.array([None]))

    # plt.ion()
    # plt.figure(2)
    # plt.clf()
    # # plt.plot(phi_sort, cut_sort)
    # plt.plot(phi_double, cut_double, linestyle='None', marker='.') #, markersize=8)
    # # plt.plot(phi_double, gauss_bf1, 'k:')
    # # plt.plot(phi_double, gauss_bf2, 'm:')
    # plt.plot(phi_double, double_gauss_bf1, '#FF6A00')
    # plt.plot(phi_double, double_gauss_bf2, 'm-')
    # plt.xlabel('PA (degrees)')
    # if diskPA is not None:
    #     plt.axvline(x=diskPA, c='#FF6A00', linestyle='--')
    #     # plt.axvline(x=diskPA_rad - 2*np.pi, c='k', linestyle='--')
    #     # plt.axvline(x=diskPA_rad + 2*np.pi, c='k', linestyle='--')
    #     # plt.axvline(x=diskPA_rad - np.pi, c='C1', linestyle='--')
    #     # plt.axvline(x=diskPA_rad + np.pi, c='C1', linestyle='--')
    #     plt.axvline(x=diskPA + 360, c='#FF6A00', linestyle='--')
    #     plt.axvline(x=diskPA - 180, c='m', linestyle='--')
    #     plt.axvline(x=diskPA + 180, c='m', linestyle='--')
    # # ax_top = plt.twiny()
    # # ax_top.set_xlim(0, len(cut_double))
    # # ax_top.set_xlabel('Slice Index')
    # plt.draw()
    # 
    # # breakpoint()
    # if np.abs(fitvals1['mu1'] - diskPA) > 30:
    #     pdb.set_trace()
    # if np.abs(fitvals2['mu1'] - 180 - diskPA) > 30:
    #     pdb.set_trace()
    
    # pdb.set_trace()

    return peakPhi_1, peakPhi_2, peakYX_1, peakYX_2, fitvals1, fitvals2


def vertical_profile_gp(fp, rad=60, highpass=None, diskPA=None, star=None,
                     maxDiskWidth_pix=35, tight=False, data=None, hdr=None, hdu=None,
                     bounds1=None, bounds2=None, errMap=None):
    """
    Use Gaussian processes to model an azimuthal slice of disk at constant
    radius and find the position angle of the brightness peak.

    Inputs:
      rad: radii of profile in [pix]
      diskPA: disk major axis PA in [degrees]
      maxDiskWidth_pix: maximum "vertical" width of the disk in [pixels]
    """
    
    # DEFINE some data cleaning and fitting parameters.
    dr = 0.5 # half-width of each slice in [pix]
    wx = 1 # [pix]
    dvert = 40 # half-length of each slice in vertical direction from star [pix]
    
    if (data is None) | (hdr is None) | (hdu is None):
        with fits.open(os.path.expanduser(fp)) as hduf:
            hdu = hduf
            data = hdu[0].data
            hdr = hdu[0].header

# # TESTING!!! Smooth image first.
#             data = median_filter(data, 5)

    if highpass is not None:
        data = unsharp('', hdu, '', B=highpass, ident='_999', output=True, silent=True,
                        save=False, parOK=True, gauss=False)

    if errMap is not None:
        with fits.open(os.path.expanduser(errMap)) as hduf:
            hduErr = hduf
            err = hduErr[0].data
            hdrErr = hduErr[0].header

    if diskPA is not None:
        diskPA_rad = np.radians(diskPA)
    else:
        diskPA_rad = 0.

    if star is None:
        star = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])
    radii = make_radii(data, star)
    phi = make_phi(data, star)
    phi -= np.pi/2
    phi[phi < 0] += 2*np.pi
    # if diskPA is not None:
    #     phi -= (diskPA_rad + np.pi/2)
    
    # Width of a single pixel in phi at radius rr.
    pixel_width_phi = np.degrees(np.arcsin(1./rad)) # [deg/pixel]
    # Estimate disk FWHM to get initial guess of Gaussian sigma for fit.
    fwhm_guess_phi = 0.5*maxDiskWidth_pix*pixel_width_phi/2.355 # [deg]
    
    phiCut_1 = phi[(radii < rad + dr) & (radii > rad - dr) & (phi < np.pi)]
    cut_1 = data[(radii < rad + dr) & (radii > rad - dr) & (phi < np.pi)]
    
    phiCut_2 = phi[(radii < rad + dr) & (radii > rad - dr) & (phi >= np.pi)]
    cut_2 = data[(radii < rad + dr) & (radii > rad - dr) & (phi >= np.pi)]
    
    # Optionally, if tight == True, set the entire slice to 0 outside of the
    # maxDiskWidth_pix range around diskPA.
    if tight:
        tightInds_1 = ((phiCut_1 > diskPA_rad - np.radians(fwhm_guess_phi)) & \
                       (phiCut_1 < diskPA_rad + np.radians(fwhm_guess_phi)))
        tightInds_2 = ((phiCut_2 > diskPA_rad + np.pi - np.radians(fwhm_guess_phi)) & \
                       (phiCut_2 < diskPA_rad + np.pi + np.radians(fwhm_guess_phi)))
        trimInds_1 = ((phiCut_1 > diskPA_rad - np.radians(3*fwhm_guess_phi)) & \
                       (phiCut_1 < diskPA_rad + np.radians(3*fwhm_guess_phi)))
        trimInds_2 = ((phiCut_2 > diskPA_rad + np.pi - np.radians(3*fwhm_guess_phi)) & \
                       (phiCut_2 < diskPA_rad + np.pi + np.radians(3*fwhm_guess_phi)))
        # cut_1[phiCut_1 < diskPA_rad - np.radians(fwhm_guess_phi)] = 0.
        # cut_1[phiCut_1 > diskPA_rad + np.radians(fwhm_guess_phi)] = 0.
        # cut_2[phiCut_2 < diskPA_rad + np.pi - np.radians(fwhm_guess_phi)] = 0.
        # cut_2[phiCut_2 > diskPA_rad + np.pi + np.radians(fwhm_guess_phi)] = 0.
    else:
        tightInds_1 = ()
        tightInds_2 = ()
        trimInds_1 = ()
        trimInds_2 = ()

    # cut_1 = cut_1[tightInds_1]
    # cut_2 = cut_2[tightInds_2]
    # phiCut_1 = phiCut_1[tightInds_1]
    # phiCut_2 = phiCut_2[tightInds_2]
    cut_1[~tightInds_1] = 0
    cut_2[~tightInds_2] = 0
    cut_1 = cut_1[trimInds_1]
    cut_2 = cut_2[trimInds_2]
    phiCut_1 = phiCut_1[trimInds_1]
    phiCut_2 = phiCut_2[trimInds_2]

    sortInds_1 = np.argsort(phiCut_1)
    phiCut_sort_1 = phiCut_1[sortInds_1]
    cut_sort_1 = cut_1[sortInds_1]
    sortInds_2 = np.argsort(phiCut_2)
    phiCut_sort_2 = phiCut_2[sortInds_2]
    cut_sort_2 = cut_2[sortInds_2]

    # phi_double_1 = np.degrees(np.append(phiCut_sort_1, phiCut_sort_1 + 2*np.pi))
    # cut_double_1 = np.append(cut_sort_1, cut_sort_1)
    
# # # TESTING!!! Smoothing the slice.
#     cut_double = gaussian_filter1d(cut_double.copy(), sigma=2, mode='reflect')
    
    try:
        cutErr = err[(radii < rad + dr) & (radii > rad - dr) & (phi < np.pi)]
        cutErr = err[trimInds_1]
        cut2Err = err[(radii < rad + dr) & (radii > rad - dr) & (phi >= np.pi)]
        cut2Err = err[trimInds_2]
    except:
        cutErr = np.ones(cut_sort_1.shape)
        cut2Err = np.ones(cut_sort_2.shape)

    # dataRange = np.arange(cut_sort_1.shape[0])

    # cut[(dataRange < star[0] - dvert) | (dataRange > star[0] + dvert)] = np.nanmedian(cut)
    # cut2[(dataRange < star[0] - dvert) | (dataRange > star[0] + dvert)] = np.nanmedian(cut2)
    # Set all NaN values to zero to avoid fitter failing.
    cut_clean_1 = np.nan_to_num(cut_sort_1.copy(), 0.)
    cut_clean_2 = np.nan_to_num(cut_sort_2.copy(), 0.)
    cutErr_clean = np.nan_to_num(cutErr.copy(), np.inf)
    cut2Err_clean = np.nan_to_num(cut2Err.copy(), np.inf)

    # # If the slice is flat zeros, skip the fit and move on.
    # if (np.all(cut_double_clean == 0) or np.all(np.isnan(cut_double_clean))):
    #     print("r={}; No disk: empty data slice".format(rad))
    #     return np.nan, np.nan, (np.array([None]), np.array([None])), (np.array([None]), np.array([None])), None, None
    

# # # TESTING!!! Smoothing the slice.
#     cut_double = gaussian_filter1d(cut_double.copy(), sigma=2, mode='reflect')

    # # If the slice is flat zeros, skip the fit and move on.
    # if (np.all(cut_clean == 0) or np.all(np.isnan(cut_clean))):
    #     print("r={}; No disk: empty data slice".format(rad))
    #     return (np.array([None]), np.array([None])), (np.array([None]), np.array([None])), None, None

    # Set bounds for the double Gaussian model fit parameters.
    # Param order: mu1, sig1, mu2, sig2, C1, C2
    # Locate slice's brightest pixel on both sides.
    max1 = np.max(cut_clean_1)
    max1_ind = np.argmax(cut_clean_1)
    max2 = np.max(cut_clean_2)
    max2_ind = np.argmax(cut_clean_2)
    # # Guess that disk midplane is centered on slice's brightest pixel, to start.
    # p0_1 = (max1_ind, fwhm_guess_phi, max1_ind-5, fwhm_guess_phi, 30., 30.)
    # p0_2 = (max2_ind, fwhm_guess_phi, max2_ind+5, fwhm_guess_phi, 30., 30.)

    # if bounds1 is None:
    #     bounds1 = (np.array([p0_1[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, p0_1[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, 0, 0]),
    #                np.array([p0_1[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_1[2]+2*fwhm_guess_phi, fwhm_guess_phi*2, 1e5, 1e5]))
    # if bounds2 is None:
    #     bounds2 = (np.array([p0_2[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, p0_2[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, 0, 0]),
    #                np.array([p0_2[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_2[2]+2*fwhm_guess_phi, fwhm_guess_phi*4, 1e5, 1e5]))

 #--- SIDE 1 FITTING ---#
    try:
        xPeak1_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, snr_1 = fit_gp_1d(cut_clean_1, phiCut_sort_1,
                        indMinMax=(0, -1),
                        err=cutErr_clean)
        # xPeak1_gp += (diskPA_rad + np.pi/2)
    except Exception as ee:
        print(ee)
        xPeak1_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, snr_1 = np.nan, np.nan, np.nan, np.nan

 #--- SIDE 2 FITTING ---#
    try:
        xPeak2_gp, xPeak2Err_lo_gp, xPeak2Err_up_gp, snr_2 = fit_gp_1d(cut_clean_2, phiCut_sort_2,
                        indMinMax=(0, -1),
                        err=cut2Err_clean)
        # xPeak2_gp += (diskPA_rad + np.pi/2)
    except Exception as ee:
        print(ee)
        xPeak2_gp, xPeak2Err_lo_gp, xPeak2Err_up_gp, snr_2 = np.nan, np.nan, np.nan, np.nan

    # plt.figure(31)
    # plt.clf()
    # plt.title('Side 1')
    # plt.plot(np.degrees(phiCut_sort_1), cut_clean_1, c='k', marker='o')
    # try:
    #     plt.axvline(x=np.degrees(xPeak1_gp), c='m')
    # except:
    #     pass
    # plt.draw()
    
    # plt.figure(32)
    # plt.clf()
    # plt.title('Side 2')
    # plt.plot(np.degrees(phiCut_sort_2), cut_clean_2, c='0.6', marker='^')
    # try:
    #     plt.axvline(x=np.degrees(xPeak2_gp), c='m')
    # except:
    #     pass
    # plt.draw()
    
    # # try:
    # #     peakErr2 = np.abs(np.array([confInterval2[peakKey][2][1] - confInterval2[peakKey][3][1], confInterval2[peakKey][4][1] - confInterval2[peakKey][3][1]]))
    # # except:
    # #     peakErr2 = np.array([0, 0])
    
    # pdb.set_trace()
    
    return xPeak1_gp, xPeak2_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, xPeak2Err_lo_gp, xPeak2Err_up_gp, snr_1, snr_2


def vertical_profile_y(fp, rad=60, highpass=None, diskPA=None, star=None,
                     maxDiskWidth_pix=35, data=None, hdr=None, hdu=None,
                     bounds1=None, bounds2=None, errMap=None):
    """
    Use a double Gaussian function to fit the vertical slice of a disk and find
    its brightness peak.
    
    Inputs:
      rad: radius of profile in [pix]
      diskPA: disk major axis PA in [degrees]
      maxDiskWidth_pix: maximum "vertical" width of the disk in [pixels]
    """
    
    if (data is None) | (hdr is None) | (hdu is None):
        with fits.open(os.path.expanduser(fp)) as hduf:
            hdu = hduf
            data = hdu[0].data
            hdr = hdu[0].header

# TESTING!!! Smooth image first.
            data = median_filter(data, 5)

    if highpass is not None:
        data = unsharp('', hdu, '', B=highpass, ident='_999', output=True, silent=True,
                        save=False, parOK=True, gauss=False)

    if errMap is not None:
        with fits.open(os.path.expanduser(errMap)) as hduf:
            hduErr = hduf
            err = hduErr[0].data
            hdrErr = hduErr[0].header

    if diskPA is not None:
        diskPA_rad = np.radians(diskPA)

    if star is None:
        star = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])
    radii = make_radii(data, star)
    dr = 1.5 # half-width of each slice in [pix]
    wx = 1 # [pix]

    cut = np.nanmean(data[:, star[1]+rad-wx:star[1]+rad+wx+1], axis=1)
    cut2 = np.nanmean(data[:, star[1]-rad-wx:star[1]-rad+wx+1], axis=1)
    
    try:
        cutErr = np.nanmean(err[:, star[1]+rad-wx:star[1]+rad+wx+1], axis=1)
        cut2Err = np.nanmean(err[:, star[1]-rad-wx:star[1]-rad+wx+1], axis=1)
    except:
        cutErr = np.ones(cut.shape)
        cut2Err = np.ones(cut.shape)

    dataRange = np.arange(cut.shape[0])

    cut[(dataRange < star[0] - 150) | (dataRange > star[0] + 151)] = np.nanmedian(cut)
    cut2[(dataRange < star[0] - 150) | (dataRange > star[0] + 151)] = np.nanmedian(cut2)
    # Set all NaN values to zero to avoid fitter failing.
    cut_clean = np.nan_to_num(cut.copy(), 0.)
    cut2_clean = np.nan_to_num(cut2.copy(), 0.)
    cutErr_clean = np.nan_to_num(cutErr.copy(), np.inf)
    cut2Err_clean = np.nan_to_num(cut2Err.copy(), np.inf)

# # # TESTING!!! Smoothing the slice.
#     cut_double = gaussian_filter1d(cut_double.copy(), sigma=2, mode='reflect')

    # Width of a single pixel in phi at radius rr.
    pixel_width_phi = np.degrees(np.arcsin(1./rad)) # [deg/pixel]
    # Estimate disk FWHM to get initial guess of Gaussian sigma for fit.
    fwhm_guess_phi = 0.5*maxDiskWidth_pix*pixel_width_phi/2.355 # [deg]

    # # If the slice is flat zeros, skip the fit and move on.
    # if (np.all(cut_clean == 0) or np.all(np.isnan(cut_clean))):
    #     print("r={}; No disk: empty data slice".format(rad))
    #     return (np.array([None]), np.array([None])), (np.array([None]), np.array([None])), None, None

    # Set bounds for the double Gaussian model fit parameters.
    # Param order: mu1, sig1, mu2, sig2, C1, C2
    # Locate slice's brightest pixel on both sides.
    max1 = np.max(cut_clean)
    max1_ind = np.argmax(cut_clean)
    max2 = np.max(cut2_clean)
    max2_ind = np.argmax(cut2_clean)
    # Guess that disk midplane is centered on slice's brightest pixel, to start.
    # p0_1 = (max1_ind, 0.5*fwhm_guess_phi, max1_ind+5, 0.5*fwhm_guess_phi, 30., 30.)
    # p0_2 = (max2_ind, 0.5*fwhm_guess_phi, max2_ind+5, 0.5*fwhm_guess_phi, 30., 30.)
    p0_1 = (max1_ind, fwhm_guess_phi, max1_ind-5, fwhm_guess_phi, 30., 30.)
    p0_2 = (max2_ind, fwhm_guess_phi, max2_ind+5, fwhm_guess_phi, 30., 30.)
    # if (max1_phi <= diskPA + 30) and (max1_phi >= diskPA - 30):
    #     p0_1 = (max1_phi, 0.5*fwhm_guess_phi, max1_phi+5, 0.5*fwhm_guess_phi, 30., 30.)
    # else:
    #     p0_1 = (diskPA, 0.5*fwhm_guess_phi, diskPA+5, 0.5*fwhm_guess_phi, 30., 30.)
    # if (max2_phi <= diskPA + 180 + 30) and (max2_phi >= diskPA + 180 - 30):
    #     p0_2 = (max2_phi, 0.5*fwhm_guess_phi, max2_phi-5, 0.5*fwhm_guess_phi, 30., 30.)
    # else:
    #     p0_2 = (diskPA+180, 0.5*fwhm_guess_phi, diskPA+180-5, 0.5*fwhm_guess_phi, 30., 30.)
    if bounds1 is None:
        bounds1 = (np.array([p0_1[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, p0_1[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, 0, 0]),
                   np.array([p0_1[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_1[2]+2*fwhm_guess_phi, fwhm_guess_phi*2, 1e5, 1e5]))
    if bounds2 is None:
        bounds2 = (np.array([p0_2[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, p0_2[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, 0, 0]),
                   np.array([p0_2[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_2[2]+2*fwhm_guess_phi, fwhm_guess_phi*4, 1e5, 1e5]))

 #--- SIDE 1 FITTING ---#
    # Double Gaussian fit to side 1.
    try:
        double_gauss_fit1 = fit_double_gauss_1d(cut_clean, dataRange,
                                  indMinMax=(star[0] - 50, star[0] + 51),
                                  p0=p0_1, bounds=bounds1, err=cutErr_clean)
        fitvals1 = double_gauss_fit1.best_values
        confInterval1 = double_gauss_fit1.conf_interval()
        sigmas1 = double_gauss_fit1.covar.diagonal()**0.5
        double_gauss_bf1 = make_double_1d_gauss(dataRange, fitvals1['mu1'], fitvals1['sig1'],
                                        fitvals1['mu2'], fitvals1['sig2'],
                                        C1=fitvals1['C1'], C2=fitvals1['C2'])
        # Pick the correct peak in the double-Gaussian model to be the disk
        # (meaning the brightest peak) and get the actual phi value for it
        # (not just the phi of the nearest pixel center).
        peakInd_1 = np.argmax(double_gauss_bf1)
        pickPeak = np.argmax([double_gauss_bf1[int(fitvals1['mu1'])], double_gauss_bf1[int(fitvals1['mu2'])]])
        # pickPeak = np.argmin(np.abs(np.array([fitvals1['mu1'], fitvals1['mu2']]) - peakPhi_1_pix))
        if pickPeak == 0:
            peakY_1 = fitvals1['mu1']
        elif pickPeak == 1:
            peakY_1 = fitvals1['mu2']
        else:
            peakY_1 = fitvals1['mu1']
        peakYX_1 = (np.array([peakY_1]), np.array([rad]))
        # peakYX_1 = np.where((radii < rad + dr) & (radii > rad - dr) & (np.abs(phi - np.radians(peakPhi_1_pix)) == np.min(np.abs(phi - np.radians(peakPhi_1_pix)))))
        # if len(peakYX_1[0]) == 0:
        #     peakYX_1 = (np.array([None]), np.array([None]))
        # elif np.array(peakYX_1).size > 2:
        #     nearestRadInd = np.argmin(np.abs(rad - radii[peakYX_1]))
        #     peakYX_1 = (np.array([peakYX_1[0][nearestRadInd]]), np.array([peakYX_1[1][nearestRadInd]]))
        # # print("1: r={}; Peak PA = {:.2f} ; Y,X = {}".format(rad, peakPhi_1, peakYX_1))
    except Exception as ee:
        print(ee)
        print("Failed to double fit side 1")
        double_gauss_bf1 = np.nan*np.ones(cut_clean.shape)
        # peakPhi_1 = np.nan
        peakYX_1 = (np.array([np.nan]), np.array([np.nan]))
        fitvals1 = p0_1
        confInterval1 = None
        sigmas1 = None

    try:
        peakErr1 = np.abs(np.array([confInterval1[peakKey][2][1] - confInterval1[peakKey][3][1], confInterval1[peakKey][4][1] - confInterval1[peakKey][3][1]]))
    except:
        peakErr1 = np.array([0, 0])

 #--- SIDE 2 FITTING ---#
    # Double Gaussian fit to side 2.
    try:
        double_gauss_fit2 = fit_double_gauss_1d(cut2_clean, dataRange,
                                  indMinMax=(star[0] - 150, star[0] + 151),
                                  p0=p0_2,
                                  bounds=bounds2, err=cut2Err_clean)
        fitvals2 = double_gauss_fit2.best_values
        confInterval2 = double_gauss_fit2.conf_interval()
        sigmas2 = double_gauss_fit2.covar.diagonal()**0.5
        double_gauss_bf2 = make_double_1d_gauss(dataRange, fitvals2['mu1'], fitvals2['sig1'],
                                        fitvals2['mu2'], fitvals2['sig2'],
                                        C1=fitvals2['C1'], C2=fitvals2['C2'])
        # Pick the correct peak in the double-Gaussian model to be the disk
        # (meaning the brightest peak) and get the actual phi value for it
        # (not just the phi of the nearest pixel center).
        peakInd_2 = np.argmax(double_gauss_bf2)
        pickPeak = np.argmax([double_gauss_bf2[int(fitvals2['mu1'])], double_gauss_bf2[int(fitvals2['mu2'])]])
        # pickPeak = np.argmin(np.abs(np.array([fitvals1['mu1'], fitvals1['mu2']]) - peakPhi_1_pix))
        if pickPeak == 0:
            peakKey = 'mu1'
        elif pickPeak == 1:
            peakKey = 'mu2'
        else:
            peakKey = 'mu1'
        peakY_2 = fitvals2[peakKey]
        peakYX_2 = (np.array([peakY_2]), np.array([rad]))
        # fwhm_2 = measure_fwhm(cut_double[inds_2], phi_double[inds_2])
        # print("2: FWHM measured = {:.3f}, FWHM fit = {:.3f}".format(fwhm_2, gauss_fit2[0][1]*2.355))
        # if len(peakYX_2[0]) == 0:
        #     peakYX_2 = (np.array([None]), np.array([None]))
        # elif np.array(peakYX_2).size > 2:
        #     nearestRadInd = np.argmin(np.abs(rad - radii[peakYX_2]))
        #     peakYX_2 = (np.array([peakYX_2[0][nearestRadInd]]), np.array([peakYX_2[1][nearestRadInd]]))
        # print("2: r={}, Peak PA = {:.2f} ; Y,X = {}".format(rad, peakPhi_2, peakYX_2))
    except Exception as ee:
        print(ee)
        print("Failed to double fit side 2")
        double_gauss_bf2 = np.nan*np.ones(cut2_clean.shape)
        # peakPhi_2 = np.nan
        peakYX_2 = (np.array([np.nan]), np.array([np.nan]))
        fitvals2 = p0_2
        confInterval2 = None
        sigmas2 = None

    # plt.figure(30)
    # plt.clf()
    # plt.title('Side 1')
    # plt.plot(dataRange[star[0] - 75:star[0] + 75], cut_clean[star[0] - 75:star[0] + 75], 'ko')
    # try:
    #     plt.plot(dataRange[star[0] - 75:star[0] + 75], double_gauss_bf1[star[0] - 75:star[0] + 75], 'b-')
    #     plt.axvline(x=fitvals1['mu1'], c='m')
    # except:
    #     pass
    # plt.draw()
    # 
    # plt.figure(31)
    # plt.clf()
    # plt.title('Side 2')
    # plt.plot(dataRange[star[0] - 75:star[0] + 75], cut2_clean[star[0] - 75:star[0] + 75], 'ko')
    # try:
    #     plt.plot(dataRange[star[0] - 75:star[0] + 75], double_gauss_bf2[star[0] - 75:star[0] + 75], 'b-')
    #     plt.axvline(x=fitvals2['mu1'], c='m')
    # except:
    #     pass
    # plt.draw()

    try:
        peakErr2 = np.abs(np.array([confInterval2[peakKey][2][1] - confInterval2[peakKey][3][1], confInterval2[peakKey][4][1] - confInterval2[peakKey][3][1]]))
    except:
        peakErr2 = np.array([0, 0])

    # pdb.set_trace()

    return peakYX_1, peakYX_2, fitvals1, fitvals2, sigmas1, sigmas2, peakErr1, peakErr2


def vertical_profile_y_gp(fp, rad=60, highpass=None, diskPA=None, star=None,
                     maxDiskWidth_pix=35, data=None, hdr=None, hdu=None,
                     bounds1=None, bounds2=None, errMap=None):
    """
    Use Gaussian processes to model a vertical slice of disk and find the
    brightness peak.

    Inputs:
      rad: radius of profile in [pix]
      diskPA: disk major axis PA in [degrees]
      maxDiskWidth_pix: maximum "vertical" width of the disk in [pixels]
    """

    # DEFINE some data cleaning and fitting parameters.
    dr = 1.5 # half-width of each slice in [pix]
    wx = 1 # [pix]
    dvert = 40 # half-length of each slice in vertical direction from star [pix]
    
    if (data is None) | (hdr is None) | (hdu is None):
        with fits.open(os.path.expanduser(fp)) as hduf:
            hdu = hduf
            data = hdu[0].data
            hdr = hdu[0].header

# # TESTING!!! Smooth image first.
#             data = median_filter(data, 5)

    if highpass is not None:
        data = unsharp('', hdu, '', B=highpass, ident='_999', output=True, silent=True,
                        save=False, parOK=True, gauss=False)

    if errMap is not None:
        with fits.open(os.path.expanduser(errMap)) as hduf:
            hduErr = hduf
            err = hduErr[0].data
            hdrErr = hduErr[0].header

    if diskPA is not None:
        diskPA_rad = np.radians(diskPA)

    if star is None:
        star = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])
    # radii = make_radii(data, star)


    cut = np.nanmean(data[:, star[1]+rad-wx:star[1]+rad+wx+1], axis=1)
    cut2 = np.nanmean(data[:, star[1]-rad-wx:star[1]-rad+wx+1], axis=1)
    
    try:
        cutErr = np.nanmean(err[:, star[1]+rad-wx:star[1]+rad+wx+1], axis=1)
        cut2Err = np.nanmean(err[:, star[1]-rad-wx:star[1]-rad+wx+1], axis=1)
    except:
        cutErr = np.ones(cut.shape)
        cut2Err = np.ones(cut.shape)

    dataRange = np.arange(cut.shape[0])

    cut[(dataRange < star[0] - dvert) | (dataRange > star[0] + dvert)] = np.nanmedian(cut)
    cut2[(dataRange < star[0] - dvert) | (dataRange > star[0] + dvert)] = np.nanmedian(cut2)
    # Set all NaN values to zero to avoid fitter failing.
    cut_clean = np.nan_to_num(cut.copy(), 0.)
    cut2_clean = np.nan_to_num(cut2.copy(), 0.)
    cutErr_clean = np.nan_to_num(cutErr.copy(), np.inf)
    cut2Err_clean = np.nan_to_num(cut2Err.copy(), np.inf)

# # # TESTING!!! Smoothing the slice.
#     cut_double = gaussian_filter1d(cut_double.copy(), sigma=2, mode='reflect')

    # Width of a single pixel in phi at radius rr.
    pixel_width_phi = np.degrees(np.arcsin(1./rad)) # [deg/pixel]
    # Estimate disk FWHM to get initial guess of Gaussian sigma for fit.
    fwhm_guess_phi = 0.5*maxDiskWidth_pix*pixel_width_phi/2.355 # [deg]

    # # If the slice is flat zeros, skip the fit and move on.
    # if (np.all(cut_clean == 0) or np.all(np.isnan(cut_clean))):
    #     print("r={}; No disk: empty data slice".format(rad))
    #     return (np.array([None]), np.array([None])), (np.array([None]), np.array([None])), None, None

    # Set bounds for the double Gaussian model fit parameters.
    # Param order: mu1, sig1, mu2, sig2, C1, C2
    # Locate slice's brightest pixel on both sides.
    max1 = np.max(cut_clean)
    max1_ind = np.argmax(cut_clean)
    max2 = np.max(cut2_clean)
    max2_ind = np.argmax(cut2_clean)
    # Guess that disk midplane is centered on slice's brightest pixel, to start.
    # p0_1 = (max1_ind, 0.5*fwhm_guess_phi, max1_ind+5, 0.5*fwhm_guess_phi, 30., 30.)
    # p0_2 = (max2_ind, 0.5*fwhm_guess_phi, max2_ind+5, 0.5*fwhm_guess_phi, 30., 30.)
    p0_1 = (max1_ind, fwhm_guess_phi, max1_ind-5, fwhm_guess_phi, 30., 30.)
    p0_2 = (max2_ind, fwhm_guess_phi, max2_ind+5, fwhm_guess_phi, 30., 30.)

    if bounds1 is None:
        bounds1 = (np.array([p0_1[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, p0_1[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.25, 0, 0]),
                   np.array([p0_1[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_1[2]+2*fwhm_guess_phi, fwhm_guess_phi*2, 1e5, 1e5]))
    if bounds2 is None:
        bounds2 = (np.array([p0_2[0]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, p0_2[2]-2*fwhm_guess_phi, fwhm_guess_phi*0.1, 0, 0]),
                   np.array([p0_2[0]+2*fwhm_guess_phi, fwhm_guess_phi*2, p0_2[2]+2*fwhm_guess_phi, fwhm_guess_phi*4, 1e5, 1e5]))

 #--- SIDE 1 FITTING ---#

# # TEMP!!!
#     xPeak1_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, snr1 = fit_gp_1d(cut_clean,
#                         dataRange,
#                         indMinMax=(star[0] - dvert, star[0] + dvert),
#                         err=cutErr_clean)

    try:
        xPeak1_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, snr1_gp = fit_gp_1d(
                                cut_clean,
                                dataRange,
                                indMinMax=(star[0] - dvert, star[0] + dvert),
                                err=cutErr_clean)
    except:
        xPeak1_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, snr1_gp = np.nan, np.nan, np.nan, np.nan

 #--- SIDE 2 FITTING ---#
    try:
        xPeak2_gp, xPeak2Err_lo_gp, xPeak2Err_up_gp, snr2_gp = fit_gp_1d(
                                cut2_clean,
                                dataRange,
                                indMinMax=(star[0] - dvert, star[0] + dvert),
                                err=cut2Err_clean)
    except:
        xPeak2_gp, xPeak2Err_lo_gp, xPeak2Err_up_gp, snr2_gp = np.nan, np.nan, np.nan, np.nan

    # plt.figure(30)
    # plt.clf()
    # plt.title('Side 1')
    # plt.plot(dataRange[star[0] - 75:star[0] + 75], cut_clean[star[0] - 75:star[0] + 75], 'ko')
    # try:
    #     plt.plot(dataRange[star[0] - 75:star[0] + 75], double_gauss_bf1[star[0] - 75:star[0] + 75], 'b-')
    #     plt.axvline(x=fitvals1['mu1'], c='m')
    # except:
    #     pass
    # plt.draw()
    # 
    # plt.figure(31)
    # plt.clf()
    # plt.title('Side 2')
    # plt.plot(dataRange[star[0] - 75:star[0] + 75], cut2_clean[star[0] - 75:star[0] + 75], 'ko')
    # try:
    #     plt.plot(dataRange[star[0] - 75:star[0] + 75], double_gauss_bf2[star[0] - 75:star[0] + 75], 'b-')
    #     plt.axvline(x=fitvals2['mu1'], c='m')
    # except:
    #     pass
    # plt.draw()
    # 
    # try:
    #     peakErr2 = np.abs(np.array([confInterval2[peakKey][2][1] - confInterval2[peakKey][3][1], confInterval2[peakKey][4][1] - confInterval2[peakKey][3][1]]))
    # except:
    #     peakErr2 = np.array([0, 0])
    
    return xPeak1_gp, xPeak2_gp, xPeak1Err_lo_gp, xPeak1Err_up_gp, xPeak2Err_lo_gp, xPeak2Err_up_gp


def crosscorr_gauss_1d(yy, xx):

    # matTemplate = np.zeros(len(xx)*2)
    # mat = np.zeros(len(xx)*2)

    gaussTemplate = make_1d_gauss(xx, xx[len(xx)//2], 0.1)
    # matTemplate[:len(xx)] = gaussTemplate
    # mat[:len(xx)] = yy
    xcorr = correlate(gaussTemplate, yy, mode='full', method='direct')

    plt.figure(80)
    plt.clf()
    plt.plot(xcorr) #[len(xx)//4:int((3/4)*len(xx))])
    plt.draw()

    return xcorr


def fit_gauss_1d(yy, xx, indMinMax=None, p0=None, bounds=None):

    if indMinMax is not None:
        yyFit = yy[indMinMax[0]:indMinMax[1]]
        xxFit = xx[indMinMax[0]:indMinMax[1]]
    else:
        yyFit = yy
        xxFit = xx

    if p0 is None:
        p0 = (xxFit[len(xxFit)//2], 2., 10.)

    if bounds is None:
        bounds=(np.array([0., 0., 1e-1]), np.array([730., 10., np.inf]))

    fitResults = curve_fit(make_1d_gauss, xxFit, yyFit, p0=p0, bounds=bounds)
    print(fitResults)
    return fitResults


def fit_double_gauss_1d(yy, xx, indMinMax=None, p0=None, bounds=None, err=None):
    
    if indMinMax is not None:
        yyFitOrig = yy[indMinMax[0]:indMinMax[1]]
        xxFitOrig = xx[indMinMax[0]:indMinMax[1]]
    else:
        yyFitOrig = yy
        xxFitOrig = xx

    if err is None:
        errFit = np.ones(yyFitOrig.shape)
    else:
        errFit = err[indMinMax[0]:indMinMax[1]]
    
    # Interpolate vertical profile onto finer sampling to better locate the
    # peak of the profile.
    # breakpoint()
    # ftest = interp1d(xxFitOrig, yyFitOrig, kind='quadratic')
    # # breakpoint()
    # xxInterp = np.linspace(np.nanmin(xxFitOrig), np.nanmax(xxFitOrig), 1000)
    # 
    # ytest = ftest(xxInterp)
    # ytest[ytest < 0] = 0
    # 
    # xxFit = xxInterp
    # yyFit = ytest
    xxFit = xxFitOrig
    yyFit = yyFitOrig
    
    # plt.figure(4)
    # plt.clf()
    # plt.plot(xxFitOrig, yyFitOrig, 'o')
    # plt.plot(xxInterp, ytest, '.')
    # plt.draw()

    if p0 is None:
        C0 = np.nanmin(yyFit)
        p0 = (xxFit[len(xxFit)//2], 2., C0, xxFit[len(xxFit)//2], 2., C0)

    if bounds is None:
        bounds=(np.array([0., 0., 1e-1, 0., 0., 1e-1]),
                np.array([720., 10., np.inf, 720., 10., np.inf]))

    # Set weights for yyFit to 0 outside of the allowed bounds for mu so those
    # points do not influence the fit (assuming they are just noise).
    weights = 1./errFit**2 #np.ones(yyFit.shape)
    weights[(xxFit <= bounds[0][0]) | (xxFit >= bounds[1][0])] = 1e-2

    # whGood = ~((yyFit == 0) | (np.isnan(yyFit)))
    # fitResults = curve_fit(make_double_1d_gauss, xxFit[whGood], yyFit[whGood], p0=p0, bounds=bounds,
    #                        max_nfev=1e4)
    mod = Model(make_double_1d_gauss)
    params = mod.make_params(mu1=p0[0], sig1=p0[1], mu2=p0[2], sig2=p0[3],
                             C1=p0[4], C2=p0[5])
    params['mu1'].min = bounds[0][0]
    params['mu1'].max = bounds[1][0]
    params['sig1'].min = bounds[0][1]
    params['sig1'].max = bounds[1][1]
    params['mu2'].min = bounds[0][2]
    params['mu2'].max = bounds[1][2]
    params['sig2'].min = bounds[0][3]
    params['sig2'].max = bounds[1][3]
    params['C1'].min = 1e-2
    params['C2'].min = 1e-2
    fitResults = mod.fit(yyFit, params, xx=xxFit, weights=weights)

    return fitResults


def fit_gp_1d(yy, xx, indMinMax=None, err=None):
    """
    Parameters
    ----------
    yy : TYPE
        DESCRIPTION.
    xx : TYPE
        DESCRIPTION.
    indMinMax : TYPE, optional
        DESCRIPTION. The default is None.
    err : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    GPy.plotting.change_plotting_library('plotly')
    
    X = xx[indMinMax[0]:indMinMax[1]].reshape(len(xx[indMinMax[0]:indMinMax[1]]), 1)
    Y = yy[indMinMax[0]:indMinMax[1]].reshape(len(yy[indMinMax[0]:indMinMax[1]]), 1)
    if err is not None:
        E = err[indMinMax[0]:indMinMax[1]].reshape(len(err[indMinMax[0]:indMinMax[1]]), 1)
    else:
        E = 1.

    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.5)
    
    m = GPy.models.GPRegression(X, Y, kernel)
    # m.constrain_positive()
    # m.Gaussian_noise.variance = 1e-6
    
    # fig = m.plot()
    # 
    # fig[0].show()
    # 
    # pdb.set_trace()

    # m.optimize(messages=True)
    m.optimize(messages=False)
    
    # figOpt = m.plot()
    # 
    # figOpt[0].show()
    
    # GPy.plotting.show(fig) #, filename='basic_gp_regression_notebook')
    
    # pdb.set_trace()
    
    m.optimize_restarts(num_restarts=3, verbose=False)
    
    xRange_fine = np.linspace(X.min(), X.max(), 10*len(X))
    xRange_fine_shaped = xRange_fine.reshape(-1,1)
    medians = (m.predict_quantiles(xRange_fine.reshape(xRange_fine.shape[0], 1), quantiles=[50.]))[0].flatten()

    # Get posterior samples
    posteriorYs = m.posterior_samples_f(xRange_fine_shaped, full_cov=True, size=1000)
    samples = posteriorYs[:,0]
    simY, simMse = m.predict(xRange_fine_shaped)
    # Residuals from best-fit model.
    simY_orig, simMse_orig = m.predict(X)
    res_bf = (Y - simY_orig).flatten()
    res_std = np.std(res_bf)

    snr_bf_peak = np.nanmax(simY)/res_std
    print(f"SNR of best-fit model peak = {snr_bf_peak:.2f}")

    # conf_up3sig = (m.predict_quantiles(xRange_fine.reshape(10*(X.max() - X.min()), 1), quantiles=[99.7]))[0].flatten()
    # conf_lo3sig = (m.predict_quantiles(xRange_fine.reshape(10*(X.max() - X.min()), 1), quantiles=[0.3]))[0].flatten()
    # conf_up3sig = (m.predict_quantiles(xRange_fine.reshape(xRange_fine.shape[0], 1), quantiles=[97.5]))[0].flatten()
    # Calculate 1 and 3 sigma confidence lower limits on x
    conf_lo1sig = (m.predict_quantiles(xRange_fine.reshape(xRange_fine.shape[0], 1), quantiles=[50-34.13]))[0].flatten()
    conf_lo2sig = (m.predict_quantiles(xRange_fine.reshape(xRange_fine.shape[0], 1), quantiles=[50-47.72]))[0].flatten()
    conf_lo3sig = (m.predict_quantiles(xRange_fine.reshape(xRange_fine.shape[0], 1), quantiles=[50-49.87]))[0].flatten()

    # Calculate the uncertainty in the peak X position based on the spread in
    # model function Y values within some N sigma of the max likelihood X position.
    whPeak = np.argmax(medians)
    xPeak = xRange_fine[whPeak]
    # yPeak = medians[whPeak]
    peak_lo3sig = conf_lo3sig[whPeak]
    xAbove_lo3sig = xRange_fine[np.where(medians > peak_lo3sig)]
    xErr_lo = np.nanmin(xAbove_lo3sig)
    xErr_up = np.nanmax(xAbove_lo3sig)

    # print(xErr_lo, xPeak, xErr_up)

# OFF!!
    # # Alternative method to estimate uncertainty on peak X position, using the
    # # spread in the peak's X position among all GP samples returned.
    # # BUT this seems to produce unbelievable low uncertainties in some (many?)
    # # fits.
    # # Get peaks of all the samples
    # xPeakSamples = []
    # for ii in range(samples.shape[1]):
    #     whPeak = np.argmax(samples[:, ii])
    #     xPeakSamples.append(xRange_fine[whPeak])
    
    # std_xPeakSamples = np.std(xPeakSamples)
    # print("Sample peak sigma = ", std_xPeakSamples)
    
    # xErr_lo = std_xPeakSamples
    # xErr_up = std_xPeakSamples
    
    # figOpt = m.plot()
    # # figOpt = m.plot_latent()
    # figOpt[0].add_vline(x=xPeak, line_width=2, line_color="magenta")
    # figOpt[0].add_vline(x=xPeak - xErr_lo, line_width=1, line_color="magenta", line_dash='dash')
    # figOpt[0].add_vline(x=xPeak + xErr_up, line_width=1, line_color="magenta", line_dash='dash')
    # figOpt[0].show()
    
#     plt.figure(40)
#     plt.clf()
#     for ii in range(samples.shape[1]):
#         plt.plot(xRange_fine, samples[:, ii], alpha=0.5)
#     plt.plot(X, Y, 'kx')
#     plt.plot(xRange_fine, simY[:,0], 'k-')
# #    plt.plot(np.arange(X.min(), X.max(), 0.1), medians)
#     plt.draw()
    
#     plt.figure(41)
#     plt.clf()
#     plt.plot(X, res_bf, 'kx')
#     plt.draw()
    
    # if (snr_bf_peak < 3) or (snr_bf_peak > 1e3):
        
    #     plt.figure(40)
    #     plt.clf()
    #     for ii in range(samples.shape[1]):
    #         plt.plot(xRange_fine, samples[:, ii], alpha=0.5)
    #     plt.plot(X, Y, 'kx')
    #     plt.plot(xRange_fine, simY[:,0], 'k-')
    # #    plt.plot(np.arange(X.min(), X.max(), 0.1), medians)
    #     plt.draw()
        
    #     plt.figure(41)
    #     plt.clf()
    #     plt.plot(X, res_bf, 'kx')
    #     plt.draw()
        
    #     pdb.set_trace()

    if snr_bf_peak < 3:
        xPeak = np.nan
        print(f"Low SNR {snr_bf_peak:.2f} measurement set to NaN")

    return xPeak, xErr_lo, xErr_up, snr_bf_peak


def measure_fwhm(yy, xx):
    
    peak = np.nanmax(yy)
    range_fwhm = xx[yy >= peak/2]
    
    fwhm = np.nanmax(range_fwhm) - np.nanmin(range_fwhm)
    
    return fwhm


def measure_radial_profile_fits(fp, pa=None, height=None, rMax=250.,
                                yRange=None, mode='peak', output=False,
                                suffix=''):
    """
    fp: str, relative filepath to the FITS file containing image to measure from
    pa: float, position angle counterclockwise from +y axis in [deg] along which
        to measure the radial profile. Also measures at 180 deg from that angle.
    height: float, diameter in [pixels] of the azimuthal window over which to 
        measure the profile value at each radius, used to define paHW if no
        paHW value is given as input.
    rMax: float, maximum radius in [pixels] to which profile gets measured.
    yRange: list of ymin and ymax values for the Y axis.
    """
    
    hdu = fits.open(os.path.expanduser(fp))
    data = hdu[0].data
    hdr = hdu[0].header
    
    star = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])
    radii = make_radii(data, star)
    phi = make_phi(data, star)
    
    # Convert input PA (deg) to PA in phi coordinate system (radians).
    if pa is None:
        info, infoPath = load_info_json(os.path.split(fp)[0] + '/../')
        pa = info.get('diskPA_deg')
        if pa is None:
            print("FAILED: no disk PA found in info.json or given as input")
            return
    
    # Actually measure the radial profiles.
    rads, prof, profOpp, paPeak, paOppPeak = measure_radial_profile(data, star,
                                                        pa=pa, height=height,
                                                        mode=mode, rMax=rMax,
                                                        plot=False)
    
    paPeakNorm = paPeak - np.nanmean(paPeak)
    paOppPeakNorm = paOppPeak - np.nanmean(paOppPeak)
    
    rads_wmean, jnk = weighted_mean_1d(rads, np.ones(paPeakNorm.shape), n=10)
    paPeakNorm_wmean, jnk = weighted_mean_1d(paPeakNorm, np.ones(paPeakNorm.shape), n=10)
    paOppPeakNorm_wmean, jnk = weighted_mean_1d(paOppPeakNorm, np.ones(paOppPeakNorm.shape), n=10)
    
    # Output a table of the profiles.
    if output:
        tab = table.Table([rads, prof, profOpp, len(prof)*[pa], len(profOpp)*[pa+180]],
            names=['radius', 'intensity', 'intensityOpp', 'paProf', 'paProfOpp'])
        tab.write(os.path.split(fp)[0] + f'/radprofs_{mode}{height}_pa{pa}{suffix}.csv', format='csv')
    
    fs = 14 # fontsize
    
    plt.figure(2, figsize=(7,5))
    plt.clf()
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9)
    ax.errorbar(rads, prof, marker='.', linestyle='None', label='E')
    ax.errorbar(rads, profOpp, marker='.', linestyle='None', label='W')
    if yRange is not None:
        ax.set_ylim(yRange[0], yRange[1])
    # ax.set_xlim(10, )
    # ax.set_yscale('log')
    ax.set_xscale('log', basex=10)
    # Add top X axis in arcsec.
    axy = ax.twiny()
    axy.set_xscale('log', basex=10)
    axy.set_xlim(np.array(ax.get_xlim())*pscale_stis)
    # axy.set_xticks(np.arange(0., 4., 0.5)/pscale_stis)
    # axy.set_xticklabels(np.arange(0., 4., 0.5), fontsize=fs)
    axy.set_xlabel('Radius (arcsec)', fontsize=fs)
    ax.legend(numpoints=1, fontsize=fs-2, handletextpad=0.2)
    ax.set_ylabel(f'{mode} intensity', fontsize=fs)
    ax.set_xlabel('Radius (pixels)', fontsize=fs)
    plt.title(hdr.get('TARGNAME'), fontsize=fs, pad=8)
    for aa in [ax, axy]:
        aa.tick_params(axis='both', which='both', direction='in', labelsize=fs)
    plt.draw()
    
    try:
        plt.figure(3, figsize=(7,5))
        plt.clf()
        ax = plt.subplot(111)
        plt.subplots_adjust(left=0.13, bottom=0.14, right=0.98, top=0.93)
        ax.errorbar(rads, paPeak - np.nanmean(paPeak), marker='.', linestyle='None', label='E')
        ax.errorbar(rads, -1*(paOppPeak - np.nanmean(paOppPeak)), marker='.', linestyle='None', label='W')
        ax.errorbar(rads_wmean, paPeakNorm_wmean, marker='s', linestyle='-', color='b', markeredgecolor='b', markerfacecolor='None')
        ax.errorbar(rads_wmean, -1*paOppPeakNorm_wmean, marker='s', linestyle='-', color='r', markeredgecolor='r', markerfacecolor='None')
        ax.legend(numpoints=1, fontsize=fs-2)
        ax.set_ylabel('PA (degrees)', fontsize=fs)
        ax.set_xlabel('Radius (pixels)', fontsize=fs)
        plt.title(hdr.get('TARGNAME'), fontsize=fs)
        plt.draw()
    except:
        print("HELP!! Could not plot PA positions")
    
    pdb.set_trace()
    
    return


def measure_radial_profile(data, star, pa, mode='peak', rMax=250,
                           paHW=None, height=None, plot=True, expandHW_r=False,
                           expandHW=None):
    """
    data: array of image to be measured.
    star: y,x array of pixel coordinates for center of radial profile.
    pa: position angle counterclockwise from +y axis in [deg].
    mode: 'peak', 'mean', or 'median' for the azimuthal flux measurement.
    rMax: float, maximum radius in [pixels] to which profile gets measured.
    paHW: half-width of PA wedge to include on either side of pa [deg].
    height: float, diameter in [pixels] of the azimuthal window over which to 
        measure the profile value at each radius, used to define paHW if no
        paHW value is given as input.
    """
    
    radii = make_radii(data, star)
    phi = make_phi(data, star)
    
    paPhi = np.radians(pa + 90.)
    if paPhi > 2*np.pi:
        paPhi -= 2*np.pi
    paPhiOpp = paPhi + np.pi
    if paPhiOpp > 2*np.pi:
        paPhiOpp -= 2*np.pi
    
    if paHW is not None:
        paHW = np.radians(paHW)
    
    prof = []
    profOpp = []
    paPeak = []
    paOppPeak = []
    rads = np.arange(3, rMax, 1.)
    
    if paHW is None:
        paHW_by_rad = True
    else:
        paHW_by_rad = False
    
    for rad in rads:
        if paHW_by_rad:
            if height is not None:
                paHW = np.arctan(height/rad)
            else:
                paHW = np.radians(10.)
        else:
            if expandHW_r:
                if rad >= expandHW_r:
                    paHW = expandHW
        radCond = (radii < rad + 0.5) & (radii > rad - 0.5)
        # Handle PA wrapping cases around 0/2pi.
        phiMin = paPhi - paHW
        phiMax = paPhi + paHW
        if phiMin < 0:
            phiMin += 2*np.pi
        if phiMax > 2*np.pi:
            phiMax -= 2*np.pi
        if phiMin > paPhi:
            phiCondMin = (phi > phiMin) | (phi <= paPhi)
        else:
            phiCondMin = (phi > phiMin) & (phi <= paPhi)
        if phiMax < paPhi:
            phiCondMax = (phi < phiMax) | (phi >= paPhi)
        else:
            phiCondMax = (phi < phiMax) & (phi >= paPhi)
        phiCond = phiCondMin + phiCondMax

        try:
            if mode == 'peak':
                prof.append(np.nanmax(data[radCond & phiCond]))
            elif mode == 'mean':
                prof.append(np.nanmean(data[radCond & phiCond]))
            elif mode == 'median':
                prof.append(np.nanmedian(data[radCond & phiCond]))
        except:
            prof.append(np.nan)
        try:
            if np.isnan(prof[-1]):
                paPeak.append(np.nan)
            else:
                paPeak.append(phi[(data == prof[-1]) & radCond & phiCond][0])
        except:
            paPeak.append(np.nan)
        try:
            phiCondOpp = (phi < paPhiOpp + paHW) & (phi > paPhiOpp - paHW)
            try:
                if mode == 'peak':
                    profOpp.append(np.nanmax(data[radCond & phiCondOpp]))
                elif mode == 'mean':
                    profOpp.append(np.nanmean(data[radCond & phiCondOpp]))
                elif mode == 'median':
                    profOpp.append(np.nanmedian(data[radCond & phiCondOpp]))
            except:
                profOpp.append(np.nan)
            if np.isnan(profOpp[-1]):
                paOppPeak.append(np.nan)
            else:
                paOppPeak.append(phi[(data == profOpp[-1]) & radCond & phiCondOpp][0])
        except:
            continue
    
    paPeak = np.degrees(paPeak)
    paOppPeak = np.degrees(paOppPeak)
    
    try:
        paPeakNorm = paPeak - np.nanmean(paPeak)
    except:
        paPeakNorm = -1*np.ones(paPeak.shape)
    try:
        paOppPeakNorm = paOppPeak - np.nanmean(paOppPeak)
    except:
        paOppPeakNorm = -1*np.ones(paOppPeak.shape)
    
    rads_wmean, jnk = weighted_mean_1d(rads, np.ones(paPeakNorm.shape), n=10)
    paPeakNorm_wmean, jnk = weighted_mean_1d(paPeakNorm, np.ones(paPeakNorm.shape), n=10)
    paOppPeakNorm_wmean, jnk = weighted_mean_1d(paOppPeakNorm, np.ones(paOppPeakNorm.shape), n=10)
    
    if plot:
        plt.figure(2, figsize=(8,5))
        # plt.clf()
        ax = plt.subplot(111)
        plt.subplots_adjust(left=0.13, bottom=0.14, right=0.98, top=0.93)
        ax.errorbar(rads, prof, marker='.', linestyle='None', label='E')
        ax.errorbar(rads, profOpp, marker='.', linestyle='None', label='W')
        ax.set_yscale('log')
        plt.legend(numpoints=1)
        ax.set_ylabel('Peak intensity')
        ax.set_xlabel('Radius (pixels)')
        # plt.title(hdr.get('TARGNAME'), fontsize=14)
        plt.draw()
        
        # plt.figure(3, figsize=(8,5))
        # plt.clf()
        # ax = plt.subplot(111)
        # plt.subplots_adjust(left=0.13, bottom=0.14, right=0.98, top=0.93)
        # ax.errorbar(rads, paPeak - np.nanmean(paPeak), marker='.', linestyle='None', label='E')
        # ax.errorbar(rads, -1*(paOppPeak - np.nanmean(paOppPeak)), marker='.', linestyle='None', label='W')
        # ax.errorbar(rads_wmean, paPeakNorm_wmean, marker='s', linestyle='-', color='b', markeredgecolor='b', markerfacecolor='None')
        # ax.errorbar(rads_wmean, -1*paOppPeakNorm_wmean, marker='s', linestyle='-', color='r', markeredgecolor='r', markerfacecolor='None')
        # plt.legend(numpoints=1)
        # ax.set_ylabel('PA (degrees)')
        # ax.set_xlabel('Radius (pixels)')
        # # plt.title(hdr.get('TARGNAME'), fontsize=14)
        # plt.draw()
    
    return rads, prof, profOpp, paPeak, paOppPeak


def plot_radprofs(tablePaths, yRange=None):
    
    tabs = []
    for ii in tablePaths:
        tab = ascii.read(ii, format='csv')
        tabs.append(tab)
    
    fs = 14 # fontsize
    
    plt.figure(12, figsize=(7,5))
    plt.clf()
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9)
    for ii in tabs:
        ax.errorbar(ii['radius'], ii['intensity'], marker='.', linestyle='None', label='PA={} deg'.format((ii['paProf'][0])))
        ax.errorbar(ii['radius'], ii['intensityOpp'], marker='.', linestyle='None', label='PA={} deg'.format((ii['paProfOpp'][0])))
    if yRange is not None:
        ax.set_ylim(yRange[0], yRange[1])
    # ax.set_xlim(18, )
    if np.any(ii['intensity']) | np.any(ii['intensityOpp']):
        ax.set_yscale('linear')
    else:
        ax.set_yscale('log')
    ax.set_xscale('log')
    # Add top X axis in arcsec.
    axy = ax.twiny()
    axy.set_xticks(np.arange(0., 4., 0.5)/pscale_stis)
    axy.set_xticklabels(np.arange(0., 4., 0.5), fontsize=fs)
    axy.set_xscale('log')
    axy.set_xlabel('Radius (arcsec)', fontsize=fs)
    ax.legend(numpoints=1, fontsize=fs-2, handletextpad=0.2)
    ax.set_ylabel('Intensity', fontsize=fs)
    ax.set_xlabel('Radius (pixels)', fontsize=fs)
    # plt.title(hdr.get('TARGNAME'), fontsize=fs, pad=8)
    for aa in [ax, axy]:
        aa.tick_params(axis='both', which='both', direction='in', labelsize=fs)
    plt.draw()
    
    plt.figure(13, figsize=(7,5))
    plt.clf()
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9)
    allRads = []
    allProfs = []
    allPAs = []
    for ii in tabs:
        allRads.append(ii['radius'])
        allRads.append(ii['radius'])
        allProfs.append(ii['intensity'])
        allProfs.append(ii['intensityOpp'])
        allPAs.append(ii['paProf'][0])
        allPAs.append(ii['paProfOpp'][0])
    allRads = np.array(allRads)
    allProfs = np.array(allProfs)
    allPAs = np.array(allPAs)
    markers = ['o', 's', '^']
    for ii, pr in enumerate(allProfs):
        for jj in range(ii+1, len(allProfs)):
            ax.errorbar(allRads[ii], pr - allProfs[jj], marker=markers[ii], markersize=4, alpha=0.8,
                        linestyle='None', label='{} - {}'.format(allPAs[ii], allPAs[jj]))
    # if yRange is not None:
    #     ax.set_ylim(yRange[0], yRange[1])
    # ax.set_xlim(18, )
    # if np.any(ii['intensity']) | np.any(ii['intensityOpp']):
    #     ax.set_yscale('linear')
    # else:
    #     ax.set_yscale('log')
    ax.set_xscale('log')
    # Add top X axis in arcsec.
    axy = ax.twiny()
    axy.set_xticks(np.arange(0., 4., 0.5)/pscale_stis)
    axy.set_xticklabels(np.arange(0., 4., 0.5), fontsize=fs)
    axy.set_xscale('log')
    axy.set_xlabel('Radius (arcsec)', fontsize=fs)
    ax.legend(numpoints=1, fontsize=fs-2, handletextpad=0.2)
    ax.set_ylabel('Intensity Difference', fontsize=fs)
    ax.set_xlabel('Radius (pixels)', fontsize=fs)
    # plt.title(hdr.get('TARGNAME'), fontsize=fs, pad=8)
    for aa in [ax, axy]:
        aa.tick_params(axis='both', which='both', direction='in', labelsize=fs)
    plt.draw()
    
    pdb.set_trace()
    return


def plot_highpass_filter(filepath, highpassSize=30., gaussian=False, save=False):
    
    hdu = fits.open(filepath)
        
    if save:
        inputName = os.path.split(filepath)[-1]
        if gaussian:
            savepath = os.path.join(os.path.split(filepath)[0], inputName.split('.fits')[0] + '_highpass_gaussian_{}pix.fits'.format(highpassSize))
        else:
            savepath = os.path.join(os.path.split(filepath)[0], inputName.split('.fits')[0] + '_highpass_median_{}pix.fits'.format(highpassSize))
    else:
        savepath = False
    
    highpass_img = unsharp(None, hdu, fl=None, B=highpassSize, ident='_999', output=True,
                        silent=True, save=savepath, parOK=True, gauss=gaussian)
    
    # vmin = 0
    # vmax = np.percentile(highpass_img[~np.isnan(highpass_img)], 99.99)
    # 
    # fig = plt.figure(71)
    # fig.clf()
    # ax = plt.subplot(111)
    # ax.imshow(highpass_img,
    #           norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=vmin, vmax=vmax))
    # plt.draw()
    
    return


# Multiplicative factor when in units of mJy arcsec^-2.
# Optimize the colorbar for HD 114082 and scale all others around it.
rescale_Qr_fluxcal = {'AK Sco':0.05, 'AU Mic':6, 'CE Ant':10, 'GSC 07396':0,
                      'HD 30447':12, 'HD 32297':1, 'HD 35841':10, 'HD 61005':6,
                      'HD 100546':0.1, 'HD 106906':2, 'HD 110058':1,
                      'HD 111161':12, 'HD 111520':10, 'HD 114082':1, 'HD 115600':2, 
                      'HD 117214':1, 'HD 129590':1, 'HD 131835':5, 'HD 143675':3,
                      'HD 141569':3, 'HD 145560':5, 'HD 146897':1, 'HD 156623':5, 
                      'HD 157587':1, 'HD 191089':5,}

# Flux conversion factors for GPIES disks in rstokes Q, exactly matching those
# used in Esposito et al. 2020. Units of these values are Jy/(ADU/s).
gpi_Qr_fconv_Jy_ADU_s = {'HD 15115': 5.90E-07,
                        'HD 30447': 6.22E-07,
                        'HD 32297': 6.42E-07,
                        'HD 35841': 6.98E-07,
                        'Beta Pic': 5.01E-07,
                        'HD 61005': 5.11E-07,
                        'HD 100546': 6.61E-07,
                        'HD 106906': 6.56E-07,
                        'HR 4796': 7.14E-07,
                        'HD 110058': 8.29E-07,
                        'HD 111520': 7.72E-07,
                        'HD 114082': 8.65E-07,
                        'HD 115600': 6.96E-07,
                        'HD 117214': 6.36E-07,
                        'HD 131835': 8.12E-07,
                        'HD 141569': 5.82E-07,
                        'NZ Lup': 6.78E-07,
                        'HD 143675': 7.86E-07,
                        'HD 145560': 5.95E-07,
                        'HD 146897': 6.99E-07,
                        'HD 157587': 6.90E-07,
                        'HR 7012': 5.46E-07,
                        'HD 191089': 6.14E-07,
                        'AU Mic': 6.07E-07,
                        'CE Ant': 6.84E-07,
                        'HD 111161': 5.99E-07,
                        'HD 129590': 5.24E-07,
                        'AK Sco': 6.54E-07,
                        'HD 156623': 5.69E-07}