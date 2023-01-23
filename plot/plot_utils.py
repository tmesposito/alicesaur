#!/usr/bin/env python
#
# Tom Esposito

import os
import sys
import pdb
import copy
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from astropy.io import ascii, fits
from astropy import table
from astropy import wcs
from scipy.ndimage import gaussian_filter
from matplotlib.colors import SymLogNorm, LogNorm
from hst_process.utils import make_radii


#---- DEFINE CONSTANTS ----#
pscale_stis = 0.0507 # [arcsec/pixel]


# Following functions copied from plotter.py for convenience.
def easy_colorbar(im, ax, fig, Vmin, Vmax, step, widfrac=0.05, label='', labelPad=5,
                  fontSize=18, cticks=None, orientation='vertical', spine_color=None,
                  output=False, side='right'):
    """
    Easily make a colorbar for an imshow instance. Limited to simple axes.
    
    Input:
        im= imshow instance needing colorbar.
        ax= axes instance for im.
        fig= figure instance containing ax.
        Vmin
        Vmax
        step= size of value step.
        widfrac= colorbar width as fraction of axis size.
        label
        labelPad
        fontSize
        orientation= 'vertical' or 'horizontal'
        output: if True, return the colorbar axis object.
        side: 'right', 'top', '
    
    Output:
        cb= colorbar instance
    """
    plt.draw()
    
    # Axes position [top, left, bottom, right].
    axPos = ax.get_position().get_points().astype(float).reshape(4)
    caxPos = axPos.copy()

    if (orientation=='vertical') and (side=='right'):
        he_cbar = axPos[3] - axPos[1]
        wi_cbar = widfrac*(axPos[2] - axPos[0])
        caxPos[0] = axPos[2]
        caxPos[1] = axPos[1]
        caxPos[2] = wi_cbar
        caxPos[3] = he_cbar
    elif (orientation=='vertical') and (side=='left'):
        he_cbar = axPos[3] - axPos[1]
        wi_cbar = widfrac*(axPos[2] - axPos[0])
        caxPos[0] = axPos[0] - wi_cbar
        caxPos[1] = axPos[1]
        caxPos[2] = wi_cbar
        caxPos[3] = he_cbar
    elif (orientation=='horizontal') and (side=='top'):
        he_cbar = widfrac*(axPos[3] - axPos[1])
        wi_cbar = axPos[2] - axPos[0]
        caxPos[0] = axPos[0]
        caxPos[1] = axPos[3]
        caxPos[2] = wi_cbar
        caxPos[3] = he_cbar
    else:
        he_cbar = widfrac*(axPos[3] - axPos[1])
        wi_cbar = axPos[2] - axPos[0]
        caxPos[0] = axPos[0]
        caxPos[1] = axPos[1] - he_cbar
        caxPos[2] = wi_cbar
        caxPos[3] = he_cbar

    caxPos = caxPos.tolist()
    cax = fig.add_axes(caxPos)
    
    if cticks is None:
        cticks = np.arange(Vmin, Vmax + step/2., step)
    cb = plt.colorbar(im, cax=cax, orientation=orientation, ticks=cticks)
    cb.set_label(label, fontsize=fontSize, labelpad=labelPad)
    if orientation == 'vertical':
        for tick in cax.get_yticklabels():
            tick.set_fontsize(fontSize)
    else:
        if side == 'top':
            cax.xaxis.tick_top()
            cax.xaxis.set_label_position('top')
        for tick in cax.get_xticklabels():
            tick.set_fontsize(fontSize)

    
    if spine_color is not None:
        cb.outline.set_edgecolor(spine_color)
    
    # Make colorbar gradients save properly as eps, pdf.
    cb.solids.set_rasterized(True)

    plt.draw()

    if output:
        return cax
