#!/usr/bin/env python

import os
import sys
import pdb
import getpass
import numpy as np
from glob import glob
from tqdm import tqdm
from astropy.io import ascii, fits
from astropy import table
from astropy import wcs
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Internal imports
from ..pipeline import Pipeline
from alicesaur.psfsub import stis_psfsub
from alicesaur import utils
from alicesaur.calibration.bad_pix import fix_bad_pix
from alicesaur.calibration.align import find_star_radon, shift_pix_to_pix
from alicesaur.calibration.flux import convert_intensity
from alicesaur.improcess.mask import mask_exclusions, mask_spikes_offaxis


class PipelineSTIS(Pipeline):
    """
    Main reduction pipeline class for STIS images.
    """

    # Image pixel scale for STIS imager.
    pscale = 0.0507 # [arcsec/pixel]
