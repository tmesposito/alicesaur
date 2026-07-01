#!/usr/bin/env python

import numpy as np

# Internal imports
from ..pipeline import Pipeline


class PipelineSTIS(Pipeline):
    """
    Main reduction pipeline class for STIS images.
    """

    def __init__(self, **kwargs):

        self.instrument = 'stis'
        # STIS image plate scale.
        self.pscale = 0.05075 # [arcsec/pixel] Nguyen et al. 2021
        # Image dimensions.
        self.imgShape = np.array([1024, 1024]) # [Y,X pixels]

        super().__init__(**kwargs)

