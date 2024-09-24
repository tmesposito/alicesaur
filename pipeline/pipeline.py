#!/usr/bin/env python

import os
import platform
import sys
import shutil
import pdb
import logging
import getpass
import json
import numpy as np
from datetime import datetime
from glob import glob
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from astropy.io import ascii, fits
from astropy import table
from astropy import wcs
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation

# Internal imports
from alicesaur import pipeline
from alicesaur.psfsub import stis_psfsub
from alicesaur import utils
from alicesaur.calibration.cti import CTI
from alicesaur.calibration.bad_pix import fix_bad_dq_knn, mask_bad_pix
from alicesaur.calibration.distortion import correct_distortion
from alicesaur.calibration.align import find_star_radon, shift_pix_to_pix
from alicesaur.calibration.flux import convert_intensity
from alicesaur.improcess.mask import mask_exclusions, mask_spikes_offaxis
from alicesaur.improcess.manipulate import zero_pad
from alicesaur.gaia.astrometry import main
from alicesaur.gaia.gaia_utils import get_gaia_id
from alicesaur.plot.disk_plot import plot_radprof_1d


# Set matplotlib backend based on OS.
if platform.system() == 'Darwin':
    matplotlib.use('MACOSX')
# Set matplotlib imshow origin to bottom left.
matplotlib.rcParams["image.origin"] = "lower"


class Pipeline(object):
    """
    Base reduction pipeline class for Hubble Space Telescope image processing.
    """

    debug = False
    saveFinal = False
    saveAuxiliary = True
    # Locate the path to the alicesaur package. Prioritize the user environment
    # variable ALICESAUR_HOME over retrieving the path from imported modules.
    if os.environ.get('ALICESAUR_HOME') not in [None, '']:
        packageHome = os.path.abspath(os.environ.get('ALICESAUR_HOME'))
    else:
        packageHome = os.path.dirname(pipeline.__path__[0])

    # Default to ingesting flt files as input.
    inputType = 'flt'
    # Default to STIS instrument.
    instrument = 'stis'
    # Image plate scale.
    pscale = 0.0507 # [arcsec/pixel]
    # Image dimensions.
    imgShape = np.array([1024, 1024]) # [Y,X pixels]
    # Minimum PA rotation to allow reference images in ADI PSF subtractions.
    # None allows all reference images.
    deltaPAMin = None
    # Custom ID for this pipeline.
    cid = ''

    def __init__(self, **kwargs):

        self.pipelineStartTime = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')

        self.pids = []

        # Define attributes with kwargs items
        for key, val in kwargs.items():
            setattr(self, key, val)

        if hasattr(self, 'dataDir'):
            if self.dataDir is None:
                # Invent a data directory from the target name,
                # pid (if possible), instrument, and obsMode.
                if len(self.pids) > 0:
                    self.dataDir = os.path.join(os.path.realpath('.'),
                                    f'{self.targ}_{self.instrument.lower()}_{self.pids[0]}',
                                    self.obsMode.lower())
                else:
                    self.dataDir = os.path.join(os.path.realpath('.'),
                                    f'{self.targ}_{self.instrument.lower()}',
                                    self.obsMode.lower())

            # Ensure trailing slash in data directory name.
            self.dataDir = os.path.join(os.path.expanduser(self.dataDir),'')
        else:
            self.dataDir = None

        # Create the data directory if it doesn't exist yet.
        try:
            utils.check_mkdir(self.dataDir)
        except Exception as ee:
            print(ee)
            sys.exit(0)

        # Set up the logger for this pipeline.
        self.set_up_logger(startTime=self.pipelineStartTime, level='INFO')
        self.logger.info(f"Using data directory {self.dataDir}")

        # Adjust custom identifier string to add underscore.
        if self.cid != '':
            if self.cid[0] not in ['_', '-']:
                self.cid = '_' + self.cid
                self.logger.warning("Custom identifier modified to "\
                                    f"{self.cid} to conform to file naming "\
                                    "convention.\n")

        # Path to instrument- and filetype-specific occulter mask FITS.
        if self.instrument == 'stis':
# FIX ME!!! Tailor occulter masks to flt and sx2 files separately???
            self.occultMaskPath = os.path.join(self.packageHome, 'masks',
                                    'mask_stis_occulters_sx2_bar_wedgeB.fits')
        else:
            self.occultMaskPath = ''

        # Load dataset info and reduction parameters from info.json.
        self.load_info_json(self.dataDir)

        self.alignStar = None


    def set_up_logger(self, startTime=None, level='INFO'):
        """
        Create a logging object.

        Parameters
        ----------
        startTime : TYPE, optional
            DESCRIPTION. The default is None.
        level : TYPE, optional
            DESCRIPTION. The default is 'INFO'.

        Returns
        -------
        None

        """
        if startTime is None:
            startTime = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')

        # Set up a logger.
        logPath = os.path.join(self.dataDir, 'logs',
                      f'alicesaur_{self.targ}_{self.obsMode}_{startTime}.log')

        formatter = logging.Formatter('%(asctime)s :: %(levelname)s: %(message)s',
                                      '%H:%M:%S')
        logger = logging.getLogger('pipeline_logger')
        logger.setLevel(logging.getLevelName(level))

        # Create a stream handler to print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.getLevelName(level))
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)

        # Create a file handler to write logs to a file
        try:
            # Create the log directory if it doesn't exist yet.
            utils.check_mkdir(os.path.join(self.dataDir, 'logs'))

            file_handler = logging.FileHandler(logPath)
            file_handler.setLevel(logging.getLevelName(level))
            file_handler.setFormatter(formatter)
            # Add the file handler to the logger
            logger.addHandler(file_handler)
        except Exception as ee:
            logger.error(ee)
            logger.error(f"*** FAILED to write log file to {logPath}. Will only"\
                         " log to stdout\n")

        # logging.basicConfig(filename=logPath, filemode='a',
        #                     format='%(asctime)s:%(levelname)s: %(message)s',
        #                     datefmt='%Y-%m-%d %H:%M:%S',
        #                     level=logging.getLevelName(level))
        # self.logger = logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.logger = logger
        self.logger.info(f'Started alicesaur pipeline at UTC {startTime}')

        return


    def load_info_json(self, infoDir):
        """
        Load dataset info and reduction parameters from an info.json file.
        """
        try:
            infos = glob(os.path.join(infoDir, "info.json"))
            if len(infos) < 1:
                infos = glob(os.path.join(infoDir, "../info.json"))
                if len(infos) < 1:
                    self.logger.warning("No info.json found. Using default values.\n")
            with open(infos[0]) as ff:
                self.info = json.load(ff)
            self.infoPath = os.path.normpath(infos[0])
            self.logger.info("Loaded info from {}".format(self.infoPath))
        except:
            self.info = {}
            self.infoPath = ''
            self.logger.warning(f"*** FAILED to load info.json from directory {infoDir}. "
                  + "Filling it with default values.\n")
            self.logger.exception("info.json exception:")

        # Set info dict defaults if overriding values were not given in the
        # json file.
        self.info.setdefault('targetName', '')
        self.info.setdefault('psfRefName', '')
        self.info.setdefault('obsLogPath', '')
        self.info.setdefault('diskPA_deg', 0.)
        self.info.setdefault(self.obsMode.lower(), {})
        # Background sampling region centers for science and reference images.
        self.info[self.obsMode].setdefault('bgCen_yx', None)
        self.info[self.obsMode].setdefault('bgCenRef_yx', None)
        # if not (np.all(bgCenRef is None) | (bgCenRef == '')):
        #     bgCenRef = np.array(bgCenRef.split(' '), dtype=float)
        # else:
        #     bgCenRef = bgCen
        self.info[self.obsMode].setdefault('bgCenFinal_yx', None)
        # Background sampling radius [pix]
        self.info[self.obsMode].setdefault('bgRadius', 40)
        # Diffraction spike mask width [pix]
        self.info[self.obsMode].setdefault('spWidth', 5)
        self.info[self.obsMode].setdefault('radProfSub',
                                           {"rMax": 200,
                                            "postCombine": True})
        # Radon transform inner working angle [pix]
        if self.obsMode.lower() in ['wedgeb1.8']:
            self.info[self.obsMode].setdefault('radonIWA', 70)
        else:
            self.info[self.obsMode].setdefault('radonIWA', 30)
        # starToUse is the optional forced star position
        self.starToUse = self.info[self.obsMode].setdefault('starForce_yx',
                                                            None)
        if self.starToUse is not None:
            self.forceStar = True
            self.starToUse = np.array(self.starToUse)
        else:
            self.forceStar = False
         # Masked exclusion region definitions for Science and Reference images
        self.exclusions = self.info[self.obsMode].setdefault('exclude', {})
        self.exclusionsSci = self.exclusions.setdefault('sci', {})
        self.exclusionsRef = self.exclusions.setdefault('ref', {})
        # PSF subtraction inner and outer radii.
        self.exclusionsSci.setdefault('r_in', 30)
        self.exclusionsSci.setdefault('r_out', 130)

        # Set the background attributes.
        # Background sampling center and radius for science images.
        bgCen = self.info[self.obsMode]['bgCen_yx']
        if not (np.all(bgCen is None) | (bgCen == '')):
            self.bgCen = np.array(bgCen.split(' '), dtype=float)
        else:
            self.bgCen = None
        self.bgRadius = self.info[self.obsMode]['bgRadius'] # [pix]
        # Background sampling center for reference images.
        bgCenRef = self.info[self.obsMode]['bgCenRef_yx']
        if not (np.all(bgCenRef is None) | (bgCenRef == '')):
            self.bgCenRef = np.array(bgCenRef.split(' '), dtype=float)
        else:
            self.bgCenRef = None

        self.subFinalRadProf = False
        if self.info[self.obsMode].get('radProfSub') is not None:
            if self.info[self.obsMode]['radProfSub'].get('postCombine') in ['True', True]:
                self.subFinalRadProf = True

        return


    def find_imgs(self, suffix='flt'):
        """
        Find paths to all images with the provided suffix in self.dataDir.

        Assign that list of paths to self.fileList attribute.

        """
        if not os.path.exists(self.dataDir):
            self.logger.error("*** Directory {} does not exist ** !!\n".format(self.dataDir))
            raise OSError

        self.fileList = np.sort(glob(self.dataDir + '*_{}.fits*'.format(suffix)))
        # Force an unlimited string length for this array.
        self.fileList = np.asarray(self.fileList, dtype='O')

        if len(self.fileList) == 0:
            self.logger.warning(f"*** NO FITS files found at {self.dataDir + '*_' + suffix +'.fits'} *** !!\n")

        return


    def load_imgs(self, suffix='flt'):
        """
        List of arrays containing data [0] and headers [1].

        suffix: str suffix of FITS files to summarize.
        """

        imgsHdrs = [[],[]]
        targs = []

        if len(self.fileList) == 0:
            self.logger.error(f"*** NO FITS files found at {self.dataDir + '*_' + suffix +'.fits'} ** !!\n")

        if suffix == 'flt':
            pass
        elif suffix == 'sx2':
            for ii, ff in enumerate(self.fileList):
                with fits.open(ff) as hdu:
                    imgs = [hdu[jj].data for jj in range(len(hdu))]
                    hdrs = [hdu[jj].header for jj in range(len(hdu))]
                    imgsHdrs[0].append(imgs)
                    imgsHdrs[1].append(hdrs)
                    targs.append(hdrs[0]['TARGNAME'].lower())
            self.imgShape = np.array(imgsHdrs[0][0][1].shape)

        return imgsHdrs, np.array(targs)


    def load_flt_imgs(self, file_names, plot_images=False, scienceOnly=False):
        """
        Load and organize extensions of 'flt' FITS files.
        """
        sci_data, err_data, dq_data = [], [], []
        all_headers, sci_headers, err_headers, dq_headers = [], [], [], []
        dq_bool_16 = None
        dq_bool_8192 = None
        target_names = []

        for file_name in file_names:
            crsplits_sci = []
            crsplits_hdrs_sci = []
            crsplits_err = []
            crsplits_hdrs_err = []
            crsplits_dq = []
            crsplits_hdrs_dq = []
            crsplits_hdrs_all = []
            with fits.open(file_name) as hdu_list:
                for ii, hdu in enumerate(hdu_list):

                    # Put all headers into one long list, and then also
                    # split them up by category.
                    crsplits_hdrs_all.append(hdu.header)

                    if 'SCI' in hdu.header.get('EXTNAME', ''):
                        crsplits_sci.append(hdu.data)
                        crsplits_hdrs_sci.append(hdu.header)
                    elif 'ERR' in hdu.header.get('EXTNAME', ''):
                        crsplits_err.append(hdu.data)
                        crsplits_hdrs_err.append(hdu.header)
                    elif 'DQ' in hdu.header.get('EXTNAME', ''):
                        crsplits_dq.append(hdu.data)
                        crsplits_hdrs_dq.append(hdu.header)
                    elif hdu.header.get('EXTNAME', None) is None:
                        base_header = hdu.header.copy()
                        crsplits_hdrs_sci.append(base_header)
                        target_names.append(hdu.header.get('TARGNAME', ''))

                    if plot_images and hdu.data is not None and hdu.is_image:
                        plt.imshow(hdu.data, cmap='gray',
                                   norm=SymLogNorm(vmin=0.01, vmax=100,
                                                   linthresh = 0.01))
                        plt.colorbar()
                        plt.title('HDU: ' + hdu.header.get('EXTNAME', 'N/A'))
                        plt.show()

            sci_data.append(crsplits_sci)
            err_data.append(crsplits_err)
            dq_data.append(crsplits_dq)
            all_headers.append(crsplits_hdrs_all)
            sci_headers.append(crsplits_hdrs_sci)
            err_headers.append(crsplits_hdrs_err)
            dq_headers.append(crsplits_hdrs_dq)

        # Check if all images have the same number of CRSPLITS.
        n_crsplits = np.array([len(ii) for ii in sci_data])
        if len(np.unique(n_crsplits)) > 1:
            self.logger.warning("WARNING: Uneven numbers of CRSPLITS among input images:" \
                                f"{n_crsplits}\n")
            # max_crsplits = max(n_crsplits)
        #     print(f"Inserting empty CRSPLITS to reach {max_crsplits} for all")
        #     for ii, ncr in enumerate(n_crsplits):
        #         if ncr < max_crsplits:
        #             sci_data[ii].append(np.nan*np.ones(sci_data[ii][0].shape))
        #             err_data[ii].append(np.nan*np.ones(err_data[ii][0].shape))
        #             dq_data[ii].append(np.nan*np.ones(dq_data[ii][0].shape))
        #             # for jj in range(3):
        #             #     all_headers.append({'EXTNAME':'EMPTY'})
        #             sci_headers[ii].append({'EXTNAME':'EMPTY'})
        #             err_headers[ii].append({'EXTNAME':'EMPTY'})
        #             dq_headers[ii].append({'EXTNAME':'EMPTY'})
            min_crsplits = min(n_crsplits)
            self.logger.info(f"Trimming extra CRSPLITS to reach {min_crsplits} for all")
            for ii, ncr in enumerate(n_crsplits):
                if ncr > min_crsplits:
                    sci_data[ii] = sci_data[ii][:min_crsplits]
                    err_data[ii] = err_data[ii][:min_crsplits]
                    dq_data[ii] = dq_data[ii][:min_crsplits]
                    all_headers[ii] = all_headers[ii][:3*min_crsplits+1]
                    sci_headers[ii] = sci_headers[ii][:min_crsplits+1]
                    err_headers[ii] = err_headers[ii][:min_crsplits]
                    dq_headers[ii] = dq_headers[ii][:min_crsplits]

        sci_data = np.array(sci_data)
        err_data = np.array(err_data)
        dq_data = np.array(dq_data)

        # Convert DQ values to binary strings to read the individual flags.
        # Only do this if bad pixels are going to be fixed.
        if not self.noFixPix:
            self.logger.info("Reading DQ array binary flags...")
            dq_binary = np.empty(dq_data.shape, dtype='U14')

            for index in np.ndindex(dq_data.shape):
                dq_binary[index] = f"{dq_data[index]:014b}"
            dq_bool_16 = np.zeros(dq_binary.shape, dtype=bool)
            dq_bool_8192 = dq_bool_16.copy()
            for index in np.ndindex(dq_binary.shape):
                # Binary string is read from the right, so -5th element in from the
                # right is the 16 bit. 0th element is the 8192 bit.
                dq_bool_16[index] = dq_binary[index][-5] == '1'
                dq_bool_8192[index] = dq_binary[index][0] == '1'

            self.logger.info(f"\nNumber of CRSPLITS per FITS: {[len(ii) for ii in sci_data]}")

            if dq_data.size > 0:

                n_dq_16 = np.sum(np.sum(dq_bool_16, axis=3), axis=2)
                n_dq_8192 = np.sum(np.sum(dq_bool_8192, axis=3), axis=2)

                self.logger.info(f"\nDQ 16 pixels by FITS (row) and CRSPLIT (column):\n{n_dq_16}")
                self.logger.info(f"\nDQ 8192 pixels by FITS (row) and CRSPLIT (column):\n{n_dq_8192}")

                self.logger.info("DQ flag : total count in dataset")
                for flag in np.unique(dq_binary):
                    self.logger.info(f"{int(flag, 2)} : {np.sum(dq_binary==flag)}")
        else:
            dq_bool_16 = np.zeros(dq_data.shape, dtype=bool)
            dq_bool_8192 = np.zeros(dq_data.shape, dtype=bool)

        if scienceOnly:
            return sci_data, sci_headers, np.array(target_names)
        else:
            return sci_data, err_data, dq_data, dq_bool_16, dq_bool_8192, all_headers, sci_headers, err_headers, dq_headers, np.array(target_names)


    def pixelfixing(self, data, dq_8192_mask=None, dq_masks=[],
                    fix_other=False, verbose=False):
        """
        Fix bad pixels iteratively in all images.
        """

        data_fixed = data.copy()
        # Fix the DQ=8192 pixels separately.
        if dq_8192_mask is not None:
            if data.ndim > 3:
                # Number of 8192 flagged pixels in each CRSPLIT.
                n_dq_8192 = np.sum(np.sum(dq_8192_mask, axis=3), axis=2)
                med_8192 = np.nanmedian(n_dq_8192)
                # std_8192 = np.nanstd(n_dq_8192)
                for ii, da in tqdm(enumerate(data_fixed), desc="Fixing DQ=8192 bad pixels"):
                    for jj in range(da.shape[0]):
                        # Check for unusually high number of pixels
                        # with 8192 DQ flag, which can sometimes be
                        # wrong from the HST pipeline.
                        # Do not fix 8192 pixels for those.
                        if n_dq_8192[ii][jj] > 2*med_8192:
                            self.logger.warning("Too many pixels with DQ=8192 flag "\
                                    f"in image {ii}, crsplit {jj}; not fixing them "\
                                    f"({n_dq_8192[ii][jj]} > {2*med_8192} [2*median])\n")
                            continue
                        data_fixed[ii][jj] = fix_bad_dq_knn(da[jj],
                                                dq_8192_mask[ii][jj], k=5,
                                                max_distance=np.inf,
                                                iterate=True,
                                                verbose=verbose)

        # Then fix all other DQ flags.
        for dqm in dq_masks:
            if data.ndim > 3:
                for ii, da in tqdm(enumerate(data_fixed), desc="Fixing other DQ flagged bad pixels"):
                    for jj in range(da.shape[0]):
                        data_fixed[ii][jj] = fix_bad_dq_knn(da[jj],
                                                dqm[ii][jj], k=9,
                                                max_distance=30,
                                                iterate=True,
                                                verbose=verbose)

        if fix_other:
            if data.ndim > 3:
                for ii, da in tqdm(enumerate(data_fixed), desc="Fixing other (non-DQ) bad pixels"):
                    for jj in range(da.shape[0]):
                        data_fixed[ii][jj] = mask_bad_pix(data_fixed[ii][jj],
                                                    inst=None, Nsig=7,
                                                    neighborDist=1,
                                                    low_only=False,
                                                    iterate=True)

        return data_fixed

    
    def summarize_obs(self, suffix='sx2', dsName=None):
        """
        Summarize FITS header info in a table.

        suffix: str suffix of FITS files to summarize.
        dsName: str optional name of dataset for log filename; defaults to the
            lowest level directory of dataDir.
        """

        if not os.path.exists(self.dataDir):
            self.logger.error("*** Directory {} does not exist ** !!\n".format(self.dataDir))
            raise OSError
    
        fl = np.sort(glob(self.dataDir + '*_{}.fits*'.format(suffix)))
    
        rows = []
        col_names = ['I', 'FILENAME', 'TARGNAME', 'ORIENTAT', 'TDATEOBS', 'TTIMEOBS', 'PROPAPER', 'NCOMBINE',
                     'EXPTIME', 'NDATAARR']
        col_dtypes = ['int32', 'U80', 'U40', 'U40', 'U40', 'U40', 'U40', 'int32',
                      'float', 'int32']
    
        for ii, ff in enumerate(fl):
            hdr0 = fits.getheader(ff, ext=0)
            hdr1 = fits.getheader(ff, ext=1)
            # Get number of data arrays in _flt.fits file.
            try:
                ff_flt = ff.split(suffix)[0] + 'flt.fits'
                hdr_flt = fits.getheader(ff_flt, ext=0)
                nDataDim = hdr_flt['NEXTEND'] + 1
            except:
                nDataDim = -1
            rows.append((ii+1, os.path.split(ff)[-1], hdr0.get('TARGNAME'), hdr1.get('ORIENTAT'), hdr0.get('TDATEOBS'),
                         hdr0.get('TTIMEOBS'), hdr0.get('PROPAPER'),
                         hdr1.get('NCOMBINE'), hdr1.get('EXPTIME'), nDataDim))
    
        sum_table = table.Table(rows=rows, names=col_names, dtype=col_dtypes)
    
        if dsName is None:
            dsName = os.path.split(self.dataDir[:-1])[-1]
        table_name = 'obs_log_{}.csv'.format(dsName)
        sum_table.write(self.dataDir + table_name, format='csv', overwrite=True)
        self.logger.info(f"Wrote observation summary table to {self.dataDir + table_name}")
    
        return

    # def summarize_obs(self, suffix='flt', dsName=None):
    #     """
    #     Summarize FITS header info in a table.

    #     suffix: str suffix of FITS files to summarize.
    #     dsName: str optional name of dataset for log filename; defaults to the
    #         lowest level directory of dataDir.
    #     """

    #     if not os.path.exists(self.dataDir):
    #         print("\n!! **Directory {} does not exist ** !!\n".format(self.dataDir))
    #         raise OSError
    
    #     fl = np.sort(glob(self.dataDir + '*_{}.fits*'.format(suffix)))
    
    #     rows = []
    #     col_names = ['I', 'FILENAME', 'TARGNAME', 'ORIENTAT', 'TDATEOBS', 'TTIMEOBS', 'PROPAPER', 'NCOMBINE',
    #                  'EXPTIME', 'NDATAARR']
    #     col_dtypes = ['int32', 'U80', 'U40', 'U40', 'U40', 'U40', 'U40', 'int32',
    #                   'float', 'int32']
    
    #     for ii, ff in enumerate(fl):
    #         hdr0 = fits.getheader(ff, ext=0)
    #         hdr1 = fits.getheader(ff, ext=1)
    #         # Get number of data arrays in _flt.fits file.
    #         try:
    #             ff_flt = ff.split(suffix)[0] + 'flt.fits'
    #             hdr_flt = fits.getheader(ff_flt, ext=0)
    #             nDataDim = hdr_flt['NEXTEND'] + 1
    #         except:
    #             nDataDim = -1
    #         rows.append((ii+1, os.path.split(ff)[-1], hdr0.get('TARGNAME'), hdr1.get('ORIENTAT'), hdr0.get('TDATEOBS'),
    #                      hdr0.get('TTIMEOBS'), hdr0.get('PROPAPER'),
    #                      hdr1.get('NCOMBINE'), hdr1.get('EXPTIME'), nDataDim))
    
    #     sum_table = table.Table(rows=rows, names=col_names, dtype=col_dtypes)
    
    #     if dsName is None:
    #         dsName = os.path.split(self.dataDir[:-1])[-1]
    #     table_name = 'obs_log_{}.csv'.format(dsName)
    #     sum_table.write(self.dataDir + table_name, format='csv', overwrite=True)
    #     print("Wrote observation summary table to {}".format(self.dataDir + table_name))
    
    #     return
       
    
    def load_obs_log(self, logPath=None):
        """
        """

        if logPath is not None:
            logList = [logPath]
        else:
            logList = glob(self.dataDir + 'obs_log*.csv')
            if len(logList) == 0:
                logList = glob(self.dataDir + '../obs_log*.csv')

        try:
            self.obsTable = ascii.read(os.path.expanduser(logList[0]),
                                       delimiter=",",
                                       header_start=0,
                                       data_start=1,
                                       guess=True)
        except:
            self.obsTable = None

        return self.obsTable


    def combineOrbitImgs(self, imgs, orientats):
        """
        Combine (nanmean) all images from a single orbit for a given target.
        """
        self.sciInds_precombine = self.sciInds.copy()
        self.refInds_precombine = self.refInds.copy()
        self.allHdrs_precombine = self.allHdrs.copy()

        # Maintain the input order of the sci and ref images.
        orientatsUnique, indsUnique = np.unique(orientats, return_index=True)
        self.indsOrbits = indsUnique # index of first image in each orbit
        sameOrder = np.argsort(indsUnique)
        orientatsUnique = orientatsUnique[sameOrder]

        orientatsSci = np.unique(orientats[self.sciInds])
        orientatsRef = np.unique(orientats[self.refInds])

        sciIndsNew = np.array([np.where(orientatsUnique == oo)[0][0] for oo in orientatsSci])
        refIndsNew = np.array([np.where(orientatsUnique == oo)[0][0] for oo in orientatsRef])
        self.sciInds = sciIndsNew
        self.refInds = refIndsNew
        # Reset the header list as well.
# FIX ME!!! Do better to combine headers after the orbit combination.
# Right now, we just take the first header of each orbit.
        self.sciHdrs = [self.allHdrs[ind] for ind in indsUnique[self.sciInds]]
        self.refHdrs = [self.allHdrs[ind] for ind in indsUnique[self.refInds]]
        self.allHdrs = [self.allHdrs[ind] for ind in indsUnique]

        combineImgs = np.zeros((len(orientatsSci) + len(orientatsRef), imgs.shape[1], imgs.shape[2]))

        for ii, orient in enumerate(orientatsSci):
            indsOrbit = np.where(orientats == orient)[0]
            combineImgs[np.where(orientatsUnique == orient)[0]] = np.nanmean(imgs[indsOrbit], axis=0)

        for ii, orient in enumerate(orientatsRef):
            indsOrbit = np.where(orientats == orient)[0]
            combineImgs[np.where(orientatsUnique == orient)[0]] = np.nanmean(imgs[indsOrbit], axis=0)

        return combineImgs, orientatsUnique


    def find_star(self, imgs, mask, starGuess=None):
        """
        Locate the target star position in pixel coordinates. The default is to
        use a Radon transform-based algorithm to locate the center of the
        occulted primary star via its diffraction spikes.

        starGuess: array of guessed (y,x) pixel coordinates for the star in
            the input images. If None, defaults to self.starFromWCS
        """

        # Find star.
        stars = []

        if starGuess is None:
            starGuess = self.starFromWCS

        for ii, im in enumerate(imgs):
            # Mask out the occulters before doing the radon transform.
            imgIter = im # + mask
            imgIter[imgIter < -1e3] = 0.
            # Do a radon transform to find star from diffraction spike pattern.
            # radonIWA = 30 pix generally good for bar10
            # sp_width = 30 pix generally good for bar10
            if self.noRadon or self.forceStar:
                stars.append(self.starFromWCS)
                self.logger.warning("Assuming all stars at {}\n".format(self.starFromWCS))
            else:
                self.logger.info("Radon transforming image {}...".format(ii))
                stars.append(find_star_radon(imgIter,
                                    starGuess,
                                    self.spikeAngles, IWA=self.radonIWA,
                                    sp_width=20, r_mask=None)) # [pixels] y,x

        return np.array(stars)


    def align_imgs(self, imgs, indImg, masks=[], commonMask=None, pad=True,
                   starGuess=None, finalStarYX=None):

        # Find the occulted star's coordinates in every image.
        stars = self.find_star(imgs, commonMask, starGuess=starGuess)

        # Define new aligned star position.
        if pad:
            matSize = np.array([2048, 2048])
            if finalStarYX is not None:
                alignStar = np.array(finalStarYX)
            else:
                alignStar = matSize//2
        else:
            matSize = None
            if finalStarYX is not None:
                alignStar = np.array(finalStarYX)
            else:
                alignStar = np.round(np.mean(stars, axis=0))

# FIX ME!!! Might be bad to redefine alignStar here.
        self.alignStar = alignStar

        # Register images to align stars.
        # Also pad the images to matSize if self.noPad is not set.
        self.logger.info("ALIGNING images and masks to common star position...")
        alignImgs = []
        alignMasks = []
        for ii in tqdm(range(len(imgs)), desc="Align images"):
            alignImgs.append(shift_pix_to_pix(imgs[ii], stars[ii],
                                              finalYX=self.alignStar,
                                              outputSize=matSize,
                                              order=3, fill=0.))

        # Align and pad associated mask arrays as well.
        if commonMask is not None:
            alignMasks = shift_pix_to_pix(commonMask, stars[ii],
                                          finalYX=self.alignStar,
                                          outputSize=matSize,
                                          order=1, fill=-1e4)
        else:
            for ii in tqdm(range(len(masks)), desc="Align masks"):
                alignMasks.append(shift_pix_to_pix(masks[ii], stars[ii],
                                                   finalYX=self.alignStar,
                                                   outputSize=matSize,
                                                   order=1, fill=-1e4))
            alignMasks = np.array(alignMasks)

        alignImgs = np.array(alignImgs)

        # Preserve original star positions before overwriting stars with new
        # aligned position. Keep the offsets around for posterity.
# FIX ME!!! Better handle the original CRSPLIT star positions.
        self.starsOriginal = stars.copy()
        self.stars = np.tile(self.alignStar, (len(imgs), 1))
        self.alignStarOffsets = self.stars - self.starsOriginal

        # # Make aligned images the working images from now on.
        # self.workingImgs = alignImgs

        # Update header WCS reference coordinates and image dimensions.
        self.update_wcs(alignImgs, indImg=indImg)

        return alignImgs, alignMasks


    def pad_imgs(self, imgs, outputShape=(1100,1100), fill=0.):

        mat = np.zeros(outputShape)
        mat += fill

        imgsPadded = []
        for im in imgs:
            dimsOrig = im.shape
            shiftY = (outputShape[0] - dimsOrig[0])//2
            shiftX = (outputShape[1] - dimsOrig[1])//2

            imgsPadded.append(mat.copy())
            imgsPadded[-1][shiftY:shiftY + dimsOrig[0],
                           shiftX:shiftX + dimsOrig[1]] = im

        return np.array(imgsPadded)


    def subtract_background(self):
        """
        Subtract a global background from each image in self.workingImgs.
        """
        bgs = []
        for ii, im in enumerate(self.workingImgs):
            if self.bgCen is not None:
                # Assume all reference images are taken at the same PA.
                if ii not in self.refInds:
                    bgCenRot = utils.rotate_yx(self.bgCen, self.alignStar,
                                               self.orientats[ii])
    # # FIX ME!!! For modern system of padding images, don't add the star offsets to the bg positions.
    #                 if self.targ not in noOffsetDatasets:
    #                     bgCenRot += self.alignStarOffsets[ii]
                # Still need to offset the reference image for these modern cases
                # because bg positions are chosen from the raw images.
                else:
                    bgCenRot = self.bgCenRef + self.alignStarOffsets[ii]

                if self.workingImgs.ndim > 3:
                    for jj in range(self.workingImgs.shape[1]):
                        # Work on a source-masked copy of the aligned image.
                        imSub, bg = utils.subtract_bg(im[jj].copy(), bgCenRot, self.bgRadius)
                        # Can't use subtract_bg image output here because it is masked.
                        self.workingImgs[ii][jj] = im[jj] - bg
                        bgs.append(bg)
                else:
                    imSub, bg = utils.subtract_bg(im.copy(), bgCenRot, self.bgRadius)
                    # Can't use subtract_bg image output here because it is masked.
                    self.workingImgs[ii] = im - bg
                    bgs.append(bg)
            # Handle the default case of sampling background many places in
            # the image and take a median.
            else:
                if self.workingImgs.ndim > 3:
                    for jj in range(self.workingImgs.shape[1]):
                        # Work on a source-masked copy of the aligned image.
                        bg = utils.randomly_sample_bg(im[jj].copy(),
                                                 excludeYX=[self.stars[ii]],
                                                 bgRadius=40,
                                                 exclusionRadius=200,
                                                 mask=self.alignMasks[ii].astype(bool))
                        bgs.append(bg)
                        self.workingImgs[ii][jj] = im[jj] - bg

        self.bgs = np.array(bgs)

        return

    def update_dimensions(self, imgs, hdrs):
        """
        Update header array dimensions given image arrays.
        """
        for ii, im in enumerate(imgs):
            for jj, hdr in enumerate(hdrs[ii]):
                if 'NAXIS1' in hdr.keys():
                    hdrs[ii][jj]['NAXIS1'] = im.shape[1] # x
                    hdrs[ii][jj]['NAXIS2'] = im.shape[0] # y

        return hdrs


# FIX ME!!! Is setting CRPIX1 and 2 to the new star center the right thing?
# Should we instead simply shift CRPIX by the same shift applied to the star center?
    def update_wcs(self, imgs, indImg):
        """
        Update header WCS reference coordinates and array dimensions given
        image arrays and taking self.stars coordinates.
        """

        if hasattr(self, 'sciInds'):
            # Select the headers associated to the correct image.
            hdrs = self.allHdrs[indImg]
            # Loop over headers for the image.
            indSplit = 0
            for ii, hdr in enumerate(hdrs):
                if ('CRPIX1' in hdr.keys()) and (hdr.get('EXTNAME', '') == 'SCI'):
                    hdrs[ii]['CRPIX1'] = self.stars[indSplit][1] # x
                    hdrs[ii]['CRPIX2'] = self.stars[indSplit][0] # y
                if ('NAXIS1' in hdr.keys()) and (hdr.get('EXTNAME', '') == 'SCI'):
                    hdrs[ii]['NAXIS1'] = imgs[indSplit].shape[1] # x
                    hdrs[ii]['NAXIS2'] = imgs[indSplit].shape[0] # y

            self.sciHdrs = [self.allHdrs[ind] for ind in self.sciInds]
            self.refHdrs = [self.allHdrs[ind] for ind in self.refInds]

            # for ii in self.sciInds:
            #     for jj, hdr in enumerate(self.allHdrs[ii]):
            #         if 'CRPIX1' in hdr.keys():
            #             self.allHdrs[ii][jj]['CRPIX1'] = self.stars[ii][1] # x
            #             self.allHdrs[ii][jj]['CRPIX2'] = self.stars[ii][0] # y
            #         if 'NAXIS1' in hdr.keys():
            #             self.allHdrs[ii][jj]['NAXIS1'] = imgs[ii].shape[1] # x
            #             self.allHdrs[ii][jj]['NAXIS2'] = imgs[ii].shape[0] # y
            # for ii in self.refInds:
            #     for jj, hdr in enumerate(self.allHdrs[ii]):
            #         if 'CRPIX1' in hdr.keys():
            #             self.allHdrs[ii][jj]['CRPIX1'] = self.stars[ii][1] # x
            #             self.allHdrs[ii][jj]['CRPIX2'] = self.stars[ii][0] # y
            #         if 'NAXIS1' in hdr.keys():
            #             self.allHdrs[ii][jj]['NAXIS1'] = imgs[ii].shape[1] # x
            #             self.allHdrs[ii][jj]['NAXIS2'] = imgs[ii].shape[0] # y

            # self.sciHdrs = [self.allHdrs[ind] for ind in self.sciInds]
            # self.refHdrs = [self.allHdrs[ind] for ind in self.refInds]

# FIX ME!!! This 'else' might not work correctly anymore. Check it.
        else:
            for ii in range(len(imgs)):
                for jj, hdr in enumerate(self.allHdrs[ii]):
                    if 'CRPIX1' in hdr.keys():
                        self.allHdrs[ii][jj]['CRPIX1'] = self.stars[ii][1] # x
                        self.allHdrs[ii][jj]['CRPIX2'] = self.stars[ii][0] # y
                    if 'NAXIS1' in hdr.keys():
                        self.allHdrs[ii][jj]['NAXIS1'] = imgs[ii].shape[1] # x
                        self.allHdrs[ii][jj]['NAXIS2'] = imgs[ii].shape[0] # y

        return

    def derotate(self, imgs, orientats, cens):
    
        rotImgs = []
        for ii in range(len(imgs)):
            angle = orientats[ii]
            rotImgs.append(utils.rotate_array(imgs[ii], cens[ii], angle,
                                              preserve_nan=True, cval=np.nan))
    
        return np.array(rotImgs)


    def update_fits(self, filePath, newData=None, newHeader=None,
                    sciOnly=False):
        """

        """

        with fits.open(filePath, mode='update') as hdul:
            if newData is not None:
                if sciOnly:
                    sciInds = []
                    for ii in range(len(hdul)):
                        if hdul[ii].header.get('EXTNAME', '') == 'SCI':
                            sciInds.append(ii)
                    for ii in range(len(newData)):
                        # if not np.all(np.isnan(newData[ii])):
                        if ii >= len(sciInds):
                            for jj in range(3):
                                hdul.append(fits.ImageHDU(data=newData[ii].astype(np.float32)))
                        else:
                            hdul[sciInds[ii]].data = newData[ii].astype(np.float32)

            hdul.flush()

        # if unit in ['Jy', 'Jy arcsec-2', 'mJy', 'mJy arcsec-2']:
        #     saveName = f"unified_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}_{unit.replace(' ', '_')}.fits"
        #     # saveName = "{}_{}_stis_{}_{}_a{}_{}_{}.fits".format(targ, obsDate,
        #     #                                 propAper, psfSubMode.lower(), ann,
        #     #                                 imType, newUnit.replace(' ', '_'))
        # else:
        #     saveName = f"unified_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}.fits"
        #     # saveName = "{}_{}_stis_{}_{}_a{}_final.fits".format(targ, obsDate,
        #     #                                 propAper, psfSubMode.lower(), ann)

#         # Create primary header.
#         hdul = fits.HDUList()
#         if priHdr is not None:
#             newPriHdr = priHdr.copy()
#         else:
#             newPriHdr = None
#         newPriHdr['FILENAME'] = os.path.basename(filePath)
#         newPriHdr.add_comment(f'Aligned CRSPLIT images from {self.inputType}.')
#         hdul.append(fits.PrimaryHDU(header=newPriHdr))

#         # Create headers for all data extensions.
#         for ii, da in enumerate(data):
#             bitpix = headers[ii]['BITPIX']
#             # Enforce data types to stabilize file size.
#             if bitpix in [8, 16, 32, 64]:
#                 hdul.append(fits.ImageHDU(data=np.array(da).astype(f'uint{bitpix}'), header=headers[ii].copy()))
#             else:
#                 hdul.append(fits.ImageHDU(data=np.array(da).astype('float32'), header=headers[ii].copy()))

# # # FIX ME!!! This n_sci is a small hack to get around the fact that we only
# # # record starsOriginal for the SCI extensions during alignment.
# #         # Track which SCI header we are at in the CRSPLIT sequence.
# #         n_sci = 0

#         # Add the following keys to all headers.
#         for ii, hdu in enumerate(hdul):
#             if not 'HISTORY' in hdu.header:
#                 try:
#                     hdu.header.add_history('Created by {}'.format(getpass.getuser()))
#                 except:
#                     hdu.header.add_history('Created by unknown user')
#             hdu.header['TARGNAME'] = (newPriHdr.get('TARGNAME', ''))
#             hdu.header['PSFNAME'] = (self.psfRefName, 'Reference PSF name')
#             hdu.header['FILETYPE'] = (filetype)
#             hdu.header['NCOMBINE'] = (1, 'Number of images combined')
#             hdu.header['INPUTTYP'] = (self.inputType, 'Type of input data')
#             hdu.header['PSFSUBMD'] = (self.psfSubMode, 'PSF-subtraction mode')

#             if hdu.header.get('EXTNAME', '') == 'SCI':
#                 hdu.header['FIXPIX'] = (not self.noFixPix, 'Bad pixels were fixed?')
#                 hdu.header['ORBCOMBI'] = (not self.noCombine, 'Stacked images per orbit before PSF sub?')
#                 hdu.header['CENTRADN'] = (not self.noRadon, 'Radon transformed to get center?')
#                 hdu.header['ALIGNED'] = (True, 'Images aligned to PSFCENT?')
#                 hdu.header['PSFCENTY'] = (self.alignStar[0], 'Y location of target star center')
#                 hdu.header['PSFCENTX'] = (self.alignStar[1], 'X location of target star center')
#                 if ii >= 1:
#                     hdu.header['ORIGCENY'] = (np.mean(self.starsOriginal, axis=0)[0], 'Un-padded pre-alignment star center Y')
#                     hdu.header['ORIGCENX'] = (np.mean(self.starsOriginal, axis=0)[1], 'Un-padded pre-alignment star center X')
#                     # n_sci += 1
#                 if 'x' in self.inputType:
#                     hdu.header['DISTCORR'] = (True, 'Distortion corrected images?')
#                 else:
#                     hdu.header['DISTCORR'] = (False, 'Distortion corrected images?')
#                 # hdu.header['TEXPTIME'] = (np.sum(self.exptimes_s), 'Total combined integration time in s')
#                 # hdu.header['BUNIT'] = (self.bunit, 'brightness units')
#                 # hdu.header['PHOTFLAM'] = (self.photflam_avg, 'inverse sensitivity, ergs/s/cm2/Ang per count/s')
#                 # Propagate certain keys from individual raw FITS.
#                 # for key in ['TELESCOP', 'INSTRUME', 'EQUINOX', 'RA_TARG', 
#                 #             'DEC_TARG', 'PROPOSID', 'TDATEOBS',
#                 #             'CCDAMP', 'CCDGAIN', 'CCDOFFST', 'OBSTYPE', 'OBSMODE',
#                 #             'PHOTMODE', 'SUBARRAY', 'DETECTOR', 'OPT_ELEM',
#                 #             'APERTURE', 'PROPAPER', 'FILTER', 'APER_FOV',
#                 #             'CRSPLIT', 'PHOTFLAM', 'PHOTZPT', 'PHOTPLAM', 'PHOTBW', ]:
#                 #     try:
#                 #         hdu.header[key] = (headers[ii][key],
#                 #                             headers[ii].comments[key])
#                 #     except:
#                 #         print(f"Could not propagate header keyword {key}")
#             # if self.bgCen is not None:
#             #     hdu.header['BGCENTY'] = (self.bgCen[0], 'Science Y center background sample')
#             #     hdu.header['BGCENTX'] = (self.bgCen[1], 'Science X center background sample')
#             # else:
#             #     hdu.header['BGCENTY'] = (None, 'Science Y center background sample')
#             #     hdu.header['BGCENTX'] = (None, 'Science X center background sample')
#             # if self.bgCenRef is not None:
#             #     hdu.header['BGCENTYR'] = (self.bgCenRef[0], 'Reference Y center background sample')
#             #     hdu.header['BGCENTXR'] = (self.bgCenRef[1], 'Reference X center background sample')
#             # else:
#             #     hdu.header['BGCENTYR'] = (None, 'Reference Y center background sample')
#             #     hdu.header['BGCENTXR'] = (None, 'Reference X center background sample')
#             # hdu.header['BGRADIUS'] = (self.bgRadius, 'Radius background sample region (pix)')
#             # hdu.header['ANNULI'] = (self.ann, 'Number of subtraction region annuli')
#             # hdu.header['SPWIDTH'] = (self.spWidth, 'Diff. spike mask width (pix)')
#             # hdu.header['RADPROFS'] = (self.subRadProf, 'Residual radial profile subtracted?')
#             # hdu.header.add_comment('Image ORIENTAT angles: '\
#             #                         + str(self.orientats).replace('\n', ''))
#             hdu.header.add_comment(cmt1)
#             hdu.header.add_comment('Reduction info file: {}'.format(self.infoPath))
#             # hdu.header.add_comment(f'Image integration times (s): {self.exptimes_all_s}')

#         hdul.writeto(filePath, overwrite=True)
#         print(f"\n{filetype} saved as {filePath}")

        return


    def write_aligned_fits(self, data, headers, filePath, priHdr=None):
        """

        """

        # if unit in ['Jy', 'Jy arcsec-2', 'mJy', 'mJy arcsec-2']:
        #     saveName = f"unified_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}_{unit.replace(' ', '_')}.fits"
        #     # saveName = "{}_{}_stis_{}_{}_a{}_{}_{}.fits".format(targ, obsDate,
        #     #                                 propAper, psfSubMode.lower(), ann,
        #     #                                 imType, newUnit.replace(' ', '_'))
        # else:
        #     saveName = f"unified_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}.fits"
        #     # saveName = "{}_{}_stis_{}_{}_a{}_final.fits".format(targ, obsDate,
        #     #                                 propAper, psfSubMode.lower(), ann)

        filetype = 'Aligned CRSPLIT image cube'
        cmt1 = f'{self.inputType} input file type'

        # Create primary header.
        hdul = fits.HDUList()
        if priHdr is not None:
            newPriHdr = priHdr.copy()
        else:
            newPriHdr = None
        newPriHdr['FILENAME'] = os.path.basename(filePath)
        newPriHdr.add_comment(f'Aligned CRSPLIT images from {self.inputType}.')
        hdul.append(fits.PrimaryHDU(header=newPriHdr))

        # Create headers for all data extensions.
        for ii, da in enumerate(data):
            if (np.all(np.isnan(da)) | np.all(da == 0)):
                continue
            bitpix = headers[ii].get('BITPIX')
            # Enforce data types to stabilize file size.
            if bitpix in [8, 16, 32, 64]:
                hdul.append(fits.ImageHDU(data=np.array(da).astype(f'uint{bitpix}'), header=headers[ii].copy()))
            else:
                hdul.append(fits.ImageHDU(data=np.array(da).astype('float32'), header=headers[ii].copy()))

# # FIX ME!!! This n_sci is a small hack to get around the fact that we only
# # record starsOriginal for the SCI extensions during alignment.
#         # Track which SCI header we are at in the CRSPLIT sequence.
#         n_sci = 0

        # Add the following keys to all headers.
        for ii, hdu in enumerate(hdul):
            if not 'HISTORY' in hdu.header:
                try:
                    hdu.header.add_history('Created by {}'.format(getpass.getuser()))
                except:
                    hdu.header.add_history('Created by unknown user')
            hdu.header['TARGNAME'] = (newPriHdr.get('TARGNAME', ''))
            hdu.header['PSFNAME'] = (self.psfRefName, 'Reference PSF name')
            hdu.header['FILETYPE'] = (filetype)
            hdu.header['INPUTTYP'] = (self.inputType, 'Type of input data')
            hdu.header['PSFSUBMD'] = (self.psfSubMode, 'PSF-subtraction mode')

            if hdu.header.get('EXTNAME', '') == 'SCI':
                hdu.header['NCOMBINE'] = (1, 'Number of images combined')
                hdu.header['FIXPIX'] = (not self.noFixPix, 'Bad pixels were fixed?')
                hdu.header['ORBCOMBI'] = (not self.noCombine, 'Stacked images per orbit before PSF sub?')
                hdu.header['CENTRADN'] = (not self.noRadon, 'Radon transformed to get center?')
                hdu.header['ALIGNED'] = (True, 'Images aligned to PSFCENT?')
                hdu.header['PSFCENTY'] = (self.alignStar[0], 'Y location of target star center')
                hdu.header['PSFCENTX'] = (self.alignStar[1], 'X location of target star center')
                if ii >= 1:
                    hdu.header['ORIGCENY'] = (np.mean(self.starsOriginal, axis=0)[0], 'Un-padded pre-alignment star center Y')
                    hdu.header['ORIGCENX'] = (np.mean(self.starsOriginal, axis=0)[1], 'Un-padded pre-alignment star center X')
                    # n_sci += 1
                if 'x' in self.inputType:
                    hdu.header['DISTCORR'] = (True, 'Distortion corrected images?')
                else:
                    hdu.header['DISTCORR'] = (False, 'Distortion corrected images?')
                # hdu.header['TEXPTIME'] = (np.sum(self.exptimes_s), 'Total combined integration time in s')
                # hdu.header['BUNIT'] = (self.bunit, 'brightness units')
                # hdu.header['PHOTFLAM'] = (self.photflam_avg, 'inverse sensitivity, ergs/s/cm2/Ang per count/s')
                # Propagate certain keys from individual raw FITS.
                # for key in ['TELESCOP', 'INSTRUME', 'EQUINOX', 'RA_TARG', 
                #             'DEC_TARG', 'PROPOSID', 'TDATEOBS',
                #             'CCDAMP', 'CCDGAIN', 'CCDOFFST', 'OBSTYPE', 'OBSMODE',
                #             'PHOTMODE', 'SUBARRAY', 'DETECTOR', 'OPT_ELEM',
                #             'APERTURE', 'PROPAPER', 'FILTER', 'APER_FOV',
                #             'CRSPLIT', 'PHOTFLAM', 'PHOTZPT', 'PHOTPLAM', 'PHOTBW', ]:
                #     try:
                #         hdu.header[key] = (headers[ii][key],
                #                             headers[ii].comments[key])
                #     except:
                #         print(f"Could not propagate header keyword {key}")
            # if self.bgCen is not None:
            #     hdu.header['BGCENTY'] = (self.bgCen[0], 'Science Y center background sample')
            #     hdu.header['BGCENTX'] = (self.bgCen[1], 'Science X center background sample')
            # else:
            #     hdu.header['BGCENTY'] = (None, 'Science Y center background sample')
            #     hdu.header['BGCENTX'] = (None, 'Science X center background sample')
            # if self.bgCenRef is not None:
            #     hdu.header['BGCENTYR'] = (self.bgCenRef[0], 'Reference Y center background sample')
            #     hdu.header['BGCENTXR'] = (self.bgCenRef[1], 'Reference X center background sample')
            # else:
            #     hdu.header['BGCENTYR'] = (None, 'Reference Y center background sample')
            #     hdu.header['BGCENTXR'] = (None, 'Reference X center background sample')
            # hdu.header['BGRADIUS'] = (self.bgRadius, 'Radius background sample region (pix)')
            # hdu.header['ANNULI'] = (self.ann, 'Number of subtraction region annuli')
            # hdu.header['SPWIDTH'] = (self.spWidth, 'Diff. spike mask width (pix)')
            # hdu.header['RADPROFS'] = (self.subRadProf, 'Residual radial profile subtracted?')
            # hdu.header.add_comment('Image ORIENTAT angles: '\
            #                         + str(self.orientats).replace('\n', ''))
            hdu.header.add_comment(cmt1)
            hdu.header.add_comment('Reduction info file: {}'.format(self.infoPath))
            # hdu.header.add_comment(f'Image integration times (s): {self.exptimes_all_s}')

        hdul.writeto(filePath, overwrite=True)
        self.logger.info(f"{filetype} SAVED as {filePath}")

        return


    def save_unified_to_fits(self, data, unit, cid=''):
        """

        """

        if unit in ['Jy', 'Jy arcsec-2', 'mJy', 'mJy arcsec-2']:
            saveName = f"unified_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}_{unit.replace(' ', '_')}{cid}.fits"
            # saveName = "{}_{}_stis_{}_{}_a{}_{}_{}.fits".format(targ, obsDate,
            #                                 propAper, psfSubMode.lower(), ann,
            #                                 imType, newUnit.replace(' ', '_'))
        else:
            saveName = f"unified_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}{cid}.fits"
            # saveName = "{}_{}_stis_{}_{}_a{}_final.fits".format(targ, obsDate,
            #                                 propAper, psfSubMode.lower(), ann)

        filetype = 'Unified images from combined CRSPLITS'
        cmt1 = f'{self.inputType} input file type'

        hdu = fits.PrimaryHDU(data=np.array(data).astype('float32'))
        if not 'HISTORY' in hdu.header:
            try:
                hdu.header.add_history('Created by {}'.format(getpass.getuser()))
            except:
                hdu.header.add_history('Created by unknown user')
        hdu.header['TARGNAME'] = (self.targ)
        hdu.header['PSFNAME'] = (self.psfRefName, 'Reference PSF name')
        hdu.header['FILETYPE'] = (filetype)
        hdu.header['NCOMBINE'] = (self.sciHdrs[0][1].get('NCOMBINE', 1),
                                  'Number of images combined')
        hdu.header['INPUTTYP'] = (self.inputType, 'Type of input data')
        hdu.header['PSFSUBMD'] = (self.psfSubMode, 'PSF-subtraction mode')
        # hdu.header['CENTRADN'] = (not self.noRadon, 'Radon transformed to get center?')
        # hdu.header['PSFCENTY'] = (self.alignStar[0], 'Y location of target star center')
        # hdu.header['PSFCENTX'] = (self.alignStar[1], 'X location of target star center')
        # hdu.header['ORIGCENY'] = (np.mean(self.starsOriginal, axis=0)[0], 'Un-padded mean star center Y')
        # hdu.header['ORIGCENX'] = (np.mean(self.starsOriginal, axis=0)[1], 'Un-padded mean star center X')
        hdu.header['TEXPTIME'] = (np.sum(self.exptimes_s), 'Total combined integration time in s')
        hdu.header['BUNIT'] = (self.bunit, 'brightness units')
        hdu.header['PHOTFLAM'] = (self.photflam_avg, 'inverse sensitivity, ergs/s/cm2/Ang per count/s')
        # Propagate certain keys from individual raw FITS.
        for key in ['TELESCOP', 'INSTRUME','EQUINOX','RA_TARG', 'DEC_TARG',
                    'PROPOSID', 'TDATEOBS',
                    'CCDAMP', 'CCDGAIN', 'CCDOFFST', 'OBSTYPE', 'OBSMODE',
                    'PHOTMODE', 'SUBARRAY', 'DETECTOR', 'OPT_ELEM',
                    'APERTURE', 'PROPAPER', 'FILTER', 'APER_FOV',
                    'CRSPLIT', 'PHOTZPT', 'PHOTPLAM', 'PHOTBW', ]:
            try:
                hdu.header[key] = (self.sciHdrs[0][0][key],
                                   self.sciHdrs[0][0].comments[key])
            except:
                self.logger.warning(f"Could not propagate header keyword {key}")
        # if self.bgCen is not None:
        #     hdu.header['BGCENTY'] = (self.bgCen[0], 'Science Y center background sample')
        #     hdu.header['BGCENTX'] = (self.bgCen[1], 'Science X center background sample')
        # else:
        #     hdu.header['BGCENTY'] = (None, 'Science Y center background sample')
        #     hdu.header['BGCENTX'] = (None, 'Science X center background sample')
        # if self.bgCenRef is not None:
        #     hdu.header['BGCENTYR'] = (self.bgCenRef[0], 'Reference Y center background sample')
        #     hdu.header['BGCENTXR'] = (self.bgCenRef[1], 'Reference X center background sample')
        # else:
        #     hdu.header['BGCENTYR'] = (None, 'Reference Y center background sample')
        #     hdu.header['BGCENTXR'] = (None, 'Reference X center background sample')
        # hdu.header['BGRADIUS'] = (self.bgRadius, 'Radius background sample region (pix)')
        hdu.header['FIXPIX'] = (not self.noFixPix, 'Bad pixels were fixed?')
        # hdu.header['ORBCOMBI'] = (not self.noCombine, 'Stacked images per orbit before PSF sub?')
        # hdu.header['ANNULI'] = (self.ann, 'Number of subtraction region annuli')
        # hdu.header['SPWIDTH'] = (self.spWidth, 'Diff. spike mask width (pix)')
        # hdu.header['RADPROFS'] = (self.subRadProf, 'Residual radial profile subtracted?')
        hdu.header.add_comment('Image ORIENTAT angles: '\
                               + str(self.orientats).replace('\n', ''))
        hdu.header.add_comment(cmt1)
        hdu.header.add_comment('Reduction info file: {}'.format(self.infoPath))
        hdu.header.add_comment(f'Image integration times (s): {self.exptimes_all_s}'.replace('\n', ' '))

        hdu.writeto(self.dataDir + saveName, overwrite=True)
        self.logger.info(f"{filetype} SAVED as {self.dataDir + saveName}")

        return


    def save_psfsub_to_fits(self, data, imType, unit, cid=''):
        """
        imType: str
            Either 'final' for the collapsed final image, or 'psfcube' for the
            individual PSF-subtracted frames.
        """

        if unit in ['Jy', 'Jy arcsec-2', 'mJy', 'mJy arcsec-2']:
            saveName = f"{imType}_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}_{unit.replace(' ', '_')}{cid}.fits"
            # saveName = "{}_{}_stis_{}_{}_a{}_{}_{}.fits".format(targ, obsDate,
            #                                 propAper, psfSubMode.lower(), ann,
            #                                 imType, newUnit.replace(' ', '_'))
        else:
            saveName = f"{imType}_{self.targ}_{self.obsDate}_{self.instrument}_{self.inputType}_{self.propAper}_{self.psfSubMode.lower()}_a{self.ann}{cid}.fits"
            # saveName = "{}_{}_stis_{}_{}_a{}_final.fits".format(targ, obsDate,
            #                                 propAper, psfSubMode.lower(), ann)

        # Adjust based on the image type being saved.
        if imType == 'final':
            filetype = 'Final combined PSF-subtracted image'
            cmt1 = f'{self.psfSubMode} PSF-subtracted combined image'
            if np.array(data).ndim == 3:
                cmt1 += 's. Index 0 = all subtractions; Index 1 = no '\
                    'post-combine radial profile subtraction; Index 2 = no '\
                    'radial profile subtractions at all.'
        elif imType == 'psfcube':
            filetype = 'Individual PSF-subtracted image cube'
            cmt1 = f'{self.psfSubMode} PSF-subtracted individual images'
        elif imType == 'error':
            filetype = 'Error map for final combined PSF-subtracted image'
            cmt1 = f'{self.psfSubMode} PSF-subtracted error map'
        elif imType == 'snr':
            filetype = 'SNR map for final combined PSF-subtracted image'
            cmt1 = f'{self.psfSubMode} PSF-subtracted SNR map'
            if np.array(data).ndim == 3:
                cmt1 += 's. Index 0 = all subtractions; Index 1 = no '\
                    'post-combine radial profile subtraction; Index 2 = no '\
                    'radial profile subtractions at all.'

        if np.array(data).ndim > 3:
            hdul = fits.HDUList()
            priHdu = fits.PrimaryHDU()
            newPriHdr = priHdu.header
            newPriHdr['FILENAME'] = os.path.basename(saveName)
            hdul.append(fits.PrimaryHDU(header=newPriHdr))
        else:
            hdu = fits.PrimaryHDU(data=np.array(data).astype('float32'))
            newPriHdr = hdu.header
        if not 'HISTORY' in newPriHdr:
            try:
                newPriHdr.add_history('Created by {}'.format(getpass.getuser()))
            except:
                newPriHdr.add_history('Created by unknown user')
        newPriHdr['TARGNAME'] = (self.targ)
        newPriHdr['PSFNAME'] = (self.psfRefName, 'Reference PSF name')
        newPriHdr['FILETYPE'] = (filetype)
        newPriHdr['NCOMBINE'] = (self.sciHdrs[0][1].get('NCOMBINE', 1), 'Number of images combined')
        newPriHdr['INPUTTYP'] = (self.inputType, 'Type of input data')
        newPriHdr['PSFSUBMD'] = (self.psfSubMode, 'PSF-subtraction mode')
        newPriHdr['CENTRADN'] = (not self.noRadon, 'Radon transformed to get center?')
        newPriHdr['PSFCENTY'] = (self.alignStar[0], 'Y location of target star center')
        newPriHdr['PSFCENTX'] = (self.alignStar[1], 'X location of target star center')
        newPriHdr['ORIGCENY'] = (np.mean(self.starsOriginal, axis=0)[0], 'Un-padded mean star center Y')
        newPriHdr['ORIGCENX'] = (np.mean(self.starsOriginal, axis=0)[1], 'Un-padded mean star center X')
        newPriHdr['TEXPTIME'] = (np.sum(self.exptimes_s), 'Total combined integration time in s')
        newPriHdr['BUNIT'] = (self.bunit, 'brightness units')
        newPriHdr['PHOTFLAM'] = (self.photflam_avg, 'inverse sensitivity, ergs/s/cm2/Ang per count/s')
        # Propagate certain keys from individual raw FITS.
        for key in ['TELESCOP', 'INSTRUME','EQUINOX','RA_TARG', 'DEC_TARG',
                    'PROPOSID', 'TDATEOBS',
                    'CCDAMP', 'CCDGAIN', 'CCDOFFST', 'OBSTYPE', 'OBSMODE',
                    'PHOTMODE', 'SUBARRAY', 'DETECTOR', 'OPT_ELEM',
                    'APERTURE', 'PROPAPER', 'FILTER', 'APER_FOV',
                    'CRSPLIT', 'PHOTZPT', 'PHOTPLAM', 'PHOTBW', ]:
            try:
                newPriHdr[key] = (self.sciHdrs[0][0][key],
                                   self.sciHdrs[0][0].comments[key])
            except:
                self.logger.warning(f"Could not propagate header keyword {key}")
        if self.bgCen is not None:
            newPriHdr['BGCENTY'] = (self.bgCen[0], 'Science Y center background sample')
            newPriHdr['BGCENTX'] = (self.bgCen[1], 'Science X center background sample')
        else:
            newPriHdr['BGCENTY'] = (None, 'Science Y center background sample')
            newPriHdr['BGCENTX'] = (None, 'Science X center background sample')
        if self.bgCenRef is not None:
            newPriHdr['BGCENTYR'] = (self.bgCenRef[0], 'Reference Y center background sample')
            newPriHdr['BGCENTXR'] = (self.bgCenRef[1], 'Reference X center background sample')
        else:
            newPriHdr['BGCENTYR'] = (None, 'Reference Y center background sample')
            newPriHdr['BGCENTXR'] = (None, 'Reference X center background sample')
        newPriHdr['BGRADIUS'] = (self.bgRadius, 'Radius background sample region (pix)')
        newPriHdr['FIXPIX'] = (not self.noFixPix, 'Bad pixels were fixed?')
        newPriHdr['ORBCOMBI'] = (not self.noCombine, 'Stacked images per orbit before PSF sub?')
        newPriHdr['ANNULI'] = (self.ann, 'Number of subtraction region annuli')
        newPriHdr['SPWIDTH'] = (self.spWidth, 'Diff. spike mask width (pix)')
        newPriHdr['PSFRIN'] = (self.exclusionsSci.get('r_in', np.nan), 'PSF subtraction inner radius (pix)')
        newPriHdr['PSFROUT'] = (self.exclusionsSci.get('r_out', np.nan), 'PSF subtraction outer radius (pix)')
        newPriHdr['RADPROFS'] = (self.subRadProf, 'Residual radial profile subtracted?')
        try:
            newPriHdr.add_comment('Science images used: '\
                                  f'{[os.path.basename(fp) for fp in self.fileList[self.sciInds]]}')
        except:
            pass
        try:
            newPriHdr.add_comment('Reference images used: '\
                                  f'{[os.path.basename(fp) for fp in self.fileList[self.refInds]]}')
        except:
            pass
        newPriHdr.add_comment('Constituent image ORIENTAT angles: '\
                               + str(self.orientats[self.sciInds]).replace('\n', ''))
        if self.subFinalRadProf and imType in ['final']:
            newPriHdr.add_comment('Post-collapse radial profile subtracted')
        newPriHdr.add_comment(cmt1)
        newPriHdr.add_comment('Reduction info file: {}'.format(self.infoPath))
        try:
            newPriHdr.add_comment('PSF-subtraction exclusions (science images): ' \
                                   + str(self.exclusionsSci).replace('\n', ''))
            newPriHdr.add_comment('PSF-subtraction exclusions (reference images): ' \
                                   + str(self.exclusionsRef).replace('\n', ''))
        except:
            newPriHdr.add_comment('Failed to comment on exclusion regions')
        try:
            newPriHdr.add_comment('PSF-subtraction scale factors for reference images by science image: ' \
                                   + str(np.round(self.refScaleFactors, 5)).replace('\n', ''))
        except:
            newPriHdr.add_comment('PSF-subtraction scale factors for reference images by science image: ' \
                                   + str(self.refScaleFactors).replace('\n', ''))
        try:
            newPriHdr.add_comment('Constituent image exposure times (s): ' \
                                  + str(str(self.exptimes_s).replace('\n', '')))
        except:
            newPriHdr.add_comment('Failed to comment on constituent image exposure times.')

        if np.array(data).ndim > 3:
            # Create headers for all data extensions.
            for ii, da in enumerate(data):
                # if (np.all(np.isnan(da)) | np.all(da == 0)):
                #     continue
                # Enforce data types to stabilize file size.
                hdul.append(fits.ImageHDU(data=np.array(da).astype('float32'), header=newPriHdr.copy()))
                hdr = hdul[-1].header
                if ii == 0:
                    hdr['FILETYPE'] = ('Final combined PSF-subtracted image')
        else:
            hdu.writeto(self.dataDir + saveName, overwrite=True)


        self.logger.info(f"{filetype} SAVED as {self.dataDir + saveName}")

        return


    def post_reduction_analysis(self, im):
        # Measure an azimuthally averaged radial profile outward from the star
        # as a basic check for PSF subtraction quality and/or disk signal.

# FIX ME!!! Stop hardcoding the rMin and rMax values for the profile.
        rMax = 200
        meanRadProf2d = stis_psfsub.measure_mean_radial_prof(im,
                                                        self.alignStar,
                                                        paList=[0],
                                                        paHW=180, rMax=rMax,
                                                        interpInf=False,
                                                        smooth=False,
                                                        mode='mean')
        meanRadProf1d = meanRadProf2d[int(np.round(self.alignStar[0])),
                                      int(np.round(self.alignStar[1])) + 3:int(np.round(self.alignStar[1])) + rMax]

        rads = range(3, rMax, 1)
        plot_radprof_1d(rads, meanRadProf1d, yRange=None,
                        savePath=os.path.join(self.dataDir,
                                              f'final_radprof{self.cid}.png'))

        return


    def run(self):
        """
        Run the complete reduction pipeline from start to finish.
        """

        if not self.saveFinal:
            self.logger.warning("*** Final products will NOT be saved to disk "\
                  "because saveFinal==False. Add --saveFinal to the main "\
                  "script if you want to save these files.\n")

# TEMP!!! Clean this up later to just use the self attributes directly.
        obsMode = self.obsMode
        targ = self.targ
        ann = self.ann
        psfSubMode = self.psfSubMode
        noCombine = self.noCombine
        pad = not self.noPad

        roiY = [-200, 200]
        roiX = [-200, 200]

# FIX ME!!! Instrument-specific stuff here needs to go into instrument modules.
        # Angles at which diffraction spikes occur in STIS data [deg].
        self.spikeAngles = np.array([44.9, 134.7]) # [deg] clockwise from 0 at +X


        # # Summarize the dataset info from FITS headers.
        # if len(glob(dataDir + 'obs_log*.csv')) < 1:
        #   summarize_obs(dataDir, suffix=self.inputType, dsName=None)

        # Fetch dataset info and reduction parameters from info.json.
        info, infoPath = self.info, self.infoPath

        # Set reduction parameters based on info.json if overriding values were not given
        # as input args.
        if targ in ['', None]:
            targ = self.info['targetName']
        self.psfRefName = self.info['psfRefName']
        if self.spWidth is None:
            self.spWidth = self.info[self.obsMode]['spWidth'] # diffraction spike mask width [pix]
        self.radonIWA = self.info[self.obsMode]['radonIWA']
        exclusions = self.info[self.obsMode]['exclude'] # masked region definitions
        exclusionsSci = self.exclusionsSci # science masking
        sub_r_in = self.exclusionsSci['r_in'] # PSF subtraction inner radius
        sub_r_out = self.exclusionsSci['r_out'] # PSF subtraction outer radius

        # Load observation log.
        self.load_obs_log(logPath=info.get('obsLogPath'))

        # Find image filepaths in self.dataDir.
        self.find_imgs(suffix=self.inputType)

    # ========== CHARGE TRANSFER INEFFICIENCY CORRECTION ========== #

        if not self.noFixCTI:

            # Correct the flt files, creating new "flc.fits" files
            if self.inputType in ['flc', 'xfc', 'axc', 's2c']:
                self.logger.warning(f"{self.inputType} images already CTI-corrected; will"\
                      " not correct them again\n")
            # Check that proposal IDs are provide. If not, skip correction.
            elif len(self.pids) == 0:
                self.logger.warning("No proposal IDs given, so we are skipping "\
                      "CTI correction. Use main_reduce_stis.py --pids [#####]"\
                      " if you want to fetch MAST data for CTI correction.\n")
            else:
                self.logger.info("Start data fetching and CTI correction step")
                # Create CTI object instance; set up directories and environment.
                stisCTI = CTI(obsMode=self.obsMode,
                              loggerName=self.logger.name)
                stisCTI.setup_directories(base_dir=self.dataDir)
                stisCTI.setup_env()

                # Do the CTI correction.
                cti_success = stisCTI.run_cti(self.pids, target_name=self.targ,
                                              clean=True)

                if cti_success:
                    # Update the inputType.
                    if self.inputType == 'flt':
                        newSuffix = 'flc'
                    elif self.inputType == 'xft':
                        newSuffix = 'xfc'
                    elif self.inputType == 'axt':
                        newSuffix = 'axc'
                    elif self.inputType == 'sx2':
                        newSuffix = 's2c'
                    else:
                        newSuffix = 'idk'

                    if self.inputType in ['flt', 'xft', 'axt', 'sx2']:
                        self.inputType = newSuffix
                else:
                    self.logger.error("CTI correction unsuccessful. Downloading uncorrected flt files instead.")
                    stisCTI.download_without_cti(self.pids, target_name=self.targ,
                                                 clean=True)

            # Update the image paths based on CTI correction outcome.
            self.find_imgs(suffix=self.inputType)

            self.logger.info("Finished data fetching and CTI correction step")


    # ========== LOAD IMAGES ========== #

        # Returns list where [0] = data from each fits, [1] = headers from each fits.
        # Each item in [0] and [1] is also a list of arrays.
        self.logger.info("LOADING DATA...")
        imgsHdrs, targs = self.load_imgs(suffix=self.inputType)

        if self.inputType in ['flt', 'flc', 'xft', 'xfc', 'axt', 'axc']:
            sci_data, err_data, dq_data, dq_bool_16, dq_bool_8192, all_headers, sci_headers, \
                err_headers, dq_headers, targs = self.load_flt_imgs(self.fileList,
                                                                    plot_images=False,
                                                                    scienceOnly=False)
            self.allHdrs = all_headers
            self.imgShape = np.array(sci_data[0][0].shape)
            try:
                for hdr in sci_headers[0]:
                    if (hdr.get('EXTNAME', '') == 'SCI') and \
                        ('PSFCENTX' in hdr.keys()) and \
                        ('PSFCENTY' in hdr.keys()):
                        self.alignStar = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])
                        break
            except:
                if self.inputType in ['axt', 'axc']:
                    self.logger.warning("*** Failed to retrieved aligned star "\
                          "coordinates from FITS header. Images may not be "\
                          "aligned.\n")
                self.alignStar = None

    # ========== BAD PIXEL FIXING ========== #

# FIX ME!!! Turn these hard-coded DQ fixing flags into input options.
# DQ=8192 flags (cosmic-ray rejection) are not fixed by default because they
# are too agressive, including real sources like stars and diffraction spikes.
            fix_dq_16 = True
            fix_dq_8192 = False

            if fix_dq_8192:
                dq_8192_mask = dq_bool_8192
            else:
                dq_8192_mask = None

            # Fix bad pixels based on data quality flags from FITS.
            if not self.noFixPix:
                if fix_dq_16:
                    sci_data = self.pixelfixing(sci_data,
                                                dq_8192_mask=dq_8192_mask,
                                                dq_masks=[dq_bool_16],
                                                fix_other=True, verbose=True)
                else:
                    sci_data = self.pixelfixing(sci_data,
                                                dq_8192_mask=dq_8192_mask,
                                                dq_masks=[],
                                                fix_other=True, verbose=True)

                # Overwrite input FITS cube with the pixel-fixed CRSPLIT images.
                for ii, fp in enumerate(self.fileList):
                    self.update_fits(fp, newData=sci_data[ii],
                                     newHeader=None, sciOnly=True)

        # Load occulter mask from repository file.
        # Mask out occulting wedges (later replacing with NaN values).
        if self.instrument == 'stis':
            occultMask = fits.getdata(self.occultMaskPath)
        else:
            self.logger.warning("*** Occulter masks are not implemented yet "\
                  "for instruments other than STIS!!!\n")
            occultMask = np.zeros(self.imgShape)
        occultMask[occultMask < 0] = -1.e4


 # FIX ME!!! Move most of this into the data loading method??

        # Define important variable based on header info.
        # Get input image intensity units.
        self.bunit = self.allHdrs[0][1]['BUNIT']

        # Separate science frames from PSF reference frames via the target name.
        self.sciInds = np.where(np.char.lower(targs) == targ.lower())[0]
        self.refInds = np.where(np.char.lower(targs) != targ.lower())[0]
        # Number of input science and reference frames (before any combinations)
        self.nSci = np.size(self.sciInds)
        self.nRef = np.size(self.refInds)

        self.logger.info("Science image indices: {}".format(self.sciInds))
        self.logger.info("Ref image indices ({}): {}\n".format(targs[self.refInds], self.refInds))

        # Get UT observation date, proposed aperture name, orientat angles, and
        # rough star coordinates from headers.
        self.obsDate = self.allHdrs[self.sciInds[0]][0]['TDATEOBS'] # [UT]
        self.propAper = self.allHdrs[self.sciInds[0]][0]['PROPAPER']
        self.orientats = np.array([hdrs[1]['ORIENTAT'] for hdrs in self.allHdrs])
# FIX ME!!! Need to fully switch to using the self.orientats attribute.
        orientats = self.orientats

        self.exptimes_all_s = np.array([hdrs[0].get('TEXPTIME', -1) for hdrs in self.allHdrs]) # [s]
        self.exptimes_s = np.array(self.exptimes_all_s)[self.sciInds] # SCI exptimes only [s]
        self.photflam_avg = np.mean([self.allHdrs[ii][0].get('PHOTFLAM', -1.) for ii in self.sciInds]) # SCI exptimes only


    # ========== DISTORTION CORRECTION ========== #

        if self.inputType in ['flt', 'flc']:
            try:
                # Update file paths to new distortion-corrected fits files.
                # Placeholder for the CTI correction.
                if self.inputType == 'flt':
                    newSuffix = 'xft'
                elif self.inputType == 'flc':
                    newSuffix = 'xfc'
                x2dFileList = []
                for ii in range(len(self.fileList)):
                    x2dPath = os.path.join(os.path.dirname(self.fileList[ii]),
                                           os.path.basename(self.fileList[ii]).replace(self.inputType, newSuffix))
                    x2dFileList.append(x2dPath)

                # Correct images for distortion and write new x2d.fits files.
                distortion_success_list = correct_distortion(self.fileList, outputPaths=x2dFileList,
                                                             refDir=None, inst=self.instrument,
                                                             overwrite=True)
                if np.sum(distortion_success_list) == 0:
                    self.logger.error('Distortion Correction: FAILED distortion correction on all images! Proceeding without it.')
                    newSuffix = self.inputType
                    x2dFileList = self.fileList
            except:
                self.logger.exception('Distortion Correction: FAILED distortion correction! Proceeding without it.')
                newSuffix = self.inputType
                x2dFileList = self.fileList

            self.inputType = newSuffix
            self.fileList = np.asarray(x2dFileList, dtype='O')

            # # Load the distortion-corrected fits as the working images.
            # sci_data, sci_headers, targs = self.load_flt_imgs(self.fileList,
            #                                                   plot_images=False,
            #                                                   scienceOnly=True)
            # self.allHdrs = sci_headers
            # self.imgShape = np.array(sci_data[0][0].shape)

            sci_data, err_data, dq_data, dq_bool_16, dq_bool_8192, all_headers, sci_headers, \
                err_headers, dq_headers, targs = self.load_flt_imgs(self.fileList,
                                                                    plot_images=False,
                                                                    scienceOnly=False)
            self.allHdrs = all_headers
            self.imgShape = np.array(sci_data[0][0].shape)


    # ========== IMAGE ALIGNMENT (REGISTRATION) ========== #

# FIX ME!!! Could move this inside find_star.
        # Estimate star position based on headers or force it with input.
        if self.forceStar:
            self.starFromWCS = self.starToUse
        else:
            self.starFromWCS_list = []
            if self.inputType in ['flt', 'flc', 'xft', 'xfc']:
                for ii in range(len(sci_headers)):
                    priHdr = sci_headers[ii][0]
                    targRA = priHdr['RA_TARG']
                    targDec = priHdr['DEC_TARG']
                    starFromWCS_flt = []
                    for jj, hdr in enumerate(sci_headers[ii][1:]):
                        ww = wcs.WCS(hdr)
                        starFromWCS_flt.append(ww.wcs_world2pix([[targRA, targDec]], 0)[0][::-1]) # [pixels] y,x
                    self.starFromWCS_list.append(np.mean(starFromWCS_flt, axis=0))
            else:
                for ii, hdr in enumerate(imgsHdrs[1]):
                    # Get estimate of star position from target RA/Dec and WCS in header.
                    ww = wcs.WCS(hdr[1])
                    targRA = hdr[0]['RA_TARG']
                    targDec = hdr[0]['DEC_TARG']
                    self.starFromWCS_list.append(ww.wcs_world2pix([[targRA, targDec]], 0)[0][::-1]) # [pixels] y,x
            self.starFromWCS = np.mean(self.starFromWCS_list, axis=0)

# FIX ME!!! The following alignment only aligns the SCI extensions of the
# FITS cubes. This means the ERR and DQ arrays become out of alignment with
# the science images after this stage.

        # # Define common star center based on WCS estimate.
        # # This is the point to which all images will be aligned.
        # self.alignStar = np.round(self.starFromWCS, 0)

        # Align CRSPLIT exposures, if they are present.
        # Write aligned images to new alc.fits (if CTI-corrected) or alt.fits
        # files (if not CTI-corrected).
        self.starsAll = []
        self.starsOriginalAll = []
        self.alignStarOffsetsAll = []
        alignMasksAll = []
        if self.inputType in ['flt', 'flc', 'xft', 'xfc']:
            for ii in range(len(sci_data)):
                # imgsPadded = self.pad_imgs(sci_data[ii], outputShape=(1100,1100), fill=0.)

                # # Pop off "empty" images before alignment.
                # empty_to_remove = []
                # for jj in range(len(sci_data[ii])):
                #     if np.all(np.isnan(sci_data[ii][jj])):
                #         empty_to_remove.append(jj)
                # for ind in empty_to_remove[::-1]:
                #     sci_data[ii].pop(ind)

                alignImgs, alignMasks = self.align_imgs(sci_data[ii],
                                                    indImg=ii, masks=[],
                                                    commonMask=occultMask,
                                                    pad=True,
                                                    starGuess=self.starFromWCS_list[ii],
                                                    finalStarYX=np.array([1024., 1024.]))
                alignMasksAll.append(alignMasks.copy())
                self.starsAll.append(self.stars)
                self.starsOriginalAll.append(self.starsOriginal)
                self.alignStarOffsetsAll.append(self.alignStarOffsets)

                # Placeholder for the CTI correction.
                if self.inputType == 'flt':
                    newSuffix = 'aft'
                elif self.inputType == 'flc':
                    newSuffix = 'afc'
                elif self.inputType == 'xft':
                    newSuffix = 'axt'
                elif self.inputType == 'xfc':
                    newSuffix = 'axc'
                outPath = os.path.join(os.path.dirname(self.fileList[ii]),
                                       os.path.basename(self.fileList[ii]).replace(self.inputType, newSuffix))

                # Add ID suffix to output filename.
                outPath = os.path.splitext(outPath)[0] + self.cid + '.fits'

                # # Reassemble aligned CRSPLIT images into a new cube with same
                # # extensions as an flt.
                # self.write_aligned_fits(list(sum(list(zip(sci_data[ii], err_data[ii], dq_data[ii])), ())),
                #                         headers=list(sum(list(zip(sci_headers[ii][1:], err_headers[ii], dq_headers[ii])), ())),
                #                         filePath=outPath,
                #                         priHdr=sci_headers[ii][0])
                # Assembled aligned CRSPLIT images into a new cube with only
                # the science extensions.
                # Isolate the SCI extension headers first.
                aligned_sci_headers = []
                for hdr in self.allHdrs[ii]:
                    if hdr.get('EXTNAME') == 'SCI':
                        aligned_sci_headers.append(hdr)
                self.write_aligned_fits(alignImgs, headers=aligned_sci_headers,
                                        filePath=outPath,
                                        priHdr=self.allHdrs[ii][0])
                # Update the file path and inputType to the new aligned FITS cube.
                self.fileList[ii] = outPath

# FIX ME!!! Convert all alignMasks to self.alignMasks.
            alignMasks = np.array(alignMasksAll)
            self.alignMasks = alignMasks
            self.stars = np.mean(self.starsAll, axis=1)
            self.starsOriginal = np.mean(self.starsOriginalAll, axis=1)
            self.alignStarOffsets = np.mean(self.alignStarOffsetsAll, axis=1)

            self.inputType = newSuffix

# FIX ME!!! We can avoid loading from file again here, since the aligned
# images are stored in memory.
            sci_data, sci_headers, targs = self.load_flt_imgs(self.fileList,
                                                              plot_images=False,
                                                              scienceOnly=True)
            self.imgShape = np.array(sci_data[0][0].shape)
            self.workingImgs = sci_data

        else:
            self.workingImgs = sci_data
            self.stars = np.tile(self.alignStar, (sci_data.shape[0], 1))
            self.starsOriginal = []
            for hdrs in self.allHdrs:
                self.starsOriginal.append(np.array([hdrs[1]['ORIGCENY'], hdrs[1]['ORIGCENX']]))
            self.starsOriginal = np.array(self.starsOriginal)
            self.alignStarOffsets = self.stars - self.starsOriginal

            for ii in range(len(sci_data)):
                # imgsPadded = self.pad_imgs(sci_data[ii], outputShape=(1100,1100), fill=0.)
                alignMasks = shift_pix_to_pix(occultMask, self.starsOriginal[ii],
                                              finalYX=self.alignStar,
                                              outputSize=np.array([2048, 2048]),
                                              order=1, fill=-1e4)
                alignMasksAll.append(alignMasks.copy())
# FIX ME!!! Convert all alignMasks to self.alignMasks.
            alignMasks = np.array(alignMasksAll)
            self.alignMasks = alignMasks


    # ==== BACKGROUND SUBTRACTION ==== #
        # Subtract background/sky. If bgCen is not given by the info.json,
        # then a multi-region sampling of the background is performed and the
        # median of all background samples is subtracted from the image.
        if np.all(self.bgCen != -1):
            self.logger.info("SUBTRACTING BACKGROUND from all images...")
            self.subtract_background()
            self.logger.info("Background means subtracted:\n{}".format(self.bgs))
        else:
            self.logger.warning("Skipping background subtraction (bgCen is -1)\n")


    # ========== COMBINE CRSPLITS ========== #

# FIX ME!!! CRSPLIT combination should move to a method eventually.
        # Combine CRSPLITS into one "integrated" image per FITS.
# FIX ME!!! Should maybe base the logic here on the dimensions of the input
# files rather than their suffixes.
        if self.inputType in ['flt', 'flc', 'alt', 'aft', 'alc', 'axt', 'axc']:
            self.logger.info("COMBINING CRSPLITS into unified images...")
            yy, xx = np.ogrid[:sci_data[0].shape[1], :sci_data[0].shape[2]]
            for ii in tqdm(range(sci_data.shape[0]), desc="Images being unified"):
                # # Clip highest valued pixel in every CRSPLIT stack.
                # # WARNING: This distorts the bg star shapes -- don't like it
                # sci_data[ii][np.argmax(sci_data[ii], axis=0), yy, xx] = np.nan
                # Sigma clip pixels more than 5-sigma discrepant in every CRSPLIT stack.
                # No pixels actually crossed the 5-sigma threshold in the first
                # dataset tested (HD 114082 flt wedge).
                # std = np.nanstd(sci_data[ii], axis=0)
                # nsigma = np.abs((sci_data[ii] - np.nanmedian(sci_data[ii], axis=0))/std)
                # median absolute deviation = the median over the absolute
                # deviations from the median (ignoring NaN)
                mad = median_abs_deviation(sci_data[ii], axis=0,
                                           nan_policy='omit')
                nmad = np.abs((sci_data[ii] - np.nanmedian(sci_data[ii], axis=0))/mad)
                # sci_data[ii][np.where(nsigma > 5), yy, xx] = np.nan
                # sci_data[ii][nsigma > 5] = np.nan
                sci_data[ii][nmad > 5] = np.nan
            # Get an array of the number of CRSPLITS per image.
            n_crsplits = np.array([hdr[0].get('CRSPLIT') for hdr in sci_headers])
            # Average together the CRSPLITS.
            unifiedImgs = np.nanmean(sci_data, axis=1)

            for ii in range(sci_data.shape[0]):
                # Update headers for the unified images to reflect correct
                # number of combined CRSPLITS.
                for jj in range(1,len(sci_headers[ii])):
                    sci_headers[ii][jj]['NCOMBINE'] = n_crsplits[ii]
                imgsHdrs[0].append([None, unifiedImgs[ii]])
                imgsHdrs[1].append(sci_headers[ii][:2])

            del unifiedImgs # don't need this anymore

        # Zero-pad images to uniform dimensions, depending on instrument.
        if self.instrument == 'stis':
            outsize = np.array([2048, 2048])
            for ii in range(len(imgsHdrs[0])):
                im = imgsHdrs[0][ii][1]
                if im.shape != tuple(outsize):
                    imgsHdrs[0][ii][1] = zero_pad(im, outsize=outsize,
                                                  method='simple')
                    for jj, hdr in enumerate(imgsHdrs[1][ii]):
                        if 'CRPIX1' in hdr.keys():
                            imgsHdrs[1][ii][jj]['CRPIX1'] += (outsize[1] - im.shape[1])//2 # x
                            imgsHdrs[1][ii][jj]['CRPIX2'] += (outsize[0] - im.shape[0])//2 # y
            # Update header image dimensions and WCS reference coordinates.
            imgsHdrs[1] = self.update_dimensions([imgsHdrs[0][ii][1] for ii in range(len(imgsHdrs[0]))],
                                                  imgsHdrs[1])


        # Separate images into their own 3-d array. Includes Sci and Ref images.
        # Then define them as the current working images.
        self.workingImgs = np.array([imgsHdrs[0][ii][1] for ii in range(len(imgsHdrs[0]))])

        # Split out science headers for later convenience.
        self.allHdrs = imgsHdrs[1]
        self.sciHdrs = [imgsHdrs[1][ii] for ii in self.sciInds]
        self.refHdrs = [imgsHdrs[1][ii] for ii in self.refInds]

        # Optionally output the integrated images as FITS here.
        if self.saveAuxiliary:
            self.save_unified_to_fits([ii[1] for ii in imgsHdrs[0]], unit='DN',
                                      cid=self.cid)


    # ========== CALIBRATE FLUX ========== #
        # Convert intensity to counts per second.
        newUnit = 'COUNTS S-1'
        # newUnit = 'mJy arcsec-2'
        self.workingImgs = convert_intensity(self.workingImgs, imgsHdrs[1],
                                    unitEnd=newUnit,
                                    pscale=self.pscale) # [counts/s]
        self.bunit = newUnit
        self.logger.info("Converted image intensity units to {}".format(newUnit))


    # ========== AVERAGE IMAGES BY ORBIT ========== #
        # Combine individual exposures in an orbit to make one image.
        # Redefine orientats and indices based on these combined images.
        if (not noCombine) and (psfSubMode.lower() not in ['pyklip-rdi']):
            self.workingImgs, orientats = self.combineOrbitImgs(self.workingImgs, orientats)

        if self.debug:
            fig = plt.figure(3)
            for ii in range(len(self.workingImgs)):
                st = np.round(self.stars[ii]).astype(int)
                fig.clf()
                ax = plt.subplot(111)
                ax.imshow(self.workingImgs[ii][st[0]+roiY[0]:st[0]+roiY[1],
                                               st[1]+roiX[0]:st[1]+roiX[1]],
                          norm=SymLogNorm(linthresh=1., linscale=1.,
                                          vmin=0., vmax=5000.),
                          extent=[st[1]+roiX[0], st[1]+roiX[1],
                                  st[0]+roiY[0], st[0]+roiY[1]])
                ax.scatter(x=st[1], y=st[0], marker='+', s=60, c='m')
                ax.set_title("Aligned Image {}".format(ii))
                plt.draw()
                pdb.set_trace()


    # ==== CREATE IMAGE MASKS ==== #

        psfsubOnSpikesOnly = False

        self.logger.info("Making PSF subtraction masks...")
        # Make mask for occulters and diffraction spikes.
        radii = utils.make_radii(self.workingImgs[0], self.alignStar)
        masks = []
        for ii, img in enumerate(self.workingImgs):
            spikemask = utils.make_spikemask_stis(self.workingImgs[0], self.alignStar,
                                                  self.spikeAngles, width=self.spWidth)
    # FIX ME!!! Combine spikemask with occulter mask here.
            masks.append(spikemask)
        masks = np.array(masks)
        psfSubMasks = masks.copy()
        # Make a different set of masks that does not do the radial masking.
        if psfsubOnSpikesOnly:
            sourceMasks = np.zeros(masks.shape).astype(bool)
        else:
            sourceMasks = masks.copy()
        # Make special masks for background star charge bleed.
        bgStarMasks = []
        # Specialize science masks.
        for ii, ind in tqdm(enumerate(self.sciInds), desc="Science masks"):
    # FIX ME!!! May want to move this radial masking to the RDI PSF subtraction function
    # so it doesn't conflict with pyKLIP?
            # Mask out the occulted sections too, established earlier as very negative valued.
            sourceMasks[ind][alignMasks[ind] < 0] = np.nan
            # Now mask all of the excluded sources given in "exclude" json key.
            sourceMasks[ind] = mask_exclusions(mask=sourceMasks[ind],
                                       exclusions=exclusionsSci,
                                       cen=self.alignStar, cenOffset=np.zeros(2),
                                       paOffset=-1*orientats[ind], spikeAngles=self.spikeAngles)
            # Now make a new mask by folding in the radial masking, specifically for
            # PSF subtraction only.
    # TEMP!!!
            # psfSubMasks[ind][radii >= sub_r_out] = np.nan
            # psfSubMasks[ind][radii < sub_r_in] = np.nan
            # psfSubMasks[ind] += sourceMasks[ind]
            # breakpoint()

    # # TEMP!!! TEST ONLY PSF SUBTRACTING BASED ON DIFFRACTION SPIKES.
            if psfsubOnSpikesOnly:
                # psfSubMasks[ind] = ~masks[ind]
                try:
                    spikePSF_rmin, spikePSF_rmax = info[obsMode]['spikePSF_rminmax']
                    psfSubMasks[ind][radii >= spikePSF_rmax] = np.nan
                    psfSubMasks[ind][radii < spikePSF_rmin] = np.nan
                except:
                    psfSubMasks[ind][radii >= sub_r_out] = np.nan
                    psfSubMasks[ind][radii < sub_r_in] = np.nan
                psfSubMasks[ind] += sourceMasks[ind].copy()
    # # END TEMP!!! TEST ONLY PSF SUBTRACTING BASED ON DIFFRACTION SPIKES.
                sourceMasks[ind] += masks[ind].copy()
            else:
                psfSubMasks[ind][radii >= sub_r_out] = np.nan
                psfSubMasks[ind][radii < sub_r_in] = np.nan
                psfSubMasks[ind] += sourceMasks[ind].copy()

            # Add off-axis diffraction spike masks to the other masks.
            # IMPORTANT: cen here is in the padded image coordinate frame-- NOT the original.
            for excl in exclusionsSci.setdefault('spikes_yxr_anglesDeg', []):
                maskSpikesOffaxis = mask_spikes_offaxis(np.zeros(alignMasks[ind].shape),
                                                    excl,
                                                    cen=self.alignStar,
                                                    cenOffset=None,
                                                    paOffset=-1*orientats[ind],
                                                    spikeAngles=excl[3])
                maskSpikesOffaxis *= -1e4
                masks[ind][maskSpikesOffaxis < 0] = True
                alignMasks[ind] += maskSpikesOffaxis
                psfSubMasks[ind][maskSpikesOffaxis < 0] = True
                sourceMasks[ind][maskSpikesOffaxis < 0] = True

            if self.debug:
                plt.figure(4)
                plt.clf()
                plt.imshow(alignMasks[ind])
                plt.title("Occulter ('align') mask")
                plt.draw()
                plt.show()

                pdb.set_trace(header=f"DEBUG: Science masking for ii={ii}, image ind={ind}")

        # Specialize reference masks. Default radius masks to match science masks.
        exclusionsRef = exclusions.setdefault('ref', {})
        # if exclusionsRef.get('r_out') is None:
        #   exclusionsRef['r_out'] = exclusionsSci.get('r_out')
        # if exclusionsRef.get('r_in') is None:
        #   exclusionsRef['r_in'] = exclusionsSci.get('r_in')
        if self.psfSubMode in ['rdi']:
            for ii, ind in tqdm(enumerate(self.refInds), desc="Reference masks"):
        # TEMP!!! TEST ONLY PSF SUBTRACTING BASED ON DIFFRACTION SPIKES.
                if psfsubOnSpikesOnly:
                    psfSubMasks[ind] = np.zeros(psfSubMasks[ind].shape).astype(bool)

                psfSubMasks[ind][alignMasks[ind] < 0] = np.nan
                # Don't offset PA of reference mask because coords should
                # already be given in rotated frame.
                # Always offset the coordinates based on the new aligned star
                # center for reference images because those mask coordinates
                # are only ever measured from the raw images.
                # Special case of HD 106906 wedgeb1.8 has two dither positions
                # on same wedge, so handle that offset correctly.
                if (targ == 'HD-106906') and (obsMode == 'wedgeb1.8'):
                    psfSubMasks[ind] = mask_exclusions(mask=psfSubMasks[ind],
                                                exclusions=exclusionsRef,
                                                cen=self.alignStar, cenOffset=self.alignStarOffsets[self.refInds[0]],
                                                paOffset=0, spikeAngles=self.spikeAngles)
                else:
                    psfSubMasks[ind] = mask_exclusions(mask=psfSubMasks[ind],
                                                exclusions=exclusionsRef,
                                                cen=self.alignStar, cenOffset=self.alignStarOffsets[ind],
                                                paOffset=0, spikeAngles=self.spikeAngles)
        # ADI and all other non-RDI cases.
        else:
            psfSubMasks_refs = alignMasks.copy()
            # Lazy bookkeeping to convert occulter spike mask to boolean,
            # where masked elements are set to True.
            psfSubMasks_refs[psfSubMasks_refs >= 0] = 0
            psfSubMasks_refs[psfSubMasks_refs < 0] = 1
            psfSubMasks_refs = psfSubMasks_refs.astype(bool)
            # Add in diffraction spike masks.
            # psfSubMasks_refs = psfSubMasks_refs + (-1*masks.astype(int))
            psfSubMasks_refs = psfSubMasks_refs + masks
            for ii, ind in tqdm(enumerate(self.sciInds), desc="Reference masks"):
                # Don't offset PA of reference mask because coords should
                # already be given in rotated frame.
                # Always offset the coordinates based on the new aligned star
                # center for reference images because those mask coordinates
                # are only ever measured from the 1100x1100 xfc or sx2 images.
                psfSubMasks_refs[ind] = mask_exclusions(mask=psfSubMasks_refs[ind],
                                            exclusions=exclusionsRef,
                                            cen=self.alignStar, cenOffset=self.alignStarOffsets[ind],
                                            paOffset=0, spikeAngles=self.spikeAngles)

        if self.debug:
            for ii in range(len(psfSubMasks)):
                plt.figure(5)
                plt.clf()
                plt.imshow(psfSubMasks[ii])
                plt.title(f"PSF Subtraction mask (1=masked, 0=not masked): img {ii}")

                plt.figure(6)
                plt.clf()
                plt.imshow(self.workingImgs[ii],
                            norm=SymLogNorm(linthresh=0.01, linscale=1,
                                            vmin=0, vmax=100))
                plt.title(f"Aligned image (no mask shown): img {ii}")

                plt.figure(7)
                plt.clf()
                plt.imshow(self.workingImgs[ii] * ~psfSubMasks[ii],
                            norm=SymLogNorm(linthresh=0.01, linscale=1,
                                            vmin=0, vmax=100))
                plt.title(f"Aligned image with PSF Subtraction mask: img {ii}")
                plt.draw()
                plt.show()

                pdb.set_trace(header=f"DEBUG: PSF subtraction masks for ii={ii}")


    # ======== PSF SUBTRACTION ======== #

        # Basic RDI PSF subtraction.
        if psfSubMode.lower() == 'rdi':
            self.logger.info("Performing RDI PSF subtraction...")
            rmin = 1 # PSFsub masking supercedes this value.
            getRadProf = info[obsMode].get('radProfSub')
            if getRadProf and (getRadProf is not None):
                radProfPaList = info[obsMode]['radProfSub'].setdefault('paList', [0])
                if radProfPaList is not None:
                    radProfPaList = np.array(radProfPaList)
                    self.subRadProf = True
                else:
                    self.subRadProf = False
                radProfPaHW = info[obsMode]['radProfSub'].setdefault('paHW', 50)
                radProfMax = info[obsMode]['radProfSub'].setdefault('rMax', 200)
            else:
                radProfPaList = None
                radProfPaHW = None
                radProfMax = None
                self.subRadProf = False

            # Estimate initial brightness scaling for reference PSFs.
            # Make a set of combination PSF subtraction masks that are the
            # science masks + all (logical or) ref masks.
            # Apply those combo masks to both science and ref images, then
            # take their ratios and sum to get the approximate ratio of
            # PSF brightnesses between ref and science images.
            comboRefSciMasks = psfSubMasks[self.sciInds] | np.any(psfSubMasks[self.refInds], axis=0)
            ratioSums_ref_sci = []
            for ii, ind in enumerate(self.sciInds):
                ratio_ref_sci_ii = np.nansum(np.nanmedian(self.workingImgs[self.refInds] * ~comboRefSciMasks[ii], axis=0)) / np.nansum(self.workingImgs[ind] * ~comboRefSciMasks[ii])
                ratioSums_ref_sci.append(np.round(np.nansum(ratio_ref_sci_ii), 4))
            self.logger.info("Ratios of reference PSF to science PSF by "\
                             f"science image: {ratioSums_ref_sci}")
            ratio_ref_sci = np.nanmedian(ratioSums_ref_sci)
            self.logger.info("Median ratio of reference PSF to science PSF "\
                             f"brightness = {ratio_ref_sci:.3f}")
            # Don't let ratio stray too far from unity.
            if (ratio_ref_sci < 0.05) or (ratio_ref_sci > 20):
                ratio_ref_sci = 1.
            self.logger.info(f"Using {1/ratio_ref_sci:.3f} (1/{ratio_ref_sci:.3f}) as initial reference PSF "\
                             "scaling ratio\n")
            C0 = np.log10(1/ratio_ref_sci)

            # Do the PSF subtraction.
            psfSubImgs, psfSubImgs_subRadProf, refScaleFactors = stis_psfsub.rdi_subtract_psf(
                                    self.workingImgs[self.sciInds],
                                    self.workingImgs[self.refInds],
                                    psfSubMasks[self.sciInds],
                                    psfSubMasks[self.refInds],
                                    self.stars[self.sciInds], C0=C0,
                                    rmin=rmin, rmax=sub_r_out, ann=ann,
                                    deltaPAMin=self.deltaPAMin,
                                    orientats=orientats[self.sciInds],
                                    radProfPaList=radProfPaList,
                                    radProfPaHW=radProfPaHW,
                                    radProfMax=radProfMax,
                                    radProfMasks=sourceMasks[self.sciInds],
                                    subRadProf=self.subRadProf,
                                    bgCen=self.bgCen, bgRadius=self.bgRadius,
                                    optimize_dither=True)

            self.psfSubImgs = psfSubImgs
            self.psfSubImgs_subRadProf = psfSubImgs_subRadProf
            self.refScaleFactors = refScaleFactors
            self.logger.info(f"Ref scale factors by science image: {refScaleFactors}")

        # Basic ADI PSF subtraction.
        elif psfSubMode.lower() == 'adi':
            self.logger.info("Performing ADI PSF subtraction...")
            rmin = 1 # PSFsub masking supercedes this value.
            getRadProf = info[obsMode].get('radProfSub')
            if getRadProf and (getRadProf is not None):
                radProfPaList = info[obsMode]['radProfSub'].setdefault('paList', [0])
                if radProfPaList is not None:
                    radProfPaList = np.array(radProfPaList)
                    self.subRadProf = True
                else:
                    self.subRadProf = False
                radProfPaHW = info[obsMode]['radProfSub'].setdefault('paHW', 50)
                radProfMax = info[obsMode]['radProfSub'].setdefault('rMax', 200)
            else:
                radProfPaList = None
                radProfPaHW = None
                radProfMax = None
                self.subRadProf = False

            psfSubImgs, refScaleFactors = stis_psfsub.adi_subtract_psf(
                                    self.workingImgs[self.sciInds],
                                    self.workingImgs[self.sciInds],
                                    psfSubMasks[self.sciInds],
                                    psfSubMasks_refs,
                                    self.stars[self.sciInds], self.deltaPAMin,
                                    self.orientats[self.sciInds],
                                    C0=0., rmin=rmin, rmax=sub_r_out, ann=ann,
                                    radProfPaList=radProfPaList,
                                    radProfPaHW=radProfPaHW,
                                    radProfMax=radProfMax,
                                    radProfMasks=sourceMasks[self.sciInds],
                                    subRadProf=self.subRadProf,
                                    bgCen=self.bgCen, bgRadius=self.bgRadius,
                                    optimize_dither=False)
            self.psfSubImgs = psfSubImgs
            self.refScaleFactors = refScaleFactors

        # PyKLIP RDI subtraction.
        elif psfSubMode.lower() == 'pyklip-rdi':
            self.logger.info("Performing pyKLIP RDI PSF subtraction...")
            IWA = 3.
            OWA = 200.
            # Mask out the occulters in the input images.
            alignImgsMasked = self.workingImgs + alignMasks
            alignImgsMasked[alignMasks < 0] = np.nan

            stis_psfsub.do_klip_stis(targ, self.fileList[self.sciInds],
                inputImgs=alignImgsMasked[self.sciInds], inputHdrs=None,
                psfPaths=self.fileList, psfImgs=alignImgsMasked, mode='RDI',
                ann=int(OWA-IWA), subs=1, minrot=0, mvmt=0, IWA=IWA, OWA=OWA,
                numbasis=[1,2,3,5,15], maxnumbasis=None, star=self.stars,
                highpass=False, pre_sm=None, spWidth=8., ps_spWidth=0., PAadj=0.,
                parangs=None, aligned_center=self.alignStar[::-1],
                collapse="mean", prfx=targ,
                cubes=False, save_aligned=False, restored_aligned=None,
                lite=False, do_snmap=False, numthreads=4, sufx='', output=False,
                compute_correlation=True)

            self.psfSubImgs = []
            self.psfSubImgs_subRadProf = []
            self.refScaleFactors = None

        # PyKLIP ADI subtraction.
        elif psfSubMode.lower() == 'pyklip-adi':
            self.logger.info("Performing pyKLIP ADI PSF subtraction...")
            stis_psfsub.do_klip_stis(targ, self.fileList[self.sciInds],
                inputImgs=self.workingImgs[self.sciInds], inputHdrs=None,
                psfPaths=None, psfImgs=None, mode='ADI', 
                ann=1, subs=1, minrot=0, mvmt=5, IWA=10., OWA=200.,
                numbasis=[1,2,3,5,10,15], maxnumbasis=None, star=self.stars[self.sciInds],
                highpass=False, pre_sm=None, spWidth=8., ps_spWidth=0., PAadj=0.,
                parangs=orientats[self.sciInds], aligned_center=self.alignStar[::-1],
                collapse="mean", prfx=targ,
                save_psf_cubes=False, save_aligned=False, restored_aligned=None,
                lite=True, do_snmap=False, numthreads=4, sufx='', output=False,
                compute_correlation=False)

            self.psfSubImgs = []
            self.psfSubImgs_subRadProf = []
            self.refScaleFactors = None

        # Optionally save the individual PSF-subtracted images as a FITS cube.
        if self.saveAuxiliary:
            self.save_psfsub_to_fits(self.psfSubImgs, 'psfcube', unit=newUnit,
                                     cid=self.cid)

        # Do another background subtraction before combination??
    # SKIP -- for now.


        # Perform astrometry on occulted primary using background star
        # Gaia positions.
        if self.do_gaia:
            self.logger.info("*** Running GAIA ASTROMETRY thingy ***\n")
            self.targSimbad = utils.format_target_name(self.targ)
            gaiaID = get_gaia_id(self.targSimbad)
            if gaiaID is not None:
                gaiaIDNumber = int(gaiaID.split(' ')[-1])
                # target_gaia_id = 6055854551117476480 #6072902994276659200 is HD-106906
                target_rv = [0., 0.]
                all_gaia_out = []
                for ii, img in enumerate(self.psfSubImgs):
                    # Star input order is x,y
                    # Output order is:
                    # final_x_median, final_y_median, final_ps_x_median, final_ps_y_median,
                    # final_tn_median, final_x_std, final_y_std, final_ps_x_std, final_ps_y_std,
                    # final_tn_std
                    gaia_out = main(f"{self.targ}_{self.obsMode}_{self.inputType}_{ii}",
                                    self.instrument, gaiaIDNumber, target_rv,
                                    self.stars[self.sciInds][ii][::-1], gaia_catalogue='DR3',
                                    exclude_extra=[], im=img, hdr=self.sciHdrs[ii][1],
                                    out_dir=self.dataDir)
                    all_gaia_out.append(gaia_out)
                    self.logger.info(f"Gaia X: {gaia_out[0]:.2f} +/- {gaia_out[5]:.2f}, Gaia Y median: {gaia_out[1]:.2f} +/- {gaia_out[6]:.2f}")
            else:
                self.logger.error("*** FAILED to measure Gaia astrometry: "\
                      f"invalid Gaia ID number for target ({gaiaIDNumber})\n")

        if psfSubMode.lower() in ["adi", "rdi"]:
            # Apply mask for occulters and diffraction spikes to
            # PSF-subtracted images before the final collapse.
            for ii in range(len(psfSubImgs)):
                aM = alignMasks[self.sciInds][ii].copy()
                if len(bgStarMasks) > 0:
                    aM[bgStarMasks[ii]] = -1
                aM[aM >= 0] = 0
                aM[aM < 0] = 1
                aM = aM.astype(bool)
                self.psfSubImgs[ii][masks[self.sciInds][ii] + aM] = np.nan
                try:
                    self.psfSubImgs_subRadProf[ii][masks[self.sciInds][ii] + aM] = np.nan
                except:
                    self.logger.warning("*** FAILED to mask radial profile-subtracted PSF-subtracted images. Proceeding.")
                    pass

            # Derotate and combined PSF-subtracted images.
            ("Derotating PSF-subtracted images...")
            rotImgs = self.derotate(self.psfSubImgs, orientats[self.sciInds],
                                    self.stars[self.sciInds])
            rotImgs_subRadProf = self.derotate(self.psfSubImgs_subRadProf,
                                               orientats[self.sciInds],
                                               self.stars[self.sciInds])

            # # Also make a copy with images derotated in the wrong direction.
            # # Make the widest deltaPA = 90 deg from being fully aligned.
            # spreadPAs = orientats[self.sciInds].copy()
            # spreadPAs[1:] -= np.arange(1, len(self.sciInds))*(90/(len(self.sciInds)-1))
            # rotImgsBkwd = self.derotate(self.psfSubImgs, spreadPAs,
            #                             self.stars[self.sciInds])


            # Optimize combination based on background levels, if needed.
            self.logger.info("Combining derotated images...")
            finalImg = np.nanmean(rotImgs, axis=0)
            finalImg_subRadProf = np.nanmean(rotImgs_subRadProf, axis=0)
            # bkwdImg = np.nanmedian(rotImgsBkwd, axis=0)

            # Subtract radial profile from final combined image to remove
            # residual halo.
            if self.subFinalRadProf:
                tmpImg = finalImg_subRadProf.copy()
                # tmpImg = bkwdImg.copy()
                # Remove disk mask from exclusions.
                exclusionsFinal = exclusionsSci.copy()
                exclusionsFinal['rect_cenYX_widthYX_angleDeg'] = []
                finalMask = np.zeros(tmpImg.shape, dtype=bool)
                finalMask = mask_exclusions(mask=finalMask,
                                   exclusions=exclusionsFinal,
                                   cen=self.alignStar, cenOffset=np.zeros(2),
                                   paOffset=0, spikeAngles=self.spikeAngles)
                tmpImg[finalMask.astype(bool)] = np.nan
                meanRadProf = stis_psfsub.measure_mean_radial_prof(tmpImg, self.alignStar,
                                                       paList=info[obsMode]['radProfSub']['paList'],
                                                       paHW=info[obsMode]['radProfSub']['paHW'],
                                                       rMax=info[obsMode]['radProfSub']['rMax'])
                meanRadProf = np.nan_to_num(meanRadProf, 0)
                # Keep the "pure" final image pure (no radial profile
                # subtraction at all), still defined as finalImg.
                # Keep the final image with radial profile subtraction only
                # done during the PSF subtraction phase.
                finalImg_noFinalProfSub = finalImg_subRadProf
                # Subtract the final radial profile from the final image
                # that was already radial profile-subtracted during the
                # PSF subtraction phase.
                finalImg_finalProfSub = finalImg_subRadProf - meanRadProf


                if self.debug:
                    plt.figure(15)
                    plt.clf()
                    plt.imshow(finalMask)
                    plt.title("Mask for post-combine\nradial profile measurement")

                    plt.figure(16)
                    plt.clf()
                    plt.imshow(tmpImg,
                               norm=SymLogNorm(linthresh=0.01, linscale=1,
                                               vmin=-0.1, vmax=10))
                    plt.title("Post-combine image masked\nfor radial profile measurement")

                    plt.figure(17)
                    plt.clf()
                    plt.imshow(meanRadProf,
                               norm=SymLogNorm(linthresh=0.01, linscale=1,
                                               vmin=-0.1, vmax=10))
                    plt.title("Post-combine radial profile measured")

                    pdb.set_trace()

                del tmpImg

            else:
                finalImg_noFinalProfSub = None

            # Do one final background subtraction.
            if info[obsMode].get('bgCenFinal_yx') is not None:
                finalImg, bgFinal = utils.subtract_bg(finalImg, np.array(info[obsMode].get('bgCenFinal_yx').split(' '), dtype=float), self.bgRadius)
                finalImg_noFinalProfSub, bgFinal = utils.subtract_bg(finalImg_noFinalProfSub, np.array(info[obsMode].get('bgCenFinal_yx').split(' '), dtype=float), self.bgRadius)
                finalImg_finalProfSub, bgFinal = utils.subtract_bg(finalImg_finalProfSub, np.array(info[obsMode].get('bgCenFinal_yx').split(' '), dtype=float), self.bgRadius)

            # Write the final combined PSF-subtracted image(s) to FITS file.
            if self.saveFinal:
                if finalImg_noFinalProfSub is None:
                    self.save_psfsub_to_fits(finalImg, 'final',
                                             unit=newUnit, cid=self.cid)
                else:
                    # Index order: 0 = all subtractions;
                    # 1 = no final radial profile subtraction
                    # 2 = no radial profile subtracts at all
                    self.save_psfsub_to_fits(np.array([finalImg_finalProfSub,
                                                       finalImg_noFinalProfSub,
                                                       finalImg]),
                                             'final', unit=newUnit,
                                             cid=self.cid)

# FIX ME!!! Compute error maps for each version of the final image.
# Currently only compute for the version with no radial profile subtractions at all.
            # Compute error maps.
            if not self.noErrorMaps:
                self.logger.info("Computing ERROR MAPS...")
                rotImgs_electrons = convert_intensity(rotImgs.copy(),
                                                      self.sciHdrs,
                                                      unitStart='counts s-1',
                                                      unitEnd='e-',
                                                      pscale=self.pscale)
                sourceMaskDerot = mask_exclusions(mask=spikemask,
                                           exclusions=exclusionsSci,
                                           cen=self.alignStar,
                                           cenOffset=np.zeros(2), paOffset=0,
                                           spikeAngles=self.spikeAngles)
                sourceMaskDerot[sourceMaskDerot == 1] = np.nan

                # Sample the noise around the disk minor axis on its back edge.
                # Define the angles for this region.
                stdPACen = info['diskPA_deg'] + 90.
                if stdPACen < 0:
                    stdPACen += 360.
                stdPAMin = stdPACen - 60.
                if stdPAMin < 0:
                    stdPAMin += 360.
                    stdPARange1 = [stdPAMin, 360]
                else:
                    stdPARange1 = [stdPAMin, stdPACen]
                stdPAMax = stdPACen + 60.
                if stdPAMax > 360:
                    stdPAMax -= 360.
                    stdPARange2 = [0, stdPAMax]
                else:
                    stdPARange2 = [stdPACen, stdPAMax]
                if stdPAMin > stdPAMax:
                    stdPARange2 = [0, stdPAMax]
                stdPARange = stdPARange1 + stdPARange2

                errorMaps_electrons = []
                for ii in tqdm(range(len(rotImgs)), desc="Error maps"):
                    poissonMap = np.sqrt(np.abs(rotImgs_electrons[ii]))
                    theta = utils.make_phi(rotImgs_electrons[ii], self.alignStar,
                                           zeroAxis='+y')
                    # Get background noise map as std deviations of annuli.
                    stdMap = utils.get_partialann_stdmap(rotImgs_electrons[ii]+sourceMaskDerot,
                                        self.alignStar, radii, theta, stdPARange,
                                        r_max=400, rdelta=3)
                    # Check if std deviation map is all NaN (i.e., the sampled
                    # region of the image was masked). If so, use the entire
                    # image for sampling as a not-so-great backup option.
                    if np.all(np.isnan(stdMap[radii < 400])):
                        self.logger.warning("*** Could not sample errors "\
                              f"within PA range(s) {stdPARange} because image"\
                              " was fully masked in that region. Defaulting "\
                              "to sampling errors from unmasked regions at "\
                              "all PA's.\n")
                        stdMap = utils.get_partialann_stdmap(rotImgs_electrons[ii]+sourceMaskDerot,
                                            self.alignStar, radii, theta, [-1, 361],
                                            r_max=400, rdelta=3)
                    # Extrapolate background noise into inner regions where we
                    # don't sample it well or at all but we do have data.
                    stdStrip1 = gaussian_filter(stdMap[int(self.alignStar[0]),
                                                       int(self.alignStar[1]-70):int(self.alignStar[1])], 3)
                    stdStrip2 = gaussian_filter(stdMap[int(self.alignStar[0]),
                                                       int(self.alignStar[1]+1):int(self.alignStar[1]+1+70)], 3)
                    stdStrip3 = gaussian_filter(stdMap[int(self.alignStar[0]-70):int(self.alignStar[0]),
                                                       int(self.alignStar[1])], 3)
                    stdStrip4 = gaussian_filter(stdMap[int(self.alignStar[0]+1):int(self.alignStar[0]+1+70),
                                                       int(self.alignStar[1])], 3)
                    stdStrip = np.nanmean([stdStrip1, stdStrip2[::-1], stdStrip3, stdStrip4[::-1]], axis=0)

                    fe = interp1d(np.arange(self.alignStar[1]-70, self.alignStar[1])[~np.isnan(stdStrip)],
                                  gaussian_filter(stdMap[int(self.alignStar[0]), int(self.alignStar[1]-70):int(self.alignStar[1])], 3)[~np.isnan(stdStrip)],
                                  kind='linear', fill_value="extrapolate")
                    interpStd = fe(np.arange(self.alignStar[1]-70, self.alignStar[1]))
                    for rr in range(1, 30): #len(interpStd)):
                        try:
                            stdMap[(radii >= rr - 0.5) & (radii < rr + 0.5)] = interpStd[::-1][rr]
                        except:
                            stdMap[(radii >= rr - 0.5) & (radii < rr + 0.5)] = np.nan
                    # Combine noise terms.
                    errorMaps_electrons.append(np.sqrt(np.nansum([poissonMap**2, stdMap**2], axis=0)))
                
                # Fractional error maps.
                fracerr = np.array(errorMaps_electrons)/rotImgs_electrons
                # Error maps in units of counts per second.
                errorMaps = fracerr*rotImgs # [counts s-1]
                # Number of non-NaN values in each pixel across all PSF-subtracted images.
                countAveragedPix = len(errorMaps) - np.sum(np.isnan(errorMaps), axis=0)
                
                # Propagated standard error of the mean.
                finalErrorMap = np.sqrt(np.nansum(errorMaps**2, axis=0))/np.sqrt(countAveragedPix)
                
                finalSNR = finalImg/finalErrorMap # [SNR]
                finalSNR_noFinalProfSub = finalImg_noFinalProfSub/finalErrorMap # [SNR]
                finalSNR_finalProfSub = finalImg_finalProfSub/finalErrorMap # [SNR]

                if self.saveFinal:
                    # Save the error map and SNR map too.
                    try:
                        self.save_psfsub_to_fits(finalErrorMap, 'error',
                                                 unit=newUnit, cid=self.cid)
                    except:
                        self.logger.error("*** FAILED to save Error map!!!\n")
                    try:
                        self.save_psfsub_to_fits([finalSNR_finalProfSub,
                                                  finalSNR_noFinalProfSub,
                                                  finalSNR],
                                                 'snr',
                                                 unit=newUnit, cid=self.cid)
                    except:
                        self.logger.error("*** FAILED to save SNR map!!!\n")


        try:
            self.logger.info("Running post-reduction analysis...")
            self.post_reduction_analysis(finalImg)
        except:
            self.logger.error("*** FAILED post-reduction analysis\n")

        self.logger.info(" ***  FIN  ***\n")
