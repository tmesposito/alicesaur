#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:51:08 2024

To do Charge Transfer Inefficiency correction on HST images.

Based originally on STScI's tutorial notebook at https://github.com/spacetelescope/STIS-Notebooks/tree/main/drizpac_notebook
Based afterwards on modifications by Bin Ren <>.

"""


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Load packages
import os
import sys
import glob
import shutil
import subprocess
import numpy as np
from tempfile import TemporaryDirectory
from astropy.io import fits
from astropy.table import vstack

# STIS packages
from stis_cti import stis_cti, archive_dark_query
from stis_cti import __version__ as stis_cti_version

# STScI packages
# from drizzlepac import tweakreg
# from drizzlepac import astrodrizzle as ad
from astroquery.mast import Observations

# Custom package, copy_files.py should be in the same directory
import copy_files as cf

from alicesaur.utils import check_mkdir, set_up_logger


class CTI():
    """
    Class for Charge-Transfer Inefficiency correction.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor of the CTI class.
        """

        # # Check astroquery Observations options
        # Observations.get_metadata("observations")
        # Observations.get_metadata("products")

        self.obsMode = None
        self.loggerName = None
        self.loginToken = None

        # Define attributes with kwargs items
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Log output if possible, otherwise just print it.
        self.logger = set_up_logger(loggerName=self.loggerName)
        # if not hasattr(self, 'loggerName'):
        #     self.logger = None

        self.logger.debug("CTI correction: Created CTI correction object")

        self.logger.info(f'CTI correction: stis_cti v{stis_cti_version}')


    def setup_directories(self, base_dir=None):

        # Get the current working directory to handle relative paths
        cwd = os.getcwd()

        if base_dir is None:
            self.base_dir = os.path.abspath(cwd)
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.logger.info(f"CTI correction: Creating CTI directories inside {self.base_dir}")

        # Set temporary data download store
        self.cache_dir = os.path.join(self.base_dir, 'data_cache')

        # Set the reference file directory
        self.ref_dir = os.path.join(self.base_dir, 'reference_files')

        # Set root and data directories
        self.root_dir = os.path.join(self.base_dir, 'drizpac')
        self.dat_dir = os.path.join(self.root_dir, 'data')
        self.ccd_dir = os.path.join(self.dat_dir, 'ccd_data')
        # mama_dir = os.path.join(dat_dir, 'mama_data')

        # Set directories for CTI correction of CCD data
        self.cti_dir = os.path.join(self.dat_dir, 'ccd_cti')
        self.science = os.path.join(self.cti_dir, 'science')
        self.darks = os.path.join(self.cti_dir, 'darks')
        self.ref_cti = os.path.join(self.cti_dir, 'ref')

        # Set directories for alignment with tweakreg
        self.ccd_twk = os.path.join(self.dat_dir, 'ccd_twk')
        # nuv_twk = os.path.join(dat_dir, 'nuv_twk')
        # fuv_twk = os.path.join(dat_dir, 'fuv_twk')

        # Set directories for drizzling with astrodrizzle
        self.ccd_drz = os.path.join(self.dat_dir, 'ccd_drz')
        # nuv_drz = os.path.join(dat_dir, 'nuv_drz')
        # fuv_drz = os.path.join(dat_dir, 'fuv_drz')
        # Check for directories and make them if they don't exist
        # for d in [cache_dir, ref_dir, root_dir, dat_dir, ccd_dir, mama_dir, cti_dir, science, darks, ref_cti, ccd_twk, fuv_twk, nuv_twk, ccd_drz, fuv_drz, nuv_drz]:
        for d in [self.cache_dir, self.ref_dir, self.root_dir, self.dat_dir, self.ccd_dir, self.cti_dir, self.science, self.darks, self.ref_cti, self.ccd_twk, self.ccd_drz]:
            check_mkdir(d)

        return


    def setup_env(self):

        # Set environment variables
        os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
        os.environ['CRDS_PATH'] = os.path.abspath(self.ref_dir)
        os.environ['oref'] = os.path.join(os.path.abspath(self.ref_dir), 'references', 'hst', 'stis') + os.path.sep   # Trailing slash important

        self.logger.info("CTI correction: Set CTI environment variables")
        for envvar in ['CRDS_SERVER_URL', 'CRDS_PATH', 'oref']:
            self.logger.debug(f"CTI correction: {envvar} = {os.environ[envvar]}")

        return


# FIX ME!!! Handle case of same target being observed in multiple blocks
# separated by significant time within a single PID. Could take an input date
# to specify which block to fetch, otherwise all will get rolled together.
    def get_oids(self, pids, target_name='HD-114082'):
        """
        Get observation IDs from proposal IDs.

        From a list of input proposal IDs (``pids``) get a list of observation IDs 
        (``oids``) using astroquery.

        Parameters
        ----------
        pids : list or array_like of str or int
            List or array of proposal IDs to find observation IDs for

        Returns
        -------
        oids : list of str
            List of observation IDs within the input proposal IDs

        """

        oids = []

        # For each PID get obs IDs
        for pid in pids:
            # Only get observations from the given Proposal ID taken with
            # STIS/CCD in MIRVIS filter that are "image" type and calibrated
            # to a certain level (3).
            obs = Observations.query_criteria(proposal_id='{}'.format(pid),
                                              instrument_name='STIS/CCD',
                                              filters='MIRVIS',
                                              dataproduct_type='image',
                                              calib_level=3)
            # obs columns: obs_id is filename identifier, whereas obsid is a
            # numerical id roughly (but not necessarily) in order of execution.
            # calib_level is 3 for normal images, t_min and t_max are start and
            # end time of exposure.
            # obs columns to ignore: em_min and em_max seem useless (all NaN),
            # dataproduct_type (already specified),
            # t_min, obs_collection (all HST), provenance_name (all CALSTIS),
            # proposal id (we already specified), t_obs_release (don't care),
            # obs_title (don't care), s_region, sequence_number (all masked),
            # srcDen, 
            # Sort observations by t_min to roughly sort by date recorded.
            obs.sort('t_min')
            # Get the list of data products associated with the obs.
            products = Observations.get_product_list(obs)

# FIX ME!!! Figure out how to log this table output nicely.
            # Print the table of observations found.
            print("\nObservations found:")
            obs['intentType', 'target_name', 'target_classification', 't_min',
                'obs_id', 't_exptime'].pprint(max_lines=100, max_width=200)

            if len(obs) == 0:
                self.logger.error("*** NO OBSERVATIONS FOUND in archive with" \
                                  f" program ID {pid}. Check your program ID."\
                                  " Aborting.")
                return [], [], None

            # Separate out obs for only PSF reference stars.
            # First grab anything with both 'PSF' and the science target's name
            # in the obs target name.
            if target_name is not None:
                PSFrefs_by_name = np.array([(('PSF' in tn) and ((target_name in tn) | (target_name.replace('-', '') in tn))) for tn in obs['target_name'].data])
            else:
                PSFrefs_by_name = np.array(len(obs)*[False])
            # If none found like that, expand to just 'PSF' in the target name.
            if np.sum(PSFrefs_by_name) == 0:
                PSFrefs_by_name = np.array(['PSF' in tn for tn in obs['target_name'].data])
                # Then grab anything with a specific target classification.
                obs_calPSF = obs[(obs['target_classification'] == 'CALIBRATION;POINT SPREAD FUNCTION') |
                                 PSFrefs_by_name]
            else:
                obs_calPSF = obs[PSFrefs_by_name]
            # obs_calPSF['instrument_name', 'filters', 'wavelength_region', 'target_name', 'target_classification', 'obs_id', 'obsid'].pprint(max_width=-1)

            if target_name is not None:
                obs_target = obs[obs['target_name'] == target_name]
                # Get obs_id's for this target's images only.
                oids_for_target = list(np.unique(obs_target['obs_id']))

                # Abort if no observations found with the target name.
                if len(obs_target) == 0:
                    self.logger.error("*** NO OBSERVATIONS FOUND in archive with" \
                                      f" program ID {pid} AND target name {target_name}."\
                                      "Check both inputs before trying again."\
                                      " Aborting.")
                    unique_targets = list(np.unique(obs['target_name']))
                    self.logger.info(f"Target names found for program ID {pid}: {unique_targets}")
                    return [], [], None

                t_start_target = min(obs_target['t_min'])
                t_end_target = max(obs_target['t_max'])

                # Locate PSF reference stars observed in the middle of the target obs.
                # If none, look immediately before and after the target obs.
                # Make sure to match the camera settings.
                obs_calPSF_match = obs_calPSF[(obs_calPSF['t_min'] >= t_start_target) & (obs_calPSF['t_min'] <= t_end_target)]
                self.logger.info(f"Data Fetch: Found {len(obs_calPSF_match)} clearly labeled PSF reference star image(s) amid science images")
                if len(obs_calPSF_match) > 0:
                    self.logger.info(f"Data Fetch: PSF reference stars found: {list(np.unique(obs_calPSF_match['target_name']))}")
                # Get obs_id's for the PSF reference images.
                oids_for_refs = list(np.unique(obs_calPSF_match['obs_id']))
                obs_calPSF = vstack([obs_calPSF, obs_calPSF_match])
                # Combine all target and ref oids into one list.
                oids = oids_for_target
                oids += oids_for_refs
# FIX ME!!! This method will grab too many non-PSF reference images if there
# are multiple chunks of target images spread out in time.
                # Try to identify reference images by timing if they are not
                # labeled via target_classification.
                # Assume any target observed between the first and last target
                # image are for a related PSF reference star.
                if len(oids_for_refs) == 0:
                    obs_amidst = obs[(obs['t_min'] >= t_start_target) & (obs['t_max'] <= t_end_target)]
                    unique_target_names = np.unique(obs_amidst['target_name'])
                    assumed_ref_names = unique_target_names[unique_target_names != target_name]
                    ref_match = [ob in assumed_ref_names for ob in obs_amidst['target_name']]
                    obs_ref_match = obs_amidst[ref_match]
                    oids_for_refs_more = list(np.unique(obs_ref_match['obs_id']))
                    oids_for_refs += oids_for_refs_more
                    oids += oids_for_refs_more
                    obs_calPSF = vstack([obs_calPSF, obs_ref_match])
                    # obs_ref_match = obs_amidst[obs_amidst['target_name'] in ]
                # If still no PSF references were identified, look immediately
                # before and after science target images (within 3 hours).
                if len(oids_for_refs) == 0:
                    # Limit search to 3 hours before and after the science images.
                    obs_adjacent = obs[((obs['t_min'] < t_start_target) & (obs['t_min'] >= t_start_target - 0.125))
                                       | ((obs['t_min'] > t_end_target) & (obs['t_min'] <= t_end_target + 0.125))]
                    unique_target_names = np.unique(obs_adjacent['target_name'])
                    assumed_ref_names = unique_target_names[unique_target_names != target_name]
                    ref_match = [ob in assumed_ref_names for ob in obs_adjacent['target_name']]
                    obs_ref_match = obs_adjacent[ref_match]
                    oids_for_refs_more = list(np.unique(obs_ref_match['obs_id']))
                    oids_for_refs += oids_for_refs_more
                    oids += oids_for_refs_more
                    obs_calPSF = vstack([obs_calPSF, obs_ref_match])
                # for oid in oids_for_target:
                #     product_row = products[products['obs_id'] == oid]
                #     # if product_row is not None:
                #     #     print(product_row['obsID', 'obs_id', 'productFilename', 'parent_obsid'])
                # # oids += list(np.unique(products['obs_id']))
            else:
                oids += list(np.unique(products['obs_id']))

        self.logger.info(f"Data Fetch: Found {len(obs_calPSF)} total PSF reference star image(s) near science images")
        if len(obs_calPSF) > 0:
            self.logger.info(f"Data Fetch: PSF reference stars found: {list(np.unique(obs_calPSF['target_name']))}")

# FIX ME!!! Figure out how to log this table output nicely.
        # Print the table of observations found.
        print("\nPSF reference images:")
        obs_calPSF['intentType', 'target_name', 'target_classification',
                   't_min', 'obs_id', 't_exptime'].pprint(max_lines=100,
                                                          max_width=200)
        print("\n")

        # Weed out duplicate oids.
        oids = np.unique(oids).tolist()

        self.logger.info('Data Fetch: {} observation IDs found for {} proposal IDs:'.format(len(oids), len(pids)))
        self.logger.info(oids)

        return oids, products, obs


    def download_data(self, ids, destination, product_types=None,
                      cache_dir=None, download_mode='general'):
        '''Downloads MAST data products into a flattened location.
        
        Downloads data products (``product_type``) from input observation IDs (``ids``) 
        from MAST and copies them to a single directory (``destination``) from the 
        temporary download directory (``cache_dir``). Similar to stisteam.getdata(). 
        Written by Sean Lockwood.
        
        Parameters
        ----------
        ids : list of str
            List of observation IDs to download data products from
        destination : str
            Full path to final destination directory for data
        product_types : list of str, optional
            Names of product types to download for each observation ID (default is None, 
            means all data products will be downloaded)
        cache_dir : str, optional
            Full path to temporary data download storage directory (default 
            ``cache_dir`` as defined above)
        
        '''

        if cache_dir is None:
            cache_dir = self.cache_dir

        assert os.path.isdir(destination), 'Destination must be a directory'

        self.logger.info('Data Fetch: Downloading & copying data to {}\n'.format(destination))

        # Authenticate with a MyST account token. Needed for proprietary
        # data download only.
        if self.loginToken is not None:
            session = Observations.login(token=self.loginToken)

        # Get data products for each observation ID
        obs = Observations.query_criteria(obs_id=ids)    
        products = Observations.get_product_list(obs)
        if product_types is not None:
            products = products[[x.upper() in product_types for x in products['productSubGroupDescription']]]

        # Download data and combine into the destination directory
        with TemporaryDirectory(prefix='downloads', dir=cache_dir) as d:
            dl = Observations.download_products(products, mrp_only=False, download_dir=d)
            for filename in dl['Local Path']:
                # Ignore FITS with wrong occulter position.
                with fits.open(filename, mode='readonly') as hdu:
                    hdr = hdu[0].header
                    hdr_sci = hdu[1].header
                    occulter = hdr.get('PROPAPER', '')
                    if hdr_sci.get('EXPFLAG', '') == 'INTERRUPTED':
                        self.logger.info(f"Ignoring {os.path.basename(filename)} for "\
                                         f"EXPFLAG = {hdr_sci.get('EXPFLAG')}")
                        continue
                    if (download_mode == 'science') and \
                            (self.obsMode is not None) and \
                            (occulter != self.obsMode.upper()):
                        self.logger.info(f"Ignoring {os.path.basename(filename)} for "\
                                         f"non-matching occulter ({occulter})")
                        continue
                shutil.copy(filename, destination)


    # Find and update the headers with the best reference files and download them
    def download_ref_files(self, image_list):
        self.logger.info(f'CTI correction: Downloading reference files for following {len(image_list)} images:\n{image_list}\n')
        images = ' '.join(image_list)
        subprocess.check_output('crds bestrefs --files {} --sync-references=1 --update-bestrefs'.format(images),
                                shell=True, stderr=subprocess.DEVNULL)


    def download_darks(self):
        """
        Determine dark exposures required for CCD data and download them (~1.2 GB)
        """
        needed_darks = archive_dark_query(glob.glob(os.path.join(self.science, '*_raw.fits')))
        # needed_darks = archive_dark_query(glob.glob(os.path.join(self.science, '*_flt.fits')))

        dark_exposures = set()
        for anneal in needed_darks:
            for dark in anneal['darks']:
                dark_exposures.add(dark['exposure'])

        self.download_data(dark_exposures, self.darks,
                           product_types={'FLT', 'EPC', 'SPT'})
        self.logger.info("CTI correction: Finished downloading dark data\n")


    def make_working_copies(self, exts=['*_raw.fits', '*_epc.fits', '*_spt.fits', '*_asn.fits', '*_flt.fits', '*_crj.fits']):
        """
        exts: Set file extensions for CCD image data to copy
        """
        
        # Copy over all CCD files for each extension to the CTI correction directory
        # os.chdir(cwd)
        for ext in exts:
            cf.copy_files_check(self.ccd_dir, self.science, files=f'{ext}')


    def run_cti(self, pids, target_name=None, clean=True, download=True):
        """

        Parameters
        ----------
        pids : TYPE
            DESCRIPTION.
        target_name : TYPE, optional
            DESCRIPTION. The default is None.
        clean : TYPE, optional
            DESCRIPTION. The default is True.
        download : bool, optional
            False to skip downloading of files for CTI correction; if this is
            done, will attempt to use existing files in the established
            directories. Default is True.

        Returns
        -------
        bool
            DESCRIPTION.

        """

        # Get the obs_id values from the PIDs.
        self.pids = pids
        self.ccd_oids, products, obs = self.get_oids(pids, target_name=target_name)

        # Abandon CTI correction if no matching observations were found.
        if len(self.ccd_oids) == 0:
            return False

        # Download data.
        # Choices for product_types: 'RAW', 'FLT', 'CRJ', 'EPC', 'SPT', 'ASN', 'SX2'
        if download:
            self.download_data(self.ccd_oids, self.ccd_dir, product_types={'RAW'},
                               download_mode='science')
            self.logger.info("CTI correction: Finished downloading CCD data\n")

        # Download reference files.
        # # Check in the Notebook directory
        # os.chdir(cwd)
        # Get a list of images for the CCD and download reference files
        self.ccd_list = sorted(glob.glob(os.path.join(self.ccd_dir, '*_raw.fits')))

        # Check if the images are subarrayed. If they are, we cannot do the
        # CTI correction and must skip it.
        subarrayed_images = []
        for pth in self.ccd_list:
            try:
                hdr = fits.getheader(pth, ext=1)

                if (hdr.get('NAXIS1', 1024) < 1024) or (hdr.get('NAXIS2', 1024) < 1024):
                    subarrayed_images.append(True)
                else:
                    subarrayed_images.append(False)
            except:
                subarrayed_images.append(False)

        if np.all(subarrayed_images):
            self.logger.warning("CTI correction: All images are subarrayed. "\
                                "CTI correction cannot be performed. "\
                                "SKIPPING CTI CORRECTION.")
            if clean:
                self.cleanup()

            return False
        else:
            self.logger.info("CTI correction: At least "\
                             f"{np.sum(~np.array(subarrayed_images))} images "\
                            "are not subarrayed. Continuing correction...")

        if download:
            self.download_ref_files(self.ccd_list)
            self.logger.info("CTI correction: Finished downloading reference files\n")

        # Copy files for processing.
        self.make_working_copies()

        if download:
            self.download_darks()
        # Run the CTI correction.
        try:
            stis_cti(science_dir=os.path.abspath(self.science),
                     dark_dir=os.path.abspath(self.darks),
                     ref_dir=os.path.abspath(self.ref_cti),
                     num_processes=15,
                     verbose=2,
                     clean=True)
                     # clean_all= True)

            self.post_cti_checks()

            # Copy CTI-corrected FITS files to the main data directory.
            cf.copy_files_check(self.science, self.base_dir, files='*_flc.fits')

            if clean:
                self.cleanup()

            return True

        except Exception:
            self.logger.exception("CTI correction: CTI correction FAILED!")

            if clean:
                self.cleanup()

            return False


    def post_cti_checks(self):

        self.logger.info("CTI correction: Post-CTI checks...")
        # Print information for each file, check those that have been CTI corrected
        # os.chdir(cwd)
        # files = glob.glob(os.path.join(os.path.abspath(science), '*raw.fits'))
        files = glob.glob(os.path.join(os.path.abspath(self.science), '*flt.fits'))
        files.sort()

        psm4 = 0
        ampd = 0
        ampo = 0
        for f in files:

            # Open file header
            hdr = fits.getheader(f, 0)

            # Get year obs
            yr = int(hdr['TDATEOBS'].split('-')[0])

            # Get file info
            if yr <= 2004: 
                self.logger.info('\n...PRE-SM4 ({}), NOT CTE CORRECTED...'.format(yr))
                psm4 += 1
            # elif os.path.exists(f.replace('raw.fits', 'cte.fits')):
            elif os.path.exists(f.replace('flt.fits', 'flc.fits')):
                self.logger.info('\n***CTE CORRECTED, AMP {} ({})***'.format(hdr['CCDAMP'], yr))
                ampd += 1
            else: 
                ampo += 1
                self.logger.info('\n~NON-CTE CORRECTED, AMP {} ({})~'.format(hdr['CCDAMP'], yr))
            self.logger.info('FILE: {}, PID: {}, ROOT: {}'.format(os.path.basename(f), hdr['PROPOSID'], hdr['ROOTNAME']))
            self.logger.info('INST: {}, DETECTOR: {}, AP: {}'.format(hdr['INSTRUME'], hdr['DETECTOR'], hdr['APERTURE']))
            self.logger.info('DATE OBS:{}, PROCESSED: {}'.format(hdr['TDATEOBS'], hdr['DATE']))

        # Run checks on data
        nraw = len(glob.glob(os.path.join(self.science, '*raw.fits')))
        ncte = len(glob.glob(os.path.join(self.science, '*cte.fits')))
        self.logger.info(f'{nraw} raw CCD files input, {ncte} CTE corrected files output')   # Should be equal if all data taken on amp D and post-SM4
        self.logger.info(f'{psm4} pre-SM4, {ampd} amp D (CTE corr), {ampo} other amp (non-CTE)')


    def download_without_cti(self, pids, target_name=None, clean=True):
        """
        Download fits files but do not attempt to correct them. Just move them
        to the destination directory.

        Parameters
        ----------
        pids : TYPE
            DESCRIPTION.
        target_name : TYPE, optional
            DESCRIPTION. The default is None.
        clean : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        # Get the obs_id values from the PIDs.
        self.pids = pids
        self.ccd_oids, products, obs = self.get_oids(pids, target_name=target_name)

        # Abandon CTI correction if no matching observations were found.
        if len(self.ccd_oids) == 0:
            return False

        # Download data.
        check_mkdir(self.cache_dir)
        # Choices for product_types: 'RAW', 'FLT', 'CRJ', 'EPC', 'SPT', 'ASN', 'SX2'
        self.download_data(self.ccd_oids, destination=self.base_dir,
                           cache_dir=self.cache_dir,
                           product_types={'FLT'}, download_mode='science')
        self.logger.info("Data Fetch: Finished downloading uncorrected FLT data\n")

        if clean:
            self.cleanup()

        return


    def cleanup(self):
        
        # Delete cache directories.
        if os.path.isdir(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                self.logger.info("CTI correction: Deleted CTI cache directories")
            except:
                self.logger.exception("CTI correction: WARNING: FAILED to deleted CTI cache directories at "\
                                      f"{self.cache_dir}")
        # Delete CTI data directories.
        if os.path.isdir(self.root_dir):
            try:
                shutil.rmtree(self.root_dir)
                self.logger.info("CTI correction: Deleted CTI temporary data directories")
            except:
                self.logger.exception("CTI correction: WARNING: FAILED to deleted CTI temporary data "\
                                      f"directories at {self.root_dir}")
        # # Delete CTI reference file directories.
        # if os.path.isdir(self.ref_dir):
        #     try:
        #         shutil.rmtree(self.ref_dir)
        #         print("Deleted CTI reference file directories\n")
        #     except Exception as ee:
        #         print("\nWARNING: FAILED to deleted CTI reference file "\
        #               f"directories at {self.ref_dir}\n")
        #         print(ee)

        return
        
