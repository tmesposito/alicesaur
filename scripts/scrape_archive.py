#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:20:12 2024

@author: tom
"""

import os
import sys
import shutil
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.table import Table, vstack
from astroquery.mast import Observations
from urllib import request as urlrequest, parse as urlparse, error as urlerror
from tempfile import TemporaryDirectory
from pprint import pprint
from tqdm import tqdm

from alicesaur.utils import check_mkdir


# # Get data products for each observation ID
# obs = Observations.query_criteria(obs_id=ids)    
# products = Observations.get_product_list(obs)
# if product_types is not None:
#     products = products[[x.upper() in product_types for x in products['productSubGroupDescription']]]

# # Download data and combine into the destination directory
# with TemporaryDirectory(prefix='downloads', dir=cache_dir) as d:
#     dl = Observations.download_products(products, mrp_only=False, download_dir=d)
    

# EXAMPLE NOTES
def get_proposal_ids(abstract='stis'): #', +ccd'): #, title=''):
    '''
    Perform an HST Abstract search for STIS programs.
    '''
    url = 'https://archive.stsci.edu/hst/abstract.html'
    form_data = urlparse.urlencode({
        'abstract': abstract,      # String to be searched for within the abstract
        # 'atitle':   title,         # String to be searched for within the title
        'checkbox': 'no',          # Display abstract?
        'submit':   'submit'}).encode()

    # Submit POST request:
    try:
        response = urlrequest.urlopen(url, form_data)  # Needs better error handling!
        lines = response.readlines()
        # Note 'utf-8' decoding fails here on accented letters. Use latin-1.
        lines_decoded = [line.decode('latin-1') for line in lines]
        text = [line.rsplit('\n',1)[0] for line in lines_decoded]
        response.close()
    except (urlerror.URLError, urlerror.HTTPError) as e:
        print('Please check your internet connection!')
        raise e

    # Parse returned text for proposal IDs:
    text = [x for x in text if 'proposal_search.php' in x]
    proposals = [int(y.split('>')[1].split('<')[0]) for y in text]

    return proposals


def find_observations(pids, instrument_name='STIS/CCD', filters='MIRVIS',
                      dataproduct_type='image', calib_level=3,
                      dateStart=None, dateEnd=None,
                      verbose=False):
    """

    Inputs:

        dateStart: str
            Start date and time of allowed observation range, in isot format
            like "YYYY-MM-DDThh:mm:ss". Only the "YYYY-MM-DD" is required.

        dateEnd: str
            End date and time of allowed observation range, in isot format
            like "YYYY-MM-DDThh:mm:ss". Only the "YYYY-MM-DD" is required.
    """

    if dateStart is not None:
        timeStart = Time(dateStart)
        timeStart_mjd = timeStart.mjd

        if dateEnd is not None:
            timeEnd = Time(dateEnd)
        else:
            timeEnd = timeStart + TimeDelta(1, format='jd')
        timeEnd_mjd = timeEnd.mjd

        print("\nRestricting observation date range to between UTC "\
              f"{dateStart} and {timeEnd.isot} (inclusive)\n")

    obs_list = []
    for pid in tqdm(pids, desc="Searching observations by pid"):
        # Optionally filter by observation date range.
        if dateStart is not None:
            obs = Observations.query_criteria(proposal_id=f'{pid}',
                                              instrument_name=instrument_name,
                                              filters=filters,
                                              dataproduct_type=dataproduct_type,
                                              calib_level=calib_level,
                                              t_min=[timeStart_mjd, timeEnd_mjd])
        else:
            obs = Observations.query_criteria(proposal_id=f'{pid}',
                                              instrument_name=instrument_name,
                                              filters=filters,
                                              dataproduct_type=dataproduct_type,
                                              calib_level=calib_level)

        if len(obs) <= 0:
            if verbose: print(f"No observations found for pid {pid}")
            continue
        else:
            # Sort observations by t_min to roughly sort by date recorded.
            obs.sort('t_min')
            # # Filter observations by start and end date, if they are given.
            # if dateStart is not None:
            #     # try:
            #     wh_date_ok = (obs['t_min'] >= timeStart_mjd) & (obs['t_min'] <= timeEnd_mjd)
            #     if np.sum(wh_date_ok) == 0:
            #         if verbose: print(f"pid {pid}: All {len(obs)} "\
            #                           "observations were outside the"\
            #                           "requested date range.")
            #         continue
            #     else:
            #         obs_all = obs.copy()
            #         obs = obs_all[wh_date_ok]
            if verbose: pprint(obs)
            obs_list.append(obs)
            
    print(f"{len(obs_list)} observations found for parameters: "\
          f"{instrument_name} ; {filters} ; {dataproduct_type} ; "\
          f"calib level = {calib_level}")

    return obs_list


# FIX ME!!! Handle case of same target being observed in multiple blocks
# separated by significant time within a single PID. Could take an input date
# to specify which block to fetch, otherwise all will get rolled together.
def get_oids(obs_list, target_name=None, dataDir='.'):
    """
    Get observation IDs from proposal IDs.

    From a list of input proposal IDs (``pids``) get a list of observation IDs 
    (``oids``) using astroquery.

    Parameters
    ----------


    Returns
    -------
    oids : list of str
        List of observation IDs within the input proposal IDs

    """
    
    unique_ids_all = []
    
    pids_science = []
    obs_list_science = []
    oids_science = []

    # For each PID get obs IDs
    for ii, obs in enumerate(obs_list):
        pid = np.unique(obs['proposal_id'])[0]

        science_row_inds = np.where(obs['intentType'] == 'science')[0]
        N_science = len(science_row_inds)
        if N_science == 0:
            print(f"Zero science images found for obs {ii}. Skipping.")
            continue
        else:
            pids_science.append(pid)
            obs_list_science.append(obs)

# FIX ME!!! Figure out how to log this table output nicely.
        # Print the table of observations found.
        print("\nObservations found:")
        obs['intentType', 'target_name', 'target_classification', 't_min',
            'obs_id', 't_exptime'].pprint(max_lines=100, max_width=200)

        if len(obs) == 0:
            print("*** NO OBSERVATIONS FOUND in archive with" \
                  f" program ID {pid}. Check your program ID."\
                  " Aborting.")
            return [], [], None

        # Get products.
        products = np.unique(Observations.get_product_list(obs))

        # Separate out obs for only PSF reference stars.
        # First grab anything with both 'PSF' and the science target's name
        # in the obs target name.
        target_names = np.array(np.unique(obs[science_row_inds]['target_name']))

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
                print("*** NO OBSERVATIONS FOUND in archive with" \
                      f" program ID {pid} AND target name {target_name}."\
                      "Check both inputs before trying again. Aborting.")
                unique_targets = list(np.unique(obs['target_name']))
                print(f"Target names found for program ID {pid}: {unique_targets}")
                return [], [], None

            t_start_target = min(obs_target['t_min'])
            t_end_target = max(obs_target['t_max'])

            # Locate PSF reference stars observed in the middle of the target obs.
            # If none, look immediately before and after the target obs.
            # Make sure to match the camera settings.
            obs_calPSF_match = obs_calPSF[(obs_calPSF['t_min'] >= t_start_target) & (obs_calPSF['t_min'] <= t_end_target)]
            print(f"Data Fetch: Found {len(obs_calPSF_match)} clearly labeled PSF reference star image(s) amid science images")
            if len(obs_calPSF_match) > 0:
                print(f"Data Fetch: PSF reference stars found: {list(np.unique(obs_calPSF_match['target_name']))}")
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
            # Weed out duplicate oids.
            oids = list(np.unique(products['obs_id']))
            oids = np.unique(oids).tolist()
            oids_science.append(oids)

            # oids = np.unique(oids).tolist()
            # oids.append(list(np.unique(products['obs_id'])))

    # self.logger.info(f"Data Fetch: Found {len(obs_calPSF)} total PSF reference star image(s) near science images")
    # if len(obs_calPSF) > 0:
    #     self.logger.info(f"Data Fetch: PSF reference stars found: {list(np.unique(obs_calPSF['target_name']))}")

# FIX ME!!! Figure out how to log this table output nicely.
    # Print the table of observations found.
    # print("\nPSF reference images:")
    # obs_calPSF['intentType', 'target_name', 'target_classification',
    #            't_min', 'obs_id', 't_exptime'].pprint(max_lines=100,
    #                                                   max_width=200)
    # print("\n")

        # # Weed out duplicate oids.
        # oids = np.unique(oids).tolist()

        # self.logger.info('Data Fetch: {} observation IDs found for {} proposal IDs:'.format(len(oids), len(pids)))
        # print(oids)

        # unique_ids_obs = download_data(oids, obs, os.path.abspath(dataDir),
        #                                product_types='FLT', download_mode='general')
        
        # unique_ids_all += unique_ids_obs

        # print(unique_ids_all)

    print(f"\nSCIENCE proposal ids ({len(pids_science)}):\n"\
          f"{pids_science}\n")

    table = Table(data=None, names=["pid", "dataset_id"], dtype=[str, str])

    # Identify all the data by downloading it -- sooo slow and messy.
    for ii, oid in enumerate(oids_science):
        try:
            unique_ids_obs = download_data(oid, obs_list_science[ii], os.path.abspath(dataDir),
                                           product_types='FLT', download_mode='general')

            for uid_obs in unique_ids_obs:
                table.add_row(vals=[pids_science[ii], uid_obs])
            table.write(os.path.join(dataDir, 'dataset_ids.csv'), delimiter=',',
                        format='ascii.fast_no_header',
                        overwrite=True)

            unique_ids_all += unique_ids_obs

            print(unique_ids_all)
        except Exception as ee:
            print(ee)
            print(f"\n*** FAILED to generate dataset_ids for obs {ii}\n")

    # table = Table(np.array(test).reshape(2,1))
    # table.write(os.path.join(dataDir, 'dataset_ids.csv'), delimiter=',',
    #             format='ascii.fast_no_header',
    #             overwrite=True)

    return table


def download_data(ids, obs, destination, product_types=None,
                  cache_dir=None, download_mode='general',
                  copy_to_destination=False):
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
        cache_dir = os.path.join(destination, 'data_cache')

    assert os.path.isdir(destination), 'Destination must be a directory'

    check_mkdir(cache_dir)

    print('Data Fetch: Downloading & copying data to {}\n'.format(destination))

    # Get data products for each observation ID
    obs = Observations.query_criteria(obs_id=ids)    
    products = Observations.get_product_list(obs)
    if product_types is not None:
        products = products[[x.upper() in product_types for x in products['productSubGroupDescription']]]

    occulters = []
    target_names = []
    dates= []
    pids = []
    
    unique_id_list = []

    # Download data and combine into the destination directory
    with TemporaryDirectory(prefix='downloads', dir=cache_dir) as d:
        dl = Observations.download_products(products, mrp_only=False, download_dir=d)
        for filename in dl['Local Path']:
            # Ignore FITS with wrong occulter position.
            with fits.open(filename, mode='readonly') as hdu:
                hdr = hdu[0].header
                occulter = hdr.get('PROPAPER', '')
                occulters.append(occulter)
                target_names.append(hdr.get('TARGNAME', ''))
                dates.append(hdr.get('TDATEOBS', ''))
                pids.append(hdr.get('PROPOSID', ''))
                # if (download_mode == 'science') and \
                #         (self.obsMode is not None) and \
                #         (occulter != self.obsMode.upper()):
                #     self.logger.info(f"Ignoring {os.path.basename(filename)} for "\
                #                       f"non-matching occulter ({occulter})")
                #     continue

            # Copy FITS to destination.
            if copy_to_destination:
                shutil.copy(filename, destination)

        print(occulters)
        dataset_ids = []
        if np.all(np.array(occulters) == '50ccd'):
            print("No occulted images found. Skipping.")
            return []
        else:
            # pid = np.unique(obs['proposal_id'])[0]
            # target_class = 
            
            for ii in range(len(occulters)):
                dataset_ids.append(f"{target_names[ii]}_{dates[ii]}_{occulters[ii].lower()}_stis")

        print(dataset_ids)

        unique_ids = np.unique(dataset_ids)

        print(unique_ids)
        
        # # Merge datasets that are identical but one day apart. Probably just
        # # crossed midnight.
        # if len(unique_ids) > 1:
        #     times = [Time(uid.split('_')[1]) for uid in unique_ids]
        #     for uid in unique_ids:
        #         parts = unique_ids[0].split('_')
        #         parts.pop(1)
        #         uid_parts.append(parts)
        #     uid_parts = np.array(uid_parts)
        #     for jj, uid in enumerate(unique_ids):
        #         if jj == 0:
        #             continue
        #         wh_match = np.where(uid_parts[jj] == uid_parts)
        #         if len(wh_match[0]) > 1:

        for uid in unique_ids:
            unique_id_list.append(uid)

    return unique_id_list



if __name__ == "__main__":

    desc = 'Scrape STIS observation information from the MAST archive.'
    parser = argparse.ArgumentParser(description=desc)

    desc = 'Start date of the time range to scrape. Must be in isot format '
    desc += 'like YYYY-MM-DDThh:mm:ss. Only the YYYY-MM-DD day is required.'
    desc += 'If None (default), no time constraint is applied.'
    parser.add_argument('--date-start', default=None, type=str,
                        nargs='?', help=desc)

    desc = 'End date of the time range to scrape. Must be in isot format '
    desc += 'like YYYY-MM-DDThh:mm:ss. Only the YYYY-MM-DD day is required.'
    desc += 'If None (default), no time constraint is applied.'
    parser.add_argument('--date-end', default=None, type=str,
                        nargs='?', help=desc)

    desc = 'Path to output directory for files created.'
    parser.add_argument('--output-dir', default='.', type=str,
                        nargs='?', help=desc)

    args = parser.parse_args()

    if (args.date_start is not None) & (args.date_end is None):
        print("\n*** HELP!!! If --date-start is given, you must also give "\
              "-date-end. Exiting.\n")
        sys.exit(0)

    # Get all proposal ids that contain "stis" in the abstract.
    pids = get_proposal_ids(abstract='stis')

    # List all observations for those pids.
    obs_list = find_observations(pids, dateStart=args.date_start,
                                 dateEnd=args.date_end)

    # Get Observation IDs for all observations.
    dataset_id_table = get_oids(obs_list,
                                dataDir=os.path.abspath(args.output_dir))