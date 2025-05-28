#!/usr/bin/env python

import os
import sys
import argparse
import pdb
import time
from glob import glob

# Internal imports
try:
    from alicesaur.pipeline.stis.pipeline_stis import PipelineSTIS
except ImportError as ee:
    print(f"\n{ee}")
    print("\nUh oh! The alicesaur directory may not have been found. Make "\
          "sure its parent directory is in your PYTHONPATH and try again."\
          " Exiting.\n")
    sys.exit(1)


if __name__ == "__main__":

    desc = 'Reduce STIS imaging data'
    parser = argparse.ArgumentParser(description=desc)

    desc = 'Path to directory containing FITS data for processing.'
    parser.add_argument('--dataDir', metavar='dataDir',
                        default=None, type=str, nargs='?',
                        help=desc)

    desc = 'Instrument from which the data originated: stis (default) or acs.'
    parser.add_argument('--instrument', choices=['stis', 'acs'],
                        default="stis", type=str, nargs='?', help=desc)

    desc = 'Image input file type; sx2 is default.'
    parser.add_argument('--inputType', choices=['flt', 'flc', 'xft', 'xfc',
                                                'axt', 'axc', 'sx2'],
                        default="sx2", type=str, nargs='?', help=desc)

    desc = 'Observation mode: bar10 or wedgeb1.0. Default is generic "nomode".'
    parser.add_argument('--obsMode', metavar='obsMode',
                        default="nomode", type=str, nargs='?',
                        help=desc)

    desc = 'PSF subtraction mode: rdi, adi, pyklip-rdi, or pyklip-adi'
    parser.add_argument('--psfSubMode', metavar='psfSubMode',
                        default="rdi", type=str, nargs='?',
                        help=desc)

    desc = 'Minimum PA rotation in [degrees] relative to science image ' \
            + 'required to allow a reference image when using ADI PSF ' \
            + 'subtraction. Default is None (allow all references).'
    parser.add_argument('--deltaPAMin',
                        default=None, type=float, nargs='?',
                        help=desc)

    desc = 'Target name (e.g., HD-32297). For names starting with special characters '\
        'like a - (dash), escape them with a leading forward slash like \-AU-MIC'
    parser.add_argument('--targ', metavar='targ',
                        default="", type=str, nargs='?',
                        help=desc)

    desc = 'HST proposal ID numbers for which to fetch data from MAST to run '\
            + 'CTI correction. Separate multiple IDs with spaces.'
    parser.add_argument('--pids', default=[], type=int, nargs='*', help=desc)

    desc = 'Width in [pixels] of diffraction spike masks. Default is to use the value in the info.json.'
    parser.add_argument('--spWidth', metavar='spWidth',
                        default=None, type=int, nargs='?',
                        help=desc)

    desc = 'Number of annuli to use for subtraction optimization (default 1)'
    parser.add_argument('--ann', metavar='ann',
                        default=1, type=int, nargs='?',
                        help=desc)

    desc = 'Flag to save the final image to FITS, overwriting any existing file.'
    parser.add_argument('--saveFinal', dest="saveFinal",
                        action="store_true", help=desc)

    desc = 'Custom identifier for the reduction, which will be appended to '
    desc += 'the ends of all saved files. Default is an empty string.'
    parser.add_argument('--cid', default='', type=str,
                        nargs='?', help=desc)

    desc = 'Central UTC date of the time span to include in this processing. '
    desc += 'If given, the pipeline will only include images that were '
    desc += 'observed during the window <--date-incl> - <--date-incl-span> to '
    desc += '<--date-incl> + <--date-incl-span>. Format must be YYYY-MM-DD '
    desc += '(for which the assumed time is 00:00) or YYYY-MM-DDThh:mm. '
    desc += 'If None (default), no time constraint is applied.'
    parser.add_argument('--date-incl', default=None, type=str,
                        nargs='?', help=desc)

    desc = 'Number of days before and after --date-incl to include in the '
    desc += 'dataset. Default is 2.'
    parser.add_argument('--date-incl-span', default=2., type=float,
                        nargs='?', help=desc)

    desc = 'Use --noFixCTI to NOT correct for Charge-Transfer Inefficiency in'\
        ' each individual .flt image. Only applies if --input-type flt and '\
        ' --pids [proposal ID] are supplied.'
    parser.add_argument('--noFixCTI', dest="noFixCTI",
                        action="store_true", help=desc)

    desc = 'Use --noFixPix to NOT fix bad pixels in each individual image.'
    parser.add_argument('--noFixPix', dest="noFixPix",
                        action="store_true", help=desc)

    desc = 'Use --noAutoMask to NOT apply automatic source masks.'
    parser.add_argument('--noAutoMask', action="store_true", help=desc)

    desc = 'Use --noMaskSaturation to NOT mask saturated pixels in each '\
        'individual image when doing PSF subtraction and final image collapse.'
    parser.add_argument('--noMaskSaturation', dest="noMaskSaturation",
                        action="store_true", help=desc)

    desc = 'Use --noRadon to NOT do radon star centering on each image.'
    parser.add_argument('--noRadon', dest="noRadon",
                        action="store_true", help=desc)

    desc = 'Use --noCombine to NOT combine images from each orbit before processing.'
    parser.add_argument('--noCombine', dest="noCombine",
                        action="store_true", help=desc)

    desc = 'Use --noPad to NOT pad images into 2048x2048 arrays.'
    parser.add_argument('--noPad', dest="noPad",
                        action="store_true", help=desc)

    desc = 'Use --noErrorMaps to NOT make error maps or SNR maps.'
    parser.add_argument('--noErrorMaps', dest="noErrorMaps",
                        action="store_true", help=desc)

    desc = 'Use --debug to enter debugger at various steps and plot some info.'
    parser.add_argument('--debug', action="store_true", help=desc)

    desc = 'Use --do-gaia to do the Gaia astrometry thingy.'
    parser.add_argument('--do-gaia', action="store_true", help=desc)

    desc = 'Use --iterate to run the pipeline with multiple consecutive '\
        'iterations to refine the final result.'
    parser.add_argument('--iterate', action="store_true", help=desc)

    desc = 'MAST login token; only required for downloading proprietary data. '
    desc += 'You can create a new token at https://auth.mast.stsci.edu/tokens'
    parser.add_argument('--mastToken', default=None, type=str,
                        nargs='?', help=desc)

    args = parser.parse_args()


    # Handle target names that contain escape characters "\".
    if len(args.targ) > 0:
        if '\\' in args.targ:
            vars(args)["targ"] = args.targ.replace('\\', '')
            print(f"\nWARNING: Revised --targ target name to {args.targ} "\
                  "by removing escape character '\\'\n")

    # Iterative pipeline run.
    if args.iterate:
        print("\nRunning iterative pipeline\n")
        # Create a "detault" reduction pipeline instance.
        updated_args = vars(args).copy()
        # Add an "iter1" suffix to files in first iteration.
        if args.cid == "":
            updated_args['cid'] = "iter1"
        else:
            updated_args['cid'] = "iter1_" + args.cid
        pl_1 = PipelineSTIS(**updated_args)
        # Run the default reduction.
        pl_1.run()

        # Determine the highest level intermediate product present and
        # use that as the inputType for the second iteration.
        inputType_options = ['xfc', 'xft', 'flc', 'flt']
        for it in inputType_options:
            files_present = glob(os.path.join(pl_1.dataDir, f"*_{it}.fits"))
            if len(files_present) > 0:
                new_inputType = it
                break
            else:
                new_inputType = None

        if new_inputType is None:
            try:
                pl_1.logger.warning("No high-level intermediate FITS files "\
                                  f"found in {pl_1.dataDir}. Skipping second "\
                                  "iteration of pipeline.")
            except:
                print("No high-level intermediate FITS files "\
                      f"found in {pl_1.dataDir}. Skipping second "\
                      "iteration of pipeline.")
        else:
            try:
                pl_1.logger.info("Running second pipeline iteration with "\
                                 f"inputType = {new_inputType}")
            except:
                print("Running second pipeline iteration with "\
                      f"inputType = {new_inputType}")

            # Now re-run the pipeline from the axf/axc inputs a second time,
            # using input from the first run's output.
            updated_args = vars(args).copy()
            # Maintain the same data directory.
            updated_args['dataDir'] = pl_1.dataDir
            updated_args['inputType'] = new_inputType
            # Remove pids so we don't download data again.
            if 'pids' in updated_args.keys():
                updated_args.pop('pids')
            updated_args['noFixPix'] = True
            updated_args['logger'] = pl_1.logger
            pl_2 = PipelineSTIS(**updated_args)
            # Run the second reduction.
            pl_2.run()

    # Normal (single) pipeline run.
    else:
        # Create the reduction pipeline instance.
        pl = PipelineSTIS(**vars(args))

        # Run the reduction.
        pl.run()
