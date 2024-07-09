# alicesaur
Archival Legacy Investigation of Circumstellar Environments: STIS and ACS Uniform Reduction

This Python-based data reduction pipeline is designed to process high-contrast, coronagraphic images from the Hubble Space Telescope's STIS and ACS cameras to detect circumstellar disks and point-source companions (exoplanets, brown dwarfs, and stars).

Alicesaur is open source and GNU General Public Licensed.

## Features

* Download raw and calibration data from the MAST archive in FITS format
* Calibrate images
* Subtract the primary point-spread function via RDI or KLIP ADI
* Output a fully processed image with various levels of PSF
* Start the pipeline from various intermediate image products and pipeline steps to avoid wasting time reprocessing from scratch

## Installation

Installing alicesaur requires the cloning of this repository, installing several underlying Python packages, and installing some non-Python executables and libraries.

Before starting, we suggest you create a new virtual environment as a home for alicesaur. You can use whichever environment manager you prefer: e.g., conda, virtualenv, pipenv, etc.

Be sure to activate this new environment and work within it for all following installation steps.

### Python
Python 3.9 or higher is recommended. You can install it with a package manager or from source. Using your operating system's existing Python installation is discouraged because changes or damage to it can create serious problems at the system level.

### Install alicesaur source code
To install the core alicesaur functionality, clone this repository. We recommend setting up an ssh key to access your github account via ssh authentication, in which case you would clone the repository with:
```git clone git@github.com:tmesposito/alicesaur.git```
Otherwise, you can clone the repository via HTTPS authenticaion with:
```git clone https://github.com/tmesposito/alicesaur.git```

### Install libcfitsio library
Some of the STScI tools used by alicesaur depend on the CFITSIO library named libcfitsio. The easiest way to install this is via Homebrew, which we recommend. Otherwise, it can be installed from source. Instructions for both methods can be found at https://heasarc.gsfc.nasa.gov/fitsio/fitsio_macosx.html.

*After you install libcfitsio*, you will also need to manually copy the "libcfitsio.10.x.y.z.dylib" file (where x,y,z will be some subversion numbers) from the installed cfitsio directory into the directory where the STScI code will look for it. That's typically inside your environment's `lib/` directory. After doing so, you must either A) rename the copied file to the specific filename the code expects, which for versions named libcfitsio.10.x.y.z will be `libcfitsio.10.dylib`, or B) make a symbolic link with that name linked to the copied libcfitsio.10.x.y.z.dylib file.

### Install Python dependencies
alicesaur is dependent on several publicly available Python packages. To install the bulk of these, first `cd` into your cloned local repository of alicesaur. Then, do a pip install of the repository's `requirements.txt` file:
```pip install -r requirements.txt```

A few dependencies cannot be pip installed. For these, do the following:
1. Clone the github repository for STScI's `crds` package to use their Calibration Reference Data System: ```git clone git@github.com:spacetelescope/crds.git```
2. Go to the github repository for STScI's STIS data reduction notebook at https://github.com/spacetelescope/STIS-Notebooks/tree/main/drizpac_notebook and download the single file `copy_files.py` to the same parent directory that houses your local alicesaur repository.

### Install CALSTIS executables
Optional calibration steps within alicesaur for charge transfer inefficiency (CTI) correction and geometric distortion (x2d) correction rely on C executables from older CALSTIS code. At the moment, the easiest way to install them is to ask the alicesaur developers for copies.

Once you have them, place all the cs#.e files into your environment's PATH. The suggested location is inside your environment's `bin/` directory, but any path that you then add to the environment's PATH variable should work.

Next, on some operating systems such as MacOS, you will need to adjust your security preferences to "trust" two of these executables, as they are not recognized as safe by default. One of the following approaches should work on Mac to solve the issue. You may or may not need to find similar workarounds for other OS's.
1. Via command line, run `xattr -dr com.apple.quarantine "/your/path/here/bin/cs0.e"`. Repeat this for each `cs#.e` file.
2. Try to run the executable, either by calling it individually from the command line or by running the alicesaur pipeline. When an error message appears saying the executable can't be run because it is not trusted, open System Preferences, go to Privacy & Security, select "Allow applications downloaded from App Store and identified developers", then below that where there is a message that says "cs#.e was blocked from used because it is not from an identified developer" click the "Allow anyway" button. You will need to repeat this for each cs#.e file, which may require running the pipeline multiple times (if you chose that method). Fortunately, you only need to do this entire process once.

### Add alicesaur to your PYTHONPATH
To run, the parent directory containing your local alicesaur repository must be in your environment's PYTHONPATH. To add it temporarily for your currently active shell, you can run either:
```export PYTHONPATH=/path/to/parent/directory:$PYTHONPATH```  for a bash shell
OR
```setenv PYTHONPATH /path/to/parent/directory:$PYTHONPATH```  for a csh shell like csh, tcsh, or zsh
If you want this change to be permanent, you can add the above line to your general shell "rc" file, typically named something like ~/.bashrc or ~/.cshrc, or to your environment's startup script.

## Quick Start

### Example STIS reduction
You can run a basic STIS reduction from raw MAST archive files to RDI PSF-subtracted final images using the script
`scripts/main_reduce_stis.py`. The default outputs to the input image directory will be:
1. The final, time-collapsed, PSF-subtracted image named `final_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`.
  * Default intensity units are DN/s.
  * Default final star location is pixel (1024, 1024) in Python coordinates; i.e., (1025, 1025) in DS9.
2. An error map with the 1-sigma noise per pixel in units of DN/s named `error_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`.
3. A signal-to-noise ratio map named `snr_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`.

The way to run it is as follows:

First, create a directory to contain the data. We recommend making an outer directory for the target, observation date, and instrument (e.g., "hd115600_20200210_stis"), then a subdirectory within that for the observing mode or occulter position (e.g., "bar10").

Then, call the main reduction script like:
```
python alicesaur/scripts/main_reduce_stis.py --dataDir ~/path/to/data/directory/hd129590_20200413_stis/bar10
--instrument stis --obsMode bar10 --psfSubMode rdi --targ HD-129590 --noCombine --inputType flt --pids 15653 --saveFinal [--noFixPix --noErrorMaps]
```
where
* `dataDir` is the path to the directory containing your input sx2 FITS files ("raw" data),
* `obsMode` is the observation mode; either "bar10", "wedgeB1.0", "wedbgeB1.8", "wedgeA1.6"
* `psfSubMode` should always be "rdi" at this point
* `targ` is the target name as written in the FITS headers
* `--noCombine` is used to NOT combine individual science images by orbit before PSF subtracting them
* `--inputType` determines the type of input data to use: "flt" (for reduction from raw archive images or local flat-field images), "flc" for flat-fielded and CTI-corrected, "xfc" for distortion- & CTI-corrected images, "axc" for aligned distortion- & CTI-corrected images (the highest level intermediate product).
* `--saveFinal` is required to save the final PSF-subtracted image to file
* `--noFixPix` can be used to skip bad pixel fixing if the input files were already fixed, saving time
* `--noErrorMaps` is used to NOT save the error map and SNR map FITS

This will download raw STIS data for target HD 129590 with the program ID 15653 and the "BAR10" occulter position. It will then perform CTI correction, which takes up to ~10 minutes (depending on length of data set) and output new "flc.fits" files to the data directory. Then it will perform the rest of the calibration steps, outputting "xfc.fits" and "axc.fits" image files with individual CRSPLIT slices.

Next, it will output an intermediate product named `unified_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits` containing the unified (summed over CRSPLITs) images aligned to a common star center. Following that will be a cube of the individual PSF-subtracted images named `psfcube_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`. Finally, the final image products will be output.

**_NOTE:_** The Pipeline class currently looks for an `info.json` file just above the directory
containing the input FITS files. That file contains
the basic reduction parameters, like filtering choices and PSF-subtraction optimization
region definitions. This file is optional and the pipeline will run without one, but it will use default values for those
reduction parameters which may not be optimal for every data set. In the meantime, here is a basic
template if you want to try a reduction, with parameters for one Bar10 dataset and one WedgeB1.0 dataset for the same target (HD 115600).

```
{
    "targetName": "HD-129590",
    "psfRefName": "PSF-HD-127271",
    "obsLogPath": "/Users/tom/Research/data/hst/hd129590_20200413_stis/obs_log_hd129590_20200413_stis.csv",
    "diskPA_deg": 119.0,
    "bar10": {
    	"bgCen_yx": "1475 835",
    	"bgCen_yx_noPad": "984 747",
    	"bgCenRef_yx": "603 531",
	    "bgRadius": 60,
    	"spWidth": 8,
    	"radProfSub": {
    	    "paList": [90],
    	    "paHW": 179,
    	    "rMax": 230,
            "postCombine": "True"
	    },
    	"exclude": {
            "sci": {
            	"r_in": 12,
            	"r_out": 80	,
            	"pa_deg": [],
            	"rect_cenYX_widthYX_angleDeg": [[[1054, 1005], [100, 400], 30]],
            	"OLDrect_cenYX_widthYX_angleDeg": [[[1032, 1012], [49, 260], 28]],
            	"point_yxr": [[1122, 970, 9], [1172, 962, 9], [941, 780, 15]],
            	"rect_cenYX_widthYX_angleDeg_noPad": [[[750, 973], [49, 260], 28]]
            },
            "ref": {
                "point_yxr": [[854, 1037, 9]]
            }
        }
    },
    "wedgeb1.0": {
    	"bgCen_yx": "988 1345",
    	"bgCen_yx_noPad": "173 719",
        "bgCenRef_yx": "601 946",
        "bgRadius": 50,
        "spWidth": 10,
    	"radProfSub": {
    	    "paList": [220],
    	    "paHW": 50,
    	    "rMax": 230,
            "postCombine": "False"
    	},
    	"exclude": {
    	    "sci": {
            	"r_in": 14,
            	"r_out": 77,
            	"pa_deg": [],
            	"rect_cenYX_widthYX_angleDeg": [[[1037, 1012], [53, 338], 28]],
            	"point_yxr": [[1122, 965, 12], [896, 1071, 13], [1172, 961, 8]],
            	"rect_cenYX_widthYX_angleDeg_noPad": [[[358, 242], [49, 267], 28]],
            	"point_yxr_noPad": [[444, 199, 8], [218, 302, 10], [493, 191, 8]]
            },
            "ref": {
            	"point_yxr": [[457, 305, 15]]
            }
    	}
    }
}
```

Brief glossary of info.json parameters:
* `bgCen_yx`: Y and X pixel coordinates for the center of a patch where the image background is measured in PADDED science images
  * CAUTION: These coordinates assume the image array has the same dimensions of the final output image. This is DIFFERENT from the reference image coordinate frame.
* `bgCenRef_yx`: Y and X pixel coordinates for the center of a patch where the image background is measured in RAW reference PSF images.
* `bgCenFinal_yx`: Y and X pixel coordinates for the center of a patch where the image background is measured in the FINAL output image.
  * CAUTION: These coordinates assume the image array has the same dimensions of the raw input image.
* `bgRadius`: Radius of the circular background patches in [pixels]
* `spWidth`: Full width of the diffraction spike masks in [pixels]
* `radProfSub`: Series of parameters controlling the optional radial profile subtraction after PSF-subtraction
  * `paList`: List of PAs in [degrees E of N] that set the center of the radial profile sampling wedge
  * `paHW`: Half-width (in azimuth) of the radial profile sampling wedge, in [degrees]; i.e., the sampling wedge extends this many degrees in PA on either side of the value in `paList`.
  * `rMax`: Maximum radius from the star to which the radial profile is measured and subtracted, in [pixels]
  * `postCombine`: Boolean that (if True) will do a second radial profile subtraction on the time-collapsed PSF-subtracted image at the very end of the reduction process.
* `exclude`: Series of parameters placing masks in the science (`sci`) and reference (`ref`) images.
  * See improcess/mask.py function documentation for details
  * `sci` coordinates are in the final image frame
  * `ref` coordinates are in the raw input reference image frame
  * More documentation on these to come eventually...
