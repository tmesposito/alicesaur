# alicesaur
Archival Legacy Investigation of Circumstellar Environments: STIS and ACS Uniform Reduction

This Python-based data reduction pipeline is designed to process high-contrast, coronagraphic images from the Hubble Space Telescope's STIS and ACS cameras to detect circumstellar disks and point-source companions (exoplanets, brown dwarfs, and stars). Currently, only the STIS portion is implemented.

Alicesaur is open source and GNU General Public Licensed.

## Features

* Download raw and calibration data from the MAST archive in FITS format
* Process input files in flt or sx2 formats
* Calibrate images with charge-transfer inefficiency correction (CTI), dark subtraction, flat fielding, distortion correction, bad pixel fixing, saturation masking, and image alignment (registration)
* Subtract the primary star's point-spread function (PSF) via Reference Differential Imaging (RDI) or Angular Differential Imaging (ADI)
* Output intermediate products at major stages of processing, such as post-calibration images, post-alignment images, and PSF-subtracted cubes
* Output the fully processed image cube with extensions for different levels of post-processing
* Define processing parameters for each data set via a single json file
* Start the pipeline from various intermediate image products and pipeline steps to avoid wasting time reprocessing from scratch

## Installation

Installing alicesaur requires the cloning of this repository, installing several underlying Python packages, and installing some non-Python executables and libraries.

Before starting, we suggest you create a new virtual environment as a home for alicesaur. You can use whichever environment manager you prefer: e.g., conda, virtualenv, pipenv, etc.

Be sure to activate this new environment and work within it for all following installation steps.


### Supported systems

Alicesaur is supported on Linux and Mac operating systems.

### Python
Python 3.9 or higher is required. You can install it with a package manager or from source. Using your operating system's existing Python installation is discouraged because changes to it can create serious problems at the system level.

### Install alicesaur source code
To install the core alicesaur functionality, clone this repository. We recommend setting up an [ssh key to access your github account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) via ssh authentication, in which case you would clone the repository with:
```
git clone git@github.com:tmesposito/alicesaur.git
```
Otherwise, you can clone the repository via HTTPS authentication with:
```
git clone https://github.com/tmesposito/alicesaur.git
```

### Install libcfitsio library
Some of the STScI tools used by alicesaur depend on the CFITSIO library named libcfitsio. The easiest way to install this is via a package manager like apt (linux) or Homebrew (Mac), which we recommend. Otherwise, it can be installed from source. Instructions for both methods can be found at https://heasarc.gsfc.nasa.gov/fitsio.

*After you install libcfitsio*, you will likely also need to manually copy the "libcfitsio.10.x.y.z.dylib" file (where x,y,z will be some subversion numbers) from the installed cfitsio directory into the directory where the STScI code will look for it. That's typically inside your environment's `lib/` directory. After doing so, you must either A) rename the copied file to the specific filename the code expects, which for versions named libcfitsio.10.x.y.z will be `libcfitsio.10.dylib`, or B) make a symbolic link with that name linked to the copied libcfitsio.10.x.y.z.dylib file.

### Install Python dependencies
alicesaur is dependent on several publicly available Python packages. To install the bulk of these, first `cd` into your cloned local repository of alicesaur. Then, do a pip install of the repository's `requirements.txt` file:
```
pip install -r requirements.txt
```

A few dependencies cannot be pip installed. For these, do the following:
1. Clone the github repository for STScI's `crds` package to use their Calibration Reference Data System:
   ```
   git clone git@github.com:spacetelescope/crds.git
   ```
3. Go to the github repository for STScI's STIS data reduction notebook at https://github.com/spacetelescope/STIS-Notebooks/tree/main/drizpac_notebook and download the single file [copy_files.py](https://github.com/spacetelescope/STIS-Notebooks/blob/main/drizpac_notebook/copy_files.py) to the same parent directory that houses your local alicesaur repository.

4. Install STScI's `hstcal` package to use the CALSTIS executables contained therein. If you are using a conda virtual environment, you can try this via:
   ```
   conda install -c conda-forge hstcal==X.Y.Z
   ```
   where the X.Y.Z is the desired version number (choose the latest available).

   Otherwise, for those not using conda, you can install it from source by cloning the github repository:
   ```
   git clone git@github.com:spacetelescope/hstcal.git
   ```
   You will then need to follow the instructions in the hstcal/INSTALL.md file to build the package and its executables.

6. Add the full path to the hstcal `bin` directory, wherever you installed it, to your environment's PATH variable. Example for bash:
   ```
   export PATH=/$HOME/src/hstcal/bin:$PATH
   ```

### Add alicesaur to your PYTHONPATH
To run, the parent directory containing your local alicesaur repository must be in your environment's PYTHONPATH. To add it temporarily for your currently active shell, you can run either:
(for a bash or zsh shell)
```
export PYTHONPATH=/path/to/parent/directory:$PYTHONPATH
```
or (for a csh shell like csh or tcsh)
```
setenv PYTHONPATH /path/to/parent/directory:$PYTHONPATH
```
If you want this change to be permanent, you can add the above line to your general shell "rc" file, typically named something like ~/.bashrc or ~/.cshrc, or to your environment's startup script.

## Quick Start

### Example STIS reduction
You can run a basic STIS reduction from raw MAST archive files to RDI PSF-subtracted final images using the script
`scripts/main_reduce_stis.py`.

Access the script help docs with `python main_reduce_stis.py -h`.

Here is an example of a standard call to process a data set from raw archive data, as if you were processing it for the first time:
```
python alicesaur/scripts/main_reduce_stis.py --dataDir ~/path/to/data/directory/hd129590_20200413_stis/bar10 --instrument stis --pids 15653 --targ HD-129590 --obsMode bar10 --psfSubMode rdi --noCombine --inputType flt --saveFinal --do-gaia
```
where
* `--dataDir [dataDir]` sets the path to the "data directory" that will contain the data and outputs. A new directory will be created if one does not exist.
* `--instrument [{stis,acs}]` sets the instrument to either "stis" or "acs". In this case, `main_reduce_stis.py` is hardcoded to use the STIS pipeline, so this argument isn't necessary here- we only include it for the sake of clarity.
* `--pids [pids]` sets the HST program ID associated with the observation for which you want to fetch data from the archive
* `--targ [targetName]` sets the target name as written in the archive (and FITS headers)
* `--obsMode [obsMode]` sets the observation mode; typically for coronagraphic data this is the occulter position, such as "bar10", "wedgeB1.0", "wedbgeB1.8", "wedgeA1.6"
* `--psfSubMode [{rdi,adi}]` sets the preferred PSF subtraction mode; either "rdi" or "adi"
* `--inputType [{flt,flc,xft,xfc,axt,axc,sx2}]` sets the type of input data to use: "flt" (for reduction from raw archive images or local flat-field images), "flc" for flat-fielded and CTI-corrected, "xfc" for distortion- & CTI-corrected images, "axc" for aligned distortion- & CTI-corrected images (the highest level intermediate product).
* `--noCombine` flag is used to NOT combine individual science images by orbit before PSF subtracting them; we recommend using --noCombine because it typically provides the best PSF subtraction results
* `--saveFinal` flag is required to save the final PSF-subtracted image to file; without this, no final output products will be saved
* `--do-gaia` flag will run the Gaia-based astrometric reference alignment step at the end of the pipeline. See below for more details on this.

Other optional arguments and flags to the `main_reduce_stis.py` script are:
* `--noFixPix` flag will skip bad pixel fixing if the input files were already fixed, saving time
* `--noFixCTI` flag will skip the correction of charge-transfer inefficiency noise
* `--noAutoMask` will prevent automatic masking of background stars
* `--noMaskSaturation` flag will prevent masking of saturated pixels in each individual image when doing PSF subtraction and final image collapse
* `--noRadon` flag will prevent using radon transforms to identify the star center in each image
* `--noErrorMaps` flag will prevent creation of the error maps and SNR map FITS
* `--noPad` will avoid padding images into 2048x2048 pixel arrays; NOTE that this may cause pipeline crashes
* `--debug` flag will run the pipeline in debug mode with periodic breakpoints and more verbose output
* `--iterate` flag will run the pipeline with two consecutive iterations to refine the final result
* `--maskToken [token]` is the MAST login token you want to use for downloading proprietary (non-public) archive data
* `--cid [CID]` defines a custom identifier for the reduction, which will be appended to the ends of all saved files. Default is an empty string (i.e., no identifier).
* `--deltaPAMin [DELTAPAMIN]` sets the minimum PA rotation (in degrees) relative to each science image required to allow a reference image when using ADI PSF subtraction. Default is None (allow all references).
* `--spWidth [spWidth]` sets the width (in pixels) of diffraction spike masks. Default is to use the value in the info.json.
* `--ann [ann]` sets the number of annuli to use for PSF subtraction optimization. Default is 1, which is the recommended value.
* `--date-incl [DATE_INCL]` applies only if you want to specify a date range from which to include data. --date-incl defines a central UTC date of the time span to include. If given, the pipeline will only include images that were observed during the window `<--date-incl> - <--date-incl-span> to <--date-incl> + <--date-incl-span>`. Format must be YYYY-MM-DD (for which the assumed time is 00:00) or YYYY-MM-DDThh:mm. If None (default), no time constraint is applied.
* `--date-incl-span [DATE_INCL_SPAN]` also requires `--date-incl` to be given and defines the number of days before and after `--date-incl` to include in the data set. Default is 2.


The example above will download raw STIS data from MAST for target HD-129590 with the program ID 15653 and the "BAR10" occulter position. It will then perform CTI correction, dark subtraction, and flat fielding which takes up to ~10 minutes (depending on length of data set) and output new "flc.fits" files to the data directory. Then it will perform the rest of the calibration steps, outputting distortion-corrected "xfc.fits" and aligned "axc.fits" image files with individual CRSPLIT slices.

Next, it will output an intermediate product named `unified_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits` containing the unified (summed over CRSPLITs) images aligned to a common star center. Following that will be a cube of the individual PSF-subtracted images named `psfcube_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`. Then, because we gave the `--do-gaia` option, the Gaia astrometric calibration will run and output several figures.

Lastly, the final image products will be output. The default outputs to the input image directory will be:
1. The final, time-collapsed, PSF-subtracted image named `final_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`.
  * Default intensity units are DN/s.
  * Default final star location is pixel (1024, 1024) in Python coordinates; i.e., (1025, 1025) in DS9.
2. An error map with the 1-sigma noise per pixel in units of DN/s named `error_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`.
3. A signal-to-noise ratio map named `snr_[targetName]_[obsDate]_stis_axc_[obsMode]_rdi_a1.fits`.



## Alicesaur details

### Fundamentals

This section describes the basics of alicesaur that might be most relevant to users. Information here will evolve over time.

The heart of the pipeline is the `Pipeline` class defined in `alicesaur/pipeline/pipeline.py`. All of alicesaur's essential variables and functions are contained therein.
Once a `Pipeline` object is initialized with your desired input parameters, the only required step is to run the `Pipeline.run()` method.

STIS reductions specifically use the `PipelineSTIS` child class defined in `alicesaur/pipeline/stis/pipeline_stis.py`.
It inherits everything from its parent `Pipeline` and then enforces a few STIS-specific attributes such as instrument name, pixel scale, and image dimensions.

Pipeline info, warnings, and errors are partially logged to files inside "logs" in the data directory.


### Parameters

The Pipeline class looks for an `info.json` file in the top data directory; e.g., `~/path/to/data/directory/hd129590_20200413_stis/` for the example above. That file contains the basic reduction parameters for that data set, like masking, filtering choices, and PSF-subtraction optimization region definitions. This file is optional and the pipeline will run without one, but doing so will use default values for those reduction parameters which may not be optimal for every data set. In the meantime, here is a basic template if you want to try a reduction, with parameters for one Bar10 dataset and one WedgeB1.0 dataset for the same target (HD-129590).

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
            	"point_yxr": [[1122, 970, 9], [1172, 962, 9], [941, 780, 15]]
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
            	"point_yxr": [[1122, 965, 12], [896, 1071, 13], [1172, 961, 8]]
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
  * `paHW`: Half-width (in azimuth) of the radial profile sampling wedge, in [degrees]; i.e., the sampling wedge extends this many degrees in position angle (PA) on either side of the value in `paList`.
  * `rMax`: Maximum radius from the star to which the radial profile is measured and subtracted, in [pixels]
  * `postCombine`: Boolean that (if True) will do a second radial profile subtraction on the time-collapsed PSF-subtracted image at the very end of the reduction process.
* `exclude`: Series of parameters placing masks in the science (`sci`) and reference (`ref`) images before and during PSF subtraction.
  * See improcess/mask.py function doctrings for full details
  * Note: `sci` coordinates are in the final image frame
  * Note: `ref` coordinates are in the raw input reference image frame
  * `r_in` sets the inner radius (in pixels) of a circular mask centered on the target star for PSF subtraction
  * `r_out` sets the outer radius (in pixels) of a circular mask centered on the target star for PSF subtraction
  * `point_yxr` defines centers and radii of circular masks anywere in the image for PSF subtraction. The syntax is "[[center Y, center X, radius in pixels], ..."
  * `rect_cenYX_widthYX_angleDeg` defines a rectangular mask with any center, dimensions, and rotation for PSF subtraction. The syntax is "[[[center Y, center X], [width X, height Y], rotation counterclockwise from horizontal in degrees]]"
  * `pa_deg` defines a cone-shaped mask radiating outward from the target star for PSF subtraction. The syntax is "[PA minimum, PA maximum]" where angles are in degrees measured east of north (PA=0 is at +y axis)


### PSF subtraction

The default PSF subtraction method is RDI. If no suitable reference star images are found in the archive or the preexisting event directory, then alicesaur will perform ADI instead. You can also force ADI as the preferred option with the argument `--psfSubMode adi`.

In a nutshell, the RDI algorithm works on each science image one by one. For a given science image, it starts by creating a reference PSF image from a linear combination of all individual reference images where each reference image's global intensity is multiplied by a scalar and then all scaled reference images are combined via average. A least-squares optimization is then used to determine the scalar coefficients of the linear combination that minimize the residuals of the difference of the science image and the combined reference PSF image.
Once the optimal intensity scalings are determined, a second phase optimizes the alignment of the PSFs by dithering (translating) the combined reference PSF image in X and Y to minimize the PSF residuals via another least-squares optimization. After the optimal dither is found, the PSF subtraction is finished by subtracting the intensity-scaled and optimally-aligned combined reference PSF image from the science image. The algorithm then moves on to the next science image.

After all images have been PSF-subtracted, they are derotated to uniformly place north up and east left and then collapsed in time with a simple mean.

For ADI reductions, the base RDI algorithm is still used but the list of reference PSF images fed to it is identical to the list of science images. A minimum parallactic rotation criterion (default is 0.1 degrees) prevents a science image from serving as its own reference image, but no other minimum rotation threshold is used by default. To apply one, you must pass the `deltaPAMin` foption to the `main_reduce_stis.py` script or keyword argument to the `Pipeline` object with a value of the minimum PA rotation (in degrees) required to allow a reference image.


### Gaia astrometric calibration

When run with the `--do-gaia` option, alicesaur will use background stars visible in the individual PSF-subtracted images and the final collapse image that are also identified in the Gaia DR3 catalog to establish an astrometric reference frame and thereby measure the primary star's position in X,Y image coordinates as well as RA,Dec. This 

Briefly, the calibration works by first fitting the position of each background star in an image via MCMC to get its X,Y position. It then assembles a list of stars that should be within the field of view from the Gaia DR3 catalog and matches them to the observed stars. From there, it computes the image's astrometric reference frame (including pixel scale and true north angle) and runs another MCMC with the combined positions of the observed Gaia stars to determine the X,Y position of the primary (occulted) star in the image. That position and its uncertainties is written to the final image cube's SCI extension header in a series of keys:
* GAIATRA = Target (primary) star Right Ascension (in degrees) at the observation epoch
* GAIATDEC = Target star Declination (in degrees) at the observation epoch
* GAIAPLX = Target star parallax (in milliarcseconds) at the observation epoch
* GAIACENX, GAIACENY = Target star X,Y pixel coordinates in the image frame
* GAIAERRX, GAIAERRY = Target star X,Y pixel coordinate 1-sigma uncertainties in the image frame
* GAIATRN, GAIATRNE = Measured true north angle and associated 1-sigma uncertainty (in degrees)
* GAIAPSX, GAIAPSY, GAIAPSEX, GAIAPSEY = Measured pixel scale along X,Y axes and their associated 1-sigma uncertainties (in arcseconds per pixel)

The tool also outputs png figures showing the results of each step:
* "gaia_overview-*" shows the stars included in the fitting circled in blue and unused Gaia stars as magenta X's
* "gaia_psffits-*" shows the postage stamps of each star, the best-fit model, and the 2-d posterior distribution of the MCMC used to determine its X,Y position
* "gaia_mcmc-corner-*" and "gaia_mcmc-chains-*" show the corner plot and walker chains from the astrometric reference frame MCMCs


### Miscellanea

* Where image frame coordinates are used inside alicesaur, they are usually given in Y,X order to match Python's row,column indexing convention. For example, variables are defined like `coordinates = np.array([Y, X])`.
That said, some modules and external dependencies may require inputs in X,Y order, so keep that in mind if diving into the source code.