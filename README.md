# alicesaur
Archival Legacy Investigation of Circumstellar Environments: STIS and ACS Uniform Reduction

## Example STIS reduction
You can run a basic STIS reduction from sx2 to RDI PSF-subtracted final image using the script
`scripts/main_reduce_stis.py`. The default outputs to the input image directory will be:
1. The final, time-collapsed, PSF-subtracted image named `[targetName]_[obsDate]_stis_[obsMode]_rdi_a1_final.fits`.
  * Default intensity units are DN/s.
  * Default final star location is pixel (1024, 1024) in Python coordinates; i.e., (1025, 1025) in DS9.
2. An error map with the 1-sigma noise per pixel in units of DN/s named `error_[targetName]_[obsDate]_stis_[obsMode]_rdi_a1_final.fits`.
3. A signal-to-noise ratio map named `snr_[targetName]_[obsDate]_stis_[obsMode]_rdi_a1_final.fits`.

The way to run it is as follows:

```
python alicesaur/scripts/main_reduce_stis.py --dataDir ~/path/to/sx2_fits/hd115600_20200210_stis/bar10
--obsMode bar10 --psfSubMode rdi --targ HD-115600 --saveFinal --noCombine [--noErrorMaps --noRadon
--noFixPix --noPad --spWidth [int] --bgCen [int int]]
```
where
* `dataDir` is the path to the directory containing your input sx2 FITS files ("raw" data),
* `obsMode` is the observation mode; either "bar10", "wedgeB1.0", "wedbgeB1.8", "wedgeA1.6"
* `psfSubMode` should always be "rdi" at this point
* `targ` is the target name as written in the FITS headers
* `--saveFinal` is required to save the final PSF-subtracted image to file
* `--noCombine` is used to NOT combine individual science images by orbit before PSF subtracting them
* `--noErrorMaps` is used to NOT save the error map and SNR map FITS

**_WARNING:_** The Pipeline class currently expects an `info.json` file just above the directory
containing the input FITS files and may not run a reduction without one. That file contains
the basic reduction parameters, like filtering choices and PSF-subtraction optimization
region definitions. This will be made optional soon. In the meantime, here is a basic
template if you want to try a reduction, with parameters for one Bar10 dataset and one WedgeB1.0 dataset for the same target (HD 115600).

```
{
    "targetName": "HD-115600",
    "psfRefName": "PSF-HD-115778",
    "obsLogPath": "/Users/Tom/Research/data/hst/hd115600_20200210_stis/obs_log_hd115600_20200210_stis.csv",
    "diskPA_deg": 204,
    "bar10": {
	"bgCen_yx": "360 590",
        "bgCenRef_yx": "500 883",
        "bgRadius": 40,
        "spWidth": 5,
	"radProfSub": {
	    "paList": [115],
	    "paHW": 20,
	    "rMax": 180
	},
        "exclude": {
            "sci": {
                "r_in": 14,
                "r_out": 90,
                "pa_deg": [],
                "rect_cenYX_widthYX_angleDeg": [],
                "point_yxr": [[817, 875, 11], [773, 922, 14], [745, 883, 8], [716, 852, 8], [787, 1088, 10]]
            },
            "ref": {
            }
        }
    },
    "wedgeb1.0": {
        "bgCen_yx": "695 952",
    	"bgCenRef_yx": "762 923",
    	"bgCenFinal_yx": "656 979",
    	"bgRadius": 55,
    	"spWidth": 10,
	"radProfSub": {
	    "paList": [115],
	    "paHW": 20,
	    "rMax": 180,
            "postCombine": "True"
	},
    	"exclude": {
    	    "sci": {
            	"r_in": 14,
            	"r_out": 60,
            	"pa_deg": [[23, 110]],
            	"rect_cenYX_widthYX_angleDeg": [[[1092, 872], [98, 227], 340]],
	        "point_yxr": [[1055,  962, 16], [957, 1034, 7], [1082, 1017, 7], [980, 1058, 7], [1100, 915, 16], [1107,  895, 7], [998,  892, 12], [1027,  923, 12], [1070, 1126, 12], [981, 1151, 12], [1114,  846, 8], [1091, 1035, 9], [913,  850, 9], [1075,  902, 8]]
            },
            "ref": {
            	"point_yxr": [[457, 305, 7]]
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
