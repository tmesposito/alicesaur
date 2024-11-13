from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel, detect_sources
from astropy.convolution import convolve
import numpy as np
import pandas as pd
import os

def main_masking(self):
    """
    Main function for loading FITS data, processing it, generating a segmentation map, 
    excluding a central region, saving masked pixel coordinates, and saving the segmentation map as a FITS file.
    """
    data = None
    original_filename = None

    for filename in os.listdir(self.dataDir):
        if filename.startswith('final_') and filename.endswith('.fits'):
            file_path = os.path.join(self.dataDir, filename)
            with fits.open(file_path) as file:
                data = file[0].data[0] 
            original_filename = filename
            print(f"Loaded data from {filename}")
            break

    if data is None:
        print("No final FITS images found in the directory. Skipping masking process.")
        return None

    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (250, 250), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    final_data = data - bkg.background
    threshold = 3 * np.median(bkg.background_rms)
    
    kernel = make_2dgaussian_kernel(3, size=7)
    convolved_data = convolve(final_data, kernel)
    seg_map = detect_sources(convolved_data, threshold, npixels=10)
    seg_map.data[seg_map.data > 0] = 1

    exclude_center = (1024, 1024)
    exclude_radius = 50
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    distance_from_center = np.sqrt((x - exclude_center[0])**2 + (y - exclude_center[1])**2)
    exclusion_mask = distance_from_center <= exclude_radius
    seg_map.data[exclusion_mask] = 0

    all_pixels = np.argwhere(seg_map.data == 1)
    df = pd.DataFrame(all_pixels, columns=['y', 'x'])
    csv_file_path = os.path.join(self.dataDir, 'masked_pixel_yx.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Saved masked pixel coordinates to {csv_file_path}")

    output_file = f"segmap_{original_filename.split('final_')[1]}"
    output_path = os.path.join(self.dataDir, output_file)
    hdu = fits.PrimaryHDU(seg_map.data)
    hdu.writeto(output_path, overwrite=True)
    print(f"Segmentation map saved as {output_path}")

    return seg_map

