This file contains references and AI disclosures specific to the codebase in this repository.

**Academic Citations:** For bibliography and references related to the written report, please see [`deliverables/references.bib`](./deliverables/references.bib).

# Code Repositories

* **HAUNet_RSISR**
    * **Date Used:** 23/06/2026
    * **Repository:** https://github.com/likakakaka/HAUNet_RSISR
    * **Paper:** J. Wang, B. Wang, X. Wang, Y. Zhao and T. Long, "Hybrid Attention-Based U-Shaped Network for Remote Sensing Image Super-Resolution," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-15, 2023, Art no. 5612515, doi: 10.1109/TGRS.2023.3283769.
    * **Usage:** Used their `codes/model/haunet.py` to build the core HAUNet architecture.

# AI Usage

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 10/06/2026
    * **Scope:** Generated the initial test pipeline code in data processing pipeline in `src/data/dataset.py`. Also generated code to be used for testing/visualising the initial test pipeline in `notebooks/exploratory-notebook.ipynb`. NOTE: This is not used in the final pipeline. This is purely for test/exploratory purposes only.
    * **Prompt:** (with information on Pereira et al. dataset pasted) "Create a function that fetches GOES data corresponding to the Pereira et al. Landsat dataset. And generate a code to copy and paste to an exploratory ipynb file to test and check the data that's been stored in GCP bucket."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 13/06/2026
    * **Scope:** Generated the initialisation code for the data processing pipeline in `src/data/dataset.py`.
    * **Prompt:** "Create initialisation for the data pipeline, including initialising the buckets in GCP, the resolution of the data, extent covariates (COPERNICUS/DEM/GLO30 and ESA/WorldCover/v100)"
    * **Human Modification:** Added context ratio to make GOES/VIIRS image bounds larger to better capture spatial context.

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 13/06/2026
    * **Scope:** Generate the code for the main loop that runs the dataset pipeline in `src/data/dataset.py`. Including `process_flat_zip` function to handle zip file processing.
    * **Prompt:** (pasting existing data pipeline, and the file structure of the Pereira et al. Landsat files) "Generate the main loop that runs the dataset pipeline. Also create a function that process the zip files in the Pereira et al. dataset."
    * **Human Modification:** Add processing of manually annotated Landsat mask data.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 13/06/2026
    * **Scope:** Generated data acquisition pipeline in `src/data/dataset.py`, where it acquires GOES and covariates data that corresponds to the same location (and time, for GOES) as the VIIRS and Landsat data.
    * **Prompt:** (pasting existing code, inclusing initialisation and utility functions) "Generate a function that uses this code to fetch GOES and covariates data that matches spatially and temporally. Make it so that it filters for data where there are minimum of 2 fire pixels in the Landsat data. Use the context ROI as the bounds."
    * **Human Modification:** Add code that writes relevant information about the data using the `write_manifest()` function.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 16/06/2026
    * **Scope:** Generated the core loop (`process_public_zip`, `process_flat_zip`) for unzipping and accessing the Landsat files in `src/data/dataset.py`. The loop also processed and filters for `MIN_ACTIVE_PIXELS`.
    * **Prompt:** (pasting existing code, including link to the Pereira et al. dataset) "Create a function that unzips and processes the Landsat master archive containing raw fire scenes, filter out data based on the minimum active pixels, extract spatial metadata and checks their bounding boxes. Use Multithreaded Task Dispatching for faster processing."
    * **Human Modification:** Modified so that it only locates the files contianing `"intersection"`.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 16/06/2026
    * **Scope:** Generated the function `fetch_goes_fire_features` in `src/data/dataset.py`, where it takes the ROI, centre, and date/time and takes the closest matching GOES patch.
    * **Prompt:** (pasting existing code) "Generate a function that takes the ROI, lon centre, and date/time to get the corresponding data for GOES, calculating the thermal index."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 16/06/2026
    * **Scope:** Generated `process_landsat_sample` function in `src/data/dataset.py`. This generates unique paths for each files, checks for existing GCS blob and skips if it exists. The function extracts the center longitude and latitude from the sample's metadata and maps them to the appropriate local UTM CRS and create a ROI boundng box. Then uses the information to extract the corresponding GOES and covariates.
    * **Prompt:** (pasting existing code) "Create a function that processes the Landsat data, gets a 7680m ROI, and gets the corresponding GOES and covariates with same ROI"
    * **Human Modification:** Add manifest writing.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 16/06/2026
    * **Scope:** Generated the code for processing VIIRS data and acquiring/processing the corresponding GOES and covariate data (`process_viirs_cluster`, `run_viirs_branch`). This includes determining the projections, ROI, rasterisation of VIIRS data, and extracting the GOES and covariates data, and saving them to GCP.
    * **Prompt:** (pasting existing code, including the Landsat/GOES/covairates acquisition and processing branch) "Generate code similar to the Landsat branch, but for VIIRS. The VIIRS data are in csv and needs to be rasterised. Ensure the physical extent covered are consistent with the context extent ratio, and consistent between the input datasets."
    * **Human Modification:** Add manifest writing. Also add code after loading VIIRS data to parse to datetime, and define time window for grouping.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 17/06/2026
    * **Scope:** Generated data exploration code in `notebooks/explore-dataset.ipynb`, where it checks for number of active pixel distributions and the number of files contained in Pereira et al. dataset. It also counts the number of raster data we get from the VIIRS csv data, and the number of active pixels in the rasterised VIIRS.
    * **Prompt:** (pasting existing data pipeline for reference) "Create a code to paste to an ipynb notebook to explore the distribution of the number of active pixels, and the number of masks contained in each files. Also, create code to count how much raster data we get, and the number of active pixels from the VIIRS data."
    * **Human Modification:** Add the gdrive ID for the relevant folders.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 18/06/2026
    * **Scope:** Generated data exploration code in `notebooks/explore-dataset.ipynb`, where it plots the Landsat/GOES/Covariates and VIIRS/GOES/Covariates data.
    * **Prompt:** (pasting existing data pipeline) "Create a code to paste in an ipynb notebook to check and visualise the data so that you plot for Landsat/GOES/Covariates and VIIRS/GOES/Covariates dataset pairs."
    * **Human Modification:** Change dataset path names.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 18/06/2026
    * **Scope:** Debug the issue with the blank GOES data fetched for the VIIRS target. Generated diagnostic code and the corresponding fix. Bug was fixed by fixing the geographical split to enforce GEE to use either GOES-16 or GOES-17 accordingly based on their geographical coverage.
    * **Prompt:** (pasting existing data pipeline and the plot code and outputs) "The GOES data for the VIIRS target is black. Generate a diagnostic code to identify the problem, and give me the fix."

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 18/06/2026
    * **Scope:** Generate test script `tests/test_dataset.py` to test the dataset that's downloaded in GCP.
    * **Prompt:** (pasting existing data pipeline) "Generate a test script to verify the data stored using this pipeline."