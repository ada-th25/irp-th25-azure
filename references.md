This file contains references and AI disclosures specific to the codebase in this repository.

**Academic Citations:** For bibliography and references related to the written report, please see [`deliverables/references.bib`](./deliverables/references.bib).

# Code Repositories

* **HAUNet_RSISR**
    * **Date Used:** 23/06/2026
    * **Repository:** https://github.com/likakakaka/HAUNet_RSISR
    * **Paper:** J. Wang, B. Wang, X. Wang, Y. Zhao and T. Long, "Hybrid Attention-Based U-Shaped Network for Remote Sensing Image Super-Resolution," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-15, 2023, Art no. 5612515, doi: 10.1109/TGRS.2023.3283769.
    * **Usage:** Used their `codes/model/haunet.py` to build the core HAUNet architecture.

# Packages & Dependencies

* **NumPy**
    * **Version/Date:** v2.4 / 08/06/2026
    * **Repository:** https://github.com/numpy/numpy
    * **Paper/Documentation:** C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, Sep. 2020, doi: 10.1038/s41586-020-2649-2.
    * **Usage:** Multidimensional array manipulations and structural data transformations.

* **PyTorch**
    * **Version/Date:** v2.12 / 23/06/2026
    * **Repository:** https://github.com/pytorch/pytorch
    * **Usage:** Core deep learning framework for building, training, and executing neural network architectures.

* **scikit-learn**
    * **Version/Date:** v1.3 / 27/06/2026
    * **Repository:** https://github.com/scikit-learn/scikit-learn
    * **Paper/Documentation:** F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
    * **Usage:** Used for evaluation metrics, dataset splitting, or specific data preprocessing pipelines.

# Online Resources

* **Medium**
    * **Date Used:** 30/06/2026
    * **URL:** https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
    * **Author/Organisation:** Amit Yadav
    * **Usage:** Used their implementation of focal and dice loss.

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

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 17/06/2026
    * **Scope:** Generate the initial data loader script in `src/data/wildfire_dataset.py`.
    * **Prompt:** (pasting existing data pipeline) "Generate a data loader script to load the dataset inside the GCP bucket. The shapes of the data are as follows: Landsat pairs: GOES (3, 15, 15), covariates (2, 256, 256), target Landsat (1, 256, 256). VIIRs pairs: GOES (3, 15, 15), covariates (2, 80, 80), target VIIRS (1, 20, 20). Include a function that filters GOES data where it's all zero masks."

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

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 20/06/2026
    * **Scope:** Edit `src/data/wildfire_dataset.py` to handle data splitting by reading a file with designated splits and the new changes with GOEs context ratio = 1. The script returns a (5, H, W) data that is ready to run into model. The new edit also adds a helper function `compute_pos_weight()` to return a tensor to use for BCEWithLogitsLoss.
    * **Prompt:** (pasting existing wildfire_dataset) "Edit the script so that it handles data splitting, including a helper function that scans the training split manifests to estimate the ratio of background to fire pixels, returning a scalar tensor for use in BCEWithLogitsLoss."

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 25/06/2026
    * **Scope:** Modify the original HAUNet (`src/models/haunet/haunet.py`) to fit the requirements for the wildfire downscaling model.
    * **Prompt:** (pasting the original HAUNet model) "Make edits to this to fit with my wildfire downscaling task (input GOES+covariates -> VIIRS or Landsat). This includes removing the residual learning, accepting more input channels (5), outputting 1 channel."
    * **Human Modification:** Change up_scale from 4 to 1, as spatial upscaling will occur before the network.

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 26/06/2026
    * **Scope:** Modify the original HAUNet (`src/models/haunet/haunet.py`) to fit the requirements for the wildfire downscaling model.
    * **Prompt:** (pasting the original HAUNet model) "Make edits to this to fit with my wildfire downscaling task (input GOES+covariates -> VIIRS or Landsat). This includes removing the residual learning, accepting more input channels (5), outputting 1 channel."
    * **Human Modification:** Change up_scale from 4 to 1, as spatial upscaling will occur before the network.

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 27/06/2026
    * **Scope:** Generate function for data splitting (`src/data/split_data.py`) where it assigns each data to percentile bins based on number of active fire pixels. This assignent is used to ensure the split produces balanced dataset.
    * **Prompt:** (pasting the existing code for wildfire_dataset and dataset) "Write a function that splits the data id into trining/test/val, where the data is assigned into quartile bins based on number of active fire pixels for a balanced dataset. Then split the data according to the bins for balanced dataset."
    * **Human Modification:** Modified to use sklearn's StratifiedShuffleSplit after assigning bin categories.

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 27/06/2026
    * **Scope:** Generate `main()` for `src/data/split_data.py` to implement argument parser to handle splits for viirs or landsat or both.
    * **Prompt:** (pasting existing code in the script) "Generate def main() function that takes either viirs, landsat or both as argument to run different splitting, and another argument for overwriting to regenrate the split."

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 28/06/2026
    * **Scope:** Generate test script (`tests/test_wildfire_dataset`) to test `wildfire_dataset` pipeline.
    * **Prompt:** (pasting code in wildfire_dataset) "Generate a test script to test whether the data processed is correct, including shape checking, checking for zeros etc."

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 28/06/2026
    * **Scope:** Generate test script (`tests/test_splits`) to test if splitting in `split_dataset.py` is done correctly.
    * **Prompt:** (pasting code in split_dataset) "Generate a test script to test whether the data split is correct."

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 30/06/2026
    * **Scope:** Generate `losses.py` where it contains loss calculations for focal loss, dice loss and weighted sum of both.
    * **Prompt:** "Given the model is trained on raster images, generate a script that contains classes with the following losses: Focal loss, dice loss, and focal dice loss (weighted sum of both losses)."
    * **Human Modification:** Modified alpha (0.75) for focal loss to give more weights to active pixels over the background.

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 30/06/2026
    * **Scope:** Generate script that calculates evaluation metrics, including the IOU, IPSNR, F1, and precision and recall (`metrics.py`).
    * **Prompt:** "Generate a metrics script that takes target and output fire mask image, and computes the IOU, IPSNR, F1 and precision and recall."

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 01/07/2026
    * **Scope:** Generate training script to run model training (`train.py`).
    * **Prompt:** (pasting existing HAUNet model) "Generate a training script that trains the model, and logs the metrics calculated during training. Include W&B logging"
    * **Human Modification:** Edit the script to integrate my current metrics script.

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 07/07/2026
    * **Scope:** Alter dataset pipeline to rasterise VIIRS locally, and to fetch GOES data from public GCP bucket. The rasterised VIIRS is downloaded so that data is evenly collected across the years (by month) and the number of active pixels are distributed evenly. The script `goes_local_fetch.py` is generated.
    * **Prompt:** (pasting existing pipeline) "Change the pipeline so that instead of rasterising VIIRS via GEE, rasterise them locally, and also fetch GOES data from public GCP bucket for GOES 16/17. Also, collect VIIRs data so that data is evenly collected across the years (by month) and the number of active pixels are distributed evenly."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 08/07/2026
    * **Scope:** Generate `goes_local_fetch` to fetch GOES data from public GCP bucket. This includes finding the closest GOEs data, reprojection, uploading as tiff, filtering for all zero data etc.
    * **Prompt:** (pasting existing pipeline) "Create a script to replace the GEE GOES fetching code, so that it obtains the GOES data from the public GCP bucket, and does necessary searching and processing, such as reprojection to match with the original GEE pipeline."
    * **Human Modification:** Delete unecessary processing, and fix datetime bugs.

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 09/07/2026
    * **Scope:** Debug `goes_local_fetch` to automatically evict cache to prevent dataset pipeline from failing. This adds the function `_evict_lru_if_needed`. And also fixes threading errors.
    * **Prompt:** (pasting existing pipeline and error message) "Debug `goes_local_fetch` to automatically evict cache to prevent dataset pipeline from failing by capping total cache size and evict older files before new ones are downloaded."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 10/07/2026
    * **Scope:** Fix the code crash caused by threading.
    * **Prompt:** (pasting existing pipeline and error message) "Debug the code to prevent crashed caused by threading."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 12/07/2026
    * **Scope:** Generate code that calculates Tversky loss in `losses.py`.
    * **Prompt:** (pasting existing loss script): "Add Tversky loss in the script. Set beta > alpha to penalise missed positives."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 12/07/2026
    * **Scope:** Debug the forking issue in the `wildfire_dataset` by checking each worker process builds its own
        fresh client on first access, rather than reusing one inherited from the parent at fork time.
    * **Prompt:** (pasting existing code and error message): "Debug the workers forking issue in the wildfire_dataset script."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 13/07/2026
    * **Scope:** Change `wildfire_dataset` so that the output VIIRS will be evaluated at their native 20x20 pixel dimensions, and dynamically process the data so that in the cascade model, it will accept the output VIIRS from the stage 1 model and upsamples to covariates' dimensions.
    * **Prompt:** (pasting existing code): "output VIIRS will be evaluated at their native 20x20 pixel dimensions, and dynamically process the data so that in the cascade model, it will accept the output VIIRS from the stage 1 model and upsamples to covariates' dimensions."