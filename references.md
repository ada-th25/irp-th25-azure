This file contains references and AI disclosures specific to the codebase in this repository.

**Academic Citations:** For bibliography, papers/articles, and references related to the written report, please see [`deliverables/references.bib`](./deliverables/references.bib).

# Code Repositories

* **HAUNet_RSISR**
    * **Date Used:** 23/06/2026
    * **Repository:** https://github.com/likakakaka/HAUNet_RSISR
    * **Paper:** J. Wang, B. Wang, X. Wang, Y. Zhao and T. Long, "Hybrid Attention-Based U-Shaped Network for Remote Sensing Image Super-Resolution," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-15, 2023, Art no. 5612515, doi: 10.1109/TGRS.2023.3283769.
    * **Usage:** Used their `codes/model/haunet.py` to build the core HAUNet architecture.

# Packages & Dependencies

* **NumPy**
    * **Version/Date:** v2.4
    * **Repository:** https://github.com/numpy/numpy
    * **Paper/Documentation:** C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, Sep. 2020, doi: 10.1038/s41586-020-2649-2.
    * **Usage:** Multidimensional array manipulations and structural data transformations.

* **PyTorch**
    * **Version/Date:** v2.12
    * **Repository:** https://github.com/pytorch/pytorch
    * **Usage:** Core deep learning framework for building, training, and executing neural network architectures.

* **scikit-learn**
    * **Version/Date:** v1.3
    * **Repository:** https://github.com/scikit-learn/scikit-learn
    * **Paper/Documentation:** F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
    * **Usage:** Used for evaluation metrics, dataset splitting, or specific data preprocessing pipelines.

* **GeoPandas**
    * **Version/Date:** v1.1
    * **Repository:** https://github.com/geopandas/geopandas
    * **Paper/Documentation:** Kelsey Jordahl, Joris Van den Bossche, Martin Fleischmann, Jacob Wasserman, James McBride, Jeffrey Gerard, … François Leblanc. (2020, July 15). geopandas/geopandas: v0.8.1 (Version v0.8.1). Zenodo. http://doi.org/10.5281/zenodo.3946761
    * **Usage:** Vector data processing, geometric manipulations, and coordinate reference system (CRS) transformations.

* **Google Earth Engine Python API (earthengine-api)**
    * **Version/Date:** v1.7
    * **Repository:** https://github.com/google/earthengine-api
    * **Paper/Documentation:** N. Gorelick *et al.*, "Google Earth Engine: Planetary-scale geospatial analysis for everyone," *Remote Sensing of Environment*, 2017, doi: 10.1016/j.rse.2017.06.031.
    * **Usage:** Querying, filtering, and executing planetary-scale cloud computations on geospatial datasets.

* **Xarray**
    * **Version/Date:** v2026.7
    * **Repository:** https://github.com/pydata/xarray
    * **Paper/Documentation:** S. Hoyer and J. Hamman, "xarray: N-D labeled Arrays and Datasets in Python," *Journal of Open Research Software*, 2017, doi: 10.5334/jors.148.
    * **Usage:** Managing multi-dimensional gridded datasets with labeled dimensions, coordinates, and metadata.

* **rioxarray**
    * **Version/Date:** v0.22.0
    * **Repository:** https://github.com/corteva/rioxarray
    * **Paper/Documentation:** Corteva Open Source, "rioxarray: Geospatial xarray extension powered by rasterio," https://corteva.github.io/rioxarray.
    * **Usage:** Geospatial raster operations on Xarray structures, including clipping, reprojecting, and exporting to GeoTIFF formats.

* **SciPy**
    * **Version/Date:** v1.18.0
    * **Repository:** https://github.com/scipy/scipy
    * **Paper/Documentation:** P. Virtanen *et al.*, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2.
    * **Usage:** Advanced mathematical algorithms, statistics, and scientific calculations complementing NumPy operations.

# Online Resources

* **Medium**
    * **Date Used:** 30/06/2026
    * **URL:** https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
    * **Author/Organisation:** Amit Yadav
    * **Usage:** Used their implementation of focal and dice loss.

* **Medium**
    * **Date Used:** 12/07/2026
    * **URL:** https://medium.com/@saba99/tversky-loss-902f5f8cc35f
    * **Author/Organisation:** Saba Hesaraki
    * **Usage:** Understand the use of Tversky loss in context of image segmentation.

* **Medium**
    * **Date Used:** 15/07/2026
    * **URL:** https://sh-tsang.medium.com/review-cornernet-detecting-objects-as-paired-keypoints-object-detection-ffb23026291b
    * **Author/Organisation:** Sik-Ho Tsang
    * **Usage:** Understand how the variation of focal loss used in CornerNet works.

* **Wikipedia**
    * **Date Used:** 08/08/2026
    * **URL:** https://en.wikipedia.org/wiki/Otsu%27s_method
    * **Author/Organisation:** Wikipedia
    * **Usage:** Understand how Otsu thresholding works.

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

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 14/07/2026
    * **Scope:** Debug the OOM error by accumulating the the gradients over a set number of of batches instead of updating after every batch. And do a gradient clipping to limit the magnitude of the gradients. This edits `src/train.py`
    * **Prompt:** (pasting existing train script and error message): "Debug the OOM error by accumulating the the gradients over a set number of of batches instead of updating after every batch. And do a gradient clipping to limit the magnitude of the gradients."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 14/07/2026
    * **Scope:** Generate code that calculates Two Branch Loss and Local RMSE Loss in `src/losses.py` from Badhan et al. Autoencoder Architecture.
    * **Prompt:** (uploaded a copy of Badhan et a.) "Generate a code that calculates Two branch loss and the Local RMSE loss from this paper."
    * **Human Modification:** Implement it as a class inside the loss script.

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 15/07/2026
    * **Scope:** Create a train/val loss plotting script `src/utils/plot_loss.py` and edit the CSV logger in `src/train.py` to ensure it writes the logs on new files if training was rerun.
    * **Prompt:** (pasting existing train script): "Create a train/val loss plotting script `src/utils/plot_loss.py` and edit the CSV logger in `src/train.py` to ensure it writes the logs on new files if training was rerun."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 15/07/2026
    * **Scope:** Add implementation of the variation of focal loss from CornerNet (`GaussianFocalLoss` in `src/losses.py`). And add a function that converst the binary fire data into a Gaussian heatmap.
    * **Prompt:** (pasting existing `wildfire_dataset` code and link to the CornerNet paper) "Implement a Gaussian Focal Loss based on this paper, and edit wildfire_dataset to add a Gaussian heatmap function that converts every pixel with pixel with value 1.0, a Gaussian distribution is rendered around it.
    * **Human Modification:** Implement the loss as a class inside the loss script.

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 18/07/2026
    * **Scope:** Create a predictions plotting script `src/utils/plot_predictions.py` 
    * **Prompt:** "Create a prediction plotting script that plots the input GOES the covariates, the true and predicted output."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 22/07/2026
    * **Scope:** Create autoencoder model (`src/models/autoencoder.py`) based on the model in the paper "Deep Learning Approach to Improve Spatial Resolution of GOES-17 Wildfire Boundaries Using VIIRS Satellite Data" by Badhan et al.
    * **Prompt:** (Uploaded the Badhan et al. paper) "Generate the autoencoder model based on this paper."
    * **Human Modification:** Accept covariates input, remove the two-branch structure.

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 25/07/2026
    * **Scope:** Fix `src/data/wildfire_dataset.py` to parallelise the downloads for faster execution via thread pooling. It also filters the data based on split_ids (if given) so it does not load everything first.
    * **Prompt:** (pasting existing wildfire_dataset) "Parallelise the downloads for faster execution. Also filter the data based on split_ids (if given) so it does not load everything first."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 25/07/2026
    * **Scope:** Add new metrics: Area Under the Precision-Recall Curve (AUPRC) that calculates precision and recall at different thresholds, and centroid distance that calculates Euclidean distance in m between the location of the predicted heatmap and the predicted heatmap.
    * **Prompt:** "Generate a code that calculates the Area Under the Precision-Recall Curve (AUPRC) that calculates precision and recall at different thresholds, and centroid distance that calculates Euclidean distance in m between the location of the predicted heatmap and the predicted heatmap. This metrics will be calculated to assess the predicted heatmaps of an imbalanced fire dataset."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 27/07/2026
    * **Scope:** Generate a script that calculates fractinon of active pixels in the target data and prints outs the stats.
    * **Prompt:** (pasting existing data pipeline) "Generate a script that fetches data from the GCP bucket, and calculates the number of active pixels in the target data and print out the results."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 27/07/2026
    * **Scope:** Generate a post processing script to process the final Landsat output into a binary fire mask. They can be done either by fixed thresholding or percentile-based. Includes a function that finds the best threshold value from the validation data based on F1 score.
    * **Prompt:** "Generate a post-processing script that applies either fixed threshold or percentile threshold to post process the probability map into a binary output. Add a function that finds the best threshold value from the validation data based on F1 score."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 27/07/2026
    * **Scope:** Generate a script that load the trained cascaded downscaling stage 1 model, takes the Landsat dataset, uses GOES+covariates as input to generate the predicted VIIRS data, and saves them in the GCP bucket. This is so that it can be used as input for training the cascaded downscaling stage 2 model.
    * **Prompt:** (pasting existing training script) "Generate a python script that load the trained cascaded downscaling stage 1 model, takes the Landsat dataset, uses GOES+covariates as input to generate the predicted VIIRS data, and saves them in the GCP bucket."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 28/07/2026
    * **Scope:** Optimise the `src/train.py` script so the weighted sampler builder is faster
    * **Prompt:** (pasting existing training script) "Optimise this script so that the load_positive_pixel_counts run faster."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 29/07/2026
    * **Scope:** Generate `src/utils/compute_dem_stats.py` that mirrors `src/utils/compute_goes_stats.py` so it calculates DEM stats to be used for standardisation. Only calculated with train data.
    * **Prompt:** (pasting goes stat calculation script) "Generate a script that calculates the mean and standard deviation of the DEM data in the train set only."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 29/07/2026
    * **Scope:** Generate `src/evaluate.py` where trained model is laoded and inferenced using test dataset, then metrics are calculated and logged into a csv file.
    * **Prompt:** (pasting train and metrics script) "Write a script that evaluates the trained model by inferencing the test dataset, and calculates the metrics. Write the results in a csv file."

* **Tool:** Google Gemini (3.6)
    * **Date Used:** 30/07/2026
    * **Scope:** Generate draft section for draft README file, including Quickstart.
    * **Prompt:** (pasting existing draft README) "Add suggested section for repository README"
    * **Human Modification:** Replace placeholders with my repo/project info.

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 31/07/2026
    * **Scope:** Change `src/train.py` so that it saves best checkpoint based on multiple metrics.
    * **Prompt:** (pasting train script) "Change this code so that I can choose more than one metric to keep the 'best'. For example, if I want to checkpoint ones where either auprc or soft_iou is better, it saves it as best_auprc or best_soft_iou etc"

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 03/08/2026
    * **Scope:** Add a function in `src/evaluate` to add a baseline metric for all-zero prediction and also for prevalence-rate cases so it can be compared against my model metrics.
    * **Prompt:** (pasting evaluation script) "Add a function so that it calculates all the metrics in the evaluation loop so that it also calculates the all-zero prediction and also prevalence-rate cases so it can be comapred against my main results."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 04/08/2026
    * **Scope:** Edit `src/utils/plot_predictions.py` so that it uses `use_cov_pyramid`, and ensure the model is built from the checkpoint's saved config.
    * **Prompt:** (pasting existing plotting script, model script and YAML file) "Edit this so that is uses use_cov_pyramid. Also ensure the model is built from the checkpoint's saved config."

* **Tool:** Google Gemini (3.6)
    * **Date Used:** 05/08/2026
    * **Scope:** Change `src/data/wildfire_dataset.py` to be able to configure dilation kernel size.
    * **Prompt:** (pasting existing wildfire_dataset) "Change so that user can configure the kernel size for dilation"
    * **Human Modification:** Set different default kernel size for Landsat and VIIRS (5 and 3 respectively).

* **Tool:** Google Gemini (3.6)
    * **Date Used:** 05/08/2026
    * **Scope:** Update plotting code in `notebooks/explore_dataset.ipynb` to choose random data patch. Additionally plots random VIIRS/Landsat data with original vs dilated targets.
    * **Prompt:** (pasting existing plotting code in notebook, and the dilation logic in wildfire_dataset) "Change this so that it plots random data ids, and also plot all the GOES channels. Additionally, plot original vs dilated (for both Landsat and VIIRS)."
    * **Human Modification:** Rewrite configuration to match GCP bucket details.

* **Tool:** Google Gemini (3.6)
    * **Date Used:** 07/08/2026
    * **Scope:** Write code that loads JSON files from `logs/` and explore the results summary data. This is written in `notebooks/explore_results_summary.ipynb`
    * **Prompt:** (uploading examlple JSON file) "Write code to explore/visualise these results summaries. Primary metric include auprc, soft iou etc. but omit the centroid distance metric."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 09/08/2026
    * **Scope:** Edit `src/evaluate.py`, `src/data/generate_stage1_predictions.py` and `src/utils/plot_predictions.py` to fetch checkpoint from GCS bucket if not provided locally.
    * **Prompt:** (pasting existing prediction and evaluation scripts) "Edit this so that if the checkpoints are not found locally, it fetches it from the GCS bucket."

* **Tool:** Anthropic Claude (Sonnet 5)
    * **Date Used:** 12/08/2026
    * **Scope:** Write code in `notebooks/explore_results_summary.ipynb` that plots the predictions of the models.
    * **Prompt:** (pasting the plotting script for reference) "Write code based on this that 1. builds model, loads checkpoint, runs inference on a chosen batch of samples (random or by explicit sample_id) 2. plot input reference 3. plot per-experiment 4. plots the cascade flow"