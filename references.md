This file contains references and AI disclosures specific to the codebase in this repository.

**Academic Citations:** For bibliography and references related to the written report, please see [`deliverables/references.bib`](./deliverables/references.bib).

# AI Usage

* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 10/06/2026
    * **Scope:** Generated the initial test pipeline code in data processing pipeline in `src/data/dataset.py`. Also generated code to be used for testing/visualising the initial test pipeline in `notebooks/exploratory-notebook.ipynb`.
    * **Prompt:** (with information on Pereira et al. dataset pasted) "Create a function that fetches GOES data corresponding to the Pereira et al. Landsat dataset. And generate a code to copy and paste to an exploratory ipynb file to test and check the data that's been stored in GCP bucket."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 13/06/2026
    * **Scope:** Generated the initialisation code for the data processing pipeline in `src/data/dataset.py`.
    * **Prompt:** "Create initialisation for the data pipeline, including initialising the buckets in GCP, the resolution of the data, extent covariates (COPERNICUS/DEM/GLO30 and ESA/WorldCover/v100)"
    * **Human Modification:** Added context ratio to make GOES/VIIRS image bounds larger to better capture spatial context.

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 13/06/2026
    * **Scope:** Generated data acquisition pipeline in `src/data/dataset.py`, where it acquires GOES and covariates data that corresponds to the same location (and time, for GOES) as the VIIRS and Landsat data.
    * **Prompt:** (pasting existing code, inclusing initialisation and utility functions) "Generate a function that uses this code to fetch GOES and covariates data that matches spatially and temporally. Make it so that it filters for data where there are minimum of 2 fire pixels in the Landsat data. Use the context ROI as the bounds."
    * **Human Modification:** Add code that writes relevant information about the data using the `write_manifest()` function.