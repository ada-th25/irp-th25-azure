This file contains references, licenses, and AI disclosures specific to the codebase in this repository.

**Academic Citations:** For bibliography and references related to the written report, please see [`deliverables/references.bib`](./deliverables/references.bib).

# AI Usage
    
* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 10/06/2026
    * **Scope:** Generated the initial rasterization code in data processing pipeline in `src/data/dataset.py`. Also generated code to be used for testing the initial data pipeline in `notebooks/exploratory-notebook.ipynb`.
    * **Prompt:** (with initial data pipeline pasted) "Add a function to rasterize the VIIRS data. And generate a code to copy and paste to an exploratory ipynb file to test and check the data that's been stored in GCP bucket."

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 13/06/2026
    * **Scope:** Generated the initialisation code for the data processing pipeline in `src/data/dataset.py`.
    * **Prompt:** "Create initialisation for the data pipeline, including initialising the buckets in GCP, the resolution of the data, extent covariates (COPERNICUS/DEM/GLO30 and ESA/WorldCover/v100)"
    * **Human Modification:** Added context ratio to make GOES/VIIRS image bounds larger to better capture spatial context.