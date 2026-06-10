This file contains references, licenses, and AI disclosures specific to the codebase in this repository.

**Academic Citations:** For bibliography and references related to the written report, please see [`deliverables/references.bib`](./deliverables/references.bib).

# AI Usage

* **Tool:** Google Gemini (3.5)
    * **Date Used:** 10/06/2026
    * **Scope:** Generated the core algorithm for the data processing pipeline in `src/data/dataset.py`.
    * **Prompt:** (with Badhan et al. 2024 paper attached) "Create a dataset pipeline for a .py file based on this paper. The pipeline will be done via Google Cloud Platform. The data will be stored in a dedicated bucket."
    * **Human Modification:** Modified the bucket names, API key, and corrected thefile name format.
    
* **Tool:** Anthropic Claude (Sonnet 4.6)
    * **Date Used:** 10/06/2026
    * **Scope:** Generated the initial rasterization code in data processing pipeline in `src/data/dataset.py`. Also generated code to be used for testing the initial data pipeline in `notebooks/exploratory-notebook.ipynb`.
    * **Prompt:** (with initial data pipeline pasted) "Add a function to rasterize the VIIRS data. And generate a code to copy and paste to an exploratory ipynb file to test and check the data that's been stored in GCP bucket."