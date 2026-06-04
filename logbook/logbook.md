# Logbook

### First In-Person Meeting (26 May 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Discussed the main scope of the project: downsclaing of wildfire data for near-real-time wildfire evolution monitoring.
    - Discussed the possible coarse inpuut/high-res target datasets to be used for the downscaling model.
    - Also discussed about the possibility of exploring direct (2km/1km -> 30m/20m) and cascaded (2km/1km -> 375m -> 30m/20m) approaches.
- **Work Plan Before Next Meeting:**
    - For coarse inputs, explore GOES vs MTG. For high-res targets, explore Landsat vs Sentinel 2.
    - Start looking into the literature for direct and cascaded downscaling approaches.

### First Meeting With Second Supervisor (28 May 2026)
- **Present:** Tsuyu Huang (Student), Anna Korre (Second Supervisor)
- **Key Points:**
    - Personal introductions, understanding the second supervisor's role.
    - Update on the overall scope of the project that was discussed with the main suprvisor (Jorge).
- **Work Plan Before Next Meeting:**
    - Check both supervisors' availability to set up a meeting with the three of us.
    - Make a decision on what input/target dataset to use with justifications.
    - Create a broad plan of the timeline for the project.
    - Start creating a list of papers/references and write down how each of them are relevant to/supports my research.

### In-Person Meeting (2 June 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Discussed progress from previous week, including possible model architectures to use.
    - Raised the problem about temporal mismatch of VIIRS and Landsat data.
    - Discussed the resources available for processing/training with large amounts of data
- **Feedback:**
    - For the temporal mismatch of VIIRS/Landsat, use GOES -> VIIRS model to obtain VIIRS/Landsat data pairs.
    - Look into about 3 architectures to compare.
- **Work Plan Before Next Meeting:**
    - Look into what covariate data to use, and find out what bands we will be using.
    - Look into the evaluation metrics to use.
    - Find 3 architectures to use and compare.
    - Look into how to use Google Cloud Platform for training.

### Meeting With Main and Second Supervisor (4 Jun 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor), Anna Korre (Second Supervisor)
- **Key Points:**
    - Explained the project progress up to date.
    - Discussed the data to use for the project, including considerations for weather data.
    - Explained how the project progress is monitored (via Google Docs).
- **Feedback:**
    - Consider weather data as extra input.
    - Read the GOES -> VIIRS paper (Badhan et al.) carefully to understand how they use the data.
    - There may be no need for NDWI data for covariates.
- **Work Plan Before Next Meeting:**
    - Look into the literature carefully, finalise a decision on what architecture to use for downscaling.
    - Start writing the project plan, even if the final selected methods are different.
    - Look into how to process the data.