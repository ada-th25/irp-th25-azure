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

### Meeting with Main Supervisor (8 Jun 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Explained that the main research will focus on comparing downscaling pipelines with different types of architectures rather than comparing ones of the same kind.
    - Discussed how to process and store the large amounts of data for this project. Google Cloud Platform will be used.
- **Feedback:**
    - Google Cloud Platform will be used. Access to paid computing resources will be given.
- **Work Plan Before Next Meeting:**
    - Finish up the project plan.
    - Look into how to use Google Cloud Platform for this project.
    - Start looking into building a data acquisition pipeline.

### Meeting with Main Supervisor (10 Jun 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Received feedback on project plan.
    - Briefly explained the progress on working on the data pipeline.
- **Feedback:**
    - Edit the project plan to highlight the novel research, and highlight the research motivation in context of insurance.
- **Work Plan Before Next Meeting:**
    - Develop a pipline to extract and process the data needed via Google Cloud Platform.

### Meeting with Main Supervisor (17 Jun 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Discussed the progress on the data pipeline development.
    - Raised concerns about the sparsity of the data and the time it takes for the pipeline to process.
- **Feedback:**
    - Check how much files the Preira et al. dataset contains.
    - Filter the Landsat mask data to keep images with >=10 active pixels.
    - Check how to change the machine on GCP for faster processing.
- **Work Plan Before Next Meeting:**
    - Create a exploratory notebook exploring the sparsity of the data.
    - Check which machine to use for running the pipeline.

### Catch-up Email to Main Supervisor (29 Jun 2026)
- **People:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Discussed the progress up to date, including on wildfire_dataset implementation, data split, and HAUNet implementation.
    - Raised point on whether to use GOES data with greater physical extent (greater number of pixels) or GOES data without (smaller image).
    - Received feedback on loss functions, and possible implementation of progressive frames rather than just static snapshots.
    - More will be discussed on Wednesday before emailing Anna (second supervisor) about next meeting.
- **Feedback:**
    - Use GOES data with greater physical extent.
    - Can consider using past GOES data rather than static snapshots if the GOES context extent option does not work.
    - Consider using focal loss or dice loss over BCEWithLogitsLoss due to severe class imbalance.
- **Work Plan Before Next Meeting:**
    - Look into other loss function to handle class imbalance.
    - Start building the training script.
    - Look into the feasibility of using dynamic snapshots over static ones.

### Meeting with Main Supervisor (01 Jul 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Discussed the problem with lack of data for Landsat target.
    - Discussed what the main focus should be for now.
- **Feedback:**
    - Look into other Landsat datasets that may potentially contain more data (e.g. Landsat Collection 2 Level-3 Burned Area Science Product).
    - Generate more GOES/VIIRS pairs to get about 10,000 images.
    - Work on training the GOES->VIIRS pipeline first.
- **Work Plan Before Next Meeting:**
    - Finish generating the training script.
    - Obtain more VIIRS data by increasing the temporal extent of the dataset.
    - Start training the first model (HAUNet) on the VIIRs dataset.
    - Look into the feasibility of using the suggested Landsat dataset as ground truth target.

### Meeting with Main Supervisor (07 Jul 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Discussed the problem with GEE request taking too long to load data.
    - Discussed about how we can reduce the requests.
- **Feedback:**
    - Consider looking into public buckets for GOES 16/17 instead of downloading from GEE.
    - Consider rasterising VIIRS without GEE API.
- **Work Plan Before Next Meeting:**
    - Look into how to use the public bucket in the data pipeline.
    - Look into how to rasterise without GEE.

### Chat with Main Supervisor (14 Jul 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Update on the VIIRs data acquisition (finished).
    - Update on model training (HAUNet) progress. The HAUNet currently trained on small sample for testing/tuning, but losses and results does not look great.
- **Feedback:**
    - Look into the Autoencoder model from Badhan et al. and see if we can implement it.
- **Work Plan Before Next Meeting:**
    - Try to tweak HAUNet further and try to get the model to NOT predict everything as 'no fire'
    - Implement the Badhan et al. and see if it is any better.

### Meeting With Main and Second Supervisor (23 Jul 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor), Anna Korre (Second Supervisor)
- **Key Points:**
    - Explained the project progress up to date, including data pipeline and model building.
    - Discussed how I can help Anna understand project progress.
- **Feedback:**
    - Write and send her a document of the summaries and details of my project progress.
    - Consider reducing the area of interest/forcing the model to focus on certain parts to remove unnecessary noise and improve accuracy.
- **Work Plan Before Next Meeting:**
    - Write up a document with my project progress to send to Anna.

### Chat with Main Supervisor (30 Jul 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Presented the results for GOES -> VIIRS training.
    - Discuss next steps, including possible things to try to improve model and experiments to do for the report.
- **Feedback:**
    - If time allows, look into using climate data.
    - Try to prioritise overestimation over underestimation, e.g. dilate target data to predict larger area.
    - Try different experiments, such as different losses, see results if covariates are removed etc. to see what contributes the most.
    - Try to train on more data.
- **Work Plan Before Next Meeting:**
    - Download more VIIRS data and train with it.
    - Train the model with dilated targets, try different losses etc and log results.

### Meeting with Main Supervisor (04 Aug 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Presented some results from direct pipeline training.
    - Discussed the main design chanages in HAUNet from the original.
    - Discussed the overall experimental design for the project.
- **Feedback:**
    - Once we have results, try to look into comparing results with other papers.
    - Continue with the model training and performing th experiments.
    - In the report, write down the computing resources used, including GCP, GPUs etc.
- **Work Plan Before Next Meeting:**
    - Add compute resources into report.
    - Continue training and obtain evaluation results for some experiments.
    - Write down & send document of progress to Anna (second supervisor).

### Email with Main Supervisor (18 Aug 2026)
- **Present:** Tsuyu Huang (Student), Jorge Veiras (Main Supervisor)
- **Key Points:**
    - Received feedback on current report draft.
- **Feedback:**
    - Bring Artio/insurance use-case more clearly. e.g. assymetry between under- and over-estimation.
    - Add normalised AUPRC and confidence intervals if space left in report.
    - Overall, try focusing on strengthening the results section.
- **Work Plan Before Next Meeting:**
    - Try implementing the confidence interval/normalised AUPRC implementation and see if it can be added to report.
    - Look at feedback comments on the report and make relevant changes.