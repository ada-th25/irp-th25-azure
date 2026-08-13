## Configurations & Experiments

Experiments are managed using modular YAML files. Each configuration file controls data pipelines, dataset transforms, model architecture variations, training hyper-parameters, loss functions, and evaluation/post-processing metrics.

To train a model from an experiment:
```bash
# Example:
python src/train.py --config experiments/e1_haunet_direct.YAML
```

### Key Configuration Blocks

* `experiment`: Defines the unique run identifier (`name`) and dataset target branch (`landsat` or `viirs`).
* `data`: Controls GCS paths, dataset split locations, input normalisation statistics (computed per train split), target transformations (`binarize_target`, `dilate_target`), and feature ablation flags (`drop_landcover`, `drop_dem`).
* `model`: Architecture selection (`haunet`, `autoencoder`) and model-specific parameters (e.g. `prior_fire_fraction`, `use_cov_pyramid`).
* `training`: Optimisation parameters including learning rate, gradient accumulation steps, weight decay, and sampling schemes (`use_weighted_sampler`).
* `evaluation` & `postprocess`: Decision thresholding, metric calculation settings, and binarization methods (`otsu`, `percentile`, `fixed`) for output post-processing.
* `checkpointing`: Local and GCS target paths, retention rules (`keep_last_n`), and validation criteria for tracking the best model checkpoint (`best_metric`).
* `loss`: Loss function parameters (`focal_dice`, `focal`, `dice`), including focal exponents ($\gamma$), class balancing ($\alpha$), and smooth factors.