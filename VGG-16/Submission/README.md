1. To start the augmentation and normalization of the raw database, you have to run the "augmentation_init.py". The configuration of the augmentation can be changed in the "augmentation.py" file.

2.1. In the "config.py" file the "BASE_PATH" has to be configured. For using the cluster it must look like "/home/username/...".
2.2 In the "load_dataset.py" file the path in line 16 must be configured to path="/scratch/username"

3. In the empty folders "plots", "evaluation" and "best_params" the results get stored.
