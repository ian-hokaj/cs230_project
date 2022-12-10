# CS 230 Project
In this project we attempt to use a modified Fourier Neural Operator to Resolve Shock Conditions in the Euler System of PDEs. Additional data files/models/etc. can be found on our [Google Drive](https://drive.google.com/drive/folders/13pPSY2vLNwXmbB4IbxtWHWgwSej-lyWS?usp=sharing)

The reporistory is organized as follows:

## Files
- `train_model.py`: Script for training the models. Draws configurations and models from `configs/config_train.py`
- `eval_models.py`: Evaluates a single model according to `configs/config_eval.py`
- `eval_all_upscales.py`: Evaluates all of the models listed in `configs/config_all_upscales.py` and outputs a `.mat` file to `eval/`
- `plot_epochs.py`: Plot the train/test MSE over the epochs, from the data stored in `pred/`
- `plot_upscales.py`: Plots the data from the models/losses in `configs/config_all_upscales.py` for increasing reolutions, as specified in the config file
- `EulerDataGen.m`: Generate boundary conditions/solutions to the Euler Fluid equations (system of PDEs)

## Folders
- `configs/`: Config files for training/evaluation of models. Must set before running most of the above scripts
- `data/`: Train/test data for training and evaluating the models. Can be generated with `EulerDataGen.m` or downloaded form the Google Drive
- `eval/`: Model outputs and losses/MSEs on test data
- `plots/`: Output directory for plotting scripts
- `pred/`: Predictions on test set generated immediately after training, helpful for visualizing outputs using `EulerDataGen.m`
- `report/`: PDF and slides for the project deliverables
- `scripts/`: Helper files for training and testing models. Includes the modified FNO architecture, basline FNO, U-Net, loss functions, file readers, and Adam optimizer

