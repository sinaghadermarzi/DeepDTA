# DeepDTA - easy run 

This is a forked version of the [DeepDTA](https://github.com/hkmztrk/DeepDTA) to make training and testing on new drug-protein pairs easier. 

## Usage

These instructions assumes that you are working on linux and have Miniconda/Anaconda installed.

After cloning or downloading the repository, to prepare the python environment, enter the following command in the root folder of the project to prepare the environment. 

> `conda env create -p cenv/ -f enviornment.yml `

then activate the environment by entering:

> `conda activate cenv/ `

Then you can train and test on your dataset by following the instructions in the next sections.

### Training

For training, we need a set of drug-protein pairs and their affinity. The format of the input dataset should look like the example provided in `datasets/example.csv` 

to start training on the given example dataset, enter the following command in the root folder of the project:

> `python ./source/train_model.py datasets/example.csv `

the model will be saved in the folder `saved_model`

### Testing

Assuming there is a trained model stored in the  `saved_model` folder, you can make predictions for a set of drug-protein pairs by the following command:

`python ./source/predict.py datasets/example.csv `

the dataset used as input for prediction has the same format as the `datasets/example.csv` 

the results will be saved in the same directory as the input file with a `_predicted` suffix
