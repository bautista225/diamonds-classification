# Diamond cut: a machine learning classification project

Authors:

* Juan Bautista García Traver
* Alfredo del Río Moldes
* Duarte Salvado Rubio
* Alejandro González Santos

## How to execute

The code for processing the dataset and training the model is provided both as a Julia script (`ml_group_practice.jl`) and a Jupyter notebook (`ml_group_practice.ipynb`). 

When executed, it will perform cross-validation with the different ML approaches and hyperparameters, print the validation metrics for each, train the final model for each approach and print the test metrics.

The code for installing the required packages is commented out in the `practice_functions.jl` and `assignments_functions.jl` files. If required, uncomment those lines.

The code for generating plots in `practice_functions.jl` is also commented out, as it takes a significant amount of time to run. The plots are stored as PNG files in the working directory.

### Required packages

This is a list of the Julia packages that are required for running the code:
* PlotlyJS
* Flux
* Statistics
* ScikitLearn
* PrettyTables
* ScikitLearnBase