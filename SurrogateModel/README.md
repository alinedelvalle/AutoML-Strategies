**SurrogateModel** contains the implementation for creating meta-datasets and surrogate models. 

The inputs to create_meta_dataset.py are:
* project: project path.
* dataset: are the metrics of a dataset resulting from the execution of the AutoML algorithm (AutoMLS) and the Random Algorithm (Random). The dataset value is the name of the dataset followed by '-automl' or '-random' depending on the algorithm executed.
* n_labels: number of labels.
* n_features: number of features.
