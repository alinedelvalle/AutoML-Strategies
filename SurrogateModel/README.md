**SurrogateModel** contains the implementation for creating meta-datasets and surrogate models. 

The inputs to create_meta_dataset.py are:
* project: project path.
* dataset: are the metrics of a dataset resulting from the execution of the AutoML algorithm (AutoMLS) and the Random Algorithm (Random). The metrics file are in metrics/name_dataset-automl or metrics/name_dataset+'-random' depending on the algorithm executed. For the flags dataset, for example, the metrics/flags-automl folder contains the metrics of the flags dataset resulting from the execution of AutoMLS.
* n_labels: number of labels.
* n_features: number of features.
The outputs of create_meta_dataset.py are the meta-dataset and it is in the meta-dataset/name_dataset-automl.csv or meta-dataset/name_dataset-random.csv.
