**AutoMLS** is the implementation of the AutoML multi-objective strategy for multi-label classification. To execute the algorithm, we run the main.py file. The inputs to this algorithm are:
* n_labels: number of labels.
* n_features: number of features.
* name_dataset: dataset name (ARFF). The k-folds of this dataset must be in folder datasets.
* is_dataset_sparse: if the dataset is sparse.
* java_command: path to the environment java.
* meka_classpath: path to the lib folder of the MEKA library.
* len_population: population size.
* number_generation: number of generations.
* k_folds: number of folds.
* n_threads: number of threads
* limit_time = limit time in seconds to train the models.
* project: project path.
* name_dir_res = folder name in folder results/name_dataset.

The outputs of this algorithm are in the results/name_dataset/name_dir_res folder and are:
* Hypervolume graph.
* Objective space graph.
* Results File.
* Metrics file (CSV) with the history of the evaluated algorithms.
