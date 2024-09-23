import os
import sys
import shlex
import subprocess
import tempfile

from skmultilearn.ext import Meka

import numpy as np
import scipy.sparse as sparse


class MekaAdapted(Meka):
    
    def __init__(self, meka_classifier=None, weka_classifier=None, java_command=None, meka_classpath=None, timeout=None):
        super().__init__(meka_classifier, weka_classifier, java_command, meka_classpath)
        # new attributes
        self.timeout = timeout
        self.len_model_file = 0
        self.len_model_memory = 0
        
    
    # Function adapted to handle MLC ensembles
    # This function throws the exception: TimeoutExpired
    def _run_meka_command(self, args):
        array = self.meka_classifier.strip().split(' ')
        
        mlc = array[0] # mlc algorithm
        params_mlc = array[1:] # mlc algorithm hyperparameters 
        
        command_args = [
            self.java_command,
            "-cp",
            '"{}*"'.format(self.meka_classpath),
            mlc
        ] # it contains: java, meka path and mlc algorithm
        
        command_args += args # train data, test data, threshold and verbosity
        command_args += params_mlc # mlc algorithm hyperparameters
            
        # weka classifier and hyperparameters
        if self.weka_classifier is not None:
            command_args += ["-W", self.weka_classifier]

        meka_command = " ".join(command_args)
        
        # -----------------
        # print(meka_command)
        # -----------------
        
        if sys.platform != "win32":
            meka_command = shlex.split(meka_command)

        '''pipes = subprocess.Popen(
            meka_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.output_, self._error = pipes.communicate()'''
        
        process = subprocess.run(
            meka_command,
            capture_output=True, 
            timeout=self.timeout,
            text=True,
            check=True
        )
        
        self.output_ = process.stdout
        self._error = process.stderr
        
        if type(self.output_) == bytes:
            self.output_ = self.output_.decode(sys.stdout.encoding)
        if type(self._error) == bytes:
            self._error = self._error.decode(sys.stdout.encoding)

        #if process.returncode != 0:
        #    raise Exception(self.output_ + self._error)

        
    # New function that receives:
    # * the number of test instances
    # * the number of labels
    # * arff with the training data
    # * arff with the test data
    # Stores the size of the model's binary file (self.len_model_file).
    # And returns the predictions.
    def fit_predict(self, instances_test_count, label_count, train_arff, test_arff):  
        # sets attributes with None
        self._clean()
        self._instance_count = instances_test_count # de teste
        self._label_count = label_count
        self.len_model_file = 0
        
        try:
            # tempfile
            classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)
            
            # run
            self.output_ = None
            self._warnings = None
            self._verbosity = 5
            
            args = [
                       '-t', '"{}"'.format(train_arff),
                       '-T', '"{}"'.format(test_arff),
                       '-verbosity', str(self._verbosity),
                       # save model in tempfile
                       "-d", '"{}"'.format(classifier_dump_file.name)
                   ]
    
            self._run_meka_command(args)     
            self._parse_output()
            
            # get size of tempfile
            # st_size : represents the size of the file in bytes.
            self.len_model_file = os.stat(classifier_dump_file.name).st_size

        finally:
            # remove tempfile
            self._remove_temporary_files([classifier_dump_file])

        return self._results
    
    
    # Function adapted to work with arff files. The function receives:
    # * number of labels
    # * arff with training data
    # Stores the trained model (self.classifier_dump).
    # Stores the size of the binary file of the model (self.len_model_file).
    def fit(self, label_count, train_arff):
        # sets attributes with None
        self._clean()
        self._label_count = label_count

        try:
            classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)
            
            input_args = [
                "-verbosity", "0",
                "-split-percentage", "100",
                "-t", '"{}"'.format(train_arff),
                "-d", '"{}"'.format(classifier_dump_file.name),
            ]

            self._run_meka_command(input_args)
            
            self.classifier_dump = None
            with open(classifier_dump_file.name, "rb") as fp:
                self.classifier_dump = fp.read()
                
            # get size of tempfile
            # st_size : represents the size of the file in bytes.
            self.len_model_file = os.stat(classifier_dump_file.name).st_size
            # getsizeof: return the size of an object in bytes.
            self.len_model_memory = sys.getsizeof(self.classifier_dump)
            
        finally:
            self._remove_temporary_files([classifier_dump_file])

        return self
    
    
    # Function adapted to work with arff files. The function receives:
    # * number of test instances
    # * arff with training data
    # * arff with test data
    # Opens and saves the trained model (self.classifier_dump) in a file.
    # Returns the predictions.
    def predict(self, instances_test_count, train_arff, test_arff):
        self._instance_count = instances_test_count

        if self.classifier_dump is None:
            raise Exception("Not classified")

        try:
            classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)

            with open(classifier_dump_file.name, "wb") as fp:
                fp.write(self.classifier_dump)
            
            # run
            self._verbosity = 5
            args = ["-l", '"{}"'.format(classifier_dump_file.name)]
            self._run(train_arff, test_arff, args)
            self._parse_output()

        finally:
            self._remove_temporary_files([classifier_dump_file])

        return self._results


    def predict_proba(self, instances_test_count, train_arff, test_arff):
        self._instance_count = instances_test_count

        if self.classifier_dump is None:
            raise Exception("Not classified")

        try:
            classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)

            with open(classifier_dump_file.name, "wb") as fp:
                fp.write(self.classifier_dump)
            
            # run
            self._verbosity = 8
            args = ["-l", '"{}"'.format(classifier_dump_file.name)]
            self._run(train_arff, test_arff, args)
            self._parse_output()

        finally:
            self._remove_temporary_files([classifier_dump_file])

        return self._results
    
    
    # Adapted function to get verbosity from self._verbosity
    def _run(self, train_arff, test_arff, additional_arguments=[]):
        self.output_ = None
        self._warnings = None

        args = [
            "-t",
            '"{}"'.format(train_arff),
            "-T",
            '"{}"'.format(test_arff),
            "-verbosity",
            str(self._verbosity),
        ] + additional_arguments

        self._run_meka_command(args)
        
        return self
    
    
    # Function adapted to handle label prediction and confidence outputs (probabilities)
    # If self._verbosity == 5 get predicted labels
    # If self._verbosity == 8 get probability
    def _parse_output(self):
        """Internal function for parsing MEKA output."""
        if self.output_ is None:
            self._results = None
            self._statistics = None
            return None

        predictions_split_head = "==== PREDICTIONS"
        predictions_split_foot = "|==========="

        if self._label_count is None:
            self._label_count = map(
                lambda y: int(y.split(")")[1].strip()),
                [x for x in self.output_.split("\n") if "Number of labels" in x],
            )[0]

        if self._instance_count is None:
            self._instance_count = int(
                float(
                    filter(
                        lambda x: "==== PREDICTIONS (N=" in x, self.output_.split("\n")
                    )[0]
                    .split("(")[1]
                    .split("=")[1]
                    .split(")")[0]
                )
            )
            
        # ----------    
        if self._verbosity == 5: # predict
            predictions = (
                self.output_.split(predictions_split_head)[1]
                .split(predictions_split_foot)[0]
                .split("\n")[1:-1]
            )
    
            predictions = [
                y.split("]")[0] for y in [x.split("] [")[1] for x in predictions]
            ]
            
            predictions = [
                [a for a in [f.strip() for f in z.split(",")] if len(a) > 0]
                for z in predictions
            ]
            
            predictions = [[int(a) for a in z] for z in predictions]
    
            assert self._verbosity == 5
    
            self._results = sparse.lil_matrix(
                (self._instance_count, self._label_count), dtype="int"
            )
            
            for row in range(self._instance_count):
                for label in predictions[row]:
                    self._results[row, label] = 1
                    
        elif self._verbosity == 8: # proba
        
            predictions = (
                self.output_.split(predictions_split_head)[1]
                .split(predictions_split_foot)[0]
                .split("\n")[1:-1]
            )
            
            predictions = [
                y.split("]")[0] for y in [x.split("] [")[1] for x in predictions]
            ]
            
            predictions = [
                [a for a in [f.strip() for f in z.replace(',','.').split(" ")] if len(a) > 0]
                for z in predictions
            ]
            
            predictions = [[float(a) for a in z] for z in predictions]
            
            assert self._verbosity == 8

            self._results = np.array(predictions)
        # ----------  
        
        statistics = [
            x
            for x in self.output_.split("== Evaluation Info")[1].split("\n")
            if len(x) > 0 and "==" not in x
        ]
        
        statistics = [y for y in [z.strip() for z in statistics] if "  " in y]
        array_data = [z for z in statistics if "[" in z]
        non_array_data = [z for z in statistics if "[" not in z]

        self._statistics = {}
        for row in non_array_data:
            r = row.strip().split("  ")
            r = [z for z in r if len(z) > 0]
            r = [z.strip() for z in r]
            if len(r) < 2:
                continue
            try:
                test_value = float(r[1])
            except ValueError:
                test_value = r[1]

            r[1] = test_value
            self._statistics[r[0]] = r[1]

        for row in array_data:
            r = row.strip().split("[")
            r = [z.strip() for z in r]
            r[1] = r[1].replace(", ", " ").replace(",", ".").replace("]", "").split(" ")
            r[1] = [x for x in r[1] if len(x) > 0]
            self._statistics[r[0]] = r[1] 
            
    # get
    @property
    def statistics(self):
         return self._statistics
     
        
    @property
    def error(self):
         return self._error