import time
import signal

from skmultilearn.ext import download_meka
from subprocess import TimeoutExpired

from meka.meka_adapted import MekaAdapted

from skmultilearn.dataset import load_from_arff

from sklearn.metrics import f1_score


if __name__ == '__main__':
    project = '' # project paths
    meka_classpath = download_meka()
       
    meka = MekaAdapted(       
        meka_classifier = 'meka.classifiers.multilabel.BR',
        weka_classifier = 'weka.classifiers.trees.RandomForest -- -I 100',
        meka_classpath = meka_classpath, # download_meka
        java_command = '', # path to the java do environment
        timeout = 100 # seconds 
    )
    
    x_test, y_test = load_from_arff(
        project + 'datasets/emotions/emotions-test-0.arff', 
        label_count=6, # labels
        label_location='end',
        load_sparse=False,
        return_attribute_definitions=False
    )
    
    n_example_test = x_test.shape[0]
    
    print(n_example_test)
    
    p = meka.fit_predict(n_example_test, 
                         6, # labels
                         project + 'datasets/emotions/emotions-train-0.arff', 
                         project + 'datasets/emotions/emotions-test-0.arff')

    print(meka.output_)
    
    print('---')
    
    print(meka.statistics)
    
    print('---')
    
    print(f1_score(y_test, p, average='samples'))
    print(f1_score(y_test, p, average='micro'))
    print(f1_score(y_test, p, average='macro'))