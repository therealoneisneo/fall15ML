#! /bin/bash  

python number_recognition.py dtree image_data/zip.train image_data/zip.test


python number_recognition.py knn image_data/zip.train image_data/zip.test

python number_recognition.py net image_data/zip.train image_data/zip.test

python number_recognition.py svm image_data/zip.train image_data/zip.test

python number_recognition.py pcaknn image_data/zip.train image_data/zip.test

python number_recognition.py pcasvm image_data/zip.train image_data/zip.test


# python number_recognition.py svm image_data/zip.train image_data/zip.test

# python naiveBayes.py data_sets data_sets/test_set