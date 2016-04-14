import os
import sys
import ParameterWebSocketClient
import ParameterServer
import pyspark
from operator import add
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import MnistDNN
import HiggsDNN
import MolecularDNN
import tensorflow as tf
import time
import random
import cStringIO
import numpy as np

# todo: make this config. refactor CLI, etc.
dataDirectory = "./data"
outputDirectory = "./output"
model_keyword = 'mnist'

# might be better to read a model json file
# and instantiate correct model basedon modeltype 
# assuming all models have same below
if model_keyword == 'mnist':
    training_rdd_filename = '%s/mnist_train.csv' % dataDirectory
    test_filename = '%s/mnist_test.csv' % dataDirectory
    local_test_path = '%s/mnist_test.csv' % dataDirectory
    partitions = 48
    warmup = 2000
    batch_sz = 50
    epochs = 5
    repartition = True
    time_lag = 100
    model = MnistDNN.MnistDNN(batch_sz)
elif model_keyword == 'higgs':
    training_rdd_filename = '%s/higgs_train_all.csv' % dataDirectory
    test_filename = '%s/higgs_test_all.csv' % dataDirectory
    local_test_path = '%s/higgs_test_all.csv' % dataDirectory
    warmup = 20000
    epochs = 1
    partitions = 64
    batch_sz = 128
    time_lag = 20
    repartition = True
    model = HiggsDNN.HiggsDNN(batch_sz)
elif model_keyword == 'molecular':
    training_rdd_filename = '%s/molecular_train_all.csv' % dataDirectory
    test_filename = '%s/molecular_test_all.csv' % dataDirectory
    local_test_path = '%s/molecular_test_all.csv' % dataDirectory
    warmup = 10000
    repartition = True
    epochs = 3
    partitions = 128
    batch_sz = 64
    time_lag = 130
    model = MolecularDNN.MolecularDNN(batch_sz)
else:
    print("KEYWORD HAS TO BE 'mnist', 'higgs' or 'molecular'")
    sys.exit(1)
    
t = int(time.time())
error_rates_path = '%s/error_rates_%s_%d.txt' % (outputDirectory,model_keyword,t)

conf = pyspark.SparkConf()
conf.setExecutorEnv('LD_LIBRARY_PATH', ':/usr/local/cuda-7.0/lib64')
conf.setExecutorEnv('PATH', '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/hadoop/bin:/usr/local/cuda-7.0/bin') 
conf.setExecutorEnv('HADOOP_CONF_DIR', '/usr/local/hadoop/etc/hadoop')
conf.setExecutorEnv('JAVA_HOME','/usr/lib/jvm/java-7-openjdk-amd64')
sc = pyspark.SparkContext(conf=conf)

# todo: cli for websocket port
websocket_port = random.randint(30000, 60000)
print 'websocket_port %d' % websocket_port

def train_partition(partition):
    return ParameterWebSocketClient.TensorSparkWorker(
            model_keyword,
            batch_sz,
            websocket_port).train_partition(partition)                                              
def test_partition(partition):
    return ParameterWebSocketClient.TensorSparkWorker(
            model_keyword,
            batch_sz,
            websocket_port).test_partition(partition)                                               
def train_epochs(num_epochs, training_rdd, num_partitions):
    print 'train epochs %d' % num_epochs
    for i in range(num_epochs):
        print '--------> training epoch %d' % i
        if repartition:
            training_rdd = training_rdd.repartition(num_partitions)
        mapped_training = training_rdd.mapPartitions(train_partition)
        mapped_training.collect()

def test_all():
    testing_rdd = sc.textFile(test_filename).cache()
    mapped_testing = testing_rdd.mapPartitions(test_partition)
    return mapped_testing.reduce(add)/mapped_testing.getNumPartitions()                                                                                                
def start_parameter_server(
        websocket_port,
        model,
        warmup_data,
        test_data):
    parameter_server = ParameterServer.ParameterServer(
            websocket_port,
            model=model,
            warmup_data=warmup_data,
            test_data=test_data)
    parameter_server.start()
    return parameter_server                                                                                                                                            
def main(warmup_iterations, num_epochs, num_partitions):
    try:
        training_rdd = sc.textFile(training_rdd_filename,
                minPartitions=num_partitions).cache()
        print 'num_partitions = %s' % training_rdd.getNumPartitions()
        time.sleep(2)
        warmup_data = training_rdd.take(warmup_iterations)
        with open(local_test_path) as test_file:
            test_data_lines = test_file.readlines()

        with open(error_rates_path, 'w') as f:
            f.write('')
        test_data = test_data_lines[0:100]
        
        parameter_server = start_parameter_server(
                websocket_port=websocket_port,
                model=model,
                warmup_data=warmup_data,
                test_data=test_data)
        
        train_epochs(num_epochs, training_rdd, num_partitions)
        print 'done'
    finally: 
        tornado.ioloop.IOLoop.current().stop()                                                                                                                     
main(warmup_iterations=warmup,
        num_epochs=epochs,
        num_partitions=partitions)
