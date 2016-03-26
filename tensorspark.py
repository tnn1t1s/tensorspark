import parameterwebsocketclient
import pyspark
from operator import add
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import os
import mnistdnn
import higgsdnn
import moleculardnn
import tensorflow as tf
import time
import random
import cStringIO
import numpy as np

directory = "/user/root/"

model_keyword = 'higgs'
if model_keyword == 'mnist':
    training_rdd_filename = '%smnist_train.csv' % directory
    test_filename = '%smnist_test.csv' % directory
    local_test_path = '/home/ubuntu/mnist_test.csv'
    partitions = 48
    warmup = 2000
    batch_sz = 50
    epochs = 5
    repartition = True
    time_lag = 100
    model = mnistdnn.MnistDNN(batch_sz)
elif model_keyword == 'higgs':
    training_rdd_filename = '%shiggs_train_all.csv' % directory
    test_filename = '%shiggs_test_all.csv' % directory
    local_test_path = '/home/ubuntu/higgs_test_all.csv'
    warmup = 20000
    epochs = 1
    partitions = 64
    batch_sz = 128
    time_lag = 20
    repartition = True
    model = higgsdnn.HiggsDNN(batch_sz)
elif model_keyword == 'molecular':
    training_rdd_filename = '%smolecular_train_all.csv' % directory
    test_filename = '%smolecular_test_all.csv' % directory
    local_test_path = '/home/ubuntu/molecular_test_all.csv'
    warmup = 10000
    repartition = True
    epochs = 3
    partitions = 128
    batch_sz = 64
    time_lag = 130
    model = moleculardnn.MolecularDNN(batch_sz)
else:
    print("KEYWORD HAS TO BE 'mnist', 'higgs' or 'molecular'")
    sys.exit(1)
    
t = int(time.time())
error_rates_path = '/home/ubuntu/error_rates_%s_%d.txt' % (model_keyword, t)
conf = pyspark.SparkConf()

conf.setExecutorEnv('LD_LIBRARY_PATH', ':/usr/local/cuda-7.0/lib64')
conf.setExecutorEnv('PATH', '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/hadoop/bin:/usr/local/cuda-7.0/bin') 
conf.setExecutorEnv('HADOOP_CONF_DIR', '/usr/local/hadoop/etc/hadoop')
conf.setExecutorEnv('JAVA_HOME','/usr/lib/jvm/java-7-openjdk-amd64')
sc = pyspark.SparkContext(conf=conf)

websocket_port = random.randint(30000, 60000)
print 'websocket_port %d' % websocket_port

def train_partition(partition):
    return parameterwebsocketclient.TensorSparkWorker(model_keyword,
            batch_sz,
            websocket_port).train_partition(partition)                                              
def test_partition(partition):
    return parameterwebsocketclient.TensorSparkWorker(model_keyword,
            batch_sz,
            websocket_port).test_partition(partition)                                               
# you can find the mnist csv files here http://pjreddie.com/projects/mnist-in-csv/                                                                                         
def train_epochs(num_epochs, training_rdd, num_partitions):
    for i in range(num_epochs):
        print 'training epoch %d' % i
        if repartition:
            training_rdd = training_rdd.repartition(num_partitions)
        mapped_training = training_rdd.mapPartitions(train_partition)
        mapped_training.collect()

def test_all():
    testing_rdd = sc.textFile(test_filename).cache()
    mapped_testing = testing_rdd.mapPartitions(test_partition)
    return mapped_testing.reduce(add)/mapped_testing.getNumPartitions()                                                                                                
def start_parameter_server(model, warmup_data,test_data):
    parameter_server = ParameterServer(model=model,
            warmup_data=warmup_data,
            test_data=test_data)
    parameter_server.start()
    return parameter_server                                                                                                                                            
def main(warmup_iterations, num_epochs, num_partitions):
    try:
        training_rdd = sc.textFile(training_rdd_filename,
                minPartitions=num_partitions).cache()
        print 'num_partitions = %s' % training_rdd.getNumPartitions()
        time.sleep(5)
        warmup_data = training_rdd.take(warmup_iterations)
        with open(local_test_path) as test_file:
            test_data_lines = test_file.readlines()

        with open(error_rates_path, 'w') as f:
            f.write('')
        test_data = test_data_lines[0:100]
        
        parameter_server = start_parameter_server(model=model,
                warmup_data=warmup_data,
                test_data=test_data)
        
        train_epochs(num_epochs, training_rdd, num_partitions)
        print 'done'
    finally: tornado.ioloop.IOLoop.current().stop()                                                                                                                     
main(warmup_iterations=warmup, num_epochs=epochs, num_partitions=partitions)
