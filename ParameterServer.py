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

class ParameterServer(threading.Thread):
    def __init__(self, model, warmup_data=None, test_data=None):
        threading.Thread.__init__(self)
        self.model = model
        test_labels, test_features = model.process_data(test_data)
        self.test_features = test_features
        self.test_labels = test_labels
        self.warmup(warmup_data)
        self.gradient_count = 0
        self.application = tornado.web.Application([(r"/",
            ParameterServerWebsocketHandler,
            {'server':self})])
        
    def warmup(self, data=None):
        if data is not None:
            self.model.train_warmup(partition=data,
                    error_rates_filename=error_rates_path)
    
    def run(self):
        self.application.listen(websocket_port)
        tornado.ioloop.IOLoop.current().start()


