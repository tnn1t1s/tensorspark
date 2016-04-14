import ParameterWebSocketClient
import ParameterServerWebsocketHandler
import pyspark
from operator import add
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import os
import time
import random
import cStringIO
import numpy as np

class ParameterServer(threading.Thread):
    def __init__(self,
            websocket_port,
            model,
            warmup_data=None,
            test_data=None):
        threading.Thread.__init__(self)
        self.websocket_port = websocket_port
        self.model = model
        test_labels, test_features = model.process_data(test_data)
        self.test_features = test_features
        self.test_labels = test_labels
        self.warmup(warmup_data)
        self.gradient_count = 0
        self.application = tornado.web.Application([(r"/",
            ParameterServerWebsocketHandler.ParameterServerWebsocketHandler,
            {'server':self})])
        
    def warmup(self, data=None):
        if data is not None:
            self.model.train_warmup(partition=data,
                    error_rates_filename="./output/error_rates")
    
    def run(self):
        self.application.listen(self.websocket_port)
        tornado.ioloop.IOLoop.current().start()


