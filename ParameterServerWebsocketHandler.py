from operator import add
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import os
import tensorflow as tf
import time
import random
import cStringIO
import numpy as np

class ParameterServerWebsocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        self.server = kwargs.pop('server')
        self.model = self.server.model
        with self.model.session.graph.as_default():
            self.saver = tf.train.Saver()
            self.local = threading.Lock()
            super(ParameterServerWebsocketHandler,self).__init__(*args, **kwargs)

    def open(self):
        self.send_parameters()

    def send_parameters(self):
        #self.lock.acquire()
        parameters = self.model.get_parameters()
        #self.lock.release()
        serialized = self.model.serialize(parameters)
        self.write_message(serialized, binary=True)

    def on_close(self):
        pass

    def on_message(self,message):
        # now assuming every message is a gradient
        time_gradient = self.model.deserialize(message)
        self.server.gradient_count += 1
        print 'gradient_count %s' % self.server.gradient_count
        time_sent = time_gradient[0][0]
        if time.time() - time.sent < time_lag:
            self.lock.acquire()
            gradient = time_gradient[1:]
            self.model.apply(gradient)
            if self.server.gradient_count % 10 == 0:
                error_rate = self.model.test(self.server.test_labels,
                        self.server.test_features)
                print 'gradients received: %d    error_rate: %f' % (self.server.gradient_count, error_rate)
                t = time.time()
                with open(error_rates_path, 'a') as f:
                    f.write('%f, %d, %f\n' % (t, self.server.gradient_count, error_rate))
            self.lock.release()
        else:
            print "Rejected"
        del time_gradient
        self.send_parameters()                                                                                                                                     
