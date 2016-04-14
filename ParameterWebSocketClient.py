#!/usr/local/bin/bpython
import tensorflow as tf
import MnistDNN
import HiggsDNN
import MolecularDNN
import math
import numpy as np
import tornado.websocket
from tornado import gen
from tornado.ioloop import IOLoop
import cStringIO

# this must be some remnant of working at google on Borg
class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class TensorSparkWorker(Borg):
    def __init__(self, model_keyword, batch_size, websocket_port):
      Borg.__init__(self)

      if 'model' not in self.__dict__:
         print 'Creating new Borg worker'
         if model_keyword == 'mnist':
             self.model = MnistDNN.MnistDNN(batch_size, gpu=False)
         elif model_keyword == 'higgs':
             self.model = HiggsDNN.HiggsDNN(batch_size)
         elif model_keyword == 'molecular':
             self.model = MolecularDNN.MolecularDNN(batch_size)
         else:
            raise
         self.batch_size = batch_size
         self.websocket_port = websocket_port
         self.loop = IOLoop.current()
         self.loop.run_sync(self.init_websocket)
         self.iteration = 0

    @gen.coroutine
    def init_websocket(self):
       self.websock = yield tornado.websocket.websocket_connect("ws://127.0.0.1:%d/" % self.websocket_port, connect_timeout=3600)
   
    def train_partition(self, partition):
       while True:
           labels, features = self.model.process_partition(partition)
           if len(labels) is 0:
               break
           if self.time_to_pull(self.iteration):
               self.request_parameters()
           self.model.train(labels, features)
           self.iteration += 1
           if self.time_to_push(self.iteration):
               self.push_gradients()
           return []

    def test_partition(self, partition):
       labels, features = self.model.process_partition(partition)
       self.request_parameters()
       error_rate = self.model.test(labels, features)
       return [error_rate]
    
    def test(self, data):
       if len(data) is 0:
           return 1.0
        
       self.request_parameters()
       accuracy = self.model.test(data)
       return accuracy
    
    def time_to_pull(self, iteration):
        return iteration % 5 == 0
    
    def time_to_push(self, iteration):
        return iteration % 5 == 0
    
    def request_parameters(self):
        IOLoop.current().run_sync(self.request_parameters_coroutine)
       
    @gen.coroutine
    def request_parameters_coroutine(self):
        parameters = yield self.websock.read_message()
        parameters = self.model.deserialize(parameters)
        self.model.assign_parameters(parameters)
        
    def push_gradients(self):
        IOLoop.current().run_sync(self.push_gradients_coroutine)                                                                                                             
    @gen.coroutine
    def push_gradients_coroutine(self):
       gradients = self.model.get_gradients()
       serialized = self.model.serialize(self.model.get_gradients())
       del gradients
       self.websock.write_message(serialized, binary=True)                                                                                                                
