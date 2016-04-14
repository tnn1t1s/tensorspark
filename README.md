A repo for Distributed TensorFlow on Spark, first presented at the 2016 Spark Summit East 

Slide deck: http://www.slideshare.net/arimoinc/distributed-tensorflow-scaling-googles-deep-learning-library-on-spark-58527889

# Running the Model
## First, build a zip file of the model and parameter server
zip pyfile.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mymodel.py
## Then, submit to spark using spark-submit
spark-submit ..... --pyfiles ./pyfiles.zip ./tensorspark.py

# hows it work?
Distributed Tensorflow partitions the input data set using Sparks' Resilient Distributed Dataset partition method. For each iteration of the optimization, each spark worker gets a subset of the dataset, udates parameters by communicating to the parameter server. When an iteration is complete, the score is measured using parameters of the model and 'hot' data from the test set and the process is repeated. 

