Distributed TensorFlow on Spark.

This work was presented at the 2016 Spark Summit East by Christopher Nguyen @ Adatao.
     - http://www.slideshare.net/arimoinc/distributed-tensorflow-scaling-googles-deep-learning-library-on-spark-58527889

I've since taken it and done a cleanup and will run/publish tests in near future.

# Running the Model
<pre>
$ source setup
$ python tensorspark.py
</pre>

# How does it work?
Distributed Tensorflow partitions the input data set using Sparks' Resilient Distributed Datasets i.e  for each iteration of the optimization, each spark worker processes a subset of the dataset. The workers update parameters by communicating to the Parameter Server (synchronously for now, but really, do we care?). When an iteration is complete, the score is measured vs test data and the process is repeated. 

