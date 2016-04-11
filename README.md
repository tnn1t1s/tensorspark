A repo for Distributed TensorFlow on Spark, first presented at the 2016 Spark Summit East 

Slide deck: http://www.slideshare.net/arimoinc/distributed-tensorflow-scaling-googles-deep-learning-library-on-spark-58527889

# Running the Model
## First, build a zip file of the model and parameter server
zip pyfile.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mymodel.py
## Then, submit to spark using spark-submit
spark-submit ..... --pyfiles ./pyfiles.zip ./tensorspark.py


