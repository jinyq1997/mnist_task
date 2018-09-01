import sys
import tensorflow as tf
from flask import Flask, request
from PIL import Image, ImageFilter
from redis import Redis, RedisError
import os
import socket
from cassandra.cluster import Cluster
import time
import logging

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)

### Model Setup
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "model/model")
##############

### Connect to the cassandra database
cluster = Cluster(contact_points=["172.17.0.1"],port=9042)
session = cluster.connect()
### Create KEYSPACE and TABLE if not exist
session.execute("create KEYSPACE if not exists mnist_database WITH replication = {'class':'SimpleStrategy', 'replication_factor': 2};")
session.execute("use mnist_database")
session.execute("create table if not exists mnist(id uuid, digits int, image_name text, upload_time timestamp, primary key(id));")
#############

@app.route("/prediction", methods=['GET','POST'])
def predictint():
    imname = request.files["file"]   #Input
    file_name = request.files["file"].filename
    imvalu = prepareImage(imname)
    prediction = tf.argmax(y,1)
    pred = prediction.eval(feed_dict={x: [imvalu]}, session=sess)
    #grab time
    uploadtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    #insert data to cassandra
    session.execute("INSERT INTO mnist(id, digits, image_name, upload_time) values(uuid(), %s, %s, %s)",[int(str(pred[0])), file_name, uploadtime])
    return "The prediction answer is: %s" % str(pred[0])

def prepareImage(i):
    im = Image.open(i).convert('L')
    im = im.resize((28,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    tv = list(im.getdata()) #get pixel values
    tva = [x*1.0/255.0 for x in tv]
    return tva  # Vector of values

@app.route('/')
def index():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = '''
	<!doctype html>
	<html>
	<body>
	<form action='/prediction' method='post' enctype='multipart/form-data'>
  		<input type='file' name='file'>
	<input type='submit' value='Upload'>
	</form>
	'''   
    return html.format(name=os.getenv("NAME", "MNIST"), hostname=socket.gethostname(), visits=visits) 

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)

