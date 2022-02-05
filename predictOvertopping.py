import numpy as np
import logging
import math as math
from overtoppingEve.tfrecord import *
import tensorflow.compat.v1 as tf  #for tensorflow 2.0
tf.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)
# print( tf.__version__)
qmax=-0.0788
qmin=-7.397

trained_model='overtoppingEve/'
def Predicted_overtopping(data):
  with tf.Session(graph=tf.Graph(),config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                      allow_soft_placement=True, log_device_placement=False)) as sess:
      
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], trained_model)
      
      Hm0Toe=data['Hm0Toe']
      data['ht']=data['ht']/data['Hm0Toe']
      data['Bt']=data['Bt']/data['lm0t']
      data['hb']=data['hb']/data['Hm0Toe']
      data['B']=data['B']/data['lm0t']
      data['Ac']=data['Ac']/data['Hm0Toe']
      data['Rc']=data['Rc']/data['Hm0Toe']
      data['Gc']=data['Gc']/data['lm0t']
      data['D50_d']=data['D50_d']/data['Hm0Toe']
      data['D50_u']=data['D50_u']/data['Hm0Toe']
      data['Hm0Toe']=data['Hm0Toe']/data['lm0t']
      data['h']=data['h']/data['lm0t']
      inputvalues = [list(data.values())[:-1]]
      print(inputvalues)
      inputvalues=np.reshape(inputvalues, (1, 16, 1))
      true_value=np.transpose([[0]])
      # print("Input_shape:",np.shape(data))
      # print("Inputs:", data)
      # print("Output_shape:",np.shape(true_value))
      save_tfrecords(inputvalues, true_value , 'test',  'test0.tfrecord')

      # H1 = tf.get_default_graph().get_tensor_by_name("Inputdata_:0")
      Input = tf.get_default_graph().get_tensor_by_name("InputDatabase:0")
      keep_ =tf.get_default_graph().get_tensor_by_name("keep:0")
      dataset_init_op = tf.get_default_graph().get_operation_by_name('dataset_init')
      rnd=tf.get_default_graph().get_tensor_by_name("rndvalue:0")
      # output = sess.run(H1, feed_dict = {Input:parameters.test_file,H1: data, keep_:np.float32(1.0)})
      # print("output:",output)
      H2 = tf.get_default_graph().get_tensor_by_name("Outputdata_:0")
      
      sess.run(dataset_init_op,feed_dict = {Input:'test0.tfrecord',rnd:1})
      predicted = sess.run(H2, feed_dict = { keep_:1.0})

      # print(predicted)
      
      qx=predicted*(qmax-qmin)+qmin
      q=(math.pow(10,qx)*math.sqrt(9.8*(math.pow(Hm0Toe,3))))
      print("q:",q)
      
print("Model Load! ")
