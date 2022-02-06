import numpy as np
import logging
import math as math
from overtoppingEve.tfrecord import *
import tensorflow.compat.v1 as tf  #for tensorflow 2.0
tf.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)
# print( tf.__version__)
#Scaling of q
qmax=-0.0788
qmin=-7.397

trained_model='overtoppingEve/'

#Figure plot
def plot1D(x,y,name):
  """Create a pyplot plot and save to buffer."""
  import matplotlib.pyplot as plt
  figure = plt.figure(figsize=(10,5))
  x=np.hstack(x)
  plt.plot(x,y,'-o')
  csfont = {'fontname':'Times','size'   : 14}
  plt.xlabel(name)
  plt.ylabel('q measured [m^3/s/m]', **csfont)
  plt.grid(True)
  plt.show()

def Predicted_overtopping(data,k=0,kname=""):
  with tf.Session(graph=tf.Graph(),config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                      allow_soft_placement=True, log_device_placement=False)) as sess:
      
      #Load pre-trained model as the .pb model trained from Tensorflow
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], trained_model)
      
      #Scale normalized with wave height and structure scale factor
      if(isinstance(data,dict)):
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
        inputvalues=np.reshape(inputvalues, (1, 16, 1))
        true_value=np.transpose([[0]])
      else:
        inputvalues=np.expand_dims(data, 2)
        leng=len(inputvalues)
        true_value=np.zeros([leng,1])
      #print("input parameter with normalized scale:",inputvalues)
      
      #load input parameter to pre-trained model
      save_tfrecords(inputvalues, true_value , 'test',  'test0.tfrecord')

      #The pretrained-model setup
      Input = tf.get_default_graph().get_tensor_by_name("InputDatabase:0")
      keep_ =tf.get_default_graph().get_tensor_by_name("keep:0")
      dataset_init_op = tf.get_default_graph().get_operation_by_name('dataset_init')
      rnd=tf.get_default_graph().get_tensor_by_name("rndvalue:0")
      H2 = tf.get_default_graph().get_tensor_by_name("Outputdata_:0")
      
      #Execute to predict wave overtopping value
      sess.run(dataset_init_op,feed_dict = {Input:'test0.tfrecord',rnd:1})
      predicted = sess.run(H2, feed_dict = { keep_:1.0})

      #Scale inverse
      if(isinstance(data,dict)):
        qx=predicted*(qmax-qmin)+qmin
        q=(math.pow(10,qx)*math.sqrt(9.8*(math.pow(Hm0Toe,3))))
      else:
        Hm0Toe= np.array(data)[:,1]
        q=[]
        for n in range(len(Hm0Toe)):
          qx=predicted[n]*(qmax-qmin)+qmin
          q.append(math.pow(10,qx)*math.sqrt(9.8*(math.pow(Hm0Toe[n],3))))
        plot1D(np.array(data)[:,k],q,kname)
      print("Wave overtopping q:",q)
      
print("Model Loaded successful! ")
