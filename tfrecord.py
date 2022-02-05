import numpy as np
import logging
import math as math
import tensorflow.compat.v1 as tf  #for tensorflow 2.0
tf.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)

def save_tfrecords(Intput_data, Output_data, sname, dest_file):
	""" 
	Save numpy array to TFRecord. 
	:params Intput_data: Save first array to TFRecor,each data[i] is a numpy.ndarray. (Note: Int64List or FloatList would be transfer to BytesList) 
	 :params  Output_data: Save 2nd array to TFRecor,each data[i] is a numpy.ndarray. (Note: Int64List or FloatList would be transfer to BytesList)
	 :params   sname:  Save 3rd array to TFRecor,each data[i] is a string array. (Note: StringList would be transfer to BytesList)
	 :params   dest_file: path of the output file。 
	"""
	with tf.io.TFRecordWriter(dest_file) as writer:
		for i in range(len(Intput_data)):
			# X_data_array = serialize_array(Intput_data[i,:])
			# Out_data_array = serialize_array(Output_data)
			features = tf.train.Features(
				feature={
					"Intput_data": _bytes_feature(Intput_data[i,:]),
					"Output_data": _bytes_feature(Output_data[i,:]),
					"Name_data": _string_feature(sname.encode() )
				}
			)
			tf_example = tf.train.Example(features=features)
			serialized = tf_example.SerializeToString()
			writer.write(serialized)

def _bytes_feature(value):
	"""Returns a bytes_list from a float32."""
	return tf.train.Feature(
		bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

def _string_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 