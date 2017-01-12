# python3

# TensorBoard
# python3 ~/.local/lib/python3.5/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=logs --port=6007

import os
import sys
import h5py
import math
import urllib.request
import numpy as np
import tensorflow as tf

sys.path.append('models/slim')
from datasets import dataset_utils
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

image_size = inception.inception_v3.default_image_size

url = 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'
checkpoints_dir = '/tmp/checkpoints'

def make_padding(padding_name, conv_shape):
  padding_name = padding_name.decode("utf-8")
  if padding_name == "VALID":
    return [0, 0]
  elif padding_name == "SAME":
    #return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
    return [math.floor(int(conv_shape[0])/2), math.floor(int(conv_shape[1])/2)]
  else:
    sys.exit('Invalid padding name '+padding_name)


# if not tf.gfile.Exists(checkpoints_dir+'inception_v4.ckpt'):
#   tf.gfile.MakeDirs(checkpoints_dir)
#   dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

with tf.Graph().as_default():


  # Create model architecture

  inputs = np.zeros((1,299,299,3), dtype=np.float32)
  inputs[0][0][0][0] = 1
  inputs = tf.pack(inputs)

  with slim.arg_scope(inception.inception_v4_arg_scope()):
    logits, _ = inception.inception_v4(inputs, num_classes=1001, is_training=False)

  with tf.Session() as sess:

    # Initialize model

    init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
    slim.get_model_variables('InceptionV4'))  

    init_fn(sess)

    # Display model variables

    for v in slim.get_model_variables():
      print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Create graph

    #os.system('rm -rf logs')
    #os.makedirs("logs")

    tf.scalar_summary('logs', logits[0][0])
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter("logs", sess.graph)

    out = sess.run(summary_op)
    summary_writer.add_summary(out, 0)

    # Dump 
    

    

    def dump_conv2d(name='Conv2d_1a_3x3'):
      
      conv_operation = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/Conv2D')

      weights_tensor = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/weights:0')
      weights = weights_tensor.eval()

      padding = make_padding(conv_operation.get_attr('padding'), weights_tensor.get_shape())
      strides = conv_operation.get_attr('strides')

      conv_out = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/Conv2D').outputs[0].eval()
      
      beta = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/beta:0').eval()
      #gamma = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/gamma:0').eval()
      mean = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/moving_mean:0').eval()
      var = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/moving_variance:0').eval()
      
      relu_out = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/Relu').outputs[0].eval()

      os.system('mkdir -p dump/InceptionV4/'+name)
      h5f = h5py.File('dump/InceptionV4/'+name+'.h5', 'w')
      # conv
      h5f.create_dataset("weights", data=weights)
      h5f.create_dataset("strides", data=strides)
      h5f.create_dataset("padding", data=padding)
      h5f.create_dataset("conv_out", data=conv_out)
      # batch norm
      h5f.create_dataset("beta", data=beta)
      #h5f.create_dataset("gamma", data=gamma)
      h5f.create_dataset("mean", data=mean)
      h5f.create_dataset("var", data=var)
      h5f.create_dataset("relu_out", data=relu_out)
      h5f.close()

    def dump_mixed_4a_7a(name='Mixed_4a'):
      dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_0/Conv2d_1a_3x3')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0b_1x7')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0c_7x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_1a_3x3')

    def dump_mixed_5(name='Mixed_5b'):
      dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0b_3x3')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0b_3x3')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0c_3x3')
      dump_conv2d(name=name+'/Branch_3/Conv2d_0b_1x1')

    def dump_mixed_6(name='Mixed_6b'):
      dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0b_1x7')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0c_7x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0b_7x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0c_1x7')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0d_7x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0e_1x7')
      dump_conv2d(name=name+'/Branch_3/Conv2d_0b_1x1')

    def dump_mixed_7(name='Mixed_7b'):
      dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0b_1x3')
      dump_conv2d(name=name+'/Branch_1/Conv2d_0c_3x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0a_1x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0b_3x1')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0c_1x3')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0d_1x3')
      dump_conv2d(name=name+'/Branch_2/Conv2d_0e_3x1')
      dump_conv2d(name=name+'/Branch_3/Conv2d_0b_1x1')

    dump_conv2d(name='Conv2d_1a_3x3')
    dump_conv2d(name='Conv2d_2a_3x3')
    dump_conv2d(name='Conv2d_2b_3x3')

    dump_conv2d(name='Mixed_3a/Branch_1/Conv2d_0a_3x3')
    dump_mixed_4a_7a(name='Mixed_4a')
    dump_conv2d(name='Mixed_5a/Branch_0/Conv2d_1a_3x3')

    dump_mixed_5(name='Mixed_5b')
    dump_mixed_5(name='Mixed_5c')
    dump_mixed_5(name='Mixed_5d')
    dump_mixed_5(name='Mixed_5e')

    dump_conv2d(name='Mixed_6a/Branch_0/Conv2d_1a_3x3')
    dump_conv2d(name='Mixed_6a/Branch_1/Conv2d_0a_1x1')
    dump_conv2d(name='Mixed_6a/Branch_1/Conv2d_0b_3x3')
    dump_conv2d(name='Mixed_6a/Branch_1/Conv2d_1a_3x3')

    dump_mixed_6(name='Mixed_6b')
    dump_mixed_6(name='Mixed_6c')
    dump_mixed_6(name='Mixed_6d')
    dump_mixed_6(name='Mixed_6e')
    dump_mixed_6(name='Mixed_6f')
    dump_mixed_6(name='Mixed_6g')
    dump_mixed_6(name='Mixed_6h')

    dump_mixed_4a_7a(name='Mixed_7a')

    dump_mixed_7(name='Mixed_7b')
    dump_mixed_7(name='Mixed_7c')
    dump_mixed_7(name='Mixed_7d')


    # AuxLogits/Conv2d_1b_1x1
    # AuxLogits/Conv2d_2a

    # name = InceptionV4/AuxLogits/Aux_logits/weights:0, shape = (768, 1001)
    # name = InceptionV4/AuxLogits/Aux_logits/biases:0, shape = (1001,)
    # name = InceptionV4/Logits/Logits/weights:0, shape = (1536, 1001)
    # name = InceptionV4/Logits/Logits/biases:0, shape = (1001,)


    # operations = sess.graph.get_operations()
    # print(len(operations))
    
    #sess.graph.get_tensor_by_name(


    # for v in sess.graph.get_operations():
    #   print(v.name)



    # if not os.path.exists("dump"):
    #   os.makedirs("dump")

    # for v in slim.get_model_variables():
    #   print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # gname = 'InceptionV3/Conv2d_1a_3x3'
    # weights = sess.graph.get_operation_by_name(gname + '/weights:0')


    # weights = sess.graph.get_tensor_by_name(gname + '/conv2d_params:0').eval()
    # padding = make_padding(conv.get_attr("padding"), weights.shape)
    # strides = conv.get_attr("strides")

    # beta = sess.graph.get_tensor_by_name(gname + '/batchnorm/beta:0').eval()
    # gamma = sess.graph.get_tensor_by_name(gname + '/batchnorm/gamma:0').eval()
    # mean = sess.graph.get_tensor_by_name(gname + '/batchnorm/moving_mean:0').eval()
    # var = sess.graph.get_tensor_by_name(gname + '/batchnorm/moving_variance:0').eval()


    #conv = sess.graph.get_operation_by_name('InceptionV3/Conv2d_1a_3x3')

    #weights = sess.graph.get_tensor_by_name('InceptionV3/Conv2d_1a_3x3')
    #conv = slim.get_model_variables()

  # saver = tf.train.Saver(tf.global_variables())

  # with tf.Session() as sess:
  #   m = saver.restore(sess, os.path.join(checkpoints_dir, 'inception_v3.ckpt'))

  # url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
  # image_string = urllib.request.urlopen(url).read()
  # image = tf.image.decode_jpeg(image_string, channels=3)
  # processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
  # processed_images = tf.expand_dims(processed_image, 0)

#   with slim.arg_scope(inception.inception_v3_arg_scope()):
#     logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
  
#   probabilities = tf.nn.softmax(logits)

#   for v in slim.get_model_variables():
#     print('name = {}, shape = {}'.format(v.name, v.get_shape()))

  # init_fn = slim.assign_from_checkpoint_fn(
  #   os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
  #   slim.get_model_variables('InceptionV3'))

#   with tf.Session() as sess:
#     init_fn(sess)
#     np_image, probabilities = sess.run([image, probabilities])
#     probabilities = probabilities[0, 0:]
#     sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

  # print('image', image)
  # print()
  # print('np_image', np_image)
  # print()
  # print('proba', probabilities)
  # print()
  # # print('sorted_inds', sorted_inds)

  # names = imagenet.create_readable_names_for_imagenet_labels()
  # for i in range(5):
  #   index = sorted_inds[i]
  #   print('Probability %0.2f%% => [%s]' % (probabilities[index], names[index]))