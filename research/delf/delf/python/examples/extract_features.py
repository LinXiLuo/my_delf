# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from scipy import io
import numpy as np
sys.path.append('../../../../../')
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
sys.path.append('/mnt/data1/linxi/CODE/delf_models_1/research')
sys.path.append('/mnt/data1/linxi/CODE/delf_models_1/research/delf')
sys.path.append('/mnt/data1/linxi/CODE/delf_models_1/research/delf/delf')

from research.delf.delf import delf_config_pb2
from research.delf.delf import feature_extractor
from research.delf.delf import feature_io

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd_args = None

# Extension of feature files.
# _DELF_EXT = '.delf'

# Pace to report extraction log.y
_STATUS_CHECK_ITERATIONS = 100
TEST_DIR = '/mnt/data1/linxi/EVAL/delf_models_1/delf_feature/'
CLASS_DIR = TEST_DIR + 'oxford/'
SVD40_DIR = CLASS_DIR + 'qSVD40/'

# FSEL_DIR = CLASS_DIR + '/'

# IMAGES_LIST_PATH = '/mnt/data1/linxi/landmarks_data/oxford5k/oxford5k_list.txt'
SVD40_FEAT = 'fsvd40'
SVD40_NF = 'nsvd40'
# SEL_FLATTEN = 'fsel'
# SEL_NF = 'nsel'


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)

  # Read list of images.
  tf.logging.info('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path)
  num_images = len(image_paths)
  tf.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not os.path.exists(cmd_args.output_dir):
    os.makedirs(cmd_args.output_dir)
  nf = []
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)

    with tf.Session() as sess:
      # Initialize variables.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      # Loading model that will be used.
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 config.model_path)
      graph = tf.get_default_graph()
      input_image = graph.get_tensor_by_name('input_image:0')
      input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
      input_image_scales = graph.get_tensor_by_name('input_scales:0')
      input_max_feature_num = graph.get_tensor_by_name(
          'input_max_feature_num:0')
      boxes = graph.get_tensor_by_name('boxes:0')
      raw_descriptors = graph.get_tensor_by_name('features:0')
      feature_scales = graph.get_tensor_by_name('scales:0')
      attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
      attention = tf.reshape(attention_with_extra_dim,
                             [tf.shape(attention_with_extra_dim)[0]])

      locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
          boxes, raw_descriptors, config)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      
      for i in range(num_images):
        # Write to log-info once in a while.
        if i == 0:
          tf.logging.info('Starting to extract DELF features from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.logging.info('Processing image %d out of %d, last %d '
                          'images took %f seconds', i, num_images,
                          _STATUS_CHECK_ITERATIONS, elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(os.path.basename(
            image_paths[i]))[0] + '.mat'
        out_desc_fullpath = os.path.join(cmd_args.output_dir, out_desc_filename)
        if tf.gfile.Exists(out_desc_fullpath):
          tf.logging.info('Skipping %s', image_paths[i])
          continue
        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out,
         attention_out) = sess.run(
             [locations, descriptors, feature_scales, attention],
             feed_dict={
                 input_image:
                     im,
                 input_score_threshold:
                     config.delf_local_config.score_threshold,
                 input_image_scales:
                     list(config.image_scales),
                 input_max_feature_num:
                     config.delf_local_config.max_feature_num
             })

        # feature_io.WriteToFile(out_desc_fullpath, locations_out,
        #                        feature_scales_out, descriptors_out,
        #                        attention_out)

        descriptors_out = np.transpose(descriptors_out)
        io.savemat(os.path.join(SVD40_DIR, str(i+1) + '.mat'), {SVD40_FEAT: descriptors_out})
        nf.append(descriptors_out.shape[1])

        tf.logging.info('Processing image %d ', i+1)
      io.savemat(os.path.join(SVD40_DIR, SVD40_NF + '.mat'), {SVD40_NF: np.array(nf)})

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--config_path',
      type=str,
      default='delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='/mnt/data1/linxi/landmarks_data/paris6k/qparis6k_list.txt',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/mnt/data1/linxi/EVAL/delf_models_1/delf_feature/paris',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  if not os.path.exists(TEST_DIR):
      os.mkdir(TEST_DIR)
  if not os.path.exists(CLASS_DIR):
      os.mkdir(CLASS_DIR)
  if not os.path.exists(SVD40_DIR):
      os.mkdir(SVD40_DIR)


  app.run(main=main, argv=[sys.argv[0]] + unparsed)
