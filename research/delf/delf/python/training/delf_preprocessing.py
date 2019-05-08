# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 900


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]


    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def _square_crop(image_list, resize_height, resize_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    def too_heigh():
        crop_height = image_width
        crop_width = image_width
        offset_height = (image_height - image_width) / 2
        offset_width = 0
        square_image = _crop(image, offset_height, offset_width,
                             crop_height, crop_width)
        square_image.set_shape([None, None, 3])
        square_image = tf.expand_dims(square_image, 0)
        resized_image = tf.image.resize_bilinear(square_image, [resize_height, resize_width])
        return tf.squeeze(resized_image)

    def too_wide():
        crop_height = image_height
        crop_width = image_height
        offset_height = 0
        offset_width = (image_width - image_height) / 2

        square_image = _crop(image, offset_height, offset_width,
                             crop_height, crop_width)
        square_image.set_shape([None, None, 3])
        square_image = tf.expand_dims(square_image, 0)
        resized_image = tf.image.resize_bilinear(square_image, [resize_height, resize_width])
        return tf.squeeze(resized_image)


    square_image = tf.cond(tf.greater(image_height, image_width),
            too_heigh,
            too_wide)

    outputs.append(square_image)

  return outputs

def _random_scale(image_list):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """

  outputs = []
  for image in image_list:
      ratio = tf.random_uniform([], minval=0.25, maxval=1, dtype=tf.float32)
      square_720 = 720
      rescale_height = tf.to_int32(tf.rint(square_720 * ratio))
      rescale_width = tf.to_int32(tf.rint(square_720 * ratio))

      image = tf.expand_dims(image, 0)
      resized_image = tf.image.resize_bilinear(image, [rescale_height, rescale_width],
                                               align_corners=False)
      resized_image = tf.squeeze(resized_image, axis=0)

      # paddings = [[pad_height, pad_height], [pad_width, pad_width]]
      outputs.append(resized_image)

  return outputs

def _padding2fix(image_list, max_size, square_size):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.

  """
  RGB_mean = [123.68, 116.78, 103.94]
  outputs = []
  for image in image_list:
      image_height = tf.shape(image)[1]
      # image_width = tf.shape(image)[1]

      pad_height = (max_size - image_height) / 2
      # pad_width = (square_size - image_width) / 2
      pad_height = tf.to_int32(tf.rint(pad_height))
      # pad_width = tf.to_int32(tf.rint(pad_width))

      pad1 = tf.stack([pad_height, pad_height], axis=0)
      pad2 = tf.stack([pad_height, pad_height], axis=0)
      paddings = tf.stack([pad1, pad2], axis=1)

      image_slices = []
      for i in range(3):
          img = tf.squeeze(image[..., i, tf.newaxis], 2)
          img = tf.pad(img, paddings, constant_values=RGB_mean[i])
          image_slices.append(tf.expand_dims(img, 2))
      pad_image = tf.concat(image_slices, 2)

      pad_image = tf.expand_dims(pad_image, 0)
      pad_image = tf.image.resize_bilinear(pad_image, [square_size, square_size],
                                           align_corners=False)
      pad_image = tf.squeeze(pad_image, axis=0)
      outputs.append(pad_image)

  return outputs



def _distorted_bbox(image_list, image_height, image_width):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """

  outputs = []
  for image in image_list:
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=[[[0, 0, 1, 1]]])

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.summary.image('images_with_box', image_with_box)
      # Employ the bounding box to distort the image.
      image = tf.slice(image, begin, size)

      image = tf.reshape(image, size)
      image.set_shape([None, None, 3])
      resized_image = tf.expand_dims(image, axis=0)
      resized_image = tf.image.resize_bilinear(resized_image, [image_height, image_width],
                                               align_corners=False)
      resized_image = tf.squeeze(resized_image, axis=0)
      # outputs.append(image)
      outputs.append(resized_image)

      # resized_image = resized_image.set_shape([None, None, 3])
  return outputs

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))
  return new_height, new_width



def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

  image = _aspect_preserving_resize(image, resize_side)
  # image = _central_crop([image], 224, 224)[0]
  image = tf.to_float(image)
  image = _square_crop([image], 250, 250)[0]
  # image = _square_crop([image], 900, 900)[0]
  image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)/127.5
  image = tf.image.random_flip_left_right(image)
  image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  return image

def preprocess_for_att(image,
                       output_height,
                       output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """

  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

  # image = _aspect_preserving_resize(image, resize_side)
  image = _square_crop([image], 900, 900)[0]
  image = _random_crop([image], 720, 720)[0]
  # TODO light or dark
  image = _random_scale([image])[0]
  # image.set_shape([output_height, output_width, 3])
  image = _padding2fix([image], 720, output_height)[0]
  image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


  image = tf.to_float(image) / 127.5
  # image = tf.to_float(image)

  image = tf.image.random_flip_left_right(image)
  return image

def preprocess_for_att2(image,
                        output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """

  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)
  image = tf.to_float(image)
  # image = _aspect_preserving_resize(image, resize_side)
  # image = _square_crop([image], 500, 500)[0]
  image = _distorted_bbox([image], output_height, output_width)[0]
  # TODO light or dark
  image = tf.to_float(image)/127.5

  image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  image = tf.image.random_flip_left_right(image)
  return image

def preprocess_for_eval(image, output_height, output_width, resize_side):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side)
  image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)/127.5
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

def preprocess_for_test(image, resize_side):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side)
  image = tf.to_float(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image, output_height, output_width, step='test',
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """

  if step == 'train':
    return preprocess_for_train(image, output_height, output_width,
                                  resize_side_min, resize_side_max)
  elif step == 'att_train':
    return preprocess_for_att2(image, output_height, output_width,
                               resize_side_min, resize_side_max)
  elif step == 'eval':
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)
  else:
    return preprocess_for_test(image, resize_side_min)

