import tensorflow as tf
import sys
sys.path.append('../../../../../')
import tensorflow.contrib.slim as slim
from preprocessing import vgg_preprocessing
from research.delf.delf.python.training.delf_v1 import DelfV1
import deployment
from functools import partial
# from research.delf.delf.python.training.se_resnext_pa import SE_ResNeXt
from tensorflow.contrib.slim.python.slim.learning import train_step
# AUTOTUNE = tf.data.experimental.AUTOTUNE

MASTER = None
CHOOSE_NET = 'net_1'
NUM_READERS = 4
CLONE_ON_CPU = False
NUM_CLONES = 1
WORKER_REPLICAS = 1
TASK = 0
NUM_PS_TASKS = 0
MOVING_AVERAGE_DECAY = None
QUANTIZE_DELAY = -1

NUM_CLASSES = 581
BATCH_SIZE = 32
LR = 1e-5
LR_DECAY = 'exponential'
NUM_EPOCHS_PER_DECAY = 1
SYNC_REPLICAS = False
REPLICAS_TO_AGGREGATE = 1
LR_DECAY_FACTOR = 0.94
END_LR = 0.000001
EPOCH_NUM = 50
TRAIN_IMG_SIZE = 224
TRAIN_IMG_SIZE_LIST = [0.25, 0.3, 0.5, 1]  # 7

OPTIMIZER = 'sgd'

TRAIN_DATASET = '/mnt/data1/linxi/landmarks_data/The_Landmark_Data/tfrecord/LC_train27529.tfrecord'
TRAIN_LENGTH = 27529
TEST_DATASET = '/mnt/data1/linxi/landmarks_data/The_Landmark_Data/tfrecord/LC_test.tfrecord'
TEST_LENGTH = 4532

# CHECKPOINT_PATH = '/mnt/data1/linxi/CODE/delf_models_1/research/delf/delf/python/training/parameters/seresnext50/se_resnext50.ckpt'
CHECKPOINT_PATH = '/mnt/data1/linxi/CODE/delf_models_1/research/delf/delf/python/training/parameters/resnet_v1_50.ckpt'
# CHECKPOINT_PATH = None
TRAIN_DIR = '/mnt/data1/linxi/EVAL/delf_models_1/delf_resnet50/'
TRAIN_RES = True
TRAIN_ATT = False
TRAINABLE_SCOPES = None
CHECKPOINT_EXCLUDE_SCOPES = None
IGNORE_MISSING_VARS = True
TEST_EVERY_N_STEP = TRAIN_LENGTH//BATCH_SIZE


def preprocess_image_resnet_train(images):

    shape = tf.shape(images)
    small_side = tf.minimum(shape[0], shape[1])
    cropped_image = tf.image.resize_image_with_crop_or_pad(images, small_side, small_side)
    # images = tf.to_float(cropped_image) / 255.0
    images = tf.image.resize_images(cropped_image, tf.constant([250, 250]))
    images = tf.image.random_crop(images, [224, 224, 3])
    # return tf.expand_dims(images, 0)
    return images

def preprocess_image_train(data):
    return {'height': data['height'], 'width': data['width'],
            'depth': data['depth'], 'label': tf.one_hot(data['label'], NUM_CLASSES),
            'image_raw': vgg_preprocessing.preprocess_for_train(data['image_raw'], 224, 224)}


def preprocess_image_att_train(images):
    shape = tf.shape(images)
    small_side = tf.minimum(shape[0], shape[1])
    cropped_image = tf.image.resize_image_with_crop_or_pad(images, small_side, small_side)
    # images = tf.to_float(cropped_image) / 255.0
    images = tf.image.resize_images(cropped_image, tf.constant([321, 321]))
    # images = tf.image.random_crop(images, [720, 720, 3])
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(images),
        bounding_boxes=[0, 0, 1, 1],
        min_object_covered=0.1)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(images, 0),
                                                  bbox_for_draw)
    tf.summary.image('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    images = tf.slice(images, begin, size)
    return images

def preprocess_image_test(images):

    # shape = tf.shape(images)
    # small_side = tf.minimum(shape[0], shape[1])
    # cropped_image = tf.image.resize_image_with_crop_or_pad(images, small_side, small_side)
    # # images = tf.to_float(cropped_image) / 255.0
    # images = tf.image.resize_images(cropped_image, tf.constant([250, 250]))
    # images = tf.image.random_crop(images, [224, 224, 3])

    return images

def load_dataset(path, length, num_classes):
    keys_to_features = {
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/depth': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    items_to_handlers = {
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'depth': slim.tfexample_decoder.Tensor('image/depth'),
        'label': slim.tfexample_decoder.Tensor('image/label'),
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    }

    # 定义解码器，进行解码
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    # 定义dataset，该对象定义了数据集的文件位置，解码方式等元信息
    return slim.dataset.Dataset(
        data_sources=path,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=length,  # 训练数据的总数
        items_to_descriptions=None,
        num_classes=num_classes)
