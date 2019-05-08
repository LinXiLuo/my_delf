import io

import tensorflow as tf
import os
# from utils import _int64_feature, _bytes_feature, _float_feature
from PIL import Image
from io import BytesIO
import numpy as np

PATH_TO_TRAIN_FILE = '/mnt/data1/linxi/landmarks_data/GL1000_v1/GL_train'
PATH_TO_TRAIN_TFRECORD = '/mnt/data1/linxi/landmarks_data/GL1000_v1/tfrecord/GL_train1000.tfrecord'
PATH_TO_TEST_FILE = '/mnt/data1/linxi/landmarks_data/GL1000_v1/GL_test'
PATH_TO_TEST_TFRECORD = '/mnt/data1/linxi/landmarks_data/GL1000_v1/tfrecord/GL_test1000.tfrecord'




def read_image_and_label(image_path, label):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    width, height = Image.open(BytesIO(encoded_jpg)).size
    feature = {
        'image/height': _int64_feature(width),
        'image/width': _int64_feature(height),
        'image/depth': _int64_feature(3),
        'image/label': _int64_feature(int(label)),
        'image/encoded': _bytes_feature(encoded_jpg),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def build_dataset(data_path, out_path):
    image_list = []
    class_list = []

    # img_path, label = get_path_and_label(images_path)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for classID in os.listdir(data_path):
            class_path = os.path.join(data_path, classID)
            for img_index in os.listdir(class_path):
                img_path = os.path.join(class_path, img_index)
                image_list.append(img_path)
                class_list.append(classID)

        indices = np.arange(len(image_list))
        np.random.shuffle(indices)
        for i in indices:
            tf_example = read_image_and_label(image_list[i], class_list[i])
            writer.write(tf_example.SerializeToString())
        # img_path = os.path.join(dir_path, dir_names, filenames)

        # for i, l in zip(img_path, dir_names):
        #     tf_example = read_image_and_label(i, l)
        #     writer.write(tf_example.SerializeToString())
    print('Finished building dataset {}\n'.format(out_path))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

if __name__ == '__main__':
    build_dataset(PATH_TO_TRAIN_FILE, PATH_TO_TRAIN_TFRECORD)
    build_dataset(PATH_TO_TEST_FILE, PATH_TO_TEST_TFRECORD)