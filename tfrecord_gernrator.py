# based on https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io

import numpy
import pandas as pd

from tensorflow.python.framework.versions import VERSION

if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from tfmodels.research.object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', 'data/images/train_labels.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', 'data/images/train.record', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', 'data/images/train', 'Path to images')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == 'arrow':
        return 1
    if row_label == 'banner':
        return 2
    if row_label == 'binary':
        return 3
    if row_label == 'booklet':
        return 4
    if row_label == 'bright':
        return 5
    if row_label == 'calendar':
        return 6
    if row_label == 'celebration':
        return 7
    if row_label == 'cell phone':
        return 8
    if row_label == 'city':
        return 9
    if row_label == 'climber':
        return 10
    if row_label == 'cloud':
        return 11
    if row_label == 'data':
        return 12
    if row_label == 'date':
        return 13
    if row_label == 'diagram':
        return 14
    if row_label == 'document':
        return 15
    if row_label == 'employee':
        return 16
    if row_label == 'flag':
        return 17
    if row_label == 'freedom':
        return 18
    if row_label == 'graphic':
        return 19
    if row_label == 'graphic design':
        return 20
    if row_label == 'grass':
        return 21
    if row_label == 'ice':
        return 22
    if row_label == 'landscape':
        return 23
    if row_label == 'light':
        return 24
    if row_label == 'lock':
        return 25
    if row_label == 'logo':
        return 26
    if row_label == 'map':
        return 27
    if row_label == 'number':
        return 28
    if row_label == 'paper':
        return 29
    if row_label == 'pie chart':
        return 30
    if row_label == 'portrait':
        return 31
    if row_label == 'presentation':
        return 32
    if row_label == 'prize':
        return 33
    if row_label == 'robot':
        return 34
    if row_label == 'sea':
        return 35
    if row_label == 'shining':
        return 36
    if row_label == 'sky':
        return 37
    if row_label == 'skyscraper':
        return 38
    if row_label == 'snow':
        return 39
    if row_label == 'soldier':
        return 40
    if row_label == 'street':
        return 41
    if row_label == 'sunset':
        return 42
    if row_label == 'teamwork':
        return 43
    if row_label == 'text':
        return 44
    if row_label == 'time':
        return 45
    if row_label == 'tree':
        return 46
    if row_label == 'uniform':
        return 47
    if row_label == 'visualization':
        return 48
    if row_label == 'weapon':
        return 49
    if row_label == 'written report':
        return 50
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    tf_example = None
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    try:
        Im = Image.Image.getdata(image)[0]
    except OSError:
        print("errr")

    width, height = image.size

    filename = group.filename.encode('utf8')

    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if class_text_to_int(row['class']) is not None:
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        if group.filename.__str__().endswith(".jpg"):
            if not group.filename.__str__().endswith("2331.jpg"):
                tf_example = create_tf_example(group, path)
                writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
