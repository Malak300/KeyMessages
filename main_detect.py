import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
import glob
# from yolov3_tf2.models import (
# 	YoloV3, YoloV3Tiny
# )
# from yolov3_tf2.dataset import transform_images
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('output', './serving/yolov3/1', 'path to saved_model')
flags.DEFINE_string('classes', './data/images/classes.txt', 'path to classes file')
flags.DEFINE_integer('num_classes', 50, 'number of classes in the model')
flags.DEFINE_string('result_output', './data/output/ours/', 'path to output image')


def main(_argv):
    # if FLAGS.tiny:
    # 	yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    # else:
    # 	yolo = YoloV3(classes=FLAGS.num_classes)
    #
    # yolo.load_weights(FLAGS.weights)
    # logging.info('weights loaded')
    #
    # tf.saved_model.save(yolo, FLAGS.output)
    # logging.info("model saved to: {}".format(FLAGS.output))

    score_threshold = 0.45
    model = tf.saved_model.load(FLAGS.output)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img_idx = 1
    for image_path in glob.glob(f'./data/images/validation/*.jpg'):
        # image_np = cv2.imread(image_path, 1)  # load_image_into_numpy_array(image_path)
        # raw_img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
        raw_img = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, 416)

        t1 = time.time()
        outputs = infer(img)
        boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
            "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               scores[0][i].numpy(),
                                               boxes[0][i].numpy()))
        img2 = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img2 = draw_outputs(img2, (boxes, scores, classes, nums), class_names, score_threshold)
        cv2.imwrite(FLAGS.result_output + img_idx.__str__() + '_2.jpeg', img2)
        logging.info('output saved to: {}'.format(FLAGS.result_output + img_idx.__str__() + '_2.jpeg'))
        img_idx = img_idx + 1


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
