import time
import os

import numpy
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './2062_output.png', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('image', './data/images/test/2062.png', 'path to input image')


def main(_argv):
    # flags.DEFINE_string('image'+p, './data/images/'+p, 'path to input image')
    # flags.DEFINE_string('output', './output- '+p, 'path to output image')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    # if FLAGS.tfrecord:
    #     dataset = load_tfrecord_dataset(
    #         FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
    #     dataset = dataset.shuffle(512)
    #     img_raw, _label = next(iter(dataset.take(1)))
    # else:
    path = 'data/images/validation'
    for p in os.listdir(path):
        img_full_path = os.path.join(path, p)
        print(p)
        img_raw = tf.image.decode_image(open(img_full_path, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))
        ########################################################################
        logging.info('detections:')
        detected_objects = []
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))
            logging.info('\t{}, {}, {}'.format(p, class_names[int(classes[0][i])], np.array(scores[0][i])))
            detected_objects.append(class_names[int(classes[0][i])])

        c = numpy.array(detected_objects)
        unique, counts = numpy.unique(c, return_counts=True)

        if "car" in unique and counts[int(np.where(unique == "car")[0])] > 5 or "motorbike" in unique and counts[
            int(np.where(unique == "motorbike")[0])] > 2 or "bus" in unique and counts[
            int(np.where(unique == "bus")[0])] > 2 or "truck" in unique and counts[
            int(np.where(unique == "motorbike")[0])] > 3:
            logging.info('\t{}, {}, {}'.format(p, "traffic", 1.0))

        if "bird" in unique:
            logging.info('\t{}, {}, {}'.format(p, "flying", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "nature", 1.0))

        if "cat" or "dog" or "horse" or "sheep" or "cow" or "elephant" or "bear" or "zebra" or "giraffe" in unique:
            logging.info('\t{}, {}, {}'.format(p, "animal", 1.0))
        if "backpack" or "handbag" or "tie" in unique:
            logging.info('\t{}, {}, {}'.format(p, "outfit", 1.0))
        if "suitcase" in unique and "person":
            logging.info('\t{}, {}, {}'.format(p, "tourist", 1.0))
        if "suitcase" in unique and "person":
            logging.info('\t{}, {}, {}'.format(p, "tourist", 1.0))
        if "aeroplane" in unique:
            logging.info('\t{}, {}, {}'.format(p, "flying", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "travel", 1.0))
        if "skateboard" or "surfboard" or "snowboard" or "skis" in unique:
            logging.info('\t{}, {}, {}'.format(p, "snow", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "cool", 1.0))
        if "laptop" or "cell phone" or "remote" in unique:
            logging.info('\t{}, {}, {}'.format(p, "technology", 1.0))
        if "keyboard" or "cell phone" in unique:
            logging.info('\t{}, {}, {}'.format(p, "communication", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "connection", 1.0))
        if "toilet" or "vase" in unique:
            logging.info('\t{}, {}, {}'.format(p, "indoors", 1.0))
        if "chair" or "sofa" or "bed" or "diningtable" or "tvmonitor" or "microwave" or "oven" or \
                "toaster" or "hair drier" or "refrigerator" or "toothbrush" in unique:
            logging.info('\t{}, {}, {}'.format(p, "furniture", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "home", 1.0))
        if "cell phone" in unique:
            logging.info('\t{}, {}, {}'.format(p, "telephone", 1.0))
        if "boat" in unique:
            logging.info('\t{}, {}, {}'.format(p, "river", 1.0))
        if "bus" in unique or "traffic light" in unique or "car" in unique or "parking meter" in unique:
            logging.info('\t{}, {}, {}'.format(p, "street", 1.0))
        if "bus" in unique or "traffic light" in unique or "car" in unique or "truck" in unique or "train" in unique or "fire hydrant" in unique or "stop sign" in unique:
            logging.info('\t{}, {}, {}'.format(p, "road", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "city", 1.0))
            logging.info('\t{}, {}, {}'.format(p, "outdoors", 1.0))
        if "stop sign" in unique:
            logging.info('\t{}, {}, {}'.format(p, "sign", 1.0))
        if "clock" in unique:
            logging.info('\t{}, {}, {}'.format(p, "time", 1.0))
        if "bus" in unique:
            logging.info('\t{}, {}, {}'.format(p, "travel", 1.0))
        if "bottle" in unique:
            logging.info('\t{}, {}, {}'.format(p, "water", 1.0))

        for obj_cnt in zip(unique, counts):
            if obj_cnt[0] == "person":
                if obj_cnt[1] > 0:
                    logging.info('\t{}, {}, {}'.format(p, "human", 1.0))
                elif obj_cnt[1] > 2:
                    logging.info('\t{}, {}, {}'.format(p, "people", 1.0))
                elif 3 < obj_cnt[1] < 6:
                    logging.info('\t{}, {}, {}'.format(p, "meeting", 1.0))
                    logging.info('\t{}, {}, {}'.format(p, "group", 1.0))
                elif obj_cnt[1] >= 6:
                    logging.info('\t{}, {}, {}'.format(p, "crowd", 1.0))

            if obj_cnt[0] == "car" and obj_cnt[1] > 4:
                logging.info('\t{}, {}, {}'.format(p, "rally", 1.0))
            if obj_cnt[0] == "aeroplane" and obj_cnt[1] > 3:
                logging.info('\t{}, {}, {}'.format(p, "aircraft", 1.0))
                logging.info('\t{}, {}, {}'.format(p, "airplane", 1.0))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        out_img = "data/output/out_" + p
        cv2.imwrite(out_img, img)
        logging.info('output saved to: {}'.format(out_img))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

cv2.waitKey(0)
cv2.destroyAllWindows()
