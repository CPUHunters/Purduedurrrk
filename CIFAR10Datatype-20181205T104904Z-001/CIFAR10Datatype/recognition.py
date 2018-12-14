# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './data/10000_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './data/10000_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 2,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('result_txt', './data/result.txt',
                         """text file for recording bird detection result""")

IMAGE_SIZE = 54

classes = ["birds", "skys", "trees", "airplanes"]

def eval_once(saver, logits, labels, top_k_op, images, num_images, frame_count):

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
            step = 0
            while step < num_images and not coord.should_stop():
                logit, label, test_image, predictions = sess.run([logits, labels, images, top_k_op])
                step += 1
                classification = sess.run(tf.argmax(logit[0], 0))
                # print('logit[%d]s predict label is %d: ' % (0, classification), classes[classification])
                if classification == 0:
                    print('bird')
                else:
                    print('non bird')
                # with open(FLAGS.result_txt, 'a') as f:
                #     f.write(str(frame_count) + '-' + str(step) + '.jpg      ')
                #     if classification == 0:
                #         f.write(str(1) + '\n')
                #     else:
                #         f.write(str(0) + '\n')

            # num_iter = int(math.ceil(num_images / FLAGS.batch_size))
            # true_count = 0  # Counts the number of correct predictions.
            # total_sample_count = num_iter * FLAGS.batch_size
            # step = 0
            # while step < num_iter and not coord.should_stop():
            #     logit, label, test_image, predictions = sess.run([logits, labels, images, top_k_op])
            #     true_count += np.sum(predictions)
            #     print('step     :', step)
            #     print('predictions : ', predictions)
            #     print('logit    :', label)
            #     step += 1
            #
            #     step2 = 0
            #     classes = ["birds", "skys", "trees", "airplanes"]
            #     while True:
            #         classification = sess.run(tf.argmax(logit[step2], 0))
            #         print('logit[%d]    :' % step2, logit[step2])
            #         print('logit[%d]s predict label is %d: ' % (step2, classification), classes[classification])
            #         step2 = step2 + 1
            #         if step2 == FLAGS.batch_size:
            #             break
            #
            # precision = true_count / total_sample_count
            # print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def read_cifar10(filename_queue):

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 72
    result.width = 72
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def inputs(count):
    filenames = [os.path.join(FLAGS.bin_dir, str(count) + '.bin')]
    num_examples_per_epoch = 1
    batch_size = 1

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # # Ensure that the random shuffling has good mixing properties.
    # min_fraction_of_examples_in_queue = 0.4
    # min_queue_examples = int(num_examples_per_epoch *
    #                          min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, 1, batch_size)


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):

    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])


def evaluate(count, num_images):

    # Initialize eval temp directory folder
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = inputs(count)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        eval_once(saver, logits, labels, top_k_op, images, num_images, frame_count=count)


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    # evaluate()


if __name__ == '__main__':
    tf.app.run()