import os
import tensorflow as tf
import random
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import config

FLAGS = tf.app.flags.FLAGS


def get_filename_set(data_set):
    labels = [] # bird / nonbird
    filename_set = [] # ./data/eval or train/birds or nonbirds/filename 리스트

    # labels.txt를 읽어서 labels 리스트에 저장. --> bird / nonbird
    with open(FLAGS.data_dir + '/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')] # split: 구분자를 기준으로 문자열을 잘라 리스트에 넣어준다. strip: 양쪽 공백제거
            labels += inner_list

    for i, lable in enumerate(labels):  # i: 레이블 인덱스 ex) bird: 0  nonbird: 1
        list = os.listdir(FLAGS.data_dir + '/' + data_set + '/' + lable) # os.listdir: 해당 디렉터리 파일리스트 받아옴. --> ./data/eval or train/birds or nonbirds
        for filename in list:
            filename_set.append([i, FLAGS.data_dir + '/' + data_set + '/' + lable + '/' + filename]) # 파일리스트에 경로붙여서 filename_set에 리스트로 저장

    random.shuffle(filename_set)
    return filename_set


def read_jpeg(filename):
    value = tf.read_file(filename)
    decoded_image = tf.image.decode_jpeg(value, channels=FLAGS.depth)
    resized_image = tf.image.resize_images(decoded_image, [FLAGS.raw_height, FLAGS.raw_width])
    resized_image = tf.cast(resized_image, tf.uint8)

    return resized_image


def convert_images(sess, data_set):
    filename_set = get_filename_set(data_set) # 이미지 파일 경로 리스트 받아옴. [][0]: 레이블인덱스 [][1]: 파일경로

    with open('./data/' + data_set + '_data.bin', 'wb') as f:
        for i in range(0, len(filename_set)):
            resized_image = read_jpeg(filename_set[i][1])

            try:
                image = sess.run(resized_image)
            except Exception as e:
                print(e.message)
                continue

            # plt.imshow(np.reshape(image.data, [FLAGS.raw_height, FLAGS.raw_width, FLAGS.depth]))
            # plt.show()

            print(i, filename_set[i][0], image.shape)
            f.write(chr(filename_set[i][0]).encode()) # .encode() 추가함. a bytes-like object is required, not 'str' 에러나서.
            f.write(image.data)


def read_raw_images(sess, data_set):
    filename = ['./data/' + data_set + '_data.bin']
    filename_queue = tf.train.string_input_producer(filename)

    record_bytes = FLAGS.raw_height * FLAGS.raw_width * FLAGS.depth + 1
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    tf.train.start_queue_runners(sess=sess)

    for i in range(0, 100):
        result = sess.run(record_bytes)
        print(i, result[0])
        image = result[1:len(result)]

        #plt.imshow(np.reshape(image, [FLAGS.raw_height, FLAGS.raw_width, FLAGS.depth]))
        #plt.show()


def main(argv = None):
    with tf.Session() as sess:
        convert_images(sess, 'train')
        convert_images(sess, 'eval')
        #read_raw_images(sess, 'eval')


if __name__ == '__main__':
    tf.app.run()
