import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('bin_dir', './data/frame_bin',
                           """Directory where to write the binary files of each frame ROI images""")
tf.app.flags.DEFINE_string('roi_dir', './data/ROIs',
                           """Directory where the ROI images are saved according to each frame""")
RAW_SIZE = 72


def read_jpeg(image_path):
    value = tf.read_file(image_path)
    decoded_image = tf.image.decode_jpeg(value, channels=3)
    resized_image = tf.image.resize_images(decoded_image, [RAW_SIZE, RAW_SIZE])
    resized_image = tf.cast(resized_image, tf.uint8)

    return resized_image


# convert images to binary file of cifar10 datatype
# path: binary_file_dir/dir_count.bin
def convert_images(dir_count):
    dir_path = os.path.join(FLAGS.roi_dir, str(dir_count))
    imagelist = os.listdir(dir_path)
    with open(os.path.join(FLAGS.bin_dir, str(dir_count) + '.bin'), 'wb') as f:
        for image in imagelist:
            image_path = os.path.join(dir_path, image)
            resized_image = read_jpeg(image_path)
            try:
                with tf.Session() as sess:
                    image = sess.run(resized_image)
            except Exception as e:
                print(e.message)
            tmp_label = 0
            f.write(chr(tmp_label).encode())
            f.write(image.data)
