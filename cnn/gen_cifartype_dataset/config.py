import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('raw_height', 72, '')
tf.app.flags.DEFINE_integer('raw_width', 72, '')
tf.app.flags.DEFINE_integer('depth', 3, '')

tf.app.flags.DEFINE_string('data_dir', './data', '')
