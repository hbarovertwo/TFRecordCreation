import tensorflow as tf
import os
import cv2

# feature helper functions


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


a = []
# get file paths of all images
for root, directory, filenames in os.walk('/home/rahul/stackgan/car_data/train/'):
    for filename in filenames:
        a.append(os.path.join(root, filename))

b = []
# gets list of all classes
for i in range(len(a)):
    x = a[i].split('/')[6]
    if x not in b:
        b.append(x)

c = dict()
# creates mapping from class label to integers
for i in range(len(b)):
    c[b[i]] = i

d = []
for i in range(len(a)):
    x = a[i].split('/')[6]
    d.append(tuple([c[x], a[i]]))
'''
d.append(tuple([c[a[0].split('/')[6]],a[0]]))
'''
tffile = 'cars_tfrecord.tfrecords'

writer = tf.python_io.TFRecordWriter(tffile)

for label, im in d:
    img = cv2.imread(im)
    height, width, channels = img.shape
    im64 = cv2.resize(img, (64,64)).tostring()
    im128 = cv2.resize(img, (128,128)).tostring()
    im256 = cv2.resize(img, (256,256)).tostring()

    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'image_64':_bytes_feature(im64),
                'image_128':_bytes_feature(im128),
                'image_256':_bytes_feature(im256),
                'height':_int64_feature(height),
                'width':_int64_feature(width),
                'channels':_int64_feature(channels),
                'label':_int64_feature(label)
            }
        )
    )
    writer.write(example.SerializeToString())
writer.close()

