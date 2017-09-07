import tensorflow as tf
import os
import numpy as np
import pandas as pd

CLASSES = ['tea', 'fish', 'honey', 'juice', 'milk', 'nuts', 'sugar', 'jam', 'rice', 'coffee', 'oil', 'flour', 'corn', 'chocolate', 'water', 'cereal', 'pasta', 'chips', 'tomatosauce', 'vinegar', 'candy', 'beans', 'soda', 'cake', 'spices']

def getlabel_map(label_file):
    data = pd.read_csv(label_file)
    map={}
    for k,v in zip(data['image_id'],data['label']):
        map.update({k:v})
    return map

def read_images(filename_queue):
    label = filename_queue[1]
    img = tf.read_file(filename_queue[0])
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize_images(img, [128, 128])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    out_label = tf.one_hot(label,depth=len(CLASSES),on_value=1.0, off_value=0.0,axis=-1)
    return out_label,img


def pipeline(dir_path, label_file,batch_size):
    map = getlabel_map(label_file)
    filepath = []
    labels = []
    for dirpath, _, filenames in os.walk(dir_path):
        for file in filenames:
            filepath.append(os.path.join(dirpath, file))
            labels.append(CLASSES.index(map[os.path.splitext(file)[0]]))

    filepath_tensor = tf.convert_to_tensor(filepath, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    filename_queue = tf.train.slice_input_producer([filepath_tensor, labels_tensor], shuffle=True)
    keys, images = read_images(filename_queue)
    min_after_deque = batch_size * 2

    capacity = min_after_deque + 3 * batch_size

    col1_batch , col2_batch= tf.train.shuffle_batch([keys,images], batch_size=batch_size,
                                                    capacity=capacity, min_after_dequeue=min_after_deque)
    return len(filepath), col1_batch, col2_batch


if __name__ == '__main__':
    with tf.Session() as sess:
        total, col1_batch_out, col2_batch_out = pipeline('/home/mohit/hacker/train_img', '/home/mohit/hacker/train.csv', 4)
        #total, col1_batch_out, col2_batch_out = pipeline(['approved', 'rejected'], '/home/mohit/final_images/eval',4, 1)
        print total
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                print sess.run([col1_batch_out])
        except tf.errors.OutOfRangeError:
            print 'epoch limit reached'
        finally:
            coord.request_stop()

        coord.join(threads)
