import tensorflow as tf
import input
import os
import re

img_size = 128
num_channels = 3
train_path = '/home/mohit/hacker/train_img'
validation_path = '/home/mohit/hacker/eval_img'
label_file = '/home/mohit/hacker/train.csv'
classes = ['approved', 'rejected']
num_classes = len(input.CLASSES)
batch_size = 16
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('out_dir', '/home/mohit/hacker/checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(X, name):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)


def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))


def _conv_summary(kernel, scope, shape):
    with tf.variable_scope('visualization'):
        # scale weights to [0 1], type is still float
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)

        # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose(kernel_0_to_1, shape)

        # this will display random 3 filters from the 64 in conv1
        tensor_name = re.sub('%s_[0-9]*/', '', scope)
        tf.summary.image(tensor_name+'/filters', kernel_transposed, max_outputs=3)

def model(x):
    X_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    with tf.variable_scope('conv1') as scope:
        layer1_W = weight_variable([3,3,3,32], "layer1_W")
        layer1_b = bias_variable([32], "layer1_b")
        layer1_h = tf.nn.relu(conv2d(X_image, layer1_W) + layer1_b, name=scope.name)
        _activation_summary(layer1_h)
        _conv_summary(layer1_W, scope.name, [3, 0, 1, 2])
    layer1_pool = max_pool_2x2(layer1_h, name='pool1')

    with tf.variable_scope('conv2') as scope:
        layer2_W = weight_variable([3, 3, 32, 64], "layer2_W")
        layer2_b = bias_variable([64], "layer2_b")
        layer2_h = tf.nn.relu(conv2d(layer1_pool, layer2_W) + layer2_b, name=scope.name)
        _activation_summary(layer2_h)
        _conv_summary(layer2_W, scope.name, [3, 2, 0, 1])
    layer2_pool = max_pool_2x2(layer2_h, name='pool2')
    with tf.variable_scope('conv3') as scope:
        layer3_W = weight_variable([3, 3, 64, 128], "layer3_W")
        layer3_b = bias_variable([128], "layer3_b")
        layer3_h = tf.nn.relu(conv2d(layer2_pool, layer3_W) + layer3_b, name=scope.name)
        _activation_summary(layer3_h)
        _conv_summary(layer3_W, scope.name, [3, 2, 1, 0])
    layer3_pool = max_pool_2x2(layer3_h, name='pool3')

    with tf.variable_scope('local3') as scope:
        layer3_pool_flat = tf.reshape(layer3_pool, [-1, (img_size / 8) * (img_size / 8) * 128])
        fc1_W = weight_variable([(img_size / 8) * (img_size / 8) * 128, 1024], "fc1_W")
        fc1_b = bias_variable([1024], "fc1_b")
        fc1_h = tf.nn.relu(tf.matmul(layer3_pool_flat, fc1_W) + fc1_b, name=scope.name)
        _activation_summary(fc1_h)

    with tf.variable_scope('local3') as scope:
        keep_prob = tf.placeholder(tf.float32)
        fc1_h_drop = tf.nn.dropout(fc1_h, keep_prob)

        fc2_W = weight_variable([1024, num_classes], "fc2_W")
        fc2_b = bias_variable([num_classes], "fc2_b")

        y_conv = tf.add(tf.matmul(fc1_h_drop, fc2_W), fc2_b, name=scope.name)
        _activation_summary(y_conv)

    return y_conv, keep_prob


def main(_):
    with tf.Graph().as_default():
        total_train_example, train_label_batch, train_img_batch = input.pipeline(train_path,label_file,batch_size=batch_size)
        total_val_example, val_label_batch, val_img_batch = input.pipeline(validation_path, label_file, batch_size=batch_size)

        print "found {} training images".format(total_train_example)
        print "found {} validation images".format(total_val_example)
        with tf.name_scope('final_model') as scope:
            x = tf.placeholder(tf.float32, shape=[None, img_size * img_size * num_channels], name='x')

            y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

            y_conv, keep_prob = model(x)
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                          scope)
            summary_op = tf.summary.merge(summaries)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv), name='loss')
        train_loss = tf.summary.scalar('loss/train', cross_entropy)
        validation_loss = tf.summary.scalar('loss/validation', cross_entropy)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')
        train_acc = tf.summary.scalar('accuracy/train', accuracy)
        validation_acc = tf.summary.scalar('accuracy/validation', accuracy)
        saver = tf.train.Saver(tf.global_variables(),sharded=True)

        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        session = tf.Session()
        session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        summary_writer = tf.summary.FileWriter(FLAGS.out_dir, session.graph)
        step = 0
        try:
            while not coord.should_stop():
                train_img_batch = tf.reshape(train_img_batch,[batch_size, img_size * img_size * num_channels])
                val_img_batch = tf.reshape(val_img_batch,[batch_size, img_size * img_size * num_channels])
                feed_dict_train = {x: train_img_batch.eval(session=session),
                                   y_true: train_label_batch.eval(session=session), keep_prob: 0.5}
                feed_dict_validate = {x: val_img_batch.eval(session=session),
                                      y_true: val_label_batch.eval(session=session), keep_prob: 1.0}
                _, acc, loss, acc_str_train, loss_str_train= session.run([train_step,accuracy, cross_entropy,train_acc, train_loss], feed_dict=feed_dict_train)
                summary_writer.add_summary(acc_str_train, step)
                summary_writer.add_summary(loss_str_train, step)
                print "step{0}, training accuracy {1:>6.1%}, training loss {2:.3f}".format(step,acc, loss)

                if step % (total_train_example/batch_size) == 0:
                    val_acc, val_loss, acc_str_val, loss_str_val = session.run([accuracy,cross_entropy,validation_acc, validation_loss], feed_dict=feed_dict_validate)
                    summary_writer.add_summary(acc_str_val, step)
                    summary_writer.add_summary(loss_str_val, step)
                    print "validation accuracy {0:>6.1%}, validation loss {1:.3f}".format(val_acc, val_loss)
                    checkpoint_path = os.path.join(FLAGS.out_dir, 'model.ckpt')
                    saver.save(session, checkpoint_path, global_step=step)

                if step % 100 == 0:
                    summary_str = session.run(summary_op, feed_dict=feed_dict_validate)
                    summary_writer.add_summary(summary_str, step)
                step += 1
        except tf.errors.OutOfRangeError:
            print 'epoch limit reached'
        finally:
            coord.request_stop()

        coord.join(threads)
        #saver.save(session, 'img_model')

if __name__ == '__main__':
  tf.app.run(main=main)