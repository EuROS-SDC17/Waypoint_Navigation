from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


class CNNTrain():
    EPOCHS = 100
    BATCH_SIZE = 20
    PATH = "/home/nikidimi/ML/Waypoint_Navigation/ros/src/tl_detector/images/"

    def __init__(self):
        self.create_placeholders()
        self.create_layers()
        self.create_operations()

    def get_images(self, path):
        images_list = []

        for filename in glob.glob(path):
            im = Image.open(filename)
            out = im.resize((80, 60))
            out = out.convert('HSV')
            images_list.append(np.array(out))
        return images_list

    def get_all_images(self):
        green_images = self.get_images(self.PATH + "green/*.jpg")
        red_images = self.get_images(self.PATH + "red/*.jpg")
        none_images = self.get_images(self.PATH + "none/*.jpg")
        yellow_images = self.get_images(self.PATH + "yellow/*.jpg")

        go_images = green_images + none_images + yellow_images
        stop_images = red_images

        X = go_images + stop_images
        y = [0] * len(go_images) + [1] * len(stop_images)

        return X, y

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy, total_loss = 0, 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.BATCH_SIZE):
            batch_x = X_data[offset:offset+self.BATCH_SIZE]
            batch_y = y_data[offset:offset+self.BATCH_SIZE]
            loss, accuracy = sess.run([self.loss_operation,
                                       self.accuracy_operation],
                                      feed_dict={self.input_layer: batch_x,
                                                 self.output_layer: batch_y,
                                                 self.keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
            total_loss += (loss * len(batch_x))
        return total_loss / num_examples, total_accuracy / num_examples

    def create_placeholders(self):
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.input_layer = tf.placeholder(tf.float32, shape=(None, 60, 80, 3),
                                          name="input_layer")
        self.output_layer = tf.placeholder(tf.int32, shape=(None))

    def create_layers(self):
        conv1 = tf.layers.conv2d(inputs=self.input_layer,
                                 filters=64, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                        strides=2)

        dropout1 = tf.nn.dropout(pool1, self.keep_prob)

        conv2 = tf.layers.conv2d(inputs=dropout1, filters=128,
                                 kernel_size=[2, 2], padding="same",
                                 activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[30, 40],
                                        strides=2)
        dropout2 = tf.nn.dropout(pool2, self.keep_prob)

        flat = tf.contrib.layers.flatten(dropout2)
        self.logits = tf.layers.dense(inputs=flat, units=2,
                                      name="output_logits")

    def create_operations(self):
        one_hot_y = tf.one_hot(self.output_layer, 2)
        self.loss_operation = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                    labels=one_hot_y)
        )
        optimizer = tf.train.AdamOptimizer()
        self.training_operation = optimizer.minimize(self.loss_operation)
        self.correct_prediction = tf.equal(
            tf.argmax(self.logits, 1, name="output_class"),
            tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

    def train(self):
        best_result = 0
        saver = tf.train.Saver(max_to_keep=10)

        X, y = self.get_all_images()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()
            for i in range(self.EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, self.BATCH_SIZE):
                    end = offset + self.BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    loss = sess.run(self.training_operation,
                                    feed_dict={
                                        self.input_layer: batch_x,
                                        self.output_layer: batch_y,
                                        self.keep_prob: 0.5
                                    })

                test_loss, test_accuracy = self.evaluate(X_test, y_test)
                print("EPOCH {} ...".format(i+1))
                print("Test Loss     = {:.3f}".format(test_loss))
                print("Test Accuracy = {:.3f}".format(test_accuracy))
                print()

                # Early cutoff - Keeps only the epoch with best accuracy
                if test_accuracy > best_result:
                    best_result = test_accuracy
                    path = saver.save(sess, 'traffic', global_step=i)
                    print("Model saved")
                    best_epoch_path = path
        print("best epoach at {}".format(best_epoch_path))


if __name__ == "__main__":
    classifier = CNNTrain()
    classifier.train()
