import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

# 학습할 데이터의 수
n_train_data = 100
# 테스트할 데이터의 수
n_test_data = 10

train_pixels,train_list_values = mnist.train.next_batch(n_train_data) 
test_pixels,test_list_of_values = mnist.test.next_batch(n_test_data) 

train_pixel_tensor = tf.placeholder("float",[None,784])
test_pixel_tensor = tf.placeholder("float",[784])

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor,tf.negative(test_pixel_tensor))),reduction_indices=1)

pred = tf.arg_min(distance,0)

accuracy=0
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_of_values)):
        nn_index = sess.run(pred, feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i,:]})
        print("predicted number : ",np.argmax(train_list_values[nn_index]), end='')
        print(" => answer : ",np.argmax(test_list_of_values[i]))
        if np.argmax(train_list_values[nn_index])==np.argmax(test_list_of_values[i]):
            accuracy+=1.0/len(test_pixels)
    print("Accuracy = ", accuracy)
