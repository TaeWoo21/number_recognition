import tensorflow as tf
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

learning_rate = 0.1
training_epochs = 100
batch_size = 100
display_step = 1

flags_width = 28    
flags_height = 28
flags_depth = 1

# modeling 
activation = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# training
for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch) :
        batch_xs, batch_ys =mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cross_entropy, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch

    if epoch % display_step == 0 :
        print "Epoch : ", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

print "Optimization Finished"

# predict number (random)
r = random.randint(0, mnist.test.num_examples - 1)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: mnist.test.images[r:r+1]})
print "Correct Answer: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))

print '-----------------------------------------'


# load image data (image 0)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy0.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 0)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 0)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 0"

print '-----------------------------------------'

# load image data (image 1)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy1.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 1)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 1)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 1"

print '-----------------------------------------'

# load image data (image 2)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy2.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 2)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 2)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 2"


print '-----------------------------------------'

# load image data (image 3)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy3.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 3)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 3)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 3"


print '-----------------------------------------'

# load image data (image 4)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy4.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 4)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 4)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 4"


print '-----------------------------------------'

# load image data (image 5)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy5.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 5)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 5)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 5"

print '-----------------------------------------'

# load image data (image 6)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy6.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 6)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 6)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 6"

print '-----------------------------------------'


# load image data (image 7)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy7.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 7)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


# image prediction (image 7)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 7"

print '-----------------------------------------'

# load image data (image 8)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy8.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 8)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255

# image prediction (image 8)
input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 8"

print '-----------------------------------------'


# load image data (image 9)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy9.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 9)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)

# imaga prediction (image 9)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 9"

print '-----------------------------------------'


# load image data (image 1, one more time)
with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/img1_easy1.jpg", 'rb') as f:
    image = f.read()
    
# image processing (image 1, one more time)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
input_image = sess.run(resized_image)


input_image = np.asarray(input_image.data, dtype='float32') / 255


input_image = input_image.reshape(1, flags_height * flags_width * flags_depth)

# imaga prediction (image 1, one more time)
print "Prediction: ", sess.run(tf.argmax(activation,1), {x: input_image})
print "Correct Answer: 1"

print '-----------------------------------------'