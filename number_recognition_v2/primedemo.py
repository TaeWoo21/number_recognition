#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:48:46 2016

@author: taewoo
"""

import sys
import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import time

print sys.argv 

if len(sys.argv) != 2 :
    print "usage : python  *.py  FileName.jpg"
    exit()

elif os.path.exists("./test_image/"+sys.argv[1]) is False :
    print "Not exist file"
    exit()
    

from tensorflow.examples.tutorials.mnist import personal_input_data
personal_mnist = personal_input_data.read_data_sets("", one_hot=True)

def binarization(image) :
    if image.min() == image.max():
        if image.min() < 0.5 :
            return np.zeros((1,560))
        else :
            return np.ones((1,560))
            
    else:
        thresh = threshold_otsu(image)
        binary_image = image - thresh
        input_image = (sess.run(tf.sign(binary_image)) + 1) / 2
        
        return input_image
        
def timeFormat(result_time) :
    hour = result_time // 3600
    result_time -= hour*3600
    minute = result_time // 60
    second = result_time-(minute*60)
    
    return hour, minute, second
        

x = tf.placeholder("float", [None, 560])
y = tf.placeholder("float", [None, 11])


W = tf.Variable(tf.zeros([560, 11]))
b = tf.Variable(tf.zeros([11]))

learning_rate = 0.01
training_epochs = 100
batch_size = 100
personal_batch_size = 1
display_step = 1

flags_width = 20    
flags_height = 28
flags_depth = 1

slicing_width = 22     #75, 114
slicing_height = 30        #106, 162

slicing_width_point = 0     # just for notice
slicing_height_point = 0    # just for notice

previous_width_point = 0      # just for notice

repeated_length = 20

add_width = 0.7142857
add_height = 1

add_width_point = 1     # just for notice
add_height_point = 1    # just for notice

predict_result = []

# modeling 
activation = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)

# training

for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(personal_mnist.train.num_examples/personal_batch_size)

    for i in range(total_batch) :
        batch_xs, batch_ys = personal_mnist.train.next_batch(personal_batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cross_entropy, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch

    if epoch % display_step == 0 :
        print "Epoch : ", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

print "Optimization Finished"



print '-----------------------------------------'

print "    \n<From Now On, More Image Processing Image (many number Image)>\n"


# load image data (image, many number)
with tf.gfile.FastGFile("./test_image/"+sys.argv[1], 'rb') as f:
    image = f.read()



# image processing (image, many number)
decode_image = tf.image.decode_jpeg(image, channels = flags_depth)

binarization_decode_image = binarization(sess.run(decode_image).astype("float32"))



width = len(binarization_decode_image[0]) 
height = len(binarization_decode_image)

print width, " X " , height , " Pixel"


###grayscale_image = tf.image.rgb_to_grayscale(decode_image)

all_t1 = time.time()
while slicing_width <= width and slicing_height <= height :
    t1 = time.time()
    slicing_height_point = 0
    
    while slicing_height_point + slicing_height <= height :
        
        slicing_width_point = 0
        
        previous_width_point = 0      
        
        while slicing_width_point + slicing_width <= width :
            
            slice_input_image = (binarization_decode_image)[slicing_height_point:slicing_height_point + slicing_height, slicing_width_point:slicing_width_point + int(slicing_width)]
            
            
            # exclude height remain
            if slice_input_image[1].min() == 1.0 or slice_input_image[len(slice_input_image)-2].min() == 1.0 :
                slicing_width_point += add_width_point
                continue
            # exclude height white
            if not(slice_input_image[0].min() == 1.0 and slice_input_image[len(slice_input_image)-1].min() == 1.0) :
                slicing_width_point += add_width_point
                continue
            
            if slice_input_image[:,1,:].min() == 0.0 and slice_input_image[:,0,:].min() == 0.0 :
                slicing_width_point += add_width_point
                continue
            
            if slice_input_image[:,len(slice_input_image[0])-2,:].min() == 0.0 and slice_input_image[:,len(slice_input_image[0])-1,:].min() == 0.0 :
                slicing_width_point += add_width_point
                continue
                
          
            slice_input_image = (binarization_decode_image)[slicing_height_point+1:slicing_height_point+1 + slicing_height-2, slicing_width_point+1:slicing_width_point+1 + slicing_width-2]
            
            
        
            resized_image = tf.image.resize_images(slice_input_image, flags_height, flags_width, method=1)            
            
            input_image = sess.run(resized_image)            
            
            
           
            
            input_image = np.asarray(input_image.astype("uint8").data, dtype='float32') 
            
            input_image = (1-input_image).reshape(1, flags_height * flags_width * flags_depth)
            
            
            
              
            
            
            if input_image[0][:20].max() == 0 or input_image[0][541: ].max() == 0  :
                slicing_width_point += add_width_point
            
                #print "don't fit filtering(in)!", slicing_width_point, slicing_height_point
                continue
            
            # imaga prediction (image 0, predict other format)            
            temp_result = sess.run(tf.argmax(activation,1), {x: input_image})

            if temp_result != [10] :
                
            
                if slicing_width_point > previous_width_point + repeated_length or previous_width_point == 0 :
                    
                    previous_width_point = slicing_width_point
                    predict_result.extend(temp_result)
                    print temp_result
                    
                else : 
                    
                    print temp_result, "is repeated!"
                
                plt.imshow(input_image.reshape(28,20), cmap='Greys')                                        
                plt.show()     
                plt.pause(2)
            
              
            slicing_width_point += add_width_point
            
            
        
        slicing_height_point += add_height_point
        
        #print "...", slicing_height_point
        
    print "Loop! Box size: (%d, %d), Result : " % (slicing_width, slicing_height), predict_result
    t2 = time.time()
    print "spending time : ", t2 - t1    
    
    if slicing_width + add_width <= width and slicing_height + add_height <= height :
        slicing_width += add_width
        slicing_height += add_height
    
    elif slicing_width < width and slicing_height < height :
        slicing_width = width
        slicing_height = height
        
    else :
        break;

all_t2 = time.time()
result_time = timeFormat(all_t2 - all_t1)
print "Prediction: ", predict_result, "->" ,int(result_time[0]), "h ", int(result_time[1]), "min ", int(result_time[2]), "sec"
print "Correct Answer: 3, 7, 2"

print '-----------------------------------------'
