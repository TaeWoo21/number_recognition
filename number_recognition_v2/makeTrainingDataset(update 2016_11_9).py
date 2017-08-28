# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:00:41 2016

@author: taewoo
"""
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def binarization(image) :
    if image.min() == image.max():
        if image.min() < 0.5 :
            return np.zeros((1,784))
        else :
            return np.ones((1,784))
            
    else:
        thresh = threshold_otsu(image)
        binary_image = image - thresh
        input_image = (sess.run(tf.sign(binary_image)) + 1) / 2 
        
        return input_image


flags_width = 20  #fit number size    
flags_height = 28   #fit number size
flags_depth = 1

num_image = 30

non_num = 34

sess = tf.Session()

#make images idx format
img_magic_number = np.array([0, 0, 8, 3]).reshape(1,4)
img_magic_number = img_magic_number.astype(np.uint8)


img_dimension_info = np.array([num_image+non_num, flags_width, flags_height], dtype= np.int32).reshape(1,3)
img_dimension_info = img_dimension_info.astype(">i")



new_idx_imgfile = open('personal_mnist_images.idx3-ubyte', 'wb')
new_idx_imgfile.write(img_magic_number)
new_idx_imgfile.write(img_dimension_info)
    
    
#make lables idx format
lable_magic_number = np.array([0, 0, 8, 1]).reshape(1,4)
lable_magic_number = lable_magic_number.astype(np.uint8)

lable_dimension_info = np.array([num_image+non_num], dtype= np.int32).reshape(1,1)
lable_dimension_info = lable_dimension_info.astype(">i")

new_idx_lablefile = open('personal_mnist_labels.idx1-ubyte', 'wb')
new_idx_lablefile.write(lable_magic_number)
new_idx_lablefile.write(lable_dimension_info)

#make idx file contents   
for setnum in range(num_image/10) :
    
    for number in range(10) : 
        # load image data (image)
        with tf.gfile.FastGFile("./Number_img/img"+str(setnum+1)+"_easy"+str(number)+".jpg", 'rb') as f:
            image = f.read()
    
        # image processing (image)
        decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
        ###grayscale_image = tf.image.rgb_to_grayscale(decode_image)
        resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
        input_image = sess.run(resized_image)
        
        # make data type int32 for Bibarization
        input_image = np.asarray(input_image.data, dtype=np.int32)
        input_image = (1-input_image).reshape(flags_depth, flags_width * flags_height) 
        
        
        # Binarization
        """thresh = threshold_otsu(input_image)
        binary_image = input_image - thresh
            
        input_image = (sess.run(tf.sign(binary_image)) + 1) / 2 *255
        """
        input_image = binarization(input_image) * 255
        input_image = input_image.astype(np.uint8)
        
        # write image idx file contents
        new_idx_imgfile.write(input_image)
        

        # write label idx file contents
        lable_value = np.array([number], dtype = np.uint8).reshape(1,1)
        new_idx_lablefile.write(lable_value)
        
print "Finish making number image data!" 

for number in range(non_num) : 
    # load image data (image)
    with tf.gfile.FastGFile("/home/taewoo/Bib_Number/Number_img/non_num"+str(number+1)+".jpg", 'rb') as f:
        image = f.read()
    
    # image processing (image)
    decode_image = tf.image.decode_jpeg(image, channels = flags_depth)
    ###grayscale_image = tf.image.rgb_to_grayscale(decode_image)
    resized_image = tf.image.resize_images(decode_image, flags_height, flags_width, method=1)
    input_image = sess.run(resized_image)
    
    # make data type int32 for Bibarization
    input_image = np.asarray(input_image.data, dtype=np.int32)
    input_image = (1-input_image).reshape(flags_depth, flags_width * flags_height)    
    
    """input_image = np.asarray(input_image.data, dtype='float32') / 255
    input_image = (1-input_image).reshape(1, flags_height * flags_width * flags_depth)"""
####################################
    """thresh = threshold_otsu(input_image)
    binary_image = input_image - thresh
    input_image = (sess.run(tf.sign(binary_image)) + 1) / 2 *255
    """
    input_image = binarization(input_image) * 255
    input_image = input_image.astype(np.uint8)
    
    
    
    """# make data type int32 for Bibarization
    input_image = np.asarray(input_image.data, dtype=np.int32).reshape(flags_depth, flags_width * flags_height)
    print input_image
        
        
    # Binarization
    input_image = (sess.run(tf.sign((255 - input_image) - 7)) + 1) / 2 * 255
    input_image = input_image.astype(np.uint8)"""
    # write image idx file contents
    
    new_idx_imgfile.write(input_image)
    
    
    # write label idx file contents
    lable_value = np.array([10], dtype = np.uint8).reshape(1,1)
    new_idx_lablefile.write(lable_value)

print "Finish making non-number image data!" 

new_idx_imgfile.close()
new_idx_lablefile.close()

