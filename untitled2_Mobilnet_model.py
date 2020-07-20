# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:43:10 2020

@author: mechrevo
"""
import numpy as np

import tensorflow as tf
from tensorflow.layers import conv2d, batch_normalization,max_pooling2d,average_pooling2d,dense
from tensorflow.nn import relu,dropout,depthwise_conv2d_native,separable_conv2d
class Mobilnet:
    def __init__(self,x,keep_prob):
        self.input=x
        #dropout层神经元存活率
        self.keep_prob=keep_prob
    #深度可分离卷积模块
    def depthBlock(self,x,num,input_channels,out_channels,stride):
        #深度卷积核
        filter1 = tf.get_variable("depthfilter_%d_1"% num,
                                  shape=[3,3,input_channels,1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        #点卷积核
        filter2 = tf.get_variable("pointfilter_%d_2"% num,
                                  shape=[1,1,input_channels,out_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        con1=separable_conv2d(input=x,
                         depthwise_filter=filter1,
                         pointwise_filter=filter2,
                         strides=[1,stride,stride,1],
                         padding='SAME')
        b2=batch_normalization(con1)
        r2=relu(b2)
        return r2
    def Model_Struct(self):
            
        conv1=conv2d(self.input,filters=32,kernel_size=(3,3),strides=2,padding='SAME')
        b1=batch_normalization(conv1)
        r1=relu(b1)
        
        pool1 = max_pooling2d(r1,(3,3),(2,2),padding='same')    
        # 56 32
        
        block1 = self.depthBlock(pool1,1,32,64,1) #56 64
       
        block2 = self.depthBlock(block1,2,64,128,2)#28 128

        block3 = self.depthBlock(block2,3,128,128,1) #28 128
        
        block4 = self.depthBlock(block3,4,128,256,2) #14 256
        
        block5 = self.depthBlock(block4,5,256,256,1) #14 256
        
        block6 = self.depthBlock(block5,6,256,512,2) #7 512
        
        block7_1 = self.depthBlock(block6,71,512,512,1)       
        block7_2 =self.depthBlock(block7_1,72,512,512,1)
        block7_3 = self.depthBlock(block7_2,73,512,512,1)
        
        #平均池化
        aver_pool = average_pooling2d(block7_3 ,(7,7),(1,1))
        
      
  
              
        
        
        flatten = tf.layers.flatten(aver_pool)  # 把网络展平，以输入到后面的全连接层
        fc1 = dense(flatten, 700, tf.nn.relu)#700个神经元    
        fc1_dropout = dropout(fc1, keep_prob=self.keep_prob)
        
        fc2 = tf.layers.dense(fc1_dropout, 256, tf.nn.relu)
        fc2_dropout = dropout(fc2, keep_prob=self.keep_prob)
        
        fc3= dense(fc2_dropout, 2, None)  # 得到两类输出fc3
        return fc3
        
