# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:00:13 2020

@author: zhangzhao
不同模型训练
不同的模型训练的参数保存在不同的index文件：
    {
     my_Vgg_model_1.ckpt.index，
     my_Resnet_model_1.ckpt.index
     my_InceptionV1_model_1.ckpt.index，
     my_InceptionV2_model_1.ckpt.index，
     my_MobileNet_model_1.ckpt.index，     
     }
修改模块后需要在Mytensor类中修改conv_net方法

若要使用测试集验证或是要使用验证集最后分类，需要将IS_TRAIN=False


"""

import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import pickle
import matplotlib.pyplot as plt
import untitled3_vgg_model as Vgg
import untitled6_Resnet34_Model as Resnet34
import untitled5_Inception_Model as Inception
import untitled6_Inception_V2 as Inception_V2
import untitled2_Mobilnet_model as Mobilnet
import untitled4_New_Mobilnet as new_Mobilnet
import time

            
''' 全局参数 '''
IMAGE_SIZE =224
#2*(1e-4)
LEARNING_RATE = 3e-4

TRAIN_STEP = 200
TRAIN_SIZE = 85 
TEST_STEP = 100
TEST_SIZE = 50
  
IS_TRAIN = True

SAVE_PATH = '.\\model_Result\\'
data_dir = '.\\new_batch_files'
pic_path = '.\\kaggle\\test1'



def load_data(filename):
    '''从batch文件中读取图片信息'''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        return data['data'],data['label'],data['filenames']

# 读取数据的类
class InputData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        all_names = []
        for file in filenames:
            data, labels, filename = load_data(file)

            all_data.append(data)
            all_labels.append(labels)
            all_names += filename

        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._filenames = all_names

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._indicator:
            self._shuffle_data()

    def _shuffle_data(self):
        '''打乱数据顺序'''
        #生成随机数
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        '''返回每一批次的数据'''
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more examples')
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examples')
        batch_data = self._data[self._indicator : end_indicator]
        batch_labels = self._labels[self._indicator : end_indicator]
        batch_filenames = self._filenames[self._indicator : end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels, batch_filenames

# 定义一个类
class MyTensor:
    def __init__(self):
        self.batch_train_data=None
        self.batch_test_data=None
    def file_input(self):
        # 载入训练集和测试集
        train_filenames = [os.path.join(data_dir, 'train_batch_%d'%i) for i in range(1, 201)]
        test_filenames = [os.path.join(data_dir, 'test_batch')]
        self.batch_train_data = InputData(train_filenames, True)
        self.batch_test_data = InputData(test_filenames, True)

    #计算流程 
    def flow(self):
        self.x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], 'input_data')
        self.y = tf.placeholder(tf.int64, [None], 'output_data')
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = self.x / 255.0  

        #图片输入网络中
        fc = self.conv_net(self.x, self.keep_prob)
        #计算损失函数
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=fc)
        #输出两类
        self.y_ = tf.nn.softmax(fc) # 计算每一类的概率
        #将大概率设为1
        self.predict = tf.argmax(fc, 1)
        #与设定标签比较计算准确率
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.y), tf.float32))
        #使用adam优化器优化损失
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        #self.train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)
        #保留模型参数
        self.saver = tf.train.Saver(max_to_keep=1)

     
    def main(self):
        self.flow()
        #训练
        if IS_TRAIN is True:
            self.myTrain()
        #测试
        else:
            self.myTest()
    # 训练
    def myTrain(self):
        acc_list = []
        loss_list = []
        train_loss_list=[]
        train_acc_mean_list=[]
        test_acc_mean_list = []
        test_loss_mean_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if os.path.exists(SAVE_PATH + 'my_Mobilenet_model_1.ckpt.index'): 
                print("-------加载模型---------")
                model_file = tf.train.latest_checkpoint(SAVE_PATH)
                model = self.saver.restore(sess, save_path=model_file)
            else:
                print("-------创建模型----------")
                os.makedirs(SAVE_PATH, exist_ok=True)
            for i in range(TRAIN_STEP):
                '''选取混排后的85个数据'''
                train_data, train_label, _ = self.batch_train_data.next_batch(TRAIN_SIZE)

                eval_ops = [self.loss, self.acc, self.train_op]
                eval_ops_results = sess.run(eval_ops, feed_dict={
                    self.x:train_data,
                    self.y:train_label,
                    self.keep_prob:0.8   
                })
                loss_val, train_acc = eval_ops_results[0:2]
                loss_list.append(loss_val)
                acc_list.append(train_acc)
                #每两百步求平均准确率，loss
                if (i+1) % 200 == 0:
                    acc_mean = np.mean(acc_list)
                    train_acc_mean_list.append(acc_mean)
                    train_loss_list.append(loss_val)
                    print('step:{0},loss:{1:.5},acc:{2:.5},acc_mean:{3:.5}'.format(
                        i+1,loss_val,train_acc,acc_mean
                    ))
                #每两百使用测试集验证
                if (i+1) % 200 == 0:
                    test_acc_list = []
                    test_loss_list=[]
                    '''每200使用测试集进行验证'''
                    for j in range(TEST_STEP):
                        test_data, test_label, _ = self.batch_test_data.next_batch(TRAIN_SIZE)
                        loss_val,acc_val = sess.run([self.loss,self.acc],feed_dict={
                            self.x:test_data,
                            self.y:test_label,
                            self.keep_prob:1.0
                        })
                        test_loss_list.append(loss_val)
                        test_acc_list.append(acc_val)
                    test_acc_mean_list.append(np.mean(test_acc_list))
                    test_loss_mean_list.append(np.mean(test_loss_list))
                    print('[Test ] step:{0},loss:{1:.5},acc:{2:.5} ，loss_mean{3:.5}，acc_mean:{4:.5}'.format(
                        i+1,loss_val,acc_val, np.mean(test_loss_list),np.mean(test_acc_list)
                    ))
            #画图       
            plt.subplot(2,2,1) 
            plt.plot(train_loss_list  ,label='train_loss')          
            plt.title('Train Loss')
            plt.legend()
            
            plt.subplot(2,2,2)
            plt.plot(test_loss_mean_list,label='test_loss')
            plt.title('Test Loss')
            plt.legend() 
            
            plt.subplot(2,2,3)
            plt.plot(train_acc_mean_list,label='train_acc_mean')            
            plt.title('Train Mean_Accuracy')
            plt.legend()
            
            plt.subplot(2,2,4)
            plt.plot(test_acc_mean_list,label='test_acc_mean')
            plt.title('Test Mean_Accuracy')
            plt.legend()
            
            
            plt.show()
            '''保存训练后的模型''' 
            #os.makedirs(SAVE_PATH, exist_ok=True)
            self.saver.save(sess, SAVE_PATH + 'my_Mobilenet_model_1.ckpt')
         
    def myTest(self):
         with tf.Session() as sess:
            '''加载模型参数'''
            model_file = tf.train.latest_checkpoint(SAVE_PATH)
            model = self.saver.restore(sess, save_path=model_file)
            test_acc_list = []

            predict_list = []
            '''测试集再次验证'''
            for j in range(TEST_STEP):
                test_data, test_label, test_name = self.batch_test_data.next_batch(TEST_SIZE)
                for each_data, each_label, each_name in zip(test_data, test_label, test_name):
                    acc_val, y__, pre, test_img_data = sess.run(
                        [self.acc, self.y_, self.predict, self.x],
                        feed_dict={
                            self.x:each_data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                            self.y:each_label.reshape(1),
                            self.keep_prob:1.0
                        }
                    )
                    predict_list.append(pre[0])
                    test_acc_list.append(acc_val)
            
                    '''把测试结果显示出来'''
                    self.compare_test(test_img_data, each_label, pre[0], y__[0], each_name)
            print('[Test ] mean_acc:{0:.5}'.format(np.mean(test_acc_list)))
    '''返回值与标签对比验证'''
    def compare_test(self, input_image_arr, input_label, output, probability, img_name):
        classes = ['cat', 'dog']
        if input_label == output:
            result = '正确'
        else:
            result = '错误'
        print('测试【{0}】,输入的label:{1}, 预测得是{2}:{3}的概率:{4:.5}, 输入的图片名称:{5}'.format(
            result,input_label, output,classes[output], probability[output], img_name
        ))
        
    def conv_net(self, x, keep_prob):#keep_prob:dropout层的神经元保留率
    
        #fc3=Vgg.Model(x, keep_prob).Model_struct()
        #fc3=Resnet34.ResNet34(x, keep_prob).resnet_struct()
        #fc3=Inception.Inception_V1(x, keep_prob).ModelStruct()
        #fc3=Inception_V2.Inception_V2(x, keep_prob).ModelStruct()
        fc3 = Mobilnet.Mobilnet(x, keep_prob).Model_Struct()
        #fc3=new_Mobilnet.Mobilnet(x,keep_prob).Model_Struct()  
        return fc3  
 
    '''使用验证集测试'''
    def final_classify(self):
        #加载验证图片文件名
        all_test_files_dir = pic_path
        all_test_filenames = os.listdir(all_test_files_dir)
        if IS_TRAIN is False:
            #搭建计算流图
            self.flow()
            # self.classify()
            with tf.Session() as sess:
                '''加载模型'''
                model_file = tf.train.latest_checkpoint(SAVE_PATH)
                mpdel = self.saver.restore(sess,save_path=model_file)

                predict_list = []
                for each_filename in all_test_filenames:
                    each_data = self.get_img_data(os.path.join(all_test_files_dir,each_filename))
                    y__, pre, test_img_data = sess.run(
                        [self.y_, self.predict, self.x],
                        feed_dict={
                            self.x:each_data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                            self.keep_prob: 1.0
                        }
                    )
                    #分类
                    self.classify(test_img_data, pre[0], y__[0], each_filename)

        else:
            print('now is training model...')

    def classify(self, input_image_arr, output, probability, img_name):
        classes = ['cat','dog']
        single_image = input_image_arr[0] #* 255
        if output == 0:
            output_dir = 'cat/'
        else:
            output_dir = 'dog/'
        #船建文件夹保存分类结果
        os.makedirs(os.path.join('./classifyResult', output_dir), exist_ok=True)
        cv.imwrite(os.path.join('./classifyResult',output_dir, img_name),single_image)
        print('输入的图片名称:{0}，预测得有{1:5}的概率是{2}:{3}'.format(
            img_name,
            probability[output],
            output,
            classes[output]
        ))

    '''由文件名返回图片像素'''
    def get_img_data(self,img_name):
        img = cv.imread(img_name)
        resized_img = cv.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
        img_data = np.array(resized_img)    

        return img_data




if __name__ == '__main__':
    
    start_time = time.time()
    mytensor = MyTensor()
    mytensor.file_input()
    mytensor.main()  # 用于训练或测试
    #mytensor.final_classify() # 用于最后的分类
    end_time = time.time()
    print('训练结束, 用时{}秒'.format(end_time - start_time))
    