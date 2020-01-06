
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
from PIL import Image
import numpy as np


# In[2]:

#生成图片与对应标签的字典
def load_sample(sample_dir):
    #图片名列表
    lfilenames = []
    #标签名列表
    labelnames = []
    #遍历文件夹
    for (dirpath,dirnames,filenames) in os.walk(sample_dir):
        #遍历图片
        for filename in filenames:
            #每张图片的路径名
            filename_path = os.sep.join([dirpath,filename])
            #添加文件名
            lfilenames.append(filename_path)
            #添加文件名对应的标签
            labelnames.append(dirpath.split('/')[-1])
            
    #生成标签名列表
    lab = list(sorted(set(labelnames)))
    #生成标签字典
    labdict = dict(zip(lab,list(range(len(lab)))))
    #生成与图片对应的标签列表
    labels = [labdict[i] for i in labelnames]
    #图片与标签字典
    image_label_dict = dict(zip(lfilenames,labels))
    #将文件名与标签列表打乱
    lfilenames = []
    labels = []
    for key in image_label_dict:
        lfilenames.append(key)
        labels.append(image_label_dict[key])
    #返回文件名与标签列表
    return lfilenames,labels

#train数据集路径
train_directory = 'flowers/train/'
#test数据集路径
test_directory = 'flowers/test/'
train_filenames,train_labels = load_sample(train_directory)
test_filenames,test_labels = load_sample(test_directory)
print(train_filenames,train_labels)


# In[3]:

#将图片制成Dataset方法
def make_Dataset(filenames,labels,size,batch_size):
    #生成dataset对象
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    #转化为图片数据集
    dataset = dataset.map(_parseone)
    #按批次组合数据
    dataset = dataset.batch(batch_size)
    return dataset

#解析图片文件的方法
def _parseone(filename,label):
    #读取所有图片
    image_string = tf.read_file(filename)
    #将图片解码并返回空的shape
    image_decoded = tf.image.decode_image(image_string)
    #因为是空的shape，所以需要设置shape
    image_decoded.set_shape([None,None,None])
    image_decoded = tf.image.resize_images(image_decoded,size)
    #归一化
    image_decoded = image_decoded/255.
    #将归一化后的像素矩阵转化为image张量
    image_decoded = tf.cast(image_decoded,dtype=tf.float32)
    #将label转为张量
    label = tf.cast(tf.reshape(label,[]),dtype=tf.int32)
    #将标签制成one_hot
    label = tf.one_hot(label,depth=classes_num,on_value=1)
    return image_decoded,label

#从Dataset中取出数据的方法
def getdata(dataset):
    #生成一个迭代器
    iterator = dataset.make_one_shot_iterator()
    #从迭代器中取出一个数据
    one_element = iterator.get_next()
    return one_element


# In[5]:

os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"

#设置参数
#t图片尺寸
WITH = 28
HEIGHT = 28
size = [WITH,HEIGHT]
#每批次数据数量
batch_size = 400
#训练集与测试集总的数量
train_total = len(train_labels)
test_total = len(test_labels)
#图片总共由3类，用于one_hot标签
classes_num = 3
#将所有图片训练一轮所需要的总的训练次数
total_batch = int(train_total/batch_size) 
#总共循环训练轮数
train_epochs = 50
#定义初始学习率
learning_rate = 0.005

#制作训练与测试集Dataset
train_dataset = make_Dataset(train_filenames,train_labels,size,batch_size)
test_dataset = make_Dataset(test_filenames,test_labels,size,test_total)

#获取数据
train_data = []
for i in range(train_epochs):
    train_data.append(getdata(train_dataset))

test_data = getdata(test_dataset)

#定义图片和标签的占位符
#None 表示张量的第一维度可以接受任意长度,3表示图片通道数
x = tf.placeholder(tf.float32,shape = [None,WITH,HEIGHT,3])
#None 表示张量的第一维度可以接受任意长度,class_num表示标签类别个数
y = tf.placeholder(tf.float32,shape = [None,classes_num])
keep_prob  = tf.placeholder(tf.float32)

#定义权重及偏置值变量
W1 = tf.Variable(tf.random_normal(([int(WITH/4)*int(HEIGHT/4)*256,1024])))
b1 = tf.Variable(tf.constant(0.1,shape=[1024]))
W2 = tf.Variable(tf.random_normal(([1024,classes_num])))
b2 = tf.Variable(tf.constant(0.1,shape=[classes_num]))

#定义隐藏层
def hidden_layer(inputs):
    #要用激活函数
    return tf.nn.relu(tf.matmul(inputs,W1)+b1)

#定义权重方法：
def get_filter(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

#定义偏置值方法：
def get_bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

w_con1 = get_filter([5,5,3,128])
b_con1 = get_bias([128])
w_con2 = get_filter([5,5,128,256])
b_con2 = get_bias([256])

#第一层卷积输出,输出大小为 batch_size * WITH * HEIGHT * 15：
h_conv1 = tf.nn.conv2d(x,filter=w_con1,strides=[1,1,1,1],padding='SAME')+b_con1
#激活函数：
h1 = tf.nn.relu(h_conv1)
#第一层池化输出，输出大小为 batch_size * (WITH/2) *(HEIGHT/2) * 15：
h_pool1 = tf.nn.max_pool(h1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#第二层卷积输出,输出大小为 batch_size * (WITH/2) *(HEIGHT/2) * 30：
h_conv2 = tf.nn.conv2d(h_pool1,filter=w_con2,strides=[1,1,1,1],padding='SAME')+b_con2
#激活函数：
h2 = tf.nn.relu(h_conv2)
#第一层池化输出，输出大小为 batch_size * (WITH/4) *(HEIGHT/4) * 30：
h_pool2 = tf.nn.max_pool(h2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

      
#将池化后的输出改变下尺寸，
h_reshape = tf.reshape(h_pool2,[-1,int(WITH/4)*int(HEIGHT/4)*256])
    
#隐藏层输出,并使用dropout
h3 = hidden_layer(h_reshape)

h_drop1 = tf.nn.dropout(h3,keep_prob)

#预测值，这里不用激活函数，因为等下要用tensorflow定义好的softmax交叉熵函数
pred = tf.matmul(h_drop1,W2) + b2

#定义交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y)

#定义总的损失函数
loss = tf.reduce_mean(cross_entropy)

#定义优化器
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

 
#以下是测试模型
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#准确率：
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    #开始训练
    print("以下是训练模型每轮训练误差：")
    for epoch in range(train_epochs):
        #定义平均loss值
        avg_loss = 0.
        #循环所有数据
        for i in range(total_batch):
            #获取批次训练数据
            batch_xs,batch_ys = sess.run(train_data[epoch])
            _,c,acc = sess.run([opt,loss,accuracy],feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.8})
            #平均loss
            avg_loss += c / total_batch
        #显示每轮的结果
        print('Epoch:',epoch+1,',     loss:','{:.9f}'.format(avg_loss),',     accuracy:','{:.5f}'.format(acc))
        
    print("\n训练模型结束，以下是测试模型准确率：")
    
    #获取测试数据
    test_xs,test_ys = sess.run(test_data)
    acc = sess.run(accuracy,feed_dict={x:test_xs,y:test_ys,keep_prob:1.0})
    print('Accuracy:',acc)


# In[ ]:



