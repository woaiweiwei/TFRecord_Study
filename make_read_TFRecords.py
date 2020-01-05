
import tensorflow as tf
import os
from PIL import Image



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
    return image_label_dict

#train数据集路径
train_directory = 'flowers/train/'
#test数据集路径
test_directory = 'flowers/test/'
train_image_label = load_sample(train_directory)
test_image_label = load_sample(test_directory)
print(test_image_label)


#将train与test文件中的数据集制成TFRecord文件
def makeTFRec(image_label_dict,tfrecord_name):
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    for key in image_label_dict:
        #读取每张图片
        img = Image.open(key)
        #将图片大小统一尺寸
        img = img.resize((WITH,HEIGHT))
        #将图片转化成二进制
        img_raw = img.tobytes()
        #将二进制图片与其对应的标签存入tfrecords中
        example = tf.train.Example(features = tf.train.Features(feature={
                    'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label_dict[key]])),
                    'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw]))
                }))
        writer.write(example.SerializeToString())
    writer.close()
    
#将图片的尺寸统一
WITH = 40
HEIGHT = 40
makeTFRec(train_image_label,'train.tfrecords')
makeTFRec(test_image_label,'test.tfrecords')



#读取tfrecords中数据方法
def read_tfrecords(tfrecord_name):
    #将tfrecords读入流中,乱序操作并循环读取
    filename_queue = tf.train.string_input_producer([tfrecord_name]) 
    reader = tf.TFRecordReader()
    #返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    #取出文件中包含image和label的feature对象
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    #将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    #改变像素数组的大小，彩图是3通道的
    image = tf.reshape(image, [WITH, HEIGHT, 3])
    #将像素数组归一化
    image = tf.cast(image,tf.float32)*(1./255)-0.5
    #读取标签
    label = tf.cast(features['label'], tf.int32)
    #将标签制成one_hot
    label = tf.one_hot(label,depth=classes_num,on_value=1)
    #按批次大小乱序读取数据
    x_batch, y_batch = tf.train.shuffle_batch([image,label], 
                                              batch_size=batch_size, 
                                              num_threads=1, capacity=30*batch_size,
                                              min_after_dequeue=15*batch_size)
    return x_batch,y_batch


#图片总共由3类，用于one_hot标签
classes_num = 3
#批次大小
batch_size = 4
#获取训练集数据
xs_train,ys_train = read_tfrecords('train.tfrecords')
#获取测试集数据
xs_test,ys_test = read_tfrecords('test.tfrecords')

with tf.Session() as sess: 
    #必写
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    
    #获取训练集数据
    xs_train,ys_train = sess.run([xs_train,ys_train])
    print(xs_train.shape)
    print(ys_train)
    
    #必写
    coord.request_stop()
    coord.join(threads)


