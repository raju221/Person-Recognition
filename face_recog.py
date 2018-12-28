import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import tflearn
import tensorflow as tf


TRAIN_DIR = 'TRAIN_DATA_PATH'
TEST_DIR = 'TEST_DATA_PATH'
IMG_SIZE = 50
LR = 1e-3
#LR = 0.0001


MODEL_NAME = 'facerecog-{}-{}.model'.format(LR, '6conv-basic')

def label_img(img):
    word_label = img.split('.')[-3]
    #DIY One hot encoder
    if word_label == 'raju': return [0,0,0,0,0,0,0,0,0,0,1]
    elif word_label == 'pranay': return [0,0,0,0,0,0,0,0,0,1,0]
    elif word_label == 'sai': return [0,0,0,0,0,0,0,0,1,0,0]
    elif word_label == 'akhil': return [0,0,0,0,0,0,0,1,0,0,0]
    elif word_label == 'renuka': return [0,0,0,0,0,0,1,0,0,0,0]
    elif word_label == 'mohit': return [0,0,0,0,0,1,0,0,0,0,0]
    elif word_label == 'manitej': return [0,0,0,0,1,0,0,0,0,0,0]
    elif word_label == 'praneeth': return [0,0,0,1,0,0,0,0,0,0,0]
    elif word_label == 'nithin': return [0,0,1,0,0,0,0,0,0,0,0]
    elif word_label == 'charan': return [0,1,0,0,0,0,0,0,0,0,0]
    elif word_label == 'prudhvi': return [1,0,0,0,0,0,0,0,0,0,0]

def create_train_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
test_data = process_test_data()
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')


from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 11, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('MODEL_PATH/Cov_Net{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


train = train_data[:-244]
test = train_data[-244:]


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)


import matplotlib.pyplot as plt
test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
  
    img_num = data[1]
    img_data = data[0]
  
    y = fig.add_subplot(4,5,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    print("output is :")
    print(model_out)
    print(np.shape(model_out))
    print("1st element  is :")
    #print(model_out[0,0])

    model_out1=np.array(model_out).reshape(1, 11)
    print("model out1[0,0] is  :")
    print(model_out1[0,0])
    

    if model_out1[0,10] > 0.75  : str_label='Raju'
    elif model_out1[0,9] > 0.75  : str_label='Pranay'
    elif model_out1[0,8] > 0.75  : str_label='Sai'
    elif model_out1[0,7] > 0.75  : str_label='Akhil'
    elif model_out1[0,6] > 0.75  : str_label='Renuka'
    elif model_out1[0,5] > 0.75  : str_label='Mohit'
    elif model_out1[0,4] > 0.75  : str_label='Manitej'
    elif model_out1[0,3] > 0.75  : str_label='Praneeth'
    elif model_out1[0,2] > 0.75  : str_label='Nithin'
    elif model_out1[0,1] > 0.75  : str_label='Charan'
    elif model_out1[0,0] > 0.75  : str_label='Prudhvi'
    else: str_label='Unknown' 
    
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
