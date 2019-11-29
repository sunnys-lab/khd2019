import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np


from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold, KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

import keras.backend.tensorflow_backend as K
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

from model import cnn_sample
from dataprocessing import image_preprocessing, dataset_loader

from nasnet import NASNet


## setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True

def get_callback(ckpoint_model_name, patient):
    ES = EarlyStopping(
        monitor='val_loss',  # val_f1_m or val_loss
        patience=patient,
        mode='min',
        verbose=1)
    RR = ReduceLROnPlateau(
        monitor='val_loss', # val_f1_m or val_loss
        factor=0.1,
        patience=patient / 3,
        min_lr=1e-20,
        verbose=1,
        mode='min')
    MC = ModelCheckpoint(
        filepath=ckpoint_model_name,
        monitor='val_loss', # val_f1_m or val_loss
        verbose=1,
        save_best_only=True,
        mode='min')  # loss -> min or acc --> max
    #LG = CSVLogger(
    #    './logs/' + BASE_MODEL_STR + '_Native_' + datetime.now().strftime("%m%d%H%M%S") + '_log.csv',
    #    append=True,
    #    separator=',')

    return [ES, RR, MC]


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        # model.save_weights(file_path,'model')
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data, rescale=RESCALE, resize_factor=RESIZE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            # test 데이터를 training 데이터와 같이 전처리 하기
            X.append(image_preprocessing(d, rescale, resize_factor))
        X = np.array(X)

        pred = model.predict_classes(X)     # 모델 예측 결과: 0-3
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=10)                          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)                      # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)                     # DO NOT CHANGE num_classes, class 수는 항상 4

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    seed = 1234
    np.random.seed(seed)

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes


    # 상위 코드는 베이스라인 코드
    # hyper parameter 변경
    k_folds = 5
    patience = 9
    nb_epoch = 100
    batch_size = 16
    nb_classes = num_classes
    #img_dim = input_shape
    depth = 16
    nb_dense_block = 5
    growth_rate = 4
    nb_filter = 16
    dropout_rate = 0.5
    weight_decay = 1e-5
    l_rate = 1e-6
    using_pre_model = False
    checkpoint_str = '129'
    session_str = 'Sunny/ir_ph1_v2/133'


    """ Model """
    
    learning_rate = l_rate

    h, w = int(2400//RESIZE), int(3300//RESIZE)
    input_shape = (h, w, 3)  # input image shape
    #model = cnn_sample(in_shape=(h, w, 3), num_classes=num_classes)
    # the parameters for NASNetLarge
    model = NASNet(input_shape=input_shape,
                   penultimate_filters=2016,
                   nb_blocks=4,
                   stem_filters=48,
                   skip_reduction_layer_input=True,
                   use_auxiliary_branch=False,
                   filters_multiplier=2,
                   dropout=dropout_rate,
                   classes=nb_classes)

    model.summary()

    """ Initiate RMSprop optimizer """
    opt = SGD(lr=l_rate, decay=weight_decay, momentum=0.9, nesterov=True)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # Pre-trained Model Loading
    if using_pre_model:
        nsml.load(checkpoint=checkpoint_str, session=session_str)


    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    bind_model(model)

    if config.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train':  ### training mode일 때
        print('Training Start...')

        img_path = DATASET_PATH + '/train/'
        images, labels = dataset_loader(img_path, resize_factor=RESIZE, rescale=RESCALE)
        # containing optimal parameters

        ## data 섞기
        dataset = [[X, Y] for X, Y in zip(images, labels)]
        random.shuffle(dataset)
        X = np.array([n[0] for n in dataset])
        Y = np.array([n[1] for n in dataset])

        # 층화 모델 설정
        skf = StratifiedKFold(n_splits=k_folds, random_state=seed)

        j = 1
        ckpoint_file_names = []

        for (train_index, valid_index) in skf.split(X, Y):
            #traindf = df_train.iloc[train_index, :].reset_index()
            #validdf = df_train.iloc[valid_index, :].reset_index()
            print("TRAIN:", train_index, "VALID:", valid_index)
            X_train, X_val = X[train_index], X[valid_index]
            Y_train, Y_val = Y[train_index], Y[valid_index]

            print("==============================================")
            print("====== K Fold Validation step => %d/%d =======" % (j, k_folds))
            print("==============================================")

            ## Augmentation 예시
            kwargs = dict(
                rotation_range=5,
                zoom_range=0.05,
                shear_range=0.05,
                width_shift_range=0.05,
                height_shift_range=0.05,
                horizontal_flip=False,
                vertical_flip=False,
                fill_mode='nearest'
            )
            train_datagen = ImageDataGenerator(**kwargs)
            train_generator = train_datagen.flow(x=X, y=Y, shuffle=False, batch_size=batch_size, seed=seed)
            # then flow and fit_generator....

            """ Callback """
            monitor = 'val_loss'
            reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

            """ Training loop """
            STEP_SIZE_TRAIN = len(X) // batch_size
            print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))
            t0 = time.time()

            ## data를 trainin과 validation dataset으로 나누기
            #train_val_ratio = 0.75
            #tmp = int(len(Y) * train_val_ratio)
            #X_train = X[:tmp]
            #Y_train = Y[:tmp]
            #X_val = X[tmp:]
            #Y_val = Y[tmp:]

            ckpoint_file_name = str(j) + '_epoch_{epoch}' + '_acc_{val_acc:.4f}_loss_{val_loss:.4f}_' + '.hdf5'
            ckpoint_file_names.append(ckpoint_file_name)

            try:
                model.load_weights(ckpoint_file_name)
            except:
                pass

            for epoch in range(nb_epoch):
                t1 = time.time()
                print("### Model Fitting.. ###")
                print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
                print('check point = {}'.format(epoch))

                # for no augmentation case
                hist = model.fit(X_train, Y_train,
                                 validation_data=(X_val, Y_val),
                                 batch_size=batch_size,
                                 # initial_epoch=epoch,
                                 # callbacks=[reduce_lr],
                                 callbacks=get_callback(ckpoint_file_name, patience),
                                 verbose=2,
                                 shuffle=True
                                 )
                t2 = time.time()
                print(hist.history)
                print('Training time for one epoch : %.1f' % ((t2 - t1)))
                train_acc = hist.history['categorical_accuracy'][0]
                train_loss = hist.history['loss'][0]
                val_acc = hist.history['val_categorical_accuracy'][0]
                val_loss = hist.history['val_loss'][0]

                nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc)
                nsml.save(epoch)
            print('Total training time : %.1f' % (time.time() - t0))
            # print(model.predict_classes(X))






