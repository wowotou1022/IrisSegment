# coding: utf-8
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt

from keras.layers import Activation, Dropout, AveragePooling2D, AtrousConvolution2D, ZeroPadding2D, Lambda, multiply
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, Input, UpSampling2D, Conv2DTranspose, Add
from keras.layers import Reshape, Dense
from keras import backend as K
from util import *


def addLayer(previousLayer, nOutChannels):
    bn = BatchNormalization(axis=-1)(previousLayer)
    relu = Activation('relu')(bn)
    relu = Conv2D(nOutChannels, (1, 1), padding="same")(relu)
    bn_1 = BatchNormalization(axis=-1)(relu)
    relu_1 = Activation('relu')(bn_1)
    conv = Conv2D(nOutChannels, 3, 3, border_mode='same')(relu_1)
    return Add()([conv, previousLayer])

def addTransition(previousLayer, nOutChannels, dropRate, blockNum):
    bn = BatchNormalization(name='tr_BatchNorm_{}'.format(blockNum), axis=-1)(previousLayer)
    relu = Activation('relu', name='tr_relu_{}'.format(blockNum))(bn)

    if dropRate is not None:
        conv = Conv2D(nOutChannels, 1, 1, border_mode='same')(relu)
        conv = BatchNormalization(axis=-1)(conv)

        conv = Activation('relu')(conv)

        avgPool = AveragePooling2D(pool_size=(2, 2))(conv)

        return avgPool
    else:
        conv = Conv2D(nOutChannels, 1, 1, border_mode='same', name='tr_conv_{}'.format(blockNum))(relu)

        conv = BatchNormalization(axis=-1)(conv)

        conv = Activation('relu')(conv)

        return conv



def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)

def expend_as_1(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1),
                  arguments={'repnum': rep})(tensor)

#calculate js
def Jensen_Shannon_divergence(inputs):
    m, n = inputs
    n = tf.clip_by_value(n, K.epsilon(), 1)
    m= tf.clip_by_value(m, K.epsilon(), 1)

    js = (m + n) / 2

    js1 = tf.multiply(m,tf.log(tf.div(m,js)))
    js2 = tf.multiply(n, tf.log(tf.div(n, js)))
    return  0.5 * (js1 + js2)

def Re_weight(inputs):
    m, n = inputs
    C=16
    shape_x = K.int_shape(m)
    z_1 = Lambda(Jensen_Shannon_divergence)([m, n])

    #16
    z_1 = Reshape((-1, 16, C))(z_1)
    shape_z_1 = K.int_shape(z_1)
    v = Lambda(lambda z: shape_z_1[1] - tf.reduce_sum(z, 1, keep_dims=True))(z_1)
    f = Dense(C, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(v)
    x_1 = Reshape((-1, 16, C))(m)
    shape_x_1 = K.int_shape(x_1)
    f = expend_as_1(f, shape_x_1[1])
    y_1 = multiply([f, x_1])
    y = Reshape((shape_x[1], shape_x[2], shape_x[3]))(y_1)
    return y

def load_data():
    mydata = dataProcess(224, 224)
    imgs_train, imgs_mask_train = mydata.load_train_data()
    return imgs_train, imgs_mask_train

def create_model():
    inputs = Input(shape=[224, 224, 1])

    dropRate = 0.5
    growthRate = [128, 128, 128, 128, 128]
    nChannels = 128
    C=128
    N = [3, 3, 3, 3, 3]

    # encoder - 1
    conv1 = Conv2D(nChannels, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)

    dense_1 = conv1
    for i in range(N[0]):
        dense_1 = addLayer(dense_1, C)
        nChannels += int(growthRate[0])

    trans_1 = BatchNormalization(axis=-1)(dense_1)
    trans_1 = Activation('relu')(trans_1)
    dense_out_1 = Conv2D(C, (1, 1), padding="same", kernel_initializer='he_normal')(trans_1)
    trans_1 = addTransition(dense_1, C, dropRate, 1)

    # encoder - 2
    dense_2 = trans_1
    dense_2 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_2)#nChannels = 128、C=128

    dense_2 = BatchNormalization(axis=-1)(dense_2)
    dense_2 = Activation('relu')(dense_2)
    #conv->BN->Relu
    for i in range(N[1]):
        dense_2 = addLayer(dense_2, C)
        nChannels += growthRate[1]
    dense_out_2 = BatchNormalization(axis=-1)(dense_2)
    dense_out_2 = Activation('relu')(dense_out_2)
    dense_out_2 = Conv2D(C, (1, 1), padding="same", kernel_initializer='he_normal')(dense_out_2)


    trans_2 = addTransition(dense_2, C, dropRate, 2)

    # encoder - 3
    dense_3 = trans_2
    dense_3 = BatchNormalization(axis=-1)(dense_3)
    dense_3 = Activation('relu')(dense_3)
    dense_3 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_3)
    for i in range(N[2]):
        dense_3 = addLayer(dense_3, C)
        nChannels += growthRate[2]

    trans_3 = addTransition(dense_3, C, dropRate, 3)

    dense_out_3 = BatchNormalization(axis=-1)(dense_3)
    dense_out_3 = Activation('relu')(dense_out_3)
    dense_out_3 = Conv2D(C, (1, 1), padding='same', kernel_initializer='he_normal')(dense_out_3)

    # encoder - 4
    dense_4 = trans_3
    dense_4 = BatchNormalization(axis=-1)(dense_4)
    dense_4 = Activation('relu')(dense_4)
    dense_4 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_4)

    for i in range(N[3]):
        dense_4 = addLayer(dense_4, C)
        nChannels += growthRate[3]

    trans_4 = addTransition(dense_4, C, dropRate, 4)

    dense_out_4 = BatchNormalization(axis=-1)(dense_4)
    dense_out_4 = Activation('relu')(dense_out_4)
    dense_out_4 = Conv2D(C, (1, 1), padding='same', kernel_initializer='he_normal')(dense_out_4)

    #encoder - 5
    dense_5 = trans_4
    dense_5 = BatchNormalization(axis=-1)(dense_5)
    dense_5 = Activation('relu')(dense_5)
    dense_5 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_5)
    for i in range(N[4]):
        dense_5 = addLayer(dense_5, C)
        nChannels += growthRate[4]
    trans5 = addTransition(dense_5, C, None, 5)
    dense_5 = BatchNormalization(axis=-1)(dense_5)
    dense_5 = Activation('relu')(dense_5)

    dense_out_5 = Conv2D(C, (1, 1), padding='same', kernel_initializer='he_normal')(dense_5)
    dense_out_5 = BatchNormalization(axis=-1)(dense_out_5)
    dense_out_5 = Activation('relu')(dense_out_5)
    dense_out_5 = AtrousConvolution2D(C, 3, 3, atrous_rate=(2, 2))(dense_out_5)
    dense_out_5 = ZeroPadding2D(padding=(2, 2))(dense_out_5)

    dense_out_5 = BatchNormalization(axis=-1)(dense_out_5)
    dense_out_5 = Activation('relu')(dense_out_5)
    dense_out_5 = AtrousConvolution2D(C, 3, 3, atrous_rate=(2, 2))(dense_out_5)
    dense_out_5 = ZeroPadding2D(padding=(2, 2))(dense_out_5)

    dense_out_1_s=dense_out_1
    dense_out_2_s=UpSampling2D(size=(2, 2))(dense_out_2)
    dense_out_3_s = UpSampling2D(size=(4, 4))(dense_out_3)
    dense_out_4_s = UpSampling2D(size=(8, 8))(dense_out_4)
    dense_out_5_s = UpSampling2D(size=(16, 16))(dense_out_5)


    #reference layer
    ks=Add()([dense_out_1_s,dense_out_2_s,dense_out_3_s,dense_out_4_s,dense_out_5_s])

    ks = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(ks)
    shape_K = K.int_shape(dense_out_1_s)
    sk = expend_as(ks, shape_K[3])

    #decoder
    dense_out_5 = Add()([dense_out_5, trans5])
    dense_out_5 = Dropout(0.5)(dense_out_5)
    up6 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(dense_out_5)
    merge6 = Add()([dense_out_4,up6])

    merge6 = Dropout(0.5)(merge6)
    conv6 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(conv6)
    merge7 = Add()([dense_out_3,up7])
    merge7 = Dropout(0.5)(merge7)

    conv7 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(conv7)
    merge8= Add()([dense_out_2,up8])
    merge8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(C,3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(conv8)
    t_1 = Add()([dense_out_1, up9])
    t_1 = Dropout(0.5)(t_1)
    conv9 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(t_1)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(C, 3,  padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)

    op_1=conv9
    op_2=UpSampling2D(size=(2, 2))(conv8)
    op_3 = UpSampling2D(size=(4, 4))(conv7)
    op_4 = UpSampling2D(size=(8, 8))(conv6)
    op_5 = UpSampling2D(size=(16, 16))(dense_out_5)

    op_1=Re_weight([op_1,sk])
    op_2=Re_weight([op_2,sk])
    op_3=Re_weight([op_3,sk])
    op_4=Re_weight([op_4,sk])
    op_5=Re_weight([op_5,sk])

    op = Add()([op_1, op_2, op_3, op_4, op_5])
    conv10 = Conv2D(1, 1, activation='sigmoid')(op)
    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    return model

def train():
    model_path = "Model/CAV/gaussianNoise/"

    print("got model")
    model = create_model()
    print("loading data")
    imgs_train, imgs_mask_train = load_data()
    print("loading data done")

    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    model_checkpoint = ModelCheckpoint(model_path + 'model_new.hdf5', monitor='loss', verbose=1,
                                       save_best_only=True, save_weights_only=False, mode='auto', period=1)
    print('Fitting model...')
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6, verbose=0, mode='min', cooldown=0,
                           min_lr=0.00000001)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=200, verbose=1, validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_checkpoint, lr, early_stop])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_path + "accuracy.png")
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_path + "loss.png")
    plt.show()


if __name__ == '__main__':
    train()
    K.clear_session()

