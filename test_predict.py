from model import *
from util import *
from keras import backend as K
from keras.preprocessing.image import array_to_img
import cv2
import glob
import numpy as np

if __name__ == "__main__":
    data_path = "Data/CAV/guassian_noise_224/test/img/"

    model_path = "Model/CAV/gaussianNoise/"

    img_type = "jpg"

    imgs = glob.glob(data_path + "/*." + img_type)

    # import the model
    model = create_model()

    # load the model
    model.load_weights(model_path + 'model_new.hdf5')

    for imgname in imgs:
        image_rgb = (np.array(cv2.imread(imgname, 0))).astype(np.float32)
        image = np.expand_dims(image_rgb, axis=-1) / 255
        net_in = np.zeros((1, 224, 224, 1), dtype=np.float32)
        net_in[0] = image

        midname = imgname[imgname.rindex("/") + 1:imgname.rindex(".") + 1]

        imgs_mask_test = model.predict(net_in)[0]

        img = imgs_mask_test

        img = array_to_img(img)
        img.save(model_path+midname+"tiff")

    K.clear_session()
