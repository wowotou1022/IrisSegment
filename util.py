from keras.preprocessing.image import img_to_array
import numpy as np
import glob
import cv2


class dataProcess(object):
    def __init__(self, out_rows, out_cols,
                 data_path = 'Data/CAV/guassian_noise_224/train/img/',
                 label_path = 'Data/CAV/guassian_noise_224/train/mask/',
                 npy_path = "Data/CAV/guassian_noise_224/", img_type = "JPG"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0
        print('-'*30)
        print('Creating training images...')
        print('-'*30)
        imgs = glob.glob(self.data_path+"/*."+self.img_type)

        print(len(imgs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:imgname.rindex(".")+1]

            img = cv2.imread(self.data_path + "/" + midname+self.img_type, 0)
            label = cv2.imread(self.label_path + "/" + midname+"tiff", 0)
            img = img_to_array(img)
            label = img_to_array(label)

            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i +=1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        # np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        print('-'*30)
        print('load train util images...')
        print('-'*30)
        imgs_train = np.load(self.npy_path+"/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
        print(imgs_train.shape)
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train,imgs_mask_train


if __name__ == "__main__":
    mydata = dataProcess(224,224)
    mydata.create_train_data()
