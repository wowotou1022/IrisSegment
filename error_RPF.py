import numpy as np
import cv2
import glob
from keras.preprocessing.image import img_to_array

def ac_error(y_true,y_pred):
    y_pred = np.round(y_pred/255)
    y_true = np.round(y_true/255)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <=0.5] = 0
    y_true[y_true > 0.5] = 1
    y_true[y_true <= 0.5] = 0
#TP y_true=1 y_pred=1
    y_tp=y_true+y_pred
    y_tp[y_tp !=2 ] = 0
    y_tp[y_tp == 2] = 1
    tp = np.sum(y_tp)

#FN y_true=1 y_pred=0
    y_fn=y_true-y_pred
    y_fn[y_fn != 1] = 0
    y_fn[y_fn == 1] = 1
    fn = np.sum(y_fn)

#TN y_true=0 y_pred=0
    y_fn = y_true + y_pred
    y_fn[y_fn != 0] = 0
    y_fn[y_fn == 0] = 1
    tn = np.sum(y_fn)

#FP y_true=0 y_pred=1
    y_fp=y_true-y_pred
    y_fp[y_fp != -1] = 0
    y_fp[y_fp == -1] = 1
    fp=np.sum(y_fp)

    R=tp/(tp+fn)
    P=tp/(tp+fp)

    F=2*tp/(2*tp+fp+fn)
    accuracy = (tp+tn)/(tp+tn+fn+fp)

    return R,P,F,accuracy


if __name__ == "__main__":
  pred_path = "Model/CAV/gaussianNoise/"

  true_path = "Data/CAV/guassian_noise_224/test/mask/"

  print('-' * 30)
  print('load predict npydata...')
  print('-' * 30)
  imgs_pred = glob.glob(pred_path + "/*.tiff")

  print('-' * 30)
  print('load ground_true npydata...')
  print('-' * 30)
  imgs_true = glob.glob(true_path + "/*.tiff")


  img_pred = np.ndarray((len(imgs_pred), 224, 224, 1), dtype=np.uint8)
  img_true = np.ndarray((len(imgs_pred), 224, 224, 1), dtype=np.uint8)

  i=0
  R_sum=0
  P_sum=0
  F_sum=0

  R_list=[]
  P_list=[]
  F_list=[]
  A_list = []
  for imgname in imgs_pred:
      midname = imgname[imgname.rindex("/") + 1:]
      y_true = cv2.imread(true_path + midname, 0)
      y_pred = cv2.imread(imgname, 0)

      y_true = img_to_array(y_true)
      y_pred = img_to_array(y_pred)
      R, P, F, accuracy = ac_error(y_true, y_pred)
      print(30 * "_")

      R_list.append(R)
      P_list.append(P)
      F_list.append(F)
      A_list.append(accuracy)

      R_sum = np.mean(R_list)
      P_sum = np.mean(P_list)
      F_sum = np.mean(F_list)
      A_sum = np.mean(A_list)

      R_std=np.sqrt(((R_list - np.mean(R_list)) ** 2).sum() / len(R_list))
      P_std = np.sqrt(((P_list - np.mean(P_list)) ** 2).sum() / len(P_list))
      F_std = np.sqrt(((F_list - np.mean(F_list)) ** 2).sum() / len(F_list))
      A_std = np.sqrt(((A_list - np.mean(A_list)) ** 2).sum() / len(A_list))

      print(i,midname,":")
      print("F :", F_sum, F_std)
      print( "R :",R_sum,R_std)
      print("P :", P_sum, P_std)
      print("Error :", 1 - A_sum, A_std)

      i+=1

