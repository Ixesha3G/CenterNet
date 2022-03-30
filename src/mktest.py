import numpy as np
import cv2
import os
import shutil
from matplotlib import pyplot as plt

import _init_paths

from mkimpreprocess.mkpreprocess import hist_equal

def folder_create_if_not_exist(path):
  directory = path
  if not os.path.exists(directory):
    os.makedirs(directory)

def split_txt_change(file_path, prefix, output_file_path):
  #print('reading: %s' % file_path)
  with open(file_path,"r") as file: 
    text = file.readlines() 
  
  #print('changing split file: %s' % file_path)
  i=0
  new_text = text
  #print('[DEBUG] len(text): ', len(text))
  while i < len(text): 
    #print('[DEBUG] i ', i)
    new_text[i] = prefix + text[i] 
    #print('[DEBUG] text[i] ', text[i])
    i = i + 1 
  
  #print('writing to: %s' % output_file_path)
  with open(output_file_path, "w") as file: 
    file.writelines(text)
  
if __name__ == '__main__':
# dir define
  #im_path = '/home/fyp/SSJ02a-21/CenterNet/images/ezgif-frame-024.jpg'
  
  #output_path = '/home/fyp/SSJ02a-21/CenterNet/images/'
  #filename = 'test.jpg'
  
  #image process
  #img = cv2.imread(im_path)
  #adj_img = hist_equal(img)
  
  #show the adjusted image
  #cv2.imshow('Adjusted image', img)
  #waits for user to press any key 
  #(this is necessary to avoid Python kernel form crashing)
  #cv2.waitKey(0) 
  
  #closing all open windows 
  #cv2.destroyAllWindows()
  
  
  #save the adjusted image
  #os.chdir(output_path)
  #cv2.imwrite(filename, adj_img)
  
#   img = cv2.imread('Chatth_Puja_Bihar_India.jpeg',0)
#   hist,bins = np.histogram(img.flatten(),256,[0,256])
#   plt.hist(img.flatten(),256,[0,256], color = 'r')
#   plt.xlim([0,256])
#   plt.show()
  
  debug = 'test'
  
  im_folder_path = '/home/fyp/SSJ02a-21/CenterNet/data/kitti/training/image_2/'
  anno_folder_path = '/home/fyp/SSJ02a-21/CenterNet/data/kitti/training/label_2/'
  calib_folder_path = '/home/fyp/SSJ02a-21/CenterNet/data/kitti/training/calib/'
  
  output_folder_path = '/home/fyp/SSJ02a-21/CenterNet/data2/kitti/training/'
  folder_create_if_not_exist(output_folder_path)
  
  output_im_folder_path = os.path.join(output_folder_path, 'image_2')
  
  output_anno_folder_path = os.path.join(output_folder_path, 'label_2')
  output_calib_folder_path = os.path.join(output_folder_path, 'calib') 
  
  prefix= ''
  #prefix = '620'
  '''
  prefix mapping
  '620' = histrogram equalization
  
  '''
  if not os.path.exists(output_im_folder_path):
    folder_create_if_not_exist(output_im_folder_path)
    for img in sorted(os.listdir(im_folder_path)):
      im_path = os.path.join(im_folder_path, img)
      new_image_name = prefix + img
      print(im_path)
      image = cv2.imread(im_path)
      image = hist_equal(image)
    
      os.chdir(output_im_folder_path)
      cv2.imwrite(new_image_name, image)
  else:
    print('Folder exist: %s' % output_im_folder_path)
  
  if not os.path.exists(output_anno_folder_path):
    folder_create_if_not_exist(output_anno_folder_path)
    folder_create_if_not_exist(output_calib_folder_path)
    for anno in sorted(os.listdir(anno_folder_path)): 
      #new anno
    
      shutil.copy(os.path.join(anno_folder_path, anno), os.path.join(output_anno_folder_path, prefix + anno))
      shutil.copy(os.path.join(calib_folder_path, anno), os.path.join(output_calib_folder_path, prefix + anno))
      #new label
  else:
    print('Folder exist: %s' % output_anno_folder_path)
    
  split_folder_path = '/home/fyp/SSJ02a-21/CenterNet/data/kitti/ImageSets_3dop'
  val_file = os.path.join(split_folder_path, 'val.txt')
  trainval_file = os.path.join(split_folder_path, 'trainval.txt')
  test_file = os.path.join(split_folder_path, 'test.txt')
  train_file = os.path.join(split_folder_path, 'train.txt')
  
  #split_output_folder_path = split_folder_path
  split_output_folder_path_3dop = '/home/fyp/SSJ02a-21/CenterNet/data2/kitti/ImageSets_3dop'
  split_output_folder_path_subcnn = '/home/fyp/SSJ02a-21/CenterNet/data2/kitti/ImageSets_subcnn'
  folder_create_if_not_exist(split_output_folder_path_3dop)
  folder_create_if_not_exist(split_output_folder_path_subcnn)
  
  val_output_file = os.path.join(split_output_folder_path_3dop, 'val.txt')
  trainval_output_file = os.path.join(split_output_folder_path_3dop, 'trainval.txt')
  test_output_file = os.path.join(split_output_folder_path_3dop, 'test.txt')
  train_output_file = os.path.join(split_output_folder_path_3dop, 'train.txt')
  
  print('change split file: %s' % val_file)
  split_txt_change(val_file, prefix, val_output_file)
  print('change split file: %s' % trainval_file)
  split_txt_change(trainval_file, prefix, trainval_output_file)
  print('change split file: %s' % test_file)
  split_txt_change(test_file, prefix, test_output_file)
  print('change split file: %s' % train_file)
  split_txt_change(train_file, prefix, train_output_file)
  
  shutil.copy(val_output_file, os.path.join(split_output_folder_path_subcnn, 'val.txt'))
  shutil.copy(trainval_output_file, os.path.join(split_output_folder_path_subcnn, 'trainval.txt'))
  shutil.copy(test_output_file, os.path.join(split_output_folder_path_subcnn, 'test.txt'))
  shutil.copy(train_output_file, os.path.join(split_output_folder_path_subcnn, 'train.txt'))