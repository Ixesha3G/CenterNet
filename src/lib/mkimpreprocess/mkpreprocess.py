## code to plot histogram in python
import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt


def hist_equal(img_in):
# segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    #print("h_b")
    #print(h_b)
# calculate cdf    
    cdf_b = np.cumsum(h_b)  
    #print("cdf_b")
    #print(cdf_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
# mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    #print("1st step: cdf_m_b")f
    #print(cdf_m_b)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    #print("2nd step: cdf_m_b")
    #print(cdf_m_b)
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
    #print("cdf_final_b")
    #print(cdf_final_b)
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
    
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
# merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
  
    img_out = cv2.merge((img_b, img_g, img_r))
# validation
    #equ_b = cv2.equalizeHist(b)
    #equ_g = cv2.equalizeHist(g)
    #equ_r = cv2.equalizeHist(r)
    #img_out = cv2.merge((equ_b, equ_g, equ_r))
    #print(equ)
    #cv2.imwrite('output_name.png', equ)
    return img_out

def percentage_change(pr, c_ap):
# both are list
    a = np.array(pr)
    b = np.array(c_ap)
    #print(a-b)
    return np.array((b-a)/a*100).tolist()

def read_file(file_path):
# input: file path
# output: 2d list of content
# remark: read only
    with open(file_path) as reader:
      temp = reader.readlines()
    reader.close()
    content = []
    #print("len(temp)", len(temp))
    for i in range(len(temp)):
      content.append(temp[i].split())
    print(content==[])
    return content

def write_ap_record(ap, save_dir, image_mode, pr, ap_hist=None):
# input: ap_hist (2d list or NONE), ap(1d list) and save_dir 
# output: NONE
# 3 possible cases: 
# 1) ap_hist=NONE -> create file and write content
# 2) ap_hist=2d list with length 4-> update the current best percentage change if necessary and append the ap to the end
# 3) ap_hist=2d list with length 7->append the ap to the ap_hist and overwrite the file

# image_mode control which line to write/append
    ls = percentage_change(pr, ap)
    rounded_ls = [ round(item, 6) for item in ls ]
    percent_diff_s = ' '.join(map(str, rounded_ls))
    #print('rounded_ls', rounded_ls)
    cb_changed = False
    if ap_hist==None:
    
    #case1
      i=0
      with open(save_dir, 'a+') as writer:
#        print(pr)
#        print(l)
#        print(type(l))
        writer.write(percent_diff_s +'\n')
        #print(percent_diff_s)
        while(i<3):
          #print("ap[i] %6f" % ap[i])
          writer.write("{:.6f}".format(ap[i]))
          if i<2:
            writer.write('\n')
          i = i+1
        return cb_changed
    elif len(ap_hist)==4:
    #case2
      i=0
      hist = ap_hist
      #print(hist[i])
      total_percent_diff = math.fsum(list(map(float, hist[i])))
      #print('total_percent_diff', total_percent_diff)
      total_percent_diff_c = math.fsum(rounded_ls)
      #print('total_percent_diff_c', total_percent_diff_c)
      #print('total_percent_diff_c - total_percent_diff', total_percent_diff_c - total_percent_diff)
      if (total_percent_diff_c-total_percent_diff>0):
        hist[i] =  percent_diff_s
        cb_changed = True
        #print("cb_changed1", cb_changed)
      else:
        line=' '.join(map(str, hist[i]))
      with open(save_dir, 'w') as writer:
#        print(pr)
#        print(l)
#        print(type(l))
        writer.write(line +'\n')
        i=i+1#move to the next line
        
        #line 1-3
        while(i<3+1):
          line=' '.join(map(str, hist[i]))
          writer.write(line+'\n')
          i=i+1
        
        #line 4-6
        i=0 # reset
        while(i<3):
          #print("ap[i] %6f" % ap[i])
          writer.write("{:.6f}".format(ap[i]))
          if i<2:
            writer.write('\n')
          i = i+1
        #return cb_changed
    else:
    #case3: with history
    # cal the % change and replace current best if needed
      hist = ap_hist
      i=0
      for j in range(3):
        #update specific hist[i]
        # j=1,2,3 for image_mode = 0
        # j=4,5,6 for image_mode = 1
        #print("j+1+3*image_mode", j+1+3*image_mode)
        hist[j+1+3*image_mode].append("{:.6f}".format(ap[j]))
        #print("hist", hist)
      with open(save_dir,'w') as writer:
        #print('i',i)
        #print(list(map(float, hist[i])))
        total_percent_diff = math.fsum(list(map(float, hist[i]))) #orig total 
        #print('total_percent_diff', total_percent_diff)
        total_percent_diff_c = math.fsum(rounded_ls)
        #print('total_percent_diff_c', total_percent_diff_c)
        if total_percent_diff_c>total_percent_diff:
          hist[i] =  percent_diff_s
          line = hist[i]
          cb_changed = True
          #print("cb_changed2", cb_changed)
        else:
          line=' '.join(map(str, hist[i]))# first line convert back to string
        
        writer.write(line+'\n')
        i=i+1
        while(i<7):
          line=' '.join(map(str, hist[i]))
          writer.write(line+'\n')
          i = i+1
    print('cb_changed3', cb_changed)
    return cb_changed
        
def read_ap_stat(save_dir, model_performance_history_path, image_mode):
# input the opt to see the save dir

# step
# 1. cal the 3d ap
# 2. save them to the model result

# output txt file with following format
# line1 refers to highest ap 
# line2 refers to ap of easy difficulty without HE, every val_intervals do the testing once with space to separate the result
# line3 refers to ap of medium difficulty without HE, every val_intervals do the testing once with space to separate the result
# line4 refers to ap of hard difficulty without HE, every val_intervals do the testing once with space to separate the result
# line5 refers to ap of easy difficulty with HE, every val_intervals do the testing once with space to separate the result
# line6 refers to ap of medium difficulty with HE, every val_intervals do the testing once with space to separate the result
# line7 refers to ap of hard difficulty with HE, every val_intervals do the testing once with space to separate the result

# remark: assume always perform the testing without HE before testing with HE, i.e. always have line 1-4 before write to 5-7

# return boolean, True for if it is the current best
    #result_file_path = os.join(opt.save_dir, "stats_car_detection_3d.txt")
    result_file_path = save_dir
    stats = read_file(result_file_path)
    pr = [19.545927, 18.621721, 16.606655]
    ap = []
    change_cb = False
    
    # cal the 3d ap
    for i in range(3):
      j=0
      temp = 0
      while(j<len(stats[i])):
#       print("len(stats[i])",len(stats[i]))
#       print("i:", i)
#       print("j:", j)
#       print("stats[i][j]",stats[i][j]))
        temp = temp + float(stats[i][j])
        j = j+4
      ap.append(temp /11.0 * 100)
    #print(ap)
    
    # save them to the model result
    mode_opt = 0 # default testing by original image
    if image_mode == 1:
      mode_opt = 1  
    
    # read the previous history
    if not os.path.exists(model_performance_history_path):
      # create txt file and write the result
      print("create and write to file")
      change_cb = write_ap_record(ap, model_performance_history_path, image_mode, pr)
    else:
      # file exist, append to line 4-6/1-3
      hist = read_file(model_performance_history_path)
      if hist==[]:
        print("removing", model_performance_history_path)
        os.remove(model_performance_history_path)
        #go back to upper if case, i.e. ap_hist=NONE
        change_cb = write_ap_record(ap, model_performance_history_path, image_mode, pr)
      change_cb = write_ap_record(ap, model_performance_history_path, image_mode, pr, ap_hist=hist)
    return change_cb
    
    #if not os.path.exists(model_performance_result_path):
      #os.makedirs(model_performance_result_path)
      
    
    # exist some previous performance results
    # read them
    #with open(model_performance_result_path) as writer:
    #  prev_result = reader.read()
    #reader.close()
    #return False
    # 