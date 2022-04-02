from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from test import prefetch_test
from mkimpreprocess.mkpreprocess import *

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      
      print('model_{}.pth'.format(mark))
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
      pr = [19.545927, 18.621721, 16.606655] # performance reference
      for i in range(2): #0,1
      # change mode_choice
        if i==0:
          opt.mode_choice = 0
        else:
          opt.mode_choice = 1
        #print("In main.py, opt.test_scales = ", opt.test_scales)
        # run the prefetch_test to get the ap stats
        prefetch_test(opt)
        # change the current best if needed
        stat_file_dir = os.path.join(model_path, 'stats_car_detection_3d.txt')
        result_dir = os.path.join(model_path, 'model_result.txt')
        change_cb = read_ap_stat(stat_file_dir, result_dir, image_mode=i)
        
        # if the total best percentage change updated, save the model as the model_best
        if change_cb:
          save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
      
      # we save the model regardless to its performance so that we can choose to start from it if we find overfitting
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
      # check with the pr
      # if the percentage increase is the largest, save as best model.
      
      # reset to image mode 0
      opt.mode_choice = 0
#      with torch.no_grad():
#        log_dict_val, preds = trainer.val(epoch, val_loader)
#      for k, v in log_dict_val.items():
#        logger.scalar_summary('val_{}'.format(k), v, epoch)
#        logger.write('{} {:8f} | '.format(k, v))
#      if log_dict_val[opt.metric] < best:
#        best = log_dict_val[opt.metric]
#        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
#                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    #logger.write('\n')
    if epoch in opt.lr_step:
      #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)