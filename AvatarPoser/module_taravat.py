# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:30:07 2023

@author: hello
"""

import numpy as np
import torch
import zmq
import time
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
import os.path
import argparse
import logging
from utils import utils_logger
from utils import utils_option as option
from models.select_model import define_Model




window_size =10
def main(json_path='options/test_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)



    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()
    
  
    

    model.init_test()

    rot_error = []
    pos_error = []
    vel_error = []
    pos_error_hands = []

    # model.eval()
    # model = define_Model()
    # model = torch.load ('./model_zoo/avatarposer.pth', map_location = torch.device('cuda'))  

    context = zmq.Context()
    replysocket = context.socket(zmq.REP)
    replysocket.bind('tcp://*:3550')
    print("server initiated") 
    while True:
        msg = replysocket.recv()
        message = np.fromstring(msg, dtype=np.float32, sep=',')
        message = np.reshape(message, (window_size, -1))
        start = time.time()
        input = torch.from_numpy(message).float()
        input = torch.unsqueeze(input, 0)
        print("the input value is :", input)
        predicted = model.test(input)
        
        predicted = torch.squeeze(predicted)
        predicted = predicted.detach().cpu().numpy()
    
    
        output = ' '.join(str(x) for x in predicted)
        replysocket.send_string(output)
        end = time.time()
        print('the running time is:' + str(end-start))
    
    

        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        predicted_angle = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']
        predicted_body = body_parms_pred['body']

        gt_angle = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']
        gt_body = body_parms_gt['body']


        predicted_position = predicted_position#.cpu().numpy()
        gt_position = gt_position#.cpu().numpy()

        predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)                    
        gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)


        rot_error_ = torch.mean(torch.absolute(gt_angle-predicted_angle))
        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
        pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

        rot_error.append(rot_error_)
        pos_error.append(pos_error_)
        vel_error.append(vel_error_)

        pos_error_hands.append(pos_error_hands_)



    rot_error = sum(rot_error)/len(rot_error)
    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)
    pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)


    # testing log
    logger.info('Average rotational error [degree]: {:<.5f}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average positional error at hand [cm]: {:<.5f}\n'.format(rot_error*57.2958, pos_error*100, vel_error*100, pos_error_hands*100))


if __name__ == '__main__':
    main()

