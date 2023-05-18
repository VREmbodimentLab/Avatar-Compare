import os.path
import math
import argparse
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_transform
import pickle
#from utils import utils_visualize as vis
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa
from scipy.spatial.transform import Rotation as R
import gc


save_animation = False
resolution = (800,800)

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
    print(opt['path']['models'])
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


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['test']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            continue
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    model.init_test()

    pelv_error = []
    pelv_rmse_error = []
    rot_error = []
    pos_error = []
    vel_error = []
    pos_error_hands = []
    root_rot_error= []
    mpjre_global = []
    sip_global = []

    for index, test_data in enumerate(test_loader):
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))

        model.feed_data(test_data, test=True)

        model.test()

        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        predicted_angle = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']
        #print(predicted_position.shape)
        predicted_body = body_parms_pred['body']
        #predicted_global_rotation = body_parms_pred['pred_glob_rot']
        predicted_trans = body_parms_pred['trans']

        #print(predicted_trans.shape)

        gt_angle = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']
        gt_body = body_parms_gt['body']
#        gt_global_rotation = body_parms_gt['gt_glob_rot']

        predicted_root_angle = body_parms_pred['root_orient']
        gt_root_angle = body_parms_gt['root_orient']
        gt_trans = body_parms_gt['trans']
        
        #print(gt_position.shape)
        #print(gt_trans.shape)

        ##############
        predicted_root_angle_to_mat = aa2matrot(predicted_root_angle)
        gt_root_angle_to_mat = aa2matrot(gt_root_angle)

        root_rot_error_ = torch.matmul(gt_root_angle_to_mat,torch.transpose(predicted_root_angle_to_mat,1,2))
        root_rot_error_ = torch.mean((torch.linalg.norm(matrot2aa(root_rot_error_), axis = 1)))

        #root_rot_error_ = torch.mean(torch.absolute(gt_root_angle - predicted_root_angle))
        #root_rot_error_ = 
        ##############

        ##############

        predicted_global_rotation_to_aa = []

       # for i in range(len(predicted_global_rotation[0,:,0,0])):
       #     predicted_global_rotation_to_aa.append(matrot2aa(predicted_global_rotation[:,i,:,:]))

#        predicted_global_rotation_to_aa = torch.cat(predicted_global_rotation_to_aa, dim=1)

        gt_global_rotation_to_aa = []

     #   for i in range(len(gt_global_rotation[0,:,0,0])):
     #       gt_global_rotation_to_aa.append(matrot2aa(gt_global_rotation[:,i,:,:]))

#        gt_global_rotation_to_aa = torch.cat(gt_global_rotation_to_aa, dim = 1)
        
        #root_rot_error_ = torch.matmul(gt_root_angle_to_mat,torch.transpose(predicted_angle_to_mat,1,2))
        #root_rot_error_ = 
        ###############

        # Upper arm : 16, 17 upper leg : 1, 2

        ###############
#        upperarm_predicted_global_rotation_to_aa = predicted_global_rotation_to_aa[:,48:54]
 #       upperleg_predicted_global_rotation_to_aa = predicted_global_rotation_to_aa[:,3:9]
  #      sip_predicted_global_rotation_to_aa = torch.cat((upperleg_predicted_global_rotation_to_aa,upperarm_predicted_global_rotation_to_aa),dim=1)
  #      sip_predicted_global_rotmat = aa2matrot(sip_predicted_global_rotation_to_aa.reshape(-1, 3))
        
#        upperarm_gt_global_rotation_to_aa = gt_global_rotation_to_aa[:,48:54]
#        upperleg_gt_global_rotation_to_aa = gt_global_rotation_to_aa[:,3:9]
  #      sip_gt_global_rotation_to_aa = torch.cat((upperleg_gt_global_rotation_to_aa, upperarm_gt_global_rotation_to_aa),dim=1)
 #       sip_gt_global_rotmat = aa2matrot(sip_gt_global_rotation_to_aa.reshape(-1, 3))

   #     sip_global_ = torch.matmul(sip_gt_global_rotmat, torch.transpose(sip_predicted_global_rotmat,1,2))
    #    sip_global_ = torch.mean((torch.linalg.norm(matrot2aa(sip_global_), axis=1)))

        predicted_position = predicted_position#.cpu().numpy()
        gt_position = gt_position#.cpu().numpy()

        predicted_angle_to_mat = aa2matrot(predicted_angle.reshape(-1,3))                    
        gt_angle_to_mat = aa2matrot(gt_angle.reshape(-1,3))

        rot_error_ = torch.matmul(gt_angle_to_mat, torch.transpose(predicted_angle_to_mat, 1, 2))
        #print(rot_error_.shape)
        rot_error_ = torch.mean((torch.linalg.norm(matrot2aa(rot_error_), axis=1)))

        gt_pelvis = gt_position[:,0,]
        pd_pelvis = predicted_position[:,0,]

        pelv_mae_ = torch.mean(torch.sum(torch.absolute(gt_pelvis - pd_pelvis), axis = 1))
        pelv_mse_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_pelvis - pd_pelvis),axis = 1)))


#        mpjre_global_ = torch.mean(torch.absolute(gt_global_rotation_to_aa-predicted_global_rotation_to_aa))
        #sip_global_ = torch.mean(torch.absolute(sip_predicted_global_rotation_to_aa - sip_gt_global_rotation_to_aa))

        #rot_error_ = torch.mean(torch.absolute(gt_angle-predicted_angle))
        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
        pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])


        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

#        mpjre_global.append(mpjre_global_)
        pelv_error.append(pelv_mae_)
        rot_error.append(rot_error_)
        pos_error.append(pos_error_)
        vel_error.append(vel_error_)
        pelv_rmse_error.append(pelv_mse_)
        root_rot_error.append(root_rot_error_)
   #     sip_global.append(sip_global_)

        pos_error_hands.append(pos_error_hands_)


    pelv_mae = sum(pelv_error)/len(pelv_error)
    pelv_rmse_error = sum(pelv_rmse_error) / len(pelv_rmse_error)
    rot_error = sum(rot_error)/len(rot_error)
    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)
    pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)
    root_rot_error = sum(root_rot_error) / len(root_rot_error)
 #   mpjre_global = sum(mpjre_global) / len(mpjre_global)
 #   sip_global = sum(sip_global) / len(sip_global)

    # testing log
    logger.info('Average rotational error [degree]: {:<.5f}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average positional error at hand [cm]: {:<.5f}\n'.format(rot_error*57.2958, pos_error*100, vel_error*100, pos_error_hands*100))
    #print("pelv_mae is:{:.3f}".format(pelv_mae*100))

    print("pelv_rot_error is: {:<.5f}".format(root_rot_error * 57.2958))
    #print("global mpjre error is : {:<.5f}".format(mpjre_global*57.2958))
 # print(sip_global*57.2958)

    #print()
if __name__ == '__main__':
    main()
