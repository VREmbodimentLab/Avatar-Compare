import torch
import torch.nn as nn
import numpy as np
from models.network import AvatarPoser as net
from human_body_prior.body_model.body_model import BodyModel
import os

#load pretrained model
pretrained_model = torch.load('./model_zoo/avatarposer.pth')

bm_fname_male = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('male'))
dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('male'))
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters
device = torch.device('cuda')
body_model = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male)

model = net(input_dim=54,
            output_dim=132,
            num_layer=3,
            embed_dim=256,
            nhead = 8,
            body_model = body_model,
            device = device
        )


# load input data
# input must be (frame, 54)
input = np.load('./input.npy')

model.load_state_dict(pretrained_model)
model.eval()

output = model(input)


# save npz file to run vr
np.savez('./input.npz', output)
