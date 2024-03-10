import os
import json
import torch
from .LBS import LBS

class bird_model:
    '''
    Implementation of skined linear bird model
    '''
    def __init__(self, device=torch.device('cpu'), mesh='bird_eccv.json'):

        self.device = device

        # read in bird_mesh from the same dir
        this_dir = os.path.dirname(__file__)
        mesh_file = os.path.join(this_dir, mesh)
        with open(mesh_file, 'r') as infile:
            dd = json.load(infile)

        self.dd = dd
        self.kintree_table = torch.tensor(dd['kintree_table']).to(device)
        self.parents = self.kintree_table[0]
        self.weights = torch.tensor(dd['weights']).to(device)
        # vertices around the keypoints
        self.vert2kpt = torch.tensor(dd['vert2kpt']).to(device)

        # bone joints of the bird model
        self.J = torch.tensor(dd['J']).unsqueeze(0).to(device)
        # vertices of the bird model
        self.V = torch.tensor(dd['V']).unsqueeze(0).to(device)
        self.LBS = LBS(self.J, self.parents, self.weights)

        prior = torch.load(this_dir + '/pose_bone_prior.pth')
        self.p_m = prior['p_m'].to(device)
        self.b_m = prior['b_m'].to(device)
        self.p_cov = prior['p_cov'].to(device)
        self.b_cov = prior['b_cov'].to(device)
