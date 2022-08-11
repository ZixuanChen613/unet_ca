# import pdb
# pdb.set_trace()
import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.unet_blocks as blocks
import cont_assoc.models.panoptic_models as p_models
import cont_assoc.models.unet_models as u_models
import cont_assoc.utils.predict as pred
import cont_assoc.utils.contrastive as cont
import cont_assoc.utils.testing as testing
from cont_assoc.utils.evaluate_panoptic import PanopticKittiEvaluator
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator
from cont_assoc.utils.assoc_module import AssociationModule


class DSU4D(LightningModule):
    def __init__(self, ps_cfg, u_cfg):
        super().__init__()
        self.ps_cfg = ps_cfg
        self.u_cfg = u_cfg

        self.panoptic_model = p_models.PanopticCylinder(ps_cfg)
        self.unet_model = u_models.UNet(u_cfg)
        self.encoder = SparseEncoder(u_cfg)
        self.evaluator4D = PanopticKitti4DEvaluator(cfg=ps_cfg)
        feat_size = u_cfg.DATA_CONFIG.DATALOADER.POS_DATA_DIM
        self.pos_enc = cont.PositionalEncoder(max_freq=100000,
                                              feat_size=feat_size,
                                              dimensionality=3)
        weights = u_cfg.TRACKING.ASSOCIATION_WEIGHTS
        thresholds = u_cfg.TRACKING.ASSOCIATION_THRESHOLDS
        use_poses = u_cfg.MODEL.USE_POSES
        self.AssocModule = AssociationModule(weights, thresholds, self.encoder,
                                             self.pos_enc, use_poses)
        self.last_ins_id = 0

    def load_state_dicts(self, ps_dict, u_dict):
        self.unet_model.load_state_dict(u_dict)
        self.panoptic_model.load_state_dict(ps_dict)

    def merge_predictions(self, x, sem_logits, pred_offsets, pt_ins_feat):
        pt_sem_pred = pred.sem_voxel2point(sem_logits, x)
        clust_bandwidth = self.ps_cfg.MODEL.POST_PROCESSING.BANDWIDTH
        ins_pred = pred.cluster_ins(pt_sem_pred, pt_ins_feat, pred_offsets, x,
                                    clust_bandwidth, self.last_ins_id)
        sem_pred = pred.majority_voting(pt_sem_pred, ins_pred)

        return sem_pred, ins_pred
    
    def get_mean(self, features):
        ins_feats = []
        for i in range(len(features)):
            for j in range(len(features[i])):
                ins_feats.append(torch.mean(features[i][j], 0))
        ins_feat = torch.stack(ins_feats, 0)
        return ins_feat
        
    def get_ins_feat(self, x, ins_pred, raw_features):
        #Group points into instances
        pt_raw_feat = pred.feat_voxel2point(raw_features,x)
        pt_coordinates = x['pt_cart_xyz']

        coordinates, features, n_instances, ins_ids, ins_pred = cont.group_instances(pt_coordinates, pt_raw_feat, ins_pred)

        #Discard scans without instances
        features = [x for x in features if len(x)!=0]
        coordinates = [x for x in coordinates if len(x)!=0]

        if len(features)==0:#don't run tracking head if no ins
            # return [], [], [], ins_pred
            return [], [], [], ins_pred, {}

        #Get per-instance feature
        tracking_input = {'pt_features':features,'pt_coors':coordinates}

        # ins_feat = self.unet_model(tracking_input)          
        ins_feat = self.get_mean(features)                      # average the point features to get only one 128-d feature per instance

        if len(coordinates) != len(ins_ids):
            #scans without instances
            new_feats, new_coors = cont.fix_batches(ins_ids, features, coordinates)
            tracking_input = {'pt_features':new_feats,'pt_coors':new_coors}

        return ins_feat, n_instances, ins_ids, ins_pred, tracking_input

    def track(self, ins_pred, ins_feat, n_instances, ins_ids, tr_input, poses):
        #Separate instances of different scans
        points = tr_input['pt_coors']
        features = tr_input['pt_features']
        ins_feat = torch.split(ins_feat, n_instances)
        poses = [[p] for p in poses]

        #Instance IDs association
        ins_pred = self.AssocModule.associate(ins_pred, ins_feat,
                                                            points, features,
                                                            poses, ins_ids)

        self.last_ins_id = self.AssocModule.get_last_id()
        self.AssocModule.update_last_id(self.last_ins_id)

        return ins_pred

    def forward(self, x):
        sem_logits, pred_offsets, pt_ins_feat, raw_features = self.panoptic_model(x)        # ds-net ps backbone
        sem_pred, ins_pred = self.merge_predictions(x, sem_logits,
                                                    pred_offsets, pt_ins_feat)
        
        pt_raw_feat = pred.feat_voxel2point(raw_features, x)
        pt_raw_feat = [i.numpy() for i in pt_raw_feat]
        # new_feats = np.concatenate(pt_raw_feat, axis=0)
        x['feats'] = pt_raw_feat
        _,_,_,instance_feat = self.unet_model(x)                                            # u-net fine tune instance point embedings

        sem_pred, ins_pred = self.merge_predictions(x, sem_logits, pred_offsets, pt_ins_feat)
        
        ins_feat, n_ins, ins_ids, ins_pred, tracking_input = self.get_ins_feat(x, ins_pred, instance_feat)

        #if no instances, don't track
        if len(ins_feat)!=0:
            ins_pred = self.track(ins_pred, ins_feat, n_ins, ins_ids, tracking_input, x['pose'])        # ins_feats [7,128]
        return sem_pred, ins_pred

    def test_step(self, batch, batch_idx):
        x = batch
        sem_pred, ins_pred = self(x)

        if 'RESULTS_DIR' in self.ps_cfg:
            results_dir = self.ps_cfg.RESULTS_DIR
            class_inv_lut = self.panoptic_model.evaluator.get_class_inv_lut()
            testing.save_results(sem_pred, ins_pred, results_dir, x, class_inv_lut)

        if 'UPDATE_METRICS' in self.ps_cfg:
            self.panoptic_model.evaluator.update(sem_pred, ins_pred, x)
            self.evaluator4D.update(sem_pred, ins_pred, x)


# Modules
class SparseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.DATA_CONFIG.DATALOADER.DATA_DIM #128
        channels = [x * input_dim for x in cfg.MODEL.ENCODER.CHANNELS] #128, 128, 256, 512
        kernel_size = 3

        self.conv1 = SparseConvBlock(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = SparseConvBlock(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = SparseConvBlock(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            SparseLinearBlock(channels[-1], 2*channels[-1]),
            ME.MinkowskiDropout(),
            SparseLinearBlock(2*channels[-1], channels[-1]),
            ME.MinkowskiLinear(channels[-1], channels[-1], bias=True),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.global_avg_pool(y)
        return self.final(y).F

class SparseLinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                        ME.MinkowskiLinear(in_channel, out_channel, bias=False),
                        ME.MinkowskiBatchNorm(out_channel),
                        ME.MinkowskiLeakyReLU(),
                    )

    def forward(self, x):
        return self.layer(x)

class SparseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dimension=3):
        super().__init__()
        self.layer =  nn.Sequential(
                        ME.MinkowskiConvolution(
                            in_channel,
                            out_channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            dimension=dimension),
                        ME.MinkowskiBatchNorm(out_channel),
                        ME.MinkowskiLeakyReLU(),
                    )

    def forward(self, x):
        return self.layer(x)
