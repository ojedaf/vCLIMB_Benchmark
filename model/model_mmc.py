from .temporalShiftModule.ops.models import TSN
import torch.nn as nn


class TSN_MMC(nn.Module):
    """
    Main model class. Implements several Simple CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, conf_model, num_class):
        super(TSN_MMC, self).__init__()
        
        num_segments = conf_model['num_segments']
        modality = conf_model['modality']
        arch = conf_model['arch']
        consensus_type = conf_model['consensus_type']
        dropout = conf_model['dropout']
        img_feature_dim = conf_model['img_feature_dim']
        no_partialbn = conf_model['no_partialbn']
        pretrain = conf_model['pretrain']
        shift = conf_model['shift']
        shift_div = conf_model['shift_div']
        shift_place = conf_model['shift_place']
        fc_lr5 = conf_model['fc_lr5']
        temporal_pool = conf_model['temporal_pool']
        non_local = conf_model['non_local']

        self.model = TSN(num_class, num_segments, modality,
                    base_model=arch,
                    consensus_type=consensus_type,
                    dropout=dropout,
                    img_feature_dim=img_feature_dim,
                    partial_bn=not no_partialbn,
                    pretrain=pretrain,
                    is_shift=shift, shift_div=shift_div, shift_place=shift_place,
                    fc_lr5=fc_lr5,
                    temporal_pool=temporal_pool,
                    non_local=non_local, get_emb = True)
        
        feature_dim = self.model.feature_dim
        self.proj_mm = nn.Linear(768, feature_dim)
        
    def get_optim_policies(self):
        policies = self.model.get_optim_policies()
        ps = list(self.proj_mm.parameters())
        for dict_param in policies:
            if 'normal_weight' == dict_param['name']:
                dict_param['params'].append(ps[0])
            elif 'normal_bias' == dict_param['name'] and len(ps) == 2:
                dict_param['params'].append(ps[1])
        return policies
        
        
    def forward(self, videos, text_emb, class_emb):
        video_emb = self.model(videos)
        text_emb = self.proj_mm(text_emb)
        class_emb = self.proj_mm(class_emb)
        return video_emb, text_emb, class_emb