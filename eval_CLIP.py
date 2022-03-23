# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
from comet_ml import Experiment
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from utils.dataset_CLIP import CILSetTask

import clip

from model.temporalShiftModule.ops.transforms import *
from model.temporalShiftModule.opts import parser
from model.temporalShiftModule.ops import dataset_config
from model.temporalShiftModule.ops.utils import AverageMeter, accuracy
import yaml, pickle
import argparse

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

best_prec1 = 0


def main():
    
    global dict_conf, device, experiment, data, list_val_acc_ii, num_segments
    
    list_val_acc_ii = []
    parser = argparse.ArgumentParser(description="CIL TSN Rehearsal Baseline")
    parser.add_argument("-conf","--conf_path", default = './conf/conf_ucf101_cil_tsn_baseline.yaml')
    args = parser.parse_args()
    conf_file = open(args.conf_path, 'r')
    print("Conf file dir: ",conf_file)
    dict_conf = yaml.load(conf_file)

    conf_model = dict_conf['model']
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
    
    api_key = dict_conf['comet']['api_key']
    workspace = dict_conf['comet']['workspace']
    project_name = dict_conf['comet']['project_name']
    experiment = Experiment(api_key=api_key,
                            project_name=project_name, workspace=workspace)
    experiment.log_parameters(parse_conf(dict_conf))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    path_data = dict_conf['dataset']['path_data']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
    
    model, preprocess = clip.load("ViT-B/32")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
    model.to(device)
    
    crop_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 224
    
    dataset_name = dict_conf['dataset']['name']
    train_augmentation = get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True, modality = modality, input_size = crop_size)
    
    path_frames = dict_conf['dataset']['path_frames']
    memory_size = dict_conf['memory']['memory_size']
    batch_size = conf_model['batch_size']
    num_workers = conf_model['num_workers']
    
    # Data loading code
    if modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    
    val_transforms = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])
    
    val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True, 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, 
                                    train_enable = False)
    
    last_task = len(data['val']) - 1
    print('last task: ',last_task)
    best_prec1 = validate(val_cilDatasetList, model, last_task, True, True)

def get_augmentation(flip=True, modality = 'RGB', input_size = 224):
    if modality == 'RGB':
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66])])
    elif modality == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=True)])
    elif modality == 'RGBDiff':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=False)])


def validate(val_cilDatasetList, model, current_task_id, last_epoch_val, enable_print):
    top1 = AverageMeter()
    total_acc = AverageMeter()
    val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
    classes_text = list(val_cilDatasetList.classes_text.values())
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with experiment.validate():
        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for videos, texts, target in val_loader:
                    
                    target = target.to(device)
                    videos = videos.to(device)
                    batch_size = videos.size(0)
                    videos = videos.view((-1, 3) + videos.size()[-2:])
                    text_tokens = clip.tokenize(["This is " + desc for desc in classes_text]).to(device)
                    
                    # compute output
                    image_features = model.encode_image(videos).float()
                    text_features = model.encode_text(text_tokens).float()
                    
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = image_features.cpu().numpy() @ text_features.cpu().numpy().T
                    similarity = similarity.reshape(-1, num_segments, similarity.shape[1])
                    output = torch.from_numpy(np.mean(similarity, axis=1)).to(device)
                    # measure accuracy and record loss
                    prec1 = accuracy(output, target, topk=(1,))[0]

                    top1.update(prec1.item(), videos.size(0))

                if last_epoch_val:
                    experiment.log_metric("Acc_task_{}".format(n_task+1), top1.avg, step=current_task_id+1)
                total_acc.update(top1.avg, num_classes)
                top1.reset()

        output = ('Testing Results: Acc {total_acc.avg:.3f}'
                  .format(total_acc=total_acc))
        print(output)
        experiment.log_metric("Total_Acc_Per_task", total_acc.avg, step=current_task_id+1)

    return total_acc.avg


if __name__ == '__main__':
    main()
