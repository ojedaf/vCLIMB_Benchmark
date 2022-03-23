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

from utils.dataset_conText import CILSetTask
# from utils.regularizations import *
# from utils.Regularized_Training import *
from utils.EWC import *
# from model.temporalShiftModule.ops.models import TSN
from model.model_mmc import TSN_MMC
from model.temporalShiftModule.ops.transforms import *
from model.temporalShiftModule.opts import parser
from model.temporalShiftModule.ops import dataset_config
from model.temporalShiftModule.ops.utils import AverageMeter, accuracy
from model.temporalShiftModule.ops.temporal_shift import make_temporal_pool
import yaml, pickle
import argparse

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

class ContrastiveLoss(nn.Module):
    
    def __init__(self, device, temperature=0.5):
        super().__init__()
#         self.batch_size = batch_size
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature))
#         self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=device)).float())
          
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.size(0)
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=self.device)
        
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss

best_prec1 = 0


def main():
    
    global dict_conf, device, experiment, data, list_val_acc_ii
    
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
    
    num_class = len(data['train'][0].keys())
    model = TSN_MMC(conf_model, num_class)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
    model.to(device)

    crop_size = model.model.crop_size
    scale_size = model.model.scale_size
    input_mean = model.model.input_mean
    input_std = model.model.input_std
#     policies = model.model.get_optim_policies()
    
    dataset_name = dict_conf['dataset']['name']
    train_augmentation = model.model.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf_model['lr'])
#     optimizer = torch.optim.SGD(policies,
#                                 conf_model['lr'],
#                                 momentum=conf_model['momentum'],
#                                 weight_decay=conf_model['weight_decay'])
    
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
    
    train_transforms = torchvision.transforms.Compose([
        train_augmentation,
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize
    ])
    val_transforms = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])
    
    train_cilDatasetList = CILSetTask(data['train'], path_frames, memory_size, batch_size, shuffle=True, 
                                      num_workers=num_workers, drop_last=True, pin_memory=True, 
                                      num_segments=num_segments, new_length=data_length, modality=modality, 
                                      transform=train_transforms, dense_sample=False, train_enable = True)
    
    val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True, 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, 
                                    train_enable = False)

#     for group in policies:
#         print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
#             group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
    path_model = dict_conf['checkpoints']['path_model']
    
    if dict_conf['checkpoints']['train_mode']:
        best_prec1 = 0
        current_task = 0
        current_epoch = 0
        path_best_model = path_model.format('Best_Model')
        if os.path.exists(path_best_model):
            checkpoint_dict = torch.load(path_best_model)
            model.load_state_dict(checkpoint_dict['state_dict'])
            print("load parameters model - to train")
            model.model.reg_params = checkpoint_dict['reg_params']
            best_prec1 = checkpoint_dict['accuracy']
            current_task = checkpoint_dict['current_task']
            current_epoch = checkpoint_dict['current_epoch'] + 1
            
        train_loop(current_task, current_epoch, model, optimizer, train_cilDatasetList, val_cilDatasetList, device)


def train_loop(current_task, current_epoch, model, optimizer, train_cilDatasetList, val_cilDatasetList, device):
    iter_trainDataloader = iter(train_cilDatasetList)
    num_tasks = train_cilDatasetList.num_tasks
    eval_freq = dict_conf['checkpoints']['eval_freq']
    epochs = dict_conf['model']['epochs']
    lr_type = dict_conf['model']['lr_type']
    lr_steps = dict_conf['model']['lr_steps']
    path_model = dict_conf['checkpoints']['path_model']
    batch_size = dict_conf['model']['batch_size']
    temperature = dict_conf['temperature']
    criterion = ContrastiveLoss(device, temperature = temperature).to(device)
    for j in range(current_task, num_tasks):
        train_loader_i, num_next_classes = next(iter_trainDataloader)
        best_prec1 = validate(val_cilDatasetList, model, criterion, j, False, False)
        print('Best init Acc: {} Task: {}'.format(best_prec1, j+1))
        for epoch in range(current_epoch, epochs):
#             adjust_learning_rate(optimizer, epoch, lr_type, lr_steps)
            # train for one epoch
            train(train_loader_i, model, criterion, optimizer, epoch, j)
            # evaluate on validation set
            if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
                last_epoch_val = epoch == (epochs - 1)
                prec1 = validate(val_cilDatasetList, model, criterion, j, last_epoch_val, True)

                # remember best prec@1 and save checkpoint
                is_best = prec1 >= best_prec1
                best_prec1 = max(prec1, best_prec1)
                output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
                print(output_best)
                dict_to_save = {'state_dict': model.state_dict(), 'accuracy': prec1, 'current_epoch': epoch, 
                                'current_task': j, 'optimizer': optimizer.state_dict(), 'reg_params': model.model.reg_params}
                
                save_checkpoint(dict_to_save, path_model, is_best)
        
        if num_next_classes is not None:
            print('....Update model....')
            model = load_best_checkpoint(model, path_model, j)
            
            current_epoch = 0
            policies = model.get_optim_policies()
            conf_model = dict_conf['model']

            optimizer = torch.optim.Adam(model.parameters(), lr=conf_model['lr'])
#             optimizer = torch.optim.SGD(policies,
#                                     conf_model['lr'],
#                                     momentum=conf_model['momentum'],
#                                     weight_decay=conf_model['weight_decay'])

            
           
def load_best_checkpoint(model, path_model, current_task):
    path_best_model = path_model.format('Best_Model')
    if os.path.exists(path_best_model):
        checkpoint_dict = torch.load(path_best_model)
        task_to_load = checkpoint_dict['current_task']
        if task_to_load == current_task:
            model.load_state_dict(checkpoint_dict['state_dict'])
            model.model.reg_params = checkpoint_dict['reg_params']
    return model
    
            
def train(train_loader, model, criterion, optimizer, epoch, task_id):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    task_losses = AverageMeter()
    task_top1 = AverageMeter()
    
    print_freq = dict_conf['checkpoints']['print_freq']
    no_partialbn = dict_conf['model']['no_partialbn']
    if no_partialbn:
        if torch.cuda.device_count() > 1:
            model.module.model.partialBN(False)
        else:
            model.model.partialBN(False)
    else:
        if torch.cuda.device_count() > 1:
            model.module.model.partialBN(True)
        else:
            model.model.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    with experiment.train():
        for i, (videos, label_emb, target, class_emb) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            target = target.to(device)
            label_emb = label_emb.to(device)
            videos = videos.to(device)
            class_emb = class_emb.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # compute output
            video_emb, label_emb, class_emb = model(videos, label_emb, class_emb)
            loss = criterion(video_emb, label_emb)
            
            repeated_video_emb = video_emb.unsqueeze(1).repeat(1, class_emb.size(1), 1)
            output = F.cosine_similarity(repeated_video_emb, class_emb, dim=2)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]

            experiment.log_metric("Acc_task_{}".format(task_id+1), prec1.item())
            experiment.log_metric("loss_task_{}".format(task_id+1), loss.item())

            task_losses.update(loss.item(), videos.size(0))
            task_top1.update(prec1.item(), videos.size(0))

            # compute gradient and do SGD step
            loss.backward()

            clip_gradient = dict_conf['model']['clip_gradient']
            if clip_gradient is not None:
                total_norm = clip_grad_norm_(model.parameters(), clip_gradient)

            # if task_id == 0:
            optimizer.step()
            # else:
                # optimizer.step(model.reg_params)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                output = ('Num Task: {0}, Epoch: [{1}][{2}/{3}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {task_top1.val:.3f} ({task_top1.avg:.3f})\t'.format(
                    task_id, epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=task_losses, task_top1=task_top1, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                print(output)

        experiment.log_metric("Avg_Acc_task_{}".format(task_id+1), task_top1.avg)
        experiment.log_metric("Avg_loss_task_{}".format(task_id+1), task_losses.avg)


def validate(val_cilDatasetList, model, criterion, current_task_id, last_epoch_val, enable_print):
    losses = AverageMeter()
    top1 = AverageMeter()
    total_acc = AverageMeter()
    total_loss = AverageMeter()
    val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
    print_freq = dict_conf['checkpoints']['print_freq']
    BWF = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with experiment.validate():
        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for videos, label_emb, target, class_emb in val_loader:
                    
                    target = target.to(device)
                    label_emb = label_emb.to(device)
                    videos = videos.to(device)
                    class_emb = class_emb.to(device)
                    
                    # compute output
                    video_emb, label_emb, class_emb = model(videos, label_emb, class_emb)
                    loss = criterion(video_emb, label_emb)
                    
                    repeated_video_emb = video_emb.unsqueeze(1).repeat(1, class_emb.size(1), 1)
                    output = F.cosine_similarity(repeated_video_emb, class_emb, dim=2)
                    
                    # measure accuracy and record loss
                    prec1 = accuracy(output.data, target, topk=(1,))[0]

                    losses.update(loss.item(), videos.size(0))
                    top1.update(prec1.item(), videos.size(0))

                if last_epoch_val and enable_print:
                    experiment.log_metric("Acc_task_{}".format(n_task+1), top1.avg, step=current_task_id+1)
                    if n_task == current_task_id:
                        list_val_acc_ii.append(top1.avg)
                    elif n_task < current_task_id:
                        forgetting = list_val_acc_ii[n_task] - top1.avg
                        BWF.update(forgetting, num_classes)
                total_acc.update(top1.avg, num_classes)
                total_loss.update(losses.avg, num_classes)
                top1.reset()
                losses.reset()

        if enable_print:
            output = ('Testing Results: Acc {total_acc.avg:.3f} Loss {total_loss.avg:.5f}'
                      .format(total_acc=total_acc, total_loss=total_loss))
            print(output)
            experiment.log_metric("Total_Acc_task_at_{}".format(current_task_id+1), total_acc.avg)
            experiment.log_metric("Total_loss_task_at_{}".format(current_task_id+1), total_loss.avg)
            if last_epoch_val:
                experiment.log_metric("Total_Acc_Per_task", total_acc.avg, step=current_task_id+1)
                experiment.log_metric("Total_BWF_Per_task", BWF.avg, step=current_task_id+1)

    return total_acc.avg
        
def save_checkpoint(dict_to_save, path_model, is_best):
    last_model = path_model.format('Last_Model')
    print('Saving ... ')  
    torch.save(dict_to_save, last_model)
    if is_best:
        best_model = path_model.format('Best_Model')
        torch.save(dict_to_save, best_model)
        print("Save Best Networks for task: {}, epoch: {}".format(dict_to_save['current_task'] + 1, 
                                                             dict_to_save['current_epoch'] + 1), flush=True)


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = dict_conf['model']['lr']
    momentum = dict_conf['model']['momentum']
    weight_decay = dict_conf['model']['weight_decay']
    epochs = dict_conf['model']['epochs']
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = lr * decay
        decay = weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))
        decay = weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
