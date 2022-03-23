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

# from utils.dataset import CILSetTask
from utils.icarl_dataset_frames_selfsup_v2 import CILSetTask
# from utils.regularizations import *
# from utils.Regularized_Training import *
# from utils.MAS import *
# from utils.EWC import *
from model.temporalShiftModule.ops.models import TSN
from model.temporalShiftModule.ops.transforms import *
from model.temporalShiftModule.opts import parser
from model.temporalShiftModule.ops import dataset_config
from model.temporalShiftModule.ops.utils import AverageMeter, accuracy
from model.temporalShiftModule.ops.temporal_shift import make_temporal_pool
import yaml, pickle
import argparse
from torch.cuda.amp import autocast

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

best_prec1 = 0

parser = argparse.ArgumentParser(description="CIL TSN Rehearsal Baseline")
parser.add_argument("-conf","--conf_path", default = './conf/conf_ucf101_cil_tsn_baseline.yaml')
args = parser.parse_args()
conf_file = open(args.conf_path, 'r')
print("Conf file dir: ",conf_file)
dict_conf = yaml.load(conf_file)
    
type_regularization = dict_conf['model']['type_regularization']
print('Type Regularization:', type_regularization)
if type_regularization == 'EWC':
    # EWC Method
    from utils.EWC import *
else:
    # MAS Method
    from utils.MAS import *



def main():
    
    global device, experiment, data, list_val_acc_ii, type_regularization, is_activityNet
    
    list_val_acc_ii = []
#     parser = argparse.ArgumentParser(description="CIL TSN Rehearsal Baseline")
#     parser.add_argument("-conf","--conf_path", default = './conf/conf_ucf101_cil_tsn_baseline.yaml')
#     args = parser.parse_args()
#     conf_file = open(args.conf_path, 'r')
#     print("Conf file dir: ",conf_file)
#     dict_conf = yaml.load(conf_file)

    conf_model = dict_conf['model']
    
#     type_regularization = conf_model['type_regularization']
#     print('Type Regularization:', type_regularization)
#     if type_regularization == 'EWC':
#         # EWC Method
#         from utils.EWC import get_regularized_loss, on_task_update
#     else:
#         # MAS Method
#         from utils.MAS import get_regularized_loss, on_task_update
        
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
    experiment.set_name(dict_conf['comet']['name'].format(type_regularization))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    path_data = dict_conf['dataset']['path_data']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
        
    is_activityNet = dict_conf['dataset']['is_activityNet'] if 'is_activityNet' in dict_conf['dataset'] else False
    conf_model['is_activityNet'] = is_activityNet
    
    num_class = len(data['train'][0].keys())
    model = TSN(num_class, num_segments, modality,
                base_model=arch,
                consensus_type=consensus_type,
                dropout=dropout,
                img_feature_dim=img_feature_dim,
                partial_bn=not no_partialbn,
                pretrain=pretrain,
                is_shift=shift, shift_div=shift_div, shift_place=shift_place,
                fc_lr5=fc_lr5,
                temporal_pool=temporal_pool,
                non_local=non_local)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
    model.to(device)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    
    dataset_name = dict_conf['dataset']['name']
    train_augmentation = model.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True)

    optimizer = torch.optim.Adagrad(policies,
                                conf_model['lr'],
                                weight_decay=conf_model['weight_decay'])
#    optimizer = torch.optim.SGD(policies,
#                                conf_model['lr'],
#                                momentum=conf_model['momentum'],
#                                weight_decay=conf_model['weight_decay'])
    
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
    
#     train_cilDatasetList = CILSetTask(data['train'], path_frames, memory_size, batch_size, shuffle=True, 
#                                       num_workers=num_workers, drop_last=True, pin_memory=True, 
#                                       num_segments=num_segments, new_length=data_length, modality=modality, 
#                                       transform=train_transforms, dense_sample=False, train_enable = True)
    
#     val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
#                                     num_workers=num_workers, pin_memory=True, 
#                                     num_segments=num_segments, new_length=data_length, modality=modality, 
#                                     transform=val_transforms, random_shift=False, dense_sample=False, 
#                                     train_enable = False)
                                    
#     test_cilDatasetList = CILSetTask(data['test'], path_frames, memory_size, batch_size, shuffle=False, 
#                                     num_workers=num_workers, pin_memory=True, 
#                                     num_segments=num_segments, new_length=data_length, modality=modality, 
#                                     transform=val_transforms, random_shift=False, dense_sample=False, 
#                                     train_enable = False)

    train_per_noise = dict_conf['dataset']['train_per_noise'] if 'train_per_noise' in dict_conf['dataset'] else 0
    val_per_noise = dict_conf['dataset']['val_per_noise'] if 'val_per_noise' in dict_conf['dataset'] else 0
    co_threshold = dict_conf['dataset']['co_threshold'] if 'co_threshold' in dict_conf['dataset'] else 0
    
    
    train_cilDatasetList = CILSetTask(data['train'], path_frames, memory_size, batch_size, shuffle=True, 
                                      num_workers=num_workers, num_frame_to_save = conf_model['num_frame_to_save'], 
                                      is_activityNet = is_activityNet, per_noise = train_per_noise, co_threshold = co_threshold, 
                                      drop_last=True, pin_memory=True, num_segments=num_segments, new_length=data_length, 
                                      modality=modality,transform=train_transforms, dense_sample=False, train_enable = True)
    
    val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, is_activityNet = is_activityNet, per_noise = val_per_noise, 
                                    co_threshold = co_threshold, pin_memory=True, num_frame_to_save = conf_model['num_frame_to_save'], 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, train_enable = False)
    
    test_cilDatasetList = None
    if not is_activityNet:
        test_cilDatasetList = CILSetTask(data['test'], path_frames, memory_size, batch_size, shuffle=False, 
                                        num_workers=num_workers, is_activityNet = is_activityNet, per_noise = val_per_noise,
                                        co_threshold = co_threshold, pin_memory=True, num_frame_to_save = conf_model['num_frame_to_save'], 
                                        num_segments=num_segments, new_length=data_length, modality=modality, 
                                        transform=val_transforms, random_shift=False, dense_sample=False, train_enable = False)
    
    

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
    path_model = dict_conf['checkpoints']['path_model']
    
    if dict_conf['checkpoints']['train_mode']:
        best_prec1 = 0
        current_task = 0
        current_epoch = 0
        path_last_model = path_model.format('Last_Model', type_regularization)
        if os.path.exists(path_last_model):
            checkpoint_dict = torch.load(path_last_model)
            model.load_state_dict(checkpoint_dict['state_dict'])
            print("load parameters model - to train")
            model.reg_params = checkpoint_dict['reg_params']
            best_prec1 = checkpoint_dict['accuracy']
            current_task = checkpoint_dict['current_task']
            current_epoch = checkpoint_dict['current_epoch'] + 1
            
        train_loop(current_task, current_epoch, model, optimizer, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList, device, dict_conf['reg_lambda'])


def train_loop(current_task, current_epoch, model, optimizer, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList, device, reg_lambda = 1):
    iter_trainDataloader = iter(train_cilDatasetList)
    num_tasks = train_cilDatasetList.num_tasks
    eval_freq = dict_conf['checkpoints']['eval_freq']
    epochs = dict_conf['model']['epochs']
    lr_type = dict_conf['model']['lr_type']
    lr_steps = dict_conf['model']['lr_steps']
    path_model = dict_conf['checkpoints']['path_model']
    for j in range(current_task, num_tasks):
        criterion = nn.CrossEntropyLoss().to(device)
#         train_loader_i, num_next_classes = next(iter_trainDataloader)
        _, _, train_loader_i, _, num_next_classes = next(iter_trainDataloader)
        best_prec1 = validate(val_cilDatasetList, model, criterion, j)
        print('Best init Acc: {} Task: {}'.format(best_prec1, j+1))
        for epoch in range(current_epoch, epochs):
#             adjust_learning_rate(optimizer, epoch, lr_type, lr_steps)
            # train for one epoch
            train(train_loader_i, model, criterion, optimizer, epoch, j)

            # evaluate on validation set
            if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
                
                with experiment.validate():
                    prec1 = validate(val_cilDatasetList, model, criterion, j)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 >= best_prec1
                    best_prec1 = max(prec1, best_prec1)
                    output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
                    print(output_best)
                    dict_to_save = {'state_dict': model.state_dict(), 'accuracy': prec1, 'current_epoch': epoch, 
                                    'current_task': j, 'optimizer': optimizer.state_dict(), 'reg_params': model.reg_params}
                    
                    save_checkpoint(dict_to_save, path_model, is_best)
  
        model = load_best_checkpoint(model, path_model, j)
        with experiment.validate():
            total_acc_val = final_validate(val_cilDatasetList, model, j)
            print('Val total Accuracy: %d %%' % total_acc_val)
        
        if not is_activityNet:
            with experiment.test():
                total_acc_test = final_validate(test_cilDatasetList, model, j)
                print('Test total Accuracy: %d %%' % total_acc_test)
        
        
        if num_next_classes is not None:
            print('....Update model....')
            init_output_size = model.new_fc.out_features
            print('init_output_size: ', init_output_size)

            # Load the best model achieved for the current task
            model = load_best_checkpoint(model, path_model, j)
            model.augment_classification(num_next_classes, device)

            # Calculate the importance of weights for current task
            # EWC Method
            model.reg_params = on_task_update(train_loader_i, device, optimizer, model)
            
            # MAS METHOD
            # model = accumulate_objective_based_weights_sparce(train_loader_i,model,norm='L2', init_task=j)
            # model.reg_params['lambda']=reg_lambda
            
            current_epoch = 0
            policies = model.get_optim_policies()
            conf_model = dict_conf['model']
            # optimizer = Weight_Regularized_SGD(policies, conf_model['lr'], momentum=conf_model['momentum'], weight_decay=conf_model['weight_decay'], L1_decay=False)
            # optimizer = torch.optim.SGD(policies,
                                    # conf_model['lr'],
                                    # momentum=conf_model['momentum'],
                                    # weight_decay=conf_model['weight_decay'])
            optimizer = torch.optim.Adagrad(policies,
                                conf_model['lr'],
                                weight_decay=conf_model['weight_decay'])

            
           
def load_best_checkpoint(model, path_model, current_task):
    path_best_model = path_model.format('Best_Model', type_regularization)
    if os.path.exists(path_best_model):
        checkpoint_dict = torch.load(path_best_model)
        task_to_load = checkpoint_dict['current_task']
        if task_to_load == current_task:
            model.load_state_dict(checkpoint_dict['state_dict'])
            model.reg_params = checkpoint_dict['reg_params']
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
            model.module.partialBN(False)
        else:
            model.partialBN(False)
    else:
        if torch.cuda.device_count() > 1:
            model.module.partialBN(True)
        else:
            model.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    with experiment.train():
        for i, (_, _, videos, _, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.to(device)
            videos = videos.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            with autocast():

                # compute output
                output = model(videos)
                # loss = criterion(output, target)
                loss = get_regularized_loss(criterion, output, target, model, dict_conf['reg_lambda'])

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target, topk=(1,))[0]

                experiment.log_metric("Acc_task_{}".format(task_id+1), prec1.item())
                experiment.log_metric("loss_task_{}".format(task_id+1), loss.item())

                task_losses.update(loss.item(), videos.size(0))
                task_top1.update(prec1.item(), videos.size(0))

                # compute gradient and do SGD or Adagrad step
            
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
        

def validate(val_cilDatasetList, model, criterion, current_task_id):
    losses = AverageMeter()
    top1 = AverageMeter()
    total_acc = AverageMeter()
    total_loss = AverageMeter()
    val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
    
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
            for _, _, videos, _, target in val_loader:
                target = target.to(device)
                videos = videos.to(device)
                # compute output
                output = model(videos)
                loss = criterion(output, target)
                
                # measure accuracy and record loss
                acc_val = accuracy(output.data, target, topk=(1,))[0]
                
                top1.update(acc_val.item(), videos.size(0))
                losses.update(loss.item(), videos.size(0))

            total_acc.update(top1.avg, num_classes)
            total_loss.update(losses.avg, num_classes)
            print('Train... task : {}, acc with classifier: {} loss: {}'.format(n_task, top1.avg, losses.avg))
            top1.reset()
            losses.reset()
    output = ('Pre val Results: Pre_Acc {total_acc.avg:.3f}'
              .format(total_acc=total_acc))
    print(output)
    return total_acc.avg
    
def final_validate(val_cilDatasetList, model, current_task_id):
    top1 = AverageMeter()
    total_acc = AverageMeter()
    val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
    BWF = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
            for _, _, videos, _, target in val_loader:
                target = target.to(device)
                videos = videos.to(device)
                # compute output
                output = model(videos)

                # check the accuracy function
                acc_val = accuracy(output.data, target, topk=(1,))[0]

                # top1.update(acc_val.item(), videos.size(0))
                top1.update(acc_val, videos.size(0))

            experiment.log_metric("Acc_task_{}".format(n_task+1), top1.avg, step=current_task_id+1)
            if n_task == current_task_id:
                list_val_acc_ii.append(top1.avg)
            elif n_task < current_task_id:
                forgetting = list_val_acc_ii[n_task] - top1.avg
                BWF.update(forgetting, num_classes)
            total_acc.update(top1.avg, num_classes)
            top1.reset()

        output = ('Testing Results: Acc {total_acc.avg:.3f}'
                      .format(total_acc=total_acc))
        print(output)
        experiment.log_metric("Total_Acc_Per_task", total_acc.avg, step=current_task_id+1)
        experiment.log_metric("Total_BWF_Per_task", BWF.avg, step=current_task_id+1)
    return total_acc.avg
        
def save_checkpoint(dict_to_save, path_model, is_best):
    last_model_path = path_model.format('Last_Model', type_regularization)
    print('Saving ... ')  
    torch.save(dict_to_save, last_model_path)
    if is_best:
        best_model_path = path_model.format('Best_Model', type_regularization)
        torch.save(dict_to_save, best_model_path)
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
