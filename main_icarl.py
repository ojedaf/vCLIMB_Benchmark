from comet_ml import Experiment
from model.iCaRL import iCaRL
import torch
from model.temporalShiftModule.ops.transforms import *
# from utils.icarl_dataset import CILSetTask
from utils.icarl_dataset_frames import CILSetTask
import argparse
import yaml, pickle
import torch.nn as nn
import os
import random
random.seed(10)

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

def main():
    
    global dict_conf, device, experiment, data, list_val_acc_ii, memory_size, type_sampling
    
    list_val_acc_ii = []
    parser = argparse.ArgumentParser(description="iCaRL TSN Baseline")
    parser.add_argument("-conf","--conf_path", default = './conf/conf_ucf101_icarl_tsn.yaml')
    args = parser.parse_args()
    conf_file = open(args.conf_path, 'r')
    print("Conf file dir: ",conf_file)
    dict_conf = yaml.load(conf_file)

    conf_model = dict_conf['model']
    
    api_key = dict_conf['comet']['api_key']
    workspace = dict_conf['comet']['workspace']
    project_name = dict_conf['comet']['project_name']
    experiment = Experiment(api_key=api_key,
                            project_name=project_name, workspace=workspace)
    experiment.log_parameters(parse_conf(dict_conf))
    experiment.set_name(dict_conf['comet']['name'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    path_data = dict_conf['dataset']['path_data']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
    
    num_class = len(data['train'][0].keys())
    
    type_sampling = dict_conf['memory']['type_mem'] if 'type_mem' in dict_conf['memory'] else 'icarl'
    print('sampling strategy:', type_sampling)
    
    model = iCaRL(conf_model, num_class, dict_conf['checkpoints'])

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    
    dataset_name = dict_conf['dataset']['name']
    
    train_augmentation = model.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True)

    optimizer = torch.optim.SGD(policies,
                                conf_model['lr'],
                                momentum=conf_model['momentum'],
                                weight_decay=conf_model['weight_decay'])
    
    path_frames = dict_conf['dataset']['path_frames']
    memory_size = dict_conf['memory']['memory_size']
    batch_size = conf_model['batch_size']
    num_workers = conf_model['num_workers']
    arch = conf_model['arch']
    modality = conf_model['modality']
    num_segments = conf_model['num_segments']

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
                                      num_workers=num_workers, num_frame_to_save = conf_model['num_frame_to_save'], drop_last=True,
                                      pin_memory=True, num_segments=num_segments, new_length=data_length, modality=modality, 
                                      transform=train_transforms, dense_sample=False, train_enable = True)
    
    val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True, num_frame_to_save = conf_model['num_frame_to_save'], 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, 
                                    train_enable = False)
    
    test_cilDatasetList = CILSetTask(data['test'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True, num_frame_to_save = conf_model['num_frame_to_save'], 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, 
                                    train_enable = False)

    # define loss function (criterion) and optimizer
    cls_loss = nn.CrossEntropyLoss().to(device)
    dist_loss = nn.BCEWithLogitsLoss().to(device)
    model.set_losses(cls_loss, dist_loss)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
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
            best_prec1 = checkpoint_dict['accuracy']
            current_task = checkpoint_dict['current_task']
            current_epoch = checkpoint_dict['current_epoch'] + 1
            
        train_loop(model, optimizer, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList)

def train_loop(model, optimizer, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList):
    iter_trainDataloader = iter(train_cilDatasetList)
    num_tasks = train_cilDatasetList.num_tasks
    eval_freq = dict_conf['checkpoints']['eval_freq']
    path_model = dict_conf['checkpoints']['path_model']
    num_epochs = dict_conf['model']['epochs']
    
    for j in range(num_tasks):
        classes, data, train_loader_i, len_data, num_next_classes = next(iter_trainDataloader)
        model.train(train_loader_i, len_data, optimizer, num_epochs, experiment, j, val_cilDatasetList)
        if torch.cuda.device_count() > 1:
            m = memory_size // model.feature_encoder.module.new_fc.out_features
        else:
            m = memory_size // model.feature_encoder.new_fc.out_features
        model.add_samples_to_mem(val_cilDatasetList, data, m, type_sampling)
        train_cilDatasetList.memory = model.memory
        model.n_known = len(model.memory)
        print('n_known_classes: ',model.n_known)
        
        batch_size = dict_conf['model']['batch_size']
        train_eval_loader = val_cilDatasetList.get_dataloader(data, batch_size, model.memory)
        
        total_train = 0.0
        correct_train = 0.0
        print('Init classification for training set')
        for _, _, videos, labels in train_eval_loader:
            videos = videos.to(device)
            preds = model.classify(videos, val_cilDatasetList)
            total_train += labels.size(0)
            correct_train += (preds.data.cpu() == labels).sum()
        acc = (100 * correct_train / total_train)
        experiment.log_metric("Train_Acc_task_{}".format(j+1), acc)
        print('Train Accuracy: %d %%' % acc)
        
        with experiment.validate():
            total_acc_val = model.final_validate(val_cilDatasetList, j, experiment)
            print('Val Accuracy: %d %%' % total_acc_val)
        
        with experiment.test():
            total_acc_test = model.final_validate(test_cilDatasetList, j, experiment)
            print('Test Accuracy: %d %%' % total_acc_test)
        
        if num_next_classes != None:
            model.augment_classification(num_next_classes, device)
            print('Classifier augmented')
            policies = model.get_optim_policies()
            conf_model = dict_conf['model']
            optimizer = torch.optim.SGD(policies,
                                    conf_model['lr'],
                                    momentum=conf_model['momentum'],
                                    weight_decay=conf_model['weight_decay'])

if __name__ == '__main__':
    main()
