from comet_ml import Experiment
from model.iCaRL_bic import iCaRL_BIC
import torch
from model.temporalShiftModule.ops.transforms import *
from utils.bic_dataset import CILSetTask
import argparse
import yaml, pickle
import torch.nn as nn
import os

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

def main():
    
    global dict_conf, device, experiment, data, list_val_acc_ii, memory_size
    
    list_val_acc_ii = []
    parser = argparse.ArgumentParser(description="iCaRL BIC TSN Baseline")
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
    
    model = iCaRL_BIC(conf_model, num_class, dict_conf['checkpoints'])

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
    
    perc = conf_model['perc']
    train_cilDatasetList = CILSetTask(data['train'], perc, path_frames, memory_size, batch_size, shuffle=True, 
                                      num_workers=num_workers, drop_last=True, pin_memory=True, 
                                      num_segments=num_segments, new_length=data_length, modality=modality, 
                                      transform=train_transforms, dense_sample=False, train_enable = True)
    
    val_cilDatasetList = CILSetTask(data['val'], perc, path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True, 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, 
                                    train_enable = False)
    
    test_cilDatasetList = CILSetTask(data['test'], perc, path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True, 
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
        classes, data, train_train_loader_i, train_val_loader_i, len_train_data, len_val_data, num_next_classes = next(iter_trainDataloader)
        model.train(train_train_loader_i, train_val_loader_i, len_train_data, len_val_data, optimizer, num_epochs, experiment, j, val_cilDatasetList)
        m = memory_size // model.feature_encoder.new_fc.out_features
        model.add_samples_to_mem(val_cilDatasetList, data, m)
        train_cilDatasetList.memory = model.memory
        model.n_known = len(model.memory)
        print('n_known_classes: ',model.n_known)
        
        with experiment.validate():
            total_acc_val = model.final_validate(val_cilDatasetList, j, experiment)
            print('Val total Accuracy: %d %%' % total_acc_val)
        
        with experiment.test():
            total_acc_test = model.final_validate(test_cilDatasetList, j, experiment)
            print('Test total Accuracy: %d %%' % total_acc_test)
        
        if num_next_classes != None:
            model.augment_classification(num_next_classes, device)
            print('Classifier augmented')
            policies = model.get_optim_policies()
            conf_model = dict_conf['model']
            optimizer = torch.optim.SGD(policies,
                                    conf_model['lr'],
                                    momentum=conf_model['momentum'],
                                    weight_decay=conf_model['weight_decay'])

        # bias_optimizer = torch.optim.Adam(model.bias_layer.parameters(), lr=0.001)

if __name__ == '__main__':
    main()
