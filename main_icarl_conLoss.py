from comet_ml import Experiment
from model.iCaRL_conLoss import iCaRL
import torch
from model.temporalShiftModule.ops.transforms import *
from utils.icarl_dataset_frames_selfsup_v2 import CILSetTask
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
    
    global dict_conf, device, experiment, data, list_val_acc_ii, memory_size, type_sampling, is_activityNet, path_memory
    
    list_val_acc_ii = []
    parser = argparse.ArgumentParser(description="iCaRL TSN Baseline")
    parser.add_argument("-conf","--conf_path", default = './conf/conf_ucf101_icarl_tsn.yaml')
    args = parser.parse_args()
    conf_file = open(args.conf_path, 'r')
    print("Conf file dir: ",conf_file)
    
    # load the config file
    dict_conf = yaml.load(conf_file)

    conf_model = dict_conf['model']
    
    # Creat and set the comet exp
    api_key = dict_conf['comet']['api_key']
    workspace = dict_conf['comet']['workspace']
    project_name = dict_conf['comet']['project_name']
    experiment = Experiment(api_key=api_key,
                            project_name=project_name, workspace=workspace)
    experiment.log_parameters(parse_conf(dict_conf))
    experiment.set_name(dict_conf['comet']['name'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # loading the CIL scenario (sequence of tasks)
    path_data = dict_conf['dataset']['path_data']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
    
    num_class = len(data['train'][0].keys())
    
    # Set the sampling strategy for the memory.
    type_sampling = dict_conf['memory']['type_mem'] if 'type_mem' in dict_conf['memory'] else 'icarl'
    conf_model['type_sampling'] = type_sampling
    print('sampling strategy:', type_sampling)
    
    is_activityNet = dict_conf['dataset']['is_activityNet'] if 'is_activityNet' in dict_conf['dataset'] else False
    conf_model['is_activityNet'] = is_activityNet
    
    # Create the model
    model = iCaRL(conf_model, num_class, dict_conf['checkpoints'])
    
    # Create the data augmentation strategies for training. 
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    
    dataset_name = dict_conf['dataset']['name']
    
    train_augmentation = model.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True)
    
    # Create optimizer
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
    path_memory = dict_conf['memory']['path_memory']

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
    
    train_per_noise = dict_conf['dataset']['train_per_noise'] if 'train_per_noise' in dict_conf['dataset'] else 0
    val_per_noise = dict_conf['dataset']['val_per_noise'] if 'val_per_noise' in dict_conf['dataset'] else 0
    co_threshold = dict_conf['dataset']['co_threshold'] if 'co_threshold' in dict_conf['dataset'] else 0
    
    # Create the CILSetTask (for training). It is the class that creates the data loader for each task sequentially. It also adds the instances in the memory buffer. 
    train_cilDatasetList = CILSetTask(data['train'], path_frames, memory_size, batch_size, shuffle=True, 
                                      num_workers=num_workers, num_frame_to_save = conf_model['num_frame_to_save'], 
                                      is_activityNet = is_activityNet, per_noise = train_per_noise, co_threshold = co_threshold, 
                                      drop_last=True, pin_memory=True, num_segments=num_segments, new_length=data_length, 
                                      modality=modality,transform=train_transforms, dense_sample=False, train_enable = True)
    
    # Create the CILSetTask (for validation)
    val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, is_activityNet = is_activityNet, per_noise = val_per_noise, 
                                    co_threshold = co_threshold, pin_memory=True, num_frame_to_save = conf_model['num_frame_to_save'], 
                                    num_segments=num_segments, new_length=data_length, modality=modality, 
                                    transform=val_transforms, random_shift=False, dense_sample=False, train_enable = False)
    
    test_cilDatasetList = None
    if not is_activityNet:
        # Create the CILSetTask (for testing)
        test_cilDatasetList = CILSetTask(data['test'], path_frames, memory_size, batch_size, shuffle=False, 
                                        num_workers=num_workers, is_activityNet = is_activityNet, per_noise = val_per_noise,
                                        co_threshold = co_threshold, pin_memory=True, num_frame_to_save = conf_model['num_frame_to_save'], 
                                        num_segments=num_segments, new_length=data_length, modality=modality, 
                                        transform=val_transforms, random_shift=False, dense_sample=False, train_enable = False)

    # define loss function (criterion) and optimizer
    cls_loss = nn.CrossEntropyLoss().to(device)
    dist_loss = nn.BCEWithLogitsLoss().to(device)
    model.set_losses(cls_loss, dist_loss)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
    path_model = dict_conf['checkpoints']['path_model']
    
    if dict_conf['checkpoints']['train_mode']:
        # To train the model
        best_prec1 = 0
        current_task = 0
        current_epoch = 0
        path_best_model = path_model.format('Best_Model')
        if os.path.exists(path_best_model):
            # Load an existing checkpoint
            checkpoint_dict = torch.load(path_best_model)
            model.load_state_dict(checkpoint_dict['state_dict'])
            print("load parameters model - to train")
            best_prec1 = checkpoint_dict['accuracy']
            current_task = checkpoint_dict['current_task']
            current_epoch = checkpoint_dict['current_epoch'] + 1
            
        train_loop(model, optimizer, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList)
    else:
        # To val the model
        path_data_pred = dict_conf['checkpoints']['data_pred']
        path_best_model = path_model.format('Best_Model')
        if os.path.exists(path_best_model):
            checkpoint_dict = torch.load(path_best_model)
            new_num_classes = num_class*(len(data['train']) - 1)
            model.augment_classification(new_num_classes, device)
            model.feature_encoder.load_state_dict(checkpoint_dict['state_dict'])

            with open(path_memory, 'rb') as f:
                memory = pickle.load(f)
            
            model.memory = memory
            
            current_task = checkpoint_dict['current_task']
#             list_pred_elems_tasks = model.final_validation_analysis(val_cilDatasetList, current_task)
            model.gradCam_validation_analysis(val_cilDatasetList, current_task)
            with open(path_data_pred, 'wb') as handle:
                pickle.dump(list_pred_elems_tasks, handle)

def train_loop(model, optimizer, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList):
    iter_trainDataloader = iter(train_cilDatasetList)
    num_tasks = train_cilDatasetList.num_tasks
    eval_freq = dict_conf['checkpoints']['eval_freq']
    path_model = dict_conf['checkpoints']['path_model']
    num_epochs = dict_conf['model']['epochs']
    
    # Loop for the num of tasks
    for j in range(num_tasks):
        # Get the dataloader for the current task
        classes, data, train_loader_i, len_data, num_next_classes = next(iter_trainDataloader)
        # Train the model on the current task
        model.train(train_loader_i, len_data, optimizer, num_epochs, experiment, j, val_cilDatasetList)
        # Compute the number of instances per class that can be saved into the memory after learning the current task. (Mem size/num learned classes).
        if memory_size != 'ALL':
            if torch.cuda.device_count() > 1:
                m = memory_size // model.feature_encoder.module.new_fc.out_features
            else:
                m = memory_size // model.feature_encoder.new_fc.out_features
        else:
            m = 'ALL'
        
        # Add the new instances to the memory and fit previous instances per class to the new size.
        model.add_samples_to_mem(val_cilDatasetList, data, m)
        # Asign the memory to the set of CIL tasks (CILSetTask).
        train_cilDatasetList.memory = model.memory
        # Asign the num of learned classes.
        model.n_known = len(model.memory)
        print('n_known_classes: ',model.n_known)
        # Save the memory
        with open(path_memory, 'wb') as handle:
            pickle.dump(model.memory, handle)
        
        if type_sampling == 'icarl':
            # If the model is iCaRL, the final training accuracy is computed in the same way as in validation.
            batch_size = dict_conf['model']['batch_size']
            train_eval_loader = val_cilDatasetList.get_dataloader(data, batch_size, model.memory)

            total_train = 0.0
            correct_train = 0.0
            print('Init classification for training set')
            for _, _, videos, _, labels in train_eval_loader:
                videos = videos.to(device)
                preds = model.classify(videos, val_cilDatasetList)
                total_train += labels.size(0)
                correct_train += (preds.data.cpu() == labels).sum()
            acc = (100 * correct_train / total_train)
            experiment.log_metric("Train_Acc_task_{}".format(j+1), acc)
            print('Train Accuracy: %d %%' % acc)
        
        # Init validation.
        with experiment.validate():
            total_acc_val = model.final_validate(val_cilDatasetList, j, experiment)
            print('Val Accuracy: %d %%' % total_acc_val)
        
        # Init testing.
        if not is_activityNet:
            with experiment.test():
                total_acc_test = model.final_validate(test_cilDatasetList, j, experiment)
                print('Test Accuracy: %d %%' % total_acc_test)
        
        # if there are more classes to learn. 
        if num_next_classes != None:
            model.augment_classification(num_next_classes, device)        # Augment the classifier
            print('Classifier augmented')
            policies = model.get_optim_policies()                         # Get the new optimizing policies.
            conf_model = dict_conf['model']
            optimizer = torch.optim.SGD(policies,
                                    conf_model['lr'],
                                    momentum=conf_model['momentum'],
                                    weight_decay=conf_model['weight_decay'])

if __name__ == '__main__':
    main()
