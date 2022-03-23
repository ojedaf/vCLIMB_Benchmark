import torch.nn.parallel
import torch
import torch.nn as nn
from .consistency_loss import get_robust_loss
from .temporalShiftModule.ops.models import TSN
from .temporalShiftModule.ops.utils import AverageMeter, accuracy
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import os
import time
import random
from torch.cuda.amp import autocast

class iCaRL(nn.Module):
    def __init__(self, conf_model, num_class, conf_checkpoint):
        super(iCaRL, self).__init__()
		
        self.conf_checkpoint = conf_checkpoint
        self.conf_model = conf_model
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
        self.fc_lr5 = conf_model['fc_lr5']
        temporal_pool = conf_model['temporal_pool']
        non_local = conf_model['non_local']
        self.feature_encoder = TSN(num_class, num_segments, modality, 
                                    base_model=arch,
                                    consensus_type=consensus_type,
                                    dropout=dropout,
                                    img_feature_dim=img_feature_dim,
                                    partial_bn=not no_partialbn,
                                    pretrain=pretrain,
                                    is_shift=shift, shift_div=shift_div, shift_place=shift_place,
                                    fc_lr5=self.fc_lr5,
                                    temporal_pool=temporal_pool,
                                    non_local=non_local)
								   
		# feature_dim = self.feature_encoder.new_fc.in_features
	    # self.new_fc = nn.Linear(feature_dim, num_class)
		
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.crop_size = self.feature_encoder.crop_size
        self.scale_size = self.feature_encoder.scale_size
        self.input_mean = self.feature_encoder.input_mean
        self.input_std = self.feature_encoder.input_std
        self.is_activityNet = conf_model['is_activityNet']

        if torch.cuda.device_count() > 1:
            self.feature_encoder = nn.DataParallel(self.feature_encoder)

        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        self.feature_encoder.to(self.device)

        self.n_classes = num_class
        self.n_known = 0

        self.compute_means = True
        self.exemplar_means = []
        self.memory = {}
        self.list_val_acc_ii = []
        self.adv_lambda = conf_model['adv_lambda']
        print('adv_lambda: ',self.adv_lambda)	

    def get_optim_policies(self):
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.get_optim_policies()
        else:
            return self.feature_encoder.get_optim_policies()

    def get_augmentation(self, flip=True):
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.get_augmentation(flip)
        else:
            return self.feature_encoder.get_augmentation(flip)

    def augment_classification(self, num_new_classes, device):
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.augment_classification(num_new_classes, device)
        else:
            return self.feature_encoder.augment_classification(num_new_classes, device)
        
        self.n_classes += num_new_classes
        
    def set_losses(self, cls_loss, dist_loss):
        self.cls_loss = cls_loss
        self.dist_loss = dist_loss
    
	# def augment_classification(self, num_new_classes):
        # # add in the policies
        # feature_dim = self.new_fc.in_features
        # out_class = self.new_fc.out_features
        # out_dim = out_class + num_new_classes
        # weight = self.new_fc.weight.data
        # bias = self.new_fc.bias.data
        # new_fc = nn.Linear(feature_dim, out_dim).to(self.device)
        # std = 0.001
        # if hasattr(new_fc, 'weight'):
            # normal_(new_fc.weight, 0, std)
            # constant_(new_fc.bias, 0)
        
        # new_fc.weight.data[:out_class] = weight[:out_class]
        # new_fc.bias.data[:out_class] = bias[:out_class]
        
        # self.new_fc = new_fc
    
    def forward(self, x):
        x = self.feature_encoder(x, get_emb = False)
        # x = self.new_fc(x)
        return x
        
    def add_samples_to_mem(self, cilsettask, data, m, type_sampling = 'icarl'): 
        # verificar que los elementos de videos vengan en el mismo orden de class_loader
        if type_sampling == 'icarl':
            for class_id, videos in data.items():
                data_class = {class_id:videos}
                class_loader = cilsettask.get_dataloader(data_class, sample_frame = True)
                features = []
                video_names = []
                for _, video_name, video, _, _ in class_loader:
                    video = video.to(self.device)
                    feature = self.feature_encoder(video, get_emb = True).data.cpu().numpy()
                    feature = feature / np.linalg.norm(feature)
                    features.append(feature[0])
                    video_names.append(video_name)

                features = np.array(features)
                class_mean = np.mean(features, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

                exemplar_set = []
                exemplar_features = [] # list of Variables of shape (feature_size,)
                list_selected_idx = []
                for k in range(m):
                    S = np.sum(exemplar_features, axis=0)
                    phi = features
                    mu = class_mean
                    mu_p = 1.0/(k+1) * (phi + S)
                    mu_p = mu_p / np.linalg.norm(mu_p)
#                     i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
                    dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                    if k <= len(dist) - 2:
                        list_idx = np.argpartition(dist, k)[:k+1]
                    elif k < len(dist):
                        fixed_k = len(dist) - 2
                        list_idx = np.argpartition(dist, fixed_k)[:fixed_k+2]
                    else:
                        break
                    
                    for idx in list_idx:
                        if idx not in list_selected_idx:
                            list_selected_idx.append(idx)
                            exemplar_set.append(video_names[idx][0])
                            exemplar_features.append(features[idx])
                            break
                    
                    """
                    print "Selected example", i
                    print "|exemplar_mean - class_mean|:",
                    print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
                    #features = np.delete(features, i, axis=0)
                    """
                if self.is_activityNet:
                    new_exemplar_set = []
                    for video_name in exemplar_set:
                        idx = video_name.split('_')[-1]
                        for vid in videos:
                            if vid['id'] == int(idx):
                                new_exemplar_set.append(vid)
                    exemplar_set = new_exemplar_set
                                
                self.memory[class_id] = exemplar_set
            
            self.memory = {class_id: videos[:m] for class_id, videos in self.memory.items()}
        else:
            self.memory = {**self.memory, **data}
            for class_id, videos in self.memory.items():
                random.shuffle(videos)
                self.memory[class_id] = videos[:m]
        
        for class_id, videos in self.memory.items():
            print('Memory... Class: {}, num videos: {}'.format(class_id, len(videos)))
                
        
    def classify(self, x, cilsettask):
       
        batch_size = x.size(0)

        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for class_id, videos in self.memory.items():
                data_class = {class_id:videos}
                class_loader = cilsettask.get_dataloader(data_class, sample_frame = True)
                features = []
                # Extract feature for each exemplar in P_y
                for _, _, video, _, _ in class_loader:
                    video = video.to(self.device)
                    feature = self.feature_encoder(video, get_emb = True).squeeze().data.cpu()
                    feature = feature / feature.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features, dim=0)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means, dim = 0) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        feature = self.feature_encoder(x, get_emb = True).cpu() # (batch_size, feature_size)
        for i in range(feature.size(0)): # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        if len(dists.size()) == 1:
            dists = dists.unsqueeze(0)
        _, preds = dists.min(1)

        return preds
    
    def load_best_checkpoint(self, path_model, current_task):
        path_best_model = path_model.format('Best_Model')
        if os.path.exists(path_best_model):
            checkpoint_dict = torch.load(path_best_model)
            task_to_load = checkpoint_dict['current_task']
            if task_to_load == current_task:
                self.feature_encoder.load_state_dict(checkpoint_dict['state_dict'])
    
    def save_checkpoint(self, dict_to_save, path_model, is_best):
        if is_best:
            print('Saving ... ')  
            best_model = path_model.format('Best_Model')
            torch.save(dict_to_save, best_model)
            print("Save Best Networks for task: {}, epoch: {}".format(dict_to_save['current_task'] + 1, 
                                                                 dict_to_save['current_epoch'] + 1), flush=True)
                                                                 
    def validate(self, val_cilDatasetList, current_task_id):
        top1 = AverageMeter()
        total_acc = AverageMeter()
        val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
        
        # switch to evaluate mode
        self.feature_encoder.eval()
        
        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for _, _, videos, _, target in val_loader:
                    target = target.to(self.device)
                    videos = videos.to(self.device)
                    # compute output
                    output = self.forward(videos)
                    
                    # measure accuracy and record loss
                    acc_val = accuracy(output.data, target, topk=(1,))[0]
                    
                    top1.update(acc_val.item(), videos.size(0))

                total_acc.update(top1.avg, num_classes)
                print('Train... task : {}, acc with classifier: {}'.format(n_task, top1.avg))
                top1.reset()
        output = ('Pre Testing Results: Pre_Acc {total_acc.avg:.3f}'
                  .format(total_acc=total_acc))
        print(output)
        return total_acc.avg
    
    def final_validation_analysis(self, val_cilDatasetList, current_task_id):
        top1 = AverageMeter()
        total_acc = AverageMeter()
        val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
        BWF = AverageMeter()

        # switch to evaluate mode
        self.feature_encoder.eval()
        list_pred_elems_tasks = []

        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                list_pred_elems = []
                for _, _, videos, _, target in val_loader:
                    # target = target.to(self.device)
                    videos = videos.to(self.device)
                    # compute output
                    preds = self.classify(videos, val_cilDatasetList)

                    # check the accuracy function
                    correct = (preds.data.cpu() == target).sum()
                    acc_val = (100 * correct / target.size(0))
                    # acc_val = accuracy(preds.data, target, topk=(1,))[0]

                    # top1.update(acc_val.item(), videos.size(0))
                    top1.update(acc_val, videos.size(0))
                    list_pred_elems.append({'targets': target,'preds':preds})

                list_pred_elems_tasks.append(list_pred_elems)
                total_acc.update(top1.avg, num_classes)
                top1.reset()

            output = ('Testing Results: Acc {total_acc.avg:.3f}'
                          .format(total_acc=total_acc))
            print(output)
        return list_pred_elems_tasks
    
        
    def final_validate(self, val_cilDatasetList, current_task_id, experiment):
        top1 = AverageMeter()
        total_acc = AverageMeter()
        val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
        BWF = AverageMeter()
        
        # switch to evaluate mode
        self.feature_encoder.eval()
        
        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for _, _, videos, _, target in val_loader:
                    # target = target.to(self.device)
                    videos = videos.to(self.device)
                    # compute output
                    preds = self.classify(videos, val_cilDatasetList)

                    # check the accuracy function
                    correct = (preds.data.cpu() == target).sum()
                    acc_val = (100 * correct / target.size(0))
                    # acc_val = accuracy(preds.data, target, topk=(1,))[0]

                    # top1.update(acc_val.item(), videos.size(0))
                    top1.update(acc_val, videos.size(0))

                experiment.log_metric("Acc_task_{}".format(n_task+1), top1.avg, step=current_task_id+1)
                if n_task == current_task_id:
                    self.list_val_acc_ii.append(top1.avg)
                elif n_task < current_task_id:
                    forgetting = self.list_val_acc_ii[n_task] - top1.avg
                    BWF.update(forgetting, num_classes)
                total_acc.update(top1.avg, num_classes)
                top1.reset()

            output = ('Testing Results: Acc {total_acc.avg:.3f}'
                          .format(total_acc=total_acc))
            print(output)
            experiment.log_metric("Total_Acc_Per_task", total_acc.avg, step=current_task_id+1)
            experiment.log_metric("Total_BWF_Per_task", BWF.avg, step=current_task_id+1)
        return total_acc.avg
    
    def set_partialBN(self):
        no_partialbn = self.conf_model['no_partialbn']

        if no_partialbn:
            if torch.cuda.device_count() > 1:
                self.feature_encoder.module.partialBN(False)
            else:
                self.feature_encoder.partialBN(False)
        else:
            if torch.cuda.device_count() > 1:
                self.feature_encoder.module.partialBN(True)
            else:
                self.feature_encoder.partialBN(True)

    
    def train(self, dataloader_cil, len_data, optimizer, num_epochs, experiment, task_id, val_cilDatasetList):

        self.compute_means = True
        eval_freq = self.conf_checkpoint['eval_freq']
        path_model = self.conf_checkpoint['path_model']
        best_acc_val = 0
        
        self.set_partialBN()

        with experiment.train():

            # Store network outputs with pre-update para 
            if torch.cuda.device_count() > 1:
                n_classes = self.feature_encoder.module.new_fc.out_features
            else:
                n_classes = self.feature_encoder.new_fc.out_features
            q = torch.zeros(len_data, n_classes).to(self.device)
            for indices, _, videos, _, labels in dataloader_cil:
                videos = videos.to(self.device)
                indices = indices.to(self.device)
                g = F.sigmoid(self.forward(videos))
                q[indices] = g.data
            q = Variable(q).to(self.device)

            for epoch in range(num_epochs):
                # switch to train mode
                start = time.time()
                self.set_partialBN()
                self.feature_encoder.train()

                acc_Avg = AverageMeter()
                loss_Avg = AverageMeter()
                for i, (indices, _, videos, videos_aug, labels) in enumerate(dataloader_cil):
                    videos = videos.to(self.device)
                    labels = labels.to(self.device)
                    indices = indices.to(self.device)
                    videos_aug = videos_aug.to(self.device)

                    optimizer.zero_grad()
                    with autocast():
                        g = self.forward(videos)
                        g_aug = self.forward(videos_aug)
                        
                        loss = get_robust_loss(self.cls_loss, g, g_aug, labels, adv_lambda=self.adv_lambda, 
                                               cr_lambda=0, l_outputs=None, l_outputs_aug=None)

		    	# Classification loss for new classes
#                         loss = self.cls_loss(g, labels)
                        acc_train = accuracy(g.data, labels, topk=(1,))[0]

		    	# Distilation loss for old classes
                        if self.n_known > 0:
                            # g = F.sigmoid(g)
                            q_i = q[indices]
                            dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y])\
                                    for y in range(self.n_known))
                            loss += dist_loss

                    loss.backward()
                    
                    clip_gradient = self.conf_model['clip_gradient']
                    if clip_gradient is not None:
                        total_norm = clip_grad_norm_(self.feature_encoder.parameters(), clip_gradient)
                        
                    optimizer.step()
                    
                    experiment.log_metric("Acc_task_{}".format(task_id+1), acc_train.item())
                    experiment.log_metric("Loss_task_{}".format(task_id+1), loss.item())
                    loss_Avg.update(loss.item(), videos.size(0))
                    acc_Avg.update(acc_train.item(), videos.size(0))
                    
                    if (i+1) % 10 == 0:
                        print('Epoch [%d/%d], Loss: %.4f' 
                               %(epoch+1, num_epochs, loss.item()))
                
                experiment.log_metric("Epoch_Acc_task_{}".format(task_id+1), acc_Avg.avg)
                experiment.log_metric("Epoch_Loss_task_{}".format(task_id+1), loss_Avg.avg)
                end = time.time()
                print('elapsed time: ',end-start)

                if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
                    acc_val = self.validate(val_cilDatasetList, task_id)
                    is_best = acc_val >= best_acc_val
                    best_acc_val = max(acc_val, best_acc_val)
                    output_best = 'Best Pre Acc Val@1: %.3f\n' % (best_acc_val)
                    print(output_best)
                    dict_to_save = {'state_dict': self.feature_encoder.state_dict(), 'accuracy': acc_val, 'current_epoch': epoch, 
                                    'current_task': task_id, 'optimizer': optimizer.state_dict()}
                    
                    self.save_checkpoint(dict_to_save, path_model, is_best)
        
        self.load_best_checkpoint(path_model, task_id)
