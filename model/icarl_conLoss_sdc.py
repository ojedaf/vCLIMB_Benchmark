import torch.nn.parallel
import torch
import torch.nn as nn
from .consistency_loss import get_robust_loss,msloss
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
import numpy as np
import pickle
from pytorch_metric_learning import losses

class iCaRL(nn.Module):
    def __init__(self, conf_model, num_class, conf_checkpoint):
        super(iCaRL, self).__init__()

        # Model Checkpoint
        self.conf_checkpoint = conf_checkpoint
        # Model Config
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
        # Model backbone (TSN)
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
        self.class_mean_arr = []
        self.exemplar_means = []
        self.memory = {}
        self.feature_mem ={}
        self.class_emb_means = {}
        self.list_val_acc_ii = {'val': [], 'test': []}
        self.adv_lambda = conf_model['adv_lambda']
        print('adv_lambda: ',self.adv_lambda)
        self.type_sampling = conf_model['type_sampling']
        print('model sampling strategy:', self.type_sampling)

    # To prepare the model for a novel task, it is essential to augment the classifier according to the novel classes. 
    # Thus, it is crucial to update the optimization policies.
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
    
    # Function to augment the classification layer
    # num_new_classes - Num of new classes.
    # device - Pytorch device (GPU).
    def augment_classification(self, num_new_classes, device):
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.augment_classification(num_new_classes, device)
        else:
            return self.feature_encoder.augment_classification(num_new_classes, device)
        
        self.n_classes += num_new_classes
    
    # Set the losses
    # cls_loss - Classification Loss
    # dist_loss - Distillation Loss
    def set_losses(self, cls_loss, dist_loss):
        self.cls_loss = cls_loss
        self.dist_loss = dist_loss
    
    # Function to do a forward pass through the model, including the linear layer.
    def forward(self, x):
        x = self.feature_encoder(x, get_emb = False)
        return x
    
    # Function to manage the memory.
    # Data - It is candidate data to save into memory.
    # m - The number of elements per class in the memory.
    def add_samples_final(self,cilsettask,data,m):
        if self.type_sampling == 'icarl':
                self.class_mean_arr = []
                for class_id, videos in data.items():
                    data_class = {class_id:videos}
                    class_loader = cilsettask.get_dataloader(data_class, sample_frame = True)
                    features = []
                    video_names = []
                    video_names_ent = []
                    video_entropy = []
                    entropies  = []

                    for _, video_name, video, _, _ in class_loader:
                        video = video.to(self.device)
                        feature = self.feature_encoder(video, get_emb = True).data.cpu().numpy()
                        feature = feature / np.linalg.norm(feature)
                        features.append(feature[0])
                        video_names.append(video_name)

                        prob_s = self.feature_encoder(video) #logits given by encoder
                        prob_s = torch.nn.Softmax()(prob_s).data.cpu().numpy()
                        feature_entropy = - prob_s * np.log(prob_s)         #/np.log(self.feature_encoder.new_fc.out_features)
                        video_entropy.append(np.sum(feature_entropy[0]))
                        video_names_ent.append(video_name)

                    features = np.array(features)
                    class_mean = np.mean(features, axis=0)
                    class_mean = class_mean / np.linalg.norm(class_mean) # Normalize
                    self.class_mean_arr.append(class_mean)   #store means of new classes in a list so as to use in classify method for Nearest mean classifier
                    

                    exemplar_set = []
                    exemplar_features = [] # list of Variables of shape (feature_size,)
                    list_selected_idx = []
                    sort_index = np.argsort(np.array(video_entropy))
                    exemplar_set_ent = []

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

                    if len(sort_index)<=m:
                        for idx in sort_index:
                        #list_selected_idx.append(idx)
                            exemplar_set_ent.append(video_names_ent[idx][0])
                        #exemplar_features.append(features[idx])

                    else:
                        high_entropy   = sort_index[len(sort_index)-m:]  #choose videos with highest entropies
                        #mid_entropy  = sort_index[(len(sort_index)-m)//2:len(sort_index)-((len(sort_index)-m)//2)]
                        for idx in high_entropy:
                            exemplar_set_ent.append(video_names_ent[idx][0])

                    self.memory[class_id] = exemplar_set_ent
                    #saving class embeddings 
                    self.class_emb_means[class_id] = class_mean
                    #self.iCarl_mem[class_id] = video_entropy,list_selected_idx
                    self.feature_mem[class_id] = exemplar_features[:20],exemplar_set_ent[:20],class_mean

        self.memory = {class_id: videos[:m] for class_id, videos in self.memory.items()}
        self.class_emb_means = {class_id: means for class_id, means in self.class_emb_means.items()}
        #self.iCarl_mem = {class_id: videos[:m] for class_id, videos in self.iCarl_mem.items()}
        return  self.feature_mem

#     def add_samples_to_mem(self, cilsettask, data, m): 
#         # Memory sampling strategy of iCaRL.
#         if self.type_sampling == 'icarl':
#             for class_id, videos in data.items():
#                 data_class = {class_id:videos}
#                 class_loader = cilsettask.get_dataloader(data_class, sample_frame = True)
#                 features = []
#                 video_names = []
#                 for _, video_name, video, _, _ in class_loader:
#                     video = video.to(self.device)
#                     feature = self.feature_encoder(video, get_emb = True).data.cpu().numpy()
#                     feature = feature / np.linalg.norm(feature)
#                     features.append(feature[0])
#                     video_names.append(video_name)

#                 features = np.array(features)
#                 class_mean = np.mean(features, axis=0)
#                 class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

#                 exemplar_set = []
#                 exemplar_features = [] # list of Variables of shape (feature_size,)
#                 list_selected_idx = []
#                 for k in range(m):
#                     S = np.sum(exemplar_features, axis=0)
#                     phi = features
#                     mu = class_mean
#                     mu_p = 1.0/(k+1) * (phi + S)
#                     mu_p = mu_p / np.linalg.norm(mu_p)
# #                     i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
#                     dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
#                     if k <= len(dist) - 2:
#                         list_idx = np.argpartition(dist, k)[:k+1]
#                     elif k < len(dist):
#                         fixed_k = len(dist) - 2
#                         list_idx = np.argpartition(dist, fixed_k)[:fixed_k+2]
#                     else:
#                         break
                    
#                     for idx in list_idx:
#                         if idx not in list_selected_idx:
#                             list_selected_idx.append(idx)
#                             exemplar_set.append(video_names[idx][0])
#                             exemplar_features.append(features[idx])
#                             break
                    
#                     """
#                     print "Selected example", i
#                     print "|exemplar_mean - class_mean|:",
#                     print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
#                     #features = np.delete(features, i, axis=0)
#                     """
#                 if self.is_activityNet:
#                     new_exemplar_set = []
#                     for video_name in exemplar_set:
#                         idx = video_name.split('_')[-1]
#                         for vid in videos:
#                             if vid['id'] == int(idx):
#                                 new_exemplar_set.append(vid)
#                     exemplar_set = new_exemplar_set
                                
#                 self.memory[class_id] = exemplar_set
            
#             self.memory = {class_id: videos[:m] for class_id, videos in self.memory.items()}
#         else:
#             # Random Memory Sampling
#             self.memory = {**self.memory, **data}
#             for class_id, videos in self.memory.items():
#                 random.shuffle(videos)
#                 if m != 'ALL':
#                     self.memory[class_id] = videos[:m]
#                 else:
#                     self.memory[class_id] = videos
                    
#         for class_id, videos in self.memory.items():
#             print('Memory... Class: {}, num videos: {}'.format(class_id, len(videos)))
                
    # This function is to classify the instances following the iCaRL pipeline.
    # x - Batch to classify
    # cilsettask - the class that handles the validation data loaders.

    def get_emb(self, cilsettask):
        #class_emb = {}
        n_c = len(self.memory.keys()) #number of classes in memory

        c_e=1000
        for i in range(len(self.memory.keys())):
            if c_e > len(self.memory[list(self.memory.keys())[i]]):     # number of examples in memory of each class(since it is variable we                                                         # keep its min as value)
                c_e = len(self.memory[list(self.memory.keys())[i]]) #len(self.memory[list(self.memory.keys())[0]]) #number of examples in memory of each class
        
        feat_size = 512  #each video's encoded feature dimensions
        class_emb_mat = np.zeros((n_c, c_e , feat_size)) 

        for class_id, videos in self.memory.items():
            k = list(self.memory.keys()).index(class_id)
            data_class = {class_id:videos}
            class_loader = cilsettask.get_dataloader(data_class, sample_frame = True)
            features = []
            counter = 0
            # Extract feature for each exemplar in P_y
            for _, _, video, _, _ in class_loader:
                video = video.to(self.device)
                feature = self.feature_encoder(video, get_emb = True).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature) # Normalize
                features.append(feature)
                counter+=1
                if counter==c_e:
                    #class_emb[class_id] = np.array(features)
                    class_emb_mat[list(self.memory.keys()).index(class_id)] =np.array(features).reshape(np.array(features).shape[0],np.array(features).shape[2]) # classes x examples x features

        return class_emb_mat
        

    
 
    def sem_drift(self,old,new):
        sig = 0.2
        W = np.zeros(new.shape)
        rec_drift = new - old
        emb_means_mat = np.zeros((len(self.class_emb_means.keys()),new.shape[2]))  #classes x feaeture_size
        delta = np.zeros((new.shape[0],new.shape[2]))      #should be of dimensions 11(classes)*512(feature_size) initially
        means_arr = []   #should be of dimensions 11*512

        for class_id,means in self.class_emb_means.items():
            emb_means_mat[list(self.memory.keys()).index(class_id)] = means

        for i in range(W.shape[0]):
            W[i] = np.exp(-(old[i] - emb_means_mat[i])/(2* (sig**2)))
            delta[i] = (np.sum(W[i] * rec_drift[i],axis=0))/(np.sum(W[i],axis=0))
            self.class_emb_means[list(self.class_emb_means.keys())[i]] = self.class_emb_means[list(self.class_emb_means.keys())[i]] + delta[i]
            means_arr.append(torch.from_numpy(self.class_emb_means[list(self.class_emb_means.keys())[i]]))

        return means_arr

        
    def classify(self, x, mean_arr):
       
        batch_size = x.size(0)

        if type(self.class_mean_arr[0])!=torch.Tensor:
            for i in range(len(self.class_mean_arr)):
                self.class_mean_arr[i] = torch.from_numpy(self.class_mean_arr[i])
        else:
            self.class_mean_arr = self.class_mean_arr

        if  mean_arr==0:
            self.exemplar_means = self.class_mean_arr
        else:
            self.exemplar_means = mean_arr + self.class_mean_arr
            
        exemplar_means = self.exemplar_means 
        means = torch.stack(exemplar_means, dim = 0) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) #(batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        feature = self.feature_encoder(x, get_emb = True).cpu() # (batch_size, feature_size)
        for i in range(feature.size(0)): 
            feature.data[i] = feature.data[i] / feature.data[i].norm()    # Normalize
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        if len(dists.size()) == 1:
            dists = dists.unsqueeze(0)
        _, preds = dists.min(1)

        return preds
    
    # Function to load the best checkpoint of the model.
    def load_best_checkpoint(self, path_model, current_task):
        path_best_model = path_model.format('Best_Model')
        if os.path.exists(path_best_model):
            checkpoint_dict = torch.load(path_best_model)
            task_to_load = checkpoint_dict['current_task']
            if task_to_load == current_task:
                self.feature_encoder.load_state_dict(checkpoint_dict['state_dict'])
    
    # Function to save the checkpoint of the model.
    # dict_to_save - A dictionary with essential model configuration for restoring the model.
    # path_model - The memory path to save the checkpoint.
    # is_best - True if with this configuration, the model achieves the best accuracy until the moment.
    def save_checkpoint(self, dict_to_save, path_model, is_best):
        if is_best:
            print('Saving ... ')  
            best_model = path_model.format('Best_Model')
            torch.save(dict_to_save, best_model)
            print("Save Best Networks for task: {}, epoch: {}".format(dict_to_save['current_task'] + 1, 
                                                                 dict_to_save['current_epoch'] + 1), flush=True)
    
    # Function to validate the model while it is trained.
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
    
    # Function to analyse the output of the model.
    def final_validation_analysis(self, val_cilDatasetList, current_task_id,mean_arr):
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
                    videos = videos.to(self.device)
                    # compute output
                    if self.type_sampling == 'icarl':
                        preds = self.classify(videos, mean_arr)
                        correct = (preds.data.cpu() == target).sum()
                        acc_val = (100 * correct / target.size(0))
                    else:
                        target = target.to(self.device)
                        preds = self.forward(videos)
                        acc_val = accuracy(preds.data, target, topk=(1,))[0]
                        acc_val = acc_val.item()
                        
                    top1.update(acc_val, videos.size(0))
                    list_pred_elems.append({'targets': target,'preds':preds})

                list_pred_elems_tasks.append(list_pred_elems)
                total_acc.update(top1.avg, num_classes)
                top1.reset()

            output = ('Testing Results: Acc {total_acc.avg:.3f}'
                          .format(total_acc=total_acc))
            print(output)
        return list_pred_elems_tasks
    
    # Function to evaluate the model at the end of the training process of each task using the metrics (Acc and BWF).
    # val_cilDatasetList - Class that handles the validation data loaders.
    # current_task_id - Id of the last learned task.
    # experiment - commet object.
    def final_validate(self, val_cilDatasetList, current_task_id, experiment ,mean_arr ,type_val = 'val'):
        top1 = AverageMeter()
        total_acc = AverageMeter()
        val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
        BWF = AverageMeter()
        
        # switch to evaluate mode
        self.feature_encoder.eval()
        
        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for _, _, videos, _, target in val_loader:
                    videos = videos.to(self.device)
                    if self.type_sampling == 'icarl':
                        preds = self.classify(videos, mean_arr)
                        correct = (preds.data.cpu() == target).sum()
                        acc_val = (100 * correct / target.size(0))
                    else:
                        target = target.to(self.device)
                        preds = self.forward(videos)
                        acc_val = accuracy(preds.data, target, topk=(1,))[0]
                        acc_val = acc_val.item()

                    top1.update(acc_val, videos.size(0))

                experiment.log_metric("Acc_task_{}".format(n_task+1), top1.avg, step=current_task_id+1)
                if n_task == current_task_id:
                    self.list_val_acc_ii[type_val].append(top1.avg)
                elif n_task < current_task_id:
                    forgetting = self.list_val_acc_ii[type_val][n_task] - top1.avg
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
    

     
            
           
                
    
    # Training function
    # dataloader_cil - data loader of a task
    # len_data - Number of training instances
    # optimizer
    # num_epochs - Number of epochs
    # experiment - It is a Commet object essential to save relevant metrics. 
    # task_id - task id
    # val_cilDatasetList - List of validation data loaders.
    def train(self, dataloader_cil, len_data, optimizer, num_epochs, experiment, task_id, val_cilDatasetList):

        self.compute_means = True
        eval_freq = self.conf_checkpoint['eval_freq']
        path_model = self.conf_checkpoint['path_model']
        best_acc_val = 0
        
        self.set_partialBN()

        with experiment.train():

            # Store network outputs before training for previous knowledge distillation.
            if torch.cuda.device_count() > 1:
                n_classes = self.feature_encoder.module.new_fc.out_features
            else:
                n_classes = self.feature_encoder.new_fc.out_features
                
            if self.type_sampling == 'icarl':
                q = torch.zeros(len_data, n_classes).to(self.device)
                for indices, video_names, videos, _, labels in dataloader_cil:
                    videos = videos.to(self.device)
                    indices = indices.to(self.device)
                    g = F.sigmoid(self.forward(videos))
                    q[indices] = g.data
                q = Variable(q).to(self.device)

            loss_func =  losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

             # Start training loop
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
                        # forward pass with the original video clips
                        if self.n_known > 0:
                            g = self.forward(videos)
                        g_emb = self.feature_encoder(videos,get_emb =True)
                        # forward pass with its temporally downsampled versions
                       # g_aug = self.forward(videos_aug)
                        g_aug_emb = self.feature_encoder(videos_aug,get_emb=True)
                        # Temporal Consistency Loss
                       # loss = get_robust_loss(self.cls_loss, g, g_aug, labels, adv_lambda=self.adv_lambda, cr_lambda=0, l_outputs=None, l_outputs_aug=None)
                        # Multi-similarity loss 
                        loss =  loss_func(g_emb,labels)  + loss_func(g_aug_emb,labels)
                        # Classification Accuracy
                        acc_train = accuracy(g.data, labels, topk=(1,))[0]
                        # Distilation loss for old classes
                        if self.n_known > 0 and self.type_sampling == 'icarl':
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
                
                # Register on Commet the final metrics for the epoch
                experiment.log_metric("Epoch_Acc_task_{}".format(task_id+1), acc_Avg.avg)
                experiment.log_metric("Epoch_Loss_task_{}".format(task_id+1), loss_Avg.avg)
                end = time.time()
                print('elapsed time: ',end-start)

                # Evaluation loop
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
