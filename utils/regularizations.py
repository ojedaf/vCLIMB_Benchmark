# class MAS:
#     """
#     @article{aljundi2017memory,
#       title={Memory Aware Synapses: Learning what (not) to forget},
#       author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
#       booktitle={ECCV},
#       year={2018},
#       url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
#     }
#     """

#     def __init__(self, device, criterion_fn, reg_coef):
#         super(MAS, self).__init__()
#         self.online_reg = True
#         self.regularization_terms = {}
#         self.device = device
#         self.criterion_fn = criterion_fn
#         self.reg_coef = reg_coef
#         self.init_class = 0
#         self.final_class = 0
    
#     def criterion(self, model, preds, targets, regularization=True):
#         loss = self.criterion_fn(preds, targets)

#         if regularization and len(self.regularization_terms)>0:
#             # Calculate the reg_loss only when the regularization_terms exists
#             reg_loss = 0
#             importance = self.regularization_terms['importance']
#             task_param = self.regularization_terms['task_param']
#             for n, p in model.named_parameters():
#                 if p.requires_grad:
#                     if p.size(0) == task_param[n].size(0):
#                         reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
#                     else:
#                         size_init = task_param[n].size(0)
#                         reg_loss += (importance[n] * (p[:size_init] - task_param[n]) ** 2).sum()
                        
#             loss += self.reg_coef * reg_loss
#         return loss

#     def calculate_importance(self, model, dataloader, out_dim):
#         print('Computing MAS')
        
#         self.init_class=self.final_class
#         self.final_class=out_dim
#         params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        
# #          # Initialize the importance matrix
# #         if self.online_reg and len(self.regularization_terms)>0:
# #             importance = self.regularization_terms['importance']
# #         else:
# #             importance = {}
# #             for n, p in params.items():
# #                 importance[n] = p.clone().detach().fill_(0)  # zero initialized
                
#         # Initialize the importance matrix
#         if self.online_reg and len(self.regularization_terms)>0:
#             importance_old = self.regularization_terms['importance']
#             importance = {}
#             for n, p in params.items():
#                 if p.size(0) == importance_old[n].size(0):
#                     importance[n] = importance_old[n]
#                 else:
#                     out_dim_init = importance_old[n].size(0)
#                     importance[n] = p.clone().detach().fill_(0)
#                     importance[n][:out_dim_init] += importance_old[n]            
#         else:
#             importance = {}
#             for n, p in params.items():
#                 importance[n] = p.clone().detach().fill_(0)  # zero initialized

#         model.eval()

#         # Accumulate the gradients of L2 loss on the outputs
#         for i, (videos, target) in enumerate(dataloader):
            
#             videos = videos.to(self.device)
#             target = target.to(self.device)

#             preds = model(videos)

#             # The flag self.valid_out_dim is for handling the case of incremental class learning.
#             # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
#             # in calculating the  loss.
#             print('init_class: ',self.init_class)
#             print('final_class: ',self.final_class)
#             print('preds: ',preds.size())
#             pred = preds[:,self.init_class:self.final_class]
#             print('pred: ',pred)
#             print('preds_v2: ',pred.size())

#             pred.pow_(2)
#             loss = pred.mean()

#             model.zero_grad()
#             loss.backward()
#             for n, p in importance.items():
#                 if params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
# #                     if 'new_fc' in n:
# #                         a_param = params[n].grad.abs() / len(dataloader)
# #                         print('size: ',a_param.size())
# #                         print('impor: ',a_param[self.init_class:self.final_class])
# #                         print('not impor: ',a_param[self.final_class:])
#                     p += (params[n].grad.abs() / len(dataloader))
                    

#         model.train()

#         return importance

import torch
import torch.nn as nn
from utils.Regularized_Training import *

def sanitycheck(model):
    for name, param in model.named_parameters():
        #w=torch.FloatTensor(param.size()).zero_()
        print(name)
        if param in model.reg_params:
        
            reg_param=model.reg_params.get(param)
            omega=reg_param.get('omega')
            
            print('omega max is',omega.max().item())
            print('omega min is',omega.min().item())
            print('omega mean is',omega.mean().item())

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=54):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
   
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def accumulate_objective_based_weights_sparce(last_task_dataloader,model_ft,norm='', init_task=0):

    use_gpu = torch.cuda.is_available()
    if init_task == 0:
        # OK
        reg_params=initialize_reg_params(model_ft)
    else:
        reg_params=initialize_store_reg_params(model_ft)
    
    model_ft.reg_params=reg_params

    optimizer_ft = MAS_OMEGA_ESTIMATE(model_ft.get_optim_policies(), lr=0.0001, momentum=0.9)

    if norm=='L2':
        print('********************objective with L2 norm***************')
        model_ft = compute_importance_l2(model_ft, optimizer_ft, exp_lr_scheduler, last_task_dataloader,use_gpu)
    else:
        model_ft = compute_importance(model_ft, optimizer_ft, exp_lr_scheduler, last_task_dataloader,use_gpu)
    
    if init_task > 0:
        reg_params=accumelate_reg_params(model_ft)
        model_ft.reg_params=reg_params

    sanitycheck(model_ft)   
    return model_ft

