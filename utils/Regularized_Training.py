import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader
#end of imports
#from  torch.optim import Optimizer, required
class Weight_Regularized_SGD(optim.SGD):
    r"""Implements stochastic gradient descent with importance weight regularizer
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,orth_reg=False,L1_decay=False):
        
        super(Weight_Regularized_SGD, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        self.orth_reg=orth_reg
        self.L1_decay=L1_decay
    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)
       
        
    def step(self, reg_params,closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
       

        loss = None
        if closure is not None:
            loss = closure()
        index=0
        reg_lambda=reg_params.get('lambda')
       
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
               

                
                if p in reg_params:
                    reg_param=reg_params.get(p)
                    omega=reg_param.get('omega')
                    init_val=reg_param.get('init_val')
                    curr_wegiht_val=p.data
                    #move the variables to cuda
                    init_val=init_val.cuda()
                    omega=omega.cuda()
                    
                    #get the difference
                    weight_dif=curr_wegiht_val.add(-1,init_val)
                    
                    if weight_dif.size(0) == omega.size(0):
                        regulizer=weight_dif.mul(2*reg_lambda*omega)
                    else:
                        omega_size = omega.size(0)
                        regulizer=weight_dif[:omega_size].mul(2*reg_lambda*omega)
                    del weight_dif
                    del curr_wegiht_val
                    del omega
                    del init_val

                    if d_p.size(0) == regulizer.size(0):
                        d_p.add_(regulizer)
                    else:
                        reg_size = regulizer.size(0)
                        d_p[:reg_size] = d_p[:reg_size] + regulizer
                    del regulizer

                if weight_decay != 0:
                    if self.L1_decay:
                        d_p.add_(weight_decay,p.data.sign())
                    else:
                        d_p.add_(weight_decay, p.data)
                #optionally you can use orthreg  
              
                if self.orth_reg:
                    d_p.add_(orth_org_hook(p))
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
               
                
                p.data.add_(-group['lr'], d_p)
                index+=1
        return loss
def set_lr(optimizer, lr,count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training=True
    if count>10:
        continue_training=False
        print("training terminated")
    if count==5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer,lr,continue_training

def traminate_protocol(since,best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def orth_org_hook(param,opt={}):
    if (len(param.size())==4):#conv2d
        opt['epsilon']=1e-10
        opt['orth_lambda']=10
        opt['beta']=0.001
      
        ##################
        
      
       
        #local m = modules[i]
        filters = param.data.clone().view(param.size(0),-1)
        norms = filters.norm(2,1).squeeze()
        #expand_as
        #norms = norms.view(-1,0).expand_as(filters)
        #
        norms = norms.view(-1,1).expand(filters.size())
        filters.div_(norms + opt['epsilon'])
        grad = torch.mm(filters, filters.transpose(1,0))
        grad = torch.exp(grad*opt['orth_lambda'])
        grad = (grad*opt['orth_lambda']).div(grad + math.exp(opt['orth_lambda'])) 
        indeces= torch.LongTensor( range(grad.size(0))).cuda()
        grad[indeces,indeces] = 0 
        grad = torch.mm(grad, filters)
        #weight = module.weight.view(module.weight.size(0), -1).clone()
        #x=module.weight.clone()
        coef =  opt['beta']
        

        grad= grad*coef
        grad=grad.view(param.size())
        return grad
    else: 
        return torch.zeros(param.size()).cuda()#


class MAS_OMEGA_ESTIMATE(optim.SGD):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(MAS_OMEGA_ESTIMATE, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        
    def __setstate__(self, state):
        super(MAS_OMEGA_ESTIMATE, self).__setstate__(state)
       

    def step(self, reg_params,batch_index,batch_size,closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        #print('************************DOING A STEP************************')
        #loss=super(Elastic_SGD, self).step(closure)
        loss = None
        if closure is not None:
            loss = closure()
        index=0
     
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
          
                #print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
                #param with zero learning rate will not be here
                if p in reg_params:
                    d_p = p.grad.data
                    unreg_dp = p.grad.data.clone()
                    #HERE MY CODE GOES
                    reg_param=reg_params.get(p)
                    
                    zero=torch.FloatTensor(p.data.size()).zero_()
                    omega=reg_param.get('omega')
                    omega=omega.cuda()
    
                    
                    #sum up the magnitude of the gradient
                    prev_size=batch_index*batch_size
                    curr_size=(batch_index+1)*batch_size
                    omega=omega.mul(prev_size)
                    
                    omega=omega.add(unreg_dp.abs_())
                    omega=omega.div(curr_size)
                    if omega.equal(zero.cuda()):
                        print('omega after zero')

                    reg_param['omega']=omega
                    #pdb.set_trace()
                    reg_params[p]=reg_param
                index+=1
        return loss

def train_model_sparce(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='',lam=0):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])

        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
            
                optimizer = lr_scheduler(optimizer, epoch,lr)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs=inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs,norm = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
        
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss=loss+lam*norm
                    loss.backward()
                    #print('step')
                    optimizer.step(model.reg_params)

                # statistics

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.data.item() / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del labels
                del inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model,best_acc

#importance_dictionary: contains all the information needed for computing the w and omega

def compute_importance_l2(model, optimizer, lr_scheduler,dset_loader,use_gpu):
    
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    
    
    
        
    epoch=1
    optimizer = lr_scheduler(optimizer, epoch,1)
    model.eval()  # Set model to training mode so we get the gradient


    running_loss = 0.0
    running_corrects = 0
   
    # Iterate over data.
    index=0
    for data in dset_loader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(),requires_grad=False), \
            Variable(labels.cuda(),requires_grad=False)
        else:
            inputs, labels = Variable(inputs,requires_grad=False), Variable(labels,requires_grad=False)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
    
        #compute the L2 norm of output 
        Target_zeros=torch.zeros(outputs.size())
        Target_zeros=Target_zeros.cuda()
        Target_zeros=Variable(Target_zeros,requires_grad=False)
    
        loss = torch.nn.MSELoss(size_average=False)

        targets = loss(outputs,Target_zeros) 
    
        targets.backward()

        optimizer.step(model.reg_params,index,labels.size(0))

        index+=1
   
    return model

def compute_importance(model, optimizer, lr_scheduler,dset_loader,use_gpu):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L1norm of the function output
    """
   
    since = time.time()

    best_model = model
    best_acc = 0.0
    
    #pdb.set_trace()
    

        
    epoch=1
    #it does nothing here, can be removed
    optimizer = lr_scheduler(optimizer, epoch,1)
    model.eval()  # Set model to training mode so we get the gradient


    running_loss = 0.0
    running_corrects = 0
   
    # Iterate over data.
    index=0
    for data in dset_loader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
            Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameters gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
    

        #compute the L1 norm of the function output
    
        Target_zeros=torch.zeros(outputs.size())
        Target_zeros=Target_zeros.cuda()
        Target_zeros=Variable(Target_zeros,requires_grad=False)
    
        loss = torch.nn.L1Loss(size_average=False)

        targets = loss(outputs,Target_zeros) 
        #compute gradients
        targets.backward()
    
        #update parameters importance
        optimizer.step(model.reg_params,index,labels.size(0))
        #nessecary index to keep the running average
        index+=1
   
    return model


def initialize_reg_params(model,freeze_layers=[]):
    
    reg_params={}
    policies = model.get_optim_policies()
    for group_param in policies:
        params = group_param['params']
        name = group_param['name']
        if not name in freeze_layers:
            for param in params: 
                print('initializing param',name)
                omega=torch.FloatTensor(param.size()).zero_()
                init_val=param.data.clone()
                reg_param={}
                reg_param['omega'] = omega
                #initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param]=reg_param
    return reg_params
   

#set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def initialize_store_reg_params(model,freeze_layers=[]):
    
    reg_params=model.reg_params
    policies = model.get_optim_policies()
    for group_param in policies:
        params = group_param['params']
        name = group_param['name']
        if not name in freeze_layers:
            print('name param: ',name)
            for param in params:
                if param in reg_params:
                    reg_param=reg_params.get(param)
                    print('storing previous omega',name)
                    prev_omega=reg_param.get('omega')
                    new_omega=torch.FloatTensor(param.size()).zero_()
                    init_val=param.data.clone()
                    reg_param['prev_omega']=prev_omega   
                    reg_param['omega'] = new_omega
                    
                    #initialize the initial value to that before starting training
                    reg_param['init_val'] = init_val
                    reg_params[param]=reg_param
        else:
            for param in params:
                if param in reg_params: 
                    reg_param=reg_params.get(param)
                    print('removing unused omega',name)
                    del reg_param['omega'] 
                    del reg_params[param] 
             
    
    
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            print('name param: ',name)
            if param in reg_params:
                reg_param=reg_params.get(param)
                print('storing previous omega',name)
                prev_omega=reg_param.get('omega')
                new_omega=torch.FloatTensor(param.size()).zero_()
                init_val=param.data.clone()
                reg_param['prev_omega']=prev_omega   
                reg_param['omega'] = new_omega
                
                #initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param]=reg_param
                
        else:
            if param in reg_params: 
                reg_param=reg_params.get(param)
                print('removing unused omega',name)
                del reg_param['omega'] 
                del reg_params[param]
    return reg_params
   
def train_model_sparce_early_stopping(model, criterion, optimizer, lr_scheduler,lr,lr_decay_epoch,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='',lam=0):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()
    val_beat_counts=0#number of time val accuracy not imporved
    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc=checkpoint['best_acc']
        lr=checkpoint['lr']
        print("lr is ",lr)
        val_beat_counts=checkpoint['val_beat_counts']
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer,lr,continue_training = set_lr(optimizer,lr,count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since,best_acc)
                    return model,best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs=inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs,norm = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
        
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss=loss+lam*norm
                    loss.backward()
                    #print('step')
                    optimizer.step(model.reg_params)

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' :
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del loss
                    del preds
                    best_acc = epoch_acc
                    #best_model = copy.deepcopy(model)
                    torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts=0
                else:
                    val_beat_counts+=1
                    print("val_beat_counts is",str(val_beat_counts))
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'lr':lr,
            'val_beat_counts':val_beat_counts,
            'model': model,
            'epoch_acc':epoch_acc,
            'best_acc':best_acc,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model,best_acc

#importance_dictionary: contains all the information needed for computing the w and omega

#set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def accumelate_reg_params(model,freeze_layers=[]):
    
    reg_params=model.reg_params
    policies = model.get_optim_policies()
    for group_param in policies:
        params = group_param['params']
        name = group_param['name']
        if not name in freeze_layers:
            print('name param: ',name)
            for param in params:
                if param in reg_params:
                    reg_param=reg_params.get(param)
                    print('restoring previous omega',name)
                    prev_omega=reg_param.get('prev_omega')
                    prev_omega=prev_omega.cuda()

                    new_omega=(reg_param.get('omega')).cuda()
                    prev_size = prev_omega.size(0)
                    new_size = new_omega.size(0)
                    if prev_size != new_size:
                        acc_omega = new_omega[:prev_size] + prev_omega
                    else:
                        acc_omega=torch.add(prev_omega,new_omega)

                    del reg_param['prev_omega']
                    reg_param['omega'] = acc_omega

                    reg_params[param]=reg_param
                    del acc_omega
                    del new_omega
                    del prev_omega
        else:
            for param in params:
                if param in reg_params: 
                    reg_param=reg_params.get(param)
                    print('removing unused omega',name)
                    del reg_param['omega'] 
                    del reg_params[param]             
    return reg_params
    
def subtract_reg_params(model,freeze_layers=[]):
  
    reg_params=model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param=reg_params.get(param)
                print('subtracting new omega',name)
                prev_omega=reg_param.get('prev_omega')
                prev_omega=prev_omega.cuda()
                
                new_omega=(reg_param.get('omega')).cuda()
                acc_omega=prev_omega.add(-1,new_omega)
                zer=torch.FloatTensor(param.size()).zero_()
                acc_omega=torch.max(acc_omega,zer.cuda())
                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega
                
                reg_params[param]=reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params: 
                reg_param=reg_params.get(param)
                print('removing unused omega',name)
                del reg_param['omega'] 
                del reg_params[param]    
    return reg_params

#set omega to zero but after storing its value in a temp omega in which later we can accumolate them both by averaging
def accumelate_avg_reg_params(model,freeze_layers=[],number_of_tasks=2):
    
    reg_params=model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param=reg_params.get(param)
                print('storing previous omega',name)
                prev_omega=(reg_param.get('prev_omega')).cuda()
                prev_omega=prev_omega*(number_of_tasks -1) #running average
                new_omega=(reg_param.get('omega')).cuda()
                acc_omega=(torch.add(prev_omega,new_omega)).div(number_of_tasks)
                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega
                reg_params[param]=reg_param
                del acc_omega
                del new_omega
                del prev_omega
                
    return reg_params
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)
