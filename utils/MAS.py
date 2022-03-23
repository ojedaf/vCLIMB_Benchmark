import torch.nn.functional as F
import torch

# return the regularized loss
def get_regularized_loss(criterion, outputs, labels, model, reg_lambda):
    loss = criterion(outputs, labels)
    mas_reg = model.reg_params
    if 'importance' in mas_reg and 'optpar' in mas_reg:
        importance_dict_list = mas_reg['importance']
        optpar_dict_list = mas_reg['optpar']

        for i in range(len(importance_dict_list)):
            for name, param in model.named_parameters():
                importance = importance_dict_list[i][name]
                optpar = optpar_dict_list[i][name]
                if optpar.size(0) == param.size(0):
                    loss += (importance * (optpar - param).pow(2)).sum() * reg_lambda
                else:
                    size_optpar = optpar.size(0)
                    loss += (importance * (optpar - param[:size_optpar]).pow(2)).sum() * reg_lambda
    return loss

def on_task_update(loader_task, device, optimizer, model):
    
    print('MAS HERE')
    model.train()
    optimizer.zero_grad()
    
    mas_reg = model.reg_params
    
    if 'importance' in mas_reg and 'optpar' in mas_reg:
        importance_dict_list = mas_reg['importance']
        optpar_dict_list = mas_reg['optpar']
    else:
        mas_reg['importance'] = []
        mas_reg['optpar'] = []

    for i, (_, _, videos, _, target) in enumerate(loader_task):
        videos, target = videos.to(device), target.to(device)
        output = model(videos)

        # uses the gradients of the squared L2-norm of the model output in an unsupervised fashion
        loss = torch.norm(output, p=2, dim=1).mean() # loss = torch.norm(torch.cat(output, dim=1), p=2, dim=1).mean() # loss = F.cross_entropy(output, target)

        loss.backward()
        
    importance_dict = {}
    optpar_dict = {}

    # gradients accumulated can be used to calculate importance
    for name, param in model.named_parameters():

        optpar_dict[name] = param.data.clone()
        importance_dict[name] = param.grad.data.clone().abs() # param.grad.data.clone().pow(2)
        
    mas_reg['importance'].append(importance_dict)
    mas_reg['optpar'].append(optpar_dict)
    
    return mas_reg

# def consolidate_reg_params(model):
#     """
#     Input:
#     1) model: A reference to the model that is being trained
#     Output:
#     1) reg_params: A dictionary containing importance weights (importance), init_val (keep a reference 
#     to the initial values of the parameters) for all trainable parameters
#     Function: This function updates the value (adds the value) of omega across the tasks that the model is 
#     exposed to
    
#     """
#     #Get the reg_params for the model 
#     reg_params = model.reg_params

#     for name, param in model.tmodel.named_parameters():
#         if param in reg_params:
#             param_dict = reg_params[param]
#             print ("Consolidating the omega values for layer", name)
            
#             #Store the previous values of omega
#             prev_omega = param_dict['prev_omega']
#             new_omega = param_dict['importance']

#             new_omega = torch.add(prev_omega, new_omega)
#             del param_dict['prev_omega']
            
#             param_dict['importance'] = new_omega

#             #the key for this dictionary is the name of the layer
#             reg_params[param] = param_dict

#     model.reg_params = reg_params

#     return model
