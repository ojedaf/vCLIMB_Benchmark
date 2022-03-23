import torch.nn.functional as F
import torch

''' l_outputs is the the output of the final layer before applying the activation 
    To use the consistency loss without the adversarial loss, set adv_lambda=0
    To turn off the consistency loss, set cr_lambda=0
    A suggested value is cr_lambda = 1, but for this one l_outputs and l_outputs_aug has to be passed
    To turn off the adversarial loss, set adv_lambda=0
    We should probably ablate to see which one works best
    adv_lambda=0.5, cr_lambda=0
    adv_lambda=0.5, cr_lambda=1
    adv_lambda=0, cr_lambda=1
    
'''
# def get_robust_loss(criterion, videos, videos_aug, labels, model, adv_lambda=0.5, cr_lambda=0, l_outputs=None, l_outputs_aug=None):
#     outputs = model(videos)
#     outputs_aug = model(videos_aug)
#     loss = criterion(outputs, labels)
#     loss_aug = criterion(outputs_aug, labels)
#     loss_CR = 0
#     if cr_lambda > 0:
#         loss_CR = torch.mean((l_outputs - l_outputs_aug).pow(2))
#     loss_CR*=cr_lambda
#     return (adv_lambda * loss_aug) + ((1-adv_lambda) * loss) + loss_CR

def get_robust_loss(criterion, outputs, outputs_aug, labels, adv_lambda=0.5, cr_lambda=0, l_outputs=None, l_outputs_aug=None):
#     outputs = model(videos)
#     outputs_aug = model(videos_aug)
    loss = criterion(outputs, labels)
    loss_aug = criterion(outputs_aug, labels)
    print('loss: ',loss.item())
    print('loss_aug: ',loss_aug.item())
    loss_CR = 0
    if cr_lambda > 0:
        loss_CR = torch.mean((l_outputs - l_outputs_aug).pow(2))
    loss_CR*=cr_lambda
    return (adv_lambda * loss_aug) + ((1-adv_lambda) * loss) + loss_CR
