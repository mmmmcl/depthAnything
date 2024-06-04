import torch

import torch.nn.functional as F
def InvariantLoss(output, target):
    if output.shape != target.shape:
        raise ValueError('The size of output and target must be the same.')
    
   
    out_median = torch.median(output)
    target_median = torch.median(target)
    
    
    out_abs_deviation = torch.abs(output - out_median)
    target_abs_deviation = torch.abs(target - target_median)
    
    out_normalized = (output - out_median) / torch.mean(out_abs_deviation)
    target_normalized = (target - target_median) / torch.mean(target_abs_deviation)
    
    
    loss = torch.mean(torch.abs(out_normalized - target_normalized))
    
    return loss

def SemanticConstraint(features1, features2, alpha):
    assert len(features1) == len(features2), "Features lists must have the same length"
    
    
    total_loss = 0.0
    for feat1, feat2 in zip(features1, features2):
        patch_tokens1, _ = feat1
        patch_tokens2, _ = feat2
        
        cos_sim = F.cosine_similarity(patch_tokens1, patch_tokens2, dim=-1)
        layer_loss = torch.where(cos_sim > alpha, torch.tensor(0.0, device=cos_sim.device), 1 - cos_sim).mean()
        total_loss += layer_loss

    average_loss = total_loss / len(features1)

    return average_loss


# features1 = [
#     (torch.randn(1, 196, 384), None),
#     (torch.randn(1, 196, 384), None),
#     (torch.randn(1, 196, 384), None),
#     (torch.randn(1, 196, 384), None)
# ]
# features2 = [
#     (torch.randn(1, 196, 384), None),
#     (torch.randn(1, 196, 384), None),
#     (torch.randn(1, 196, 384), None),
#     (torch.randn(1, 196, 384), None)
# ]
# alpha = 0.85


# average_loss = SemanticConstraint(features1, features2, alpha)
