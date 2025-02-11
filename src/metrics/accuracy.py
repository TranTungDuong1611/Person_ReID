import os
import sys
import torch
sys.path.append(os.getcwd())

def accuracy(output, target, topk=[1, 5]):
    """ computes accuracy over the k top predictions

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.
    Returns:
        list: accuracy at top-k.
    """
    
    batch_size = target.size(0)
    
    if isinstance(output, (tuple, list)):
        output = output[0]
        
    topk_acc = []
    for k in topk:
        _, pred = torch.topk(output, k, largest=True, sorted=True)
        print(f"pred: {pred}")
        correct_ans = (pred[:, :k] == target[:, None]).any(axis=1).sum()
        
        acc = correct_ans / batch_size
        topk_acc.append(acc)
        
    return topk_acc

output = torch.randn(4, 3)
print(output)
target = torch.tensor([2, 1, 3, 2])
topk_acc = accuracy(output, target, topk=[1, 2, 3])
print(topk_acc)