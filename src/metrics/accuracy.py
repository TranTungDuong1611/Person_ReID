import os
import sys
import torch
sys.path.append(os.getcwd())

def accuracy(preds, labels, topk=[1, 5]):
    """ computes accuracy of classification between the unique class over the top k predictions

    Args:
        preds (batch_size, num_classes): prediction matrix with shape.
        labels (batch_size): ground truth labels with shape.
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.
    Returns:
        list: accuracy at top-k.
    """
    
    batch_size = labels.size(0)
    
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
        
    topk_acc = []
    for k in topk:
        _, pred = torch.topk(preds, k, largest=True, sorted=True)
        print(f"pred: {pred}")
        correct_ans = (pred[:, :k] == labels[:, None]).any(axis=1).sum()
        
        acc = correct_ans / batch_size
        topk_acc.append(acc)
        
    return topk_acc

# preds = torch.randn(4, 3)
# print(preds)
# labels = torch.tensor([2, 1, 3, 2])
# topk_acc = accuracy(preds, labels, topk=[1, 2, 3])
# print(topk_acc)