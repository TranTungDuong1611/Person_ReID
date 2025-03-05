import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_distance(outputs, target):
    """function to calculate the L2 distance between embedded vector

    Args:
        outputs: (batch_size, embedded_vector)
        target: (batch_size, embedded_vector)
    """
    
    dist = torch.norm(outputs - target, p=2)
    return dist

def calculate_cosine_similarity(vector1, vector2):
    return F.cosine_similarity(vector1, vector2, dim=0)

def search_by_l2_distance(query, gallery, gallery_labels):
    """ search query in a gallery

    Args:
        query (_type_)
        gallery (_type_)
        gallery_labels (_type_)

    Returns:
        best_pids (list): list of pids with the smallest distance with query
    """
    l2_dist_lst = []
    for feature_vec in gallery:
        l2_dist = calculate_distance(query, feature_vec)
        l2_dist_lst.append(l2_dist)
        
    _, best_idxs = torch.sort(torch.tensor(l2_dist_lst), descending=False)
    best_pids = gallery_labels[best_idxs]
    best_pids = best_pids.clone().detach()
    return best_pids

def search_by_cosine(query, gallery, gallery_labels):
    cosine_lst = []
    for feature_vec in gallery:
        cosine_sim = calculate_cosine_similarity(query, feature_vec)
        cosine_lst.append(cosine_sim)
        
    _, best_idxs = torch.sort(torch.tensor(cosine_lst), descending=True)
    best_pids = gallery_labels[best_idxs]
    return best_pids

def search_on_gallery_topk(queries, query_labels, gallery, gallery_labels, metric='euclidean', topk=[1, 5]):
    """ search the same person in gallery from queries using topk metrix

    Args:
        queries (torch.Tensor): (batch_size, embedded_vector_512D)
        query_labels (torch.Tensor): (batch_size, pids)
        gallery (torch.Tensor): (embedded_vectors_512D)
        gallery_labels (torch.Tensor): (pids)
        
    Returns:
        topk_acc (torch.Tensor): accuracy on topk search
    """
    
    batch_size = queries.size(0)
    batch_pred_pids = []
    
    if metric == 'euclidean':
        for query in queries:
            pred_pids = search_by_l2_distance(query, gallery, gallery_labels)
            batch_pred_pids.append(pred_pids.cpu())
    elif metric == 'cosine':
        for query in queries:
            pred_pids = search_by_cosine(query, gallery, gallery_labels)
            batch_pred_pids.append(pred_pids.cpu())
            
    topk_acc = []
    for k in topk:
        batch_pred_pids = np.array(batch_pred_pids)
        best_preds = torch.tensor(batch_pred_pids, device=device)
        correct = (best_preds[:, :k] == query_labels[:, None]).any(axis=1).sum()
        # print(f"top {k}: {best_preds[:, :k]}")
        
        acc = correct / batch_size
        print(f"correct: {correct} - total: {batch_size}")
        topk_acc.append(acc)
        
    return topk_acc

def search_on_gallery_map(queries, query_labels, gallery, gallery_labels, metric='euclidean', topk=[1, 5]):
    """ search the same person in gallery using map metrix

    Args:
        queries: (batch_size, feat_vec)
        query_labels: (batch_size, pids)
        gallery: (feat_vec)
        gallery_labels: (pids)
        metrics (str, optional). Defaults to 'euclidean'.
        topk (list, optional). Defaults to [1, 5].
    Return:
        map_acc (Tensor 1D): accuracy on map search
    """
    
    map_acc = []
    batch_preds = []
    if metric == 'euclidean':
        for query in queries:
            pred_pids = search_by_l2_distance(query, gallery, gallery_labels)
            batch_preds.append(pred_pids.cpu())
    elif metric == 'cosine':
        for query in queries:
            pred_pids = search_by_cosine(query, gallery, gallery_labels)
            batch_preds.append(pred_pids.cpu())
            
    for k in topk:
        batch_preds = np.array(batch_preds)
        best_preds = torch.tensor(batch_preds, device=device)
        true_pred_queries = (best_preds[:, :k] == query_labels[:, None])
        
        ap_lst = []
        for query in true_pred_queries:
            ap = 0.0
            total = 0
            correct = 0
            for i in range(len(query)):
                total += 1
                if query[i]:
                    correct += 1
                    ap += correct / total
            if correct:
                ap = ap / correct
            ap_lst.append(ap)
        
        map = np.sum(np.array(ap_lst)) / len(ap_lst)
        map_acc.append(map)
        
    map_acc = torch.from_numpy(np.array(map_acc))
    return map_acc
    

# query = torch.randint(low=1, high=5, size=(4, 4), dtype=torch.float32)
# print(f"query: {query}")
# query_labels = torch.tensor([1, 2, 2, 4])
# gallery = torch.randint(low=1, high=5, size=(6, 4), dtype=torch.float32)
# print(f"gallery: {gallery}")
# gallery_labels = torch.tensor([1, 3, 2, 4, 2, 3])

# topk_acc = search_on_gallery_topk(query, query_labels, gallery, gallery_labels)
# print(topk_acc)

# map_acc = search_on_gallery_map(query, query_labels, gallery, gallery_labels)
# print(map_acc)

# x = torch.randint(1, 3, (1, 3))
# y = torch.randint(1, 3, (1, 3))
# print(x)
# print(y)
# idx = torch.where(x == y)[0]
# print(idx.size())
# print(idx)