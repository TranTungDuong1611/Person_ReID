import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

sys.path.append(os.getcwd())

from src.data.dataloader import *
from src.metrics.accuracy import *
from src.metrics.distance import *
from src.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_search(model, query_path, gallery_loader, distance='euclidean', topk=5):
    model = model.to(device).eval()
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128))
    ])
    query = cv2.imread(query_path)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    
    # convert to tensor
    query = trans(query).unsqueeze(0).to(device)
    # embedded query
    embedded_query = model(query)
    
    # Embedded gallery to get the feature vectors
    gallery_pool = torch.tensor([], device=device)
    gallery_label_pool = torch.tensor([], device=device)
    image_path_pool = []
    for gallery_batch, gallery_label_batch, image_path_batch in gallery_loader:
        gallery_batch = gallery_batch.to(device)
        gallery_label_batch = gallery_label_batch.to(device)
        image_path_pool.extend(image_path_batch)
        
        with torch.no_grad():
            embedded_gallery = model(gallery_batch)
            gallery_pool = torch.cat((gallery_pool, embedded_gallery), dim=0)
            gallery_label_pool = torch.cat((gallery_label_pool, gallery_label_batch), dim=0)
    
    if distance == 'euclidean':
        l2_dist_lst = []
        for feature_vec in gallery_pool:
            l2_dist = calculate_distance(embedded_query, feature_vec)
            l2_dist_lst.append(l2_dist)
    
    # get the best match with query
    _, best_idxs = torch.sort(torch.tensor(l2_dist_lst), descending=False)
    best_matchs_path = [image_path_pool[idx] for idx in best_idxs.tolist()]
    best_matchs_path = best_matchs_path[:topk]
    
    best_match_pids = gallery_label_pool[best_idxs]
    best_match_pids = best_match_pids[:topk]
    print(best_matchs_path)
    
    # visualize result
    fig, axes = plt.subplots(1, topk+1, figsize=(8, 8))
    image = cv2.imread(query_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Query image')
    for i in range(1, topk+1):
        image = cv2.imread(best_matchs_path[i-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(best_match_pids[i-1].item())
    plt.savefig("retrieve_results/retrieval_result.png", dpi=300, bbox_inches='tight')
    plt.show()
    

def evaluate_retrieve(model, query_loader, gallery_loader, metric='topk', distance='euclidean', topk=[1, 5]):
    """ function to evaluate model
        metric: {'topk', 'map'}
        evaluate on each batch

    Args:
        queries (torch.Tensor): (batch_size, 512-D vec)
        gallery (torch.Tensor): (512-D vec gallery)
        metric (str, optional): Defaults to 'topk'.
        k (_type_, optional): topk for evaluate
    """
    model = model.to(device).eval()
    
    # Embedded query to get the feature vector
    query_pool = torch.tensor([], device=device)
    query_label_pool = torch.tensor([], device=device)
    for query, query_label in query_loader:
        query = query.to(device)
        query_label = query_label.to(device)
        
        with torch.no_grad():
            embedded_query = model(query)
            query_pool = torch.cat((query_pool, embedded_query), dim=0)
            query_label_pool = torch.cat((query_label_pool, query_label), dim=0)
    print(len(query_pool))
    
    # Embedded gallery to get the feature vectors
    gallery_pool = torch.tensor([], device=device)
    gallery_label_pool = torch.tensor([], device=device)
    for gallery_batch, gallery_label_batch in gallery_loader:
        gallery_batch = gallery_batch.to(device)
        gallery_label_batch = gallery_label_batch.to(device)
        
        with torch.no_grad():
            embedded_gallery = model(gallery_batch)
            gallery_pool = torch.cat((gallery_pool, embedded_gallery), dim=0)
            gallery_label_pool = torch.cat((gallery_label_pool, gallery_label_batch), dim=0)
    print(len(gallery_pool))        
    
    # Get the evaluation base on some kinds of metric
    if metric == 'topk':
        topk_acc = search_on_gallery_topk(
            queries=query_pool,
            query_labels=query_label_pool,
            gallery=gallery_pool,
            gallery_labels=gallery_label_pool,
            metric=distance,
            topk=topk
        )
    
        for i, k in enumerate(topk):
            print(f"Top {k} acc: {topk_acc[i]:.4f}")
            
    elif metric == 'map':
        map_acc = search_on_gallery_map(
            queries=query_pool,
            query_labels=query_label_pool,
            gallery=gallery_pool,
            gallery_labels=gallery_label_pool,
            metric=distance,
            topk=topk
        )
        
        for i, k in enumerate(topk):
            print(f"Top {k} map: {map_acc[i]:.4f}")

def evaluate_classification(model, query_loader, topk=[1, 5]):
    # Embedded query to get the feature vector
    query_pred_pool = torch.tensor([], device=device)
    query_label_pool = torch.tensor([], device=device)
    for query, query_label in query_loader:
        query = query.to(device)
        query_label = query_label.to(device)
        model = model.to(device)
        
        with torch.no_grad():
            query_pred = model(query)
            query_pred_pool = torch.cat((query_pred_pool, query_pred), dim=0)
            query_label_pool = torch.cat((query_label_pool, query_label), dim=0)
            
    acc = accuracy(
        preds=query_pred_pool,
        labels=query_label_pool,
        topk=topk
    )
    
    for i, k in enumerate(topk):
        print(f"Top {k} acc: {acc[i]:.4f}")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', type=str, default='retrieve', help='retrieve|classification')
    parser.add_argument('--metric', type=str, default='topk', help='topk|map')
    parser.add_argument('--distance', type=str, default='euclidean', help='euclidean|cosine')
    parser.add_argument('--topk', type=int, nargs='+', default=5, help='a list contain a value of k for evaluate topk acc')
    parser.add_argument('--visualize', type=bool, default=False, help='True|False')
    parser.add_argument('--query_path', type=str, default=None, help='Path to query image')
    
    args = parser.parse_args()
    
    num_classes = get_total_pids()
    
    if args.visualize == False:
        query_loader = get_query()
        gallery_loader = get_gallery()
        
        if args.evaluate == 'retrieve':
            # set retrieve mode
            feature_extraction = True
            
            # define model
            model = OSNet_model(num_classes=num_classes, feature_extraction=feature_extraction)
            # load weight
            model = model.to(device)
            model.load_state_dict(torch.load('./checkpoints/best_weight.pt', weights_only=True))
        
            evaluate_retrieve(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                metric=args.metric,
                distance=args.distance,
                topk=args.topk
            )
        elif args.evaluate == 'classification':
            # set retrieve mode
            feature_extraction = False
            
            # define model
            model = OSNet_model(num_classes=num_classes, feature_extraction=feature_extraction)
            # load weight
            model = model.to(device)
            model.load_state_dict(torch.load('./checkpoints/best_weight.pt', weights_only=True))
            
            evaluate_classification(
                model=model,
                query_loader=query_loader,
                topk=args.topk
            )
            
    else:
        gallery_loader = get_gallery(get_image_path=True)
        feature_extraction = True
        
        # define model
        model = OSNet_model(num_classes=num_classes, feature_extraction=feature_extraction)
        # load weight
        model = model.to(device)
        model.load_state_dict(torch.load('./checkpoints/best_weight.pt', weights_only=True))
        
        visualize_search(model, query_path=args.query_path, gallery_loader=gallery_loader)

if __name__ == '__main__':
    main()
        