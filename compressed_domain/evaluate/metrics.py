import numpy as np

def compute_metrics(sim_matrix):
    nn_idx = np.argsort(-sim_matrix, axis=1)
    y = np.eye(nn_idx.shape[0])    
    ind = np.take_along_axis(y, nn_idx, axis=1)
    ind = np.where(ind == 1)[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1    
    
    print('shape-similarity-matrix: {}'.format(nn_idx.shape))
    print("Text-to-Video: >>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
        format(metrics['R1'], metrics['R5'], metrics['R10'], metrics['MR'], metrics['MeanR']))

    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))
