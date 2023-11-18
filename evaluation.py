import pandas as pd
import numpy as np
import torch

from sklearn.metrics import roc_auc_score

def evaluate(preds, gts_path):
    """
    Given the list of all model outputs (logits), and the path to the ground
    truth (val.pkl), calculate the AUC Score of the classified segments.
    Args:
        preds (List[torch.Tensor]): The model ouputs (logits). This is a
            list of all the tensors produced by the model for all samples in
            val.pkl. It should be a list of length 4332 (size of val). All
            tensors in the list should be of size 50 (number of classes).
        gts_path (str): The path to val.pkl
    Returns:
        auc_score (float): A float representing the AUC Score
    """
    # gts = torch.load(gts_path, map_location='cpu') # Ground truth labels, pass path to val.pkl
    gts = pd.read_pickle(gts_path)
    # print(gts)
    labels = []
    model_outs = []
    for i in range(len(preds)):
        # labels.append(gts[i][2].numpy())                             # A 50D Ground Truth binary vector
        labels.append(np.array(gts.iloc[i]['label']).astype(float))    # A 50D Ground Truth binary vector
        model_outs.append(preds[i].cpu().numpy()) # A 50D vector that assigns probability to each class

    labels = np.array(labels).astype(float)
    model_outs = np.array(model_outs)
    print(labels.shape, model_outs.shape)

    auc_score = roc_auc_score(y_true=labels, y_score=model_outs, average='micro')
    print("EVALUATION METRICS:")
    print("-------------------------------------------------------------")
    print()
    print('AUC Score: {:.2f}'.format(auc_score))
    print()
    print("-------------------------------------------------------------")

    return auc_score # Return scores if you wish to save to a file