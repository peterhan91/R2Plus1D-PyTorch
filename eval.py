import torch
import pandas as pd
import sklearn
import sklearn.metrics as sklm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from dataset import MRIDataset, ToTensor

def make_pred_multilabel(model, directory, save_dir):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model
    Args:
        model: the model that previously fine tuned to training data
        directory: Directory where the data is to be loaded from
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 1

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    dataset = MRIDataset(directory=directory, 
                        mode='valid', 
                        transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, sample in enumerate(dataloader):
        inputs, labels = sample['buffers'], sample['labels']
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy() # shape: [4, 3, ]
        batch_size = len(true_labels)

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()
        scanlist = dataset.scanlists
        # get predictions and true values for each item in batch
        for j in range(0, batch_size):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = scanlist[BATCH_SIZE * i + j]
            truerow["Image Index"] = scanlist[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])
    fprs, tprs = [], []
    # calc AUCs
    for column in true_df:
        if column not in ['abnormality', 'ACL tear', 'meniscal tear']:
            continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(actual.as_matrix().astype(int), 
                                                pred.as_matrix())
            fpr, tpr, _ = sklm.roc_curve(actual.as_matrix().astype(int), pred.as_matrix())
            fprs.append(fpr)
            tprs.append(tpr)
            
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    sns.set()
    fig, ax = plt.subplots(1, 1, figsize = (6, 6), dpi = 300)
    color = ['b-', 'r-', 'g-']
    for i, label in enumerate(dataset.PRED_LABEL):
        ax.plot(fprs[i], tprs[i], color[i], alpha=0.7, label = label)
    ax.legend(loc = 4, prop={'size': 8})
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.savefig(save_dir+'RSNA_ROC.png', 
                dpi=300, bbox_inches = 'tight')

    pred_df.to_csv(save_dir+"preds.csv", index=False)
    auc_df.to_csv(save_dir+"aucs.csv", index=False)
    return pred_df, auc_df