import torch
from sklearn.metrics import roc_auc_score


def auc(model, y_true, test_loader, batch_size, prediction_dst=None):
    """
    Uses scikit-learn roc_auc_score.

    Parameters
    ----------
    model 
    
    y_true : torch.Tensor
    test_loader : Dataloader
        `shuffle` parameter must be False. Order has to be the same than
        in `` 

    Returns
    -------
    float
    """    
    preds = torch.ones_like(y_true) * -1

    for i, (x, _) in enumerate(test_loader):
        preds[
            i * batch_size:min((i + 1) * batch_size, len(y_true))
        ] = model(x)

    assert torch.all(preds != -1.0).item(), "Not all values replaced for predictions."

    if prediction_dst is not None:
        torch.save(preds, prediction_dst)

    return roc_auc_score(y_true, preds)