from sklearn.metrics import roc_auc_score


def calculate_auc(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
