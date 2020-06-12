import numpy as np
import torch


def accuracy_from_wordvecs(output, target, word2vec_mat, ks=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    This calculation is done using the cosine distance for the topK closest vectors.
    """
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)
        output_arr = output.detach().cpu().numpy()
        output_class_arr = 1 - (1 + np.dot(output_arr / np.linalg.norm(output_arr, axis=1, keepdims=True), word2vec_mat.T)) / 2
        output_class_arr_idxs = np.argsort(output_class_arr, axis=1)[:, :maxk]
        target_class_arr = target.cpu().numpy()[:, np.newaxis]
        correct_arr = output_class_arr_idxs == target_class_arr
        res_k = []
        for k in ks:
            correct_k = np.sum(correct_arr[..., :k])
            res_k.append(correct_k * 1.0 / batch_size)
        return res_k, output_class_arr_idxs


def accuracy(output, target, ks=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)
        # Get the class index of the top <maxk> scores for each element of the minibatch
        _, pred_ = output.topk(maxk, 1, True, True)
        pred = pred_.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in ks:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, pred_
