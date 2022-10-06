import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def arg_shuffle_ties(batch_rankings, descending=True, device=None):
    '''Shuffle ties, and return the corresponding indice '''
    batch_size, ranking_size = batch_rankings.size()
    if batch_size > 1:
        list_rperms = []
        for _ in range(batch_size):
            list_rperms.append(torch.randperm(ranking_size, device=device))
        batch_rperms = torch.stack(list_rperms, dim=0)
    else:
        batch_rperms = torch.randperm(ranking_size, device=device).view(1, -1)

    batch_shuffled_rankings = torch.gather(batch_rankings, dim=1, index=batch_rperms)
    batch_desc_inds = torch.argsort(batch_shuffled_rankings, descending=descending)
    batch_shuffle_ties_inds = torch.gather(batch_rperms, dim=1, index=batch_desc_inds)

    return batch_shuffle_ties_inds


def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.mean(observation_loss, dim=1))


def altListMLE(batch_preds, batch_std_labels):
    '''
    @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
    @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
    @param kwargs:
    @return:
    '''
    # shuffle per epoch rather than using the same order for a query
    batch_shuffle_ties_inds = arg_shuffle_ties(batch_rankings=batch_std_labels, descending=True,
                                               device=batch_std_labels.device)
    batch_preds_shuffled_ties = torch.gather(batch_preds, dim=1, index=batch_shuffle_ties_inds)

    # 1 using self-defined op since torch.flip() is later added
    '''
    batch_logcumsumexps = apply_LogCumsumExp(target_batch_preds)
    batch_loss = torch.sum(batch_logcumsumexps - target_batch_preds)
    '''

    # 2 since torch.flip() is available now, the loss can also be directly computed without defining a new op
    # '''
    m, _ = torch.max(batch_preds_shuffled_ties, dim=1,
                     keepdim=True)  # a transformation aiming for higher stability when computing softmax() with exp()
    y = batch_preds_shuffled_ties - m
    y = torch.exp(y)
    y_backward_cumsum = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1),
                                   dims=[1])  # row-wise cumulative sum, from tail to head
    batch_logcumsumexps = torch.log(y_backward_cumsum) + m  # corresponding to the '-m' operation
    batch_loss = torch.sum(torch.sum((batch_logcumsumexps - batch_preds_shuffled_ties), dim=1))

    return batch_loss
