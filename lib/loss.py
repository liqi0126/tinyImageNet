import torch
import torch.nn as nn


def naive_cross_entropy_loss(input, target):
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_size, keep_index):
        assert 0.0 < label_smoothing <= 1.0
        self.keep_index = keep_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (keep_index - 1)
        one_hot = torch.full((tgt_size,), smoothing_value)
        one_hot[keep_index:tgt_size] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)

        return naive_cross_entropy_loss(output, model_prob)
