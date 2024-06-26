import paddle
from ipdb import set_trace
import numpy as np


class BaseLoss(paddle.nn.Layer):

    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = paddle.ones(shape=[1], dtype=preds[0].dtype)
            errs = [self._forward(preds[n], targets[n], weight[n]) for n in
                range(N)]
            err = paddle.mean(x=paddle.stack(x=errs))
        elif isinstance(preds, paddle.Tensor):
            if weight is None:
                weight = paddle.ones(shape=[1], dtype=preds.dtype)
            err = self._forward(preds, targets, weight)
        return err


class SmoothL1Loss(BaseLoss):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return paddle.nn.functional.smooth_l1_loss(input=pred, label=target)


class L1Loss(BaseLoss):

    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return paddle.mean(x=weight * paddle.abs(x=pred - target))


class L2Loss(BaseLoss):

    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return paddle.mean(x=weight * paddle.pow(x=pred - target, y=2))


class BCELoss(BaseLoss):

    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return paddle.nn.functional.binary_cross_entropy(input=pred, label=
            target, weight=weight)


class BCEWithLogitsLoss(BaseLoss):

    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return paddle.nn.functional.binary_cross_entropy_with_logits(logit=
            pred, label=target, weight=weight)


class CELoss(BaseLoss):

    def __init__(self):
        super(CELoss, self).__init__()

    def _forward(self, pred, target, weight=None):
        return paddle.nn.functional.cross_entropy(input=pred, label=target)


class YBLoss2(paddle.nn.Layer):
    """
	Contrastive loss function.
	Based on:
	"""

    def __init__(self, margin=1.0, τ=0.05):
        super(YBLoss2, self).__init__()
        self.margin = margin
        self.τ = τ

    def forward(self, prob_x1, prob_x2, prob_joint, rand_idx, sample_idx,
        target, opt, x1, x2, x1_class, x2_class):
        v_prob_pos = paddle.zeros(shape=[tuple(prob_x2.shape)[0], 1, 25]).to(
            prob_x2.place)
        v_prob_neg = paddle.zeros(shape=[tuple(prob_x2.shape)[0], 1, 25]).to(
            prob_x2.place)
        a_prob_pos = paddle.zeros(shape=[tuple(prob_x2.shape)[0], 1, 25]).to(
            prob_x2.place)
        a_prob_neg = paddle.zeros(shape=[tuple(prob_x2.shape)[0], 1, 25]).to(
            prob_x2.place)
        feature_dic1 = []
        feature_dic2 = []
        class_dic1 = []
        class_dic2 = []
        loss = []
        x1_prob_total = []
        x2_prob_total = []
        for i in range(x1.shape[0]):
            if opt.aug_type == 'vision':
                loss.append(paddle.nn.functional.binary_cross_entropy(input
                    =prob_x2[i], label=target[sample_idx[i]]))
            elif opt.aug_type == 'audio':
                loss.append(paddle.nn.functional.binary_cross_entropy(input
                    =prob_x2[i], label=target[sample_idx[i]]))
            elif opt.aug_type == 'ada':
                if i == 0:
                    feature_dic1 = x1[i].unsqueeze(axis=0)
                    feature_dic2 = x2[i].unsqueeze(axis=0)
                    class_dic1 = x1_class[i].unsqueeze(axis=0)
                    class_dic2 = x2_class[i].unsqueeze(axis=0)
                else:
                    feature_dic1 = paddle.concat(x=(feature_dic1, x1[i].
                        unsqueeze(axis=0)), axis=0)
                    feature_dic2 = paddle.concat(x=(feature_dic2, x2[i].
                        unsqueeze(axis=0)), axis=0)
                    class_dic1 = paddle.concat(x=(class_dic1, x1_class[i].
                        unsqueeze(axis=0)), axis=0)
                    class_dic2 = paddle.concat(x=(class_dic2, x2_class[i].
                        unsqueeze(axis=0)), axis=0)
            elif opt.aug_type == 'mix' or opt.aug_type == 'yybag':
                gg_sample = paddle.concat(x=((prob_x2[i] * target[
                    sample_idx[i]]).max().unsqueeze(axis=0), (prob_x2[i + 1 *
                    len(rand_idx)] * target[sample_idx[i]]).max().unsqueeze
                    (axis=0), (prob_x2[i + 2 * len(rand_idx)] * target[
                    sample_idx[i]]).max().unsqueeze(axis=0), (prob_x2[i + 3 *
                    len(rand_idx)] * target[sample_idx[i]]).max().unsqueeze
                    (axis=0)))
                gg_rand = paddle.concat(x=((prob_x1[i] * target[rand_idx[i]
                    ]).max().unsqueeze(axis=0), (prob_x1[i + 1 * len(
                    rand_idx)] * target[rand_idx[i]]).max().unsqueeze(axis=
                    0), (prob_x1[i + 2 * len(rand_idx)] * target[rand_idx[i
                    ]]).max().unsqueeze(axis=0), (prob_x1[i + 3 * len(
                    rand_idx)] * target[rand_idx[i]]).max().unsqueeze(axis=0)))
                loss.append(paddle.nn.functional.binary_cross_entropy(input
                    =prob_x2[i + gg_sample.argmax() * len(rand_idx)], label
                    =target[sample_idx[i]]))
                loss.append(paddle.nn.functional.binary_cross_entropy(input
                    =prob_x1[i + gg_rand.argmax() * len(rand_idx)], label=
                    target[rand_idx[i]]))
                loss.append(paddle.nn.functional.binary_cross_entropy(input
                    =prob_joint[i + gg_joint.argmax() * len(rand_idx)],
                    label=joint_label))
        if opt.aug_type == 'vision':
            return paddle.mean(x=paddle.stack(x=loss))
        elif opt.aug_type == 'audio':
            return paddle.mean(x=paddle.stack(x=loss))
        elif opt.aug_type == 'yybag':
            x1_prob_total_filiter = paddle.stack(x=x1_prob_total) * target[
                rand_idx]
            x2_prob_total_filiter = paddle.stack(x=x2_prob_total) * target[
                sample_idx]
            interval = int(len(x2_prob_total_filiter) / 3)
            pos_bag = x2_prob_total_filiter[:interval].sum(axis=-1)
            neg_bag = x2_prob_total_filiter[interval:-interval].sum(axis=-1
                ) + x2_prob_total_filiter[-interval:].sum(axis=-1)
            loss = (pos_bag / (neg_bag + pos_bag)).mean() + (1 - (neg_bag /
                (neg_bag + pos_bag)).mean())
            return loss
        elif opt.aug_type == 'ada':
            feature_dic1 = paddle.nn.functional.normalize(x=feature_dic1,
                axis=-1).squeeze(axis=1)
            feature_dic2 = paddle.nn.functional.normalize(x=feature_dic2,
                axis=-1).squeeze(axis=1)
            target_audio = target[0]
            target_vis = target[1]
            corr = paddle.mm(input=target_audio, mat2=target_vis.transpose(
                perm=[1, 0]))
            corr[corr != 0] = opt.smooth
            sim = paddle.mm(input=feature_dic1, mat2=feature_dic2.transpose
                (perm=[1, 0]))
            pos = paddle.sum(x=paddle.exp(x=sim / opt.tmp) * corr.detach(),
                axis=1) + 1e-10
            neg = paddle.sum(x=paddle.exp(x=sim / opt.tmp) * (1 - corr.
                detach()), axis=1) + 1e-10
            loss = (-paddle.log(x=pos / (pos + neg))).mean()
            return loss
        elif opt.aug_type == 'mix':
            return paddle.mean(x=paddle.stack(x=loss))
        elif opt.aug_type == 'mimix':
            corr = paddle.mm(input=(target[rand_idx] + target[rand_idx]).
                clip_(min=0, max=1), mat2=(target[rand_idx] + target[
                rand_idx]).clip_(min=0, max=1).transpose(perm=[1, 0]))
            corr[corr != 0] = 1
            corr_copy = paddle.clone(x=corr)
            corr_copy[corr_copy != 0] = 1
            exact_same = target[sample_idx].sum(axis=-1)
            corr[corr == exact_same] = 1
            norm_feature_1 = paddle.nn.functional.normalize(x=x1, axis=-1)
            norm_feature_2 = paddle.nn.functional.normalize(x=x2, axis=-1)
            sim = paddle.mm(input=norm_feature_1, mat2=norm_feature_2.
                transpose(perm=[1, 0]))
            pos = paddle.sum(x=paddle.exp(x=sim / opt.tmp) * corr, axis=1
                ) + 1e-10
            neg = paddle.sum(x=paddle.exp(x=sim / opt.tmp) * (1 - corr_copy
                ), axis=1)
            loss = (-paddle.log(x=pos / (pos + neg))).mean()
            return loss


class YBLoss(paddle.nn.Layer):
    """
	Contrastive loss function.
	Based on:
	"""

    def __init__(self, margin=1.0, τ=0.05):
        super(YBLoss, self).__init__()
        self.margin = margin
        self.τ = τ

    def forward(self, all_prob, audio_idx, vis_idx, target, opt):
        v_prob_pos = paddle.zeros(shape=[tuple(all_prob.shape)[0] - len(
            audio_idx), 1, 25]).to(all_prob.place)
        v_prob_neg = paddle.zeros(shape=[tuple(all_prob.shape)[0] - len(
            audio_idx), 1, 25]).to(all_prob.place)
        v_prob_neg_counter = paddle.zeros(shape=[tuple(all_prob.shape)[0] -
            len(audio_idx), 1, 1]).to(all_prob.place)
        a_prob_pos = paddle.zeros(shape=[tuple(all_prob.shape)[0] - len(
            audio_idx), 1, 25]).to(all_prob.place)
        a_prob_neg = paddle.zeros(shape=[tuple(all_prob.shape)[0] - len(
            audio_idx), 1, 25]).to(all_prob.place)
        a_prob_neg_counter = paddle.zeros(shape=[tuple(all_prob.shape)[0] -
            len(audio_idx), 1, 1]).to(all_prob.place)
        for i in range(len(audio_idx)):
            if (target[audio_idx[i]] * target[vis_idx[i]]).sum() == 0:
                if opt.exp:
                    if opt.pos_pool == 'max':
                        a_prob_pos[audio_idx[i]] = paddle.exp(x=all_prob[
                            audio_idx[i], :, 0, :].max(0)[0])
                        v_prob_pos[vis_idx[i]] = paddle.exp(x=all_prob[
                            vis_idx[i], :, 1, :].max(0)[0])
                    elif opt.pos_pool == 'mean':
                        a_prob_pos[audio_idx[i]] = paddle.exp(x=all_prob[
                            audio_idx[i], :, 0, :].mean(axis=0))
                        v_prob_pos[vis_idx[i]] = paddle.exp(x=all_prob[
                            vis_idx[i], :, 1, :].mean(axis=0))
                    if opt.neg_pool == 'max':
                        a_prob_neg[audio_idx[i]] += paddle.exp(x=all_prob[-
                            len(audio_idx) + i, :, :, :].max(0)[0][1])
                        a_prob_neg_counter[audio_idx[i]] += 1
                        v_prob_neg[vis_idx[i]] += paddle.exp(x=all_prob[-
                            len(audio_idx) + i, :, :, :].max(0)[0][0])
                        v_prob_neg_counter[vis_idx[i]] += 1
                    elif opt.neg_pool == 'mean':
                        a_prob_neg[audio_idx[i]] += paddle.exp(x=all_prob[-
                            len(audio_idx) + i, :, :, :].mean(axis=0)[1])
                        a_prob_neg_counter[audio_idx[i]] += 1
                        v_prob_neg[vis_idx[i]] += paddle.exp(x=all_prob[-
                            len(audio_idx) + i, :, :, :].mean(axis=0)[0])
                        v_prob_neg_counter[vis_idx[i]] += 1
                else:
                    if opt.pos_pool == 'max':
                        a_prob_pos[audio_idx[i]] = all_prob[audio_idx[i], :,
                            0, :].max(0)[0]
                        v_prob_pos[vis_idx[i]] = all_prob[vis_idx[i], :, 1, :
                            ].max(0)[0]
                    elif opt.pos_pool == 'mean':
                        a_prob_pos[audio_idx[i]] = all_prob[audio_idx[i], :,
                            0, :].mean(axis=0)
                        v_prob_pos[vis_idx[i]] = all_prob[vis_idx[i], :, 1, :
                            ].mean(axis=0)
                    if opt.neg_pool == 'max':
                        a_prob_neg[audio_idx[i]] += all_prob[-len(audio_idx
                            ) + i, :, :, :].max(0)[0][1]
                        a_prob_neg_counter[audio_idx[i]] += 1
                        v_prob_neg[vis_idx[i]] += all_prob[-len(audio_idx) +
                            i, :, :, :].max(0)[0][0]
                        v_prob_neg_counter[vis_idx[i]] += 1
                    elif opt.neg_pool == 'mean':
                        a_prob_neg[audio_idx[i]] += all_prob[-len(audio_idx
                            ) + i, :, :, :].mean(axis=0)[1]
                        a_prob_neg_counter[audio_idx[i]] += 1
                        v_prob_neg[vis_idx[i]] += all_prob[-len(audio_idx) +
                            i, :, :, :].mean(axis=0)[0]
                        v_prob_neg_counter[vis_idx[i]] += 1
        v_prob_pos_filter = v_prob_pos * target.unsqueeze(axis=1)
        v_prob_neg_filter = v_prob_neg * target.unsqueeze(axis=1)
        a_prob_pos_filter = a_prob_pos * target.unsqueeze(axis=1)
        a_prob_neg_filter = a_prob_neg * target.unsqueeze(axis=1)
        v_pos = v_prob_pos_filter[v_prob_pos_filter != 0]
        v_neg = v_prob_neg_filter[v_prob_neg_filter != 0]
        a_pos = a_prob_pos_filter[a_prob_pos_filter != 0]
        a_neg = a_prob_neg_filter[a_prob_neg_filter != 0]
        loss = (-paddle.log(x=v_pos / (v_pos + v_neg))).mean() + (-paddle.
            log(x=a_pos / (a_pos + a_neg))).mean()
        return loss


class ContrastiveLoss(paddle.nn.Layer):
    """
	Contrastive loss function.
	Based on:
	"""

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3
        x0_type, x1_type, y_type = in_types
        assert tuple(x0_type.shape) == tuple(x1_type.shape)
        assert tuple(x1_type.shape)[0] == tuple(y_type.shape)[0]
        assert tuple(x1_type.shape)[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))
        diff = x0 - x1
        dist_sq = paddle.sum(x=paddle.pow(x=diff, y=2), axis=1)
        dist = paddle.sqrt(x=dist_sq)
        mdist = self.margin - dist
        dist = paddle.clip(x=mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * paddle.pow(x=dist, y=2)
        loss = paddle.sum(x=loss) / 2.0 / tuple(x0.shape)[0]
        return loss


class InfoNCELoss(paddle.nn.Layer):
    """
	Contrastive loss function.
	Based on:
	"""

    def __init__(self, margin=1.0, τ=0.05):
        super(InfoNCELoss, self).__init__()
        self.margin = margin
        self.τ = τ

    def check_type_forward(self, in_types):
        x0_type, x1_type = in_types
        assert tuple(x0_type.shape) == tuple(x1_type.shape)

    def forward(self, q, k):
        fa = q
        fv = k
        N = tuple(q.shape)[0]
        C = tuple(q.shape)[1]
        q = q.reshape(tuple(q.shape)[0] * 10, -1)
        k = k.reshape(tuple(k.shape)[0] * 10, -1)
        q = paddle.nn.functional.normalize(x=q, p=2, axis=-1)
        k = paddle.nn.functional.normalize(x=k, p=2, axis=-1)
        sim = paddle.mm(input=q, mat2=k.T)
        block = paddle.ones(shape=[10, 10])
        pos_w = paddle.eye(num_rows=tuple(sim.shape)[0]).to(q.place)
        for i in range(N):
            pos_w[i * 10:(i + 1) * 10, i * 10:(i + 1) * 10] = block
        neg_w = 1 - pos_w[:, :tuple(sim.shape)[1]]
        pos = paddle.exp(x=paddle.divide(x=sim, y=paddle.to_tensor(self.τ))
            ) * pos_w[:, :tuple(sim.shape)[1]]
        pos = pos.sum(axis=1)
        neg = paddle.sum(x=paddle.exp(x=paddle.divide(x=sim, y=paddle.
            to_tensor(self.τ))) * neg_w, axis=1)
        denominator = neg + pos
        return paddle.mean(x=-paddle.log(x=paddle.divide(x=pos, y=paddle.
            to_tensor(denominator + 1e-08)) + 1e-08))


class MaskInfoNCELoss(paddle.nn.Layer):
    """
	Contrastive loss function.
	Based on:
	"""

    def __init__(self, margin=1.0, τ=0.05):
        super(MaskInfoNCELoss, self).__init__()
        self.margin = margin
        self.τ = τ

    def forward(self, q, k, mask):
        N = tuple(q.shape)[0]
        C = tuple(q.shape)[1]
        q = q.view(tuple(q.shape)[0], -1)
        k = k.view(tuple(k.shape)[0], -1)
        q = paddle.nn.functional.normalize(x=q, p=2, axis=1)
        k = paddle.nn.functional.normalize(x=k, p=2, axis=1)
        sim = paddle.mm(input=q, mat2=k.T)
        tmp_zeros = paddle.zeros(shape=(tuple(sim.shape)[0] - tuple(mask.
            shape)[0], tuple(sim.shape)[1])).to(q.place)
        mask_pos = paddle.concat(x=(mask, tmp_zeros), axis=0)
        neg_w = 1 - mask_pos
        pos = paddle.exp(x=paddle.divide(x=sim, y=paddle.to_tensor(self.τ))
            ) * mask_pos
        pos = pos.sum(axis=1)
        neg = paddle.sum(x=paddle.exp(x=paddle.divide(x=sim, y=paddle.
            to_tensor(self.τ))) * neg_w, axis=1)
        denominator = neg + pos
        return paddle.mean(x=-paddle.log(x=paddle.divide(x=pos, y=paddle.
            to_tensor(denominator + 1e-08)) + 1e-08))
