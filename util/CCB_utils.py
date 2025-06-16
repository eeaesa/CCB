import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import torch.distributed as dist
from copy import deepcopy

#-------------------Confidence-Adaptive_Cutmix module-----------------------
def Confidence_Guided_Cutmix(
        x_u_s, pred_u_w,
        prob=0.5, min_ratio=0.4, max_ratio=0.7,
        ratio_1=0.3, ratio_2=1/0.3,
        softmax=False
):
    '''
    x_u_s: [b, in_ch, h, w]
    pred_u_w: [b, c, h, w]
    '''
    b, c, h, w = pred_u_w.shape

    if softmax:
        pred_u_w = F.softmax(pred_u_w, dim=1)

    logits_u, label_u = torch.max(pred_u_w, dim=1)

    batch_conf_u = cal_conf(pred_u_w)

    best_index = torch.argmax(batch_conf_u, dim=0)

    x_bg, label_bg, logits_bg = x_u_s, label_u, logits_u
    x_fore, label_fore, logits_fore = x_u_s, label_u, logits_u

    best_x_bg = x_bg[best_index].unsqueeze(0).repeat(b, 1, 1, 1)
    best_label_bg = label_bg[best_index].repeat(b, 1, 1)
    best_logits_bg = logits_bg[best_index].repeat(b, 1, 1)

    x_mix = torch.clone(x_fore)
    label_mix = torch.clone(label_fore)
    logits_mix = torch.clone(logits_fore)

    if np.random.rand() < prob:
        mask = torch.zeros((b, h, w), device=x_u_s.device)
        for i in range(b):
            aspect_ratio = np.random.uniform(ratio_1, ratio_2)

            cut_width = np.random.uniform(min_ratio * w, max_ratio * w)
            cut_height = cut_width / aspect_ratio

            cut_width = int(min(cut_width, w))
            cut_height = int(min(cut_height, h))

            x_start = np.random.randint(0, w - cut_width + 1)
            y_start = np.random.randint(0, h - cut_height + 1)

            mask[i, y_start:y_start + cut_height, x_start:x_start + cut_width] = 1.0

        x_mix = mask.unsqueeze(1) * x_fore + (1 - mask.unsqueeze(1)) * best_x_bg

        label_mix = mask * label_fore + (1 - mask) * best_label_bg

        logits_mix = mask * logits_fore + (1 - mask) * best_logits_bg

    return x_mix, label_mix, logits_mix

def cal_conf(pred, softmax=False):
    '''
    pred : [b, c, h, w]
    '''
    if softmax:
        pred = F.softmax(pred, dim=1)
    logits, label = torch.max(pred, dim=1)
    entropy = -torch.sum(pred * torch.log(pred + 1e-10), dim=1) # [b, h, w]
    entropy = entropy * logits * (label != 0).float()
    num_classes = pred.shape[1]
    entropy /= (np.log(num_classes) + 1e-10)
    confidence = 1.0 - entropy
    confidence = confidence.mean(dim=[1, 2])  # list[batchsize]
    return confidence

def Confidence_Guided_Cutmix_3D(
        x_u_s, pred_u_w,
        prob=0.5, min_ratio=0.4, max_ratio=0.7,
        ratio_1=0.3, ratio_2=1/0.3,
        softmax=False
):
    '''
    x_u_s: [b, in_ch, h, w, d]
    pred_u_w: [b, c, h, w, d]
    '''
    b, c, h, w, d = pred_u_w.shape

    if softmax:
        pred_u_w = F.softmax(pred_u_w, dim=1)

    logits_u, label_u = torch.max(pred_u_w, dim=1)

    batch_conf_u = cal_conf_3D(pred_u_w)

    best_index = torch.argmax(batch_conf_u, dim=0)

    x_bg, label_bg, logits_bg = x_u_s, label_u, logits_u
    x_fore, label_fore, logits_fore = x_u_s, label_u, logits_u

    best_x_bg = x_bg[best_index].unsqueeze(0).repeat(b, 1, 1, 1, 1)
    best_label_bg = label_bg[best_index].repeat(b, 1, 1, 1)
    best_logits_bg = logits_bg[best_index].repeat(b, 1, 1, 1)

    x_mix = torch.clone(x_fore)
    label_mix = torch.clone(label_fore)
    logits_mix = torch.clone(logits_fore)


    if np.random.rand() < prob:
        mask = torch.zeros((b, h, w, d), device=x_u_s.device)
        for i in range(b):
            aspect_ratio_1 = np.random.uniform(ratio_1, ratio_2)
            aspect_ratio_2 = np.random.uniform(ratio_1, ratio_2)

            cut_width = np.random.uniform(min_ratio * w, max_ratio * w)
            cut_height = cut_width / aspect_ratio_1
            cut_depth = cut_width / aspect_ratio_2

            cut_width = int(min(cut_width, w))
            cut_height = int(min(cut_height, h))
            cut_depth = int(min(cut_depth, d))

            x_start = np.random.randint(0, w - cut_width + 1)
            y_start = np.random.randint(0, h - cut_height + 1)
            z_start = np.random.randint(0, d - cut_depth + 1)

            mask[i,
                y_start:y_start + cut_height,
                x_start:x_start + cut_width,
                z_start:z_start + cut_depth] = 1.0

        x_mix = mask.unsqueeze(1) * x_fore + (1 - mask.unsqueeze(1)) * best_x_bg

        label_mix = mask * label_fore + (1 - mask) * best_label_bg

        logits_mix = mask * logits_fore + (1 - mask) * best_logits_bg

    return x_mix, label_mix, logits_mix

def cal_conf_3D(pred, softmax=False):
    '''
    pred : [b, c, h, w, d]
    '''
    if softmax:
        pred = F.softmax(pred, dim=1)
    logits, label = torch.max(pred, dim=1)
    entropy = -torch.sum(pred * torch.log(pred + 1e-10), dim=1) # [b, h, w, d]
    entropy = entropy * logits * (label != 0).float()
    num_classes = pred.shape[1]
    entropy /= (np.log(num_classes) + 1e-10)
    confidence = 1.0 - entropy
    confidence = confidence.mean(dim=[1, 2, 3])  # list[batchsize]
    return confidence

#---------------Dynamic Adaptive Threshold---------------------
class Class_Aware_Threshold:
    def __init__(self, num_classes, momentum=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        self.prob_conf = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.prob_conf.mean()
        self.prob_max = torch.ones((self.num_classes)) / self.num_classes
        self.prob_std = torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, max_probs, max_idx, eps=1e-10):

        conf_max_list = []
        conf_mean_list = []
        conf_std_list = []
        probs_mean = torch.mean(max_probs)
        for cls in range(self.num_classes):
            cls_map = (max_idx == cls)
            pred_conf_cls_all = max_probs[cls_map]
            if pred_conf_cls_all.numel() != 0:
                cls_mean_conf = torch.mean(pred_conf_cls_all)
                cls_max_conf = pred_conf_cls_all.max()
                if pred_conf_cls_all.numel() > 1:
                    cls_std_conf = torch.var(pred_conf_cls_all, unbiased=True)
                else:
                    cls_std_conf = torch.var(pred_conf_cls_all, unbiased=False)
            else:
                cls_mean_conf = self.prob_conf[cls]
                cls_max_conf = self.prob_max[cls]
                cls_std_conf = self.prob_std[cls]
            conf_mean_list.append(cls_mean_conf)
            conf_max_list.append(cls_max_conf)
            conf_std_list.append(cls_std_conf)

        conf_mean = torch.tensor(conf_mean_list).to(max_probs.device)
        conf_max = torch.tensor(conf_max_list).to(max_probs.device)
        conf_std = torch.tensor(conf_std_list).to(max_probs.device)

        self.prob_max = self.prob_max * self.m + (1 - self.m) * conf_max
        weight = self.prob_max / torch.max(self.prob_max)

        self.prob_conf = self.prob_conf * self.m + (1 - self.m) * conf_mean * weight

    @torch.no_grad()
    def masking(self, logits, label, iters):
        '''
        logits: [b, h, w]
        label: [b, h, w]
        '''
        if not self.prob_conf.is_cuda:
            self.prob_conf = self.prob_conf.to(logits.device)
            self.prob_max = self.prob_max.to(logits.device)
            self.prob_std = self.prob_std.to(logits.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits.device)

        max_probs, max_idx = logits, label
        if iters == 0:
            for cls in range(self.num_classes):
                cls_map = (max_idx == cls)
                pred_conf_cls_all = max_probs[cls_map]
                if pred_conf_cls_all.numel() != 0:
                    # top-25%
                    self.prob_conf[cls] = torch.quantile(pred_conf_cls_all, 0.75)
        else:
            self.update(max_probs, max_idx)
        cls_thresholds = self.prob_conf[max_idx]
        mask = max_probs.ge(cls_thresholds)
        valid_pix_num = mask.sum().item()
        mask = mask.to(max_probs.dtype)

        mask_ratio = torch.tensor(valid_pix_num / max_probs.numel())
        tau = self.prob_conf.mean()

        return mask, mask_ratio, tau

#----------Prototype Contrastive Learning module---------------

def BoundaryClusteringOptimization(rep, pred, valid_pixel, temp=0.07,
                         num_queries=256, num_negatives=512, sample_times=10):
    '''
    rep: [b, f, h, w]
    pred: [b, c, h, w]
    valid_pixel: [b, c, h, w], gt's one-hot encoded
    '''

    device = pred.device
    num_segments = pred.shape[1]

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    pred_soft = torch.softmax(pred, dim=1)
    pred = pred.permute(0, 2, 3, 1)  # [2B, H, W, C]
    feat = rep.permute(0, 2, 3, 1)  # [2B, H, W, D]

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_all_list = []
    seg_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class  # [2B,H,W]
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = pred_soft[:, i, :, :]  # [2B,H,W]
        strong_threshold = torch.quantile(prob_seg, 0.90)
        rep_proto_mask = (prob_seg > strong_threshold) * valid_pixel_seg.bool()  # [2B,H,W], select proto queries
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # [2B,H,W], select hard queries

        proto = feat[rep_proto_mask.bool()] # [N1, D]
        seg_proto_list.append(proto)

        feat_hard = feat[rep_mask_hard.bool()] # [N2, D]
        seg_hard_list.append(feat_hard)

        feat_all = feat[valid_pixel_seg.bool()] # [N3, D]
        seg_all_list.append(feat_all)

        seg_num_list.append(int(valid_pixel_seg.sum().item()))  # valid_pixel_seg.sum()

    feat_proto_list = process_proto_list(seg_proto_list, num_queries, need_global=True) # [C, sample_times+1, D]
    query_list = process_proto_list(seg_hard_list, num_queries, need_global=True) # [C, sample_times, D]

    loss_inner = supcon_contrastive_loss(feat_proto_list)

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:
        # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
    else:
        reco_loss = torch.tensor(0.0).to(device)
        valid_seg = len(seg_num_list)  # N

        for i in range(valid_seg):
            if i >= len(query_list) or i >= len(feat_proto_list) or query_list[i].numel() == 0 or feat_proto_list[i].numel() == 0:
                continue

            query_cls = query_list[i]  # [sample_times, d]
            key_cls = feat_proto_list[i]  # [sample_times+1, d]

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                query_cls = nn.functional.normalize(query_cls, dim=1)  # [sample_times, d]
                key_cls = nn.functional.normalize(key_cls, dim=1)  # [sample_times+1, d]


                cur_cls = query_cls
                other_cls = []
                for j in range(valid_seg):
                    if j != i and j < len(query_list) and query_list[j].numel() > 0:
                        other_cls.append(query_list[j])

                if not other_cls:
                    continue

                similarities = []
                for proto in other_cls:
                    if proto.size(1) == cur_cls.size(1):
                        sim = F.cosine_similarity(cur_cls, proto, dim=1).mean()
                        similarities.append(sim)

                if not similarities:
                    continue

                similarities = torch.tensor(similarities).to(device)

                probs = F.softmax(similarities, dim=0)

                sample_counts = (probs * num_negatives).round().int()

                while sample_counts.sum() != num_negatives:
                    diff = num_negatives - sample_counts.sum()
                    idx = torch.argmax(probs)
                    sample_counts[idx] += diff

                other_cls_queue = []
                for j in range(valid_seg):
                    if j != i and j < len(seg_all_list) and seg_all_list[j].numel() > 0:
                        other_cls_queue.append(seg_all_list[j])

                negative_vectors = []
                for k, count in enumerate(sample_counts):
                    if count > 0 and k < len(other_cls_queue):
                        proto = other_cls_queue[k]
                        available = proto.size(0)
                        if available == 0:
                            continue
                        count = min(count, available)
                        indices = torch.randperm(available)[:count]
                        selected_vectors = proto[indices]
                        negative_vectors.append(selected_vectors)

                if not negative_vectors:
                    continue

                queue_neg = torch.cat(negative_vectors, dim=0)  # [num_negatives, D]
                if queue_neg.numel() == 0:
                    continue

                queue_neg = nn.functional.normalize(queue_neg, dim=1)  # [num_negatives, D]
                queue_neg = queue_neg.permute(1, 0) # [D, num_negatives]

                if query_cls.size(0) != key_cls.size(0):
                    N = query_cls.size(0)
                    M = key_cls.size(0) - N
                    indices = torch.randint(0, N, (M,))
                    sampled_rows = query_cls[indices, :]  # [M, D]
                    query_cls = torch.cat([query_cls, sampled_rows], dim=0)  # [N+M, D]

                # positive logits: Nx1
                l_pos = torch.einsum("nd,nd->n", [query_cls, key_cls]).unsqueeze(-1)  # [n, 1]
                # negative logits: NxK
                l_neg = torch.einsum("nd,dk->nk", [query_cls, queue_neg.clone().detach()])
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)
                # apply temperature
                logits /= temp
                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

                reco_loss = reco_loss + F.cross_entropy(logits, labels)

        if valid_seg > 0:
            return reco_loss / valid_seg, loss_inner / valid_seg
        else:
            return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

def label_one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def regularization_loss(bases):
    k, c = bases.size()
    if k==0:
        print(k)
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num

    return loss_all

def supcon_contrastive_loss(out_list, temperature=0.07):

    device = out_list[0].device

    valid_features = [feat for feat in out_list if feat.numel() > 0]
    if len(valid_features) < 2:
        return torch.tensor(0.0, device=out_list[0].device if len(out_list) > 0 else 'cpu')

    features = torch.cat(valid_features, dim=0)  # [total_samples, d]
    num_classes = len(valid_features)
    samples_per_class = [feat.shape[0] for feat in valid_features]

    labels = torch.cat([
        torch.full((samples,), class_id, dtype=torch.long, device=features.device)
        for class_id, samples in enumerate(samples_per_class)
    ])  # [total_samples]

    features = F.normalize(features, p=2, dim=-1)

    # similarity_matrix = torch.matmul(features, features.T) / temperature  # [total_samples, total_samples]
    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()


    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # [total_samples, total_samples]

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(mask.size(0)).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

    loss = -mean_log_prob_pos.mean()

    return loss

def process_proto_list(seg_proto_list, sample_times, need_global=False):

    out_list = []
    d = seg_proto_list[0].shape[1] if len(seg_proto_list) > 0 else 0

    for proto in seg_proto_list:
        if proto.shape[0] == 0:
            out_list.append(proto)
            continue

        N = proto.shape[0]

        if N < sample_times:
            indices = torch.arange(N)
            extra_indices = torch.randint(0, N, (sample_times - N,))
            indices = torch.cat([indices, extra_indices])
        else:
            indices = torch.randperm(N)[:sample_times]

        local_feats = proto[indices]  # [sample_times, d]

        if need_global:
            global_feat = proto.mean(dim=0, keepdim=True)  # [1, d]
            combined = torch.cat([local_feats, global_feat], dim=0)  # [sample_times+1, d]
        else:
            combined = local_feats

        out_list.append(combined)

    return out_list

def BoundaryClusteringOptimization_3D(rep, pred, valid_pixel, temp=0.07,
                                  num_queries=256, num_negatives=512, sample_times=10):
    '''
    rep: [b, f, h, w, d]
    pred: [b, c, h, w, d]
    valid_pixel: [b, c, h, w, d]
    '''
    device = pred.device
    num_segments = pred.shape[1]

    # permute representation for indexing: batch x h x w x d x feature_channel
    pred_soft = torch.softmax(pred, dim=1)
    pred = pred.permute(0, 2, 3, 4, 1)  # [b, h, w, d, c]
    feat = rep.permute(0, 2, 3, 4, 1)  # [b, h, w, d, f]

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_all_list = []
    seg_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # [b, h, w, d]
        if valid_pixel_seg.sum() == 0:  # skip empty classes
            continue

        prob_seg = pred_soft[:, i]  # [b, h, w, d]
        strong_threshold = torch.quantile(prob_seg, 0.90)

        # 3D mask operations
        rep_proto_mask = (prob_seg > strong_threshold) * valid_pixel_seg.bool()  # [b, h, w, d]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # [b, h, w, d]

        proto = feat[rep_proto_mask]  # [N1, f]
        seg_proto_list.append(proto)

        feat_hard = feat[rep_mask_hard]  # [N2, f]
        seg_hard_list.append(feat_hard)

        feat_all = feat[valid_pixel_seg.bool()]  # [N3, f]
        seg_all_list.append(feat_all)

        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # Process prototypes and queries
    feat_proto_list = process_proto_list_3D(seg_proto_list, num_queries, need_global=True)
    query_list = process_proto_list_3D(seg_hard_list, num_queries, need_global=True)

    # intra loss using supcon
    loss_inner = supcon_contrastive_loss_3D(feat_proto_list)

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:
        return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
    else:
        reco_loss = torch.tensor(0.0).to(device)
        valid_seg = len(seg_num_list)

        for i in range(valid_seg):
            if i >= len(query_list) or i >= len(feat_proto_list) or \
                    query_list[i].numel() == 0 or feat_proto_list[i].numel() == 0:
                continue

            query_cls = query_list[i]  # [sample_times, f]
            key_cls = feat_proto_list[i]  # [sample_times+1, f]

            # normalize features
            with torch.no_grad():
                query_cls = F.normalize(query_cls, dim=1)
                key_cls = F.normalize(key_cls, dim=1)

                # sample negative keys
                other_cls = []
                for j in range(valid_seg):
                    if j != i and j < len(seg_all_list) and seg_all_list[j].numel() > 0:
                        other_cls.append(seg_all_list[j])

                if not other_cls:
                    continue

                # sample negative vectors proportionally
                negative_vectors = []
                total_available = sum([c.shape[0] for c in other_cls])
                if total_available == 0:
                    continue

                for cls_feat in other_cls:
                    available = cls_feat.shape[0]
                    if available == 0:
                        continue
                    sample_num = max(1, int(num_negatives * (available / total_available)))
                    indices = torch.randperm(available)[:sample_num]
                    negative_vectors.append(cls_feat[indices])

                if not negative_vectors:
                    continue

                queue_neg = torch.cat(negative_vectors, dim=0)  # [num_negatives, f]
                if queue_neg.shape[0] > num_negatives:
                    queue_neg = queue_neg[:num_negatives]
                elif queue_neg.shape[0] < num_negatives:
                    # pad with random samples if not enough
                    shortage = num_negatives - queue_neg.shape[0]
                    extra_indices = torch.randint(0, queue_neg.shape[0], (shortage,))
                    queue_neg = torch.cat([queue_neg, queue_neg[extra_indices]], dim=0)

                queue_neg = F.normalize(queue_neg, dim=1).permute(1, 0)  # [f, num_negatives]

            # compute logits
            l_pos = torch.einsum("nd,nd->n", [query_cls, key_cls]).unsqueeze(-1)
            l_neg = torch.einsum("nd,dk->nk", [query_cls, queue_neg])
            logits = torch.cat([l_pos, l_neg], dim=1) / temp
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            reco_loss += F.cross_entropy(logits, labels)

        return (reco_loss / valid_seg if valid_seg > 0 else torch.tensor(0.0).to(device),
                loss_inner / valid_seg if valid_seg > 0 else torch.tensor(0.0).to(device))


def label_one_hot_encoder_3D(input_tensor, n_classes):
    '''
    input_tensor: [b, 1, h, w, d]
    output: [b, c, h, w, d]
    '''
    tensor_list = []
    for i in range(n_classes):
        temp_prob = (input_tensor == i).float()  # [b, 1, h, w, d]
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)  # [b, c, h, w, d]
    return output_tensor


def supcon_contrastive_loss_3D(out_list, temperature=0.07):

    device = out_list[0].device if len(out_list) > 0 else 'cpu'
    valid_features = [feat for feat in out_list if feat.numel() > 0]

    if len(valid_features) < 2:
        return torch.tensor(0.0, device=device)

    features = torch.cat(valid_features, dim=0)  # [total_samples, f]
    samples_per_class = [feat.shape[0] for feat in valid_features]
    labels = torch.cat([
        torch.full((samples,), class_id, dtype=torch.long, device=features.device)
        for class_id, samples in enumerate(samples_per_class)
    ])

    features = F.normalize(features, p=2, dim=-1)
    logits = torch.matmul(features, features.T) / temperature

    # for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(mask.size(0)).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    loss = -mean_log_prob_pos.mean()

    return loss


def process_proto_list_3D(seg_proto_list, sample_times, need_global=False):
    out_list = []
    for proto in seg_proto_list:
        if proto.shape[0] == 0:
            out_list.append(proto)
            continue

        N = proto.shape[0]
        if N < sample_times:
            indices = torch.arange(N)
            extra_indices = torch.randint(0, N, (sample_times - N,))
            indices = torch.cat([indices, extra_indices])
        else:
            indices = torch.randperm(N)[:sample_times]

        local_feats = proto[indices]  # [sample_times, f]

        if need_global:
            global_feat = proto.mean(dim=0, keepdim=True)  # [1, f]
            combined = torch.cat([local_feats, global_feat], dim=0)
        else:
            combined = local_feats

        out_list.append(combined)

    return out_list


class ProjectionHead(nn.Module):
    """PyTorch version of projection_head"""

    def __init__(self,
                 num_input_channels,
                 num_projection_channels=256,
                 num_projection_layers=3,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_projection_layers - 1):
            self.layers.append(nn.Conv2d(num_input_channels, num_projection_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(num_projection_channels))
            self.layers.append(nn.ReLU(inplace=True))
            num_input_channels = num_projection_channels

        # Final layer
        self.final_conv = nn.Conv2d(num_input_channels, num_projection_channels, kernel_size=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return F.normalize(x, p=2, dim=-1)

class ProjectionHead3D(nn.Module):
    """PyTorch 3D version of projection_head with H, W, D ordering"""

    def __init__(self,
                 num_input_channels,
                 num_projection_channels=256,
                 num_projection_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_projection_layers - 1):
            self.layers.append(nn.Conv3d(num_input_channels, num_projection_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm3d(num_projection_channels))
            self.layers.append(nn.ReLU(inplace=True))
            num_input_channels = num_projection_channels

        # Final layer
        self.final_conv = nn.Conv3d(num_input_channels, num_projection_channels, kernel_size=1)

    def forward(self, x):
        '''
        input: (b, c_in, h, w, d)
        output: (b, c_proj, h, w, d)
        '''
        if x.dim() == 5:
            if x.shape[2] > x.shape[3]:
                x = x.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return F.normalize(x, p=2, dim=1)




