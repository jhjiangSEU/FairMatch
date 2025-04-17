import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import numpy as np
import yaml
import shutil
import pickle
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, args):
        super().__init__()
        self.temperature = args.moco_t
        self.base_temperature = args.moco_t

    def forward(self, features, mask=None, weight=None, batch_size=-1):
        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().cuda()
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * weight * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 计算未经加权的交叉熵损失
        if self.weight is not None:
            ce_loss = ce_loss * self.weight  # 对损失进行加权处理
        return torch.mean(ce_loss)


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def load_moco_model(moco_path, gpu_num, model, ema_model, requires_grad=False):
    state = torch.load(moco_path, map_location='cuda:' + str(gpu_num))['state_dict']
    model_paras = {k.replace('backbone', 'encoder'): v for k, v in state.items()}
    model_paras = {k.replace('projector', 'head'): v for k, v in model_paras.items()}
    model_paras = {k.replace('downsample', 'shortcut'): v for k, v in model_paras.items()}
    model.load_state_dict(model_paras, strict=False)

    # 冻结特征提取器参数
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = requires_grad

    ema_model_paras = {k.replace('momentum_backbone', 'encoder'): v for k, v in state.items()}
    ema_model_paras = {k.replace('momentum_projector', 'head'): v for k, v in ema_model_paras.items()}
    ema_model_paras = {k.replace('downsample', 'shortcut'): v for k, v in ema_model_paras.items()}
    ema_model.load_state_dict(ema_model_paras, strict=False)

    # 冻结特征提取器参数
    for name, param in ema_model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = requires_grad

    return model, ema_model


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().cuda()
        if len(target.shape) == 2:
            target = torch.argmax(target, dim=1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def disa_acc(train_loader_lb, confidence, file_root, epoch):
    disa_acc_lb = (train_loader_lb.dataset.targets.cuda() == torch.max(confidence, dim=1)[1]).sum() / len(confidence)
    with open(file_root + '/transductive_acc.txt', 'a') as f:
        f.write('{} epcoh transductive acc_lb: {:.4f}\n'.format(epoch, disa_acc_lb))


def sample_labeled_unlabeled_data(args, target, num_classes, lb_num_labels):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # args.balanced = False
    # if args.balanced:
    #     # get samples per class
    #     # balanced setting, lb_num_labels is total number of labels for labeled data
    #     assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
    #     lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    #
    #     ulb_samples_per_class = None
    #
    #     lb_idx = []
    #     ulb_idx = []
    #
    #     target = np.asarray(target)
    #     for c in range(num_classes):
    #         idx = np.where(target == c)[0]
    #         np.random.shuffle(idx)
    #         lb_idx.extend(idx[:lb_samples_per_class[c]])
    #         if ulb_samples_per_class is None:
    #             ulb_idx.extend(idx[lb_samples_per_class[c]:])
    #         else:
    #             ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c] + ulb_samples_per_class[c]])
    #
    #     if isinstance(lb_idx, list):
    #         lb_idx = np.asarray(lb_idx)
    #     if isinstance(ulb_idx, list):
    #         ulb_idx = np.asarray(ulb_idx)
    # else:
    idx = random.sample(range(len(target)), len(target))
    lb_idx = idx[:lb_num_labels]
    ulb_idx = idx[lb_num_labels:]

    return lb_idx, ulb_idx


def partialize(y, p):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    for i in range(n):
        row = new_y[i, :]
        row[np.where(np.random.binomial(1, p, c) == 1)] = 1
        while torch.sum(row) == 1:
            row[np.random.randint(0, c)] = 1
        avgC += torch.sum(row)
        new_y[i] = row / torch.sum(row)

    avgC = avgC / n
    return new_y, avgC


def save_checkpoint(state, is_best, filename='latest_model.pth', best_file_name='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100'

    meta = unpickle('../data/cifar100/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]: i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]

        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY
