import argparse
import copy
import random
import dataset as dataset
from utils.utils_func import *
from utils.model import mymethod
from backbone.resnet import ResNet18

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def get_config():
    parser = argparse.ArgumentParser(
        description='Revisiting Consistency Regularization for Deep Partial Label Learning')
    # basic paras
    parser.add_argument('-ep', '--epoch', help='number of epochs', type=int, default=800)
    parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', help='batch size for training', type=int, default=64)
    parser.add_argument('-ds', '--dataset', help='specify a dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10'])
    parser.add_argument('-nc', '--num_classes', help='num of classes', type=int, default=10)
    parser.add_argument('--backbone', help='backbone name', type=str, default='resnet18', choices=['resnet18'],
                        required=False)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.001)
    parser.add_argument('--cosine', action='store_false', help='use cosine lr schedule')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument("--seed", help="seed for initializing training.", type=int, default=0)
    parser.add_argument('--resume', action='store_false', help='resume model')

    # backbone settings
    parser.add_argument("--feat_dim", help='dimensions of low dimensional feature', type=int, default=128)
    parser.add_argument("--hidden_dim", help='dimensions of hidden layer', type=int, default=2048)

    # Saving & loading of the model
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--save_name", type=str, default="cifar10_rho_0.1_pr_0.1")
    parser.add_argument("--model_name", type=str, default='latest_model.pth')

    # semi-supervised settings
    parser.add_argument("--ratio", help='the ratio of unsupervised batch size', type=int, default=7)

    # dataset settings
    parser.add_argument("--hierarchical", action='store_true')
    parser.add_argument('-pr', '--partial_rate', help='partial rate (flipping)', type=float, default=0.1)
    parser.add_argument('--rho', help='partial sample ratio', type=float, default=0.1)
    parser.add_argument('--warm_up', help='warm up epoch', type=int, default=20)

    # moco settings
    parser.add_argument('--moco_queue', help='moco queue length', type=int, default=8192)
    parser.add_argument('--moco_m', help='moco ema update', type=float, default=0.99)
    parser.add_argument('--moco_t', help='moco temperature', type=float, default=0.07)
    parser.add_argument('--loss_weight', help='moco loss weight', type=float, default=0.5)

    # method settings

    parser.add_argument('--alpha_mixup', type=float, default=4)
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--beta_m', type=float, default=1)

    # config file
    parser.add_argument("--c", type=str, default="")

    # add algorithm specific parameters
    args = parser.parse_args()

    args.save_name = f'{args.dataset}_rho_{args.rho}_pr_{args.partial_rate}'
    if args.dataset == 'cifar100':
        args.save_path = os.path.join(args.save_dir, str(args.seed), args.dataset, f'hierarchical_{args.hierarchical}',
                                      args.save_name)
    else:
        args.save_path = os.path.join(args.save_dir, str(args.seed), args.dataset, args.save_name)
    args.load_path = os.path.join(args.save_dir, str(args.seed), args.dataset, args.save_name, args.model_name)

    # SET save_path, logger and tb_logger
    logger_level = "INFO"
    args.logger = get_logger(args.save_name, args.save_path, logger_level)

    args.logger.info(f"Arguments: {args}")

    if args.seed is not None:
        args.logger.info("You have chosen to seed {} training".format(args.seed))
        # random seed has to be set for the synchronization of labeled data sampling in each process.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return args


def my_train(args, train_loader_lb, train_loader_ulb, model, optimizer, epoch, criterion, confidence, loss_cont_fn,
             label_high, gt_high, acc_high):
    logger = args.logger
    model.train()

    train_loss = AverageMeter()
    supervised_loss = AverageMeter()
    hard_consistent_loss = AverageMeter()
    contrastive_loss = AverageMeter()

    iter_lb = enumerate(train_loader_lb)

    for iter, (x_weak_ulb, x_strong_ulb, y_ulb, partial_y_ulb, index_ulb) in enumerate(train_loader_ulb):
        x_weak_ulb = x_weak_ulb.cuda()
        x_strong_ulb = x_strong_ulb.cuda()
        y_ulb = y_ulb.cuda()
        partial_y_ulb = partial_y_ulb.cuda()

        try:
            _, (x_weak, y, partial_y, index) = next(iter_lb)
        except StopIteration:
            iter_lb = enumerate(train_loader_lb)
            _, (x_weak, y, partial_y, index) = next(iter_lb)

        x_weak = x_weak.cuda()
        y = y.cuda()
        partial_y = partial_y.cuda()

        logits_lb, logits_ulb, feats_ulb, logits_ulb_strong, feats_ulb_strong = model(x_weak, x_weak_ulb, x_strong_ulb,
                                                                                      partial_y, partial_y_ulb, y,
                                                                                      y_ulb, index_ulb)

        probs_lb = F.softmax(logits_lb, dim=-1)
        probs_ulb = F.softmax(logits_ulb, dim=-1)

        if epoch < args.warm_up:
            # supervised loss
            sup_loss = confidence[index] * torch.log(probs_lb + 1e-10)
            sup_loss = (-torch.sum(sup_loss)) / sup_loss.size(0)
            supervised_loss.update(sup_loss.item(), len(logits_lb))

            with torch.no_grad():
                model.momentum_update_key_encoder(args)
                _, k = model.ema_encoder(x_strong_ulb)

            add_length = len(feats_ulb)
            features = torch.cat((feats_ulb, k, model.queue_feature.clone().detach()), dim=0)
            max_probs_k, pseudo_labels_k = torch.max(logits_ulb, dim=1)

            model.dequeue_and_enqueue(k, pseudo_labels_k, max_probs_k, y_ulb, args)

            cont_loss = loss_cont_fn(features=features, mask=None, batch_size=add_length)
            contrastive_loss.update(cont_loss.item(), add_length)

            loss = sup_loss + args.loss_weight * cont_loss
            train_loss.update(loss.item())

            if iter % 50 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t total_loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                            'sup_loss: {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                            'cont_loss: {cont_loss.val:.4f} ({cont_loss.avg:.4f})'.format(epoch, iter,
                                                                                          len(train_loader_ulb),
                                                                                          loss=train_loss,
                                                                                          sup_loss=supervised_loss,
                                                                                          cont_loss=contrastive_loss))
        else:
            sum_high = 0
            for label_idx in range(probs_ulb.size(1)):
                idx_high = probs_ulb[:, label_idx].sort(descending=True)[1][:int(len(probs_ulb) / args.num_classes)]
                sum_high += probs_ulb[idx_high, label_idx].sum().item()
            selected_num = min(
                int(sum_high / len(probs_ulb) * len(partial_y_ulb) / partial_y_ulb.size(1)),
                int(len(partial_y_ulb) / partial_y_ulb.size(1)))
            selected_num = max(selected_num, 1)

            with torch.no_grad():
                pseudo_idx_high = torch.tensor([], dtype=torch.int64).cuda()
                pseudo_label_high = torch.tensor([], dtype=torch.int64).cuda()
                pseudo_max_probs_high = torch.tensor([]).cuda()
                pseudo_gt_high = torch.tensor([], dtype=torch.int64).cuda()
                for label_idx in range(probs_ulb.size(1)):
                    per_label_mask = probs_ulb[:, label_idx].sort(descending=True)[1][:selected_num]
                    pseudo_idx_high = torch.cat((pseudo_idx_high, per_label_mask))
                    pseudo_label_high = torch.cat((pseudo_label_high, torch.tensor([label_idx] * selected_num).cuda()))
                    probs_class = probs_ulb[per_label_mask, label_idx]
                    pseudo_max_probs_high = torch.cat((pseudo_max_probs_high, probs_class))
                    pseudo_gt_high = torch.cat((pseudo_gt_high, y_ulb[per_label_mask]))

            hard_cons_loss = criterion(logits_ulb_strong[pseudo_idx_high, :], pseudo_label_high.long())
            hard_consistent_loss.update(hard_cons_loss.item(), len(pseudo_idx_high))

            # 统计每轮的样本
            if iter == 0 and args.dataset != 'stl10':
                sta_label = torch.bincount(pseudo_label_high, minlength=args.num_classes)
                sta_gt = torch.bincount(pseudo_gt_high, minlength=args.num_classes)
                ratio = min(sta_label) / max(sta_label)

                confusion_matrix = torch.zeros((args.num_classes, args.num_classes)).cuda()
                for i in range(len(pseudo_label_high)):
                    confusion_matrix[pseudo_gt_high[i].cpu(), pseudo_label_high[i].cpu()] += 1
                confusion_matrix = confusion_matrix / (confusion_matrix.sum(dim=1) + 1e-10).repeat(
                    confusion_matrix.size(1),
                    1).transpose(0, 1)

                label_high.append(sta_label.cpu().tolist())
                gt_high.append(sta_gt.cpu().tolist())
                acc_high.append(torch.diagonal(confusion_matrix).cpu().tolist())

            # Beta分布
            beta_distribution = torch.distributions.Beta(args.alpha_mixup, args.alpha_mixup)
            mixup_weight = beta_distribution.sample()
            mixup_weight = torch.max(mixup_weight, 1 - mixup_weight)
            mixup_idx = torch.randperm(len(pseudo_idx_high))[:len(probs_lb)]

            if len(mixup_idx) >= len(x_weak):
                mixup_x = mixup_weight * x_weak + (1 - mixup_weight) * x_weak_ulb[pseudo_idx_high[mixup_idx], :]
                mixup_y = mixup_weight * confidence[index] + (1 - mixup_weight) * F.one_hot(
                    pseudo_label_high[mixup_idx], num_classes=args.num_classes)
            else:
                mixup_x = mixup_weight * x_weak[:len(mixup_idx), :] + (1 - mixup_weight) * x_weak_ulb[
                                                                                           pseudo_idx_high[mixup_idx],
                                                                                           :]
                mixup_y = mixup_weight * confidence[index[:len(mixup_idx)]] + (1 - mixup_weight) * F.one_hot(
                    pseudo_label_high[mixup_idx], num_classes=args.num_classes)

            mixup_logits, _ = model.encoder(mixup_x)
            mixup_probs = F.softmax(mixup_logits, dim=-1)

            # supervised loss
            sup_loss_1 = confidence[index] * torch.log(probs_lb + 1e-10)
            sup_loss_1 = (-torch.sum(sup_loss_1)) / sup_loss_1.size(0)
            supervised_loss.update(sup_loss_1.item(), len(probs_lb))

            sup_loss_2 = mixup_y * torch.log(mixup_probs + 1e-10)
            sup_loss_2 = (-torch.sum(sup_loss_2)) / sup_loss_2.size(0)
            supervised_loss.update(sup_loss_2.item(), len(mixup_probs))

            sup_loss = sup_loss_1 + args.beta_m * sup_loss_2

            loss = sup_loss + args.lambda_u * hard_cons_loss
            train_loss.update(loss.item())

            if iter % 50 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t total_loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                            'sup_loss: {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                            'hard_cons_loss: {hard_cons_loss.val:.4f} ({hard_cons_loss.avg:.4f})\t'
                            'cont_loss: {cont_loss.val:.4f} ({cont_loss.avg:.4f})'.format(
                    epoch, iter, len(train_loader_ulb), loss=train_loss, sup_loss=supervised_loss,
                    hard_cons_loss=hard_consistent_loss, cont_loss=contrastive_loss))

        with torch.no_grad():
            revisedY = partial_y.clone()
            revisedY[revisedY > 0] = 1
            revisedY = revisedY * probs_lb
            confidence[index, :] = revisedY / (revisedY.sum(dim=1) + 1e-10).repeat(revisedY.size(1), 1).transpose(0, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    disa_acc(train_loader_lb, confidence, args.save_path, epoch)


def my_test(args, test_loader, model, criterion, epoch, find_knn=False):
    losses = AverageMeter()
    top1 = AverageMeter()

    feat_all = torch.tensor([]).cuda()
    label_all = torch.tensor([]).cuda()
    pred_all = torch.tensor([]).cuda()

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()

            pred, feat = model(x, eval_only=True)
            test_loss = criterion(pred, y)

            label_all = torch.cat([label_all, y])

            pred_label = torch.max(pred, dim=1)[1]
            pred_all = torch.cat([pred_all, pred_label])

            if find_knn:
                feat_all = torch.cat((feat_all, feat))

            # measure accuracy and record loss
            prec1 = accuracy(pred.data, y)[0]
            losses.update(test_loss.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))

            if i % 100 == 0:
                args.logger.info('Test: [{0}/{1}]\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), loss=losses, top1=top1))

    return top1.avg, losses.avg


def main(args):
    data_dir = os.path.join(args.data_dir, args.dataset.lower())
    logger = args.logger

    # load data
    if args.dataset == "cifar10":
        args.num_classes = 10
        train_loader_lb, train_loader_ulb, test_loader = dataset.cifar10_dataloaders(data_dir, args)
    elif args.dataset == "cifar100":
        args.num_classes = 100
        train_loader_lb, train_loader_ulb, test_loader = dataset.cifar100_dataloaders(data_dir, args)
    elif args.dataset == "stl10":
        args.num_classes = 10
        train_loader_lb, train_loader_ulb, test_loader = dataset.stl10_dataloaders(data_dir, args)
    else:
        raise Exception("Missing function to handle this dataset")

    model = mymethod(args, ResNet18).cuda()

    # statistic
    label_high = []
    gt_high = []
    acc_high = []

    # init confidence
    confidence = copy.deepcopy(train_loader_lb.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]
    confidence = torch.tensor(confidence).cuda()
    confidence_ulb = copy.deepcopy(train_loader_ulb.dataset.partial_labels)
    confidence_ulb = confidence_ulb / confidence_ulb.sum(axis=1)[:, None]
    confidence_ulb = confidence_ulb.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    loss_cont_fn = SupConLoss(args)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # start epoch
    start_epoch = 0
    best_acc = 0
    best_epoch = 0

    # distribution
    args.lb_distribution = torch.tensor([1 / args.num_classes] * args.num_classes).cuda()
    args.ulb_distribution = torch.tensor([1 / args.num_classes] * args.num_classes).cuda()

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            checkpoint = torch.load(args.load_path, map_location="cpu")
            start_epoch = checkpoint["save_epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            label_high = checkpoint["label_high"]
            gt_high = checkpoint["gt_high"]
            acc_high = checkpoint["acc_high"]
            confidence = checkpoint["confidence"].cuda()
            best_acc = checkpoint["best_acc"]
            best_epoch = checkpoint["best_epoch"]
            args.lb_distribution = checkpoint["lb_distribution"].cuda()
            args.ulb_distribution = checkpoint["ulb_distribution"].cuda()
            logger.info(f'Resume existing model!')
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    logger.info('Model training')
    for epoch in range(start_epoch, args.epoch):
        is_best = False
        my_train(args, train_loader_lb, train_loader_ulb, model, optimizer, epoch, criterion, confidence, loss_cont_fn,
                 label_high, gt_high, acc_high)

        adjust_learning_rate(args, optimizer, epoch)

        logger.info('confidence sum: {}'.format(np.array_str(confidence.sum(dim=0).cpu().numpy()).replace('\n', '')))

        valacc, valloss = my_test(args, test_loader, model, criterion, epoch, find_knn=False)

        if valacc > best_acc:
            best_acc = valacc
            is_best = True
            best_epoch = epoch
        logger.info(f'Epoch {epoch} Val Acc: {valacc}% \t Best Val Acc: {best_acc}% on Epoch {best_epoch}\n')
        # save latest and best model
        if epoch != args.warm_up - 1:
            best_model_name = '{}/model_best.pth'.format(args.save_path)
        else:
            best_model_name = '{}/model_best_20e.pth'.format(args.save_path)
        save_checkpoint({
            'epoch': epoch + 1,
            'backbone': args.backbone,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'confidence': confidence,
            'label_high': label_high,
            'gt_high': gt_high,
            'acc_high': acc_high,
            'save_epoch': epoch,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'lb_distribution': args.lb_distribution,
            'ulb_distribution': args.ulb_distribution,
        }, is_best=is_best, filename='{}/latest_model.pth'.format(args.save_path),
            best_file_name=best_model_name)


if __name__ == "__main__":
    args = get_config()
    main(args)
