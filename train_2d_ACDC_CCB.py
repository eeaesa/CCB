import argparse
import logging
import os
import pprint
import shutil
import sys
from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.acdc import  ACDCDataSets, WeakStrongAugment_CCB

from torchvision import transforms
import random
import torch.nn.functional as F
from util import CCB_utils, losses

from networks.net_factory import net_factory
from util.utils import count_params, test_single_volume, compute_confidence_interval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str,
                    default="configs/ACDC.yaml")
# model name and save path
parser.add_argument('--method', default="CCB", type=str,
                    help='method name')
parser.add_argument('--model', type=str, default='unet_CCB',
                    help='net')
# label rat
parser.add_argument('--labeled-num', default='7', type=str)
# CCB arams
parser.add_argument('--temp', type=float, default=0.07)
parser.add_argument('--num_queries', type=int, default=256)
parser.add_argument('--num_negatives', type=int, default=512)
parser.add_argument('--cl-size', type=int, default=64, help='cl_size')
# seed
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)


def getLog(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/logging.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[patiens_num]


def get_current_consistency_weight(epoch):
    consistency = cfg['semi']['consistency']
    consistency_rampup = cfg['semi']['consistency_rampup']
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, model_teacher, ema_decay, iters):
    ema_decay = min(1 - 1 / (iters + 1), ema_decay, )
    # update weight
    for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
        param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
    # update bn
    for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
        buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def main(args):
    # save model and logging
    exp = "exp/{}/{}".format(
        cfg['data']['dataset'], args.method)

    lb_rat = args.labeled_num
    snapshot_path = "{}/{}/{}_labeled".format(
        exp, args.model, lb_rat)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    current_file = __file__  # 获取当前 train.py 文件的路径
    # 复制当前运行的 train.py 文件到 snapshot_path
    shutil.copyfile(current_file, os.path.join(snapshot_path, os.path.basename(current_file)))

    getLog(args, snapshot_path)  # 日志保存
    # 将 cfg 打印到日志中
    logging.info("Configuration settings:\n%s", yaml.dump(cfg))

    evens_path = snapshot_path + '/log'
    if not os.path.exists(evens_path):
        os.makedirs(evens_path)
    writer = SummaryWriter(evens_path)

    dataset = cfg['data']['dataset']
    root_path = cfg['data']['root_path']
    crop_size = cfg['data']['crop_size']
    in_ch = cfg['data']['in_chns']

    # train params
    base_lr = cfg['train']['base_lr']
    num_classes = cfg['train']['num_classes']
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']

    # semi params
    ema_decay = cfg['semi']['ema_decay']
    conf_thresh = cfg['semi']['conf_thresh']

    labeled_id_path = "splits/acdc/{}/labeled.txt".format(
        args.labeled_num)
    unlabeled_id_path = "splits/acdc/{}/unlabeled.txt".format(
        args.labeled_num)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    model = net_factory(net_type=args.model, in_chns=in_ch,
                        class_num=num_classes)

    optimizer = SGD(model.parameters(), lr=base_lr,
                    momentum=0.9, weight_decay=0.0001)

    logging.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.cuda()

    # teacher model initial
    model_teacher = deepcopy(model)
    model_teacher.cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data

    trainset_u = ACDCDataSets(base_dir=root_path,
                              split="train_u",
                              id_path=unlabeled_id_path,
                              transform=transforms.Compose([
                                  WeakStrongAugment_CCB(crop_size)
                              ]
                              ))

    trainset_l = ACDCDataSets(base_dir=root_path,
                              split="train_l",
                              id_path=labeled_id_path,
                              nsample=len(trainset_u.ids),
                              transform=transforms.Compose([
                                  WeakStrongAugment_CCB(crop_size)
                              ]
                              ))

    valset = ACDCDataSets(base_dir=root_path,
                          split="val",
                          )

    labeled_slice = patients_to_slices(root_path, args.labeled_num)
    unlabeled_slice = len(trainset_u)
    total_slice = labeled_slice + unlabeled_slice
    logging.info('Total silices is: {}, labeled silices is: {}\n'.format(
        total_slice, labeled_slice)
    )

    trainloader_l = DataLoader(trainset_l, batch_size=batch_size,
                               pin_memory=True, num_workers=1, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=batch_size,
                               pin_memory=True, num_workers=1, drop_last=True)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    total_iters = len(trainloader_u) * epochs
    logging.info('Total iters is: {}\n'.format(total_iters))
    previous_best_mdice, previous_best_mhd95, previous_best_acc = 0.0, 0.0, 0.0
    epoch = -1
    iters = 0

    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_mdice = checkpoint['previous_best_mdice']
        previous_best_mhd95 = checkpoint['previous_best_mhd95']
        previous_best_acc = checkpoint['previous_best_acc']

        logging.info('************ Load from checkpoint at epoch %i\n' % epoch)

    celoss_l = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
    celoss_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
    dice_loss = losses.DiceLoss(n_classes=num_classes).cuda()


    get_mask = CCB_utils.Class_Aware_Threshold(num_classes=num_classes, momentum=0.99)

    proj_head = CCB_utils.ProjectionHead(
        num_input_channels=16,
        num_projection_channels=256,
    ).cuda()
    for epoch in range(epoch + 1, epochs):
        logging.info(
            '===========> Epoch: {:}, LR: {:.5f}, \033[31m Previous best {} mdice: {:.2f}, mhd95: {:.2f}, Overall Accuracy: {:.2f}\033[0m'.format(
                epoch, optimizer.param_groups[0]['lr'], dataset, previous_best_mdice, previous_best_mhd95,
                previous_best_acc))

        loader = zip(trainloader_l, trainloader_u)

        for i, ((sample_l), (sample_u)) in enumerate(loader):

            img_x, mask_x = sample_l["image"], sample_l["label"]
            img_x, mask_x = img_x.cuda(), mask_x.cuda()

            img_u_w, img_u_s = sample_u["image_weak"], sample_u["image_strong"]
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            with torch.no_grad():
                model_teacher.eval()
                pred_u_w, feat_u_w = model_teacher(img_u_w.detach())
                pred_u_w_soft = F.softmax(pred_u_w, dim=1)

            model.train()

            # ----------CGC------------
            x_mix, label_mix, logits_mix = (CCB_utils.
            Confidence_Guided_Cutmix(
                img_u_s, pred_u_w_soft
            ))
            label_mix = label_mix.long()

            preds, feats = model(torch.cat((img_x, x_mix)))
            preds_x = preds[:num_lb]
            preds_x_soft = torch.softmax(preds_x, dim=1)

            pre_mix = preds[num_lb:]

            feats = F.interpolate(feats, size=(args.cl_size, args.cl_size), mode='nearest')

            # supevised loss
            entry = celoss_l(preds_x, mask_x.long())
            loss_ce = entry.mean()
            loss_dice = dice_loss(preds_x_soft, mask_x.long().unsqueeze(1).float(), ignore=255)
            loss_x = 0.5 * (loss_ce + loss_dice)

            #----------CAT------------
            mask_mix, mask_ratio_mix, tau_mix = get_mask.masking(logits_mix, label_mix, iters)
            loss_cat = F.cross_entropy(pre_mix, label_mix, ignore_index=255, reduction='none') * mask_mix
            loss_cat = loss_cat.mean()

            #----------BCO------------
            cls_thresholds = get_mask.prob_conf[label_mix]
            mask_u_weak = logits_mix.ge(cls_thresholds).to(logits_mix.dtype)
            mask_u_weak = F.interpolate(mask_u_weak.unsqueeze(dim=1),
                                        size=feats.shape[2:], mode='nearest')
            valid_pix_u = F.interpolate(CCB_utils.label_one_hot_encoder(label_mix.unsqueeze(1), num_classes),
                                        size=feats.shape[2:], mode='nearest')

            valid_pix_u = valid_pix_u * mask_u_weak 
            valid_pix_l = F.interpolate(CCB_utils.label_one_hot_encoder(mask_x.unsqueeze(1), num_classes),
                                        size=feats.shape[2:], mode='nearest')
            valid_pix_all = torch.cat((valid_pix_l, valid_pix_u))
            # 2、get preds = [2B, C, H, W]
            preds_s = torch.cat((preds_x, pre_mix))
            preds_s = F.interpolate(preds_s, size=feats.shape[2:], mode='nearest')
            # 3、get feats SA = [2B, D, H, W]
            reps_s = proj_head(feats)
            loss_intra, loss_inter = CCB_utils.BoundaryClusteringOptimization(
                reps_s, preds_s, valid_pix_all, temp=args.temp,
                num_queries=args.num_queries, num_negatives=args.num_negatives
            )

            consistency_weight = get_current_consistency_weight(iters // 150)

            loss_consis = loss_cat
            loss_bco = (loss_intra + loss_inter) * 0.5

            loss = loss_x + loss_consis * consistency_weight + loss_bco

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = base_lr * (1 - iters / total_iters) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # update teacher model with EMA
            with torch.no_grad():
                update_ema_variables(model, model_teacher, ema_decay, iters)

            iters = epoch * len(trainloader_u) + i

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_consis', loss_consis.item(), iters)
            writer.add_scalar('train/loss_cat', loss_cat.item(), iters)
            writer.add_scalar('train/tau_mix', tau_mix.item(), iters)
            writer.add_scalar('train/mask_ratio_mix', mask_ratio_mix.item(), iters)
            writer.add_scalar('train/loss_bco', loss_bco.item(), iters)
            writer.add_scalar('train/loss_intra', loss_intra.item(), iters)
            writer.add_scalar('train/loss_inter', loss_inter.item(), iters)

            if (i % (len(trainloader_u) // 8) == 0):
                logging.info(
                    'Iters: %d, Total loss: %f, Loss x: %f, Loss x ce: %f, Loss x dice: %f, '
                    'Loss consis: %f, Loss mix: %f, tau_mix: %f, mask ratio mix: %f, '
                    'Loss bco: %f , Loss intra: %f , Loss inter: %f ' % (
                        iters, loss.item(), loss_x.item(), loss_ce.item(), loss_dice.item(),
                        loss_consis.item(), loss_cat.item(), tau_mix.item(), mask_ratio_mix.item(),
                        loss_bco.item(), loss_intra.item(), loss_inter.item(),
                    ))


        model.eval()
        metric_list = 0.0
        pa_list = []
        mdice_list = []
        mhd95_list = []
        mjaccard_list = []
        masd_list = []

        for i, sampled_batch in enumerate(valloader):
            metric_i, PA = test_single_volume(
                sampled_batch["image"], sampled_batch["label"],
                model,
                classes=cfg['train']['num_classes'],
                patch_size=cfg['data']['crop_size']
            )

            mdice = [row[0] for row in metric_i]
            mhd95 = [row[1] for row in metric_i]
            mjaccard = [row[2] for row in metric_i]
            masd = [row[3] for row in metric_i]

            metric_list += np.array(metric_i)
            mdice_list.append(np.mean(mdice))
            mhd95_list.append(np.mean(mhd95))
            mjaccard_list.append(np.mean(mjaccard))
            masd_list.append(np.mean(masd))
            pa_list.append(PA)

        metric_list = metric_list / len(valset)

        dice_list = [row[0] for row in metric_list]
        hd95_list = [row[1] for row in metric_list]
        jaccard_list = [row[2] for row in metric_list]
        asd_list = [row[3] for row in metric_list]

        m_dice = np.nanmean(dice_list)
        m_jaccard = np.nanmean(jaccard_list)
        m_hd95 = np.nanmean(hd95_list)
        m_asd = np.nanmean(asd_list)
        m_pa = np.nanmean(pa_list)

        # 添加TensorBoard记录
        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                              metric_list[class_i, 0], iters)
            writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                              metric_list[class_i, 1], iters)
            writer.add_scalar('info/val_{}_jaccard'.format(class_i + 1),
                              metric_list[class_i, 2], iters)
            writer.add_scalar('info/val_{}_asd'.format(class_i + 1),
                              metric_list[class_i, 3], iters)

        writer.add_scalar('info/val_mean_dice', m_dice * 100.0, iters)
        writer.add_scalar('info/val_mean_hd95', m_hd95, iters)
        writer.add_scalar('info/val_mean_jaccard', m_jaccard * 100.0, iters)
        writer.add_scalar('info/val_mean_asd', m_asd, iters)
        writer.add_scalar('info/val_pa', m_pa * 100.0, iters)

        is_best = m_dice * 100.0 > previous_best_mdice
        previous_best_mdice = max(m_dice * 100.0, previous_best_mdice)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_mdice': previous_best_mdice,
            'previous_best_mhd95': previous_best_mhd95,
            'previous_best_acc': previous_best_acc * 100.0,
        }

        if is_best:
            logging.info('***** \033[31m best eval! \033[0m *****')
            previous_best_mhd95 = m_hd95
            previous_best_acc = m_pa * 100.0
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_mDSC_{}_mJac_{}_mHD95_{}.pth'.format(
                                              epoch, round(previous_best_mdice, 2),
                                              round(m_jaccard * 100.0, 2),
                                              round(previous_best_mhd95, 2)))
            torch.save(checkpoint, save_mode_path)
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        logging.info('***** \033[33m eval! \033[0m *****')


        for (cls_idx, dice) in enumerate(dice_list):
            logging.info('***** Evaluation ***** >>>> Class [{:}] Dice: {:.2f}'.format(
                cls_idx, dice * 100.0))
        logging.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(m_dice * 100.0))

        for (cls_idx, jaccard) in enumerate(jaccard_list):
            logging.info('***** Evaluation ***** >>>> Class [{:}] Jaccard: {:.2f}'.format(
                cls_idx, jaccard * 100.0))
        logging.info('***** Evaluation ***** >>>> MeanJaccard: {:.2f}\n'.format(m_jaccard * 100.0))

        for (cls_idx, hd95) in enumerate(hd95_list):
            logging.info('***** Evaluation ***** >>>> Class [{:}] hd95: {:.2f}'.format(
                cls_idx, hd95))
        logging.info('***** Evaluation ***** >>>> MeanHd95: {:.2f}\n'.format(m_hd95))

        for (cls_idx, asd) in enumerate(asd_list):
            logging.info('***** Evaluation ***** >>>> Class [{:}] ASD: {:.2f}'.format(
                cls_idx, asd))
        logging.info('***** Evaluation ***** >>>> MeanASD: {:.2f}\n'.format(m_asd))

        logging.info('***** Evaluation ***** >>>> mPA: {:.2f}\n'.format(m_pa * 100.0))

        dsc, std_dsc, ci_lower_dsc, ci_upper_dsc = compute_confidence_interval(mdice_list)
        jac, std_jac, ci_lower_jac, ci_upper_jac = compute_confidence_interval(mjaccard_list)
        hd95, std_hd95, ci_lower_hd95, ci_upper_hd95 = compute_confidence_interval(mhd95_list)
        asd, std_asd, ci_lower_asd, ci_upper_asd = compute_confidence_interval(masd_list)
        pa, std_pa, ci_lower_pa, ci_upper_pa = compute_confidence_interval(pa_list)

        logging.info(f"dsc: {dsc * 100.0:.2f}, dsc_std: ({std_dsc:.2f}), "
                     f"95% CI: ({ci_lower_dsc * 100.0:.2f}, {ci_upper_dsc * 100.0:.2f})")
        logging.info(f"jac: {jac * 100.0:.2f}, jac_std: ({std_jac:.2f}), "
                     f"95% CI: ({ci_lower_jac * 100.0:.2f}, {ci_upper_jac * 100.0:.2f})")
        logging.info(f"hd95: {hd95:.2f}, hd95_std: ({std_hd95:.2f}), "
                     f"95% CI: ({ci_lower_hd95:.2f}, {ci_upper_hd95:.2f})")
        logging.info(f"asd: {asd:.2f}, asd_std: ({std_asd:.2f}), "
                     f"95% CI: ({ci_lower_asd:.2f}, {ci_upper_asd:.2f})")
        logging.info(f"pa: {pa * 100.0:.2f}, PA_std: ({std_pa:.2f}), "
                     f"95% CI: ({ci_lower_pa * 100.0:.2f}, {ci_upper_pa * 100.0:.2f})")


if __name__ == '__main__':

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
