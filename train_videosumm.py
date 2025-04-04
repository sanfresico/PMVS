import logging
import time
import os
import numpy as np
from scipy.interpolate import interp1d  # 插值库
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from models import *
from losses import *
from datasets import *
from utils import *

from helpers.bbox_helper import nms
from helpers.vsumm_helper import bbox2summary, get_summ_f1score
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger()

def train_videosumm(args, split, split_idx):
    batch_time = AverageMeter('time')
    data_time = AverageMeter('time')

    model = Model_VideoSumm(args=args)
    model = model.to(args.device)
    calc_contrastive_loss = Dual_Contrastive_Loss().to(args.device)

    parameters = [p for p in model.parameters() if p.requires_grad] + \
                 [p for p in calc_contrastive_loss.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs('{}/checkpoint'.format(args.model_dir), exist_ok=True)

    max_train_fscore = -1
    max_val_fscore = -1
    best_val_epoch = 0

    # model testing, load from checkpoint
    checkpoint_path = None
    if args.checkpoint and args.test:
        checkpoint_path = '{}/model_best_split{}.pt'.format(args.checkpoint, split_idx)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("load checkpoint from {}".format(checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])

    train_set = VideoSummDataset(keys=split['train_keys'], args=args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=False, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    val_set = VideoSummDataset(keys=split['test_keys'], args=args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    if args.test:
        val_fscore = evaluate_videosumm(model, val_loader, args, epoch=0)
        logger.info(f'F-score: {val_fscore:.4f}')
        return val_fscore, best_val_epoch, max_train_fscore

    logger.info('\n' + str(model))

    for epoch in range(args.start_epoch, args.max_epoch):
        model.train()
        stats = AverageMeter('loss', 'cls_loss', 'loc_loss', 'ctr_loss', 'inter_contrastive_loss',
                             'intra_contrastive_loss')
        data_length = len(train_loader)
        end = time.time()
        for k, (video_name, video_list, text_list, mask_video_list, mask_text_list, \
                video_cls_label_list, video_loc_label_list, video_ctr_label_list, \
                text_cls_label_list, text_loc_label_list, text_ctr_label_list, \
                user_summary_list, n_frames_list, ratio_list, n_frame_per_seg_list, picks_list, change_points_list, \
                video_to_text_mask_list, text_to_video_mask_list) in enumerate(train_loader):
            data_time.update(time=time.time() - end)

            batch_size = len(video_list)

            video = pad_sequence(video_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)

            mask_video = pad_sequence(mask_video_list, batch_first=True)
            mask_text = pad_sequence(mask_text_list, batch_first=True)

            video_cls_label = pad_sequence(video_cls_label_list, batch_first=True)
            video_loc_label = pad_sequence(video_loc_label_list, batch_first=True)
            video_ctr_label = pad_sequence(video_ctr_label_list, batch_first=True)

            text_cls_label = pad_sequence(text_cls_label_list, batch_first=True)
            text_loc_label = pad_sequence(text_loc_label_list, batch_first=True)
            text_ctr_label = pad_sequence(text_ctr_label_list, batch_first=True)

            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

            video, text = video.to(args.device), text.to(args.device)
            mask_video, mask_text = mask_video.to(args.device), mask_text.to(args.device)

            video_cls_label = video_cls_label.to(args.device)  # [B, T]
            video_loc_label = video_loc_label.to(args.device)  # [B, T, 2]
            video_ctr_label = video_ctr_label.to(args.device)  # [B, T]

            text_cls_label = text_cls_label.to(args.device)  # [B, T]
            text_loc_label = text_loc_label.to(args.device)  # [B, T, 2]
            text_ctr_label = text_ctr_label.to(args.device)  # [B, T]

            video_pred_cls, video_pred_loc, video_pred_ctr, text_pred_cls, text_pred_loc, text_pred_ctr, contrastive_pairs = \
                model(video=video, text=text, mask_video=mask_video, mask_text=mask_text,
                      video_label=video_cls_label, text_label=text_cls_label,
                      video_to_text_mask_list=video_to_text_mask_list, text_to_video_mask_list=text_to_video_mask_list)

            cls_loss = calc_cls_loss(video_pred_cls, video_cls_label.to(torch.long), mask=mask_video) + \
                       calc_cls_loss(text_pred_cls, text_cls_label.to(torch.long), mask=mask_text)

            loc_loss = calc_loc_loss(video_pred_loc, video_loc_label, video_cls_label) + \
                       calc_loc_loss(text_pred_loc, text_loc_label, text_cls_label)

            ctr_loss = calc_ctr_loss(video_pred_ctr, video_ctr_label, video_cls_label) + \
                       calc_ctr_loss(text_pred_ctr, text_ctr_label, text_cls_label)

            inter_contrastive_loss, intra_contrastive_loss = calc_contrastive_loss(contrastive_pairs)
            inter_contrastive_loss = inter_contrastive_loss * args.lambda_contrastive_inter
            intra_contrastive_loss = intra_contrastive_loss * args.lambda_contrastive_intra

            loss = cls_loss + loc_loss + ctr_loss + inter_contrastive_loss + intra_contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item(), ctr_loss=ctr_loss.item(),
                         inter_contrastive_loss=inter_contrastive_loss.item(),
                         intra_contrastive_loss=intra_contrastive_loss.item()
                         )

            batch_time.update(time=time.time() - end)
            end = time.time()

            if (k + 1) % args.print_freq == 0:
                logger.info(f'[Train] Epoch: {epoch + 1}/{args.max_epoch} Iter: {k + 1}/{data_length} '
                            f'Time: {batch_time.time:.3f} Data: {data_time.time:.3f} '
                            f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.ctr_loss:.4f}/{stats.inter_contrastive_loss:.4f}/{stats.intra_contrastive_loss:.4f}/{stats.loss:.4f}')

        save_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'max_val_fscore': max_val_fscore,
            'max_train_fscore': max_train_fscore,
        }

        if (epoch + 1) % args.eval_freq == 0:
            # 修改评估调用以接收多个指标
            eval_metrics = evaluate_videosumm(model, val_loader, args, epoch=epoch)
            val_fscore = eval_metrics['fscore']
            val_kendall = eval_metrics['kendall']
            val_spearman = eval_metrics['spearman']

            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                best_val_epoch = epoch + 1
                torch.save(save_checkpoint, '{}/checkpoint/model_best_split{}.pt'.format(args.model_dir, split_idx))

            # 记录Kendall和Spearman到日志
            logger.info(f'[Eval]  Epoch: {epoch + 1}/{args.max_epoch} F-score: {val_fscore:.4f}/{max_val_fscore:.4f} '
                        f'Kendall: {val_kendall:.4f} Spearman: {val_spearman:.4f}\n\n')
            # logger.info(f'[Eval]  Epoch: {epoch + 1}/{args.max_epoch} '
            #             f'F-score: {val_fscore:.4f}/{max_val_fscore:.4f} '
            #             f'Kendall: {val_kendall:.4f} Spearman: {val_spearman:.4f}\n\n')
            # 记录到TensorBoard
            args.writer.add_scalar(f'Split{split_idx}/Val/max_fscore', max_val_fscore, epoch + 1)
            args.writer.add_scalar(f'Split{split_idx}/Val/fscore', val_fscore, epoch + 1)
            args.writer.add_scalar(f'Split{split_idx}/Val/kendall', val_kendall, epoch + 1)
            args.writer.add_scalar(f'Split{split_idx}/Val/spearman', val_spearman, epoch + 1)

        args.writer.add_scalar(f'Split{split_idx}/Train/loss', stats.loss, epoch + 1)
        args.writer.add_scalar(f'Split{split_idx}/Train/cls_loss', stats.cls_loss, epoch + 1)
        args.writer.add_scalar(f'Split{split_idx}/Train/loc_loss', stats.loc_loss, epoch + 1)
        args.writer.add_scalar(f'Split{split_idx}/Train/ctr_loss', stats.ctr_loss, epoch + 1)
        args.writer.add_scalar(f'Split{split_idx}/Train/inter_contrastive_loss', stats.inter_contrastive_loss, epoch + 1)
        args.writer.add_scalar(f'Split{split_idx}/Train/intra_contrastive_loss', stats.intra_contrastive_loss, epoch + 1)


    return max_val_fscore, best_val_epoch, max_train_fscore


@torch.no_grad()
def evaluate_videosumm(model, val_loader, args, epoch=None):
    model.eval()
    # 初始化统计指标
    stats = AverageMeter('fscore', 'kendall', 'spearman')
    all_pred_scores = []  # 存储所有预测分数（用于全局Kendall/Spearman）
    all_gt_labels = []  # 存储所有真实标签（用于全局Kendall/Spearman）
    data_length = len(val_loader)

    with torch.no_grad():
        for k, (video_name, video_list, text_list, mask_video_list, mask_text_list, \
                video_cls_label_list, video_loc_label_list, video_ctr_label_list, \
                text_cls_label_list, text_loc_label_list, text_ctr_label_list, \
                user_summary_list, n_frames_list, ratio_list, n_frame_per_seg_list, picks_list, change_points_list, \
                video_to_text_mask_list, text_to_video_mask_list) in enumerate(val_loader):
            batch_size = len(video_list)
            print(video_name)
            video = pad_sequence(video_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)

            mask_video = pad_sequence(mask_video_list, batch_first=True)
            mask_text = pad_sequence(mask_text_list, batch_first=True)

            video_cls_label = pad_sequence(video_cls_label_list, batch_first=True)
            video_loc_label = pad_sequence(video_loc_label_list, batch_first=True)
            video_ctr_label = pad_sequence(video_ctr_label_list, batch_first=True)

            text_cls_label = pad_sequence(text_cls_label_list, batch_first=True)
            text_loc_label = pad_sequence(text_loc_label_list, batch_first=True)
            text_ctr_label = pad_sequence(text_ctr_label_list, batch_first=True)
            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

            video, text = video.to(args.device), text.to(args.device)
            mask_video, mask_text = mask_video.to(args.device), mask_text.to(args.device)

            video_cls_label = video_cls_label.to(args.device) #[B, T]
            video_loc_label = video_loc_label.to(args.device) #[B, T, 2]
            video_ctr_label = video_ctr_label.to(args.device) #[B, T]

            text_cls_label = text_cls_label.to(args.device) #[B, T]
            text_loc_label = text_loc_label.to(args.device) #[B, T, 2]
            text_ctr_label = text_ctr_label.to(args.device) #[B, T]

            pred_cls_batch, pred_bboxes_batch = model.predict(video=video, text=text,
                                                            mask_video=mask_video, mask_text=mask_text,
                                                            video_label=video_cls_label, text_label=text_cls_label,
                                                            video_to_text_mask_list=video_to_text_mask_list,
                                                            text_to_video_mask_list=text_to_video_mask_list) #[B, T], [B, T, 2]
            mask_video_bool = mask_video.cpu().numpy().astype(bool)

        for i in range(batch_size):
            video_length = np.sum(mask_video_bool[i])
            pred_cls = pred_cls_batch[i, mask_video_bool[i]]  # [T]
            pred_bboxes = np.clip(pred_bboxes_batch[i, mask_video_bool[i]], 0, video_length).round().astype(
                np.int32)  # [T, 2]

            pred_cls, pred_bboxes = nms(pred_cls, pred_bboxes, args.nms_thresh)
            pred_summ, pred_summ_upsampled, pred_score, pred_score_upsampled = bbox2summary(
                video_length, pred_cls, pred_bboxes, change_points_list[i], n_frames_list[i], n_frame_per_seg_list[i],
                picks_list[i], proportion=ratio_list[i], seg_score_mode='mean')

            current_video_name = video_name[i]  # 获取当前视频名称
            print(video_name[i])
            if current_video_name == 'video_19':
                # 创建 DataFrame
                df = pd.DataFrame({
                    'frame': np.arange(len(pred_score_upsampled)),  # 帧序号
                    'score': pred_score_upsampled  # 预测分数
                })
                # 确定输出路径
                output_path = os.path.join('/tmp/pycharm_project_44/f_scores.xlsx')  # 假设保存到模型目录下的 results 文件夹
                df.to_excel(output_path, index=False)  # 写入 Excel
                logger.info(f'Saved frame scores for {current_video_name} to {output_path}')

            # eval_metric = 'max' if args.dataset == 'SumMe' else 'avg'
            # fscore = get_summ_f1score(pred_summ_upsampled, user_summary_list[i], eval_metric=eval_metric)
            eval_metric = 'max' if args.dataset == 'SumMe' else 'avg'
            fscore = get_summ_f1score(pred_summ_upsampled, user_summary_list[i], eval_metric=eval_metric)
            stats.update(fscore=fscore)
            user_summaries = user_summary_list[i]  # 假设形状为[num_users, n_frames]
            for user_summary in user_summaries:
                # 对齐长度（处理可能的填充）
                min_length = min(len(pred_score_upsampled), len(user_summary))
                aligned_pred = pred_score_upsampled[:min_length]
                aligned_gt = user_summary[:min_length]

                all_pred_scores.extend(aligned_pred.tolist())
                all_gt_labels.extend(aligned_gt.tolist())

            # 计算Kendall's tau和Spearman相关系数
            user_summaries = user_summary_list[i]  # 假设形状为[num_users, n_frames]
            kendall_taus = []
            spearman_corrs = []

            for user_summary in user_summaries:
                    # 确保预测和标签长度一致
                if len(pred_score_upsampled) != len(user_summary):
                        continue  # 或处理长度不匹配的情况

                # 计算指标
                kt, _ = kendalltau(pred_score_upsampled, user_summary)
                sc, _ = spearmanr(pred_score_upsampled, user_summary)

                # 处理NaN
                kt = 0.0 if np.isnan(kt) else kt
                sc = 0.0 if np.isnan(sc) else sc

                kendall_taus.append(kt)
                spearman_corrs.append(sc)

                # 根据eval_metric综合结果
            if eval_metric == 'max':
                kendall_val = max(kendall_taus) if kendall_taus else 0.0
                spearman_val = max(spearman_corrs) if spearman_corrs else 0.0
            else:  # 'avg'
                kendall_val = np.mean(kendall_taus) if kendall_taus else 0.0
                spearman_val = np.mean(spearman_corrs) if spearman_corrs else 0.0

            # 更新统计指标
            stats.update(fscore=fscore, kendall=kendall_val, spearman=spearman_val)
            if (k + 1) % args.print_freq == 0:
                logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} F-score: {stats.fscore:.4f}')
            if len(all_pred_scores) == 0 or len(all_gt_labels) == 0:
                kendall_val = 0.0
                spearman_val = 0.0
            else:
                kendall_val, _ = kendalltau(all_pred_scores, all_gt_labels)
                spearman_val, _ = spearmanr(all_pred_scores, all_gt_labels)
                kendall_val = 0.0 if np.isnan(kendall_val) else kendall_val
                spearman_val = 0.0 if np.isnan(spearman_val) else spearman_val


    # 返回所有指标
    return {
        'fscore': stats.fscore,
        'kendall': stats.kendall,
        'spearman': stats.spearman
    }