import logging
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from models import *
from losses import *
from datasets import *
from utils import *
from graphcreate import *
from rouge_score import rouge_scorer

logger = logging.getLogger()
all_keyframe_index_lists = {'keyframe':[]}

logger = logging.getLogger()
# 用于存储所有批次的关键帧索引
all_keyframe_index_lists = {'keyframe':[]}


def train_msmo(args):
    #设置初始化模型
    batch_time = AverageMeter('time')#跟踪批处理时间
    data_time = AverageMeter('time')#跟踪数据加载时间

    if args.dataset == 'BLiSS':
        model = Model_BLiSS(args=args)
    elif args.dataset in ['Daily_Mail', 'CNN']:
        model = Model_MSMO(args=args)

    model = model.to(args.device)
    calc_contrastive_loss = Dual_Contrastive_Loss().to(args.device)

    parameters = [p for p in model.parameters() if p.requires_grad] + \
                    [p for p in calc_contrastive_loss.parameters() if p.requires_grad]
    #优化器
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    #保存检查点
    os.makedirs('{}/checkpoint'.format(args.model_dir), exist_ok=True)
    #评分器
    args.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)

    max_train_R1 = max_train_R2 = max_train_RL = max_train_cos = 0
    max_val_R1 = max_val_R2 = max_val_RL = max_val_cos = 0
    best_val_epoch = 0

    if args.dataset in ['Daily_Mail', 'CNN']:
        dataset_name = 'MSMODataset'
    elif args.dataset in ['BLiSS']:
        dataset_name = 'BLiSSDataset'

    train_set = eval(dataset_name)(mode='train', args=args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, #num_workers=args.num_workers,args.batch_size
                                                drop_last=False, pin_memory=True,
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    val_set = eval(dataset_name)(mode='test', args=args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, #um_workers=args.num_workers,
                                                drop_last=False, pin_memory=True,
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)

    checkpoint_path = None
    if args.checkpoint and args.test:#加载最佳模型并进行评估
        checkpoint_path = '{}/model_best_text.pt'.format(args.checkpoint)
        print(f"load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        val_R1, val_R2, val_RL, _ = evaluate_msmo(model, val_loader, args, epoch=0)

        checkpoint_path = '{}/model_best_video.pt'.format(args.checkpoint)
        print(f"load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        _, _, _, val_cos = evaluate_msmo(model, val_loader, args, epoch=0)

        logger.info(f'R1: {val_R1:.4f} R2: {val_R2:.4f} RL: {val_RL:.4f} Cos: {val_cos:.4f}')
        return val_R1, val_R2, val_RL, val_cos, best_val_epoch, max_train_R1, max_train_R2, max_train_RL, max_train_cos

    logger.info('\n' + str(model))

    for epoch in range(args.start_epoch, args.max_epoch):#训练循环设置
        model.train()
        stats = AverageMeter('loss', 'text_loss', 'video_loss', 'inter_contrastive_loss', 'intra_contrastive_loss', 'R1', 'R2', 'RL', 'cos')

        data_length = len(train_loader)
        end = time.time()
        #数据加载和处理
        for k, (video_list, video_summ_list, text_list, \
                mask_video_list, mask_video_summ_list, mask_text_list, \
                video_label_list, text_label_list, article_segment_list, highlight_list, \
                video_to_text_mask_list, text_to_video_mask_list) in enumerate(train_loader):
            data_time.update(time=time.time() - end)

            batch_size = len(video_list)

            video = pad_sequence(video_list, batch_first=True)
            video_summ = pad_sequence(video_summ_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)


            mask_video = pad_sequence(mask_video_list, batch_first=True)
            mask_video_summ = pad_sequence(mask_video_summ_list, batch_first=True)
            mask_text = pad_sequence(mask_text_list, batch_first=True)

            video_label = pad_sequence(video_label_list, batch_first=True)
            text_label = pad_sequence(text_label_list, batch_first=True)

            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

            video, video_summ, text = video.to(args.device), video_summ.to(args.device), text.to(args.device)
            mask_video, mask_video_summ, mask_text = mask_video.to(args.device), mask_video_summ.to(args.device), mask_text.to(args.device)

            video_label = video_label.to(args.device) #[B, T]
            text_label = text_label.to(args.device) #[B, T]

            #模型前向传播
            pred_video, pred_text, contrastive_pairs = model(video=video, text=text, \
                                                             mask_video=mask_video, mask_text=mask_text, \
                                                             video_label=video_label, text_label=text_label, \
                                                             video_to_text_mask_list=video_to_text_mask_list, \
                                                             text_to_video_mask_list=text_to_video_mask_list)

            # 掩码转换和标签处理
            num_frame_selected = torch.sum(video_label, dim=-1)
            num_sentence_selected = torch.sum(text_label, dim=-1)

            mask_video_bool = mask_video.to(torch.bool)
            mask_video_summ_bool = mask_video_summ.to(torch.bool)
            mask_text_bool = mask_text.to(torch.bool)

            # 选择预测摘要，模型选择了每个批次中每个视频和文本的关键帧和关键句子。它使用模型的预测分数（pred_video, pred_text）来选择得分最高的帧和句子。
            keyframe_index_list = []
            keysentence_index_list = []
            for i in range(batch_size):
                valid_video_length = pred_video[i, mask_video_bool[i]].shape[0]
                k_video = min(30, valid_video_length)
                keyframe_index_list.append(
                    torch.topk(pred_video[i, mask_video_bool[i]], k=k_video)[1].tolist())
                valid_text_length = pred_text[i, mask_text_bool[i]].shape[0]
                k_text = min(2, valid_text_length)
                keysentence_index_list.append(
                    torch.topk(pred_text[i, mask_text_bool[i]], k=k_text)[1].tolist())
            # 将当前批次的关键帧索引添加到总列表中
            all_keyframe_index_lists['keyframe'].append('')
            # 将挑选出的关键帧进行构图，并通过GAT挑选出精细摘要
            keyframe_index_list.extend(build_edges_based_on_criteria(video,keyframe_index_list,video_summ))
            all_keyframe_index_lists['keyframe'].append(keyframe_index_list)

            # # 在训练循环结束后，保存关键帧索引到.npy文件
            # if (epoch + 1) % args.eval_freq == 0:
            #
            #     save_path = f'/tmp/pycharm_project_267/logs/BLiSS/keyframe_epoch_{epoch + 1}.npy'
            #
            #     # 使用os.makedirs创建目录
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #
            #     with open(save_path, 'wb') as f:
            #         pickle.dump(all_keyframe_index_lists['keyframe'], f)

            #这部分计算了不同类型的损失，然后将这些损失相加得到总损失（loss）。
            text_loss = calc_cls_loss(pred_text, text_label, mask=mask_text)
            if args.dataset in ['Daily_Mail', 'BLiSS']:
                video_loss = calc_cls_loss(pred_video, video_label, mask=mask_video)
            else:
                video_loss = torch.zeros(1).to(text_loss)

            inter_contrastive_loss, intra_contrastive_loss = calc_contrastive_loss(contrastive_pairs)

            inter_contrastive_loss = inter_contrastive_loss * args.lambda_contrastive_inter
            intra_contrastive_loss = intra_contrastive_loss * args.lambda_contrastive_intra
            loss = video_loss + text_loss + inter_contrastive_loss + intra_contrastive_loss

            if args.dataset in ['Daily_Mail', 'BLiSS']:#这部分代码计算视频摘要的余弦相似度（video_cos）和文本摘要的ROUGE评分（text_R1, text_R2, text_RL）。
                video_cos = calc_video_cos(video, video_summ, keyframe_index_list, mask_video_summ=mask_video_summ_bool, dataset=args.dataset)
            else:
                video_cos = 0
            text_R1, text_R2, text_RL = calc_text_rouge(article_segment_list, highlight_list, keysentence_index_list, dataset=args.dataset, rouge=args.rouge)

            #这是标准的优化过程，包括清零梯度、执行反向传播和更新模型参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #这部分代码更新了性能统计（如损失和评分）和批次处理时间，然后打印出当前训练的状态和性能指标。
            stats.update(loss=loss.item(), text_loss=text_loss.item(), video_loss=video_loss.item(),
                            inter_contrastive_loss=inter_contrastive_loss.item(), intra_contrastive_loss=intra_contrastive_loss.item(),
                            R1=text_R1, R2=text_R2, RL=text_RL, cos=video_cos)

            batch_time.update(time=time.time() - end)
            end = time.time()

            if (k + 1) % args.print_freq == 0:
                logger.info(f'[Train] Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} LR: {args.lr:.4f} '
                            f'Time: {batch_time.time:.3f} Data: {data_time.time:.3f} '
                            f'Loss: {stats.text_loss:.4f}/{stats.video_loss:.4f}/{stats.inter_contrastive_loss:.4f}/{stats.intra_contrastive_loss:.4f}/{stats.loss:.4f} '
                            f'R1: {stats.R1:.4f} R2: {stats.R2:.4f} RL: {stats.RL:.4f} Cos: {stats.cos:.4f}')
        #更新最大训练指标
        max_train_R1 = max(stats.R1, max_train_R1)
        max_train_R2 = max(stats.R2, max_train_R2)
        max_train_RL = max(stats.RL, max_train_RL)
        max_train_cos = max(stats.cos, max_train_cos)

        #使用 logger.info 打印出当前训练轮次的性能指标，使用 args.writer.add_scalar 将这些性能指标记录到 TensorBoard 日志中，方便后续的可视化和分析。
        logger.info(f'[Train] Epoch: {epoch+1}/{args.max_epoch} '
                    f'R1: {stats.R1:.4f}/{max_train_R1:.4f} '
                    f'R2: {stats.R2:.4f}/{max_train_R2:.4f} '
                    f'RL: {stats.RL:.4f}/{max_train_RL:.4f} '
                    f'Cos: {stats.cos:.4f}/{max_train_cos:.4f}\n'
        )

        args.writer.add_scalar(f'Train/max_train_R1', max_train_R1, epoch+1)
        args.writer.add_scalar(f'Train/max_train_R2', max_train_R2, epoch+1)
        args.writer.add_scalar(f'Train/max_train_RL', max_train_RL, epoch+1)
        args.writer.add_scalar(f'Train/max_train_cos', max_train_cos, epoch+1)
        args.writer.add_scalar(f'Train/train_R1', stats.R1, epoch+1)
        args.writer.add_scalar(f'Train/train_R2', stats.R2, epoch+1)
        args.writer.add_scalar(f'Train/train_RL', stats.RL, epoch+1)
        args.writer.add_scalar(f'Train/train_cos', stats.cos, epoch+1)

        #保存检查点
        save_checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'max_val_R1': max_val_R1,
            'max_val_R2': max_val_R2,
            'max_val_RL': max_val_RL,
            'max_val_cos': max_val_cos,
        }
        #在每个特定的轮次（由 args.eval_freq 控制），对模型进行评估。evaluate_msmo 函数负责在验证集上评估模型，并返回相应的性能指标。
        if (epoch + 1) % args.eval_freq == 0:
            val_R1, val_R2, val_RL, val_cos = evaluate_msmo(model, val_loader, args, epoch=epoch)

            # 更新最佳验证性能并保存检查点
            max_val_R2 = max(val_R2, max_val_R2)
            max_val_RL = max(val_RL, max_val_RL)
            if max_val_R1 < val_R1:
                max_val_R1 = max(val_R1, max_val_R1)
                best_val_epoch = epoch + 1
                torch.save(save_checkpoint, '{}/checkpoint/model_best_text.pt'.format(args.model_dir))
            if max_val_cos < val_cos:
                max_val_cos = max(val_cos, max_val_cos)
                torch.save(save_checkpoint, '{}/checkpoint/model_best_video.pt'.format(args.model_dir))

            logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} '
                        f'R1: {val_R1:.4f}/{max_val_R1:.4f} '
                        f'R2: {val_R2:.4f}/{max_val_R2:.4f} '
                        f'RL: {val_RL:.4f}/{max_val_RL:.4f} '
                        f'Cos: {val_cos:.4f}/{max_val_cos:.4f}\n\n'
            )

            #性能统计的TensorBoard记录
            args.writer.add_scalar(f'Val/max_val_R1', max_val_R1, epoch+1)
            args.writer.add_scalar(f'Val/max_val_R2', max_val_R2, epoch+1)
            args.writer.add_scalar(f'Val/max_val_RL', max_val_RL, epoch+1)
            args.writer.add_scalar(f'Val/max_val_cos', max_val_cos, epoch+1)
            args.writer.add_scalar(f'Val/val_R1', val_R1, epoch+1)
            args.writer.add_scalar(f'Val/val_R2', val_R2, epoch+1)
            args.writer.add_scalar(f'Val/val_RL', val_RL, epoch+1)
            args.writer.add_scalar(f'Val/val_cos', val_cos, epoch+1)

        args.writer.add_scalar(f'Train/loss', stats.loss, epoch+1)
        args.writer.add_scalar(f'Train/text_loss', stats.text_loss, epoch+1)
        args.writer.add_scalar(f'Train/video_loss', stats.video_loss, epoch+1)

    #返回最佳验证记录
    return max_val_R1, max_val_R2, max_val_RL, max_val_cos, best_val_epoch, \
            max_train_R1, max_train_R2, max_train_RL, max_train_cos


@torch.no_grad()
def evaluate_msmo(model, val_loader, args, epoch=None, mode='train'):
    stats = AverageMeter('R1', 'R2', 'RL', 'cos')
    data_length = len(val_loader)

    model.eval()
    for k, (video_list, video_summ_list, text_list, \
            mask_video_list, mask_video_summ_list, mask_text_list, \
            video_label_list, text_label_list, article_segment_list, highlight_list, \
            video_to_text_mask_list, text_to_video_mask_list) in enumerate(val_loader):

        batch_size = len(video_list)

        video = pad_sequence(video_list, batch_first=True)
        video_summ = pad_sequence(video_summ_list, batch_first=True)
        text = pad_sequence(text_list, batch_first=True)

        mask_video = pad_sequence(mask_video_list, batch_first=True)
        mask_video_summ = pad_sequence(mask_video_summ_list, batch_first=True)
        mask_text = pad_sequence(mask_text_list, batch_first=True)

        video_label = pad_sequence(video_label_list, batch_first=True)
        text_label = pad_sequence(text_label_list, batch_first=True)

        video, video_summ, text = video.to(args.device), video_summ.to(args.device), text.to(args.device)
        mask_video, mask_video_summ, mask_text = mask_video.to(args.device), mask_video_summ.to(args.device), mask_text.to(args.device)

        video_label = video_label.to(args.device) #[B, T]
        text_label = text_label.to(args.device) #[B, T]

        for i in range(len(video_to_text_mask_list)):
            video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
            text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

        pred_video, pred_text, contrastive_pairs = model(video=video, text=text, \
                                                            mask_video=mask_video, mask_text=mask_text, \
                                                            video_label=video_label, text_label=text_label, \
                                                            video_to_text_mask_list=video_to_text_mask_list, \
                                                            text_to_video_mask_list=text_to_video_mask_list)

        num_frame_selected = torch.sum(video_label, dim=-1)
        num_sentence_selected = torch.sum(text_label, dim=-1)

        mask_video_bool = mask_video.to(torch.bool)
        mask_video_summ_bool = mask_video_summ.to(torch.bool)
        mask_text_bool = mask_text.to(torch.bool)
        keyframe_index_list = []
        keysentence_index_list = []
        for i in range(batch_size):
            keyframe_index_list.append(torch.topk(pred_video[i, mask_video_bool[i]], k=num_frame_selected[i])[1].tolist())
            keysentence_index_list.append(torch.topk(pred_text[i, mask_text_bool[i]], k=num_sentence_selected[i])[1].tolist())

        if args.dataset in ['Daily_Mail', 'BLiSS']:
            video_cos = calc_video_cos(video, video_summ, keyframe_index_list, mask_video_summ=mask_video_summ_bool, dataset=args.dataset)
        else:
            video_cos = 0
        text_R1, text_R2, text_RL = calc_text_rouge(article_segment_list, highlight_list, keysentence_index_list, dataset=args.dataset, rouge=args.rouge)

        stats.update(R1=text_R1, R2=text_R2, RL=text_RL, cos=video_cos)


    return stats.R1, stats.R2, stats.RL, stats.cos
