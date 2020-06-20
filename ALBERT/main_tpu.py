import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import random
import re
import json
import transformers
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from Model import TweetModel
from dataset import TweetDataset
from metric import calculate_jaccard_score
import utils

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class AverageMeter(object):
    """Computes and stores the average and current values"""
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

def get_position_accuracy(logits, labels):
    predictions = np.argmax(F.softmax(logits, dim=1).cpu().data.numpy(), axis=1)
    labels = labels.cpu().data.numpy()
    total_num = 0
    sum_correct = 0
    for i in range(len(labels)):
        if labels[i] >= 0:
            total_num += 1
            if predictions[i] == labels[i]:
                sum_correct += 1
    if total_num == 0:
        total_num = 1e-7
    return np.float32(sum_correct) / total_num, total_num

def reduce_fn(vals):
    return sum(vals) / len(vals)

def loss_fn(preds, labels):
    start_preds, end_preds = preds
    start_labels, end_labels = labels

    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    return start_loss, end_loss


def train(args, train_loader, model, device, optimizer,scheduler, epoch, f):
    total_loss = AverageMeter()
    losses1 = AverageMeter() # start
    losses2 = AverageMeter() # end
    accuracies1 = AverageMeter() # start
    accuracies2 = AverageMeter() # end

    model.train()

    t = tqdm(train_loader, disable=not xm.is_master_ordinal())
    for step, d in enumerate(t):
        
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        start_position = d["start_position"].to(device)
        end_position = d["end_position"].to(device)

        model.zero_grad()

        logits1, logits2 = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            position_ids=None, 
            head_mask=None
        )

        y_true = (start_position, end_position)
        loss1, loss2 = loss_fn((logits1, logits2), (start_position, end_position))
        loss = loss1 + loss2

        acc1, n_position1 = get_position_accuracy(logits1, start_position)
        acc2, n_position2 = get_position_accuracy(logits2, end_position)

        total_loss.update(loss.item(), n_position1)
        losses1.update(loss1.item(), n_position1)
        losses2.update(loss2.item(), n_position2)
        accuracies1.update(acc1, n_position1)
        accuracies2.update(acc2, n_position2)

        
        loss.backward()
        xm.optimizer_step(optimizer)
        scheduler.step()
        print_loss = xm.mesh_reduce("loss_reduce", total_loss.avg, reduce_fn)
        print_acc1 = xm.mesh_reduce("acc1_reduce", accuracies1.avg, reduce_fn)
        print_acc2 = xm.mesh_reduce("acc2_reduce", accuracies2.avg, reduce_fn)
        t.set_description(f"Train E:{epoch+1} - Loss:{print_loss:0.2f} - acc1:{print_acc1:0.2f} - acc2:{print_acc2:0.2f}")


    log_ = f"Epoch : {epoch+1} - train_loss : {total_loss.avg} - \n \
    train_loss1 : {losses1.avg} - train_loss2 : {losses2.avg} - \n \
    train_acc1 : {accuracies1.avg} - train_acc2 : {accuracies2.avg}"

    f.write(log_ + "\n\n")
    f.flush()
    
    return total_loss.avg

def valid(args, valid_loader, model, device, tokenizer, epoch, f):
    total_loss = AverageMeter()
    losses1 = AverageMeter() # start
    losses2 = AverageMeter() # end
    accuracies1 = AverageMeter() # start
    accuracies2 = AverageMeter() # end

    jaccard_scores = AverageMeter()

    model.eval()

    with torch.no_grad():
        t = tqdm(valid_loader, disable=not xm.is_master_ordinal())
        for step, d in enumerate(t):
            
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            start_position = d["start_position"].to(device)
            end_position = d["end_position"].to(device)

            logits1, logits2 = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                position_ids=None, 
                head_mask=None
            )

            y_true = (start_position, end_position)
            loss1, loss2 = loss_fn((logits1, logits2), (start_position, end_position))
            loss = loss1 + loss2

            acc1, n_position1 = get_position_accuracy(logits1, start_position)
            acc2, n_position2 = get_position_accuracy(logits2, end_position)

            total_loss.update(loss.item(), n_position1)
            losses1.update(loss1.item(), n_position1)
            losses2.update(loss2.item(), n_position2)
            accuracies1.update(acc1, n_position1)
            accuracies2.update(acc2, n_position2)

            jac_score = calculate_jaccard_score(features_dict=d, start_logits=logits1, end_logits=logits2, tokenizer=tokenizer)

            jaccard_scores.update(jac_score)

            print_loss = xm.mesh_reduce("vloss_reduce", total_loss.avg, reduce_fn)
            print_jac = xm.mesh_reduce("jac_reduce", jaccard_scores.avg, reduce_fn)

            t.set_description(f"Eval E:{epoch+1} - Loss:{print_loss:0.2f} - Jac:{print_jac:0.2f}")

    #print("Valid Jaccard Score : ", jaccard_scores.avg)
    log_ = f"Epoch : {epoch+1} - valid_loss : {total_loss.avg} - \n\
    valid_loss1 : {losses1.avg} - \valid_loss2 : {losses2.avg} - \n\
    valid_acc1 : {accuracies1.avg} - \valid_acc2 : {accuracies2.avg} "

    f.write(log_ + "\n\n")
    f.flush()
    
    return jaccard_scores.avg

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--max_seq_len", type=int, default=192)
    parser.add_argument("--fold_index", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00002)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--spt_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sp_path", type=str, default="")

    args = parser.parse_args()

    # Setting seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model_path = args.model_path
    config = transformers.AlbertConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    tokenizer = transformers.AlbertTokenizer.from_pretrained(model_path, do_lower_case=True)
    
    MX = TweetModel(model_path, config)

    train_df = pd.read_csv(f"../input/train_5folds_seed{seed}.csv")

    args.save_path = os.path.join(args.output_dir, args.exp_name)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    f = open(os.path.join(args.save_path, f"log_f_{args.fold_index}.txt"), "w")

    num_train_dpoints = int((len(train_df)/5) * 4)

    def run():

        #torch.manual_seed(seed)

        device = xm.xla_device()
        model = MX.to(device)

        # DataLoaders
        train_dataset = TweetDataset(
            args=args,
            df=train_df,
            mode="train",
            fold=args.fold_index,
            tokenizer=tokenizer
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=False,
            num_workers=2
        )

        valid_dataset = TweetDataset(
            args=args,
            df=train_df,
            mode="valid",
            fold=args.fold_index,
            tokenizer=tokenizer
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            sampler=valid_sampler,
            num_workers=1,
            drop_last=False
        )

        param_optimizer = list(model.named_parameters())
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight"
        ]
        optimizer_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer if not any(
                        nd in n for nd in no_decay
                    )
                ], 
            'weight_decay': 0.001
            },
            {
                'params': [
                    p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay
                    )
                ], 
                'weight_decay': 0.0
            },
        ]

        

        num_train_steps = int(
            num_train_dpoints / args.batch_size / xm.xrt_world_size() * args.epochs
        )

        optimizer = AdamW(
            optimizer_parameters,
            lr=args.learning_rate * xm.xrt_world_size()
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        xm.master_print("Training is Starting ...... ")
        best_jac = 0
        #early_stopping = utils.EarlyStopping(patience=2, mode="max", verbose=True)

        for epoch in range(args.epochs):
            
            para_loader = pl.ParallelLoader(train_loader, [device])
            train_loss = train(
                args, 
                para_loader.per_device_loader(device),
                model,
                device,
                optimizer,
                scheduler,
                epoch,
                f
            )
            

            para_loader = pl.ParallelLoader(valid_loader, [device])
            valid_jac = valid(
                args, 
                para_loader.per_device_loader(device),
                model,
                device,
                tokenizer,
                epoch,
                f
            )

            jac = xm.mesh_reduce("jac_reduce", valid_jac, reduce_fn)
            xm.master_print(f"**** Epoch {epoch+1} **==>** Jaccard = {jac}")

            log_ = f"**** Epoch {epoch+1} **==>** Jaccard = {jac}"

            f.write(log_ + "\n\n")

            if jac > best_jac:
                xm.master_print("**** Model Improved !!!! Saving Model")
                xm.save(model.state_dict(), os.path.join(args.save_path, f"fold_{args.fold_index}"))
                best_jac = jac
            
            #early_stopping(jac)
            
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break


    def _mp_fn(rank, flags):
        torch.set_default_tensor_type('torch.FloatTensor')
        a = run()
    
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

if __name__ == "__main__":
    main()








