import argparse
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import itertools
# import optuna
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, classification_report, accuracy_score

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torch.nn import MSELoss
from transformers import DebertaV2Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from MCIB_trans import CIBForSequenceClassification

# Define device as a global variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global device

# Set up argument parser
argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--speaker", default='n', help="Enter y/Y for Speaker Dependent else n/N")
argParser.add_argument("-m", "--mode", default='VTA', help="VTA for Video, Text, Audio respectively")
argParser.add_argument("-c", "--context", default='y', help="y/Y for Context Dependent else n/N")

argParser.add_argument("--pre_model", type=str, default="/home/eva/Desktop/deberta-v3-base", )
argParser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei", "mustard_plus_plus"], default="mustard_plus_plus")
argParser.add_argument("-seed", "--seed", default=128, type=int, help="SEED value")
argParser.add_argument("-l", "--learning_rate", default=1e-5, type=float, help="Learning rate")
argParser.add_argument("-p", "--patience", default=20, type=int, help="Patience")
argParser.add_argument("-b", "--batch_size", default=8, type=int, help="Batch Size")
argParser.add_argument("--n_epochs", type=int, default=200)
argParser.add_argument("-dropout", "--dropout_prob", default=0.4, type=float, help="Dropout value")
argParser.add_argument("--gradient_accumulation_step", type=int, default=1)
argParser.add_argument("--warmup_proportion", type=float, default=0.1)
argParser.add_argument("-cr", "--classification_report", default='n',
                       help="Prints Classification report of Validation Set")

argParser.add_argument('--multi_head', type=int, default=8)
argParser.add_argument('--num_layers', type=int, default=6)

# argParser.add_argument("-pr", "--projection", default=256, type=int, help="Projection embedding size")
# argParser.add_argument("-sh", "--shared", default=1024, type=int, help="Shared embedding size")
argParser.add_argument("--max_seq_length", type=int, default=50)
argParser.add_argument('--p_lambda', default=4, help='coefficient -- lambda', type=float)  # For all loss
argParser.add_argument('--p_beta', default=32, help='coefficient -- beta', type=float)  # For v
argParser.add_argument('--p_gamma', default=8, help='coefficient -- gamma', type=float)   # For a
argParser.add_argument('--p_sigma', default=8, help='coefficient -- gamma', type=float)  # For t
argParser.add_argument('--beta_shift', default=1.0, help='coefficient -- shift', type=float)
# argParser.add_argument('--IB_coef', default=10, type=float)
argParser.add_argument('--B0_dim', default=256, type=float)
argParser.add_argument('--B1_dim', default=128, type=float)
# argParser.add_argument('--B1_dim', default=64, type=float)

args = argParser.parse_args()

# 保存默认参数以便在每次实验中进行更新
default_args = vars(args).copy()

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed: {}".format(seed))

def get_dataset(data, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(data):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)
        visual = np.array(visual)
        acoustic = np.array(acoustic)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        tokens = [CLS] + tokens + [SEP]

        acoustic_zero = np.zeros((1, 291))  # Assuming ACOUSTIC_DIM = 291
        acoustic = np.vstack((acoustic_zero, acoustic, acoustic_zero))
        visual_zero = np.zeros((1, 2048))  # Assuming VISUAL_DIM = 2048
        visual = np.vstack((visual_zero, visual, visual_zero))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        pad_length = max_seq_length - len(input_ids)
        pad_length_a = max_seq_length - acoustic.shape[0]
        pad_length_v = max_seq_length - visual.shape[0]
        #
        acoustic_padding = np.zeros((pad_length_a, 291))
        acoustic = np.concatenate((acoustic, acoustic_padding))
        visual_padding = np.zeros((pad_length_v, 2048))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # assert len(input_ids) == args.max_seq_length
        # assert len(input_mask) == args.max_seq_length
        # assert len(segment_ids) == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )

    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
    )
    return dataset

def create_dataloaders(all_data, fold_idx, train_idx, val_idx):
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    train_dataset = get_dataset(train_data, args.max_seq_length, get_tokenizer(args.pre_model))
    val_dataset = get_dataset(val_data, args.max_seq_length, get_tokenizer(args.pre_model))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.batch_size /
            args.gradient_accumulation_step
        ) * args.n_epochs
    )

    return (
        train_dataloader,
        val_dataloader,
        num_train_optimization_steps,
    )

def prepare_for_training(num_train_optimization_steps: int, args):
    model = CIBForSequenceClassification.from_pretrained(
        args.pre_model, multimodal_config=args, num_labels=1,
    )

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    train_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

        logits, total_loss, kl_loss_v, mse_v, kl_loss_a, mse_a, kl_loss_t, mse_t = model(
            visual_norm,
            acoustic_norm,
            input_ids,
            label_ids,
        )
        # loss_fct = MSELoss()
        loss_fct = torch.nn.BCEWithLogitsLoss()
        mse_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        loss = mse_loss + args.p_lambda * total_loss

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        # 添加梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        train_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return train_loss / nb_tr_steps



def val_epoch(model: nn.Module, val_dataloader: DataLoader):
    model.eval()
    val_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    pre = []
    labels = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, total_loss, kl_loss_v, mse_v, kl_loss_a, mse_a, kl_loss_t, mse_t = model(
                visual_norm,
                acoustic_norm,
                input_ids,
                label_ids,
            )
            # loss_fct = MSELoss()
            loss_fct = torch.nn.BCEWithLogitsLoss()
            mse_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            loss = mse_loss + args.p_lambda * total_loss

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            val_loss += loss.item()
            nb_dev_steps += 1

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            # 应用 Sigmoid 将 logits 转换为概率值
            logits = torch.sigmoid(torch.tensor(logits)).numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            # 确保 logits 和 label_ids 是列表形式
            if isinstance(logits, float):
                logits = [logits]
            if isinstance(label_ids, float):
                label_ids = [label_ids]

            pre.extend(logits)
            labels.extend(label_ids)

        pred = np.array(pre)
        preds = (pred >= 0.5).astype(int)  # 预测值进行二值化处理
        labels = np.array(labels)

    report_dict = classification_report(labels, preds, output_dict=True)

    precision = report_dict["weighted avg"]["precision"]
    recall = report_dict["weighted avg"]["recall"]
    f1 = report_dict["weighted avg"]["f1-score"]

    return val_loss / nb_dev_steps, precision, recall, f1

def main():
    torch.cuda.empty_cache()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    p_lambda_value = [4, 8]
    p_beta_values = [4, 16, 64]
    p_gamma_values = [8, 32]
    p_sigma_value = [8, 16, 64]
    B0_dim_values = [32, 64, 256]
    B1_dim_values = [32, 64, 256, 512]

    # 生成所有参数组合
    parameter_combinations = list(
        itertools.product(p_lambda_value, p_beta_values, p_gamma_values, p_sigma_value, B0_dim_values, B1_dim_values))

    # 循环遍历所有参数组合
    for p_lambda, p_beta, p_gamma, p_sigma, B0_dim, B1_dim in parameter_combinations:
        # 恢复默认参数
        args = argparse.Namespace(**default_args)
        # 更新实验参数
        args.p_lambda = p_lambda
        args.p_beta = p_beta
        args.p_gamma = p_gamma
        args.p_sigma = p_sigma
        args.B0_dim = B0_dim
        args.B1_dim = B1_dim

        # Seed everything
        set_random_seed(args.seed)

        # 获取 data
        with open(f"./features_Process/{args.dataset}.pkl", "rb") as df:
            data = pickle.load(df)
        all_data = data["train"] + data["val"]

        # 打开保存结果的文件
        result_filename = f'newCIB_trans_fine_04_{p_lambda}_{p_beta}_{p_gamma}_{p_sigma}_{B0_dim}_{B1_dim}.txt'
        with open(result_filename, 'w') as result_file:
            tee = Tee(result_file, open(result_filename, 'a'))

            # 替换标准输出为文件输出
            original_stdout = sys.stdout
            sys.stdout = tee

            # 训练、5k交叉验证
            kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
            fold_results = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_data)):
                if fold_idx > 0:
                    break  # 只训练一折
                print(f"Fold {fold_idx + 1}")

                # 得到 dataloader
                (
                    train_data_loader,
                    val_data_loader,
                    num_train_optimization_steps,
                ) = create_dataloaders(all_data, fold_idx, train_idx, val_idx)

                # 训练准备参数
                model, optimizer, scheduler = prepare_for_training(
                    num_train_optimization_steps, args
                )

                # 训练、验证过程
                best_fold_result = {
                    'epoch': 0,
                    'train_loss': float('inf'),
                    'valid_loss': float('inf'),
                    'val_precision': 0.0,
                    'val_recall': 0.0,
                    'val_f1_score': 0.0
                }
                best_f1_score = 0.0
                for epoch_i in range(int(args.n_epochs)):
                    train_loss = train_epoch(model, train_data_loader, optimizer, scheduler)
                    valid_loss, val_precision, val_recall, val_f1_score = val_epoch(model, val_data_loader)

                    print(
                        f"FOLD {fold_idx + 1} epoch {epoch_i + 1} RESULTS: "
                        f"train_loss: {train_loss}, "
                        f"valid_loss: {valid_loss}, "
                        f"val_precision: {val_precision}, "
                        f"val_recall: {val_recall}, "
                        f"val_f1_score: {val_f1_score}"
                    )

                    if val_f1_score > best_f1_score:
                        best_f1_score = val_f1_score
                        best_fold_result = {
                            'epoch': epoch_i + 1,
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                            'val_precision': val_precision,
                            'val_recall': val_recall,
                            'val_f1_score': val_f1_score}

                fold_results.append({
                    'fold': fold_idx + 1,
                    'best_epoch': best_fold_result['epoch'],
                    'train_loss': best_fold_result['train_loss'],
                    'valid_loss': best_fold_result['valid_loss'],
                    'val_precision': best_fold_result['val_precision'],
                    'val_recall': best_fold_result['val_recall'],
                    'val_f1_score': best_fold_result['val_f1_score']
                })

            # k折交叉验证的结果
            print('K-FOLD CROSS VALIDATION RESULTS')
            print('--------------------------------')
            for result in fold_results:
                print(
                    f"Fold {result['fold']}: "
                    f"Best Epoch: {result['best_epoch']}, "
                    f"Train Loss: {result['train_loss']}, "
                    f"Valid Loss: {result['valid_loss']}, "
                    f"Valid precision: {result['val_precision']}, "
                    f"Valid recall: {result['val_recall']}, "
                    f"Valid F1 Score: {result['val_f1_score']}"
                )

            avg_val_precision = np.mean([result['val_precision'] for result in fold_results])
            avg_val_recall = np.mean([result['val_recall'] for result in fold_results])
            avg_valid_f1 = np.mean([result['val_f1_score'] for result in fold_results])

            print('--------------------------------')
            print(f"Average Validation Accuracy: {avg_val_precision}")
            print(f"Average Validation Correlation: {avg_val_recall}")
            print(f"Average Validation F1 Score: {avg_valid_f1}")

            # 恢复标准输出
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()
