import argparse
from contextlib import AbstractContextManager
from enum import Enum, auto
from pathlib import Path
import sys
import typing
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from torch import nn
from torch.nn import TransformerEncoder
from torch.utils.data import StackDataset, ConcatDataset, TensorDataset, Dataset, DataLoader, Subset
from typing import Any, Dict, Set, Tuple, List, Optional
import collections
import io
import itertools
import json
import numpy as np
import sklearn.model_selection
import torch
from .seq_classification_models import *
# import torchinfo

# Label = str



def label_idx_to_class_list(label_idx: Dict[Label,int]) -> List[int]:
    class_list = list(label_idx.values())
    sorted(class_list)
    return class_list

def read_embeddings_synthetic() -> Tuple[Dataset, Dataset, List[int]]:
    class_list=[0,1,2,3]
    seq_len=5
    #iclass_embedding = torch.rand(size=[len(classes), 512])
    class_embedding = torch.tensor(
            [[1,0,0,0,0],
             [0,1,0,0,0],
             [0,0,1,0,0],
             [0,0,0,1,1]],
            dtype=torch.float32) / 2
    embed_dim = class_embedding.shape[1]
    #class_embedding_seq = torch.tensor([[class_embedding[c, :] for i in range(seq_len)] for c in classes])  # (num_classes, seq_len, embed_dimension)
    class_embedding_seq = class_embedding[:,None,:].expand(len(class_list), seq_len, embed_dim)

    label_idx={0:0, 1:1, 2:2, 3:3}

    # Example label IDs
    example_label_id_list = [0, 1, 2, 3] * 6  # Repeat for balanced examples


    # Create tensor of example label IDs
    example_label_id = torch.tensor(example_label_id_list, dtype=torch.long)  # [num_examples]

    # Generate sequence embeddings for each example
    # Each example corresponds to `seq_len` embeddings (e.g., a sequence)
    embeddings = torch.stack(
        [class_embedding_seq[label_id] for label_id in example_label_id], dim=0
    )  # [num_examples, embedding_dim]

    # Generate one-hot encoding for labels
    example_label_one_hot = nn.functional.one_hot(example_label_id, num_classes=len(class_list)).to(torch.float32)

    train_ds = StackDataset(embedding=embeddings, label_one_hot=example_label_one_hot, label_id=example_label_id)
    test_ds = StackDataset(embedding=embeddings, label_one_hot=example_label_one_hot, label_id=example_label_id)

    print("test set size:", len(test_ds))
    print("train set size:", len(train_ds))
    return train_ds, test_ds, class_list


def read_embeddings(path: Path, n_parts: int) -> Tuple[Dataset, Dataset, List[int]]:
    test_frac = 0.1
    MIN_LABEL_EXAMPLES = 10

    parts = [ torch.load(path / f'{i:04d}.pt') for i in range(n_parts) ]

    # flatten parts
    embeddings = torch.concat([xs['embeddings'] for xs in parts]) # (n_examples, sequence_length, embedding_dim)
    multilabels: List[List[Label]] = [labels for xs in parts for labels in xs['labels']] # len(multilabels) == n_examples

    # Assign single label to each example
    example_label: List[Label] # (n_examples,)
    example_label = [lbls[0] for lbls in multilabels]

    # Determine label set
    label_counts: Dict[Label, int] = collections.Counter(example_label)
    all_labels: Set[Label] = { lbl for lbl, n in label_counts.items() if n > MIN_LABEL_EXAMPLES }
    label_idx: Dict[Label, int] = { lbl: idx for idx, lbl in enumerate(all_labels) }

    # Drop examples whose labels have all been dropped
    example_mask = torch.tensor([label in all_labels for label in example_label])
    embeddings = Subset(embeddings, example_mask.nonzero()[:,0])
    example_label_id: Tensor = torch.tensor([label_idx[label] for label in example_label if label in all_labels], dtype=torch.long)
    example_label_one_hot: Tensor = nn.functional.one_hot(example_label_id, len(label_idx))

    ds = StackDataset(
            embedding=embeddings,
            label_one_hot=example_label_one_hot.to(torch.float32),
            label_id=example_label_id,
        )

    # Test/train split
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idxs, test_idxs = [x for x in sss.split(np.zeros(len(ds)), example_label_one_hot)][0]
    train_ds = Subset(ds, train_idxs)
    test_ds = Subset(ds, test_idxs)
    print('test set size', len(test_ds))
    print('train set size', len(train_ds))
    return (train_ds, test_ds, label_idx_to_class_list(label_idx))


def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn:nn.Module,  class_list: List[int], device: torch.device):
    '''Computes the avg per-example loss and eval metrics on the whole training set, not just one batch. '''
    model.eval()
    total_loss = 0
    total_grades_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['embedding'].to(device)
            labels = batch['label_one_hot'].to(device)
            grades = batch['grades_one_hot'].to(device)

            outputs, seq_logits_grades =  model(inputs)
            # (_, loss_fn, grade_fn) = model.compute_loss(final_logits = outputs, class_targets = labels)
            # TODO fixme 

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_grades_loss += grade_fn(seq_logits_grades, grades)
            all_preds.append(outputs.cpu())
            all_labels.append(batch['label_one_hot'])

    y_true = torch.cat(all_labels)
    metrics = classification_metrics(y_pred_logits=torch.cat(all_preds), y_true_one_hot=y_true, class_list=class_list)
    metrics['loss'] = total_loss / len(dataloader)
    print(f"Evaluate: total_loss: {total_loss}, grades_loss: {total_grades_loss}")
    return metrics


def evaluate_new(model: AggregateAndClassify, dataloader: DataLoader, class_list: List[int], device: torch.device):
    '''Computes the avg per-example loss and eval metrics on the whole training set, not just one batch. '''
    model.eval()
    total_loss = 0.0
    total_label_loss = 0.0
    total_grades_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['embedding'].to(device)
            labels = batch['label_one_hot'].to(device)
            grades = batch['grades_one_hot'].to(device)

            outputs, seq_logits_grades =  model(inputs)
            (loss, label_loss, grade_loss) = model.compute_loss(final_logits = outputs, class_targets = labels, seq_logits_grades=seq_logits_grades, grade_targets=grades)
            # TODO fixme 

            # loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_label_loss += label_loss.item()
            total_grades_loss += grade_loss.item()
            all_preds.append(outputs.cpu())
            all_labels.append(batch['label_one_hot'])

    y_true = torch.cat(all_labels)
    metrics = classification_metrics(y_pred_logits=torch.cat(all_preds), y_true_one_hot=y_true, class_list=class_list)
    metrics['loss'] = total_loss / len(dataloader)
    print(f"Evaluate: total_loss: {total_loss}, label_loss: {total_label_loss},  grades_loss: {total_grades_loss}")
    return metrics


def classification_metrics(y_pred_logits, y_true_one_hot, class_list: List[int]) -> Dict[str, object]:
    """
    y_pred_logits: predicted logits
    y_true_one_hot: true one-hot
    class_list: ordered list of classes (as int-labels) to evaluate
    """
    y_true_label = y_true_one_hot.argmax(dim=1)
    y_pred_label = y_pred_logits.argmax(dim=1)
    y_pred_prob = torch.nn.functional.softmax(y_pred_logits, dim=1)

    conf_matrix = confusion_matrix(y_true=y_true_label, y_pred=y_pred_label, labels=np.arange(len(class_list))) #labels=np.array(list(classes)))
    true_positives = np.diagonal(conf_matrix).sum()

    # majority class count
    true_class_counts = conf_matrix.sum(axis=1)
    majority_class_count = np.max(true_class_counts)
    majority_class_index = np.argmax(true_class_counts)
    majority_class_predictions = conf_matrix[:, majority_class_index]  # Extract the column of majority class prediction
    majority_class_false = majority_class_predictions[np.arange(len(majority_class_predictions)) != majority_class_index].sum()

    metrics = {
            'f1_micro': f1_score(y_true_label, y_pred_label, average='micro'),
            'roc_auc': roc_auc_score(y_true_one_hot, y_pred_prob, multi_class='ovo', labels=np.array(class_list)),
            'true_pos': int(true_positives), # (confusion * np.eye(n_classes)).sum(),
            # The sum of predictions where the predicted class is the majority class but the true class is not the majority class.
            'majority_class_false': int(majority_class_false),
            'accuracy': accuracy_score(y_true_label, y_pred_label),
            }
    return metrics


class Reporter:
    def __init__(self, file: io.TextIOBase):
        self.file = file

    def report(self, epoch_n: int, test_metrics, train_metrics):
        x = {
                'epoch_n': epoch_n,
                'test':  test_metrics,
                'train':  train_metrics,
        }
        json.dump(x, self.file)
        print('\n', file=self.file)
        self.file.flush()
        print(epoch_n, "test", test_metrics)
        print(epoch_n, "train", train_metrics)


class Noop(AbstractContextManager):
    def __enter__(self):
        pass
        return self

    def __exit__(self, *args):
        pass
 


class ClassificationModel(Enum):
    multi_class = auto()
    multi_label = auto()
    multi_label_packed = auto()
    multi_label_proj_packed = auto()
    multi_label_multi_seq_proj_packed = auto()
    mlp = auto()

    @staticmethod
    def from_string(arg:str):
        try:
            return ClassificationModel[arg]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid ClassificationModel choice: %s" % arg)
        

def run(root: Path
        , model_type: ClassificationModel
        ,train_ds:Dataset
        ,test_ds:Dataset
        ,class_list:List[int]
        ,out_dir: Optional[Path]=None
        ,overwrite:bool=False
        ,batch_size: int=128
        ,n_epochs: int=30
        ,inner_dim: int=64
        ,nhead: int=1
        ,device_str:str = 'cuda'
        ,snapshot_every:Optional[int]=None
        ,eval_every:Optional[int]=None
        ,epoch_timer:typing.ContextManager = Noop()
        ,target_metric:str = 'roc_auc'
        ,snapshot_best_after:Optional[int] = None
        ):
    if out_dir is None:
        out_dir = root / Path('runs') / str(model_type)
    out_dir.mkdir(parents=True, exist_ok=overwrite)

    seq_len, llm_dim = train_ds[0]['embedding'].shape
    print('llm_dim', llm_dim)
    print('context_sz', seq_len)
    print('classes', len(class_list))
    prev_highest: float = sys.float_info.min



    device = torch.device(device=device_str)
    if model_type == ClassificationModel.multi_class:
        model, loss_fn = build_model_multi_class_classifier(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead)
    elif model_type == ClassificationModel.multi_label:
        model, loss_fn = build_model_multi_label_embedding_classifier(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.multi_label_packed:
        model, loss_fn = build_model_multi_label_embedding_classifier_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.multi_label_proj_packed:
        model, loss_fn = build_model_multi_label_embedding_classifier_proj_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, inner_dim=inner_dim, class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.multi_label_multi_seq_proj_packed:
        raise ValueError(f"{model_type} needs to be called with run_num_seqs")
        # model, loss_fn = build_model_multi_label_multi_seq_embedding_classifier_proj_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, inner_dim=inner_dim, num_seqs=num_seqs,  class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.mlp:
        model, loss_fn = build_model_multi_class_classifier_mlp(layer_dims=[llm_dim//2**i for i in range(4)], n_classes=len(class_list), llm_dim=llm_dim)
    else:
        raise ValueError(f'unknown model {model_type}')

    model = model.to(device)
    # torchinfo.summary(model, input_size=(batch_size, seq_len, llm_dim), device=device)

    print('Training...')
    reporter = Reporter((out_dir / 'history-log.json').open('w'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch_t in range(n_epochs):
        with epoch_timer:
            print(f'epoch {epoch_t}...')
            model.train()
            for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
                optimizer.zero_grad()
                output = model(batch['embedding'].to(device))
                loss = loss_fn(output, batch['label_one_hot'].to(device))
                loss.backward()
                optimizer.step()

        # Save model checkpoint
        if snapshot_every and epoch_t % snapshot_every == 1 :
            torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch_t}.pt")


        if eval_every is None or epoch_t % eval_every == 1 :
            # Evaluate loss
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            test_metrics = evaluate(model, test_loader, loss_fn, class_list, device)

            train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            train_metrics = evaluate(model, train_eval_loader, loss_fn, class_list, device)
            reporter.report(epoch_t, test_metrics, train_metrics)

            # Save model checkpoint on better validation metric
            if test_metrics.get(target_metric) is None:
                raise RuntimeError(f"Snapshot target metric {target_metric} not available. Choices: {test_metrics.keys()}")
            target=test_metrics[target_metric]
            if target > prev_highest:
                print(f"Epoch {epoch_t}: Best {target_metric} on test: {target} (was: {prev_highest}).")
                prev_highest = target
                if snapshot_best_after and epoch_t >= snapshot_best_after :
                    print("Epoch {epoch_t}: Saving best snapshot")
                    torch.save(model.state_dict(), out_dir / f"model_best_epoch_{epoch_t}.pt")


    torch.save(model.state_dict(), out_dir / f"model_final.pt")

# ==========================


def run_num_seqs(root: Path
        , model_type: ClassificationModel
        ,train_ds:Dataset
        ,test_ds:Dataset
        ,class_list:List[int]
        ,grades_list:List[int]
        ,out_dir: Optional[Path]=None
        ,overwrite:bool=False
        ,batch_size: int=128
        ,n_epochs: int=30
        ,inner_dim: int=64
        ,nhead: int=1
        ,aggregation: str = "max"
        ,device_str:str = 'cuda'
        ,snapshot_every:Optional[int]=None
        ,eval_every:Optional[int]=None
        ,epoch_timer:typing.ContextManager = Noop()
        ,target_metric:str = 'roc_auc'
        ,snapshot_best_after:Optional[int] = None
        ):
    if out_dir is None:
        out_dir = root / Path('runs') / str(model_type)
    out_dir.mkdir(parents=True, exist_ok=overwrite)

    num_seqs, seq_len, llm_dim = train_ds[0]['embedding'].shape
    print('llm_dim', llm_dim)
    print('seq_len', seq_len)
    print('num_seqs', num_seqs)
    print('classes', len(class_list))
    prev_highest: float = sys.float_info.min



    device = torch.device(device=device_str)
    if model_type == ClassificationModel.multi_label_multi_seq_proj_packed:
        model, class_loss_fn, grade_loss_fn = \
            build_model_multi_label_multi_seq_embedding_classifier_proj_packed(n_classes=len(class_list)
                                                                                , n_grades=len(grades_list)
                                                                                , llm_dim=llm_dim
                                                                                , nhead = nhead
                                                                                , inner_dim=inner_dim
                                                                                , num_seqs=num_seqs
                                                                                , class_weights=torch.ones(len(class_list)).to(device)
                                                                                , grade_weights=torch.ones(len(grades_list)).to(device)
                                                                                , aggregation = aggregation
                                                                                )
    else:

        raise ValueError(f'unknown model {model_type}')

    model = model.to(device)
    # torchinfo.summary(model, input_size=(batch_size, seq_len, llm_dim), device=device)

    print('Training...')
    reporter = Reporter((out_dir / 'history-log.json').open('w'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch_t in range(n_epochs):
        with epoch_timer:
            print(f'epoch {epoch_t}...')
            model.train()
            for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
                optimizer.zero_grad()
                (pred_classes, pred_grades) = model(batch['embedding'].to(device))
                # loss = class_loss_fn(pred_classes, batch['label_one_hot'].to(device))
                # grades_loss = grade_loss_fn()
                total_loss, class_loss_fn, grade_loss_fn = model.compute_loss(final_logits= pred_classes
                                   , seq_logits_grades=pred_grades
                                   , class_targets= batch['label_one_hot'].to(device)
                                   , grade_targets= batch['grades_one_hot'].to(device))
                total_loss.backward()
                optimizer.step()

        # Save model checkpoint
        if snapshot_every and epoch_t % snapshot_every == 0 :
            torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch_t}.pt")

        if  eval_every is None or epoch_t % eval_every == 0 :
            # Evaluate loss
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            test_metrics = evaluate_new(model=model, dataloader=test_loader, class_list = class_list, device=device)
            print("test_metrics", test_metrics)

            train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            train_metrics = evaluate_new(model=model, dataloader=train_eval_loader, class_list = class_list, device=device)
            reporter.report(epoch_t, test_metrics, train_metrics)

            # Save model checkpoint on better validation metric
            if test_metrics.get(target_metric) is None:
                raise RuntimeError(f"Snapshot target metric {target_metric} not available. Choices: {test_metrics.keys()}")
            target=test_metrics[target_metric]
            if target > prev_highest:
                print(f"Epoch {epoch_t}: Best {target_metric} on test: {target} (was: {prev_highest}).")
                prev_highest = target
                if snapshot_best_after and epoch_t >= snapshot_best_after :
                    print("Epoch {epoch_t}: Saving best snapshot")
                    torch.save(model.state_dict(), out_dir / f"model_best_epoch_{epoch_t}.pt")


    torch.save(model.state_dict(), out_dir / f"model_final.pt")



def main() -> None:
    root = Path('out2')
    # train_ds, test_ds, class_list = read_embeddings(root, 95)
    train_ds, test_ds, class_list = read_embeddings_synthetic()

    cmd_args = {"snapshots":2, "n_epochs":10, "train_ds":train_ds, "test_ds":test_ds, "class_list":class_list, "device_str":"cpu", "inner_dim": 64, "nhead":1}

    # run(root=root, model_type='multi_class', **cmd_args)
    run(root=root, model_type=ClassificationModel.multi_label_proj_packed, **cmd_args)
    # run(root=root, model_type='multi_label', **cmd_args)
    # run(root=root, model_type='mlp', **cmd_args)


if __name__ == '__main__':
    main()

