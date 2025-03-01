from pathlib import Path
import sys
from enum import Enum, auto
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.utils import compute_class_weight
from torch import nn
import torch as pt
from torch.nn import TransformerEncoder
from torch.utils.data import StackDataset, ConcatDataset, TensorDataset, Dataset, DataLoader, Subset
from typing import Any, Dict, Set, Tuple, List, Optional

import torch.profiler as ptp
import collections
import io
import itertools
import json
import numpy as np
import sklearn.model_selection
import torch
import pandas as pd
import time

from . multi_seq_class_grade_model import ProblemType

from .rubric_db import *

from . test_bank_prompts import get_prompt_classes
from . import attention_classify

# import torchinfo
from .vector_db import Align, EmbeddingDb, ClassificationItemId

Label = str


# class ClassificationItemDataset(Dataset):
#     def __init__(self, db: EmbeddingDb, ):
#         self.db = db

#     def __getitem__(self, item: ClassificationItemId) -> pt.Tensor:
#         self.db.db.execute(
#             '''--sql
#             SELECT tensor_id
#             FROM classification_feature
#             WHERE classification_item_id = ?
#             ''',
#             (item,)
#         )
#         tensor_ids = self.db.db.fetch_df()
#         return self.db.fetch_tensors(tensor_ids['tensor_id'])


def get_queries(db:EmbeddingDb)-> Set[str]:
    db.db.execute(
        '''--sql
        SELECT DISTINCT classification_item.metadata->>'$.query' as query
        FROM classification_item
        '''
    )
    return db.db.fetch_df()['query']



def get_query_items(db: EmbeddingDb, query: List[str]) -> Set[ClassificationItemId]:
    db.db.execute(
        '''--sql
        SELECT classification_item_id
        FROM classification_item
        WHERE (metadata->>'$.query') in ?
        ''',
        (query,)
    )
    return db.db.fetch_df()['classification_item_id']

def lookup_queries_paragraphs_judgments(db:EmbeddingDb, classification_item_id: List[ClassificationItemId]):
    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index
    db.db.execute(
        '''--sql
        SELECT needles.i,
                   classification_item.metadata->>'$.query' as query,
                   classification_item.metadata->>'$.passage' as passage,
                   label_assignment.true_labels,
                   classification_item.classification_item_id as classification_item_id

        FROM classification_item
        INNER JOIN (SELECT * FROM classification_item_ids_df) AS needles ON classification_item.classification_item_id = needles.classification_item_id
        INNER JOIN label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
        ORDER BY needles.i ASC;
    ''')
    return db.db.df()

def lookup_queries_paragraphs(db:EmbeddingDb, classification_item_id: List[ClassificationItemId]):
    '''Same as lookup_queries_paragraphs_judgments, just without judgments'''
    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index
    db.db.execute(
        '''--sql
        SELECT needles.i,
                   classification_item.metadata->>'$.query' as query,
                   classification_item.metadata->>'$.passage' as passage,
                   classification_item.classification_item_id as classification_item_id

        FROM classification_item
        INNER JOIN (SELECT * FROM classification_item_ids_df) AS needles ON classification_item.classification_item_id = needles.classification_item_id
        INNER JOIN label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
        ORDER BY needles.i ASC;
    ''')
    return db.db.df()


def get_classification_features(db: EmbeddingDb, query:str):

    db.db.execute(
        '''--sql
        select
        classification_item.classification_item_id,
        classification_feature.tensor_id,
        classification_item.metadata->>'$.query' as query,
        classification_item.metadata->>'$.passage' as passage,
        classification_feature.metadata->>'$.prompt_class' as prompt_class,
        classification_feature.metadata->>'$.test_bank' as test_bank_id,
        label_assignment.true_labels
        from classification_item
        inner join classification_feature on classification_feature.classification_item_id = classification_item.classification_item_id
        inner join label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
        where classification_item.metadata->>'$.query' = ?
        ;
        ''',
        (query,)
    )
    return db.db.fetch_df()


# def get_tensor_ids(db:EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class:str):
#     db.db.execute(
#         '''
# select tensor_id
# from classification_feature
# where classification_item_id in `classification_item_id` classification_item.metadata->>'$.query' = ?
# ;
#         ''',
#         (query,)
#     )
#     return db.db.fetch_df()

def get_tensor_ids(db:EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class:str):
    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index
    db.db.execute(
        '''--sql
        SELECT 
            needles.i,
            array_agg(tensor_id) AS tensor_ids,
            array_agg(classification_feature.metadata->>'$.test_bank') AS test_bank_ids
        FROM classification_feature
        INNER JOIN 
            (SELECT * FROM classification_item_ids_df) AS needles 
        ON 
            classification_feature.classification_item_id = needles.classification_item_id
        WHERE 
            classification_feature.metadata->>'$.prompt_class' = ?
        GROUP BY 
            needles.i
        ORDER BY 
            needles.i ASC;
        ''', (prompt_class,))
    return db.db.df()



def annotate_with_grades(embedding_db:EmbeddingDb, data_tensors:pd.DataFrame) -> pd.DataFrame:
    embedding_db.db.execute(
        '''--sql
        SELECT needles.tensor_id, exam_grade.self_rating, exam_grade.test_bank_id
        FROM (SELECT of data_tensors.tensor_id FROM data_tensors) AS needles
        INNER JOIN classification_feature AS cf
        ON cf.tensor_id = needles.tensor_id
        INNER JOIN classification_item AS ci
        ON ci.classification_item_id = cf.classification_item_id
        INNER JOIN rubric.relevance_item AS ri
        ON ri.query_id = ci.metadata->>'$.query'
            AND ri.paragraph_id = ci.metadata->>'$.paragraph'
        INNER JOIN exam_grade
        ON exam_grade.relevance_item_id = ri.relevance_item_id
            AND exam_grade.test_bank_id = cf.metadata->>'$.test_bank'
        '''
    )
    grade_df = embedding_db.db.fetch_df()
    return data_tensors.join(grade_df, on="tensor_id")




def lookup_classification_tensors(db:EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class:str):
    # Get tensor metadata
    metadata_df = db.get_tensor_metadata(classification_item_id, prompt_class)

    # Fetch tensors and add them as a new column to the metadata DataFrame
    metadata_df = db.fetch_tensors_from_metadata(metadata_df, token_length=128, align=Align.ALIGN_BEGIN)

    # Access the computed tensors
    # for _, row in metadata_df.iterrows():
    #     print(f"Index: {row['i']}, Tensor: {row['pt_tensor']}")
    # return metadata_df

# def balanced_training_data(embedding_db: EmbeddingDb) -> pd.DataFrame:
#     embedding_db.db.execute(
#         '''--sql
#         SELECT classification_item.metadata->>'$.query' as query,
#         classification_item.metadata->>'$.passage' as passage,
#                label_assignment.true_labels as true_labels,
#                classification_item.classification_item_id as classification_item_id
#         FROM classification_item
#         INNER JOIN label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
#         WHERE len(label_assignment.true_labels) = 1
#         '''
#     )
#     df = embedding_db.db.fetch_df()
#     df['positive'] = df['true_labels'].apply(lambda x: '0' in x)
#     by_label = df.groupby('positive')
#     counts = by_label['classification_item_id'].count()
#     n = counts.min()
#     sampled = by_label.sample(n)
#     sampled.drop(columns=['positive'], inplace=True)
#     return sampled

def get_tensor_ids_with_grades(db: EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class: str):
    """
    For each classification_item_id, fetch:
      - array of tensor_ids
      - array of test_bank_ids
      - array of exam_grade fields (e.g., self_rating, is_correct)

        # df_with_grades columns might look like:
        # i | tensor_ids         | test_bank_ids       | self_ratings        | is_corrects
        # --|--------------------|---------------------|---------------------|-------------
        # 0 | [t1, t2, ...]      | ['tb1', 'tb1', ...] | [2, 3, ...]         | [False, True, ...]
        # 1 | [t17, t18, ...]    | ['tb2', ...]        | [1, 4, ...]         | [True, False, ...]
        # ...

    """    

    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index

    # Register the DataFrame as a temporary table for use in SQL
    db.db.register('classification_item_ids_df', classification_item_ids_df)

    # Execute the query
    db.db.execute(
        '''--sql
        SELECT 
            needles.i,
            array_agg(cf.tensor_id) AS tensor_ids,
            array_agg(cf.metadata->>'$.test_bank') AS test_bank_ids,
            array_agg(eg.self_rating) AS self_ratings,
            array_agg(eg.is_correct) AS correctness,
            la.true_labels AS judgment,
            ci.classification_item_id
        FROM classification_feature AS cf
        INNER JOIN classification_item_ids_df AS needles 
            ON cf.classification_item_id = needles.classification_item_id
        INNER JOIN classification_item AS ci
            ON ci.classification_item_id = cf.classification_item_id
        INNER JOIN rubric.relevance_item AS ri
            ON ri.query_id = (ci.metadata->>'$.query')
            AND ri.paragraph_id = (ci.metadata->>'$.passage')
        INNER JOIN rubric.exam_grade AS eg
            ON eg.relevance_item_id = ri.relevance_item_id
            AND eg.test_bank_id = (cf.metadata->>'$.test_bank')
        INNER JOIN label_assignment AS la
            ON la.classification_item_id = ci.classification_item_id
        WHERE 
            (cf.metadata->>'$.prompt_class') = ?
        GROUP BY 
            needles.i, ci.classification_item_id, judgment
        ORDER BY 
            needles.i ASC;
        ''', (prompt_class,))
    
    return db.db.df()


def get_tensor_ids_plain(db: EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class: str):
    """
    For each classification_item_id, fetch:
      - array of tensor_ids
      - array of test_bank_ids
      - array of exam_grade fields (e.g., self_rating, is_correct)

        # df_with_grades columns might look like:
        # i | tensor_ids         | test_bank_ids       | self_ratings        | is_corrects
        # --|--------------------|---------------------|---------------------|-------------
        # 0 | [t1, t2, ...]      | ['tb1', 'tb1', ...] | [2, 3, ...]         | [False, True, ...]
        # 1 | [t17, t18, ...]    | ['tb2', ...]        | [1, 4, ...]         | [True, False, ...]
        # ...

    """    

    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index

    # Register the DataFrame as a temporary table for use in SQL
    db.db.register('classification_item_ids_df', classification_item_ids_df)

    # Execute the query
    db.db.execute(
        '''--sql
        SELECT 
            needles.i,
            array_agg(cf.tensor_id) AS tensor_ids,
            array_agg(cf.metadata->>'$.test_bank') AS test_bank_ids,
            ci.classification_item_id
        FROM classification_feature AS cf
        INNER JOIN classification_item_ids_df AS needles 
            ON cf.classification_item_id = needles.classification_item_id
        INNER JOIN classification_item AS ci
            ON ci.classification_item_id = cf.classification_item_id
        INNER JOIN rubric.relevance_item AS ri
            ON ri.query_id = (ci.metadata->>'$.query')
            AND ri.paragraph_id = (ci.metadata->>'$.passage')
        INNER JOIN rubric.exam_grade AS eg
            ON eg.relevance_item_id = ri.relevance_item_id
            AND eg.test_bank_id = (cf.metadata->>'$.test_bank')
        WHERE 
            (cf.metadata->>'$.prompt_class') = ?
        GROUP BY 
            needles.i, ci.classification_item_id
        ORDER BY 
            needles.i ASC;
        ''', (prompt_class,))
    
    return db.db.df()



class SequenceMode(Enum):
    single_sequence = auto()
    concat_sequence = auto()
    multi_sequence = auto()

    @staticmethod
    def from_string(arg:str):
        try:
            return SequenceMode[arg]
        except KeyError:
            import argparse   
            raise argparse.ArgumentTypeError("Invalid ClassificationModel choice: %s" % arg)
        

class ClassificationItemDataset(Dataset):
    def __init__(self, tensor_df:pd.DataFrame, db:EmbeddingDb, sequence_mode:SequenceMode, max_token_len:int, align:Align):
        self.tensor_df = tensor_df
        self.db = db
        self.sequence_mode = sequence_mode
        self.max_token_len = max_token_len
        self.align = align


    def __getitem__(self, index) -> pt.Tensor:
        tensor_ids = self.tensor_df.loc[index,"tensor_ids"]
        tensor=None
        if self.sequence_mode == SequenceMode.single_sequence:
            tensor = self.db.fetch_tensors_single(tensor_ids=tensor_ids, token_length=self.max_token_len, align=self.align)
        elif self.sequence_mode == SequenceMode.multi_sequence:
                        # self.fetch_tensors(tensor_ids=tensor_ids, token_length=token_length, align=align)
            tensor = self.db.fetch_tensors(tensor_ids=tensor_ids, token_length=self.max_token_len, align=self.align)
        elif self.sequence_mode == SequenceMode.concat_sequence:
            tensor = self.db.fetch_tensors_concat(tensor_ids=tensor_ids, token_length=self.max_token_len, align=self.align)
        else:
            raise RuntimeError(f"sequence mode {self.sequence_mode} is not defined.")

        return tensor

    def __len__(self) -> int:
        return len(self.tensor_df)
    
    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])


class GradeClassificationItemDataset(Dataset):
    def __init__(self, tensor_df:pd.DataFrame):
        self.tensor_df = tensor_df


    def __getitem__(self, index) -> pt.Tensor:
        grades = self.tensor_df.loc[index,"self_ratings"]
        # grades = self.tensor_df.self_ratings

        # example_grades_list_train = [[plain_grade(r, tup.judgment) 
        #                                 for r in tup.self_ratings] 
        #                                     for tup in tensor_df.itertuples()]
        

        # Convert to a tensor
        

        # # tensor=None
        # grades_tensor = torch.tensor([[g] for g in grades], dtype=torch.long)
        # tensor_one_hot = torch.nn.functional.one_hot(grades_tensor, num_classes=6).to(torch.float)

        # range_tensor = torch.arange(6).unsqueeze(0).unsqueeze(0)  # tensor([[[0, 1, 2, 3, 4, 5]]])
        # tensor_triangle_hot = (grades_tensor >= range_tensor).to(torch.float)  # Triangle-hot. 


        # Convert grades to tensor
        grades_tensor = torch.tensor([[g] for g in grades], dtype=torch.long)

        # One-hot encoding
        tensor_one_hot = torch.nn.functional.one_hot(grades_tensor, num_classes=6).to(torch.float)

        # Triangle-hot encoding
        range_tensor = torch.arange(6).view(1, -1)  # Shape (1, 6)
        tensor_triangle_hot = (range_tensor <= grades_tensor).to(torch.float)  # Shape (3, 6)

        # Add a singleton dimension to match one-hot shape (3, 1, 6)
        tensor_triangle_hot = tensor_triangle_hot.unsqueeze(1)

        # print("tensor_one_hot.shape",tensor_one_hot.shape," tensor_triangle_hot.shape", tensor_triangle_hot.shape)
        return tensor_triangle_hot

    def __len__(self) -> int:
        return len(self.tensor_df)
    
    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])



# class GradeClassificationItemDataset(Dataset):
#     def __init__(self, tensor_df:pd.DataFrame, db:EmbeddingDb, sequence_mode:SequenceMode, max_token_len:int, align:Align):
#         self.tensor_df = tensor_df
#         self.db = db
#         self.sequence_mode = sequence_mode
#         self.max_token_len = max_token_len
#         self.align = align


#     def __getitem__(self, index) -> pt.Tensor:
#         example_num = 10
#         tok_len = 1
#         llm_dim = 6

#         grades_list_label = self.tensor_df.loc[index,"grade_label"]
#         # grades_list_dense = [grades_idx(g) for g in grades_list_label]
#         grades_list_dense = grades_list_label

        
#         # Convert them to a PyTorch tensor
#         grades_tensor = torch.tensor(grades_list_dense)

#         # One-hot encode with 6 classes (0 through 5)
#         tensor = torch.nn.functional.one_hot(grades_tensor, num_classes=6)

#         return tensor

#     def __len__(self) -> int:
#         # return len(self.tensor_df)
#         return ...a
    
#     def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
#         return ConcatDataset([self, other])


class EmbeddingStackDataset(torch.utils.data.Dataset):
    def __init__(self, embedding, label_one_hot, label_id, grades_one_hot, grades_id, grades_valid, classification_item_id):
        # Check that all datasets have the same first dimension
        if not (len(embedding) == label_one_hot.size(0) == label_id.size(0)):
            raise ValueError(f"Size mismatch between datasets:  ({len(embedding)} == {label_one_hot.size(0)} == {label_id.size(0)})")
        self.embedding = embedding
        self.label_one_hot = label_one_hot
        self.label_id = label_id
        self.grades_one_hot = grades_one_hot
        self.grades_id = grades_id
        self.grades_valid = grades_valid
        self.classification_item_id = classification_item_id

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return {
            "embedding": self.embedding[idx],
            "label_one_hot": self.label_one_hot[idx],
            "label_id": self.label_id[idx],
            "grades_one_hot": self.grades_one_hot[idx],
            "grades_id":self.grades_id[idx],
            "grades_valid":self.grades_valid[idx],
            "classification_item_id": self.classification_item_id[idx]
        }
class EmbeddingPredictStackDataset(torch.utils.data.Dataset):
    def __init__(self, embedding,classification_item_id):
        # Check that all datasets have the same first dimension
        self.embedding = embedding
        self.classification_item_id = classification_item_id

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return {
            "embedding": self.embedding[idx],
            "classification_item_id": self.classification_item_id[idx]
        }
    
    
    

def conv_class(example_label_list:list[Any], classes:list[int], label_idx:Dict[Any,int]):
        def default_label(label, d):
            l = label_idx.get(label)
            if (l is None):
                if (label != -1):
                    print(f"Warning: Dataset contains label {label}, which is not in the set of training labels: {label_idx.keys()}")
                return d
            return l
        
        example_label_id_list = [default_label(label,0)
                                        for label in example_label_list]

        example_label_id = torch.tensor(example_label_id_list, dtype=torch.long)  # [num_examples]
        # Generate one-hot encoding for labels
        example_label_one_hot = nn.functional.one_hot(example_label_id, num_classes=len(classes)).to(torch.float32)

        # print(f'label_one_hot={example_label_one_hot.shape}, label_id={example_label_id.shape}')
        # print(f'labels: example_label_id',example_label_id)
        return example_label_one_hot, example_label_id



def conv_grades(example_grades_list:list[list[Any]], grades:list[int], grade_idx:Dict[Any,int])->Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        def default_grade(grade, d):
            g = grade_idx.get(grade)
            if (g is None):
                if (grade != -1):
                    print(f"Warning: Dataset contains grade {grade}, which is not in the set of training grades: {grade_idx.keys()}")
                return d
            return g
        
        # print("example_grades_list", example_grades_list)
        
        example_grade_id_list = [ [ default_grade(grade,0) for grade in grade_list]
                                        for grade_list in example_grades_list]
        example_grade_id = torch.tensor(example_grade_id_list, dtype=torch.long)  # [num_examples, num_seq]

        example_grade_valid_list = [ [(grade in grade_idx) for grade in grade_list]
                                        for grade_list in example_grades_list]
        example_grade_valid = torch.tensor(example_grade_valid_list, dtype=torch.bool)  # [num_examples, num_seq]

        # Generate one-hot encoding for labels
        example_grade_one_hot = nn.functional.one_hot(example_grade_id, num_classes=len(grades)).to(torch.float32)

        # print(f'grade_one_hot={example_label_one_hot.shape}, label_id={example_label_id.shape}')
        # print(f'grades: example_grades_id',example_label_id)
        return example_grade_one_hot, example_grade_id, example_grade_valid

def store_dataset(db: EmbeddingDb, exp_name:str, split_name:str, classification_items:List[ClassificationItemId], metadata:Dict[str,Any]):
    if not db.read_only:
        DATASET_SCHEMA = \
            '''---sql
            CREATE SEQUENCE IF NOT EXISTS dataset_id_seq START 1;
            CREATE TABLE IF NOT EXISTS dataset (
                dataset_id INTEGER PRIMARY KEY DEFAULT nextval('dataset_id_seq'),
                exp_name TEXT,
                split_name TEXT,
                metadata JSON,
                UNIQUE (exp_name, split_name),
            );
            CREATE TABLE IF NOT EXISTS dataset_item  (
                dataset_id INTEGER REFERENCES dataset(dataset_id),
                classification_item_id INTEGER REFERENCES classification_item(classification_item_id),
                UNIQUE (dataset_id, classification_item_id),
            );
            '''
        
        db.db.execute(DATASET_SCHEMA)

        db.db.execute('''---sql
                    INSERT INTO dataset (exp_name, split_name, metadata)
                    VALUES (?,?,?)
                    ON CONFLICT DO NOTHING
                    RETURNING dataset_id
                    ''', (exp_name, split_name, metadata,))
        res = db.db.fetchone()

        if res is not None:
            ds_id, = res
            db.db.executemany('''---sql
                    INSERT INTO dataset_item (dataset_id, classification_item_id)
                    VALUES (?,?)
                    ON CONFLICT DO NOTHING
                    ''', 
                    [(ds_id, classification_item_id,) for classification_item_id in classification_items]
                    )

    else:
        print("Cannot store dataset as Embedding DB is read only.")



def compute_class_weights_mc(example_label_id_train:torch.Tensor, classes:list[int])-> torch.Tensor:
    num_classes = len(classes)
    class_counts = torch.bincount(example_label_id_train[example_label_id_train>-1], minlength=num_classes)  # [num_classes]

    # Compute total number of examples
    num_examples = example_label_id_train.size(0)

    # Compute class weights: Inverse of class frequency
    class_weights_mc = num_examples / (class_counts * num_classes)

    class_weights_mc = torch.where(torch.isinf(class_weights_mc), torch.tensor(0.0), class_weights_mc)

    # Convert to PyTorch tensor (if not already)
    class_weights_mc = (class_weights_mc+0.01).to(torch.float32)

    print("Multi-class class weights:", class_weights_mc)
    return class_weights_mc


def compute_class_weights_ml(example_label_one_hot_train:torch.Tensor)->torch.Tensor:
    # Compute the number of positive examples per class
    positive_counts = example_label_one_hot_train.sum(dim=0)  # [num_classes]

    # Compute total number of examples
    num_examples = example_label_one_hot_train.size(0)

    # Compute class weights: Inverse of class frequency
    class_weights_ml = (num_examples - positive_counts) / positive_counts

    # Convert to PyTorch tensor (if not already)
    class_weights_ml = class_weights_ml.to(torch.float32)

    class_weights_ml = torch.where(torch.isinf(class_weights_ml), torch.tensor(0.0), class_weights_ml)
    class_weights_ml = class_weights_ml+0.01

    print("Multi-label class weights:", class_weights_ml)
    return class_weights_ml

def create_dataset(embedding_db:EmbeddingDb
                   , prompt_class:str
                   , query_list: Optional[List[str]]
                   , max_queries:Optional[int]
                   , max_paragraphs:Optional[int]
                   , max_token_len:Optional[int]
                   , align: Optional[Align]
                   , sequence_mode:SequenceMode
                   , split_same_query: bool = False
                   , use_grade_as_embedding:bool = False
                   , exp_name:str = ""
                   , fold: int = 0
                   )->Tuple[Dataset, Dataset, List[int], List[int], Dict[int,Any], Dataset
                            , torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    queries = None
    if query_list:
        # print("queries_list", query_list)
        queries_fromdb:Set[str] = get_queries(db=embedding_db)
        queries = queries_fromdb[queries_fromdb.isin(query_list)]
    else:
        queries = list(get_queries(db=embedding_db))[:max_queries]

        
    classification_items_train:List[ClassificationItemId]
    classification_items_test:List[ClassificationItemId]
    classification_items_predict:List[ClassificationItemId]

    train_offset = fold % 2
    test_offset = (1+fold) % 2

    if split_same_query:
        classification_items_train =[ example for query in queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[train_offset:max_paragraphs:2] 
                                    ]
        classification_items_test = [ example for query in queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[test_offset:max_paragraphs:2] 
                                    ]
        classification_items_predict = [ example for query in queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[test_offset::2] 
                                    ]
        print(f"queries: {list(queries)}")
    else:
        train_queries = queries[train_offset::2]
        test_queries = queries[test_offset::2]
        classification_items_train =[ example for query in train_queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[:max_paragraphs] 
                                    ]
        classification_items_test = [ example for query in test_queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[:max_paragraphs] 
                                    ]
        classification_items_predict = [ example for query in test_queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[:] 
                                    ]

        print(f"train_queries: {list(train_queries)}")
        print(f"test_queries: {list(test_queries)}")


    # query, passage, labels
    classification_data_train:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_train)
    classification_data_test:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_test)
    # for a real world case, we should not rely on judgments for predict, use: lookup_queries_paragraphs
    classification_data_predict:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_predict)

    classification_items_train = classification_data_train["classification_item_id"].to_list()
    classification_items_test = classification_data_test["classification_item_id"].to_list()
    classification_items_predict = classification_data_predict["classification_item_id"].to_list()

    store_dataset(embedding_db, exp_name = exp_name, split_name="train", classification_items= classification_items_train, metadata={})
    store_dataset(embedding_db, exp_name = exp_name, split_name="test", classification_items= classification_items_test, metadata={})
    store_dataset(embedding_db, exp_name = exp_name, split_name="predict", classification_items= classification_items_predict, metadata={})

    # needles.i,
    # array_agg(cf.tensor_id) AS tensor_ids,
    # array_agg(cf.metadata->>'$.test_bank') AS test_bank_ids,
    # array_agg(eg.self_rating) AS self_ratings,
    # array_agg(eg.is_correct) AS correctness,
    # la.true_labels AS judgment,
    # ci.classification_item_id
    train_tensors_with_grades_labels:pd.DataFrame = get_tensor_ids_with_grades(embedding_db
                                                , classification_item_id=classification_items_train
                                                , prompt_class= prompt_class)
    test_tensors_with_grades_labels:pd.DataFrame = get_tensor_ids_with_grades(embedding_db
                                               , classification_item_id=classification_items_test
                                               , prompt_class= prompt_class)
    
    predict_tensors_with_grades_labels:pd.DataFrame = get_tensor_ids_with_grades(embedding_db
                                               , classification_item_id=classification_items_predict
                                               , prompt_class= prompt_class)
    
    # predict_tensors_with_grades_labels:pd.DataFrame = get_tensor_ids_plain(embedding_db
    #                                            , classification_item_id=classification_items_predict
    #                                            , prompt_class= prompt_class)
    

    example_label_list_train = [d.true_labels[0] for d in  classification_data_train.itertuples() ]
    example_label_list_test = [d.true_labels[0] for d in  classification_data_test.itertuples() ]
    example_label_list_predict = [d.true_labels[0] for d in  classification_data_predict.itertuples() ]
    classes = sorted(list(set(example_label_list_train)))
    label_idx = {c:i for i,c in enumerate(classes)}


    def filtered_grade(self_rating:int, judgment:List[str])->int:
        label = int(judgment[0])
        if label <= 0:
            return 0
        elif self_rating>=4:
            return self_rating
        else:
            return -1
        
    def plain_grade(self_rating:int, judgment: List[str])->int:
        return self_rating

    # example_grades_list_train = [[filtered_grade(r, tup.judgment)  # TODO WAS
    example_grades_list_train = [[plain_grade(r, tup.judgment) 
                                    for r in tup.self_ratings] 
                                        for tup in train_tensors_with_grades_labels.itertuples()]
    example_grades_list_test = [[plain_grade(r, tup.judgment)  
                                    for r in tup.self_ratings] 
                                    for tup in test_tensors_with_grades_labels.itertuples()]
    example_grades_list_predict = [[plain_grade(r, tup.judgment)  
                                    for r in tup.self_ratings] 
                                    for tup in predict_tensors_with_grades_labels.itertuples()]
    
    # example_grades_list_test = [tup.self_ratings for tup in test_tensors_with_grades_labels.itertuples()]

    grade_idx = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5} # Adding -1 as "Missing"
    grades = list(grade_idx.values())
    sorted(grades)

    
    # example_label_one_hot_train = [examples][seq_idx][num_grades]
    example_label_one_hot_train, example_label_id_train = conv_class(example_label_list_train, classes=classes, label_idx=label_idx)
    example_label_one_hot_test, example_label_id_test = conv_class(example_label_list_test, classes=classes, label_idx=label_idx)
    example_label_one_hot_predict, example_label_id_predict = conv_class(example_label_list_predict, classes=classes, label_idx=label_idx)
    example_grades_one_hot_train, example_grades_id_train,example_grades_valid_train = conv_grades(example_grades_list_train, grades=grades, grade_idx=grade_idx)
    example_grades_one_hot_test, example_grades_id_test, example_grades_valid_test = conv_grades(example_grades_list_test, grades=grades, grade_idx=grade_idx)
    example_grades_one_hot_predict, example_grades_id_predict, example_grades_valid_predict= conv_grades(example_grades_list_predict, grades=grades, grade_idx=grade_idx)
    

    dataset_embedding_train=None
    dataset_embedding_test=None
    dataset_embedding_predict=None
    if use_grade_as_embedding:
        dataset_embedding_train = GradeClassificationItemDataset(tensor_df=train_tensors_with_grades_labels)
        dataset_embedding_test = GradeClassificationItemDataset(tensor_df=test_tensors_with_grades_labels)
        dataset_embedding_predict = GradeClassificationItemDataset(tensor_df=predict_tensors_with_grades_labels)
    else:
        dataset_embedding_train = ClassificationItemDataset(db=embedding_db, tensor_df=train_tensors_with_grades_labels, sequence_mode=sequence_mode, max_token_len=max_token_len, align=align)
        dataset_embedding_test = ClassificationItemDataset(db=embedding_db, tensor_df=test_tensors_with_grades_labels, sequence_mode=sequence_mode, max_token_len=max_token_len, align=align)
        dataset_embedding_predict = ClassificationItemDataset(db=embedding_db, tensor_df=predict_tensors_with_grades_labels, sequence_mode=sequence_mode, max_token_len=max_token_len, align=align)



    print("dataset_embedding_train", dataset_embedding_train[0])


    train_ds = EmbeddingStackDataset(embedding=dataset_embedding_train
                                     , label_one_hot=example_label_one_hot_train
                                     , label_id=example_label_id_train
                                     , grades_one_hot=example_grades_one_hot_train
                                     , grades_id=example_grades_id_train
                                     , grades_valid = example_grades_valid_train
                                    , classification_item_id=classification_items_train)
    test_ds = EmbeddingStackDataset(embedding=dataset_embedding_test
                                    , label_one_hot=example_label_one_hot_test
                                    , label_id=example_label_id_test
                                    , grades_one_hot=example_grades_one_hot_test
                                    , grades_id=example_grades_id_test
                                    , grades_valid = example_grades_valid_test
                                    , classification_item_id=classification_items_test)
    predict_ds = EmbeddingStackDataset(embedding=dataset_embedding_predict
                                    , label_one_hot=example_label_one_hot_predict
                                    , label_id=example_label_id_predict
                                    , grades_one_hot=example_grades_one_hot_predict
                                    , grades_id=example_grades_id_predict
                                    , grades_valid = example_grades_valid_predict
                                    , classification_item_id=classification_items_predict)
    
    # predict_ds = EmbeddingPredictStackDataset(embedding=dataset_embedding_predict
    #                                           , classification_item_id=classification_items_predict)
    


    class_ids =  [label_idx[c] for c in classes]
    grade_ids =  [grade_idx[g] for g in grades]
    label_lookup = {v:k for k,v in label_idx.items()}

    
    class_weights_mc = compute_class_weights_mc(example_label_id_train=example_label_id_train, classes=class_ids)
    class_weights_ml = compute_class_weights_ml(example_label_one_hot_train=example_label_one_hot_train)


    grade_weights_mc = compute_class_weights_mc(example_label_id_train=example_grades_id_train, classes=grade_ids)
    grade_weights_ml = compute_class_weights_ml(example_label_one_hot_train=example_grades_one_hot_train)

    return (train_ds, test_ds, class_ids, grades, label_lookup, predict_ds, class_weights_mc, class_weights_ml, grade_weights_mc, grade_weights_ml)

class CachingDataset(Dataset):
    def __init__(self, cachee:Dataset):
        self.data:Dict[int,Any] = dict()
        self.cachee:Dataset = cachee
        self.length:int = len(cachee)

    def __getitem__(self, index):
        item = self.data.get(index)
        if item is None:
            item = self.cachee[index]
            self.data[index]=item
        return item

    def __len__(self):
        return self.length

class PreloadedDataset(Dataset):
    def __init__(self, cachee:Dataset):
        self.data = []
        for i in range(0,len(cachee)):
            data_item = cachee[i]
            self.data.append(data_item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class TrainingTimer:
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        print(f"{self.message}: {TrainingTimer.elapsed_time_str(self.end_time-self.start_time)}")

    @staticmethod
    def elapsed_time_str(elapsed_time:int)-> str:
        days = int(elapsed_time // (24 * 3600))
        hours = int((elapsed_time % (24 * 3600)) // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


def main(cmdargs=None) -> None:
    import argparse

    sys.stdout.reconfigure(line_buffering=True)


    print("EXAM embed train")
    desc = f'''EXAM Embed Train
             '''
    

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--embedding-db', type=str, metavar='PATH', help='Path for the database directory for recording embedding vectors')
    parser.add_argument('--rubric-db', type=str, metavar='PATH', help='Path for the database directory for rubric paragraph data')
    parser.add_argument('--db-write', action="store_true", help='If set,open embedding DB in write mode, to store dataset info', default=False)

    parser.add_argument('--exp-name', type=str, metavar='str', help='Name of the experiment (used to store dataset)')
    parser.add_argument('--fold', type=int, metavar="DIM", help="Which fold to use, currently 0 or 1", default=0)
    parser.add_argument('--load-model-path', type=str, metavar='str', help='Name of *pt file from which to load the model. Note that it must be instantiated with exactly the same parameters.')

    parser.add_argument('-o', '--root', type=str, metavar="FILE", help='Directory to write training output to', default=Path("./attention_classify"))
    parser.add_argument('--exp-db', type=str, metavar='PATH', help='Path for experiment DB')
    parser.add_argument('--device', type=str, metavar="FILE", help='Device to run on, cuda:0 or cpu', default=Path("cuda:0"))
    parser.add_argument('--epochs', type=int, metavar="T", help="How many epochs to run training for", default=30)
    parser.add_argument('--batch-size', type=int, metavar="S", help="Batchsize for training", default=128)

    parser.add_argument('--snapshots-every', type=int, metavar="T", help="Take a model shapshort every T epochs")
    parser.add_argument('--snapshots-best-after', type=int, metavar="T", help="Take a model shapshort when target metric is improved (but only after the T'th epoch, and only during evaluation epochs)")
    parser.add_argument('--snapshots-target-metric', type=str, default="roc_auc", metavar="METRIC", help="Target evaluation metric for --snapshots-best-after, such as 'roc_auc'")
    parser.add_argument('--eval-every', type=int, metavar="T", help="Take a model shapshort every T epochs", default=1)

    parser.add_argument('--inner-dim', type=int, metavar="DIM", help="Use DIM as hidden dimension", default=64)
    parser.add_argument('--nhead', type=int, metavar="N", help="Use transformer with N heads", default=1)

    parser.add_argument('--label-problem-type', type=ProblemType.from_string, required=True, choices=list(ProblemType), metavar="MODEL"
                        , help="The classification problem to use for label prediction. Choices: "+", ".join(list(x.name for x in ProblemType)))
    parser.add_argument('--grade-problem-type', type=ProblemType.from_string, required=True, choices=list(ProblemType), metavar="MODEL"
                        , help="The classification problem to use for grade prediction. Choices: "+", ".join(list(x.name for x in ProblemType)))
    parser.add_argument('--no-transformers', dest="use_transformers", action="store_false", help='If set, replaces the transformer layer with an mean pooling', default=True)
    parser.add_argument('--2-transformers', dest="two_transformers", action="store_true", help='If set, uses two lauers of transformers', default=False)
    parser.add_argument('--no-inner-proj', dest="use_inner_proj", action="store_false", help='If set, will not perform an projection to inner_dimension, use llm_dim as is', default=True)
    parser.add_argument('--no-grade-signal', dest="use_grade_signal", action="store_false", help='If set, deactivates the grade signal for loss', default=True)
    parser.add_argument('--predict-grade-from-head', dest="predict_grade_from_class_logits", action="store_false", help='If set, will not predict grades from class logits but from the packed head directly.', default=True)
    parser.add_argument('--class-aggregation', type=str, default="max", metavar="AGGR", help="How to aggregate across classification heads of sequences. Choices: max, mean, argmax")


    parser.add_argument('--prompt-class', type=str, required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--class-model', type=attention_classify.ClassificationModel.from_string, required=True, choices=list(attention_classify.ClassificationModel), metavar="MODEL"
                        , help="The classification model to use. Choices: "+", ".join(list(x.name for x in attention_classify.ClassificationModel)))
    parser.add_argument('--overwrite', action="store_true", help='will automatically replace the output directory')
    parser.add_argument('--dry-run', action="store_true", help='will automatically replace the output directory')

    parser.add_argument('--max-queries', type=int, metavar="N", help="Use up to N queries")
    parser.add_argument('--queries', type=str, metavar="query_id", nargs='+', help="Use queries with these IDS")
    parser.add_argument('--max-paragraphs', type=int, metavar="N", help="Use to a total of N paragraphs across train and test")
    parser.add_argument('--split-same-query',action="store_true", help='Train/test split on same queries, but different paragraphs.')

    parser.add_argument('--max-token-len', type=int, metavar="N", help="Use up to N embedding tokens")
    parser.add_argument('--token-align', type=Align.from_string, required=True, choices=list(Align), metavar="MODE"
                        , help=f"Whether tokens are used from beginning or end. To use last --max-token-len tokens of embedding: set to {Align.ALIGN_END}. To use the first tokens, set to {Align.ALIGN_BEGIN}. Choices: "+", ".join(list(x.name for x in Align)))

    parser.add_argument('--sequence-mode',  type=SequenceMode.from_string, required=True, choices=list(SequenceMode), metavar="MODE", help=f'Select how to handle multiple sequences for classification. Choices: {list(SequenceMode)}')
    parser.add_argument('--use-grade-as-embedding', action="store_true", help='If set, ignored the LLM embeddings and uses a triagle-hot representation of the grade')
    parser.add_argument('--caching', action="store_true", help='Dataset: build in-memory cache as needed')
    parser.add_argument('--preloaded', action="store_true", help='Dataset: preload into memory')


    parser.add_argument('--export-rubric', type=str, metavar='*.jsonl.gz', help='Export predictions in RUBRIC format')
    parser.add_argument('--rubric-data', type=str, metavar='*.jsonl.gz', help='Input data in RUBRIC format (which will be annotated with predictions)')
    

  
 
    args = parser.parse_args(args = cmdargs) 

    train_ds = None
    test_ds = None
    class_list = None

    root = Path(args.root)
    out_dir = root / Path(args.exp_name) / Path(f"fold-{args.fold}")

    load_model_path = Path(args.load_model_path)  \
                            if args.load_model_path is not None \
                            else None





    embedding_db = EmbeddingDb(Path(args.embedding_db), write=args.db_write)

    embedding_db.db.execute(f'''--sql
                            ATTACH '{args.rubric_db}' as rubric (READ_ONLY)
                            ''')

    exp_db_path = args.exp_db
    if exp_db_path is None:
        exp_db_path = root / Path(args.exp_name) / Path("exp_db.duckdb")
    else:
        exp_db_path = Path(args.exp_db)
    exp_db_path.parent.mkdir(exist_ok=True)
    
    print(f"Creating experiment DB at {exp_db_path}")
    exp_db = duckdb.connect(exp_db_path)
    exp_db.execute('''--sql
                    CREATE TABLE IF NOT EXISTS prediction (
                    classification_item_id INTEGER PRIMARY KEY,
                    predicted_label text,
                    );
                    ''')
    exp_db.execute(f'''--sql
                    ATTACH '{args.rubric_db}' as rubric (READ_ONLY)
                    ''')
    exp_db.execute(f'''--sql
                    ATTACH '{args.embedding_db}/embeddings.duckdb' as embeddings (READ_ONLY)
                    ''')




    if args.export_rubric is not None:
        print("Post Training Predict and Export")
        if args.rubric_data is None:
            raise RuntimeError("Must give RUBRIC file for annotation with --rubric-data FILE")
        else:
            print(f"Annotating {args.rubric_data}, and writing to {args.export_rubric}")
        
        exp_db.execute( '''---sql
                        select ci.classification_item_id, predicted_label, ci.metadata->>'$.query', ci.metadata->>'$.passage'
                        from prediction
                        inner join embeddings.classification_item as ci 
                            on ci.classification_item_id = prediction.classification_item_id
                        ;
                    ''', None
                )
        df = exp_db.fetch_df()
        print(df)


        qfs = parseQueryWithFullParagraphs(args.rubric_data)
        for qf in qfs:
            print(f"Query: {qf.queryId}")
            for para in qf.paragraphs:
                # print(f"Paragraph: {para.paragraph_id}")

                if para.grades is None:
                    para.grades = list()
                
                exp_db.execute( \
                    '''---sql
                        select predicted_label
                        from prediction
                        inner join embeddings.classification_item as ci 
                            on ci.classification_item_id = prediction.classification_item_id
                        where CAST(ci.metadata->>'$.query' AS TEXT) = CAST(? AS TEXT)
                             and CAST(ci.metadata->>'$.passage' AS TEXT) = CAST(? AS TEXT)
                        ;
                    ''',
                    (qf.queryId,para.paragraph_id,)
                )
                result = exp_db.fetchone()
                if result is None:
                    # print(f"no prediction for {qf.queryId} / {para.paragraph_id}. Skipping ")
                    pass
                if result is not None:
                    (predicted_label_str,) = result
                    predicted_label = int(predicted_label_str)
 
                    print(f"{qf.queryId} / {para.paragraph_id} -> predicted label: {predicted_label} ")
                    para.grades.append(Grades(correctAnswered=predicted_label>0
                                            , answer=f"{predicted_label}"
                                            , llm="exam_embed_train"
                                            , prompt_info={"prompt_class":"exam_embed_train"}
                                            , self_ratings=predicted_label
                                            , prompt_type=DIRECT_GRADING_PROMPT_TYPE
                                            , relevance_label=predicted_label
                                            , llm_options={}
                    ))
        writeQueryWithFullParagraphs(file_path=args.export_rubric, queryWithFullParagraphList=qfs)
        print("Export complete")

        sys.exit()





    with TrainingTimer("Data Loading"):

        # embedding_db = EmbeddingDb(Path("embedding_db_classify/exam_grading"))
        (train_ds, test_ds, class_list, grades_list, label_lookup, predict_ds, class_weights_mc, class_weights_ml, grade_weights_mc, grade_weights_ml) = \
            create_dataset(embedding_db
                            , prompt_class=args.prompt_class
                            , query_list = args.queries
                            , max_queries=args.max_queries
                            , max_paragraphs=args.max_paragraphs
                            , max_token_len=args.max_token_len
                            , align=args.token_align
                            , sequence_mode=args.sequence_mode
                            , use_grade_as_embedding = args.use_grade_as_embedding
                            , split_same_query=args.split_same_query
                            , exp_name=args.exp_name
                            , fold = args.fold
                            )

        if args.caching:
            train_ds = CachingDataset(train_ds)
            test_ds = CachingDataset(test_ds)
        elif args.preloaded:
            train_ds = PreloadedDataset(train_ds)
            test_ds = PreloadedDataset(test_ds)


        print(f"Device: {args.device}")


        print(f"Train data: {len(train_ds)}")
        print(f"Test data: {len(test_ds)}")
        print(f"Predict data: {len(predict_ds)}")


    # print(f"Data loading took {elapsed_time_str(end_time - start_time)}.")

    predictions = list()

    def submit_predictions(items:torch.Tensor, pred_labels:torch.Tensor):
        for item, y_pred in zip(items, pred_labels):
            classification_item = item.item()
            predicted_label = label_lookup.get(y_pred.item())
            predictions.append( (classification_item, predicted_label))

        exp_db.executemany(
            '''---sql
            INSERT OR REPLACE INTO prediction
            (classification_item_id, predicted_label)
            VALUES (?,?)
            ''', [( classification_item_id.item(),
                   label_lookup.get(y_pred.item())) 
                    for classification_item_id, y_pred in  zip(items, pred_labels)
                 ])



    with TrainingTimer("Training"):
        # with ptp.profile(activities=[ptp.ProfilerActivity.CPU, ptp.ProfilerActivity.CUDA], with_stack=True) as prof:
        # attention_classify.run(root = root
        #                 , out_dir = out_dir
        #                 , overwrite=args.overwrite
        #                 , model_type=args.class_model
        #                 , train_ds=train_ds
        #                 , test_ds=test_ds
        #                 , class_list=class_list
        #                 , batch_size=args.batch_size
        #                 , snapshot_every=args.snapshots_every
        #                 , eval_every=args.eval_every
        #                 , n_epochs=args.epochs
        #                 , device_str=args.device
        #                 , inner_dim=args.inner_dim
        #                 , nhead=args.nhead
        #                 , epoch_timer = TrainingTimer("Epoch")
        #                 , snapshot_best_after= args.snapshots_best_after
        #                 , target_metric= args.snapshots_target_metric
        #                 )



        attention_classify.run_num_seqs(root = root
                        , out_dir=out_dir
                        , overwrite=args.overwrite
                        , model_type=args.class_model
                        , train_ds=train_ds
                        , test_ds=test_ds
                        , predict_ds=predict_ds
                        , class_list=class_list
                        , class_weights_mc=class_weights_mc
                        , class_weights_ml=class_weights_ml 
                        , grades_list=grades_list
                        , grade_weights_mc=grade_weights_mc
                        , grade_weights_ml=grade_weights_ml 
                        , batch_size=args.batch_size
                        , snapshot_every=args.snapshots_every
                        , eval_every=args.eval_every
                        , n_epochs=args.epochs
                        , device_str=args.device
                        , inner_dim=args.inner_dim
                        , nhead=args.nhead
                        , epoch_timer = TrainingTimer("Epoch")
                        , snapshot_best_after= args.snapshots_best_after
                        , target_metric= args.snapshots_target_metric
                        , label_problem_type=args.label_problem_type
                        , grade_problem_type=args.grade_problem_type
                        , use_transformer=args.use_transformers
                        , two_transformers = args.two_transformers
                        , use_inner_proj=args.use_inner_proj
                        , use_grade_signal= args.use_grade_signal
                        , load_model_path= load_model_path
                        , submit_predictions=submit_predictions
                        , predict_grade_from_class_logits = args.predict_grade_from_class_logits
                        , aggregation=args.class_aggregation
                        )


        # prof.export_chrome_trace('profile.json')

    print(predictions)
if __name__ == '__main__':
    main()
