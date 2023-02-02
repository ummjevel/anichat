# 2기
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle
import pandas as pd

def make_data_set(file_path, num):
    with open(file_path, 'rb') as fr:
        df = pickle.load(fr)

    data_source = []

    for idx in tqdm(range(df.shape[0])):
        dict_data_source = {
            'context':[],
            'responses':[],
            'labels': [1] + [0] * num
        }

        list_talk = list(df.iloc[idx])
        no_nan = [i for i in list_talk if pd.isnull(i) == False]
        id = no_nan[0]
        talk = no_nan[1:-1]
        answer = no_nan[-1]

        for i in range(len(talk)):
            dict_data_source['context'].append(talk[i].strip())
  
        cond_id = (df['index']==id)

        while True:
            dict_data_source['responses'] = []
            dict_data_source['responses'].append(answer.strip())

            wrong_df = df.loc[~cond_id].sample(num)

            for wrong_idx in range(num):
                list_wrong = list(wrong_df.iloc[wrong_idx])
                wrong_no_nan = [i for i in list_wrong if pd.isnull(i) == False]
                wrong = wrong_no_nan[-1].strip()
                dict_data_source['responses'].append(wrong.strip())
            list_responses = dict_data_source['responses']

            if len(list_responses) != len(set(list_responses)):
                pass
            else:                
                break
        data_source.append(dict_data_source)
    return data_source

class SelectionDataset(Dataset):
    def __init__(self, file_path, context_transform, response_transform, concat_transform, mode='poly', num=15):
        self.context_transform = context_transform
        self.response_transform = response_transform
        self.concat_transform = concat_transform
        self.mode = mode
        self.num=num
        self.data_source = make_data_set(file_path, self.num)
        
    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        group = self.data_source[index]
        context, responses, labels = group['context'], group['responses'], group['labels']
        if self.mode == 'cross':
            transformed_text = self.concat_transform(context, responses)
            ret = transformed_text, labels
        else:
            transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
            transformed_responses = self.response_transform(responses)  # [token_ids],[seg_ids],[masks]
            ret = transformed_context, transformed_responses, labels

        return ret

    def batchify_join_str(self, batch):
        if self.mode == 'cross':
            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [], [], []
            labels_batch = []
            for sample in batch:
                text_token_ids_list, text_input_masks_list, text_segment_ids_list = sample[0]

                text_token_ids_list_batch.append(text_token_ids_list)
                text_input_masks_list_batch.append(text_input_masks_list)
                text_segment_ids_list_batch.append(text_segment_ids_list)

                labels_batch.append(sample[1])

            long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]

            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch

        else:
            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = [], [], [], []
            labels_batch = []
            for sample in batch:
                (contexts_token_ids_list, contexts_input_masks_list), (responses_token_ids_list, responses_input_masks_list) = sample[:2]

                contexts_token_ids_list_batch.append(contexts_token_ids_list)
                contexts_input_masks_list_batch.append(contexts_input_masks_list)

                responses_token_ids_list_batch.append(responses_token_ids_list)
                responses_input_masks_list_batch.append(responses_input_masks_list)

                labels_batch.append(sample[-1])

            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch,
                                            responses_token_ids_list_batch, responses_input_masks_list_batch]

            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
                          responses_token_ids_list_batch, responses_input_masks_list_batch, labels_batch
