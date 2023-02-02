import os
import numpy as np
import pickle
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from encoder import PolyEncoder
from transform import SelectionJoinTransform, SelectionSequentialTransform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("-p","--model_path", required=True, help='model file for inference')
    parser.add_argument("-bm","--base_model", default='klue/roberta-base', help='transformers model name/ .from_pretrained(bert model name)')
    parser.add_argument("-rd","--raw_data", default='./raw_df.pickle' , help='data path for make inference data generation')
    parser.add_argument("-idf","--infer_df", default='./inference_df.pickle' , help='data frame path for infer respone')
    parser.add_argument("-ced","--infer_emb_data", default='./inference_cand_embs.pickle' , help='data path for make infer respone')
    
    parser.add_argument("--poly_m", default=16, type=int, help="Number of m of polyencoder")
    parser.add_argument("-ql","--question_length", default=256, type=int, help='max length for question')
    parser.add_argument("-al","--answer_length", default=256, type=int, help='max length for answer')

    parser.add_argument("--make_inference_data", action='store_true', help="train_df to make inference data")
    parser.add_argument("--save_path", default='./', help='inference data save path')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_config = RobertaConfig.from_pretrained(args.base_model)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.base_model)
    base_model = RobertaModel.from_pretrained(args.base_model, config=model_config)

    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.question_length)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.answer_length)

    model=PolyEncoder(model_config, bert=base_model, poly_m=args.poly_m)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.device

def context_input(context):
    context_input_ids, context_input_masks = context_transform(context)
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]

    long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]

    contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)

    return contexts_token_ids_list_batch, contexts_input_masks_list_batch

def response_input(candidates):
    responses_token_ids_list, responses_input_masks_list = response_transform(candidates)
    responses_token_ids_list_batch, responses_input_masks_list_batch = [responses_token_ids_list], [responses_input_masks_list]

    long_tensors = [responses_token_ids_list_batch, responses_input_masks_list_batch]

    responses_token_ids_list_batch, responses_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)

    return responses_token_ids_list_batch, responses_input_masks_list_batch

def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):

    with torch.no_grad():
        model.eval()
        
        ctx_out = model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(model.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, model.poly_m)
        poly_codes = model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        return embs

def cand_emb_gen(responses_token_ids_list_batch, responses_input_masks_list_batch):

    with torch.no_grad():
        model.eval()
                
        batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape # res_cnt is 1 during training
        responses_token_ids_list_batch = responses_token_ids_list_batch.view(-1, seq_length)
        responses_input_masks_list_batch = responses_input_masks_list_batch.view(-1, seq_length)
        cand_emb = model.bert(responses_token_ids_list_batch, responses_input_masks_list_batch)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        return cand_emb

def loss(embs, cand_emb, contexts_token_ids_list_batch, responses_token_ids_list_batch):
    batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape

    ctx_emb = model.dot_attention(cand_emb, embs, embs) # [bs, bs, dim]
    # print(ctx_emb)
    ctx_emb = ctx_emb.squeeze()
    # print(ctx_emb)
    dot_product = (ctx_emb*cand_emb) # [bs, bs]
    # print(dot_product)
    dot_product = dot_product.sum(-1)
    # print(dot_product)
    mask = torch.eye(batch_size).to(contexts_token_ids_list_batch.device) # [bs, bs]
    # print(mask)
    loss = F.log_softmax(dot_product, dim=-1)
    # print(loss)
    loss = loss * mask
    # print(loss)
    loss = (-loss.sum(dim=1))
    # print(loss)
    loss = loss.mean()
    print(loss)
    return loss

def score(embs, cand_emb):
    with torch.no_grad():
        model.eval()

        ctx_emb = model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb*cand_emb).sum(-1)
        
        return dot_product

def inference(query, df, cand_embs):
    with torch.no_grad():
        cand_embs = cand_embs.to(device)
        embs = embs_gen(*context_input(query))
        s = score(embs, cand_embs)
        print(s)
        idx = s.argmax(1)
        print(idx)
        idx = int(idx[0])
        print(idx)

        return df['response'][idx]

if args.make_inference_data:
    try:
        with open(args.raw_data, 'rb') as fr:
            df=pickle.load(fr)
    except Exception as e:
        print('raw dataframe is not a pickle file.', e)

    with torch.no_grad():
        model.eval()
        response_input_srs = df['response'].apply(response_input)
        response_input_lst = response_input_srs.to_list()
        cand_embs_lst = []
        for i in tqdm(response_input_lst):
            cand_embs_lst.append(cand_emb_gen(*i).to('cpu'))

        df['response embedding'] = cand_embs_lst
        infer_df=df[['response', 'response embedding']]

        save_path=os.path.join(args.save_path,'inference_df.pickle')
        with open(save_path, 'wb') as fw:
            pickle.dump(infer_df, fw)

        cand_embs = cand_embs_lst[0]
        for idx in tqdm(range(1, len(cand_embs_lst))):
            y = cand_embs_lst[idx]
            cand_embs = torch.cat((cand_embs, y), 1)

        save_path=os.path.join(args.save_path,'inference_cand_embs.pickle')
        with open(save_path, 'wb') as fw:
            pickle.dump(cand_embs, fw)

        print('Inference data, dataframe created successfully, save path:',args.save_path,': inference_cand_embs.pickle, inference_df.pickle')
# inference
else:
    try:
        with open(args.infer_df, 'rb') as fr:
            df=pickle.load(fr)
    except Exception as e:
        print('inference dataframe is not a pickle file.', e)
    
    try:
        with open(args.infer_emb_data, 'rb') as fr:
            cand_embs=pickle.load(fr)
    except Exception as e:
        print('inference candidate embedding data is not a pickle file.', e)

    while True:
        text=input("Please input text, if you want exit, please input '(exit)'\n>>> ")
        query = [text.strip()]
        if query == ['(exit)']:
            print('>>>    Shut down the process    <<<')
            print('>>> Thank you for using anichat <<<')
            break
        else:
            print(inference(query, df, cand_embs))