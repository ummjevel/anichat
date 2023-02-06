import argparse
import matplotlib.pyplot as plt

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys
from os import path

sys.path.append("/home/ubuntu/alpaco/anichat/tts/vits")

# print(sys.path)
# /home/ubuntu/alpaco/anichat/tts/vits/test.py
# /home/ubuntu/alpaco/anichat/utils/text2speech2text.py

import utils
#print(path.dirname( path.dirname( path.abspath(__file__) ) ))
#

import commons
#from ..tts.vits import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import whisper

sys.path.append("/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference")

import pickle
# import pandas as pd
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from encoder import PolyEncoder
from transform import SelectionJoinTransform


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def context_input(context):
    context_input_ids, context_input_masks = context_transform(context)
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
    long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)
    return contexts_token_ids_list_batch, contexts_input_masks_list_batch

def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):
    with torch.no_grad():
        chatbot.eval()
        ctx_out = chatbot.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(chatbot.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, chatbot.poly_m)
        poly_codes = chatbot.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = chatbot.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
        return embs

def score(embs, cand_emb):
    with torch.no_grad():
        chatbot.eval()
        ctx_emb = chatbot.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb*cand_emb).sum(-1)
        return dot_product

if __name__ == "__main__":
    # arg parse
    parser = argparse.ArgumentParser(description='chatbot 2 Text 2 Speech 2 Text')
    parser.add_argument('--wav', '-w', type=str, help='output wav', default='output.wav')
    parser.add_argument('--output', '-ot', type=str, help='output text', default='output.txt')

    parser.add_argument("-p","--model_path", default='../chatbot/chatbot_only_inference/poly_16_pytorch_model.bin', help='model file for inference')
    parser.add_argument("-bm","--base_model", default='klue/roberta-base', help='transformers model name/ .from_pretrained(bert model name)')
    parser.add_argument("-idf","--infer_df", default='../chatbot/chatbot_only_inference/inference_df.pickle' , help='data frame path for infer respone')
    parser.add_argument("-ced","--infer_emb_data", default='../chatbot/chatbot_only_inference/inference_cand_embs.pickle' , help='data path for make infer respone')
    parser.add_argument("--poly_m", default=16, type=int, help="Number of m of polyencoder")
    parser.add_argument("-ql","--question_length", default=256, type=int, help='max length for question')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_config = RobertaConfig.from_pretrained(args.base_model)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.base_model)
    base_model = RobertaModel.from_pretrained(args.base_model, config=model_config)

    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.question_length)

    chatbot=PolyEncoder(model_config, bert=base_model, poly_m=args.poly_m)
    chatbot.resize_token_embeddings(len(tokenizer))
    chatbot.load_state_dict(torch.load(args.model_path))
    chatbot.to(device)

    # chatbot
    with open(args.infer_df,'rb') as fr:
        infer_df=pickle.load(fr)
    with open(args.infer_emb_data,'rb') as fr:
        cand_embs=pickle.load(fr)
    
    while True:
        text=input("Please input text, if you want exit, please input '(exit)'\n>>> ") # 텍스트 입력란
        query = [text.strip()]
        if query == ['(exit)']: # chatbot 종료조건
            print('>>>    Shut down the process    <<<')
            print('>>> Thank you for using anichat <<<')
            break
        else:
            embs = embs_gen(*context_input(query))
            s = score(embs, cand_embs).to('cpu')
            idx = s.argmax(1)
            idx = int(idx[0])
            best_answer = infer_df['response'][idx][0]
            print(best_answer)

            # VITS
            hps = utils.get_hparams_from_file("/home/ubuntu/yglee/tts_test/conan_base.json")

            net_g = SynthesizerTrn(
                len(symbols),
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model).cuda()
                
            _ = net_g.eval()
            
            _ = utils.load_checkpoint("/home/ubuntu/yglee/tts_test/G_600000.pth", net_g, None)

            stn_tst = get_text(best_answer, hps)



            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                sid = torch.LongTensor([0]).cuda()
                print(sid)
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            
            write(args.wav, hps.data.sampling_rate, audio)

            # Whisper
            model = whisper.load_model("base")
            result = model.transcribe(args.wav)
            print(result["text"])

            with open(args.output, 'w', encoding='utf8') as f:
                f.write(result["text"])