from flask import Flask, render_template, request, jsonify
import sys

# stt
import whisper

# tts

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys
from os import path
'''
sys.path.append("/home/ubuntu/alpaco/anichat/tts/vits")

import utils
import commons
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
'''

# chatbot

sys.path.append("/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference")

import pickle
# import pandas as pd
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from encoder import PolyEncoder
from transform import SelectionJoinTransform


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)


def context_input(context, context_transform):
    context_input_ids, context_input_masks = context_transform(context)
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
    long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)
    return contexts_token_ids_list_batch, contexts_input_masks_list_batch


def embs_gen(chatbot, contexts_token_ids_list_batch, contexts_input_masks_list_batch):
    with torch.no_grad():
        chatbot.eval()
        ctx_out = chatbot.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(chatbot.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, chatbot.poly_m)
        poly_codes = chatbot.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = chatbot.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
        return embs


def score(chatbot, embs, cand_emb):
    with torch.no_grad():
        chatbot.eval()
        ctx_emb = chatbot.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb*cand_emb).sum(-1)
        return dot_product


def initChatbot():
    base_model_name = 'klue/roberta-base'
    question_length = 256
    poly_m = 16
    model_path = '/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference/poly_16_pytorch_model.bin'
    infer_df_path = '/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference/inference_df.pickle'
    infer_emb_data_path = '/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference/inference_cand_embs.pickle'

    model_config = RobertaConfig.from_pretrained(base_model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name)
    base_model = RobertaModel.from_pretrained(base_model_name, config=model_config)

    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=question_length)

    chatbot=PolyEncoder(model_config, bert=base_model, poly_m=poly_m)
    chatbot.resize_token_embeddings(len(tokenizer))
    chatbot.load_state_dict(torch.load(model_path))
    chatbot.to(device)
    print('load chatbot', file=sys.stderr)

    with open(infer_df_path, 'rb') as fr:
        infer_df=pickle.load(fr)
    with open(infer_emb_data_path, 'rb') as fr:
        cand_embs=pickle.load(fr)

    print('opened pickle files', file=sys.stderr)

    return chatbot, context_transform, infer_df, cand_embs


def executeChatbot(chatbot, context_transform, infer_df, cand_embs, text):
    query = [text.strip()]
    embs = embs_gen(chatbot, *context_input(query, context_transform))
    s = score(chatbot, embs, cand_embs).to('cpu')
    idx = s.argmax(1)
    idx = int(idx[0])
    best_answer = infer_df['response'][idx][0]

    print(best_answer, file=sys.stderr)

    return best_answer


@app.route('/')
def hello_world():
    print('Hello world!', file=sys.stderr)
    initChatbot()
    return render_template('index.html')


@app.route('/webchat', methods=['GET', 'POST'])
def webchat():
    if request.method == 'POST':
        # json 형태로 풀기
        if request.is_json == True:
            params = request.get_json()
            if params.use_tts == 'true': # use tts
                print('tts 사용', file=sys.stderr)
            else: # use chatbot only
                print('chatbot만 사용', file=sys.stderr)
    return render_template('chatbot.html')

@app.route('/sendChat', methods=['POST'])
def sendChat():
    answer = '오류가 발생했습니다.'
    if request.method == 'POST':
        # json 형태로 풀기
        params = {}
        if request.is_json == True:
            params = request.get_json()
            print(params, file=sys.stderr)
            if params['use_tts'] == True: # use tts
                print('tts 사용', file=sys.stderr)
                
            else: # use chatbot only
                print('chatbot만 사용', file=sys.stderr)
                try:
                    answer = executeChatbot(chatbot, context_transform, infer_df, cand_embs, params['message'])
                except:
                    answer = '챗봇이 로드되지 않았습니다. 첫화면부터 다시 시도해주세요.'
            returns = jsonify({"message": answer, "use_tts": params['use_tts']})
        else:
            answer = '데이터 전달이 제대로 되지 않았습니다.'
            print('this is not json...', request.is_json, file=sys.stderr)
            returns = jsonify({"message": answer})
    return returns

@app.route('/c2')
def chatbot_web2():
    chatbot, context_transform, infer_df, cand_embs = initChatbot()
    answer = executeChatbot(chatbot, context_transform, infer_df, cand_embs, '란은 누구야?')
    return answer


@app.route('/c')
def chatbot_web():
    text = '진이 누구야?'
    base_model_name = 'klue/roberta-base'
    question_length = 256
    poly_m = 16
    model_path = '/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference/poly_16_pytorch_model.bin'
    infer_df_path = '/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference/inference_df.pickle'
    infer_emb_data_path = '/home/ubuntu/alpaco/anichat/chatbot/chatbot_only_inference/inference_cand_embs.pickle'

    model_config = RobertaConfig.from_pretrained(base_model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name)
    base_model = RobertaModel.from_pretrained(base_model_name, config=model_config)

    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=question_length)

    chatbot=PolyEncoder(model_config, bert=base_model, poly_m=poly_m)
    chatbot.resize_token_embeddings(len(tokenizer))
    chatbot.load_state_dict(torch.load(model_path))
    chatbot.to(device)
    print('load chatbot', file=sys.stderr)

    with open(infer_df_path, 'rb') as fr:
        infer_df=pickle.load(fr)
    with open(infer_emb_data_path, 'rb') as fr:
        cand_embs=pickle.load(fr)

    print('opened pickle files', file=sys.stderr)
    query = [text.strip()]
    embs = embs_gen(chatbot, *context_input(query, context_transform))
    s = score(chatbot, embs, cand_embs).to('cpu')
    idx = s.argmax(1)
    idx = int(idx[0])
    best_answer = infer_df['response'][idx][0]
    print(best_answer, file=sys.stderr)

    return best_answer


@app.route('/w')
def whisper2():
    print('in whisper2', file=sys.stderr)
    model = whisper.load_model("base")
    print('loaded model', file=sys.stderr)
    result = model.transcribe("/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/anichat_con_01_00007.wav",fp16=False, language='Korean')
    print(result["text"], file=sys.stderr)
    return result["text"]

'''
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


@app.route('/v')
def vits2():
    print('in vits2', file=sys.stderr)
    
    hps = utils.get_hparams_from_file("/home/ubuntu/yglee/tts_test/conan_base.json")
    print('load config', file=sys.stderr)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    print('load net_g', file=sys.stderr)
    _ = net_g.eval()
    
    _ = utils.load_checkpoint("/home/ubuntu/yglee/tts_test/G_600000.pth", net_g, None)
    print('load G_600000', file=sys.stderr)
   
    stn_tst = get_text("난 에도가와 코난 탐정이지", hps)
    print('get text', file=sys.stderr)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    print('no grad', file=sys.stderr)
    write('/home/ubuntu/yglee/tts_test/conan.wav', hps.data.sampling_rate, audio)
    print('write wav', file=sys.stderr)

'''

chatbot, context_transform, infer_df, cand_embs = initChatbot()


if __name__ == '__main__':
    print('run', file=sys.stderr)
    app.run(host="0.0.0.0", port=20000)