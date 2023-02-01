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

print(sys.path)
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

print(1)
from scipy.io.wavfile import write
import whisper


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


if __name__ == "__main__":
    # arg parse
    parser = argparse.ArgumentParser(description='Text 2 Speech 2 Text')
    parser.add_argument('--text', '-t', type=str, help='input text', default='안녕 난 에도가와 코난 탐정이지')
    parser.add_argument('--wav', '-w', type=str, help='output wav', default='output.wav')
    parser.add_argument('--output', '-ot', type=str, help='output text', default='output.txt')
    args = parser.parse_args()


    # VITS
    hps = utils.get_hparams_from_file("/home/ubuntu/alpaco/anichat/tts/vits/configs/conan_base_ms.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
        
    _ = net_g.eval()
    
    _ = utils.load_checkpoint("/home/ubuntu/alpaco/mento_log/G_233000.pth", net_g, None)
    
    stn_tst = get_text(args.text, hps)

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

    with open(args.output, 'w') as f:
        f.write(result["text"])

    print('done!')