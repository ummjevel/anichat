from django.apps import AppConfig
'''
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
import utils
import commons
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import whisper
'''
class WebchatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'webchat'
 