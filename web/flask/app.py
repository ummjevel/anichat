from flask import Flask, render_template, request, jsonify, session, send_file, url_for, redirect
import sys
import logging 
import random
from werkzeug.utils import secure_filename
import soundfile
import codecs
import pickle
import pandas as pd
import sys
import io
import os
from os import path
import traceback

app = Flask(__name__, instance_path='/Users/jeonminjeong/Documents/dev/anichat')
APP_PATH = os.path.join(app.instance_path, "web", "flask")
TTS_PATH = os.path.join(app.instance_path, "tts", "vits")
CHATBOT_PATH = os.path.join(app.instance_path, "chatbot", "chatbot_only_inference")

TTS_MODEL_PATH = {
    "conan": ("./static/model/conan_config.json"
            , "./static/model/conan_final.pth")
    , "you": ("./static/model/you_config.json"
            , "./static/model/you_final.pth")
}

MEMBER_FILE_PATH = "./static/data/members.pkl"
FEEDBACK_PATH = "./static/data/feedbacks.txt"
CHATBOT_BASE_MODEL_PATH = "./static/model/chatbot/roberta-base"
CHATBOT_MODEL_PATH = "./static/model/chatbot/model"


# stt
import whisper

# tts
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append(TTS_PATH)

import utils
import commons
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write


# chatbot
sys.path.append(CHATBOT_PATH)

from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from encoder import PolyEncoder
from transform import SelectionJoinTransform


device = torch.device("cpu") # ("cuda:0" if torch.cuda.is_available() else "cpu")


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


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def del_enter(x):
    return x.split('\n')[0].strip()


def initChatbot():

    question_length = 256
    poly_m = 8
    infer_df_path = 'inference_df.pickle'
    infer_emb_data_path = 'inference_cand_embs.pickle'

    tokenizer = RobertaTokenizerFast.from_pretrained(CHATBOT_BASE_MODEL_PATH)

    with open((os.path.join(CHATBOT_BASE_MODEL_PATH,'special_token.txt')),'r')as f:
        sp_lines=f.readlines()
    
    sp_list=list(map(del_enter,sp_lines[500:]))
    tokenizer.add_tokens(sp_list, special_tokens=True)

    model_config = RobertaConfig.from_pretrained(CHATBOT_BASE_MODEL_PATH)
    base_model = RobertaModel.from_pretrained(CHATBOT_BASE_MODEL_PATH, config=model_config)

    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=question_length)

    chatbot=PolyEncoder(model_config, bert=base_model, poly_m=poly_m)
    chatbot.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(os.path.join(CHATBOT_MODEL_PATH, 'pytorch_model.bin'), map_location=torch.device('cpu'))
    chatbot.load_state_dict(checkpoint['model'])
    chatbot.to(device)
    print('load chatbot', file=sys.stderr)

    with open(os.path.join(CHATBOT_MODEL_PATH, infer_df_path), 'rb') as fr:
        infer_df = CPU_Unpickler(fr).load()
        # infer_df=pickle.load(fr)
    with open(os.path.join(CHATBOT_MODEL_PATH, infer_emb_data_path), 'rb') as fr:
        cand_embs = CPU_Unpickler(fr).load()
        # cand_embs=pickle.load(fr)

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


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm



def initTTS(character):

    hps_path = TTS_MODEL_PATH[character][0]
    checkpoint_path = TTS_MODEL_PATH[character][1]

    print('in vits2', file=sys.stderr)
    
    hps = utils.get_hparams_from_file(hps_path)
    print('load config', file=sys.stderr)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    print('load net_g', file=sys.stderr)
    _ = net_g.eval()
    
    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    print('load G_600000', file=sys.stderr)
   
    return hps, net_g


def executeTTS(hps, net_g, text, output_path):
    stn_tst = get_text(text, hps)
    
    print('get text', file=sys.stderr)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    print('no grad', file=sys.stderr)
    write(output_path, 22050, audio)
    print('write wav', file=sys.stderr)
    return True


def checkLogin(username, pw):

    # open pickle file
    member = getLoginInfo(username)

    print(member, file=sys.stderr)
    # if there is no name and password than error
    if len(member) > 0:
        if member['password'][0] == pw:
            return True
        else:
            return False
    else:
        return False


def registerMember(username, password, name, email):
    print('this is register3', file=sys.stderr)
    # open pickle file
    members = pd.read_pickle(MEMBER_FILE_PATH)
    print(members, file=sys.stderr)
    # write pickle file
    data = [[username, password, name, email, True]]
    member = pd.DataFrame(data, columns=['username', 'password', 'name', 'email', 'first'])
    members = pd.concat([members, member])
    members.to_pickle(MEMBER_FILE_PATH)  

    print('register member: ', username, file=sys.stderr)
    # return success
    return True


def getLoginInfo(username):
    members = pd.read_pickle(MEMBER_FILE_PATH)
    print(members, file=sys.stderr)
    member = members[members['username'] == username]
    return member


def checkFirstLogin(username):

    member = getLoginInfo(username)

    if len(member) > 0:
        return member['first'][0]
    else:
        True


def setSecondLogin(username):
    members = pd.read_pickle(MEMBER_FILE_PATH)
    members.loc[members['username'] == username, 'first'] = False
    members.to_pickle(MEMBER_FILE_PATH) 


anichat_chatbot, context_transform, infer_df, cand_embs = initChatbot()
hps_conan, net_g_conan = initTTS('conan')
hps_you, net_g_you = initTTS('you')
whisper_model = whisper.load_model("base")
'''
chatbot, context_transform, infer_df, cand_embs = '', '', '', ''
hps, net_g = '', ''
whisper_model = ''
'''
from werkzeug.debug import DebuggedApplication
app.wsgi_app = DebuggedApplication(app.wsgi_app, True)

logging.basicConfig(level=logging.DEBUG) 


@app.route('/')
def hello_world():
    return redirect(url_for('login'))


@app.route('/webchat', methods=['GET', 'POST'])
def webchat():
    return render_template('webchat.html')

@app.route('/sendChat', methods=['POST'])
def sendChat():
    answer = '오류가 발생했습니다.'
    if request.method == 'POST':
        # json 형태로 풀기
        params = {}
        if request.is_json == True:
            params = request.get_json()
            text_message = params['message']
            print(params, file=sys.stderr)
            # use chatbot only
            print('chatbot만 사용', file=sys.stderr)
            try:
                answer = executeChatbot(anichat_chatbot, context_transform, infer_df, cand_embs, text_message)
            except Exception as e:
                answer = '챗봇이 로드되지 않았습니다. 첫화면부터 다시 시도해주세요.'
                print(e, file=sys.stderr)
                traceback.print_exc()
            # use tts
            wav_file_path = ''
            if params['use_tts'] == True:
                print('tts도 사용', file=sys.stderr)
                wav_file_path = '/static/record/tts_{0}.wav'.format(random.randint(0, 1000000))
                wav_file_front_path = APP_PATH
                if params['choose'] == 'conan':
                    hps = hps_conan
                    net_g = net_g_conan
                else:
                    hps = hps_you
                    net_g = net_g_you
                wavfile = executeTTS(hps, net_g, answer, wav_file_front_path + wav_file_path)
            returns = jsonify({"message": answer, "use_tts": params['use_tts'], 'wav_file': wav_file_path})
        else:
            answer = '데이터 전달이 제대로 되지 않았습니다.'
            print('this is not json...', request.is_json, file=sys.stderr)
            returns = jsonify({"message": answer})
    return returns


@app.route('/sendSTT', methods=['POST'])
def sendSTT():
    answer = '오류가 발생했습니다.'
    
    print(request.files, file=sys.stderr)
        # json 형태로 풀기
        
    if 'data' in request.files:
        file = request.files['data']
        filename = '/static/record/stt_{0}.wav'.format(random.randint(0, 1000000))
        filepath = APP_PATH + filename
        file.save(filepath)
        file.seek(0)
        # Read the audio data again.
        data, samplerate = soundfile.read(filepath)
        with io.BytesIO() as fio:
            soundfile.write(
                fio, 
                data, 
                samplerate=samplerate, 
                subtype='PCM_16', 
                format='wav'
            )
            data = fio.getvalue()
        
        audio = whisper.load_audio(filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        options = whisper.DecodingOptions(language="Korean", fp16=False)
        result = whisper.decode(whisper_model, mel, options)
        text_message = result.text
    
        print(text_message, file=sys.stderr)
        # use chatbot only
        print('chatbot만 사용', file=sys.stderr)
        try:
            answer = executeChatbot(anichat_chatbot, context_transform, infer_df, cand_embs, text_message)
        except Exception as e:
            answer = '챗봇이 로드되지 않았습니다. 첫화면부터 다시 시도해주세요.'
            print(e, file=sys.stderr)
            traceback.print_exc()
        # use tts
        wav_file_path = ''
        json_file_content = request.files['key'].read().decode('utf-8')
        #load the string readed into json object
        json_content = json.loads(json_file_content)
        print(json_content, file=sys.stderr)

        if json_content['use_tts'] == 'true':
            print('tts도 사용', file=sys.stderr)
            wav_file_path = '/static/record/tts_{0}.wav'.format(random.randint(0, 1000000))
            wav_file_front_path = APP_PATH
            if json_content['choose'] == 'conan':
                    hps = hps_conan
                    net_g = net_g_conan
            else:
                hps = hps_you
                net_g = net_g_you
            wavfile = executeTTS(hps, net_g, answer, wav_file_front_path + wav_file_path)
        
        returns = jsonify({"message": answer, "use_tts": json_content['use_tts'], 'wav_file': wav_file_path
                        , "question": text_message, "use_stt": "true"})
    else:
        answer = '데이터 전달이 제대로 되지 않았습니다.'
        print('this is not json...', request.is_json, file=sys.stderr)
        returns = jsonify({"message": answer})
    return returns


@app.route('/sendMimic', methods=['POST'])
def sendMimic():
    answer = '오류가 발생했습니다.'
    if request.method == 'POST':
        # json 형태로 풀기
        params = {}
        if request.is_json == True:
            params = request.get_json()
            text_message = params['message']
            # use tts
            wav_file_path = '/static/record/tts_{0}.wav'.format(random.randint(0, 1000000))
            wav_file_front_path = APP_PATH
            if params['choose'] == 'conan':
                hps = hps_conan
                net_g = net_g_conan
            else:
                hps = hps_you
                net_g = net_g_you
            wavfile = executeTTS(hps, net_g, text_message, wav_file_front_path + wav_file_path)
            returns = jsonify({"message": text_message, "use_tts": "true", 'wav_file': wav_file_path})
        else:
            answer = '데이터 전달이 제대로 되지 않았습니다.'
            print('this is not json...', request.is_json, file=sys.stderr)
            returns = jsonify({"message": answer})
    return returns

@app.route('/turnTTS', methods=['POST'])
def turnTTS():
    character = 'conan'
    print('this is json...', request.is_json, file=sys.stderr)
    params = request.get_json()
    print(params, file=sys.stderr)
    character = params['character']
    global hps
    global net_g
    hps, net_g = initTTS(character)
    return jsonify({"result": 'success'})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if "loginName" in request.form:
            loginName = request.form['loginName']
            loginPassword = request.form['loginPassword']

            returns = checkLogin(loginName, loginPassword)

            if returns:
                setSecondLogin(loginName)
                return render_template('select.html', username = loginName, first=returns)
            else:
                result = 'fail'

        else:
            print('this is register', file=sys.stderr)
            registerName = request.form['registerName']
            print(registerName, file=sys.stderr)
            registerUsername = request.form['registerUsername']
            print(registerUsername, file=sys.stderr)
            registerEmail = request.form['registerEmail']
            print(registerEmail, file=sys.stderr)
            registerPassword = request.form['registerPassword']
            print(registerPassword, file=sys.stderr)
            registerMember(registerUsername, registerPassword, registerName, registerEmail)

            result = 'success'

        return render_template('login.html', result = result)
    else:
        return render_template('login.html')


@app.route('/select', methods=['GET', 'POST'])
def select():
    # check first login
    firstLogin = False

    return render_template('select.html', first = firstLogin)


@app.route('/download/<filename>',methods=["GET","POST"])
def downloadFile(filename): #In your case fname is your filename
    try:
       path = f'./static/record/{filename}'
       return send_file(path, as_attachment=True)
    except Exception as e:
        print("error... during download", file=sys.stderr)
        return str(e)


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    return render_template('chatbot.html')


@app.route('/sendFeedback', methods=['POST'])
def sendFeedback():
    answer = '오류가 발생했습니다.'
    if request.method == 'POST':
        # json 형태로 풀기
        params = {}
        if request.is_json == True:
            params = request.get_json()
            ai_text = params['ai_text']
            user_text = params['user_text']
            charactor = params['charactor']
            feedback = params['feedback']
            
            with open(FEEDBACK_PATH, 'a') as f:
                f.write(user_text + '\t' + ai_text + '\t' + charactor + '\t' + feedback  + '\n')

            return jsonify({"result": 'success'})

    return jsonify({"result": 'fail'})

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
    model_path = './static/model/poly_16_pytorch_model.bin'
    infer_df_path = './static/model/inference_df.pickle'
    infer_emb_data_path = './static/model/inference_cand_embs.pickle'

    model_config = RobertaConfig.from_pretrained(base_model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name)
    base_model = RobertaModel.from_pretrained(base_model_name, config=model_config)

    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=question_length)

    chatbot=PolyEncoder(model_config, bert=base_model, poly_m=poly_m)
    chatbot.resize_token_embeddings(len(tokenizer))
    chatbot.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    chatbot.to(device)
    print('load chatbot', file=sys.stderr)

    with open(infer_df_path, 'rb') as fr:
        infer_df = CPU_Unpickler(fr).load()
        # infer_df=pickle.load(fr)
    with open(infer_emb_data_path, 'rb') as fr:
        cand_embs = CPU_Unpickler(fr).load()
        # cand_embs=pickle.load(fr)

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
    result = model.transcribe(TTS_PATH + "/DUMMY4/anichat_con_01_00007.wav", fp16=False, language='Korean')
    print(result["text"], file=sys.stderr)
    return result["text"]


@app.route('/vvv')
def vvv():
    print('Hello world!', file=sys.stderr)
    app.logger.error('str')
    return render_template('index.html')


@app.route('/vits')
def vits2():
    
    print('in vits2', file=sys.stderr)
    
    hps = utils.get_hparams_from_file("./static/model/conan_base.json")
    print('load config', file=sys.stderr)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    print('load net_g', file=sys.stderr)
    _ = net_g.eval()
    
    _ = utils.load_checkpoint("./static/model/G_410000.pth", net_g, None)
    print('load G_600000', file=sys.stderr)
   
    stn_tst = get_text("난 에도가와 코난 탐정이지", hps)
    print('get text', file=sys.stderr)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    print('no grad', file=sys.stderr)
    write('./static/model/conan.wav', hps.data.sampling_rate, audio)
    print('write wav', file=sys.stderr)
    
    return 'hello, its me.'



if __name__ == '__main__':
    print('run', file=sys.stderr)
    app.debug = True
    app.config["DEBUG"] = True
    app.run(host="0.0.0.0", port=20000, use_reloader=False)