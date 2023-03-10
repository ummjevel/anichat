import os
import argparse
import re
import json
from tqdm import tqdm
import glob

# SPEAKER_DICT = {"con": "0", "nam": "1"}
SPEAKER_DICT = {"con": "0", "you": "1" , "nam": "2"}

if __name__ == "__main__":
    # arg parse
    parser = argparse.ArgumentParser(description='Make Dataset For Project')
    # input folder path
    parser.add_argument('--input', '-i', type=str, help='dataset folder', default='/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4')
    # output wavfile path
    parser.add_argument('--woutput', '-wo', type=str, help='wavfile output folder', default='/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4')
    # output txtfile path
    parser.add_argument('--foutput', '-fo', type=str, help='txtfile output filepath', default='/home/ubuntu/alpaco/anichat/tts/vits/filelists/conan_audio_text_train_filelist.txt')
    # output wav khz
    parser.add_argument('--khz', '-khz', type=str, help='wavfile output khz', default='22050')
    # for multispeakers
    parser.add_argument('--multi', '-m', type=str, help='for multispeakers', default='False')
    
    
    args = parser.parse_args()


    # convert mp3 to wav file
    if args.khz == 44100:
        khz = ''
    else:
        khz = f'-acodec pcm_s16le -ar {args.khz}'
    
    # wav files output folder exists check
    if not os.path.exists(args.woutput):
        os.makedirs(args.woutput)

    # print("Converting mp3 to wav...")

    # # save wav files to wavfile output folder path
    # for mp3file in tqdm(glob.glob(args.input + '/*.mp3')):
    #     wavfile = args.woutput + '/' + mp3file.split('/')[-1].split('.')[0] + '.wav'
    #     command = f"ffmpeg -y -i {mp3file} {khz} {wavfile}"
    #     os.system(command)

    # print('Save txt file...')

    # write txt file
    with open(args.foutput, 'w') as txtfile:

        # load json files
        for jsonfile in tqdm(glob.glob(args.input + '/anichat_con_*.json')):
            wavfile_name = args.woutput.split('/')[-1] + '/' + jsonfile.split('/')[-1].split('.')[0] + '.wav'
            with open(jsonfile) as f:
                json_data = json.load(f)
            jsonfile_text = json_data['text']
            if args.multi == "True":
                multi_speaker = "|" + SPEAKER_DICT[jsonfile.split('_')[1]]
            else:
                multi_speaker = ""
            print(f'{wavfile_name}{multi_speaker}|{jsonfile_text}')
            txtfile.write(f'{wavfile_name}{multi_speaker}|{jsonfile_text}\n')

    print('DONE!')

