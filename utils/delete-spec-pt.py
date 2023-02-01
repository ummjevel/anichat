from tqdm import tqdm
import shutil
import glob
import os

if __name__ == "__main__":

    # jsonfile = []
    # wavfile = []
    # elsefile = []

    print(len(glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*')))
    print(len(glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*.json')))
    print(len(glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*.wav')))
    print(len(glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*.mp3')))
    print(len(glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*.spec.pt')))

    for i in glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*.wav'):
        if 'con_11_00277' in i:
            print(i)
            break

    # for file in tqdm(glob.glob('/home/ubuntu/alpaco/anichat/tts/vits/DUMMY4/*')):
    #     if 'json' in file:
    #         jsonfile.append(file)
    #     elif 'wav' in file:
    #         wavfile.append(file)
    #     else:
    #         elsefile.append(file)

    #     # if 'spec.pt' not in file:
    #     #     last.append(file)

    # print(elsefile)


        
        # if 'spec' in file:
        #     speclist.append(file)
        # if 'anichat' in file:
        #     anichat.append(file)
        # if 'json' in file:
        #     an

    # print(len(speclist))
    # print(len(anichat))
    # print(10242 - 1241)
    # print(9001 - 7706)


