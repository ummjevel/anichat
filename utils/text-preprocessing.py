import re

#for filename in glob.glob(path)[:]:

linelist = open('/home/ubuntu/anaconda3/envs/alpaco/anichat/tts/vits/filelists/conan_audio_text_train_filelist.txt').readlines()

#if not bool(linelist):   # 리스트가 비었을때
#    print(filename)

for i in linelist:   # 첫글자가 숫자가 아닐때
    #if int(i[0]) is ValueError:
    #    print(filename)
    print(re.findall('|([0-9]+)?', i))
    #print(i)
    #break