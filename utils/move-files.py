from tqdm import tqdm
import shutil
import glob

if __name__ == "__main__":

    for folders in tqdm(glob.glob('/home/ubuntu/anaconda3/envs/alpaco/anichat/tts/vits/DUMMY4/*')):
        if 'anichat' in folders:
            for files in tqdm(glob.glob(folders + '/*')):
                #print(files)
                shutil.move(files, '/home/ubuntu/anaconda3/envs/alpaco/anichat/tts/vits/DUMMY4')
