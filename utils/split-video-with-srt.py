import os
import argparse
import pysrt
import re
import json
from tqdm import tqdm


if __name__ == "__main__":
    # arg parse
    parser = argparse.ArgumentParser(description='Split Video With SRT File')
    # mp4 file
    parser.add_argument('--mp4', '-i4', type=str, help='original mp4 file', default='DC_3_wizard.mp4')
    # mp4 file
    parser.add_argument('--mp3', '-i3', type=str, help='original mp3 file', default='DC_3_wizard.mp3')
    # srt file
    parser.add_argument('--srt', '-if', type=str, help='srt file for timestamps', default='DC_3_wizard.srt')
    # output folder
    parser.add_argument('--output', '-o', type=str, help='output folder', default='DC_3_wizard')
    # specific unicode 
    parser.add_argument('--unicode', '-u', type=str, help='specific unicode encoding', default='utf-8')
    # save only mp3 
    parser.add_argument('--omp3', '-o3', type=str, help='is output save mp3', default='True')
    args = parser.parse_args()

    # execute command

    # extract start and end time from srt file
    # open srt file
    subs = pysrt.open(args.srt, encoding=args.unicode)

    # make folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for idx, sub in enumerate(tqdm(subs)): 

        output_file_name = f'{args.output}_{idx:05d}'
        # make command
        start_time = f'{sub.start.hours:02d}:{sub.start.minutes:02d}:{sub.start.seconds:02d}.{sub.start.milliseconds:03d}'
        end_time = f'{sub.end.hours:02d}:{sub.end.minutes:02d}:{sub.end.seconds:02d}.{sub.end.milliseconds:03d}'
        duration = str(sub.duration).replace(',', '.')
        if args.omp3 == False:
            command = f'ffmpeg -ss {start_time} -i {args.mp4} -strict -2 -t {duration} -c:v copy {args.output}/{output_file_name}.mp4'
        else:
            command = f'ffmpeg -ss {start_time} -i {args.mp4} -strict -2 -t {duration} -c:v copy {args.output}/{output_file_name}.mp3'

        # make json
        # file name, start timestamp, end timestamp, duration, text
        json_data = {
            "file_name": args.mp4
            , "start" : start_time
            , "end" : end_time
            , "duration" : duration
            , "text" : f'{sub.text_without_tags.replace("코난: ", "")}'
        }

        if sub.text_without_tags[:3] == '남도일':
            continue

        # save json
        with open(f"{args.output}/{output_file_name}.json", "w", encoding='utf-8') as json_file:
           json.dump(json_data, json_file, indent="\t", ensure_ascii=False)

        # execute command
        os.system(command)
        
    print('end!')

