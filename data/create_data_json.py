import json
import os

import pandas as pd


def main():
    data_dir = '/datasets01/webvid/videos_train'

    df = pd.read_csv('results_2M_train.csv')
    videoid = df['videoid'].tolist()
    caption = df['name'].tolist()
    print(len(videoid))
    print(len(caption))

    anno_dict = {}
    success = 0
    for vid, cap in zip(videoid, caption):
        vid = str(vid)
        video_file_path = os.path.join(data_dir, vid[:-4], f'{vid}.mp4')
        if os.path.isfile(video_file_path):
            success += 1
            anno_dict[video_file_path] = cap
            print(success, video_file_path)
    
    with open(f'webvid_train.json', 'w') as fp:
        json.dump(anno_dict, fp, indent=4)
    
    print(f'Found {success}/{len(videoid)} ({success/len(videoid)*100.0:.2f}%) videos')


if __name__ == '__main__':
    main()
