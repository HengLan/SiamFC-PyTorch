"""
Dataset for VID
Written by Heng Fan
"""

from torch.utils.data.dataset import Dataset
import json
from Utils import *
import os


class VIDDataset(Dataset):

    def __init__(self, imdb, data_dir, config, z_transforms, x_transforms, mode = "Train"):
        imdb_video      = json.load(open(imdb, 'r'))
        self.videos     = imdb_video['videos']
        self.data_dir   = data_dir
        self.config     = config
        self.num_videos = int(imdb_video['num_videos'])

        self.z_transforms = z_transforms
        self.x_transforms = x_transforms

        if mode == "Train":
            self.num = self.config.num_pairs
        else:
            self.num = self.num_videos

    def __getitem__(self, rand_vid):
        '''
        read a pair of images z and x
        '''
        # randomly decide the id of video to get z and x
        rand_vid = rand_vid % self.num_videos

        video_keys = self.videos.keys()
        video = self.videos[video_keys[rand_vid]]

        # get ids of this video
        video_ids = video[0]
        # how many ids in this video
        video_id_keys = video_ids.keys()

        # randomly pick an id for z
        rand_trackid_z = np.random.choice(list(range(len(video_id_keys))))
        # get the video for this id
        video_id_z = video_ids[video_id_keys[rand_trackid_z]]

        # pick a valid examplar z in the video
        rand_z = np.random.choice(range(len(video_id_z)))

        # pick a valid instance within frame_range frames from the examplar, excluding the examplar itself
        possible_x_pos = range(len(video_id_z))
        rand_x = np.random.choice(possible_x_pos[max(rand_z - self.config.pos_pair_range, 0):rand_z] + possible_x_pos[(rand_z + 1):min(rand_z + self.config.pos_pair_range, len(video_id_z))])

        z = video_id_z[rand_z].copy()    # use copy() here to avoid changing dictionary
        x = video_id_z[rand_x].copy()

        # read z and x
        img_z = cv2.imread(os.path.join(self.data_dir, z['instance_path']))
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)

        img_x = cv2.imread(os.path.join(self.data_dir, x['instance_path']))
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        # do data augmentation;
        # note that we have done center crop for z in the data augmentation
        img_z = self.z_transforms(img_z)
        img_x = self.x_transforms(img_x)

        return img_z, img_x

    def __len__(self):
        return self.num
