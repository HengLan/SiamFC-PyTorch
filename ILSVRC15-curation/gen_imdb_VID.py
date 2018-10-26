'''
Writted by Heng Fan
10/01/2018
'''
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import json


validation_ratio = 0.1


def generate_image_imdb(vid_root_path, vid_curated_path):
    '''
    # save image crops to the vid_curated_path
    '''
    anno_str = "Annotations/VID/train/"
    data_str = "Data/VID/train/"
    vid_anno_path = os.path.join(vid_root_path, anno_str)
    vid_data_path = os.path.join(vid_root_path, data_str)

    num_videos = 0

    # dirs of level1: e.g., a/, b/, ...
    all_dirs_level1 = os.listdir(vid_anno_path)

    for i in range(len(all_dirs_level1)):
        all_dirs_level2 = os.listdir(os.path.join(vid_anno_path, all_dirs_level1[i]))
        num_videos = num_videos + len(all_dirs_level2)

    train_video_num = round(num_videos * (1-validation_ratio))
    val_video_num = num_videos - train_video_num

    imdb_video_train = dict()
    imdb_video_train['num_videos'] = train_video_num
    imdb_video_train['data_str'] = data_str

    imdb_video_val = dict()
    imdb_video_val['num_videos'] = val_video_num
    imdb_video_val['data_str'] = data_str

    videos_train = dict()
    videos_val = dict()

    vid_idx = 0

    for i in range(len(all_dirs_level1)):
        all_dirs_level2 = os.listdir(os.path.join(vid_anno_path, all_dirs_level1[i]))

        # dirs of level2: e.g., a/ILSVRC2015_train_00000000/, a/ILSVRC2015_train_00001000/, ...
        for j in range(len(all_dirs_level2)):

            if vid_idx < train_video_num:
                if not videos_train.has_key(all_dirs_level2[j]):
                    videos_train[all_dirs_level2[j]] = []
            else:
                if not videos_val.has_key(all_dirs_level2[j]):
                    videos_val[all_dirs_level2[j]] = []



            frame_list = glob.glob(os.path.join(vid_anno_path, all_dirs_level1[i], all_dirs_level2[j], "*.xml"))
            frame_list.sort()

            video_ids = dict()     # store frame level information

            # level3: frame level
            for k in range(len(frame_list)):
                # read xml file
                frame_id = k
                frame_xml_name = os.path.join(vid_anno_path, all_dirs_level1[i], all_dirs_level2[j], frame_list[k])
                frame_xml_tree = ET.parse(frame_xml_name)
                frame_xml_root = frame_xml_tree.getroot()

                # crop_path = os.path.join(vid_curated_path, data_str, all_dirs_level1[i], all_dirs_level2[j])
                crop_path = os.path.join(all_dirs_level1[i], all_dirs_level2[j])
                frame_filename = frame_xml_root.find('filename').text

                print ("processing: %s, %s, %s ..." % (all_dirs_level1[i], all_dirs_level2[j], frame_filename))

                for object in frame_xml_root.iter("object"):
                    # get trackid
                    id = object.find("trackid").text
                    if not video_ids.has_key(id):
                        video_ids[id] = []
                    # get bounding box
                    bbox_node = object.find("bndbox")
                    xmax = float(bbox_node.find('xmax').text)
                    xmin = float(bbox_node.find('xmin').text)
                    ymax = float(bbox_node.find('ymax').text)
                    ymin = float(bbox_node.find('ymin').text)
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1
                    bbox = np.array([xmin, ymin, width, height])

                    tmp_instance = dict()
                    tmp_instance['instance_path'] = os.path.join(all_dirs_level1[i], all_dirs_level2[j], '{}.{:02d}.crop.x.jpg'.format(frame_filename, int(id)))
                    tmp_instance['bbox'] =bbox.tolist()

                    video_ids[id].append(tmp_instance)

            # delete the object_id with less than 1 frame
            tmp_keys = video_ids.keys()
            for ki in range(len(tmp_keys)):
                if len(video_ids[tmp_keys[ki]]) < 2:
                    del video_ids[tmp_keys[ki]]

            tmp_keys = video_ids.keys()

            if len(tmp_keys) > 0:

                if vid_idx < train_video_num:
                    videos_train[all_dirs_level2[j]].append(video_ids)
                else:
                    videos_val[all_dirs_level2[j]].append(video_ids)

                vid_idx = vid_idx + 1

    imdb_video_train['videos'] = videos_train
    imdb_video_val['videos'] = videos_val

    # save imdb information
    json.dump(imdb_video_train, open('imdb_video_train.json', 'w'), indent=2)
    json.dump(imdb_video_val, open('imdb_video_val.json', 'w'), indent=2)


if __name__ == "__main__":
    vid_root_path = "/home/hfan/Dataset/ILSVRC2015"
    vid_curated_path = "/home/hfan/Dataset/ILSVRC2015_crops"
    generate_image_imdb(vid_root_path, vid_curated_path)