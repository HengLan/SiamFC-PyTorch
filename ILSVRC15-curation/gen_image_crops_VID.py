'''
Written by Heng Fan
'''
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import cv2
import datetime


'''
# default setting for cropping
'''
examplar_size = 127.0
instance_size = 255.0
context_amount = 0.5


def get_subwindow_avg(im, pos, model_sz, original_sz):
    '''
    # obtain image patch, padding with avg channel if area goes outside of border
    '''
    avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]

    if original_sz is None:
        original_sz = model_sz

    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert (im_sz[0] > 2) & (im_sz[1] > 2), "The size of image is too small!"
    c = (sz + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[0] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = max(0, 1 - context_xmin)  # in python, index starts from 0
    top_pad = max(0, 1 - context_ymin)
    right_pad = max(0, context_xmax - im_sz[1])
    bottom_pad = max(0, context_ymax - im_sz[0])

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    im_R = im[:, :, 0]
    im_G = im[:, :, 1]
    im_B = im[:, :, 2]

    # padding
    if (top_pad != 0) | (bottom_pad != 0) | (left_pad != 0) | (right_pad != 0):
        im_R = np.pad(im_R, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant',
                      constant_values=avg_chans[0])
        im_G = np.pad(im_G, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant',
                      constant_values=avg_chans[1])
        im_B = np.pad(im_B, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant',
                      constant_values=avg_chans[2])

        im = np.stack((im_R, im_G, im_B), axis=2)

    im_patch_original = im[int(context_ymin) - 1:int(context_ymax), int(context_xmin) - 1:int(context_xmax), :]

    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch_original, (int(model_sz), int(model_sz)), interpolation=cv2.INTER_CUBIC)
    else:
        im_patch = im_patch_original

    return im_patch


def get_crops(img, bbox, size_z, size_x, context_amount):
    '''
    # get examplar and search region crops
    '''
    cx = bbox[0] + bbox[2]/2
    cy = bbox[1] + bbox[3]/2
    w = bbox[2]
    h = bbox[3]

    # for examplar
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    im_crop_z = get_subwindow_avg(img, np.array([cy, cx]), size_z, round(s_z))

    # for search region
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x
    im_crop_x = get_subwindow_avg(img, np.array([cy, cx]), size_x, round(s_x))

    return im_crop_z, im_crop_x


def generate_image_crops(vid_root_path, vid_curated_path):
    '''
    # save image crops to the vid_curated_path
    '''
    anno_str = "Annotations/VID/train/"
    data_str = "Data/VID/train/"
    vid_anno_path = os.path.join(vid_root_path, anno_str)
    vid_data_path = os.path.join(vid_root_path, data_str)

    cur_procesed_fraem = 0
    start_time = datetime.datetime.now()
    total_time = 0

    # dirs of level1: e.g., a/, b/, ...
    all_dirs_level1 = os.listdir(vid_anno_path)
    for i in range(len(all_dirs_level1)):
        all_dirs_level2 = os.listdir(os.path.join(vid_anno_path, all_dirs_level1[i]))

        # dirs of level2: e.g., a/ILSVRC2015_train_00000000/, a/ILSVRC2015_train_00001000/, ...
        for j in range(len(all_dirs_level2)):
            frame_list = glob.glob(os.path.join(vid_anno_path, all_dirs_level1[i], all_dirs_level2[j], "*.xml"))
            frame_list.sort()

            # level3: frame level
            for k in range(len(frame_list)):
                frame_xml_name = os.path.join(vid_anno_path, all_dirs_level1[i], all_dirs_level2[j], frame_list[k])
                frame_xml_tree = ET.parse(frame_xml_name)
                frame_xml_root = frame_xml_tree.getroot()

                # image file path
                frame_img_name = (frame_list[k].replace(".xml", ".JPEG")).replace(vid_anno_path, vid_data_path)
                img = cv2.imread(frame_img_name)
                if img is None:
                    print("Cannot find %s!"%frame_img_name)
                    exit(0)

                # image file name
                frame_filename = frame_xml_root.find('filename').text

                # process (all objects in) each frame
                for object in frame_xml_root.iter("object"):
                    # get trackid
                    id = object.find("trackid").text

                    # get bounding box
                    bbox_node = object.find("bndbox")
                    xmax = float(bbox_node.find('xmax').text)
                    xmin = float(bbox_node.find('xmin').text)
                    ymax = float(bbox_node.find('ymax').text)
                    ymin = float(bbox_node.find('ymin').text)
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1
                    bbox = np.array([xmin, ymin, width, height])

                    # print("processing %s, %s, %s, %s ..." % (all_dirs_level1[i], all_dirs_level2[j], frame_filename+".JPEG", id))

                    # get crops
                    im_crop_z, im_crop_x = get_crops(img, bbox, examplar_size, instance_size, context_amount)

                    # save crops
                    save_path = os.path.join(vid_curated_path, data_str, all_dirs_level1[i], all_dirs_level2[j])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    savename_crop_z = os.path.join(save_path, '{}.{:02d}.crop.z.jpg'.format(frame_filename, int(id)))
                    savename_crop_x = os.path.join(save_path, '{}.{:02d}.crop.x.jpg'.format(frame_filename, int(id)))

                    cv2.imwrite(savename_crop_z, im_crop_z, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    cv2.imwrite(savename_crop_x, im_crop_x, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                    cur_procesed_fraem = cur_procesed_fraem + 1

                    if cur_procesed_fraem % 1000 == 0:
                        end_time = datetime.datetime.now()
                        total_time = total_time + int((end_time-start_time).seconds)
                        print("finished processing %d frames in %d seconds (FPS: %d ) ..." % (cur_procesed_fraem, total_time, int(1000/(end_time-start_time).seconds)))
                        start_time = datetime.datetime.now()


if __name__ == "__main__":
    # path to your VID dataset
    vid_root_path = "/home/hfan/Dataset/ILSVRC2015"
    vid_curated_path = "/home/hfan/Dataset/ILSVRC2015_crops"
    if not os.path.exists(vid_curated_path):
        os.mkdir(vid_curated_path)
    generate_image_crops(vid_root_path, vid_curated_path)
