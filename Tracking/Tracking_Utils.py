"""
Tool functiond for tracking evaluation
Written by Heng Fan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import glob


def cat_img(image_cat1, image_cat2, image_cat3):
    """
    concatenate three 1-channel images to one 3-channel image
    """
    image = np.zeros(shape = (image_cat1.shape[0], image_cat1.shape[1], 3), dtype=np.double)
    image[:, :, 0] = image_cat1
    image[:, :, 1] = image_cat2
    image[:, :, 2] = image_cat3

    return image


def load_sequence(seq_root_path, seq_name):
    """
    load sequences;
    sequences should be in OTB format, or you can custom this function by yourself
    """
    img_dir = os.path.join(seq_root_path, seq_name, 'img/')
    gt_path = os.path.join(seq_root_path, seq_name, 'groundtruth_rect.txt')

    img_list = glob.glob(img_dir + "*.jpg")
    img_list.sort()
    img_list = [os.path.join(img_dir, x) for x in img_list]

    gt = np.loadtxt(gt_path, delimiter=',')

    init_bbox = gt[0]
    if seq_name == "Tiger1":
        init_bbox = gt[5]

    init_x = init_bbox[0]
    init_y = init_bbox[1]
    init_w = init_bbox[2]
    init_h = init_bbox[3]

    target_position = np.array([init_y + init_h/2, init_x + init_w/2], dtype = np.double)
    target_sz = np.array([init_h, init_w], dtype = np.double)

    if seq_name == "David":
        img_list = img_list[299:]
    if seq_name == "Tiger1":
        img_list = img_list[5:]
    if seq_name == "Football1":
        img_list = img_list[0:74]

    return img_list, target_position, target_sz


def visualize_tracking_result(img, bbox, fig_n):
    """
    visualize tracking result
    """
    fig = plt.figure(fig_n)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 3, edgecolor = "#00ff00", zorder = 1, fill = False)
    ax.imshow(img)
    ax.add_patch(r)
    plt.ion()
    plt.show()
    plt.pause(0.00001)
    plt.clf()


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    """
    extract image crop
    """
    if original_sz is None:
        original_sz = model_sz

    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert (im_sz[0] > 2) & (im_sz[1] > 2), "The size of image is too small!"
    c = (sz+1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c)       # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[0] - c)       # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = max(0, 1-context_xmin)       # in python, index starts from 0
    top_pad = max(0, 1-context_ymin)
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
    if (top_pad !=0) | (bottom_pad !=0) | (left_pad !=0) | (right_pad !=0):
        im_R = np.pad(im_R, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant', constant_values = avg_chans[0])
        im_G = np.pad(im_G, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant', constant_values = avg_chans[1])
        im_B = np.pad(im_B, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant', constant_values = avg_chans[2])

        im = cat_img(im_R, im_G, im_B)

    im_patch_original = im[int(context_ymin)-1:int(context_ymax), int(context_xmin)-1:int(context_xmax), :]

    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch_original, (int(model_sz), int(model_sz)), interpolation = cv2.INTER_CUBIC)
    else:
        im_patch = im_patch_original

    return im_patch


def make_scale_pyramid(im, target_position, in_side_scaled, out_side, avg_chans, p):
    """
    extract multi-scale image crops
    """
    in_side_scaled = np.round(in_side_scaled)
    pyramid = np.zeros((out_side, out_side, 3, p.num_scale), dtype = np.double)
    max_target_side = in_side_scaled[in_side_scaled.size-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side
    # size_in_search_area = beta * size_in_image
    # e.g. out_side = beta * min_target_side
    search_side = round(beta * max_target_side)

    search_region = get_subwindow_tracking(im, target_position, search_side, max_target_side, avg_chans)

    assert (round(beta * min_target_side) == out_side), "Error!"

    for s in range(p.num_scale):
        target_side = round(beta * in_side_scaled[s])
        search_target_position = np.array([1 + search_side/2, 1 + search_side/2], dtype = np.double)
        pyramid[:, :, :, s] = get_subwindow_tracking(search_region, search_target_position, out_side,
                                                   target_side, avg_chans)

    return pyramid


def tracker_eval(net, s_x, z_features, x_features, target_position, window, p):
    """
    do evaluation (i.e., a forward pass for search region)
    (This part is implemented as in the original Matlab version)
    """
    # compute scores search regions of different scales
    scores = net.xcorr(z_features, x_features)
    scores = scores.to("cpu")

    response_maps = scores.squeeze().permute(1, 2, 0).data.numpy()
    # for this one, the opencv resize function works fine
    response_maps_up = cv2.resize(response_maps, (response_maps.shape[0]*p.response_UP, response_maps.shape[0]*p.response_UP), interpolation=cv2.INTER_CUBIC)

    # choose the scale whose response map has the highest peak
    if p.num_scale > 1:
        current_scale_id =np.ceil(p.num_scale/2)
        best_scale = current_scale_id
        best_peak = float("-inf")
        for s in range(p.num_scale):
            this_response = response_maps_up[:, :, s]
            # penalize change of scale
            if s != current_scale_id:
                this_response = this_response * p.scale_penalty
            this_peak = np.max(this_response)
            if this_peak > best_peak:
                best_peak = this_peak
                best_scale = s
        response_map = response_maps_up[:, :, int(best_scale)]
    else:
        response_map = response_maps_up
        best_scale = 1
    # make the response map sum to 1
    response_map = response_map - np.min(response_map)
    response_map = response_map / sum(sum(response_map))

    # apply windowing
    response_map = (1 - p.w_influence) * response_map + p.w_influence * window
    p_corr = np.asarray(np.unravel_index(np.argmax(response_map), np.shape(response_map)))

    # avoid empty
    if p_corr[0] is None:
        p_corr[0] = np.ceil(p.score_size/2)
    if p_corr[1] is None:
        p_corr[1] = np.ceil(p.score_size/2)

    # Convert to crop-relative coordinates to frame coordinates
    # displacement from the center in instance final representation ...
    disp_instance_final = p_corr - np.ceil(p.score_size * p.response_UP / 2)
    # ... in instance input ...
    disp_instance_input = disp_instance_final * p.stride / p.response_UP
    # ... in instance original crop (in frame coordinates)
    disp_instance_frame = disp_instance_input * s_x / p.instance_size
    # position within frame in frame coordinates
    new_target_position = target_position + disp_instance_frame

    return new_target_position, best_scale