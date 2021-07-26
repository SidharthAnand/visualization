# math
import numpy as np
from scipy import stats
import random, math
from scipy.spatial import ConvexHull
import shapely
from shapely.geometry import LineString, point
from shapely.geometry.polygon import Polygon
# viz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pygame
import pygame.freetype
import pygame_gui
import shutil
# os
import time
from time import mktime, strptime, gmtime, strftime, localtime
from calendar import timegm
import json
import pickle
from os import path, listdir
import imageio
import _thread
import sys
from collections import deque
import argparse
import copy
# serial
import serial
import select
import ast
# tracking
from utilities import *
# mqtt
import paho.mqtt.client as paho
import sys
from sys import getsizeof
from bresenham import bresenham
import cv2

import config_butlr_team, rules

global printing_run_time, debug_run_time
printing_run_time = False
from json import JSONEncoder

def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0
    r = getsizeof(o)
    ids.add(id(o))
    if isinstance(o, str):
        return r
    if isinstance(o, dict):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
    if isinstance(o, list) or isinstance(o, tuple) or isinstance(o, set):
        return r + sum(d(x, ids) for x in o)
    return r

def rotate_point(point, origin=(0.0, 0.0), angle_deg=0.0):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in degree.
    """
    px, py = point
    ox, oy = origin
    angle_rad = math.radians(angle_deg)
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    point_rotated = (qx, qy)
    return point_rotated

def run_time_main(func):
    global printing_run_time, debug_run_time

    def wrapper(*args, **kw):
        if printing_run_time and debug_run_time: t1 = time.time()
        res = func(*args, **kw)
        if printing_run_time and debug_run_time: t2 = time.time()
        if printing_run_time and debug_run_time: print(
            'DEBUG: {:4.4f} seconds elapsed for {}'.format(t2 - t1, func.__name__))
        return res

    return wrapper

def run_time_special(func):
    def wrapper(*args, **kw):
        t1 = time.time()
        res = func(*args, **kw)
        t2 = time.time()
        print('DEBUG: {:4.4f} seconds elapsed for {}'.format(t2 - t1, func.__name__))
        return res

    return wrapper

def censoring(mat, left_percentile=None, right_percentile=None):
    if left_percentile:
        left_value = np.percentile(mat, left_percentile)
    else:
        left_value = mat.min()
    if right_percentile:
        right_value = np.percentile(mat, right_percentile)
    else:
        right_value = mat.max()
    return [x for x in mat.flatten() if x>=left_value and x<=right_value]

def date_time_str_to_epoch_sec_1(date_time_str):
    utc_time = time.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    epoch_time = timegm(utc_time)
    return epoch_time

def date_time_str_to_epoch_sec_2(date_time_str):
    utc_time = time.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ')
    epoch_time = timegm(utc_time)
    return epoch_time

def compress_long_traces(trackingDict, lastN=3, beforeN=2, len_th=20):
    # modify in place
    # lastN: keep last N points after compression; beforeN: uniformly sample N ponits before lastN
    for name, trace in trackingDict.items():
        full_trace = trace['trace']
        if len(full_trace) > len_th:
            every_n = int(np.ceil((len(full_trace) - lastN) / beforeN))
            tmp_a, tmp_b = full_trace[0::every_n], full_trace[-lastN:]
            if tmp_a[-1] not in tmp_b:
                compressed_trace = tmp_a + tmp_b
            else:
                compressed_trace = tmp_a[:-1] + tmp_b
            trace['trace'] = compressed_trace
    return trackingDict

@run_time_main
def generate_positions(x_start, y_start, cell_size, boxList, max_split_time=np.inf):
    """
    original plt_heatmap_pygame

    Arguments:
    ----------------------------------------------
    boxList: a list of peak cells, returned by "get_peak_by_adaptive_local_regress" or "get_peak_by_threshold"
    ax: the axis for this heatmap
    vmin/vmax: min and max temperature values, used to scale colors
    border: if true, ROIs would be visualized with black border
    title: title of this plot
    """
    ROIs = get_rois(boxList, num_neighbor=8)
    meanPositions = []
    rp1 = lambda x: ((x[1] + 1.0) * cell_size + x_start, (x[0] + 1.0) * cell_size + y_start)
    rp2 = lambda x: ((x[1] - 0.0) * cell_size + x_start, (x[0] - 0.0) * cell_size + y_start)
    rp3 = lambda x: ((x[1] - 0.0) * cell_size + x_start, (x[0] + 1.0) * cell_size + y_start)
    rp4 = lambda x: ((x[1] + 1.0) * cell_size + x_start, (x[0] - 0.0) * cell_size + y_start)
    polygons = []
    for roi_idx, roi in enumerate(ROIs):
        rectPoints = list(set([f(x) for x in roi for f in (rp1, rp2, rp3, rp4)]))
        rectPoints = np.array(rectPoints)
        hull = ConvexHull(rectPoints)
        vertices = np.asarray([rectPoints[idx] for idx in hull.vertices])
        polygon = Polygon(vertices)
        splitPolygons, diameters, majorAxisPoitsList, centroids = split_polygon_recurrent(
            polygon, scale=1 / cell_size, diameter_th=diameter_th, max_split_time=max_split_time)
        polygons += splitPolygons
        meanPositions += centroids
    return ROIs, meanPositions, polygons

@run_time_main
def generate_counts(track_info, frame_num, door_line_start, door_line_end, x_start, y_start, cell_size,
                    first_only=False):
    """
    """
    global ppl_count_global, ppl_count_in_global, ppl_count_out_global
    global count_predict_tmp

    # rename
    trks = track_info
    i = frame_num + 1
    ds = door_line_start
    de = door_line_end

    # door line horizontal or vertical
    if ds[1] == de[1]:
        door_line_dir = 'h'
    elif ds[0] == de[0]:
        door_line_dir = 'v'
    else:
        print('ERROR: door line is neither horizontal or vertical')

    line_door = LineString([ds, de])

    # loop tracks
    ppl_count = 0
    ppl_count_in = 0
    ppl_count_out = 0
    count_predict_tmp = []
    for trk in trks:
        intersections = []
        # if i in this track duration + X frames, display and count, count circles become gray afterwords
        if i >= trk['t']:
            coods = trk['cood'][:(i - trk['t'])]  # this is actually the full track if run in realtime
            for j in range(len(coods) - 1):
                pt1 = (coods[j][0], coods[j][1])
                pt2 = (coods[j + 1][0], coods[j + 1][1])
                line_test = LineString([pt1, pt2])
                int_pt_shapely = line_door.intersection(line_test)
                if not int_pt_shapely.is_empty:
                    int_pt = (int_pt_shapely.x, int_pt_shapely.y)
                    # viz counting intersection pt
                    int_pt_screen = (
                    round(x_start + int_pt[0] * cell_size * 8), round(y_start + int_pt[1] * cell_size * 8))
                    # calculate count direction: door line horizontal: up=in, down=out; vertical: left=in, right=out.
                    if door_line_dir == 'h':
                        if pt1[1] <= pt2[1]:
                            count_dir = 'out'
                        else:
                            count_dir = 'in'
                    elif door_line_dir == 'v':
                        if pt1[0] <= pt2[0]:
                            count_dir = 'out'
                        else:
                            count_dir = 'in'
                    intersections.append({'t': trk['t'] + j, 'int_pt_screen': int_pt_screen, 'dir': count_dir})
            if first_only and len(intersections) > 0:
                intersections = [intersections[0]]
            else:
                intersections = counting_remove(intersections, gap_dict=gap_dict_counting)
            ppl_count += len(intersections)
            ppl_count_in += len([x for x in intersections if x['dir'] == 'in'])
            ppl_count_out += len([x for x in intersections if x['dir'] == 'out'])
    return ppl_count, ppl_count_in, ppl_count_out

def num_of_split_polygons(diameter, diameter_th={2: 4, 3: 8}):
    if diameter >= diameter_th[2] and diameter < diameter_th[3]:
        return 2
    if diameter >= diameter_th[3]:
        return 3
    return 1

def distance_value_regress(mat, row_id, col_id, drange=1, include_self=True):
    """
    Run a simple regression for a given position, with y=temperature and x=distance to that position

    Arguments:
    -----------------------------
    mat: temperature matrix
    row_id / col_id: row and column indices of the position
    drange: searching range from the position
    include_self: if true, the temperature of the position itself would be included with the distance=0

    Returns:
    ---------------------------------
    slope, r2, p_value: regression results
    """
    dist, value = [], []
    nrow, ncol = mat.shape
    for i in range(nrow):
        for j in range(ncol):
            if i == row_id and j == col_id and include_self is False:
                continue
            if np.abs(i - row_id) > drange or np.abs(j - col_id) > drange:
                continue
            # cal distance
            d = np.sqrt((i - row_id) ** 2 + (j - col_id) ** 2)
            if d <= drange:
                dist.append(d)
                value.append(mat[i, j])
    dist, value = np.asarray(dist), np.asarray(value)
    slope, intercept, r_value, p_value, std_err = stats.linregress(dist, value)
    r2 = r_value ** 2
    return slope, r2, p_value

# @run_time_main
def get_background_slope_r2_mats(mats, nrow, ncol, drange=1):
    """
    Get the regression results (slope and r2) for background temperature,
    which are used for adaptive thresholds.

    Arguments:
    ------------------------------------------
    mats: a list of background temperature matrices.
    nrow/ncol: shape of each matrix
    drange: searching range when runing regression.

    Returns:
    --------------------------------------------
    bg_slope_mat: slope matrix of the background
    bg_r2_mat: r2 matrix of the background
    """
    n = len(mats)
    if isinstance(drange, dict):
        drange_center, drange_edge = drange['center'], drange['edge']
    else:
        drange_center, drange_edge = drange, drange
    bg_slope_mat, bg_r2_mat = np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            for mat in mats:
                if min(i, 7 - i) < 2 or min(j, 7 - j) < 2:
                    slope, r2, p_value = distance_value_regress(mat, i, j, drange=drange_edge)
                else:
                    slope, r2, p_value = distance_value_regress(mat, i, j, drange=drange_center)
                bg_slope_mat[i, j] += slope
                bg_r2_mat[i, j] += r2
    if n > 0:
        bg_slope_mat /= n
        bg_r2_mat /= n
    return bg_slope_mat, bg_r2_mat

# @run_time_special
def get_peak_by_adaptive_local_regress_before(delta_mat, slope_th=-0.1, detect_type='hot', r2_th=0.1, drange=1):
    """
    this is the old version by scipy and iteration
    """
    if not isinstance(drange, dict):
        drange_center, drange_edge = drange, drange
    else:
        drange_center, drange_edge = drange['center'], drange['edge']
    nrow, ncol = delta_mat.shape
    if isinstance(slope_th, float) or isinstance(slope_th, int):
        slope_th = np.ones((nrow, ncol)) * slope_th

    rst = []
    slope_mat, r2_mat = -100.0 * np.ones_like(delta_mat), -100.0 * np.ones_like(delta_mat)

    for i in range(nrow):
        for j in range(ncol):
            if min(i, 7 - i) < 2 or min(j, 7 - j) < 2:
                slope, r2, p_value = distance_value_regress(delta_mat, i, j, drange=drange_edge)
            else:
                slope, r2, p_value = distance_value_regress(delta_mat, i, j, drange=drange_center)
            slope_mat[i, j] = slope
            r2_mat[i, j] = r2
            if detect_type == 'hot':
                if slope <= slope_th[i, j] and r2 >= r2_th:
                    rst.append((i, j))
            elif detect_type == 'cold':
                if slope >= slope_th[i, j] and r2 >= r2_th:
                    rst.append((i, j))
    return rst, slope_mat, r2_mat

# @run_time_special
def get_drange_mask(drange=3, mat_shape=(8, 8), include_self=True, edge_resolution=(2, 2)):
    nrow, ncol = mat_shape[0], mat_shape[1]
    if not isinstance(drange, dict):
        drange_center, drange_edge = drange, drange
    else:
        drange_center, drange_edge = drange['center'], drange['edge']
    padding_width = int(np.ceil(max([drange_center, drange_edge])))
    virtual_mat = np.pad(np.ones(mat_shape), ((padding_width, padding_width), (padding_width, padding_width)),
                         'constant', constant_values=np.nan)
    virtual_nan_idx = [(m, n) for m in range(nrow + 2 * padding_width) for n in range(ncol + 2 * padding_width) if
                       np.isnan(virtual_mat[m, n])]
    mask_idx_dict, mask_dist_dict = {}, {}
    for row in range(nrow):
        for col in range(ncol):
            if min(row, mat_shape[0]-1 - row) < edge_resolution[0] or min(col, mat_shape[1]-1 - col) < edge_resolution[1]:
                use_drange = drange_edge
            else:
                use_drange = drange_center
            trans_row, trans_col = row + padding_width, col + padding_width
            crt_key = (row, col)
            crt_idx, crt_dist = [], []
            for i in range(nrow + 2 * padding_width):
                for j in range(ncol + 2 * padding_width):
                    if i == trans_row and j == trans_col and include_self is False:
                        continue
                    if np.abs(i - trans_row) > use_drange or np.abs(j - trans_col) > use_drange:
                        continue
                    d = np.sqrt((i - trans_row) ** 2 + (j - trans_col) ** 2)
                    if d <= use_drange:
                        crt_idx.append((i, j))
                        if not np.isnan(virtual_mat[i, j]):
                            crt_dist.append(d)
                        else:
                            crt_dist.append(np.nan)
            mask_idx_dict[crt_key] = crt_idx
            mask_dist_dict[crt_key] = crt_dist
    # make sure all mask have the same length
    mask_length = [len(v) for k, v in mask_idx_dict.items()]
    if len(np.unique(mask_length)) > 1:
        max_length = max(mask_length)
        for origin_ll in mask_idx_dict:
            mask_idx = mask_idx_dict[origin_ll]
            mask_dist = mask_dist_dict[origin_ll]
            crt_length = len(mask_idx)
            if crt_length < max_length:
                mask_idx += [(0, 0) for i in range(max_length - crt_length)]
                mask_dist += [np.nan for i in range(max_length - crt_length)]
    return mask_idx_dict, mask_dist_dict

def get_regress_matrix(mat, mask_idx_dict, mask_dist_dict, drange):
    """
    preparing dist matrix and value(temperature) matrix
    """
    if isinstance(drange, dict):
        drange = max(list(drange.values()))  # use the max drange if drange varies at different cell positions.
    padding_width = int(np.ceil(drange))
    padding_mat = np.pad(mat, ((padding_width, padding_width), (padding_width, padding_width)),
                         'constant', constant_values=np.nan)
    nrow, ncol = mat.shape
    dist = [mask_dist_dict[(row, col)] for row in range(nrow) for col in range(ncol)]
    value = [[padding_mat[idx] for idx in mask_idx_dict[(row, col)]] for row in range(nrow) for col in range(ncol)]
    dist = np.asarray(dist).T
    value = np.asarray(value).T
    return dist, value

def regress_vectorized(x, y):
    """
    parallelized regression for each column of y on each column of x (y_column_i = x_column_i * a_i + b_i)
    """
    mx, my = np.nanmean(x, axis=0), np.nanmean(y, axis=0)
    cov_xy = np.nansum((x - mx) * (y - my), axis=0)
    tmp1 = np.nansum((x - mx) * (x - mx), axis=0)
    tmp2 = np.sqrt(np.nansum((x - mx) * (x - mx), axis=0)) * np.sqrt(np.nansum((y - my) * (y - my), axis=0))
    # make sure no zeros
    t = []
    for x in tmp1:
        if x == 0.:
            t.append(0.0000001)
        else:
            t.append(x)
    tmp1 = t
    t = []
    for x in tmp2:
        if x == 0.:
            t.append(0.0000001)
        else:
            t.append(x)
    tmp2 = t
    # print(f'tmp1: {tmp1}')
    # print(f'tmp2: {tmp2}')
    B = cov_xy / tmp1  # slope vector
    R = cov_xy / tmp2  # r vector
    return B, R

# @run_time_special
def get_peak_by_adaptive_local_regress(delta_mat, drange, mask_idx_dict, mask_dist_dict, slope_th=-0.1,
                                       detect_type='hot', r2_th=0.1):
    """
    this is the new version by vectorized calculation
    """
    dist_matrix, temperature_matrix = get_regress_matrix(delta_mat, mask_idx_dict, mask_dist_dict, drange)
    slope_vector, r_vector = regress_vectorized(dist_matrix, temperature_matrix)
    slope_mat = slope_vector.reshape(delta_mat.shape)
    r2_mat = r_vector.reshape(delta_mat.shape) ** 2
    if isinstance(slope_th, float) or isinstance(slope_th, int):
        nrow, ncol = delta_mat.shape
        slope_th = np.ones((nrow, ncol)) * slope_th
    if detect_type == 'hot':
        judge_matrix = (slope_mat <= slope_th) * (r2_mat >= r2_th)
        # print(judge_matrix)
    elif detect_type == 'cold':
        judge_matrix = (slope_mat >= slope_th) * (r2_mat >= r2_th)
    tmp = np.where(judge_matrix)
    rst = [(row_idx, col_idx) for row_idx, col_idx in zip(tmp[0], tmp[1])]
    return rst, slope_mat, r2_mat

@run_time_main
def get_peak_by_threshold(mat, th, detect_type='hot'):
    """
    """
    if detect_type == 'hot':
        return [row_col_index for (row_col_index, value) in np.ndenumerate(mat) if value >= th]
    elif detect_type == 'cold':
        return [row_col_index for (row_col_index, value) in np.ndenumerate(mat) if value <= th]

@run_time_main
def plt_heatmap_pygame_mat_pure(x_start, y_start, cell_size, mat, viz_min=0, viz_max=30, border=True, title=None, sensor_resolution=(8,8)):
    """
    Visualize a heatmap of temperature matrix on a given axis, as well as peak cells,
    region of interests (ROI), and others. This function is used for the first subplot

    Arguments:
    ----------------------------------------------
    mat: input temperature matrix
    boxList: a list of peak cells, returned by "get_peak_by_adaptive_local_regress" or "get_peak_by_threshold"
    ax: the axis for this heatmap
    vmin/vmax: min and max temperature values, used to scale colors
    border: if true, ROIs would be visualized with black border
    title: title of this plot

    Returns:
    -----------------------------------------------
    None
    """
    global config

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    ## load pg dict
    screen = pg['screen']
    screen_width = pg['screen_width']
    screen_height = pg['screen_height']
    clock = pg['clock']
    fonts = pg['fonts']
    is_fullscreen = pg['is_fullscreen']

    # draw title text
    if config['run_mode'] == 'serial':
        x_move = 3.25 * cell_size
    else:
        x_move = -1.5 * cell_size

    if title is not None:
        fonts[1].render_to(screen, (x_start + x_move, y_start + 8.2 * cell_size), title, pygame.Color('white'))

    # adjust cell_size according to sensor resolution
    sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
    scale = min(sc1, sc2)
    cell_size *= scale

    # draw all cells
    nrow, ncol = mat.shape
    for i in range(nrow):
        for j in range(ncol):
            # draw rectangle
            reading_norm = (mat[i, j] - viz_min) / (viz_max - viz_min)
            rgb_int = int(reading_norm * 255)
            if rgb_int < 1: rgb_int = 1
            if rgb_int > 255: rgb_int = 255
            color = (rgb_int, rgb_int, rgb_int)
            pygame.draw.rect(screen, color, (x_start + j * cell_size, y_start + i * cell_size, cell_size, cell_size))
            # draw text
            # fonts[0].render_to(screen, (x_start + (j + 0.23) * cell_size, y_start + (i + 0.4) * cell_size),
            #                    str(round(mat[i, j], 2)), pygame.Color('white'))
    return None

# @run_time_main
# def plt_boundary(x_start, y_start, cell_size, mat, viz_min=0, viz_max=30, border=True, title=None, sensor_resolution=(8,8)):
#     global config
    
#     screen = pg['screen']
#     screen_width = pg['screen_width']
#     screen_height = pg['screen_height']
#     clock = pg['clock']
#     fonts = pg['fonts']
#     is_fullscreen = pg['is_fullscreen']

#     if config['run_mode'] == 'serial':
#         x_move = 3.25 * cell_size
#     else:
#         x_move = -1.5 * cell_size

#     sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
#     scale = min(sc1, sc2)
#     cell_size *= scale

#     nrow, ncol = mat.shape
#     pose_detection_mat = np.zeros(mat.shape, dtype=np.uint8)
#     for i in range(nrow):
#         for j in range(ncol):
#             # draw rectangle
#             reading_norm = (mat[i, j] - viz_min) / (viz_max - viz_min)
#             rgb_int = int(reading_norm * 255)
#             if rgb_int < 1: 
#                 rgb_int = 1
#             if rgb_int > 255: 
#                 rgb_int = 255
#             if rgb_int > 240:
#                 pose_detection_mat = 255

#     contours, hierarchy = cv2.findContours(pose_detection_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     for i in range(len(contours)):
#         current_area = cv2.contourArea(contours[i])
#         print(current_area)
#     print('bounding number', len(contours))

#     return None

@run_time_main
def plt_heatmap_pygame_polygon_pure(x_start, y_start, cell_size, boxList, polygons, border=True, detect_type='hot', sensor_resolution=(8.8), timestamp=None, device_id=None):
    """
    Visualize a heatmap of temperature matrix on a given axis, as well as peak cells,
    region of interests (ROI), and others. This function is used for the first subplot

    Arguments:
    ----------------------------------------------
    mat: input temperature matrix
    boxList: a list of peak cells, returned by "get_peak_by_adaptive_local_regress" or "get_peak_by_threshold"
    ax: the axis for this heatmap
    vmin/vmax: min and max temperature values, used to scale colors
    border: if true, ROIs would be visualized with black border
    title: title of this plot

    Returns:
    -----------------------------------------------
    None
    """
    global config, bb_save


    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    ## load pg dict
    screen = pg['screen']
    screen_width = pg['screen_width']
    screen_height = pg['screen_height']
    clock = pg['clock']
    fonts = pg['fonts']
    is_fullscreen = pg['is_fullscreen']

    # adjust cell_size according to sensor resolution
    sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
    scale = min(sc1, sc2)
    before_cell_size = cell_size
    cell_size *= scale

    if viz_detection:
        # draw detected cell boundaries
        # if detect_type == 'hot':
        #     for row_id, col_id in boxList:
        #         pygame.draw.rect(screen, pygame.Color('red'),
        #                          (x_start + col_id * cell_size, y_start + row_id * cell_size, cell_size, cell_size), 1)
        # elif detect_type == 'cold':
        #     for row_id, col_id in boxList:
        #         pygame.draw.rect(screen, pygame.Color('blue'),
        #                          (x_start + col_id * cell_size, y_start + row_id * cell_size, cell_size, cell_size), 1)

        # draw person boundaries
        if border:
            for polygon_idx in range(len(polygons)):
                polygon = polygons[polygon_idx]
                polygon_x, polygon_y = polygon.exterior.xy
                polygonPoints = [(x, y) for x, y in zip(polygon_x, polygon_y)]
                # need coords conversion because of different viz scale due to sensor_resolution
                polygonPoints_01 = [coords_conversion(coord, 'screen', cell_size=before_cell_size,
                                                      sensor_resolution=sensor_resolution)['01'] for coord in polygonPoints]
                polygonPoints = [coords_conversion(coord, '01', cell_size=cell_size,
                                                   sensor_resolution=sensor_resolution)['screen'] for coord in polygonPoints_01]
                
                polygonPointsArray = np.array(polygonPoints, dtype=np.float64)
                maxX, maxY = np.max(polygonPointsArray, axis = 0)
                minX, minY = np.min(polygonPointsArray, axis = 0)

                # pygame.draw.rect(screen, pygame.Color('blue'), 
                #                  (minX, minY, maxX-minX, maxY-minY))
                pygame.draw.polygon(screen, pygame.Color('red'), [[minX,minY], [minX,maxY], [maxX, maxY],[maxX,minY]], 7)

                # polygonPoints = [(x, y-57) for x, y in zip(polygon_x, polygon_y)] ##?????

                # cw: show centroid
                cx, cy = np.asarray([x[0] for x in polygonPoints]).mean(), np.asarray(
                    [x[1] for x in polygonPoints]).mean()
                # fonts[0].render_to(screen, (cx + cell_size * 0.0, cy), str(polygon_idx), pygame.Color('black'), size=14)
            
                # print(polygonPoints)
                if detect_type == 'hot':
                    pygame.draw.polygon(screen, pygame.Color('white'), polygonPoints, 1)
                elif detect_type == 'cold':
                    pygame.draw.polygon(screen, pygame.Color('blue'), polygonPoints, 7)

    return None

def plt_heatmap_pygame(x_start, y_start, cell_size, mat, boxList, viz_min=0, viz_max=30, border=True, title=None,
                       label=None, step=False, max_split_time=np.inf):
    """
    Visualize a heatmap of temperature matrix on a given axis, as well as peak cells,
    region of interests (ROI), and others. This function is used for the first subplot

    Arguments:
    ----------------------------------------------
    mat: input temperature matrix
    boxList: a list of peak cells, returned by "get_peak_by_adaptive_local_regress" or "get_peak_by_threshold"
    ax: the axis for this heatmap
    vmin/vmax: min and max temperature values, used to scale colors
    border: if true, ROIs would be visualized with black border
    title: title of this plot

    Returns:
    -----------------------------------------------
    None
    """
    global config

    global diameter_th

    ROIs = get_rois(boxList, num_neighbor=8)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    ## load pg dict
    screen = pg['screen']
    screen_width = pg['screen_width']
    screen_height = pg['screen_height']
    clock = pg['clock']
    fonts = pg['fonts']
    is_fullscreen = pg['is_fullscreen']

    # draw title text
    if config['run_mode'] == 'serial':
        x_move = 3.25 * cell_size
    else:
        x_move = -1.5 * cell_size
    if title is not None:
        title1 = 'Number of Persons in current frame: {}'.format(len(ROIs))
        fonts[1].render_to(screen, (x_start + x_move, y_start + 8.2 * cell_size), title, pygame.Color('white'))
        # fonts[1].render_to(screen, (x_start, y_start+8.6*cell_size), title1, pygame.Color('red'))
    else:
        title = 'Number of Persons in current frame: {}'.format(len(ROIs))
        # fonts[1].render_to(screen, (x_start, y_start+8.2*cell_size), title, pygame.Color('white'))

    # draw all cells
    nrow, ncol = mat.shape
    for i in range(nrow):
        for j in range(ncol):
            # draw rectangle
            reading_norm = (mat[i, j] - viz_min) / (viz_max - viz_min)
            rgb_int = int(reading_norm * 255)
            if rgb_int < 1: rgb_int = 1
            if rgb_int > 255: rgb_int = 255
            color = (rgb_int, rgb_int, rgb_int)
            pygame.draw.rect(screen, color, (x_start + j * cell_size, y_start + i * cell_size, cell_size, cell_size))
            # draw text
            # fonts[0].render_to(screen, (x_start + (j + 0.23) * cell_size, y_start + (i + 0.4) * cell_size),
            #                    str(round(mat[i, j], 2)), pygame.Color('white'))

    meanPositions = []
    if viz_detection:
        # draw detected cell boundaries
        for row_id, col_id in boxList:
            pygame.draw.rect(screen, pygame.Color('red'),
                             (x_start + col_id * cell_size, y_start + row_id * cell_size, cell_size, cell_size), 1)

        # draw person boundaries
        if border:
            rp1 = lambda x: ((x[1] + 1.0) * cell_size + x_start, (x[0] + 1.0) * cell_size + y_start)
            rp2 = lambda x: ((x[1] - 0.0) * cell_size + x_start, (x[0] - 0.0) * cell_size + y_start)
            rp3 = lambda x: ((x[1] - 0.0) * cell_size + x_start, (x[0] + 1.0) * cell_size + y_start)
            rp4 = lambda x: ((x[1] + 1.0) * cell_size + x_start, (x[0] - 0.0) * cell_size + y_start)
            polygons = []
            for roi_idx, roi in enumerate(ROIs):
                rectPoints = list(set([f(x) for x in roi for f in (rp1, rp2, rp3, rp4)]))
                rectPoints = np.array(rectPoints)
                # convex hull
                hull = ConvexHull(rectPoints)
                vertices = np.asarray([rectPoints[idx] for idx in hull.vertices])
                polygon = Polygon(vertices)

                splitPolygons, diameters, majorAxisPoitsList, centroids = split_polygon_recurrent(
                    polygon, scale=1 / cell_size, diameter_th=diameter_th, max_split_time=max_split_time)
                polygons += splitPolygons
                meanPositions += centroids

                for polygon_idx in range(len(polygons)):
                    polygon = polygons[polygon_idx]
                    # useless, diameter = polygon_major_axis(polygon)
                    cx, cy = meanPositions[polygon_idx]
                    fonts[0].render_to(screen, (cx, cy), str(polygon_idx), pygame.Color('blue'), size=15)
                    # ax.text(cx, cy, '{}\n{:4.2f}'.format(polygon_idx, diameter), color='r', ha='center',
                    #         va='center')
                    polygon_x, polygon_y = polygon.exterior.xy
                    polygonPoints = [(x, y) for x, y in zip(polygon_x, polygon_y)]
                    pygame.draw.polygon(screen, pygame.Color('red'), polygonPoints, 10)

    if viz_label:
        # draw ground truth label positions
        if label is not None and len(label) > 0:
            BLUE = (0, 0, 255)
            for pos in label:
                pygame.draw.circle(screen, BLUE,
                                   (int(x_start + pos[0] * cell_size * 8), int(y_start + pos[1] * cell_size * 8)),
                                   int(0.1 * cell_size), 0)

    return ROIs, meanPositions

@run_time_main
def plt_mat_text_pygame(x_start, y_start, cell_size, mat, viz_min=0, viz_max=30, title=None, sensor_resolution=(8,8)):
    """
    Visualize a heatmap of a matrix on a given axis, without additional process
    This function is used for the slope/r2 subplots.

    Arguments:
    ----------------------------------------------
    mat: input matrix
    ax: the axis for this heatmap
    title: title of this plot

    Returns:
    -----------------------------------------------
    None
    """

    ## load pg dict
    screen = pg['screen']
    screen_width = pg['screen_width']
    screen_height = pg['screen_height']
    clock = pg['clock']
    fonts = pg['fonts']
    is_fullscreen = pg['is_fullscreen']

    # draw title text
    if title is not None:
        fonts[1].render_to(screen, (x_start, y_start + 8.2 * cell_size), title, pygame.Color('white'))

    # adjust cell_size according to sensor resolution: valid for grid coord
    sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
    scale = min(sc1, sc2)
    cell_size *= scale

    # draw all cells
    nrow, ncol = mat.shape
    for i in range(nrow):
        for j in range(ncol):
            # draw rectangle
            reading_norm = (mat[i, j] - viz_min) / (viz_max - viz_min)
            rgb_int = int(reading_norm * 255)
            if rgb_int < 1: rgb_int = 1
            if rgb_int > 255: rgb_int = 255
            color = (rgb_int, rgb_int, rgb_int)
            pygame.draw.rect(screen, color, (x_start + j * cell_size, y_start + i * cell_size, cell_size, cell_size))
            # draw text
            # fonts[0].render_to(screen, (x_start + (j + 0.23) * cell_size, y_start + (i + 0.4) * cell_size),
            #                    str(round(mat[i, j], 2)), pygame.Color('white'))

def true_data_to_mat(json_input, shape=(8, 8)):
    """
    Convert a json (path or file) to a temperature matrix

    Arguments:
    ----------------------------------------------
    json_input: the path of a json file or the dict read from a json file
    shape: the shape of the matrix

    Returns:
    -----------------------------------------------
    mat: temperature matrix
    """
    global data_scale, zero_noise_filter

    if isinstance(json_input, str):
        # print('Input is a path str')
        data = json.load(open(json_input, 'r'))
    elif isinstance(json_input, dict):
        # print('Input is a json dict')
        data = json_input
    else:
        print('[Error] Input is neither path nor json dict')
        return
    if len(data['data']) == shape[0] * shape[1]:
        # print(f'data["data"]: {data["data"]}')
        data_1d = data['data'].copy()
        if zero_noise_filter:
            median = int(np.median([x for x in data_1d if x > 0]))
            # print(f'median: {median}')
            for i in range(len(data_1d)):
                if data_1d[i] == 0:
                    # print('\n!!!!!!!!!!!!!!!!!! zero noise replaced !!!!!!!!!!!!!!!!!!!!!\n')
                    data_1d[i] = median
        # print(f'data_1d: {data_1d}')
        mat = np.asarray(data_1d).reshape(shape) * data_scale
        return mat
    else:
        print('[Error] reading matrix shape wrong')
        return None

def get_rois(cells, num_neighbor=4):
    """
    Get region of interest (ROI) from list of peak cells. Each ROI is a continous area of cells and is regarded as 1 person.

    Arguments:
    ----------------------------------------------
    cells: a list of peak cells, each defined by a tuple of (row_id, col_id)

    Returns:
    -----------------------------------------------
    a list of ROIs
    """
    # 4 neighbors of a cell
    n1 = lambda x: (x[0] + 1, x[1])
    n2 = lambda x: (x[0] - 1, x[1])
    n3 = lambda x: (x[0], x[1] + 1)
    n4 = lambda x: (x[0], x[1] - 1)

    n5 = lambda x: (x[0] + 1, x[1] + 1)
    n6 = lambda x: (x[0] - 1, x[1] - 1)
    n7 = lambda x: (x[0] - 1, x[1] + 1)
    n8 = lambda x: (x[0] + 1, x[1] - 1)

    ROIs = []
    for cell in cells:
        belong = False
        for roi in ROIs:
            if cell in roi['adjacent']:
                belong = True
                # belong to this roi, modify the roi
                roi['region'].append(cell)
                if num_neighbor == 4:
                    roi['adjacent'] = list(set(roi['adjacent'] + [n1(cell), n2(cell), n3(cell), n4(cell)]))
                elif num_neighbor == 8:
                    roi['adjacent'] = list(set(
                        roi['adjacent'] + [n1(cell), n2(cell), n3(cell), n4(cell), n5(cell), n6(cell), n7(cell),
                                           n8(cell)]))
                break
        # do not belong to any roi, create a roi by itself
        if not belong:
            if num_neighbor == 4:
                newROI = {
                    'region': [cell],
                    'adjacent': list(set([f(x) for x in [cell] for f in (n1, n2, n3, n4)]))
                }
            elif num_neighbor == 8:
                newROI = {
                    'region': [cell],
                    'adjacent': list(set([f(x) for x in [cell] for f in (n1, n2, n3, n4, n5, n6, n7, n8)]))
                }
            ROIs.append(newROI)
    if len(ROIs) > 1:
        while True:
            change = False
            for i in range(len(ROIs) - 1):
                for j in range(i + 1, len(ROIs)):
                    roi_i, roi_j = ROIs[i], ROIs[j]
                    intersect = [x for x in roi_i['region'] if x in roi_j['adjacent']] + \
                                [x for x in roi_j['region'] if x in roi_i['adjacent']]
                    if len(intersect) == 0:
                        continue
                    else:
                        mergedROI_region = roi_i['region'] + roi_j['region']
                        if num_neighbor == 4:
                            mergedROI_adjacent = list(set([f(x) for x in mergedROI_region for f in (n1, n2, n3, n4)]))
                        elif num_neighbor == 8:
                            mergedROI_adjacent = list(
                                set([f(x) for x in mergedROI_region for f in (n1, n2, n3, n4, n5, n6, n7, n8)]))
                        roi_i = {'region': mergedROI_region, 'adjacent': mergedROI_adjacent}
                        roi_j = {'region': [], 'adjacent': []}
                        ROIs[i] = roi_i
                        ROIs[j] = roi_j
                        change = True
                    if change is True:
                        break
                if change is True:
                    break
            if change is False:
                break

    return [x['region'] for x in ROIs if len(x['region']) > 0]

@run_time_main
def viz_tracks(track_info, frame_num, x_start, y_start, cell_size, show_reflect=False, sensor_resolution=(8,8)):
    """
    """
    ## load pg dict
    screen = pg['screen']
    screen_width = pg['screen_width']
    screen_height = pg['screen_height']
    clock = pg['clock']
    fonts = pg['fonts']
    is_fullscreen = pg['is_fullscreen']

    # rename
    trks = track_info
    i = frame_num + 1

    # adjust cell_size according to sensor resolution
    sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
    scale = min(sc1, sc2)
    cell_size *= scale

    # loop tracks
    for trk in trks:

        # track color
        random.seed(trk['t'])
        rnd_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        # if i in this track duration + X frames, tracks disappear afterwards
        if i >= trk['t'] and i < trk['t'] + len(trk['cood']) + tracking_disappear_delay:
            # draw dots
            coods = trk['cood'][:(i - trk['t'])]
            for cood in coods:
                cood_screen = [round(x_start + cood[0] * cell_size * sensor_resolution[1]),
                               round(y_start + cood[1] * cell_size * sensor_resolution[0])]
                pygame.draw.circle(screen, rnd_color, cood_screen, 5, 0)
            for j in range(len(coods) - 1):
                pt1 = [round(x_start + coods[j][0] * cell_size * sensor_resolution[1]),
                       round(y_start + coods[j][1] * cell_size * sensor_resolution[0])]
                pt2 = [round(x_start + coods[j + 1][0] * cell_size * sensor_resolution[1]),
                       round(y_start + coods[j + 1][1] * cell_size * sensor_resolution[0])]
                pygame.draw.line(screen, rnd_color, pt1, pt2, 3)
            if show_reflect:
                reflect_point_cood = trk['reflect']
                reflect_point_cood_screen = [round(x_start + reflect_point_cood[0] * cell_size * sensor_resolution[1]),
                                             round(y_start + reflect_point_cood[1] * cell_size * sensor_resolution[0])]
                # print('reflect_coord: {}'.format(reflect_point_cood ))
                # print('reflect_coord_screen: {}'.format(reflect_point_cood_screen))
                pygame.draw.circle(screen, rnd_color, reflect_point_cood_screen, 5, 1)

@run_time_main
def viz_counts(track_info, frame_num, door_line_start, door_line_end, x_start, y_start, cell_size, sensor_resolution=(8,8)):
    """
    """
    global ppl_count_global, ppl_count_in_global, ppl_count_out_global
    global count_predict_tmp

    ## load pg dict
    screen = pg['screen']
    screen_width = pg['screen_width']
    screen_height = pg['screen_height']
    clock = pg['clock']
    fonts = pg['fonts']
    is_fullscreen = pg['is_fullscreen']

    # rename
    trks = track_info
    i = frame_num
    ds = door_line_start
    de = door_line_end

    # door line horizontal or vertical
    if ds[1] == de[1]:
        door_line_dir = 'h'
    elif ds[0] == de[0]:
        door_line_dir = 'v'
    else:
        print('ERROR: door line is neither horizontal or vertical')

    line_door = LineString([ds, de])

    # adjust cell_size according to sensor resolution
    sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
    scale = min(sc1, sc2)
    cell_size *= scale

    # viz door line
    ds_screen = (round(x_start + ds[0] * cell_size * 8), round(y_start + ds[1] * cell_size * 8))
    de_screen = (round(x_start + de[0] * cell_size * 8), round(y_start + de[1] * cell_size * 8))
    if display_count:
        pygame.draw.line(screen, pygame.Color('white'), ds_screen, de_screen, 2)

    # counting mark colors
    mark_color_1 = pygame.Color('red')
    mark_color_2 = pygame.Color('gray')

    # loop tracks
    ppl_count = 0
    ppl_count_in = 0
    ppl_count_out = 0
    count_predict_tmp = []
    for trk in trks:
        intersections = []
        # if i in this track duration + X frames, display and count, count circles become gray afterwords
        if i >= trk['t']:
            coods = trk['cood'][:(i - trk['t'])]  # this is actually the full track if run in realtime
            for j in range(len(coods) - 1):
                pt1 = (coods[j][0], coods[j][1])
                pt2 = (coods[j + 1][0], coods[j + 1][1])
                line_test = LineString([pt1, pt2])
                int_pt_shapely = line_door.intersection(line_test)
                if not int_pt_shapely.is_empty:
                    int_pt = (int_pt_shapely.x, int_pt_shapely.y)
                    # viz counting intersection pt
                    int_pt_screen = (
                    round(x_start + int_pt[0] * cell_size * 8), round(y_start + int_pt[1] * cell_size * 8))
                    if i < trk['t'] + len(trk['cood']) + tracking_disappear_delay:
                        mark_color = mark_color_1
                    else:
                        mark_color = mark_color_2
                    # calculate count direction: door line horizontal: up=in, down=out; vertical: left=in, right=out.
                    if door_line_dir == 'h':
                        if pt1[1] <= pt2[1]:
                            count_dir = 'out'
                        else:
                            count_dir = 'in'
                    elif door_line_dir == 'v':
                        if pt1[0] <= pt2[0]:
                            count_dir = 'out'
                        else:
                            count_dir = 'in'
                    intersections.append(
                        {'t': trk['t'] + j, 'int_pt_screen': int_pt_screen, 'mark_color': mark_color, 'dir': count_dir})
            # print('intersections: \n{}'.format(intersections))
            # removing possible duplicated intersections
            intersections = counting_remove(intersections, gap_dict=gap_dict_counting)
            # draw dots
            # pygame.draw.circle(screen, mark_color, int_pt_screen, 10, 1)
            # count
            # ppl_count += 1

            # draw plots and count
            if display_count:
                for point in intersections:
                    pygame.draw.circle(screen, point['mark_color'], point['int_pt_screen'], 10, 1)
                    # draw direction
                    if door_line_dir == 'h':
                        if point['dir'] == 'out':
                            pygame.draw.line(screen, point['mark_color'], point['int_pt_screen'],
                                             (point['int_pt_screen'][0], point['int_pt_screen'][1] + 10), 1)
                        elif point['dir'] == 'in':
                            pygame.draw.line(screen, point['mark_color'], point['int_pt_screen'],
                                             (point['int_pt_screen'][0], point['int_pt_screen'][1] - 10), 1)
                    elif door_line_dir == 'v':
                        if point['dir'] == 'out':
                            pygame.draw.line(screen, point['mark_color'], point['int_pt_screen'],
                                             (point['int_pt_screen'][0] + 10, point['int_pt_screen'][1]), 1)
                        elif point['dir'] == 'in':
                            pygame.draw.line(screen, point['mark_color'], point['int_pt_screen'],
                                             (point['int_pt_screen'][0] - 10, point['int_pt_screen'][1]), 1)

            # calc count eval
            if count_eval:
                for point in intersections:
                    count_predict_tmp.append({'i': point['t'], 'count': 1})

            ppl_count += len(intersections)
            # hack
            if disable_lower_count:
                if not disable_in_count:
                    ppl_count_in += len([x for x in intersections if
                                         x['dir'] == 'in' and x['int_pt_screen'][1] < round(y_start + 5 * cell_size)])
                if not disable_out_count:
                    ppl_count_out += len([x for x in intersections if
                                          x['dir'] == 'out' and x['int_pt_screen'][1] < round(y_start + 5 * cell_size)])
            elif disable_upper_count:
                if not disable_in_count:
                    ppl_count_in += len([x for x in intersections if
                                         x['dir'] == 'in' and x['int_pt_screen'][1] > round(y_start + 3 * cell_size)])
                if not disable_out_count:
                    ppl_count_out += len([x for x in intersections if
                                          x['dir'] == 'out' and x['int_pt_screen'][1] > round(y_start + 3 * cell_size)])
            else:
                if not disable_in_count:
                    ppl_count_in += len([x for x in intersections if x['dir'] == 'in'])
                if not disable_out_count:
                    ppl_count_out += len([x for x in intersections if x['dir'] == 'out'])

    # viz ppl counting number
    # text = f'ppl_count = {round(ppl_count_global+ppl_count)}; ppl_count_in = {round(ppl_count_in_global+ppl_count_in)}; ppl_count_out = {round(ppl_count_out_global+ppl_count_out)}'
    text = f'ppl_count_in = {round(ppl_count_in_global + ppl_count_in)}; ppl_count_out = {round(ppl_count_out_global + ppl_count_out)}'
    fonts[2].render_to(screen, (x_start - 1.5 * cell_size, y_start + 9.0 * cell_size), text, pygame.Color('red'))
    return ppl_count, ppl_count_in, ppl_count_out

def pygame_init(screen_width=1280, screen_height=720, fullscreen=False, title=None, logo_path=None, font_path=None):
    """
    Initialize PyGame screen surface.
    This is normally one of the earliest functions to run in __main__

    Arguments:
    ----------------------------------------------
    ...

    Returns:
    -----------------------------------------------
    pg as a dict that includes:
    screen: the pygame screen
    screen_width
    screen_height
    clock: the pygame clock
    fonts: a list of font objects (s, m, l) None if font_path is not secified
    is_fullscreen
    """

    global fps

    # initialize pygame
    pygame.init()
    if logo_path is not None:
        logo = pygame.image.load(logo_path)
        pygame.display.set_icon(logo)
    if title is not None:
        pygame.display.set_caption(title)
    if fullscreen:
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        is_fullscreen = True
    else:
        screen = pygame.display.set_mode((screen_width, screen_height))
        is_fullscreen = False

    # pygame gui
    manager = pygame_gui.UIManager((screen_width, screen_height))
    # manager = pygame_gui.UIManager((screen_width, screen_height), 'theme.json')

    # font object
    font_s = pygame.freetype.Font(font_path, 9)
    font_m = pygame.freetype.Font(font_path, 14)
    font_l = pygame.freetype.Font(font_path, 21)
    fonts = [font_s, font_m, font_l]

    # for pygame fps limit
    clock = pygame.time.Clock()
    time_delta = clock.tick(fps)

    pg = {
        'screen': screen,
        'screen_width': screen_width,
        'screen_height': screen_height,
        'clock': clock,
        'fonts': fonts,
        'is_fullscreen': is_fullscreen,
        'manager': manager,
        'time_delta': time_delta
    }

    return pg

def save_trace_to_file(trace_dict, save_trace_path):
    if not os.path.exists(save_trace_path):
        json.dump([trace_dict], open(save_trace_path, 'w'), indent=2)
        print('New trace saved to the new file.')
    else:
        saved_trace = json.load(open(save_trace_path, 'r'))
        saved_trace.append(trace_dict)
        json.dump(saved_trace, open(save_trace_path, 'w'), indent=2)
        print('New trace saved to the existing file.')

def coords_conversion(origin_coords, origin_coords_type='01',
                      cell_size=30, sensor_resolution=(8,8), screen_width=1280, screen_height=720) :
    x_start, y_start = int(screen_width / 3 * 0.2), int(screen_height / 2 * 0.2)
    if origin_coords_type == '01':
        coords_01 = tuple(origin_coords)
        coords_grid = (
            min(int(origin_coords[1] * sensor_resolution[0]), sensor_resolution[0] - 1),
            min(int(origin_coords[0] * sensor_resolution[1]), sensor_resolution[1] - 1)
        )
        coords_screen = (
            origin_coords[0] * sensor_resolution[1] * cell_size + x_start,
            origin_coords[1] * sensor_resolution[0] * cell_size + y_start
        )
    if origin_coords_type == 'screen':
        coords_screen = tuple(origin_coords)
        coords_grid = (
            min(int((origin_coords[1] - y_start) / cell_size), sensor_resolution[0] - 1),
            min(int((origin_coords[0] - x_start) / cell_size), sensor_resolution[1] - 1)
        )
        coords_01 = (
            (origin_coords[0] - x_start) / (cell_size * sensor_resolution[1]),
            (origin_coords[1] - y_start) / (cell_size * sensor_resolution[0])
        )
    if origin_coords_type == 'grid':
        coords_grid = tuple(origin_coords)
        coords_01 = (
            origin_coords[1] / sensor_resolution[1],
            origin_coords[0] / sensor_resolution[0],
        )
        coords_screen = (
            origin_coords[1] * cell_size + x_start,
            origin_coords[0] * cell_size + y_start
        )
    coords = {'01':coords_01, 'screen':coords_screen, 'grid':coords_grid}
    return coords


class Activity_Map():
    def __init__(self, sensor_resolution=(8,8), map_resolution=None, resolution_scale=None,
                 buffer_dist=1, ell_ob_th=0.5, ell_area_th=0.2, hours_elapsed_for_level_down=6, enable_level_down=True,
                 cell_size=30, screen_width=1280, screen_height=720):
        """
        Args:
            sensor_resolution: tuple of int, resolution of sensor
            map_resolution: tuple of int, resolution of activity map, not neccessarily equals to sensor_resolution
            resolution_scale: None of int, if int, the map_resolution would be resolution_scale * sensor_resolution
            buffer_dist: specify the radius of buffer area from the center cell, all cells in the buffer area of the trace/points
                would be leveled up or leveld down
            ell_ob_th: float, ellipse oblateness threshold, oblateness is ranged from 0~1; 0=circle, 1=line;
                a trace with oblateness (of its std ellipse) >= ell_ob_th would be regarded as reliable
            ell_area_th: float, ellipse area threshold, a trace with area (of its std ellipse) >= ell_area_th would be regarded as reliable
            hours_elapsed_for_level_down: float, when too long time has passed for a cell without a new detection,
                i.e. longer than hours_elapsed_for_level_down, we can decrease its level in activity map
            enable_level_down: bool, whether or not to decrease the level of the cells without new detections for too long time
            cell_size: int, size of a cell in pygame screen (in pixel), just for coordinates conversion
            screen_width: int, pygame screen width, just for coordinates conversion
            screen_height: int, pygame screen height, just for coordinates conversion
        """
        self.sensor_resolution = sensor_resolution
        if resolution_scale:
            self.map_resolution = tuple([int(x*resolution_scale) for x in sensor_resolution])
        elif map_resolution and isinstance(map_resolution, tuple) and len(map_resolution)==2:
            self.map_resolution = map_resolution
        else:
            self.map_resolution = sensor_resolution
        self.activity_map = np.zeros(self.map_resolution)
        self.cell_indices = [(i,j) for i in range(self.map_resolution[0]) for j in range(self.map_resolution[1])]
        self.last_detection_time = {cell_index:None for cell_index in self.cell_indices}
        self.dist_among_cells = None
        self.cell_buffer_indices_lookup = None
        self.buffer_dist = buffer_dist
        self.ell_ob_th = ell_ob_th
        self.ell_area_th = ell_area_th
        self.hours_elapsed_for_level_down = hours_elapsed_for_level_down
        self.enable_level_down = enable_level_down
        self.cell_size = cell_size
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.status = 'only updating'

    def mqtt_subscribe_for_reading_activity_map(self, mqtt_read_am_in_topic='', client=None, mqtt_qos=1):
        """
        if need to read saved activity map from mqtt, subscribe to corresponding topic
        basically do not need this, because when lauching mqtt_processes_for_am, we have already subscribed
        Args:
            mqtt_read_am_in_topic: str, if read_from='mqtt', specify the top to read
            client: object, if read_from='mqtt', specify the mqtt client
            mqtt_qos: if read_from='mqtt', specify qos
        Returns:
        """
        client.subscribe(mqtt_read_am_in_topic, mqtt_qos)

    def read_activity_map(self, read_from='mqtt', saved_file_path='', data_type='values'):
        """
        Args:
            read_from: str, 'mqtt'=read from mqtt, 'saved_file'=read from save file
            saved_file_path: if read_from='saved_file', specify the path of this file
            data_type: if 'values', only read activity map values (matrix);
                       if 'params', also read init params, such as sensor_resolution, map_resolution, etc,
                       and use these params to reconstruct an activity map with the same spec
        Returns:
        """
        global AM_FROM_MQTT
        # AM_FROM_MQTT = None
        if not os.path.exists(saved_file_path) and read_from == 'saved_file':
            print(os.path.abspath(saved_file_path))
            print('Error: activity map saved file path does not exist')
            return
        try:
            if read_from == 'mqtt':
                while AM_FROM_MQTT is None:
                    continue
                activity_map_dict = AM_FROM_MQTT
            elif read_from == 'saved_file':
                activity_map_dict = pickle.load(open(saved_file_path, 'rb'))
        except:
            print('Error: failed to retrieve activity map')
            return
        if data_type == 'values':
            retrived_activity_map = activity_map_dict['activity_map']
            if retrived_activity_map.shape != self.map_resolution:
                print('Error: invalid map resolution: expect = {}, read = {}'.format(self.map_resolution, retrived_activity_map.shape ))
                raise
            self.activity_map = retrived_activity_map
        elif data_type == 'params':
            # self.dist_among_cells = activity_map_dict['dist_among_cells']
            # self.cell_buffer_indices_lookup = activity_map_dict['cell_buffer_indices_lookup']
            # self.last_detection_time = activity_map_dict['last_detection_time']
            retrived_activity_map_params = activity_map_dict
            activity_map_params = [
                'sensor_resolution', 'map_resolution', 'buffer_dist', 'ell_ob_th', 'ell_area_th', 'activity_map',
                'hours_elapsed_for_level_down', 'enable_level_down', 'cell_size', 'screen_width', 'screen_height'
            ]
            for attr in activity_map_params:
                setattr(self, attr, retrived_activity_map_params[attr])
            self.activity_map = np.zeros(self.map_resolution)
            self.cell_indices = [(i, j) for i in range(self.map_resolution[0]) for j in range(self.map_resolution[1])]
            self.last_detection_time = {cell_index: None for cell_index in self.cell_indices}
        # print('Successfully read activity map.')

    def dump_activity_map_pickle(self, save_path=None, data_type='values'):
        """
        Args:
            save_path: save activity map in dict to this file, if None, do not save
            data_type: if 'values', only save activity map values (matrix);
                       if 'params', also save init params, such as sensor_resolution, map_resolution, etc,
                       and use these params so that we can reconstruct an activity map with the same spec later
        Returns:
            dumps_result: the dumped am dict in bytes, use pickle.loads(dumps_result) to retrieve the am dict
        """
        if data_type == 'values':
            activity_map_dict = {'activity_map': self.activity_map}
        elif data_type == 'params':
            activity_map_params = [  # do not include 'status'
                'sensor_resolution', 'map_resolution', 'buffer_dist', 'ell_ob_th', 'ell_area_th', 'activity_map',
                'hours_elapsed_for_level_down', 'enable_level_down', 'cell_size', 'screen_width', 'screen_height'
            ]
            activity_map_dict = {}
            for attr in activity_map_params:
                activity_map_dict[attr] = getattr(self, attr)
        dumps_result = pickle.dumps(activity_map_dict)
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(activity_map_dict, f)
        return dumps_result

    def level_up_for_detections(self, detection_coords, coords_mode='screen', assigned_level=2, mark_time=True):
        """
        For a single point, find its buffer area, and increase the level for cells in this area to certain level
        Args:
            detection_coords: tuple, detection coordinate of ONE point => (x,y)
            coords_mode: string, if '01`, then the coordinate is scaled to 0~1; if 'grid', then the coordinate is the grid cell indices;
                if 'screen', then the coordinate is for pygame screen viz.
            assigned_level: int, specify the level for this detection point, generally 1 or 2
            mark_time: bool, if true, update the last detection time for the cells whose level were leveled up.

        Returns:
        """
        if self.status in ['only applying', 'stop working']: return
        if self.cell_buffer_indices_lookup is None:
            self.get_cell_buffer_indices()
        if coords_mode == 'map_grid':
            detection_coords_grid_in_map = detection_coords
        else:
            if coords_mode != '01':
                detection_coords_01 = [
                    coords_conversion(coord, coords_mode, self.cell_size, self.sensor_resolution, self.screen_width, self.screen_height)['01']
                    for coord in detection_coords
                ]
            else:
                detection_coords_01 = detection_coords
            detection_coords_grid_in_map = [
                coords_conversion(coord, '01', self.cell_size, self.map_resolution, self.screen_width, self.screen_height)['grid']
                for coord in detection_coords_01
            ]
        cells_to_level_up = []
        for idx, coord in enumerate(detection_coords_grid_in_map):
            this_cell = (int(coord[0]), int(coord[1]))
            try:
                cells_to_level_up.extend(self.cell_buffer_indices_lookup[this_cell])
            except:
                if coords_mode == 'map_grid':
                    print('Activity Map Error: grid_coords = {}, 01_coords = unavailable'.format(coord))
                else:
                    print('Activity Map Error: grid_coords = {}, 01_coords = {}'.format(coord, detection_coords_01[idx]))
        cells_to_level_up = list(set(cells_to_level_up))
        rows_to_level_up = [cell[0] for cell in cells_to_level_up]
        cols_to_level_up = [cell[1] for cell in cells_to_level_up]
        crt_activity_map = self.activity_map.copy()
        new_activity_map = crt_activity_map.copy()
        new_activity_map[rows_to_level_up, cols_to_level_up]= assigned_level
        # make sure the level could only go up but not down
        new_activity_map = np.maximum(new_activity_map, crt_activity_map)
        self.activity_map = new_activity_map
        if mark_time:
            crt_timestamp = time.time()
            self.last_detection_time.update({cell_index: crt_timestamp for cell_index in cells_to_level_up})

    def get_continuous_trace_coords_grid(self, trace_coords):
        """
        For a given trace defined only by the coordinates of some nodes, get continuous coordinates along the trace
        For example: for trace [(0,0), (0,2), (2,2)], it will generate [(0,0), (0,1), (0,2), (1,2), (2,2)]
        Args:
            trace_coords: list, coordinates of the nodes (in grid mode) to define a trace
        Returns:
            full_continuous_trace_coords_grid: list, coordinates of all cells (in grid mode) along the trace
        """
        # trace_coords must be in grid mode
        full_continuous_trace_coords_grid = []
        for idx in range(len(trace_coords) - 1):
            this_coord, next_coord = trace_coords[idx], trace_coords[idx+1]
            crt_continuous_trace_coords_grid = list(bresenham(this_coord[0], this_coord[1], next_coord[0], next_coord[1]))
            if len(full_continuous_trace_coords_grid) > 0 and full_continuous_trace_coords_grid[-1] == crt_continuous_trace_coords_grid[0]:
                full_continuous_trace_coords_grid += crt_continuous_trace_coords_grid[1:]
            else:
                full_continuous_trace_coords_grid += crt_continuous_trace_coords_grid
        return full_continuous_trace_coords_grid


    def level_up_for_traces(self, trace, coords_mode='01'):
        """
        For a trace, find its buffer area, and increase the level for cells in that area to certain level.
        If the trace is reliaable enough (meet ellipse thresholds), level up to 2; other wise level up to 1
        Args:
            trace: dict, trace generated by analyze_single_frame function, where trace['cood'] is a list of node points (tuple, (x,y))
            coords_mode:string, if '01`, then the coordinate is scaled to 0~1; if 'grid', then the coordinate is the grid cell indices;
                if 'screen', then the coordinate is for pygame screen viz.
        Returns:
        """
        if 'to_activity_map' in trace and trace['to_activity_map'] is True:
            return
        if self.status in ['only applying', 'stop working']: return
        x_coords = [x[0] for x in trace['cood']]
        y_coords = [x[1] for x in trace['cood']]
        ellipse_result = std_ellipse(x_coords, y_coords, n_std=2)
        ell_area, ell_ob = ellipse_result['area'], ellipse_result['ob']  #area and obesity of the std ellipse
        # print('Ellapse indices for current trace: area = {}, ob = {}'.format(ell_area, ell_ob))
        trace_coords = trace['cood']
        if coords_mode != '01':
            trace_coords_01 = [
                coords_conversion(coord, coords_mode, self.cell_size, self.sensor_resolution, self.screen_width, self.screen_height)['01']
                for coord in trace_coords
            ]
        else:
            trace_coords_01 = trace_coords
        trace_coords_grid_in_map = [
            coords_conversion(coord, '01', self.cell_size, self.map_resolution, self.screen_width, self.screen_height)['grid']
            for coord in trace_coords_01
        ]
        continuous_trace_coords_grid_in_map = self.get_continuous_trace_coords_grid(trace_coords_grid_in_map)
        if ell_area >= self.ell_area_th or ell_ob >= self.ell_ob_th:
            self.level_up_for_detections(continuous_trace_coords_grid_in_map, coords_mode='map_grid', assigned_level=2, mark_time=True)
        else:
            self.level_up_for_detections(continuous_trace_coords_grid_in_map, coords_mode='map_grid', assigned_level=1, mark_time=False)
        trace['to_activity_map'] = True

    def filter_for_new_detections(self, detection_coords, coords_mode='screen'):
        """
        When acitivty map is in the status of "applying", when a list of new detection points come in, it will figure out each point falls
            in which level area
        Args:
            detection_coords: list of tuple, coordinates of new detection points
            coords_mode: string, if '01`, then the coordinate is scaled to 0~1; if 'grid', then the coordinate is the grid cell indices;
                if 'screen', then the coordinate is for pygame screen viz.
        Returns:
            rst: dict, in format of {0:[], 1:[], 2:[]}, indicating coordinates that fall in each level. rst[0] will give coordinates
                that are in the area of level 2.
        """
        if coords_mode != '01':
            detection_coords_01 = [
                coords_conversion(coord, coords_mode, self.cell_size, self.sensor_resolution, self.screen_width,
                                  self.screen_height)['01']
                for coord in detection_coords
            ]
        else:
            detection_coords_01 = detection_coords
        detection_coords_grid_in_map = [
            coords_conversion(coord, '01', self.cell_size, self.map_resolution, self.screen_width, self.screen_height)[
                'grid']
            for coord in detection_coords_01
        ]
        rst = {0: [], 1: [], 2:[]}
        for coord_origin, coord_grid_in_map in zip(detection_coords, detection_coords_grid_in_map):
            if self.activity_map[coord_grid_in_map] == 0:
                rst[0].append(coord_origin)
            elif self.activity_map[coord_grid_in_map] == 1:
                rst[1].append(coord_origin)
            elif self.activity_map[coord_grid_in_map] == 2:
                rst[2].append(coord_origin)
        return rst


    def level_down_for_inactive_cells(self, assigned_level='auto'):
        """
        When self.eanable_level_down is True, the cells that have too long time (controlled by self.hours_elapsed_for_level_down)
            without new reliable detections will be leveled down to certain level.
        Args:
            assigned_level: int / string, for cells to be leveled down, specify the new level. If 'auto', then new
                level would be original level minus 1
        Returns:

        """
        if self.status in ['only applying', 'stop working']: return
        if self.enable_level_down is False: return
        crt_time = time.time()
        time_eplased_since_last_detection = {cell: (crt_time - last_detection_time) / 3600
            for cell, last_detection_time in self.last_detection_time.items() if last_detection_time is not None
        }
        cells_to_level_down = [cell
            for cell, time_elapsed in time_eplased_since_last_detection.items()
            if time_elapsed > self.hours_elapsed_for_level_down
        ]
        rows_to_level_down = [cell[0] for cell in cells_to_level_down]
        cols_to_level_down = [cell[0] for cell in cells_to_level_down]
        if assigned_level == 'auto':
            self.activity_map[rows_to_level_down, cols_to_level_down] = self.activity_map[rows_to_level_down, cols_to_level_down] - 1
            self.activity_map = np.maximum(self.activity_map, np.zeros_like(self.activity_map))
        elif assigned_level in [0,1]:
            self.activity_map[rows_to_level_down, cols_to_level_down] = assigned_level
        else:
            self.activity_map[rows_to_level_down, cols_to_level_down] = 0

    def change_status(self, new_status):
        """
        Change status of activity map
        Args:
            new_status: string, new status of the activity map, must be one of 'only updating', 'only applying',
                'updating and applying', 'stop working'
        Returns:
        """
        if new_status not in ['only updating', 'only applying', 'updating and applying', 'stop working']:
            print('Error: invalid status, must be one of only updating, only applying, updating and applying, stop working')
            return
        self.status = new_status


    def get_distance_among_cells(self):
        """
        To calculate distance between each pair of two cells in the activity map, used to generate cell_buffer_indices_lookup
        """
        dist_among_cells = {}
        cell_indices_len = len(self.cell_indices)
        for i in range(cell_indices_len - 1):
            for j in range(i, cell_indices_len):
                cell_i, cell_j = self.cell_indices[i], self.cell_indices[j]
                crt_dist = np.sqrt((cell_i[0] - cell_j[0])**2 + (cell_i[1] - cell_j[1])**2)
                if cell_i not in dist_among_cells:
                    dist_among_cells[cell_i] = {cell_j: crt_dist}
                else:
                    dist_among_cells[cell_i][cell_j] = crt_dist
                if cell_j not in dist_among_cells:
                    dist_among_cells[cell_j] = {cell_i: crt_dist}
                else:
                    dist_among_cells[cell_j][cell_i] = crt_dist
        self.dist_among_cells = dist_among_cells

    def get_cell_buffer_indices(self, buffer_dist=None):
        """
        To generate cell_buffer_indices_lookup, which indicates the buffer area for each cell, so that we do not calculate distance
            afterwards
        Args:
            buffer_dist: radius of the buffer area from the center cell. Notice: this distance is meausred in grid coordinate and
                map resolution, not sensor resolution!
        Returns:
        """
        if buffer_dist is None: buffer_dist = self.buffer_dist
        if self.dist_among_cells is None:
            self.get_distance_among_cells()
        cell_buffer_indices_lookup = {}
        for cell in self.cell_indices:
            cell_buffer_indices_lookup[cell] = [
                target_cell for target_cell, dist in self.dist_among_cells[cell].items() if dist <= buffer_dist
            ]
        self.cell_buffer_indices_lookup = cell_buffer_indices_lookup

    def draw_activity_map(self, save_fname='auto', path_name='auto'):
        """
        Plot or save activity map to image file
        Args:
            save_fname: string or None, the name of saved image file, if None, do not save, if auto, use local time as file name
            path_name: string, the folder path of saved image file, if auto, use 'activity_map_log' at the same folder as folder path.
        Returns:

        """
        im = plt.matshow(self.activity_map)
        values = [int(x) for x in np.unique(self.activity_map.ravel())]
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in
                   range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(False)
        plt.axis(False)
        if save_fname is None:
            plt.show()
        else:
            if save_fname == 'auto':
                save_fname = 'activity-map-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            if path_name == 'auto':
                path_name = 'activity_map_log'
            if not os.path.exists(path_name):
                os.mkdir(path_name)
            full_path = os.path.join(path_name, save_fname)
            plt.savefig(full_path,  bbox_inches = 'tight')

    def calc_area_ratio(self, level=2):
        area = (self.activity_map == 2).sum()
        ratio = area / (self.map_resolution[0] * self.map_resolution[1])
        return ratio

def mqtt_processes_for_am(client, usn, pw, address, mqtt_topic_trace_in=None, mqtt_topic_am_in=None, mqtt_qos=1,
                          activity_map=None):
    global ALL_TRACES, AM_FROM_MQTT
    # AM_FROM_MQTT = None
    def on_subscribe(client, userdata, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(client, userdata, msg):
        global AM_FROM_MQTT
        try:
            if msg.topic == mqtt_topic_trace_in:
                # this is a new trace
                this_trace = json.loads(msg.payload)
                if type(this_trace) != dict:
                    print('Error: Invalid payload')
                    return
                if 'to_activity_map' in this_trace and this_trace['to_activity_map'] is True:
                    return
                ALL_TRACES.append(this_trace)
                print('New trace received at {}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
            elif mqtt_topic_am_in is not None and msg.topic.startswith(mqtt_topic_am_in):
                # this is saved activity map
                AM_FROM_MQTT = pickle.loads(msg.payload)
                if activity_map is not None:
                    if 'sensor_resolution' not in AM_FROM_MQTT:
                        print('New saved activity map values received at {}'.format(
                            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        ))
                        activity_map.read_activity_map(read_from='mqtt', data_type='values')
                    else:
                        print('Activity map parameters reeived at {}, reconstruct activity map'.format(
                            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        ))
                        activity_map.read_activity_map(read_from='mqtt', data_type='params')
        except Exception as e:
            print('\n========================================================================')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('Error type: {}, happended in line {} on <mtqq_receiver>'.format(exc_type, exc_tb.tb_lineno))
            print('Error: {}'.format(e))
            print('Error: Invalid payload')
            print('========================================================================\n')

    client.username_pw_set(usn, pw)
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(address, 1883)
    if mqtt_topic_trace_in is not None:
        client.subscribe(mqtt_topic_trace_in, mqtt_qos)
    if mqtt_topic_am_in is not None:
        client.subscribe(mqtt_topic_am_in + '/params', mqtt_qos)
        client.subscribe(mqtt_topic_am_in + '/values', mqtt_qos)
    client.loop_forever()


@run_time_main
def analyze_single_frame(mat, mat_raw, imgName, i,  # data for this frame
                         records,  # history data
                         screen_width, screen_height,  # screen setting to set coords...to be deleted
                         bg_slope_mat, bg_r2_mat,  # detecting settings
                         detect_method='regress', drange=3, mask_idx_dict=None, mask_dist_dict=None,
                         # detecting settings
                         delta_slope_th=-0.1, delta_r2_th=0.1,  # detecting settings
                         diff=False,  # detecting settings
                         round2=False, drange2=1.5, delta_slope_th2=-0.1, delta_r2_th2=0.1,
                         bg_r2_mat2=None,
                         detect_type='hot',
                         delta_mat=None,
                         th_value='auto',
                         sensor_resolution=(8,8), activity_map=None,
                        timestamp=None, deviceId=None):
    """
    mat: normalized data for a single frame
    meanPositionsList: before append
    imgNames: before append
    """
    # use a try-except block to find errors

    global printing_run_time, debug_run_time
    if printing_run_time and debug_run_time: t1 = time.time()

    global ppl_count_global, ppl_count_in_global, ppl_count_out_global
    global ppl_count_global_history, ppl_count_global_in_history, ppl_count_global_out_history
    global last_ppl_count_global, last_ppl_count_global_in, last_ppl_count_global_out
    global config
    global remove_hot_object, hot_obj_temp_th, hot_obj_slope_th, center_slope, border_slope
    global center_r2, border_r2, diameter_th, trackingPara, gap_dict_combine, step_th_remove, extendPara, momentum_eta, counting_line_key, gap_dict_counting
    global max_split_time
    global printAuxFlag
    global data_recording
    global GLOBAL_WINDOW_MATS
    global enable_tracking, use_saved_bg, saved_bg
    global activity_map_wait_time, activity_change_status_threshold, enable_black_list_map
    global RECENT_MATS_RAW_DICT

    ## viz parameters
    cell_size = 30
    # records = copy.deepcopy(records_in)
    if printing_run_time and debug_run_time:
        print(
            'records size: {}; trackingDict size: {}; imgNames size: {}, pair size: {}, out size: {}, in size: {}'.format(
                deep_getsizeof(records, set()),
                deep_getsizeof(records['trackingDict'], set()),
                deep_getsizeof(records['imgNames'], set()),
                deep_getsizeof(records['pairs'], set()),
                deep_getsizeof(records['outFromLastFrame'], set()),
                deep_getsizeof(records['inToNextFrame'], set()),
            ))

    records['imgNames'].append(imgName)

    if printing_run_time and debug_run_time: t2 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: ** {t2 - t1:4.4f} seconds elapsed for "beginning part" in "analyze_single_frame" **')

    # =========================================#
    #              Detecting                  #
    # =========================================#
    if printing_run_time and debug_run_time: t1 = time.time()
    if True:
        # "if true" is used just to make a block to be folded
        if detect_method == 'th':
            if detect_type == 'hot':
                # if th_value == 'auto':
                # th_value = (delta_mat.mean() + delta_mat.std()) * 5 / 4
                boxList = get_peak_by_threshold(delta_mat, th_value, 'hot')

            elif detect_type == 'cold':
                # if th_value == 'auto':
                # th_value = (delta_mat.mean() - delta_mat.std()) * 5 / 4
                boxList = get_peak_by_threshold(delta_mat, th_value, 'cold')
            slope_mat, r2_mat = np.zeros((8, 8)), np.zeros((8, 8))
        elif detect_method == 'regress':
            boxList, slope_mat, r2_mat = get_peak_by_adaptive_local_regress(delta_mat, drange=drange,
                                                                            mask_idx_dict=mask_idx_dict,
                                                                            mask_dist_dict=mask_dist_dict,
                                                                            slope_th=delta_slope_th,
                                                                            detect_type=detect_type, r2_th=delta_r2_th)
        elif detect_method == 'both':
            if detect_type == 'hot':
                boxList_th = get_peak_by_threshold(delta_mat, th_value, 'hot')
                boxList_regress, slope_mat, r2_mat = get_peak_by_adaptive_local_regress(delta_mat, drange=drange,
                                                                                        mask_idx_dict=mask_idx_dict,
                                                                                        mask_dist_dict=mask_dist_dict,
                                                                                        slope_th=delta_slope_th,
                                                                                        detect_type=detect_type,
                                                                                        r2_th=delta_r2_th)

            elif detect_type == 'cold':
                boxList_th = get_peak_by_threshold(delta_mat, th_value, 'hot')
                boxList_regress, slope_mat, r2_mat = get_peak_by_adaptive_local_regress(delta_mat, drange=drange,
                                                                                        mask_idx_dict=mask_idx_dict,
                                                                                        mask_dist_dict=mask_dist_dict,
                                                                                        slope_th=delta_slope_th,
                                                                                        detect_type=detect_type,
                                                                                        r2_th=delta_r2_th)
            boxList = [x for x in boxList_th if x in boxList_regress]
        # x_start = int(screen_width/2 - cell_size*4)
        x_start = int(screen_width / 3 * 0.2)
        y_start = int(screen_height / 2 * 0.2)
        ROIs, meanPositions, polygons = generate_positions(x_start, y_start, cell_size, boxList,
                                                           max_split_time=max_split_time)

        # check time to change activity map status
        if activity_map is not None and activity_map.status == 'only updating':
            this_time = int(time.time())
            if this_time - AM_START_TIME >= activity_map_wait_time and activity_map.calc_area_ratio(level=2) > activity_change_status_threshold:
                activity_map.change_status('only applying')
                print('The status of activity map has changed to "only applying"')
        # # apply activity map => do not filter here, let all positions pass, and do filter for
        # if activity_map is not None and activity_map.status in ['only applying', 'updating and applying']:
        #     detections_filtered_by_activity_map = activity_map.filter_for_new_detections(meanPositions, coords_mode='screen')
        #     # print(detections_filtered_by_activity_map)    #debug_cw
        #     meanPositions = detections_filtered_by_activity_map[2]
        
        # apply black list map
        if activity_map is not None and enable_black_list_map is True:
            detections_filtered_by_black_list_map = activity_map.filter_for_new_detections(meanPositions, coords_mode='screen')
            meanPositions = detections_filtered_by_black_list_map[False]
        records['meanPositionsList'].append(meanPositions)
    else:
        boxList = []
        slope_mat, r2_mat = np.zeros((8, 8)), np.zeros((8, 8))
        ROIs = []
        meanPositions = []
        polygons = []
    if printing_run_time and debug_run_time: t2 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: ** {t2 - t1:4.4f} seconds elapsed for "Detecting" in "analyze_single_frame" **')
        
    # if detect_type == 'hot': print('detections: {}'.format(meanPositions))   #debug_cw

    # =========================================#
    #               Tracking                  #
    # =========================================#
    # print('data: \n{}'.format(mat2))
    # print('slope:\n{}'.format(np.round(slope_mat, 3)))
    # print('backgroup slope: \n{}'.format(np.round(bg_slope_mat, 3)))
    # print('\n')

    # print('\ndebug_cw: check traces info\n' + '=='*30)
    # print(records['meanPositionsList'])
    # for k, v in records['trackingDict'].items():
    #     print(k)
    #     print(v)
    #     print('-'*40)
    if printing_run_time and debug_run_time: t1 = time.time()
    if enable_tracking:
        if printing_run_time and debug_run_time: t3 = time.time()
        # build a lookup dict with key=current, value=start
        if len(records['trackingDict']) > 0:
            currentAsKey = {route['current']: start for start, route in records['trackingDict'].items()}
        if printing_run_time and debug_run_time: t4 = time.time()
        if printing_run_time and debug_run_time: print(
            f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "build a lookup dict" in "Tracking" ===')

        if i == 0:
            pairs, outFromLastFrame = [], []
            meanPositions_c = records['meanPositionsList'][-1]
            inToNextFrame = [idx for idx, meanPosition in enumerate(meanPositions_c)]
        elif i > 0:
            meanPositions_b = records['meanPositionsList'][-2]
            meanPositions_c = records['meanPositionsList'][-1]
            if len(meanPositions_b) > 0 and len(meanPositions_c) > 0:
                if i == 1:
                    meanPositions_a = meanPositions_b
                else:
                    meanPositions_a = []
                    meanPositions_a_origin = records['meanPositionsList'][-3]
                    for id, meanPosition_b in enumerate(meanPositions_b):
                        if id in records['pairToNextFrame']:
                            meanPositions_a.append(
                                meanPositions_a_origin[
                                    records['pairFromLastFrame'][records['pairToNextFrame'].index(id)]])
                        elif id in records['inToNextFrame']:
                            meanPositions_a.append(meanPosition_b)
                        else:
                            print('!!!!!problem')

                vdirections = []
                for idx in range(len(meanPositions_b)):
                    # current_idx = '{}:{}'.format(records['imgNames'][i-1], idx)
                    current_idx = '{}:{}'.format(records['imgNames'][-2], idx)
                    vdirections.append(records['trackingDict'][currentAsKey[current_idx]]['vdirection'])

                if printing_run_time and debug_run_time: t3 = time.time()
                distMatrix, vdirections_new, reflectin_points = reflecting_distance(meanPositions_a, meanPositions_b,
                                                                                    meanPositions_c, vdirections,
                                                                                    eta=momentum_eta)
                if printing_run_time and debug_run_time: t4 = time.time()
                if printing_run_time and debug_run_time: print(
                    f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "reflecting_distance" in "Tracking" ===')
                if printing_run_time and debug_run_time: t3 = time.time()
                P, indicesMatrix = indices_allpeople_with_allpeople_in_two_frames(meanPositions_b, meanPositions_c,
                                                                                  meanPositions_a, printAuxFlag=False,
                                                                                  b_d=trackingPara['b_d'],
                                                                                  b_lr=trackingPara['b_lr'])
                if printing_run_time and debug_run_time: t4 = time.time()
                if printing_run_time and debug_run_time: print(
                    f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "indices_allpeople_with_allpeople_in_two_frames" in "Tracking" ===')

                if printing_run_time and debug_run_time: t3 = time.time()
                P = dist_to_p(distMatrix)
                if printing_run_time and debug_run_time: t4 = time.time()
                if printing_run_time and debug_run_time: print(
                    f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "dist_to_p" in "Tracking" ===')
                if printing_run_time and debug_run_time: t3 = time.time()
                pairs, outFromLastFrame, inToNextFrame, pairFromLastFrame, pairToNextFrame, P_afterRules = choose_and_apply_rules(
                    P, indicesMatrix, uncondintionRules, conditionRules)
                if printing_run_time and debug_run_time: t4 = time.time()
                if printing_run_time and debug_run_time: print(
                    f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "choose_and_apply_rules" in "Tracking" ===')

                if printing_run_time and debug_run_time: t3 = time.time()
                # update momentum (vdirections)
                for idx in range(len(meanPositions_b)):
                    # current_idx = '{}:{}'.format(records['imgNames'][i-1], idx)
                    current_idx = '{}:{}'.format(records['imgNames'][-2], idx)
                    records['trackingDict'][currentAsKey[current_idx]]['vdirection'] = vdirections_new[idx]
                    records['trackingDict'][currentAsKey[current_idx]]['reflecting_point'] = scale(
                        reflectin_points[idx], x_start, y_start, cell_size, sensor_resolution=sensor_resolution)
                if printing_run_time and debug_run_time: t4 = time.time()
                if printing_run_time and debug_run_time: print(
                    f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "update momentum" in "Tracking" ===')
            elif len(meanPositions_b) > 0:
                pairs, pairFromLastFrame, pairToNextFrame = [], [], []
                outFromLastFrame = list(range(len(meanPositions_b)))
                inToNextFrame = []
            elif len(meanPositions_c) > 0:
                pairs, pairFromLastFrame, pairToNextFrame = [], [], []
                outFromLastFrame = []
                inToNextFrame = list(range(len(meanPositions_c)))
            else:
                pairs, pairFromLastFrame, pairToNextFrame = [], [], []
                outFromLastFrame = []
                inToNextFrame = []

            records['pairs'] = pairs
            records['outFromLastFrame'] = outFromLastFrame
            records['inToNextFrame'] = inToNextFrame
            records['pairFromLastFrame'] = pairFromLastFrame
            records['pairToNextFrame'] = pairToNextFrame

            # cw_debug
            # if len(meanPositions_c) > 0:
            #     print('New detections: pairs = {}, out = {}, in = {}'.format(pairs, outFromLastFrame, inToNextFrame))

        if printing_run_time and debug_run_time: t3 = time.time()
        # convert coords to 0-1
        meanPositions_c_01 = [scale(x, x_start, y_start, cell_size, sensor_resolution=sensor_resolution) for x in meanPositions_c]
        meanPositions_c_grid = [coords_conversion(coord_01, '01', cell_size, sensor_resolution)['grid'] for coord_01 in meanPositions_c_01]
        # print('i = ', i, 'type = ', delta_slope_type)
        # print('inToNextFrame: ', inToNextFrame)
        # print('pairs: ', pairs)
        # print('outFromLastFrame: ', outFromLastFrame)
        if printing_run_time and debug_run_time: t4 = time.time()
        if printing_run_time and debug_run_time: print(
            f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "convert coords to 0-1" in "Tracking" ===')
        if printing_run_time and debug_run_time: t3 = time.time()
        if len(inToNextFrame) > 0:
            for idx in inToNextFrame:
                # start_idx = '{}:{}'.format(records['imgNames'][i], idx)
                start_idx = '{}:{}'.format(records['imgNames'][-1], idx)
                newRoute = {'current': start_idx, 'trace': [meanPositions_c_01[idx]], 'trace_grid': [meanPositions_c_grid[idx]],
                            'stop': False, 'vdirection': [0, 0], 'reflecting_point': meanPositions_c_01[idx], 
                            'show': False, 'waiting_for_combine': [True], 'img_names':[records['imgNames'][-1]], 
                            'temp_raw':[mat_raw[meanPositions_c_grid[idx]]], 
                            'delta_temp':[delta_mat[meanPositions_c_grid[idx]]]}
                if activity_map is not None: newRoute['to_activity_map'] = False
                records['trackingDict'][start_idx] = newRoute
        if len(pairs) > 0:
            for pair in pairs:
                # from_idx = '{}:{}'.format(records['imgNames'][i - 1], pair[0])
                from_idx = '{}:{}'.format(records['imgNames'][-2], pair[0])
                # to_idx = '{}:{}'.format(records['imgNames'][i], pair[1])
                to_idx = '{}:{}'.format(records['imgNames'][-1], pair[1])
                start_idx = currentAsKey[from_idx]
                records['trackingDict'][start_idx]['trace'].append(meanPositions_c_01[pair[1]])
                records['trackingDict'][start_idx]['current'] = to_idx
                records['trackingDict'][start_idx]['trace_grid'].append(meanPositions_c_grid[pair[1]])
                records['trackingDict'][start_idx]['img_names'].append(records['imgNames'][-1])
                records['trackingDict'][start_idx]['temp_raw'].append(mat_raw[meanPositions_c_grid[pair[1]]])
                records['trackingDict'][start_idx]['delta_temp'].append(delta_mat[meanPositions_c_grid[pair[1]]])
        if len(outFromLastFrame) > 0:
            for idx in outFromLastFrame:
                # start_idx = currentAsKey['{}:{}'.format(records['imgNames'][i - 1], idx)]
                start_idx = currentAsKey['{}:{}'.format(records['imgNames'][-2], idx)]
                records['trackingDict'][start_idx]['stop'] = True

        traces = [{'t': records['imgNames'].index(key.split(':')[0]) - 1, 'cood': value['trace'],
                    'cood_grid': value['trace_grid'], 'fnsh': value['stop'], 'img_names_first_last': [value['img_names'][0], value['img_names'][-1]], 
                   'reflect': value['reflecting_point'], 'id': key, 'combined_ids': [key], 'show': value['show'], 
                   'waiting_for_combine': value['waiting_for_combine'],
                   'to_activity_map': value.get('to_activity_map', False),
                   'temp_raw': value['temp_raw'], 'delta_temp': value['delta_temp']}
                  for key, value in
                  records['trackingDict'].items()]
                  
        # if detect_type == 'hot':
            # print('\n'+'='*20+'debug_cw1'+ '='*20)
            # for t in traces: print('{}\n'.format(t))
            # print('='*20+'debug_cw1'+ '='*20+'\n')
        # if len(traces) > 0:
        #     print('Num of traces: ', len(traces))           #cw_debug
        if printing_run_time and debug_run_time: t4 = time.time()
        if printing_run_time and debug_run_time: print(
            f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "records updates" in "Tracking" ===')

        # tracking post-processing
        if printing_run_time and debug_run_time: t3 = time.time()
        if track_post_combine:
            post_combine_debug = False
            # if any([t['fnsh'] for t in traces if t['t']!=-1 and len(t['cood'])>3]): post_combine_debug = True
            traces, combine_log = post_combine(traces, uncondintionRules, conditionRules, gap_dict=gap_dict_combine,
                                               coodScale=coordScale_combine, debug=post_combine_debug)
            # after combine, make sure that each trace being combined has the same "waiting_for_combine" with the combined trace
            for trace in traces:
                if trace['fnsh']:
                    for trace_idx in trace['combined_ids']:
                        records['trackingDict'][trace_idx]['waiting_for_combine'] = trace['waiting_for_combine']
        if printing_run_time and debug_run_time: t4 = time.time()
        if printing_run_time and debug_run_time: print(
            f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "track_post_combine" in "Tracking" ===')
        if printing_run_time and debug_run_time: t3 = time.time()
        if tracking_post_remove:
            traces = post_remove(traces, step_th=step_th_remove)
        if printing_run_time and debug_run_time: t4 = time.time()
        if printing_run_time and debug_run_time: print(
            f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "tracking_post_remove" in "Tracking" ===')
        # if tracking_extend_to_border:
        # traces = extent_traces_to_border(traces, lastN=extendPara['lastN'], max_th=extendPara['max_th'], min_th=extendPara['min_th'])
    else:
        traces = []
        combine_log = []
    if printing_run_time and debug_run_time: t2 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: ** {t2 - t1:4.4f} seconds elapsed for "Tracking" in "analyze_single_frame" **')

    if detect_type == 'hot':
        sensor_resolution = delta_mat.shape
        sc1, sc2 = 8 / sensor_resolution[0], 8 / sensor_resolution[1]
        scalee = min(sc1, sc2)
        before_cell_size = cell_size
        cell_size *= scalee
         
        x_start = int(screen_width/3*0.2)
        y_start = int(screen_height/2*0.2)

        saving_bb_list = []

        for polygon_idx in range(len(polygons)):
            polygon = polygons[polygon_idx]
            polygon_x, polygon_y = polygon.exterior.xy
            polygonPoints = [(x, y) for x, y in zip(polygon_x, polygon_y)]
            # need coords conversion because of different viz scale due to sensor_resolution
            polygonPoints_01 = [coords_conversion(coord, 'screen', cell_size=before_cell_size,
                                                  sensor_resolution=sensor_resolution)['01'] for coord in polygonPoints]
            polygonPoints = [coords_conversion(coord, '01', cell_size=cell_size,
                                               sensor_resolution=sensor_resolution)['screen'] for coord in polygonPoints_01]
            
            polygonPointsArray = np.array(polygonPoints, dtype=np.float64)
            maxX, maxY = np.max(polygonPointsArray, axis = 0)
            minX, minY = np.min(polygonPointsArray, axis = 0)
            maxXCoord = (maxX - x_start)/(cell_size*8)
            maxYCoord = (maxY - y_start)/(cell_size*8)
            minXCoord = (minX - x_start)/(cell_size*8)
            minYCoord = (minY - y_start)/(cell_size*8)
            polysize = (maxXCoord - minXCoord)*(maxYCoord - minYCoord)
            if polysize > 0.05:
                saving_bb_list.append([[minXCoord, minYCoord], [maxXCoord, maxYCoord]])
        global bb_save
        saving_dict = {"bounding box": saving_bb_list,"timestamp":timestamp,"ID": deviceId}
        jsObj = json.dumps(saving_dict, cls=NumpyArrayEncoder)
        f = open(bb_save, 'a')
        f.write(jsObj)
        f.write('\n')
        f.close()

    ## return
    return traces, boxList, polygons, records, slope_mat, r2_mat, combine_log

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def detecting(delta_mat, screen_width, screen_height,
              detect_method='regress', drange=3, mask_idx_dict=None, mask_dist_dict=None,
              delta_slope_th=-0.1, delta_r2_th=0.1,
              detect_type='hot',
              th_value='auto',
              timestamp=None, deviceId=None):
    global max_split_time

    cell_size = 30
    if detect_method == 'th':
        if detect_type == 'hot':
            boxList = get_peak_by_threshold(delta_mat, th_value, 'hot')
        elif detect_type == 'cold':
            boxList = get_peak_by_threshold(delta_mat, th_value, 'cold')
        slope_mat, r2_mat = np.zeros((8, 8)), np.zeros((8, 8))
    elif detect_method == 'regress':
        boxList, slope_mat, r2_mat = get_peak_by_adaptive_local_regress(delta_mat, drange=drange,
                                                                        mask_idx_dict=mask_idx_dict,
                                                                        mask_dist_dict=mask_dist_dict,
                                                                        slope_th=delta_slope_th,
                                                                        detect_type=detect_type, r2_th=delta_r2_th)

    elif detect_method == 'both':
        if detect_type == 'hot':
            boxList_th = get_peak_by_threshold(delta_mat, th_value, 'hot')
            boxList_regress, slope_mat, r2_mat = get_peak_by_adaptive_local_regress(delta_mat, drange=drange,
                                                                                    mask_idx_dict=mask_idx_dict,
                                                                                    mask_dist_dict=mask_dist_dict,
                                                                                    slope_th=delta_slope_th,
                                                                                    detect_type=detect_type,
                                                                                    r2_th=delta_r2_th)

        elif detect_type == 'cold':
            boxList_th = get_peak_by_threshold(delta_mat, th_value, 'hot')
            boxList_regress, slope_mat, r2_mat = get_peak_by_adaptive_local_regress(delta_mat, drange=drange,
                                                                                    mask_idx_dict=mask_idx_dict,
                                                                                    mask_dist_dict=mask_dist_dict,
                                                                                    slope_th=delta_slope_th,
                                                                                    detect_type=detect_type,
                                                                                    r2_th=delta_r2_th)
        boxList = [x for x in boxList_th if x in boxList_regress]

    # x_start = int(screen_width/2 - cell_size*4)
    x_start = int(screen_width / 3 * 0.2)
    y_start = int(screen_height / 2 * 0.2)
    ROIs, meanPositions, polygons = generate_positions(x_start, y_start, cell_size, boxList,
                                                       max_split_time=max_split_time)
    detecting_rst = {'boxList': boxList, 'polygons': polygons,
                     'meanPositionsList': meanPositions, 'slope_mat': slope_mat, 'r2_mat': r2_mat}
   
    return detecting_rst

def tracking(records, i, screen_width, screen_height, sensor_resolution=(8,8)):
    global printing_run_time, debug_run_time

    cell_size = 30
    x_start = int(screen_width / 3 * 0.2)
    y_start = int(screen_height / 2 * 0.2)
    if printing_run_time and debug_run_time: t3 = time.time()
    # build a lookup dict with key=current, value=start
    if len(records['trackingDict']) > 0:
        currentAsKey = {route['current']: start for start, route in records['trackingDict'].items()}
    if printing_run_time and debug_run_time: t4 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "build a lookup dict" in "Tracking" ===')

    if i == 0:
        pairs, outFromLastFrame = [], []
        meanPositions_c = records['meanPositionsList'][-1]
        inToNextFrame = [idx for idx, meanPosition in enumerate(meanPositions_c)]
    elif i > 0:
        meanPositions_b = records['meanPositionsList'][-2]
        meanPositions_c = records['meanPositionsList'][-1]
        if len(meanPositions_b) > 0 and len(meanPositions_c) > 0:
            if i == 1:
                meanPositions_a = meanPositions_b
            else:
                meanPositions_a = []
                meanPositions_a_origin = records['meanPositionsList'][-3]
                for id, meanPosition_b in enumerate(meanPositions_b):
                    if id in records['pairToNextFrame']:
                        meanPositions_a.append(
                            meanPositions_a_origin[records['pairFromLastFrame'][records['pairToNextFrame'].index(id)]])
                    elif id in records['inToNextFrame']:
                        meanPositions_a.append(meanPosition_b)
                    else:
                        print('!!!!!problem')

            vdirections = []
            for idx in range(len(meanPositions_b)):
                current_idx = '{}:{}'.format(records['imgNames'][-2], idx)
                vdirections.append(records['trackingDict'][currentAsKey[current_idx]]['vdirection'])

            if printing_run_time and debug_run_time: t3 = time.time()
            distMatrix, vdirections_new, reflectin_points = reflecting_distance(meanPositions_a, meanPositions_b,
                                                                                meanPositions_c, vdirections,
                                                                                eta=momentum_eta)
            if printing_run_time and debug_run_time: t4 = time.time()
            if printing_run_time and debug_run_time: print(
                f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "reflecting_distance" in "Tracking" ===')
            if printing_run_time and debug_run_time: t3 = time.time()
            P, indicesMatrix = indices_allpeople_with_allpeople_in_two_frames(meanPositions_b, meanPositions_c,
                                                                              meanPositions_a, printAuxFlag=False,
                                                                              b_d=trackingPara['b_d'],
                                                                              b_lr=trackingPara['b_lr'])
            if printing_run_time and debug_run_time: t4 = time.time()
            if printing_run_time and debug_run_time: print(
                f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "indices_allpeople_with_allpeople_in_two_frames" in "Tracking" ===')

            if printing_run_time and debug_run_time: t3 = time.time()
            P = dist_to_p(distMatrix)
            if printing_run_time and debug_run_time: t4 = time.time()
            if printing_run_time and debug_run_time: print(
                f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "dist_to_p" in "Tracking" ===')
            if printing_run_time and debug_run_time: t3 = time.time()
            pairs, outFromLastFrame, inToNextFrame, pairFromLastFrame, pairToNextFrame, P_afterRules = choose_and_apply_rules(
                P, indicesMatrix, uncondintionRules, conditionRules)
            if printing_run_time and debug_run_time: t4 = time.time()
            if printing_run_time and debug_run_time: print(
                f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "choose_and_apply_rules" in "Tracking" ===')

            if printing_run_time and debug_run_time: t3 = time.time()
            # update momentum (vdirections)
            for idx in range(len(meanPositions_b)):
                # current_idx = '{}:{}'.format(records['imgNames'][i-1], idx)
                current_idx = '{}:{}'.format(records['imgNames'][-2], idx)
                records['trackingDict'][currentAsKey[current_idx]]['vdirection'] = vdirections_new[idx]
                records['trackingDict'][currentAsKey[current_idx]]['reflecting_point'] = scale(reflectin_points[idx],
                                                                                               x_start, y_start,
                                                                                               cell_size)
            if printing_run_time and debug_run_time: t4 = time.time()
            if printing_run_time and debug_run_time: print(
                f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "update momentum" in "Tracking" ===')
        elif len(meanPositions_b) > 0:
            pairs, pairFromLastFrame, pairToNextFrame = [], [], []
            outFromLastFrame = list(range(len(meanPositions_b)))
            inToNextFrame = []
        elif len(meanPositions_c) > 0:
            pairs, pairFromLastFrame, pairToNextFrame = [], [], []
            outFromLastFrame = []
            inToNextFrame = list(range(len(meanPositions_c)))
        else:
            pairs, pairFromLastFrame, pairToNextFrame = [], [], []
            outFromLastFrame = []
            inToNextFrame = []

        records['pairs'] = pairs
        records['outFromLastFrame'] = outFromLastFrame
        records['inToNextFrame'] = inToNextFrame
        records['pairFromLastFrame'] = pairFromLastFrame
        records['pairToNextFrame'] = pairToNextFrame
    if printing_run_time and debug_run_time: t3 = time.time()
    # convert coords to 0-1
    meanPositions_c_01 = [scale(x, x_start, y_start, cell_size, sensor_resolution) for x in meanPositions_c]
    if printing_run_time and debug_run_time: t4 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "convert coords to 0-1" in "Tracking" ===')
    if printing_run_time and debug_run_time: t3 = time.time()
    if len(inToNextFrame) > 0:
        for idx in inToNextFrame:
            # start_idx = '{}:{}'.format(records['imgNames'][i], idx)
            start_idx = '{}:{}'.format(records['imgNames'][-1], idx)
            newRoute = {'start_i': i, 'current': start_idx, 'trace': [meanPositions_c_01[idx]], 'stop': False,
                        'vdirection': [0, 0], 'reflecting_point': meanPositions_c_01[idx], 
                        'show': False, 'waiting_for_combine': [True]}
            records['trackingDict'][start_idx] = newRoute
    if len(pairs) > 0:
        for pair in pairs:
            # from_idx = '{}:{}'.format(records['imgNames'][i - 1], pair[0])
            from_idx = '{}:{}'.format(records['imgNames'][-2], pair[0])
            # to_idx = '{}:{}'.format(records['imgNames'][i], pair[1])
            to_idx = '{}:{}'.format(records['imgNames'][-1], pair[1])
            start_idx = currentAsKey[from_idx]
            records['trackingDict'][start_idx]['trace'].append(meanPositions_c_01[pair[1]])
            records['trackingDict'][start_idx]['current'] = to_idx
    if len(outFromLastFrame) > 0:
        for idx in outFromLastFrame:
            # start_idx = currentAsKey['{}:{}'.format(records['imgNames'][i - 1], idx)]
            start_idx = currentAsKey['{}:{}'.format(records['imgNames'][-2], idx)]
            records['trackingDict'][start_idx]['stop'] = True

    traces = [{'t': value['start_i'], 'cood': value['trace'],
               'fnsh': value['stop'], 'reflect': value['reflecting_point'], 'id': key, 'combined_ids': [key],
               'show': value['show'], 'waiting_for_combine': value['waiting_for_combine']}
              for key, value in
              records['trackingDict'].items()]
    if printing_run_time and debug_run_time: t4 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "records updates" in "Tracking" ===')

    # tracking post-processing
    if printing_run_time and debug_run_time: t3 = time.time()
    if track_post_combine:
        traces, combine_log = post_combine(traces, uncondintionRules, conditionRules, gap_dict=gap_dict_combine,
                                           coodScale=coordScale_combine)
    if printing_run_time and debug_run_time: t4 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "track_post_combine" in "Tracking" ===')
    if printing_run_time and debug_run_time: t3 = time.time()
    if tracking_post_remove:
        traces = post_remove(traces, step_th=step_th_remove)
    if printing_run_time and debug_run_time: t4 = time.time()
    if printing_run_time and debug_run_time: print(
        f'DEBUG: === {t4 - t3:4.4f} seconds elapsed for "tracking_post_remove" in "Tracking" ===')
    return traces, records, combine_log

def update_sensitivity(sensitivity, sensitivity_bounds):
    cr2_l = sensitivity_bounds[0]
    cr2_h = sensitivity_bounds[1]
    br2_l = sensitivity_bounds[2]
    br2_h = sensitivity_bounds[3]
    csh_l = sensitivity_bounds[4]
    csh_h = sensitivity_bounds[5]
    bsh_l = sensitivity_bounds[6]
    bsh_h = sensitivity_bounds[7]
    # scale parameters
    center_r2 = cr2_l - (cr2_l - cr2_h) * sensitivity
    border_r2 = br2_l - (br2_l - br2_h) * sensitivity
    center_slope_hot = csh_l - (csh_l - csh_h) * sensitivity
    border_slope_hot = bsh_l - (bsh_l - bsh_h) * sensitivity
    # center_border_mapping
    delta_r2_th = center_border_mapping(center_r2, border_r2)
    delta_slope_th = center_border_mapping(center_slope_hot, border_slope_hot)
    return(delta_r2_th, delta_slope_th)


##############
def pygame_run_mqtt(min_interval=0.01,
                    detect_method_hot='regress', detect_method_cold='th',
                    step=False, bg_frame_num=10, drange=3,
                    delta_slope_th_hot=0.1, delta_slope_th_cold=0.1,
                    delta_r2_th=0.1, th_value_hot='auto', th_value_cold='auto',
                    th_auto_std_scale_hot=8, th_auto_std_scale_cold=8,
                    diff=False, bg_r2_mat=None, tracking_separate=True,
                    round2=False, drange2=1.5, delta_slope_th2=-0.1, delta_r2_th2=0.1,
                    bg_r2_mat2=None, mem_clr_num_frames=300,
                    counting_first_only=False,
                    fps=0, bg=pygame.Color('black'), activity_map=None,
                    sensor_resolution=(8,8), edge_resolution=(2,2),
                    output_wait_for_post_combine_steps=3):
    """
    Visualization for a mqtt stream from data of a sensor and a corresponding ground truth video using a 1-subplot animation.
    This is the principle function to run in __main__
    PyGame runtime.
    Key press: ESC to exit, F to switch fullscreen.
    This is the functions to run in __main__ after pygame_init

    Arguments:
    ----------------------------------------------
    ...

    Returns:
    -----------------------------------------------
    None
    """

    global ppl_count_global, ppl_count_in_global, ppl_count_out_global
    global display_trace, display_count
    global auto_disp_trace, th_num_clear_frame
    global config
    global remove_hot_object, hot_obj_temp_th, hot_obj_slope_th, center_slope, border_slope
    global center_r2, border_r2, diameter_th, trackingPara, gap_dict_combine, step_th_remove, extendPara, momentum_eta, counting_line_key, gap_dict_counting
    global max_split_time
    global printAuxFlag
    global data_recording
    global viz

    global minV, maxV
    global mat, mat_raw
    global flag_mat_updated_mm, flag_mat_updated_bg
    global flag_detection_mm, flag_detection_bg
    global flag_long_trace_mm, flag_long_trace_bg
    global flag_valid_trace_mm, flag_valid_trace_bg

    global bg_mat, bg_mat_raw, bg_slope_mat

    global client_global, client_for_am
    client_global = None

    global mqtt_id, publish, publish_for_activity_map, normalize, long_trace_num, valid_trace_num, norm_scale, data_queue, delay_prevention
    global mqtt_topic_in, mqtt_topic_out, mqtt_topic_trace, mqtt_topic_activity_map
    global mqtt_address, mqtt_port, mqtt_qos, mqtt_keepalive, mqtt_topic_status
    global data_start_time_epoch, data_start_time_str, data_end_time_epoch, data_end_time_str, playback_slider_value, read_restart, is_paused
    global printing_run_time, debug_run_time, is_exiting, batch_eval, play_speed
    global window_num_frame, frame_interval, init_minV, init_maxV
    global version
    global waking_filtering, play_speed_lower_bound, play_speed_upper_bound
    global flag_bg_mat_raw_defined
    global detection_write_file, detection_write_path, detection_to_mqtt, mqtt_topic_detection, detection_send_empty_list
    global world_coord, spatial_config_path
    global msg_convert, detection_fp_filter, world_coord_rotation, world_coord_flip_x, world_coord_flip_y
    global enable_tracking
    global flag_wake, waking_frame_num, waking_trigger, waking_trigger_sec, horizontal_weights
    global bg_initiation, bg_ini_file_path
    global use_saved_bg, saved_bg
    global save_trace_file_path, saved_activity_map_file_path, AM_FROM_MQTT, AM_START_TIME, activity_map_wait_time, activity_change_status_threshold
    global SAVE_DATA_OVER
    global RECENT_MATS_RAW_DICT, CHECK_FUTON_METHOD, ENABLE_FUTON_EFFECT_DETECTION
    global sensitivity, sensitivity_bounds

    SAVE_DATA_OVER = False
    # hack
    global disable_lower_count, disable_upper_count, disable_in_count, disable_out_count
    if viz:
        ## load pg dict
        screen = pg['screen']
        screen_width = pg['screen_width']
        screen_height = pg['screen_height']
        clock = pg['clock']
        fonts = pg['fonts']
        is_fullscreen = pg['is_fullscreen']
        manager = pg['manager']
        time_delta = pg['time_delta']
    else:
        screen_width = 1280
        screen_height = 720
        time_delta = 0.001

    ## viz parameters
    cell_size = 30

    ## pygame start()
    mats = []
    imgNames = []
    epochs = []
    serial_json_strs_recording = []
    epochs_msec_recording = []
    epoch = 0
    flag_long_trace = False
    trajectories_prev = deque([])
    list_delta_time_sec = []
    i_waking = 0
    paused_slider_value = 0
    
    if sensitivity != -1:
        delta_r2_th, delta_slope_th_hot = update_sensitivity(sensitivity, sensitivity_bounds)

    ## get mask_dict
    mask_idx_dict, mask_dist_dict = get_drange_mask(drange=drange, mat_shape=sensor_resolution, 
        include_self=True, edge_resolution=edge_resolution)

    ## load word coord spatial config json
    if world_coord:
        with open(spatial_config_path, 'r') as f:
            spatial_config_data = json.load(f)

    if viz:
        ## pygame gui
        # debug buttons
        clear_memory_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, screen_height - 30), (20, 20)),
                                                           text='M', manager=manager)
        reset_count_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((35, screen_height - 30), (20, 20)),
                                                          text='R', manager=manager)
        display_trace_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((85, screen_height - 30), (20, 20)),
                                                            text='T', manager=manager)
        display_count_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((110, screen_height - 30), (20, 20)),
                                                            text='C', manager=manager)
        remove_hot_object_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((135, screen_height - 30), (20, 20)), text='H', manager=manager)
        auto_disp_trace_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((160, screen_height - 30), (20, 20)), text='A', manager=manager)

        # serial only buttons
        reset_background_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((60, screen_height - 30), (20, 20)), text='B', manager=manager)

        # data recording buttons
        start_data_recording_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, screen_height - 90), (200, 20)), text='Start Data Recording', manager=manager)
        finish_data_recording_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, screen_height - 65), (200, 20)), text='Finish Data Recording', manager=manager)

        # multi-modes buttons
        standard_mode_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, screen_height - 175), (200, 20)),
                                                            text='Standard Mode', manager=manager)
        object_detection_mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, screen_height - 150), (200, 20)), text='Object Detection Mode', manager=manager)
        high_density_mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, screen_height - 125), (200, 20)), text='High Density Mode', manager=manager)

        # data playback slider
        if run_mode == 'saved_data':
            data_playback_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((485, 11.5), (685, 15)),
                                                                          start_value=0., value_range=(0., 1.),
                                                                          manager=manager)
            play_speed_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((870, 33), (300, 15)),
                                                                       start_value=play_speed, value_range=(
                play_speed_lower_bound, play_speed_upper_bound), manager=manager)

        # mqtt device name dropdown menu
        device_name_menu = pygame_gui.elements.UIDropDownMenu(
            options_list=['mqtt_id', '0242E33552C8', '0242DC39E5EB', '024225C4F614', '024251BC8518', '0242D5E131AB',
                          '024293C09661'], starting_option='mqtt_id', relative_rect=pygame.Rect((400, 27.5), (150, 17.5)),
            manager=manager)

    ## pygame update
    init = True
    i = 0
    i_clear = 0
    i_bg = 0
    i_wake = 0
    running = True
    cnt_skip = 0
    epoch_msec_prev = 0
    coords_prev = []
    count_frame = 0
    mean_bg_value = []
    meanPositionsList_hot = []
    meanPositionsList_cold = []
    if tracking_separate:
        trackingDict_hot = {}
        trackingDict_cold = {}
        records_hot = {'imgNames': [], 'trackingDict': trackingDict_hot, 'meanPositionsList': meanPositionsList_hot}
        records_cold = {'imgNames': [], 'trackingDict': trackingDict_cold, 'meanPositionsList': meanPositionsList_cold}
    else:
        trackingDict = {}
        records = {'imgNames': [], 'trackingDict': trackingDict, 'meanPositionsList': []}
    delta_value_mat_list = []  # for auto detecting threshold
    RECENT_MATS_RAW_DICT = {}

    # adaptive min max init
    if adaptive_min_max:
        window_mats = deque([])
        window_flags = deque([])
        print('window_mats and window_flags for adaptive min max initiated')
        minV, maxV = init_minV, init_maxV

    # adaptive bg init
    if adaptive_bg != 0:
        window_flags_abg = deque([])
        print(''
              ''
              ' for adaptive bg initiated')

    flag_bg_mat_raw_defined = False
    bg_mats_raw = []

    # if use saved bg from txt file as 1st bg
    if bg_initiation == 1:
        with open(bg_ini_file_path, 'r') as file:
            line = file.readline().strip()
            while line:
                if line != '\n' and line != '':
                    # get rid of receiving time
                    if line[:1] == '1':
                        line = line[14:]
                    # convert to standard data format
                    if msg_convert == 'seamless':
                        data: dict = eval(line.replace('\n', ''))
                        fields: dict | int = data.get('fields', 0)
                        if fields != 0:
                            data_inside = fields.get('data', 0)
                            if data_inside != 0:
                                readings = data_inside[6:70]
                                data_json = {'data': readings,
                                           'deviceName': fields['macAddress'],
                                           'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                                           'timestamp': data['timestamp'], 
                                           'utcSecs': fields['utcSecs'], 
                                           'utcUsecs': fields['utcUsecs']}
                    if data_json['deviceName'] == mqtt_id:
                        bg_mats_raw.append(true_data_to_mat(data_json, shape=sensor_resolution))
                line = file.readline()

    while running:
        if viz:
            # handle pygame quit
            for event in pygame.event.get():
                # QUIT
                if event.type == pygame.QUIT:
                    running = False
                # ESC: quit this scene
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                # F: toggle fullscreen
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    if is_fullscreen:
                        screen = pygame.display.set_mode((screen_width, screen_height))
                        is_fullscreen = False
                    else:
                        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                        is_fullscreen = True
                # T: toggle display_trace
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    display_trace = not display_trace
                # C: toggle display_count
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    display_count = not display_count
                # 1-0: reset counting line
                if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                    reset_counting_line('v_l')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                    reset_counting_line('v_ml')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_3:
                    reset_counting_line('v_m')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_4:
                    reset_counting_line('v_mr')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_5:
                    reset_counting_line('v_r')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_6:
                    reset_counting_line('h_t')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_7:
                    reset_counting_line('h_mt')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_8:
                    reset_counting_line('h_m')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_9:
                    reset_counting_line('h_mb')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_0:
                    reset_counting_line('h_b')
                # M: force clear memory
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    mats = []
                    epochs = []
                    meanPositionsList = [[], [], []]
                    trackingDict = {}

                    # ============ a new way to clear memory============#
                    records_hot['meanPositionsList'] = records_hot['meanPositionsList'][-2:]
                    records_cold['meanPositionsList'] = records_cold['meanPositionsList'][-2:]
                    currentNames_hot, currentNames_cold = [], []
                    for idx, mp in enumerate(records_hot['meanPositionsList'][-1]):
                        currentNames_hot.append('{}:{}'.format(records_hot['imgNames'][-1], idx))
                    for idx, mp in enumerate(records_cold['meanPositionsList'][-1]):
                        currentNames_cold.append('{}:{}'.format(records_cold['imgNames'][-1], idx))
                    # get inital names (keys of trackingDict) of traces related to latest meanPositions
                    currentAsKey_hot = {route['current']: start for start, route in records_hot['trackingDict'].items()}
                    currentAsKey_cold = {route['current']: start for start, route in records_cold['trackingDict'].items()}
                    initalNames_hot = [currentAsKey_hot[x] for x in currentNames_hot]
                    initalNames_cold = [currentAsKey_cold[x] for x in currentNames_cold]
                    # check again if there is any trace to be combined
                    combined = []
                    for tt in initalNames_hot + initalNames_cold:
                        combined += [pair for pair in combine_log if tt in pair]
                    combined = [x for pair in combined for x in pair]
                    keep_ids = list(set(initalNames_hot + initalNames_cold + combined))

                    # only keep traces with intial names in list above
                    records_hot['trackingDict'] = {key: value for key, value in records_hot['trackingDict'].items() if
                                                   key in keep_ids}
                    records_cold['trackingDict'] = {key: value for key, value in records_cold['trackingDict'].items() if
                                                    key in keep_ids}

                    # for long trace
                    compress_long_traces(records_hot['trackingDict'])
                    compress_long_traces(records_cold['trackingDict'])

                    # imgNames: find the earliest imgName for the routes still existing, delete all imgNames earlier than that
                    all_imgNames = records_hot['imgNames']
                    keep_imgNames_idx = [all_imgNames.index(id.split(':')[0]) for id in keep_ids]
                    if len(keep_imgNames_idx) > 0:
                        ealiest_imgName_idx = min(keep_imgNames_idx)
                    else:
                        ealiest_imgName_idx = len(all_imgNames) - 2
                    if len(all_imgNames) - ealiest_imgName_idx < 2:
                        ealiest_imgName_idx = len(all_imgNames) - 2  # at least keep last 2 imgNames
                    records_hot['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                    records_cold['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                    # ============ a new way to clear memory============#

                    # traces_to_show = []
                    ppl_count_global += ppl_count
                    ppl_count_in_global += ppl_count_in
                    ppl_count_out_global += ppl_count_out
                    i_clear = 0

                    print('memory cleared by pressing "M" key')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s and step is False:
                    step = True
                # R: reset ppl counting
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    mats = []
                    epochs = []
                    meanPositionsList = [[], [], []]
                    trackingDict = {}
                    # traces_to_show = []

                    # ============ a new way to clear memory============#
                    records_hot['meanPositionsList'] = records_hot['meanPositionsList'][-2:]
                    records_cold['meanPositionsList'] = records_cold['meanPositionsList'][-2:]
                    currentNames_hot, currentNames_cold = [], []
                    for idx, mp in enumerate(records_hot['meanPositionsList'][-1]):
                        currentNames_hot.append('{}:{}'.format(records_hot['imgNames'][-1], idx))
                    for idx, mp in enumerate(records_cold['meanPositionsList'][-1]):
                        currentNames_cold.append('{}:{}'.format(records_cold['imgNames'][-1], idx))
                    # get inital names (keys of trackingDict) of traces related to latest meanPositions
                    currentAsKey_hot = {route['current']: start for start, route in records_hot['trackingDict'].items()}
                    currentAsKey_cold = {route['current']: start for start, route in records_cold['trackingDict'].items()}
                    initalNames_hot = [currentAsKey_hot[x] for x in currentNames_hot]
                    initalNames_cold = [currentAsKey_cold[x] for x in currentNames_cold]
                    # check again if there is any trace to be combined
                    combined = []
                    for tt in initalNames_hot + initalNames_cold:
                        combined += [pair for pair in combine_log if tt in pair]
                    combined = [x for pair in combined for x in pair]
                    keep_ids = list(set(initalNames_hot + initalNames_cold + combined))

                    # only keep traces with intial names in list above
                    records_hot['trackingDict'] = {key: value for key, value in records_hot['trackingDict'].items() if
                                                   key in keep_ids}
                    records_cold['trackingDict'] = {key: value for key, value in records_cold['trackingDict'].items() if
                                                    key in keep_ids}

                    # for long trace
                    compress_long_traces(records_hot['trackingDict'])
                    compress_long_traces(records_cold['trackingDict'])

                    # imgNames: find the earliest imgName for the routes still existing, delete all imgNames earlier than that
                    all_imgNames = records_hot['imgNames']
                    keep_imgNames_idx = [all_imgNames.index(id.split(':')[0]) for id in keep_ids]
                    if len(keep_imgNames_idx) > 0:
                        ealiest_imgName_idx = min(keep_imgNames_idx)
                    else:
                        ealiest_imgName_idx = len(all_imgNames) - 2
                    if len(all_imgNames) - ealiest_imgName_idx < 2:
                        ealiest_imgName_idx = len(all_imgNames) - 2  # at least keep last 2 imgNames
                    records_hot['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                    records_cold['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                    # ============ a new way to clear memory============#

                    ppl_count = 0
                    ppl_count_in = 0
                    ppl_count_out = 0
                    ppl_count_global = 0
                    ppl_count_in_global = 0
                    ppl_count_out_global = 0

                    print('counting reset and memory cleared by pressing "R" key')
                # B: reset background recording
                if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    # i_bg = 0  # now handled by bg thred
                    # trigger bg reset and clear memory
                    flag_long_trace = flag_long_trace_bg = flag_long_trace_mm = True
                # H: toggle remove_hot_object
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    remove_hot_object = not remove_hot_object
                    print(f'remove_hot_object = {remove_hot_object}')
                # # A: toggle alternative algorithm
                # if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                #     auto_disp_trace = not auto_disp_trace
                # SPACE: toggle data play/pause
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                    if is_paused:
                        paused_slider_value = data_playback_slider.get_current_value()
                # LEFT: play backward 20 seconds
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    playback_slider_value = data_playback_slider.get_current_value()
                    slider_step_20sec = 1 / (data_end_time_epoch - data_start_time_epoch) * 20
                    playback_slider_value -= slider_step_20sec
                    if is_paused:
                        paused_slider_value = playback_slider_value
                    read_restart = True
                    flag_long_trace_mm = flag_long_trace_bg = flag_long_trace = True
                # RIGHT: play forward 20 seconds
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    playback_slider_value = data_playback_slider.get_current_value()
                    slider_step_20sec = 1 / (data_end_time_epoch - data_start_time_epoch) * 20
                    playback_slider_value += slider_step_20sec
                    if is_paused:
                        paused_slider_value = playback_slider_value
                    read_restart = True
                    flag_long_trace_mm = flag_long_trace_bg = flag_long_trace = True
                ## pygame gui
                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        # T: toggle display_trace
                        if event.ui_element == display_trace_button:
                            display_trace = not display_trace
                        # C: toggle display_count
                        elif event.ui_element == display_count_button:
                            display_count = not display_count
                        # M: force clear memory
                        if event.ui_element == clear_memory_button:
                            mats = []
                            epochs = []
                            meanPositionsList = [[], [], []]
                            trackingDict = {}
                            ppl_count_global += ppl_count
                            ppl_count_in_global += ppl_count_in
                            ppl_count_out_global += ppl_count_out
                            i_clear = 0
                        # R: reset ppl counting
                        elif event.ui_element == reset_count_button:
                            mats = []
                            epochs = []
                            meanPositionsList = [[], [], []]
                            trackingDict = {}
                            i_clear = 0
                            ppl_count = 0
                            ppl_count_in = 0
                            ppl_count_out = 0
                            ppl_count_global = 0
                            ppl_count_in_global = 0
                            ppl_count_out_global = 0
                        # H: toggle remove_hot_object
                        elif event.ui_element == remove_hot_object_button:
                            remove_hot_object = not remove_hot_object
                            print(f'remove_hot_object = {remove_hot_object}')
                        # B: reset background recording
                        elif event.ui_element == reset_background_button:
                            # i_bg = 0  # now handled by bg thred
                            # trigger bg reset and clear memory
                            flag_long_trace = flag_long_trace_bg = flag_long_trace_mm = True
                        # A: toggle alternative algorithm
                        elif event.ui_element == auto_disp_trace_button:
                            auto_disp_trace = not auto_disp_trace
                        # Start Data Recording
                        elif event.ui_element == start_data_recording_button:
                            print(f'Start Data Recording')
                            data_recording = True
                        # Finish Data Recording
                        elif event.ui_element == finish_data_recording_button:
                            if data_recording:
                                print(f'Finish Data Recording')
                                data_recording = False
                                recording_file_name = f'sensor_data_{epochs_msec_recording[0]}-{epochs_msec_recording[-1]}'
                                save_data_to_file(serial_json_strs_recording, epochs_msec_recording, recording_file_name)
                                serial_json_strs_recording = []
                                epochs_msec_recording = []
                            else:
                                print(f'Data recording hasn\'t started yet')
                        # Standard Mode
                        elif event.ui_element == standard_mode_button:
                            print(f'Standard Mode')
                            diameter_th = config['diameter_th']
                            diameter_th = {int(key): value for key, value in diameter_th.items()}
                            max_split_time = config['max_split_time']
                            center_slope = config['center_slope']
                            border_slope = config['border_slope']
                            delta_slope_th = center_border_mapping(center_slope, border_slope)
                            center_r2 = config['center_r2']
                            border_r2 = config['border_r2']
                            bg_r2_mat = center_border_mapping(center_r2, border_r2)
                            gap_dict_combine = config['gap_dict_combine']
                            gap_dict_combine = {int(key): value for key, value in gap_dict_combine.items()}
                            step_th_remove = config['step_th_remove']
                            extendPara = config['extendPara']
                        # Object Detection Mode
                        elif event.ui_element == object_detection_mode_button:
                            print(f'Object Detection Mode')
                            diameter_th = config['diameter_th_hot_obj']
                            diameter_th = {int(key): value for key, value in diameter_th.items()}
                            max_split_time = config['max_split_time_hot_obj']
                            center_slope = config['center_slope']
                            border_slope = config['border_slope']
                            delta_slope_th = center_border_mapping(center_slope, border_slope)
                            center_r2 = config['center_r2']
                            border_r2 = config['border_r2']
                            bg_r2_mat = center_border_mapping(center_r2, border_r2)
                            gap_dict_combine = config['gap_dict_combine']
                            gap_dict_combine = {int(key): value for key, value in gap_dict_combine.items()}
                            step_th_remove = config['step_th_remove']
                            extendPara = config['extendPara']
                        # High Density Mode
                        elif event.ui_element == high_density_mode_button:
                            print(f'High Density Mode')
                            center_slope = config['center_slope_crossing']
                            border_slope = config['border_slope_crossing']
                            delta_slope_th = center_border_mapping(center_slope, border_slope)
                            center_r2 = config['center_r2_crossing']
                            border_r2 = config['border_r2_crossing']
                            bg_r2_mat = center_border_mapping(center_r2, border_r2)
                            diameter_th = config['diameter_th_crossing']
                            diameter_th = {int(key): value for key, value in diameter_th.items()}
                            max_split_time = config['max_split_time_crossing']
                            gap_dict_combine = config['gap_dict_combine_crossing']
                            gap_dict_combine = {int(key): value for key, value in gap_dict_combine.items()}
                            step_th_remove = config['step_th_remove_crossing']
                            extendPara = config['extendPara_crossing']
                    if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        if event.ui_element == data_playback_slider:
                            print('data playback slider value:', event.value)
                            playback_slider_value = event.value
                            read_restart = True
                            flag_long_trace_mm = flag_long_trace_bg = flag_long_trace = True
                        if event.ui_element == play_speed_slider:
                            # print('play speed slider value:', event.value)
                            play_speed = event.value
                    if event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                        if event.ui_element == device_name_menu:
                            print("Selected option for device_name_menu:", event.text)
                            mqtt_id = event.text

                # pygame gui process events
                manager.process_events(event)

            # SPACE: toggle data play/pause
            if is_paused:
                playback_slider_value = paused_slider_value
                read_restart = True
                flag_long_trace_mm = flag_long_trace_bg = flag_long_trace = True

        # tmp_t1 = time.time()
        # read data_queue
        if SAVE_DATA_OVER == True and len(data_queue) == 0:
            sys.exit()

        # pause while data_queue is empty
        while len(data_queue) == 0:
            # sleep to prevent lag
            time.sleep(0.0001)
            # time.sleep(0.05)
        # tmp_t2 = time.time()
        # tmp_delta_time = tmp_t2 - tmp_t1
        # print(f'DEBUG: time waiting for new data_queue item: {tmp_delta_time:4.4f} seconds')

        # when queue is not empty
        # not skip delayed frames at all
        if delay_prevention == 0:
            data = data_queue.popleft()
        # skip delayed frames adaptively
        elif delay_prevention == 1:
            if len(data_queue) < 10:
                data = data_queue.popleft()
            elif len(data_queue) < 20:
                cnt_skip += 1
                print(f'data_queue longer than 10. 1 frame skipped. i = {i}; cnt_skip = {cnt_skip}')
                data = data_queue.popleft()
                data = data_queue.popleft()
                # clear memory
                flag_long_trace = True
                # # reset background recording and clear memory
                # flag_long_trace_bg = flag_long_trace_mm = True
            elif len(data_queue) < 50:
                cnt_skip += 2
                print(f'data_queue longer than 20. 2 frame skipped. i = {i}; cnt_skip = {cnt_skip}')
                data = data_queue.popleft()
                data = data_queue.popleft()
                data = data_queue.popleft()
                # clear memory
                flag_long_trace = True
                # reset background recording and clear memory
                flag_long_trace_bg = flag_long_trace_mm = True
            elif len(data_queue) < 150:
                cnt_skip += 3
                print(f'data_queue longer than 50. 3 frame skipped. i = {i}; cnt_skip = {cnt_skip}')
                data = data_queue.popleft()
                data = data_queue.popleft()
                data = data_queue.popleft()
                data = data_queue.popleft()
                # clear memory
                flag_long_trace = True
                # reset background recording and clear memory
                flag_long_trace_bg = flag_long_trace_mm = True
            elif len(data_queue) < 500:
                cnt_skip += 4
                print(f'data_queue longer than 150. 4 frame skipped. i = {i}; cnt_skip = {cnt_skip}')
                data = data_queue.popleft()
                data = data_queue.popleft()
                data = data_queue.popleft()
                data = data_queue.popleft()
                data = data_queue.popleft()
                # clear memory
                flag_long_trace = True
                # reset background recording and clear memory
                flag_long_trace_bg = flag_long_trace_mm = True
            else:
                cnt_skip += len(data_queue) - 1
                print(f'data_queue longer than 500!!! data_queue cleared!!! i = {i}; cnt_skip = {cnt_skip}')
                data = data_queue.popleft()
                data_queue = deque([])
                # clear memory
                flag_long_trace = True
                # reset background recording and clear memory
                flag_long_trace_bg = flag_long_trace_mm = True
        # skip all delayed frames
        elif delay_prevention == 2:
            if len(data_queue) > 1:
                cnt_skip += len(data_queue) - 1
                print(f'data_queue longer than 1! data_queue cleared!!! i = {i}; cnt_skip = {cnt_skip}')
                data = data_queue.popleft()
                data_queue = deque([])
                # # clear memory
                # flag_long_trace = True
                # # reset background
                # flag_long_trace_bg = flag_long_trace_mm = True
            else:
                data = data_queue.popleft()
        else:
            print('ERROR: Unknow Delay Prevention Method!!!')
            quit()

        data_valid = True
        try:
            # true_data_to_mat(ast.literal_eval(data))
            # print(f'msg: {data}')
            # print('debug_cw: \n{}\nlen = {}'.format(data, len(data['data'])))
            mat_raw = true_data_to_mat(data, shape=sensor_resolution)
            if waking_trigger == 2:
                flag_wake = data["flag"]
        except Exception as e:
            data_valid = False
            print('\n========================================================================')
            print('data invalid! ')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('Error type: {}, happended in line {} on <true_data_to_mat>'.format(exc_type, exc_tb.tb_lineno))
            print('Error: {}'.format(e))
            print('========================================================================\n')


        time_now = time.time()
        delta_time = 0
        

        if viz:
            # pygame gui
            manager.update(time_delta)

            ## clear pygame screen
            screen.fill(bg)

            ## static draw
            # draw the logo img
            img = pygame.image.load(logo_img)
            img = pygame.transform.scale(img, (100, 100))
            x_start = screen_width / 100 * 6 - 50
            y_start = screen_height / 100 * 68 - 50
            screen.blit(img, (x_start, y_start))

        # serial data handling and delta time calculation
        if init:
            # print('skip first reading which is potentially incomplete')
            init = False
            time_prev = time_now
        # # calculate a new data point only if the previous one finished (manual)
        # elif time_now - time_prev >= delta_time + min_interval:
        elif data_valid:
            time_1 = time.time()

            ## print run time regularly
            if i % 500 in [0, 1, 2]:
                printing_run_time = True
            else:
                printing_run_time = False

            ## DATA

            # normalization
            # mat_raw = mat.copy()
            if normalize:
                mat = norm_scale * (mat_raw.copy() - minV) / (maxV - minV)
            else:
                mat = mat_raw.copy()
            # adaptive minV maxV, adaptive bg
            flag_mat_updated_mm = flag_mat_updated_bg = True

            # adaptive_min_max
            if adaptive_min_max:
                update_adaptive_min_max(i, window_mats, window_flags, window_num_frame=window_num_frame,
                                        frame_interval=frame_interval, min_delta=min_max_delta_limit,
                                        init_minV=init_minV, init_maxV=init_maxV)

            # adaptive bg 1, 3
            if adaptive_bg in [1, 3]:
                i_bg = update_adaptive_bg(i, i_bg, bg_mats_raw, window_flags_abg, reset_frames=reset_frames, bg_frame_num=bg_frame_num)

            mats.append(mat.copy())
            # fake imgName
            epoch_msec = int(data['timestamp'])  # round(time_now*1000)
            # epoch_id = data['device_id']
            epoch_id = data['deviceName']
            epoch = int(epoch_msec / 1000)
            imgName = f'Sensor_Frame_{epoch_msec}'
            # print(f'imgName: {imgName}')
            imgNames.append(imgName)
            
            # record recent mat_raw to RECENT_MATS_RAW_DICT
            if len(RECENT_MATS_RAW_DICT) >= 200:
                RECENT_MATS_RAW_DICT.pop(next(iter(RECENT_MATS_RAW_DICT)))
            RECENT_MATS_RAW_DICT[imgName] = mat_raw

            if data_recording:
                epochs_msec_recording.append(epoch_msec)
                serial_json_strs_recording.append(data)

            if step:
                tmp = input('Press "Enter" for the next step: ')
                if tmp == 'c':
                    step = False

            # secondary normalization for waking period
            if waking_filtering == 0:
                pass

            # simple waking filter
            elif waking_filtering == 1:
                # waking filter trigger
                if waking_trigger == 1:
                    # check if delta time larger than threshold
                    if epoch_msec - epoch_msec_prev >= int(waking_trigger_sec * 1000):
                        i_wake = waking_frame_num
                        print(f'simple waking filter triggered by delta time! ')
                elif waking_trigger == 2:
                    # check if flag=1
                    if flag_wake == 1:
                        i_wake = waking_frame_num
                        print(f'simple waking filter triggered by flag==1! ')
                # waking period
                if i_wake > 0:
                    # calc mat average for this frame
                    mat_average = np.average(mat)
                    # compare with mat average in bg mat
                    mat_average_diff = mat_average - np.average(bg_mat)
                    # shift data to secondary normalize the mat
                    mat = mat.copy() - mat_average_diff
                    i_wake -= 1
                # normal period
                else:
                    pass
                # update epoch msec prev
                epoch_msec_prev = epoch_msec

            # complete waking filter
            elif waking_filtering == 2:
                # waking filter trigger
                if waking_trigger == 1:
                    # check if delta time larger than threshold
                    if epoch_msec - epoch_msec_prev >= int(waking_trigger_sec * 1000):
                        i_wake = waking_frame_num
                        print(f'complete waking filter triggered by delta time! ')
                elif waking_trigger == 2:
                    # check if flag=1
                    if flag_wake == 1:
                        i_wake = waking_frame_num
                        print(f'complete waking filter triggered by flag==1! ')
                # waking period
                if i_wake > 0:
                    # calc roll average for this frame
                    roll_average_list = []
                    for n in range(mat.shape[0]):
                        roll_average_list.append(np.average(mat[n, :]))
                    # compare with roll average in bg mat
                    roll_average_list_diff = roll_average_list - np.average(bg_mat)
                    # shift data to secondary normalize the rolls, and apply horizontal weights
                    for n in range(mat.shape[0]):
                        for m in range(mat.shape[1]):
                            mat[n, m] = (mat[n, m] - roll_average_list_diff[n]) * (1.0 - (i_wake / waking_frame_num) * (1.0 - horizontal_weights[m]))
                    i_wake -= 1
                # normal period
                else:
                    pass

                # update epoch msec prev
                epoch_msec_prev = epoch_msec

            # adaptive bg 2
            if adaptive_bg == 2:
                i_bg = update_adaptive_bg(i, i_bg, bg_mats_raw, window_flags_abg, reset_frames=reset_frames, bg_frame_num=bg_frame_num)

            if viz:
                # gui slider
                if run_mode == 'saved_data':
                    # if not using the slider, update the slider value
                    mpos = pygame.mouse.get_pos()
                    if not (pygame.mouse.get_pressed()[0] and mpos[1] < 30 and mpos[0] > 485):
                        playback_slider_value = (epoch_msec - data_start_time_epoch * 1000) / (
                                    data_end_time_epoch * 1000 - data_start_time_epoch * 1000)
                        # print(f'playback_slider_value = {playback_slider_value}')
                        data_playback_slider.set_current_value(playback_slider_value)

            ## 1st BACKGROUND
            # if use first bg_frame_num frames as 1st bg
            if bg_initiation == 0:
                # if recording background
                if i_bg < bg_frame_num and i < bg_frame_num:
                    set_bg_mat(i_bg, bg_frame_num, bg_mats_raw)
                    if i_bg == bg_frame_num - 1:
                        flag_bg_mat_raw_defined = True
            # if use saved bg from txt as 1st bg
            elif bg_initiation == 1 and i == 1:
                # bg_mats_raw has already been defined in init
                bg_mat_raw = np.mean(np.array(bg_mats_raw), axis=0)
                if normalize:
                    bg_mat = norm_scale * (bg_mat_raw.copy() - minV) / (maxV - minV)
                else:
                    bg_mat = bg_mat_raw.copy()
                print('background has been reset')
                flag_bg_mat_raw_defined = True
            i_bg += 1

            # auto detecting threshold calc
            delta_value_mat_list.append(censoring(mat - bg_mat, left_percentile=20, right_percentile=80))
            prompt_str_list = []
            if len(delta_value_mat_list) > 10:
                delta_value_mat_list.pop(0)
            if (detect_method_hot in ['th', 'both'] and th_value_hot == 'auto') or (
                    detect_method_cold in ['th', 'both'] and th_value_cold == 'auto'):
                tmp = np.asarray([x for sublist in delta_value_mat_list for x in sublist])
                delta_mat_mean, delta_mat_std = tmp.mean(), tmp.std()
                # print('mean={}, std={}'.format(delta_mat_mean, delta_mat_std))
            if detect_method_hot in ['th', 'both'] and th_value_hot == 'auto':
                use_th_value_hot = delta_mat_mean + th_auto_std_scale_hot * delta_mat_std
                # print('detecting threshold for hot person is automatically set to: ', use_th_value_hot)
                prompt_str_list.append(
                    'detecting threshold for hot person is automatically set to: {:4.2f}'.format(use_th_value_hot))
            else:
                use_th_value_hot = th_value_hot
            if detect_method_cold in ['th', 'both'] and th_value_cold == 'auto':
                use_th_value_cold = delta_mat_mean - th_auto_std_scale_cold * delta_mat_std
                # use_th_value_cold = (np.percentile(tmp, 20) - np.percentile(tmp, 50)) * 4
                prompt_str_list.append(
                    'detecting threshold for cold person is automatically set to: {:4.2f}'.format(use_th_value_cold))
                # print('detecting threshold for cold person is automatically set to: ', use_th_value_cold)
            else:
                use_th_value_cold = th_value_cold
            if viz: fonts[0].render_to(screen, (5, 30), '; '.join(prompt_str_list), pygame.Color('white'))

            # analysis: detecting, tracking
            # if use_saved_bg: bg_mat = saved_bg
            if tracking_separate:
                traces_hot, boxList_hot, polygons_hot, records_hot, slope_mat_hot, r2_mat_hot, combine_log_hot = analyze_single_frame(
                    mat=mat, mat_raw=mat_raw, imgName=imgName, i=i,
                    records=records_hot, detect_type='hot',
                    screen_width=screen_width, screen_height=screen_height,
                    detect_method=detect_method_hot, drange=drange, delta_slope_th=delta_slope_th_hot,
                    delta_r2_th=delta_r2_th,
                    bg_slope_mat=bg_slope_mat, bg_r2_mat=bg_r2_mat, diff=False,
                    round2=False, drange2=1.5, delta_slope_th2=-0.1, delta_r2_th2=0.1,
                    bg_r2_mat2=bg_r2_mat2, th_value=use_th_value_hot,
                    delta_mat=mat - bg_mat, mask_idx_dict=mask_idx_dict, mask_dist_dict=mask_dist_dict,
                    sensor_resolution=sensor_resolution,
                    activity_map=activity_map, 
                    timestamp=data['timestamp'], deviceId=data['deviceName'])#deviceId=data['device_id'])

                traces_cold, boxList_cold, polygons_cold, records_cold, slope_mat_cold, r2_mat_cold, combine_log_cold = analyze_single_frame(
                    mat=mat, mat_raw=mat_raw, imgName=imgName, i=i,
                    records=records_cold, detect_type='cold',
                    screen_width=screen_width, screen_height=screen_height,
                    detect_method=detect_method_cold, drange=drange, delta_slope_th=delta_slope_th_cold,
                    delta_r2_th=delta_r2_th,
                    bg_slope_mat=bg_slope_mat, bg_r2_mat=bg_r2_mat, diff=False,
                    round2=False, drange2=1.5, delta_slope_th2=-0.1, delta_r2_th2=0.1,
                    bg_r2_mat2=bg_r2_mat2, th_value=use_th_value_cold,
                    delta_mat=mat - bg_mat, mask_idx_dict=mask_idx_dict, mask_dist_dict=mask_dist_dict,
                    sensor_resolution=sensor_resolution,
                    activity_map=activity_map)

                traces = traces_cold + traces_hot
                slope_mat = slope_mat_hot
                r2_mat = r2_mat_hot

                # check if there are detections for adaptive min max and adaptive bg
                positions = copy.deepcopy(records_hot['meanPositionsList'][-1])
                positions.extend(copy.deepcopy(records_cold['meanPositionsList'][-1]))
                # flag_detection_mm = flag_detection_bg = len(positions) != 0

            else:  # do not tracking separately for hot and cold
                records['imgNames'].append(imgName)
                detecting_rst_hot = detecting(delta_mat=mat - bg_mat, screen_width=screen_width,
                                              screen_height=screen_height,
                                              detect_method=detect_method_hot, drange=drange,
                                              mask_idx_dict=mask_idx_dict, mask_dist_dict=mask_dist_dict,
                                              delta_slope_th=delta_slope_th_hot, delta_r2_th=delta_r2_th,
                                              detect_type='hot', th_value=use_th_value_hot)
                detecting_rst_cold = detecting(delta_mat=mat - bg_mat, screen_width=screen_width,
                                               screen_height=screen_height,
                                               detect_method=detect_method_cold, drange=drange,
                                               mask_idx_dict=mask_idx_dict, mask_dist_dict=mask_dist_dict,
                                               delta_slope_th=delta_slope_th_cold, delta_r2_th=delta_r2_th,
                                               detect_type='cold', th_value=use_th_value_cold)
                if detect_method_hot in ['regress', 'both']:
                    slope_mat = detecting_rst_hot['slope_mat']
                    r2_mat = detecting_rst_hot['r2_mat']
                elif detect_method_cold in ['regress', 'both']:
                    slope_mat = detecting_rst_cold['slope_mat']
                    r2_mat = detecting_rst_cold['r2_mat']
                boxList_hot, boxList_cold = detecting_rst_hot['boxList'], detecting_rst_cold['boxList']
                polygons_hot, polygons_cold = detecting_rst_hot['polygons'], detecting_rst_cold['polygons']
                meanPositionsList_hot, meanPositionsList_cold = detecting_rst_hot['meanPositionsList'], \
                                                                detecting_rst_cold['meanPositionsList']
                records['meanPositionsList'].append(meanPositionsList_hot + meanPositionsList_cold)
                if enable_tracking:
                    traces, records, combine_log = tracking(records, i, screen_width=screen_width, 
                        screen_height=screen_height, sensor_resolution=sensor_resolution)

                # check if there are detections for adaptive min max and adaptive bg
                positions = records['meanPositionsList'][-1]
                # flag_detection_mm = flag_detection_bg = len(positions) != 0

            # for saved_data format activity_map, update activity map if we have new detections
            if activity_map is not None and len(positions) > 0 and mqtt_topic_activity_map is None and saved_activity_map_file_path is not None and os.path.exists(saved_activity_map_file_path + '_values.p'):
                activity_map.read_activity_map(read_from='saved_file', data_type='values',
                                                  saved_file_path=saved_activity_map_file_path + '_values.p')
                # print('Activity map value has been updated to the latest')

            # do filter here for adaptive background using activity map
            if activity_map is not None and activity_map.status in ['only applying', 'updating and applying']:
                detections_filtered_by_activity_map = activity_map.filter_for_new_detections(positions, coords_mode='screen')
                # print(detections_filtered_by_activity_map)    #debug_cw
                positions = detections_filtered_by_activity_map[2]
            flag_detection_mm = flag_detection_bg = len(positions) != 0
            
            ## detection coordinates publish to mqtt or write to text file per frame
            if detection_write_file or detection_to_mqtt:
                stamp_curr = epoch_msec
                sensor_id = mqtt_id
                positions = records_hot['meanPositionsList'][-1]
                positions.extend(records_cold['meanPositionsList'][-1])
                coords = []
                for pair in positions:
                    # from coord(screen) to coord(0~1)
                    x = (pair[0] - int(screen_width / 3 * 0.2)) / (cell_size * sensor_resolution[0])
                    y = (pair[1] - int(screen_height / 2 * 0.2)) / (cell_size * sensor_resolution[1])
                    coords.append((x, y))

                # only publish or write when there are detections, and it is not a false positive detection 
                # FP filter, logic: if previous frame doesn't have detection, it's a single pop up FP
                fp_dtc = True
                if detection_fp_filter == 0:
                    fp_dtc = False
                if detection_fp_filter == 1 and len(coords_prev) > 0:
                    fp_dtc = False
                is_empty_list = len(coords) == 0
                # send when allow empty list, or list is no empty when doesn't allow empty list; not send if it's an FP
                if ((not is_empty_list) or (detection_send_empty_list)) and (not fp_dtc):
                    detections = {'deviceName': sensor_id,
                                  'timestamp': stamp_curr,
                                  'detectionsLocal': coords}
                    # convert local coordinates to world coordinates
                    if world_coord:
                        d = [x for x in spatial_config_data['sensors'] if x['deviceName'] == sensor_id][0]
                        if 'room' in d.keys(): detections['room'] = d['room']
                        coords_wld = []
                        for coord in coords:
                            x = d['center'][0] + (coord[0] - 0.5) * d['coverage_dim'][0]
                            y = d['center'][1] + ((1.0-coord[1]) - 0.5) * d['coverage_dim'][1]
                            # rotate around center
                            coord_new = (x, y)
                            if 'rotation' in d.keys():
                                coord_new = rotate_point(coord_new, d['center'], d['rotation'])
                            # unified world coord rotation, for backwards compatibility
                            if world_coord_rotation != 0:
                                coord_new = rotate_point(coord_new, d['center'], world_coord_rotation)
                            # flip
                            if world_coord_flip_x:
                                coord_new = (d['center'][0]-(coord_new[0]-d['center'][0]), coord_new[1])
                            if world_coord_flip_y:
                                coord_new = (coord_new[0], d['center'][1]-(coord_new[1]-d['center'][1]))
                            coords_wld.append(coord_new)
                        detections['detectionsWorld'] = coords_wld
                    if msg_convert == 'butlrOfficeMeetingRoom':
                        detections['frame'] = data['frame']
                    elif msg_convert == 'seamless' or msg_convert == 'seamless64':
                        detections['utcSecs'] = data['utcSecs']
                        detections['utcUsecs'] = data['utcUsecs']

                    # write to txt file
                    if detection_write_file:
                        with open(detection_write_path, 'a+') as out_file:
                            out_file.write(str(detections))
                            out_file.write("\n")

                    # publish to mqtt
                    if detection_to_mqtt and client_global is not None:
                        (rc, mid) = client_global.publish(mqtt_topic_detection, json.dumps(detections), qos=mqtt_qos)
                coords_prev = coords
                        
            boxList = boxList_cold + boxList_hot
            polygons = polygons_cold + polygons_hot


            # viz
            x_start = int(screen_width / 3 * 0.2)
            y_start = int(screen_height / 2 * 0.2)
            if viz:
                plt_heatmap_pygame_mat_pure(x_start, y_start, cell_size, mat,
                                            viz_min=0, viz_max=30, sensor_resolution=sensor_resolution,
                                            title=imgName + '\n' + strftime('%Y-%m-%d %H:%M:%S', localtime(epoch / 1000)))
                # plt_boundary(x_start, y_start, cell_size, mat,
                #                             viz_min=0, viz_max=30, sensor_resolution=sensor_resolution)
                plt_heatmap_pygame_polygon_pure(x_start, y_start, cell_size, boxList_cold, polygons_cold,
                                                detect_type='cold', sensor_resolution=sensor_resolution)
                plt_heatmap_pygame_polygon_pure(x_start, y_start, cell_size, boxList_hot, polygons_hot,
                                                detect_type='hot', sensor_resolution=sensor_resolution, timestamp=data['timestamp'], device_id=epoch_id)

                x_start_tmp = int(screen_width / 3 * 2.2)
                y_start_tmp = int(screen_height / 2 * 0.2)
                plt_mat_text_pygame(x_start_tmp, y_start_tmp, cell_size, mat - bg_mat, viz_min=-2.5, viz_max=2.5,
                                    title='DELTA_VALUE', sensor_resolution=sensor_resolution)

                x_start_tmp = int(screen_width / 3 * 1.2)
                y_start_tmp = int(screen_height / 2 * 0.2)
                plt_mat_text_pygame(x_start_tmp, y_start_tmp, cell_size, mat_raw, viz_min=0, viz_max=30,
                                    title='RAW', sensor_resolution=sensor_resolution)

                x_start_tmp = int(screen_width / 3 * 1.2)
                y_start_tmp = int(screen_height / 2 * 1.1)
                plt_mat_text_pygame(x_start_tmp, y_start_tmp, cell_size, slope_mat, viz_min=-2, viz_max=2,
                                    title='SLOPE', sensor_resolution=sensor_resolution)
                x_start_tmp = int(screen_width / 3 * 2.2)
                y_start_tmp = int(screen_height / 2 * 1.1)
                plt_mat_text_pygame(x_start_tmp, y_start_tmp, cell_size, r2_mat, viz_min=0, viz_max=1,
                                    title='R2', sensor_resolution=sensor_resolution)

            if enable_tracking:
                # post combine for cold+hot traces when tracking separately
                if track_post_combine and tracking_separate:
                    # post_combine was initally designed for purely cold/hot traces, and do not allow time_window=0
                    # but in order to combine cold and hot traces, this case should be considered
                    gap_dict_combine_hot_and_cold = copy.deepcopy(gap_dict_combine)
                    if '0' not in gap_dict_combine_hot_and_cold: gap_dict_combine_hot_and_cold[0] = \
                    gap_dict_combine_hot_and_cold[1]
                    traces, combine_log_after = post_combine(traces, uncondintionRules, conditionRules,
                                                             gap_dict=gap_dict_combine_hot_and_cold,
                                                             coodScale=coordScale_combine, uncombinableFlag=False)
                    combine_log = combine_log_hot + combine_log_cold + combine_log_after

                if tracking_extend_to_border:
                    traces = extent_traces_to_border(traces, lastN=extendPara['lastN'], max_th=extendPara['max_th'],
                                                     min_th=extendPara['min_th'])
                if viz:
                    # display traces
                    if display_trace:
                        frame_num = i_clear
                        viz_tracks(traces, frame_num, x_start, y_start, cell_size, show_reflect=True,
                                   sensor_resolution=sensor_resolution)

                # flag long trace
                for trace in traces:
                    # print(f'trace: {trace}')
                    # print(f'len(trace["cood"]): {len(trace["cood"])}')
                    if len(trace['cood']) >= long_trace_num:
                        print('long trace detected! ')
                        flag_long_trace_mm = flag_long_trace_bg = flag_long_trace = True
                    elif len(trace['cood']) >= valid_trace_num[0] and len(trace['cood']) < valid_trace_num[1]:
                        # print('valid trace detected! ')
                        flag_valid_trace_mm = flag_valid_trace_bg = True

                ## COUNTING
                if viz:
                    # viz counting result
                    if viz_counting:
                        frame_num = i_clear
                        ppl_count, ppl_count_in, ppl_count_out = viz_counts(traces, frame_num, door_line_start, door_line_end,
                                                                        x_start, y_start, cell_size)

                # for publish
                # if len(traces) > 0:
                #     print('Back to main: len(traces) = ', len(traces))   #cw_debug
                # print('\n'+'='*20+'debug_cw2'+ '='*20)
                # for t in traces: print('{}\n'.format(t))
                # print('='*20+'debug_cw2'+ '='*20+'\n')
                # print([t['waiting_for_combine'] for t in traces])
                if printing_run_time and debug_run_time: t1 = time.time()
                traces_to_show = [
                    trace for trace in traces if trace['fnsh'] is True and 
                    trace['show'] is False and 
                    len(trace['waiting_for_combine']) >= output_wait_for_post_combine_steps and 
                    any(trace['waiting_for_combine'][-output_wait_for_post_combine_steps:]) is False
                ]
                ppl_count_show_list, ppl_count_in_show_list, ppl_count_out_show_list = [], [], []
                if len(traces_to_show) > 0:  # do counting only when there is a new finished trace
                    for trace_to_show in traces_to_show:
                        this_ppl_count_show, this_ppl_count_in_show, this_ppl_count_out_show = \
                            generate_counts([trace_to_show], i, door_line_start, door_line_end, x_start, y_start, cell_size,
                                            first_only=counting_first_only)
                        this_ppl_count_show = 1  #cw_debug
                        ppl_count_show_list.append(this_ppl_count_show)
                        ppl_count_in_show_list.append(this_ppl_count_in_show)
                        ppl_count_out_show_list.append(this_ppl_count_out_show)
                    ppl_count_show = np.asarray(ppl_count_show_list).sum()
                else:
                    ppl_count_show = -100
                if printing_run_time and debug_run_time: t2 = time.time()
                if printing_run_time and debug_run_time: print(
                    f'DEBUG: ** {t2 - t1:4.4f} seconds elapsed for "Counting for publish" in "pygame_run_mqtt" **')

                # output
                try:
                    crt_epoch = data['timestamp'] / 1000
                    crt_epoch_msec = data['timestamp']
                except:
                    crt_epoch = time.time()
                crt_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(crt_epoch))
                if len(traces_to_show) > 0 and ppl_count_show > 0:
                    # once it is going to output, all finised traces would be marked as show=True
                    # print('Curret traces to show: ', len(traces_to_show)) #cw_debug
                    for idx, trace in enumerate(traces_to_show):
                        if tracking_separate:
                            if trace['id'] in records_hot['trackingDict']:
                                # records_hot['trackingDict'][trace['id']]['show'] = True
                                for trace_idx in trace['combined_ids']:
                                    records_hot['trackingDict'][trace_idx]['show'] = True
                            elif trace['id'] in records_cold['trackingDict']:
                                # records_cold['trackingDict'][trace['id']]['show'] = True
                                for trace_idx in trace['combined_ids']:
                                    records_cold['trackingDict'][trace_idx]['show'] = True
                            else:
                                print('Error: cannot find the trace id from hot/cold trackingDict')
                        else:
                            if trace['id'] in records['trackingDict']:
                                # records['trackingDict'][trace['id']]['show'] = True
                                for trace_idx in trace['combined_ids']:
                                    records['trackingDict'][trace_idx]['show'] = True
                            else:
                                print('Error: cannot find the trace id from trackingDict')
                        
                        # check for futon-effect
                        if ENABLE_FUTON_EFFECT_DETECTION:
                            futon_flag = check_futon_effect(
                                trace, printFlag=True, recent_mats_raw=RECENT_MATS_RAW_DICT, 
                                window=None, method=CHECK_FUTON_METHOD
                            )
                        else:
                            futon_flag = False
                        
                        # if ppl_count_show_list[idx] > 0:
                        if True:        # do not care count, cw_debug
                            output_dict = {}
                            this_trace = tuple2list(trace['cood'])
                            output_dict['DeviceName'] = mqtt_id
                            output_dict['Time'] = crt_time
                            output_dict['Epoch'] = crt_epoch_msec
                            output_dict['Trajectory'] = [this_trace]
                            output_dict['Exit'] = [this_trace[-1]]
                            output_dict['InAndOut'] = [ppl_count_in_show_list[idx], ppl_count_out_show_list[idx]]
                            to_Mqtt = output_dict

                            # prevent repeated msgs
                            trajectory_now = to_Mqtt['Trajectory'][0]
                            # print(f'DEBUG: len(trajectories_prev) before: {len(trajectories_prev)}')
                            if len(trajectories_prev) == 0:
                                # print and publish to mqtt server
                                print('\n{}\n'.format(json.dumps(to_Mqtt)))
                                if futon_flag: 
                                    print('The trace above is suspected as False due to futon effect and will not be counted')
                                    continue
                                if save_trace_file_path is not None and os.path.exists(save_trace_file_path):
                                    save_trace_to_file(trace, save_trace_file_path)  #hack_cw
                                if client_for_am is not None and publish_for_activity_map:
                                    (rc, mid) = client_for_am.publish(mqtt_topic_trace, json.dumps(trace),
                                                                      qos=mqtt_qos)
                                    print('New trace published to mqtt for activity map at {}'.format(
                                        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                                    ))
                                if client_global is not None and publish:
                                    (rc, mid) = client_global.publish(mqtt_topic_out, json.dumps(to_Mqtt), qos=mqtt_qos)
                            else:
                                match_list = False
                                for t in range(len(trajectories_prev)):
                                    trajectory_prev = trajectories_prev[t]
                                    match_item = True
                                    n = 1
                                    while n < 6:
                                        if len(trajectory_prev) > n and len(trajectory_now) > n:
                                            if trajectory_prev[n] != trajectory_now[n]:
                                                match_item = False
                                                break
                                        n += 1
                                    # print(f'DEBUG: checking prev msg #{t+1}/{len(trajectories_prev)}; n={n}; match_item={match_item}. ')
                                    if match_item:
                                        match_list = True
                                        break
                                if match_list:   # cw_debug
                                # if False:           # always output, do not compare
                                    # print(f'WARNING: repeated message detected in #{t+1} of previous {len(trajectories_prev)} messages, ignored!!! ')
                                    pass
                                else:
                                    # print and publish to mqtt server
                                    print('\n{}\n'.format(json.dumps(to_Mqtt)))
                                    if futon_flag: 
                                        print('The trace above is suspected as False due to futon effect and will not be counted')
                                        continue
                                    if save_trace_file_path is not None and os.path.exists(save_trace_file_path):
                                        save_trace_to_file(trace, save_trace_file_path)  #hack_cw
                                    if client_for_am is not None and publish_for_activity_map:
                                        (rc, mid) = client_for_am.publish(mqtt_topic_trace, json.dumps(trace),
                                                                          qos=mqtt_qos)
                                        print('New trace published to mqtt for activity map at {}'.format(
                                            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                                        ))
                                    if client_global is not None and publish:
                                        (rc, mid) = client_global.publish(mqtt_topic_out, json.dumps(to_Mqtt), qos=mqtt_qos)
                            trajectories_prev.append(trajectory_now)
                            if len(trajectories_prev) > 10:
                                trajectories_prev.popleft()
                            # print(f'DEBUG: len(trajectories_prev) after: {len(trajectories_prev)}')
                            # print(f'DEBUG: trajectories_prev: {trajectories_prev}\n\n')

                # alternative algorithms
                if auto_disp_trace:
                    frame_num = i

                # calculate delta time
                time_2 = time.time()
                delta_time = time_2 - time_1
                # print(f'delta_time = {delta_time} seconds')

                # clear memory but keep ppl_count_global
                if step:
                    print('\n' + '-' * 50)
                    # print(records['meanPositionsList'][-1])
                    print('MeanPostions:')
                    for mp in records['meanPositionsList'][-1]:
                        x, y = mp
                        print(((x - int(screen_width / 3 * 0.2)) / cell_size, (y - int(screen_height / 2 * 0.2) / cell_size)))
                        # print(((x - 85) / 30, (y - 72) / 30))

                    print('traces: ')
                    for t in traces:
                        # print([[int(85+x*30), int(72+y*30)] for x, y in t['cood']])
                        print([(x * 8, y * 8) for x, y in t['cood']])

            if i_clear > mem_clr_num_frames + 2 or flag_long_trace:
                mats = []
                # imgNames = []  # can't clear because tracking need it
                epochs = []
                meanPositionsList = [[], [], []]  # need to have 3 items to avoid error when meanPositionsList[-3]
                trackingDict = {}
                # input('clear memory：')

                if enable_tracking:
                    # ============ a new way to clear memory============#
                    if tracking_separate:
                        records_hot['meanPositionsList'] = records_hot['meanPositionsList'][-2:]
                        records_cold['meanPositionsList'] = records_cold['meanPositionsList'][-2:]
                        currentNames_hot, currentNames_cold = [], []
                        for idx, mp in enumerate(records_hot['meanPositionsList'][-1]):
                            currentNames_hot.append('{}:{}'.format(records_hot['imgNames'][-1], idx))
                        for idx, mp in enumerate(records_cold['meanPositionsList'][-1]):
                            currentNames_cold.append('{}:{}'.format(records_cold['imgNames'][-1], idx))
                        # get inital names (keys of trackingDict) of traces related to latest meanPositions
                        currentAsKey_hot = {route['current']: start for start, route in records_hot['trackingDict'].items()}
                        currentAsKey_cold = {route['current']: start for start, route in
                                             records_cold['trackingDict'].items()}
                        initalNames_hot = [currentAsKey_hot[x] for x in currentNames_hot]
                        initalNames_cold = [currentAsKey_cold[x] for x in currentNames_cold]
                        # check again if there is any trace to be combined
                        combined = []
                        for tt in initalNames_hot + initalNames_cold:
                            combined += [pair for pair in combine_log if tt in pair]
                        combined = [x for pair in combined for x in pair]
                        keep_ids = list(set(initalNames_hot + initalNames_cold + combined))

                        # only keep traces with intial names in list above
                        records_hot['trackingDict'] = {key: value for key, value in records_hot['trackingDict'].items() if
                                                       key in keep_ids}
                        records_cold['trackingDict'] = {key: value for key, value in records_cold['trackingDict'].items() if
                                                        key in keep_ids}
                        # records_hot['trackingDict'] = {key: value for key, value in records_hot['trackingDict'].items() if key in initalNames_hot}
                        # records_cold['trackingDict'] = {key: value for key, value in records_cold['trackingDict'].items() if key in initalNames_cold}

                        # for long trace
                        compress_long_traces(records_hot['trackingDict'])
                        compress_long_traces(records_cold['trackingDict'])

                        # imgNames: find the earliest imgName for the routes still existing, delete all imgNames earlier than that
                        all_imgNames = records_hot['imgNames']
                        keep_imgNames_idx = [all_imgNames.index(id.split(':')[0]) for id in keep_ids]
                        if len(keep_imgNames_idx) > 0:
                            ealiest_imgName_idx = min(keep_imgNames_idx)
                        else:
                            ealiest_imgName_idx = len(all_imgNames) - 2
                        if len(all_imgNames) - ealiest_imgName_idx < 2:
                            ealiest_imgName_idx = len(all_imgNames) - 2  # at least keep last 2 imgNames
                        records_hot['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                        records_cold['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                    else:
                        records['meanPositionsList'] = records['meanPositionsList'][-2:]
                        currentNames = []
                        for idx, mp in enumerate(records['meanPositionsList'][-1]):
                            currentNames.append('{}:{}'.format(records['imgNames'][-1], idx))
                        # get inital names (keys of trackingDict) of traces related to latest meanPositions
                        currentAsKey = {route['current']: start for start, route in records['trackingDict'].items()}
                        initalNames = [currentAsKey[x] for x in currentNames]
                        # check again if there is any trace to be combined
                        combined = []
                        for tt in initalNames:
                            combined += [pair for pair in combine_log if tt in pair]
                        combined = [x for pair in combined for x in pair]
                        keep_ids = list(set(initalNames + combined))

                        # only keep traces with intial names in list above
                        records['trackingDict'] = {key: value for key, value in records['trackingDict'].items() if
                                                   key in keep_ids}

                        # for long trace
                        compress_long_traces(records['trackingDict'])

                        # imgNames: find the earliest imgName for the routes still existing, delete all imgNames earlier than that
                        all_imgNames = records['imgNames']
                        keep_imgNames_idx = [all_imgNames.index(id.split(':')[0]) for id in keep_ids]
                        if len(keep_imgNames_idx) > 0:
                            ealiest_imgName_idx = min(keep_imgNames_idx)
                        else:
                            ealiest_imgName_idx = len(all_imgNames) - 2
                        if len(all_imgNames) - ealiest_imgName_idx < 2:
                            ealiest_imgName_idx = len(all_imgNames) - 2  # at least keep last 2 imgNames
                        records['imgNames'] = all_imgNames[ealiest_imgName_idx:]
                    # ============ a new way to clear memory============#

                if viz and enable_tracking and viz_counting:
                    ppl_count_global += ppl_count
                    ppl_count_in_global += ppl_count_in
                    ppl_count_out_global += ppl_count_out
                i_clear = 0
                flag_long_trace = False

                print('memory cleared')

            # proceed i
            i += 1
            i_clear += 1

        # when a mqtt input is skipped
        else:
            print('!!! a mqtt input is skipped !!!')

        if viz:
            # draw pygame gui
            manager.draw_ui(screen)
            if run_mode == 'saved_data':
                # playback slider text
                fonts[0].render_to(screen, (400, 15), data_start_time_str, pygame.Color('white'))
                fonts[0].render_to(screen, (1175, 15), data_end_time_str, pygame.Color('white'))
                playback_slider_value = data_playback_slider.get_current_value()
                x_pos = playback_slider_value * 620 + 515 - 40
                data_slider_time_epoch = data_start_time_epoch + int(
                    (data_end_time_epoch - data_start_time_epoch) * playback_slider_value)
                data_slider_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data_slider_time_epoch))
                fonts[0].render_to(screen, (x_pos, 5), data_slider_time_str, pygame.Color('white'))
                # play speed slider text
                fonts[0].render_to(screen, (795, 37), f'play speed： {play_speed_lower_bound}x', pygame.Color('white'))
                fonts[0].render_to(screen, (1175, 37), f'{play_speed_upper_bound}x', pygame.Color('white'))
                play_speed_slider_value = play_speed_slider.get_current_value()
                x_pos = (play_speed_slider_value - play_speed_lower_bound) / (
                            play_speed_upper_bound - play_speed_lower_bound) * 235 + 890
                fonts[0].render_to(screen, (x_pos, 26), f'{play_speed_slider_value:2.2f}x', pygame.Color('white'))

        # time_delta pygame gui
        time_delta = round((time_now - time_prev) * 1000)  # for pygame gui
        delta_time_sec = time_now - time_prev  # for debugging
        if printing_run_time and debug_run_time:
            print('DEBUG: ***** {:4.4f} seconds elapsed for 1 frame *****'.format(delta_time_sec))
            # list_delta_time_sec.append(delta_time_sec)
            # average_delta_time_sec = np.average(np.asarray(list_delta_time_sec))
            # print('DEBUG: ***** {:4.4f} seconds elapsed for 1 frame. average_time_delta = {:4.4f} *****'.format(delta_time_sec, average_delta_time_sec))

        # pass time_prev
        time_prev = time_now

        ## display info
        # fps
        # fps_str = f'fps: {round(clock.get_fps(),1)}'
        fps = round(1 / (time_delta / 1000 + 0.00000001), 1)
        fps_str = f'fps: {fps}'
        if viz: fonts[0].render_to(screen, (5, 5), fps_str, pygame.Color('white'))
        # length of data queue
        q_len_str = f'q_len: {len(data_queue)}'
        if viz: fonts[0].render_to(screen, (5, 15), q_len_str, pygame.Color('white'))
        # data timestamp
        data_time_run = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(epoch))
        data_time_str = 'data time GMT: ' + data_time_run
        if viz: fonts[0].render_to(screen, (55, 5), data_time_str, pygame.Color('white'))
        # machine current time
        epoch_machine_time = int(time.time())
        machine_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(epoch_machine_time))
        machine_time_str = 'machine time GMT: ' + machine_time
        if viz: fonts[0].render_to(screen, (225, 5), machine_time_str, pygame.Color('white'))
        # mqtt_id
        mqtt_id_str = f'mqtt_id: {mqtt_id}'
        if viz: fonts[0].render_to(screen, (55, 15), mqtt_id_str, pygame.Color('white'))
        # min max
        min_max_str = f'min: {minV}; max: {maxV}'
        if viz: fonts[0].render_to(screen, (225, 15), min_max_str, pygame.Color('white'))

        ## print info regularly
        if i % 500 == 0:
            # data time latestly received
            if len(data_queue) > 0:
                epoch_data_receive = int(int(data_queue[len(data_queue) - 1]['timestamp']) / 1000)
            else:
                epoch_data_receive = epoch
            data_time_receive = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(epoch_data_receive))
            # print
            print(f'{min_max_str}; {q_len_str}; {fps_str}; {data_time_str}; {machine_time_str}')
            # publish json to status
            ## TODO: 1) flag data received continouity 2) bg reset flag 3) clear memory flag 4) long trace flag 5) direct sun light flag 6) bg img 7) session delta times 8) min max reset flag
            output_dict = {}
            output_dict['version'] = version
            output_dict['mqtt_id'] = mqtt_id
            output_dict['min'] = minV
            output_dict['max'] = maxV
            output_dict['q_len'] = len(data_queue)
            output_dict['fps'] = fps
            output_dict['delta_time_frame'] = round(delta_time_sec, 4)
            output_dict['machine_time_gmt'] = machine_time
            output_dict['machine_time_epoch'] = epoch_machine_time
            output_dict['data_time_run_gmt'] = data_time_run
            output_dict['data_time_run_epoch'] = epoch
            output_dict['data_time_receive_gmt'] = data_time_receive
            output_dict['data_time_receive_epoch'] = epoch_data_receive
            status_to_mqtt = output_dict
            # status_to_mqtt = {'version':version, 'mqtt_id':mqtt_id, 'min':minV, 'max':maxV, 'q_len':len(data_queue), 'fps':fps, 'delta_time_frame':delta_time_sec, 'machine_time_gmt':machine_time, 'data_time_run_gmt':data_time_run, 'data_time_receive_gmt':data_time_receive}
            if client_global is not None and publish:
                (rc, mid) = client_global.publish(mqtt_topic_status, json.dumps(status_to_mqtt), mqtt_qos)

        if viz: 
            ## update the pygame screen
            pygame.display.flip()

        # # pygame fps limit
        # clock.tick(fps)

        # if batch evaluate, exit when finish all data queue
        if batch_eval:
            if len(data_queue) <= 0 and i > 10:
                is_exiting = True
            if is_exiting:
                sys.exit()

def load_config(configFile='config.json'):
    json_data = json.load(open(path.abspath(configFile), 'r'))
    # print(json_data)
    return json_data

def reset_counting_line(key):
    global door_line_start
    global door_line_end
    if key == 'v_l':
        # left
        door_line_start = (0.167, 0.001)
        door_line_end = (0.167, 0.999)
    elif key == 'v_ml':
        # mid-left
        door_line_start = (0.334, 0.001)
        door_line_end = (0.334, 0.999)
    elif key == 'v_m':
        # middle (vertical)
        door_line_start = (0.501, 0.001)
        door_line_end = (0.501, 0.999)
    elif key == 'v_mr':
        # mid-right
        door_line_start = (0.667, 0.001)
        door_line_end = (0.667, 0.999)
    elif key == 'v_r':
        # right
        door_line_start = (0.834, 0.001)
        door_line_end = (0.834, 0.999)
    elif key == 'h_t':
        # top
        door_line_start = (0.001, 0.167)
        door_line_end = (0.999, 0.167)
    elif key == 'h_mt':
        # mid-top
        door_line_start = (0.001, 0.334)
        door_line_end = (0.999, 0.334)
    elif key == 'h_m':
        # middle (horizontal)
        door_line_start = (0.001, 0.501)
        door_line_end = (0.999, 0.501)
    elif key == 'h_mb':
        # mid-bottom
        door_line_start = (0.001, 0.667)
        door_line_end = (0.999, 0.667)
    elif key == 'h_b':
        # bottom
        door_line_start = (0.001, 0.834)
        door_line_end = (0.999, 0.834)
    else:
        print('key for counting line reset not defined!!!')

def mat_remove_hot_object(mat_src, mat_slope, mat_bg, hot_obj_temp_th, hot_obj_slope_th):
    mat_dst = mat_src
    nrow, ncol = mat_src.shape
    for i in range(nrow):
        for j in range(ncol):
            if mat_src[i, j] > hot_obj_temp_th and mat_slope[i, j] < hot_obj_slope_th:
                mat_dst[i, j] = mat_bg[i, j]
    return mat_dst

def count_evaluate(count_label_path, count_predict, gap_th=12):
    global eval_result

    correct = 0
    missing = 0
    false_positive = 0
    total = 0
    accuracy = 0

    # load count label
    count_label = json.load(open(count_label_path, 'r'))
    # print(f'count_label: {count_label}')
    label_list = []
    for l in count_label:
        for i in range(l['count']):
            label_list.append(l['i'])
    label_list.sort()
    print(f'label_list: {label_list}')

    # load count pridict
    predict_list = []
    for l in count_predict:
        for i in range(l['count']):
            predict_list.append(l['i'])
    predict_list.sort()
    print(f'predict_list: {predict_list}')

    # eval
    total = len(label_list)
    for label in label_list:
        # # print(f'label: {label}')
        gap_list = []
        for predict in predict_list:
            gap_list.append(abs(predict - label))
        # # print(f'gap_list: {gap_list}')
        tmp = list(zip(predict_list, gap_list))
        tmp.sort(key=lambda x: x[1])
        predict_list = [x[0] for x in tmp]
        # # print(f'predict_list: {predict_list}')
        if len(predict_list) > 0:
            gap_min = abs(predict_list[0] - label)
            # # print(f'gap_min: {gap_min}')
            if gap_min <= gap_th:
                correct += 1
                # # print('correct += 1')
                predict_list = predict_list[1:]
            else:
                missing += 1
                # # print('missing += 1')
        elif len(predict_list) == 0:
            false_positive += 1
        # # print(f'predict_list: {predict_list}')
    false_positive += len(predict_list)
    # # print(f'false_positive: {false_positive}')

    correct = total - missing
    if not total == 0:
        accuracy = (total - missing - false_positive) / total
        precision = correct / (correct + false_positive)
        recall = correct / (correct + missing)
        false_negative_rate = missing / total  # 漏检率 = 漏检人次 / 实际人次
        false_positive_rate = false_positive / (correct + false_positive)  # 误检率 = 误检人次 / 检测出总人次
        total_error_rate = false_negative_rate + false_positive_rate  # 总错误率 = 漏检率 + 误检率
        print(f'correct = {correct}, missing = {missing}, false-positive = {false_positive}, total = {total}')
        print(f'precision 误检准确率 = correct/(correct+false_positive): {round(precision * 100, 2)}%')
        print(f'recall 漏检准确率 = correct/(correct+missing): {round(recall * 100, 2)}%')
        print(f'worst-case accuracy = (total-missing-false_positive)/total: {round(accuracy * 100, 2)}%')
        print(f'false-negative rate 漏检率 = missing/total: {round(false_negative_rate * 100, 2)}%')
        print(
            f'false-positive rate 误检率 = false_positive/(correct+false_positive): {round(false_positive_rate * 100, 2)}%')
        print(f'total error rate 总错误率 = false_negative_rate+false_positive_rate: {round(total_error_rate * 100, 2)}%')
    else:
        print('error: total == 0 !!!')
    eval_result.append(
        {'correct': correct, 'missing': missing, 'false_positive': false_positive, 'total': total, 'accuracy': accuracy,
         'precision': precision, 'recall': recall, 'false_negative_rate': false_negative_rate,
         'false_positive_rate': false_positive_rate, 'total_error_rate': total_error_rate})
    return accuracy, correct, missing, false_positive, total

def load_config_internally():
    config = {
        '---------------------------APP-----------------------------': '',
        'run_mode': 'batch_wiwide',
        'com_name': 'COM3',
        'data_folder': 'C:/Users/RYAN/Google Drive/01_Work/Butlr/_Data/200205_Wiwide PoC Test Data/sensor_log/sensor_log_09',
        'viz_label': False,
        'label_folder': 'C:/Users/RYAN/Dropbox (MIT)/01_Work/Butlr/RH_GH_PoC_sim_viz/frames/006/positions',
        'count_labeling': False,
        'count_eval': False,
        'count_label_path': 'C:/Users/RYAN/Google Drive/01_Work/Butlr/_Data/200109_Wiwide PoC Rehearsal Data/Data Gen/16/count_label.json',
        'fps': 0,
        'start_id': 0,
        'step': False,
        'auto_memo_clear': 200,
        '------------------------DETECTION--------------------------': '',
        'normalize': True,
        'minV': 15,
        'maxV': 25,
        'remove_hot_object': False,
        'hot_obj_temp_th': -2,
        'hot_obj_slope_th': -0.5,
        'bg_frame_num_serial': 10,
        'bg_index_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'detection_method': 'regress',
        'drange': {'center': 3.5, 'edge': 3.0},
        'center_slope': -0.9,  # -1.4003917121605036,
        'border_slope': -0.9,  # -1.0714296342152252,
        'center_r2': 0.0,  # 0.06395072267985241,
        'border_r2': 0.0,  # -0.0204679963355553,
        'delta_r2_th': 0.0,
        'diameter_th': {'2': 4, '3': 8},
        '------------------------TRACKING---------------------------': '',
        'trackingPara': {'b_d': -0.9, 'b_lr': -0.1},  # {'b_d': -0.9985747848416009, 'b_lr': -0.11843827395935666},
        'gap_dict_combine': {'1': 85, '2': 85, '3': 105, '4': 110},
        'step_th_remove': 3,
        'extendPara': {'lastN': 18, 'max_th': 0.7, 'min_th': 0},
        'momentum_eta': 0.9,  # 0.9993351363282554,
        '------------------------COUNTING---------------------------': '',
        'counting_line_key': 'v_mr',
        'gap_dict_counting': {'1': 60, '2': 60, '3': 90}
    }
    return config

def save_data_to_file(json_strs, epochs, file_name):
    """
    """
    data_save_dir = '../data'

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
        print('made path: {}'.format(data_save_dir))

    f = open(f'{data_save_dir}/{file_name}', 'w+')
    f.close()

    f = open(f'{data_save_dir}/{file_name}', 'a')
    for json_str, epoch in zip(json_strs, epochs):
        line_str = f'{epoch / 1000.0},{json_str}\n'
        f.write(line_str)
    f.close()

    print(f'Recorded data saved to file: \'{data_save_dir}/{file_name}\'. ')
    mats_recording = []
    epochs_msec_recording = []

def center_border_mapping(center_value, border_value, sensor_resolution=(8,8), edge_resolution=(2,2)):
    """
    """
    # mat = np.array([[border_value, border_value, border_value, border_value, border_value, border_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, border_value, border_value, border_value, border_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, center_value, center_value, center_value, center_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, center_value, center_value, center_value, center_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, center_value, center_value, center_value, center_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, center_value, center_value, center_value, center_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, border_value, border_value, border_value, border_value, border_value,
    #                  border_value],
    #                 [border_value, border_value, border_value, border_value, border_value, border_value, border_value,
    #                  border_value]])
    mat = np.ones(sensor_resolution) * center_value
    mat[:edge_resolution[0], :] = border_value
    mat[sensor_resolution[0]-edge_resolution[0]:, :] = border_value
    mat[:, :edge_resolution[1]] = border_value
    mat[:, sensor_resolution[1]-edge_resolution[1]:0] = border_value
    return mat

def mqtt_receiver(device_name=None, address=None, port=None):
    global t_latest, i
    global data_queue
    global client_global
    global mqtt_id, mqtt_topic_in, mqtt_topic_out, mqtt_address, mqtt_port, mqtt_qos, mqtt_keepalive, mqtt_topic_status, mqtt_username, mqtt_password
    global alt_hr, receiving_mqtt, allow_reverse_time, msg_convert, buffer_frames
    t_latest = 0
    i = 0
    mqtt_id = device_name
    receiving_mqtt = True
    buffer_frames = []

    # global time_now_mqtt_receiver, time_prev_mqtt_receiver, list_delta_time_mqtt_receiver
    # time_now_mqtt_receiver = time.time()
    # time_prev_mqtt_receiver = time_now_mqtt_receiver
    # list_delta_time_mqtt_receiver = []

    def on_subscribe(client, userdata, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(client, userdata, msg):
        global t_latest, i
        global data_queue
        global alt_hr, receiving_mqtt, allow_reverse_time, reformat_msg, buffer_frames
        # global time_now_mqtt_receiver, time_prev_mqtt_receiver, list_delta_time_mqtt_receiver
        # print(json.loads(msg.payload))

        machine_time_gmt_hr = int(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())[11:13])
        if alt_hr == 0:
            # receive msg in all hours
            receiving_mqtt = True
        elif alt_hr == 1:
            # receive msg only in odd hours
            if machine_time_gmt_hr % 2 == 1:
                receiving_mqtt = True
            else:
                receiving_mqtt = False
        elif alt_hr == 2:
            # receive msg only in even hours
            if machine_time_gmt_hr % 2 == 0:
                receiving_mqtt = True
            else:
                receiving_mqtt = False
        else:
            print(f'ERROR: alt_hr = {alt_hr}; alt_hr should either be 0, 1 or 2!!!')
            quit()
        # print(f'DEBUG: alt_hr = {alt_hr}; machine_time_gmt_hr = {machine_time_gmt_hr}; receiving_mqtt = {receiving_mqtt}')

        if receiving_mqtt:
            try:
                skip = False
                if msg_convert == 'wiwideOper':
                    msg_payload_converted = str(msg.payload).replace('\'', '\"')[2:-1]
                    # print(f'msg_payload_reformat: {msg_payload_reformat}')
                    data_json_pre_buffer = json.loads(msg_payload_converted)
                elif msg_convert == 'seamless' or 'seamless64':
                    #print(msg.payload)
                    data: dict = eval(str(msg.payload).replace('\n', '')[2:-1])
                    fields: dict | int = data.get('fields', 0)
                    if fields != 0:
                        data_inside = fields.get('data', 0)
                        if data_inside != 0:
                            if msg_convert == 'seamless':
                                readings = data_inside[6:70]
                            elif msg_convert == 'seamless64':
                                readings = data_inside
                            data_json_pre_buffer = {'data': readings,
                                       'deviceName': fields['macAddress'],
                                       'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                                       'timestamp': data['timestamp'], 
                                       'utcSecs': fields['utcSecs'], 
                                       'utcUsecs': fields['utcUsecs']}
                        else:
                            skip = True
                    else:
                        skip = True
                elif msg_convert == 'butlrOfficeMeetingRoom':
                    # print(f'str(msg.payload): {str(msg.payload)}')
                    data: dict = eval(str(msg.payload)[2:-1])
                    data_json_pre_buffer = {'data': data['p'],
                               'deviceName': str(data['id']),
                               'thermistor': data['t'],
                               'timestamp': int(date_time_str_to_epoch_sec_2(data['time'])*1000), 
                               'frame': data['frame']}
                    # print(f'butlrOfficeMeetingRoom data_json converted: {data_json}')
                else:
                    data_json_pre_buffer = json.loads(msg.payload)
                if not skip:
                    if data_json_pre_buffer['deviceName']==mqtt_id:
                        # update and sort buffer frames
                        buffer_frames.append(data_json_pre_buffer)
                        buffer_frames = sorted(buffer_frames, key = lambda x: x['timestamp'])
                        buffer_time_span_secs = (buffer_frames[-1]['timestamp']-buffer_frames[0]['timestamp'])/1000
                        if buffer_time_span_secs >= buffer_secs:
                            data_json = buffer_frames.pop(0)
                            delta_timestamp = (data_json['timestamp'] - t_latest) / 1000
                            # data queue
                            if allow_reverse_time:
                                data_queue.append(data_json)
                                t_latest = data_json['timestamp']
                            else:
                                if delta_timestamp > 0:
                                    data_queue.append(data_json)
                                    t_latest = data_json['timestamp']
                                else:
                                    print(f'WARNING: reversed timestamp detected! data timestamp = {data_json["timestamp"]}; delta_timestamp = {delta_timestamp:4.4f}')
                                    pass
                        else:
                            print(f'Filling the frame buffer list! buffer_time_span_secs < buffer_secs. buffer_time_span_secs: {buffer_time_span_secs}. ')
                        i += 1
            except Exception as e:
                    print('\n========================================================================')
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print('Error type: {}, happended in line {} on <mtqq_receiver>'.format(exc_type, exc_tb.tb_lineno))
                    print('Error: {}'.format(e))
                    print('========================================================================\n')


    client = paho.Client()
    if mqtt_username != 'none' and mqtt_password != 'none':
        client.username_pw_set(mqtt_username, mqtt_password)
    client_global = client
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(mqtt_address, mqtt_port)
    client.subscribe(mqtt_topic_in, qos=mqtt_qos)
    client.loop_forever()

def mqtt_pure_publisher():
    global client_global
    global mqtt_address, mqtt_port, mqtt_username, mqtt_password
    global alt_hr, receiving_mqtt, allow_reverse_time, msg_convert, buffer_frames
    client = paho.Client()
    if mqtt_username != 'none' and mqtt_password != 'none':
        client.username_pw_set(mqtt_username, mqtt_password)
    client_global = client
    client.connect(mqtt_address, mqtt_port)
    client.loop_forever()

def serial_receiver(com_name='COM4', time_out=2):
    global i
    global data_queue
    i = 0

    ## serial setup
    ser = serial.Serial(com_name, 115200, timeout=time_out)

    # loop forever
    while True:
        # read serial port
        line = ser.readline().decode("utf-8")
        # print(data)

        data_valid = True
        try:
            data_json = ast.literal_eval(data)
        except:
            data_valid = False
            print('WARNING: invalid serial line received! ')

        if data_valid:
            # print(data_json)
            # data queue
            data_queue.append(data_json)
        i += 1

def bufcount(filename):
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read  # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines

def delta_expansion(input_list):
    # maxVal = max(input_list)
    # minVal = min(input_list)
    # newMax = maxVal+100
    # newMin = minVal-100
    res = []
    # for i in range(len(input_list)):
    #     # newVal = newMin+(newMax-newMin)*(input_list[i]-minVal)/(maxVal-minVal)
    #     if i < 32*4 or i%32 > 27 or i > 32*28 or i%32 <4:
    #         res.append(0)
    #     elif input_list[i] > 24:
    #         res.append(55)
    #     else:
    #         res.append(input_list[i])

    # return res
    for i in range(len(input_list)):
        # newVal = newMin+(newMax-newMin)*(input_list[i]-minVal)/(maxVal-minVal)
        # if i < 32*4 or i%32 > 27 or i > 32*28 or i%32 <4:
        #     res.append(0)
        if input_list[i] > 25:
            res.append(55)
        else:
            res.append(input_list[i])

    return res

def synthetic_data_reader_one(data_path=None, sleep=0.0975):
    global data_queue, play_speed, SAVE_DATA_OVER
    SAVE_DATA_OVER = False
    try:
        f = open(data_path, 'r')
    except:
        print('Invalid data path')
        return
    while True:
        if len(data_queue) > 10000:
            pass
        this_line = f.readline()
        if this_line == '':
            f.close()
            f = open(data_path, 'r')
        else:
            crt_data = eval(this_line)
            if 'p' in crt_data and 'data' not in crt_data:
                temp = np.array(crt_data.pop('p'))/4
                # crt_data['data'] = temp.tolist()
                crt_data['data'] = delta_expansion(temp.tolist())
            if 'id'in crt_data and 'device_id' not in crt_data:
                crt_data['device_id'] = crt_data.pop('id')
            # try:
            #     timestamp =  int(crt_data['t'])
            # except:
            #     timestamp = int(time.time() * 1000)
            if 't' in crt_data and 'timestamp' not in crt_data:
                crt_data['timestamp'] = crt_data.pop('t')
            data_queue.append(crt_data)
            time.sleep(sleep / play_speed)


def synthetic_data_reader(data_folder=None, sleep=0.0975, dtype='m'):
    global data_queue, play_speed, SAVE_DATA_OVER
    SAVE_DATA_OVER = False
    if not (os.path.exists(data_folder) and os.path.isdir(data_folder)):
        print(os.path.abspath(data_folder))
        print('Invalid data folder')
        return
    if dtype == 'm':
        files = [fname for fname in os.listdir(data_folder) if fname.endswith('.json')]
        files.sort()
        f_idx = 0
        while True:
            if len(data_queue) > 10000:
                continue
            if f_idx >= len(files):
                f_idx = 0
                # SAVE_DATA_OVER = True
                # continue
            crt_fpath = os.path.join(data_folder, files[f_idx])
            crt_data = json.load(open(crt_fpath, 'r'))
            if 'p' in crt_data and 'data' not in crt_data:
                # crt_data['data'] = crt_data.pop('p')
                crt_data['data'] = delta_expansion(crt_data.pop('p'))
        
            ## fake timestamp, subject to data format
            try:
                timestamp =  int(files[f_idx].split('.')[0].split('_')[-1])
            except:
                timestamp = int(time.time() * 1000)
            crt_data['timestamp'] = timestamp
            data_queue.append(crt_data)
            time.sleep(sleep / play_speed)
            f_idx += 1
    elif dtype == 's':
        file = [fname for fname in os.listdir(data_folder) if fname.endswith('.json')][0]
        data = json.load(open(os.path.join(data_folder, file), 'r'))
        data = sorted(data, key=lambda x: x['timestamp'])
        # print('data len = {}'.format(len(data)))
        data_idx = 0
        while True:
            if len(data_queue) > 10000:
                continue
            if data_idx >= len(data):
                data_idx = 0
            crt_data = data[data_idx]
            # print('\n\n')
            # print(crt_data)
            # crt_data['data'] = crt_data['data'][6:70]
            crt_data_new = {'data': crt_data['data'][6:70].copy(), 'timestamp': crt_data['timestamp']}  #'macAddress'
            # print(crt_data)
            data_queue.append(crt_data_new)
            time.sleep(sleep / play_speed)
            data_idx += 1

def saved_data_reader(data_path=None, device_name=None, start_epoch=0, sleep=0.0):  # sleep=0.0975
    global t_latest
    global data_queue
    global data_start_time_epoch, data_start_time_str, data_end_time_epoch, data_end_time_str, playback_slider_value, read_restart, is_paused
    global mqtt_id, is_exiting, batch_eval, msg_convert
    global play_speed, allow_reverse_time, buffer_secs
    i = 0
    src_file = data_path
    shape = (8, 8)
    mqtt_id = device_name

    print(f'start_epoch = {start_epoch}')

    file_line_cnt = bufcount(data_path)
    print(f'file_line_cnt = {file_line_cnt}')

    if path.isfile(src_file):

        # get last valid line's time, optimized for large file
        with open(src_file, 'rb') as file:
            # get last line's time
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
            last_line = file.readline().decode()
            if msg_convert == 'seamless' or msg_convert == 'seamless64':
                data: dict = ast.literal_eval(last_line)
                fields: dict | int = data.get('fields', 0)
                if fields != 0:
                    data_inside = fields.get('data', 0)
                    if data_inside != 0:
                        if msg_convert == 'seamless':
                            readings = data_inside[6:70]
                        elif msg_convert == 'seamless64':
                            readings = data_inside
                        data_json = {'data': readings,
                            'deviceName': fields['macAddress'],
                            'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                            'timestamp': data['timestamp'],
                            'utcSecs': fields['utcSecs'],
                            'utcUsecs': fields['utcUsecs']}
            else:
                data_json = ast.literal_eval(last_line)
            data_end_time_epoch = round(data_json['timestamp'] / 1000)
            # print(f'data end epoch: {data_end_time_epoch}')
            data_end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data_end_time_epoch))
            print(f'data end time GMT: {data_end_time_str}')

        # get first valid line's time
        with open(src_file, 'r') as file:
            line = file.readline().strip()
            while line:
                if line != '\n' and line != '':
                    if msg_convert == 'seamless' or msg_convert == 'seamless64':
                        data: dict = ast.literal_eval(line)
                        fields: dict | int = data.get('fields', 0)
                        if fields != 0:
                            data_inside = fields.get('data', 0)
                            if data_inside != 0:
                                if msg_convert == 'seamless':
                                    readings = data_inside[6:70]
                                elif msg_convert == 'seamless64':
                                    readings = data_inside
                                data_json = {'data': readings,
                                    'deviceName': fields['macAddress'],
                                    'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                                    'timestamp': data['timestamp'],
                                    'utcSecs': fields['utcSecs'],
                                    'utcUsecs': fields['utcUsecs']}
                    else:
                        data_json = ast.literal_eval(line)
                    if data_json['deviceName'] == device_name:
                        data_start_time_epoch = round(data_json['timestamp'] / 1000)
                        # print(f'data start epoch: {data_start_time_epoch}')
                        data_start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data_start_time_epoch))
                        print(f'data start time GMT: {data_start_time_str}')
                        break
                line = file.readline()

        # initial restart_slider_value
        restart_slider_value = (start_epoch - data_start_time_epoch) / (data_end_time_epoch - data_start_time_epoch)

        if batch_eval:
            # no loop, exit when finish
            t_latest = 0
            fp = open(src_file)
            buffer_frames = []
            for i, line in enumerate(fp):
                if i >= int(file_line_cnt * restart_slider_value):
                    if msg_convert == 'seamless' or msg_convert == 'seamless64':
                        data: dict = ast.literal_eval(line)
                        fields: dict | int = data.get('fields', 0)
                        if fields != 0:
                            data_inside = fields.get('data', 0)
                            if data_inside != 0:
                                if msg_convert == 'seamless':
                                    readings = data_inside[6:70]
                                elif msg_convert == 'seamless64':
                                    readings = data_inside
                                data_json_pre_buffer = {'data': readings,
                                    'deviceName': fields['macAddress'],
                                    'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                                    'timestamp': data['timestamp'],
                                    'utcSecs': fields['utcSecs'],
                                    'utcUsecs': fields['utcUsecs']}
                    else:
                        data_json_pre_buffer = ast.literal_eval(line)
                    if data_json_pre_buffer['deviceName'] == mqtt_id:
                        # update and sort buffer frames
                        buffer_frames.append(data_json_pre_buffer)
                        buffer_frames = sorted(buffer_frames, key = lambda i: i['timestamp'])
                        buffer_time_span_secs = (buffer_frames[-1]['timestamp']-buffer_frames[0]['timestamp'])/1000
                        if buffer_time_span_secs >= buffer_secs:
                            data_json = buffer_frames.pop(0)
                            delta_timestamp = (data_json['timestamp'] - t_latest) / 1000
                            # data queue
                            if allow_reverse_time:
                                data_queue.append(data_json)
                                # sleep to mimic mqtt fps
                                time.sleep(sleep)
                                t_latest = data_json['timestamp']
                            else:
                                if delta_timestamp > 0:
                                    data_queue.append(data_json)
                                    t_latest = data_json['timestamp']
                                else:
                                    print(f'WARNING: reversed timestamp detected! data timestamp = {data_json["timestamp"]}; delta_timestamp = {delta_timestamp:4.4f}')
                                    pass
                                # sleep to mimic mqtt fps
                                time.sleep(sleep)
                        else:
                            print(f'Filling the frame buffer list! buffer_time_span_secs < buffer_secs. buffer_time_span_secs: {buffer_time_span_secs}. ')
                if read_restart:
                    restart_slider_value = playback_slider_value
                    break
                # while is_paused:
                #     time.sleep(sleep)
            fp.close()
            # is_exiting = True
            read_restart = False
        # if not batch_eval
        else:
            # loop forever
            while True:
                t_latest = 0
                fp = open(src_file)
                buffer_frames = []
                for i, line in enumerate(fp):
                    if i >= int(file_line_cnt * restart_slider_value):
                        if msg_convert == 'seamless' or msg_convert == 'seamless64':
                            data: dict = ast.literal_eval(line)
                            fields: dict | int = data.get('fields', 0)
                            if fields != 0:
                                data_inside = fields.get('data', 0)
                                if data_inside != 0:
                                    if msg_convert == 'seamless':
                                        readings = data_inside[6:70]
                                    elif msg_convert == 'seamless64':
                                        readings = data_inside
                                    data_json_pre_buffer = {'data': readings,
                                        'deviceName': fields['macAddress'],
                                        'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                                        'timestamp': data['timestamp'],
                                        'utcSecs': fields['utcSecs'],
                                        'utcUsecs': fields['utcUsecs']}
                        else:
                            data_json_pre_buffer = ast.literal_eval(line)
                        if data_json_pre_buffer['deviceName'] == mqtt_id:
                            # update and sort buffer frames
                            buffer_frames.append(data_json_pre_buffer)
                            buffer_frames = sorted(buffer_frames, key = lambda i: i['timestamp'])
                            buffer_time_span_secs = (buffer_frames[-1]['timestamp']-buffer_frames[0]['timestamp'])/1000
                            if buffer_time_span_secs >= buffer_secs:
                                data_json = buffer_frames.pop(0)
                                delta_timestamp = (data_json['timestamp'] - t_latest) / 1000
                                # data queue
                                if allow_reverse_time:
                                    data_queue.append(data_json)
                                    # sleep to mimic mqtt fps
                                    time.sleep(sleep / play_speed)
                                    t_latest = data_json['timestamp']
                                else:
                                    if delta_timestamp > 0:
                                        data_queue.append(data_json)
                                        t_latest = data_json['timestamp']
                                    else:
                                        print(f'WARNING: reversed timestamp detected! data timestamp = {data_json["timestamp"]}; delta_timestamp = {delta_timestamp:4.4f}')
                                        pass
                                    # sleep to mimic mqtt fps
                                    time.sleep(sleep / play_speed)
                            else:
                                print(f'Filling the frame buffer list! buffer_time_span_secs < buffer_secs. buffer_time_span_secs: {buffer_time_span_secs}. ')
                    if read_restart:
                        restart_slider_value = playback_slider_value
                        break
                    # while is_paused:
                    #     time.sleep(sleep)
                fp.close()
                read_restart = False

    else:
        print('The src file does not exist')
        exit()

def update_adaptive_min_max(i, window_mats, window_flags, window_num_frame=150, frame_interval=5, min_percentile=1,
                            max_percentile=100, min_delta=10, init_minV=5, init_maxV=25):
    # global min_max_mats_window
    global minV, maxV
    global mat_raw
    global flag_mat_updated_mm
    global flag_detection_mm
    global flag_long_trace_mm
    global flag_valid_trace_mm
    global flag_min_max_updated
    global enable_tracking

    # print('mat_updated')
    if i % frame_interval == 0:
        # print('mat not skipped')
        # update window - use valid trace flag if tracking is enabled, otherwise use detection flag
        window_mats.append(mat_raw.copy())
        if enable_tracking:
            window_flags.append(flag_valid_trace_mm)
        else:
            window_flags.append(flag_detection_mm)
        if len(window_mats) > window_num_frame / frame_interval:
            # print('window size max')
            window_mats.popleft()
            window_flags.popleft()
        # 10th frame of initiation, or ppl detected in the window
        if i == 10 or True in window_flags:
            minV_t = round(np.percentile(window_mats, min_percentile, interpolation='linear') - 2, 2)
            maxV_t = round(np.percentile(window_mats, max_percentile, interpolation='linear') + 5, 2)
            # delta of min max shouldn't be too small, or algo will be too sensitive
            maxV_t = max(minV_t + min_delta, maxV_t)
            # minV should not be smaller than -5, maxV should not be larger than 45
            if minV_t >= -5 and maxV_t <= 55:
                minV = minV_t
                maxV = maxV_t
                flag_min_max_updated = True
            else:
                print(f'WARNING: min max not updated due to out of normal range (-5, 55)! minV_t = {minV_t}; maxV_t = {maxV_t}. ')

        # print(f'minV, maxV = {minV}, {maxV}')
    else:
        # print('mat skipped')
        # skip interval mats
        pass
    flag_detection_mm = False
    flag_valid_trace_mm = False
    flag_long_trace_mm = False

def update_adaptive_bg(i, i_bg, bg_mats_raw, window_flags_abg, reset_frames=5000, bg_frame_num=10):
    global bg_mat_raw, bg_mat
    global minV, maxV
    global mat, mat_raw
    global flag_mat_updated_bg
    global flag_detection_bg
    global flag_long_trace_bg
    global flag_valid_trace_bg
    global flag_min_max_updated
    global flag_bg_mat_raw_defined
    # global flag_force_bg_reset
    global normalize, norm_scale

    global printing_run_time, debug_run_time
    global bg_by_flag, bg_flag_list, flag_wake

    global adaptive_bg
    global bg_mean_normalized_list
    global enable_tracking

    # use a try-except block to find errors
    try:
        ## adaptive bg
        if printing_run_time and debug_run_time: print('i_bg = ', i_bg)

        # keep a detection check sliding window that is <prev_frame_num> frames long
        window_flags_abg.append(flag_detection_bg)
        if len(window_flags_abg) > prev_frame_num:
            window_flags_abg.popleft()

        # if recording background
        if i_bg < bg_frame_num and i > bg_frame_num:
            set_bg_mat(i_bg, bg_frame_num, bg_mats_raw, window_flags_abg)

        if flag_min_max_updated and normalize and flag_bg_mat_raw_defined:
            bg_mat = norm_scale * (bg_mat_raw.copy() - minV) / (maxV - minV)
            flag_min_max_updated = False
        
        flag_mat_updated_bg = False
        flag_detection_bg = False
        flag_valid_trace_bg = False

        # original reset bg logic: when regular period reached or long trace detected
        if adaptive_bg in [1, 2, 3]:
            # if i_bg == reset_frames - 1 or flag_long_trace_bg or i in [200, 500, 1000]:
            if i_bg == reset_frames - 1 or flag_long_trace_bg:
                # print('reset i_bg = 0')
                i_bg = -1
                flag_long_trace_bg = False

        ## bg by flag
        if bg_by_flag:
            if flag_wake == 0:
                print(f'flag==0 detected')
                if normalize:
                    bg_flag_list.append(norm_scale * (mat_raw.copy() - minV) / (maxV - minV))
                else:
                    bg_flag_list.append(mat_raw.copy())
                # print(f'len(bg_flag_list): {len(bg_flag_list)}')
                if len(bg_flag_list) > bg_frame_num:
                    bg_flag_list.pop(0)
                    # print(f'len(bg_flag_list): {len(bg_flag_list)}')
                if len(bg_flag_list) >= 5:
                    bg_flag_mat = np.mean(np.array(bg_flag_list), axis=0)
                    bg_mat = bg_flag_mat
                    print(f'len(bg_flag_list) >= 5! bg_flag_mat and bg_mat updated!!! len(bg_flag_list): {len(bg_flag_list)}')

        ## bg mean normalized - recording X number or normalized frames and use their mean as bg
        if adaptive_bg == 2:
            bg_mean_normalized_list.append(mat.copy())
            if len(bg_mean_normalized_list) > bg_frame_num:
                bg_mean_normalized_list.pop(0)
            if len(bg_mean_normalized_list) >= bg_frame_num:
                bg_mat = np.mean(np.array(bg_mean_normalized_list), axis=0)
                # print(f'len(bg_mean_normalized_list) >= 1000! bg_mat updated!!! len(bg_mean_normalized_list): {len(bg_mean_normalized_list)}')
            else:
                print(f'len(bg_mean_normalized_list) < bg_frame_num: {bg_frame_num}! filling the list! bg_mat not updated!!! len(bg_mean_normalized_list): {len(bg_mean_normalized_list)}')

        return i_bg

    except Exception as e:
        print('\n========================================================================')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error type: {}, happended in line {} on <update_adaptive_bg>'.format(exc_type, exc_tb.tb_lineno))
        print('Error: {}'.format(e))
        print('========================================================================\n')


def set_bg_mat(i_bg, frame_num, bg_mats_raw, window_flags_abg=deque([])):
    global mat_raw, bg_mat_raw, bg_mat
    global minV, maxV, normalize, norm_scale
    global adaptive_bg, flag_bg_mat_raw_defined

    # TO DO: stop update and delay regular check if there's detection during bg reset

    if not adaptive_bg in [3] or not flag_bg_mat_raw_defined:
        if i_bg == 0:
            bg_mats_raw.clear()
        bg_mats_raw.append(mat_raw.copy())
        if i_bg == frame_num - 1:
            bg_mat_raw = np.mean(np.array(bg_mats_raw), axis=0)
            if normalize:
                bg_mat = norm_scale * (bg_mat_raw.copy() - minV) / (maxV - minV)
            else:
                bg_mat = bg_mat_raw.copy()
            print('background has been reset')
    
    # if adaptive_bg in [3]
    else:
        if i_bg == 0:
            bg_mats_raw.clear()
        bg_mats_raw.append(mat_raw.copy())
        if i_bg == frame_num - 1:
            if not True in window_flags_abg:
                bg_mat_raw = np.mean(np.array(bg_mats_raw), axis=0)
                if normalize:
                    bg_mat = norm_scale * (bg_mat_raw.copy() - minV) / (maxV - minV)
                else:
                    bg_mat = bg_mat_raw.copy()
                print('background has been reset')
            # if detection in the window
            else:
                print('DEBUG: *** background reset has been skipped due to detection in the window ***')


def main():
    # ===================================================
    # Define paths of json files and ground truth video
    # ===================================================
    np.set_printoptions(threshold=np.inf)

    global version
    version = 'v020084_201231'

    yn_dict = {'y': True, 'n': False}

    global data_folder
    # data_folder = r'C:\Users\RYAN\Google Drive\000_Butlr\191221_Ziran PoC Test Data\data dump\4_01_06\IR_Sensor_Data_Frames_3'

    global videoPath

    global label_src_files
    label_src_files = []

    global viz_detection
    viz_detection = True

    global viz_video
    viz_video = False

    global viz_label
    viz_label = False
    global label_folder
    label_folder = None

    global save_label
    save_label = False
    global label_src_path
    label_src_path = None
    global label_dst_path
    label_dst_path = None

    global viz_tracking
    viz_tracking = True
    global track_post_combine
    track_post_combine = True
    global tracking_post_remove
    tracking_post_remove = True
    global tracking_extend_to_border
    # tracking_extend_to_border = True
    global tracking_disappear_delay
    global display_trace
    # display_trace = True

    global sample_track_info
    # sample_track_path = r'C:\Friends\butlr\Butlr_POC-master\Butlr_POC-master\Data_Management\Track Data Example.json'
    sample_track_path = None
    # sample_track_info = json.load(open(sample_track_path, 'r'))

    global viz_counting
    viz_counting = False
    global door_line_start
    global door_line_end
    global display_count
    # display_count = True

    global auto_disp_trace
    # auto_disp_trace = False
    global th_num_clear_frame
    th_num_clear_frame = 30

    global is_paused
    is_paused = False

    global pg

    global logo_img, logo
    logo_img = 'logo/butlr.logo.png'

    global ppl_count_global
    ppl_count_global = 0
    global ppl_count_in_global
    ppl_count_in_global = 0
    global ppl_count_out_global
    ppl_count_out_global = 0

    global nn_prediction_json
    # nn_prediction_json = r'C:\Users\RYAN\Google Drive\000_Butlr\191221_Ziran PoC Test Data\data dump\1\nn_prediction.json'
    nn_prediction_json = None

    global eval_data_info
    # eval_data_info = '../Evaluation/evaluation_data_info_200211.json'

    global eval_result
    eval_result = []

    # for config.json
    global config, run_mode, normalize, counting_line_key, detection_method, minV, maxV, drange, delta_slope_th, delta_r2_th, bg_r2_mat
    global nn_model, mem_clr_num_frames, fps, fullscreen, step, remove_hot_object, hot_obj_temp_th, hot_obj_slope_th, momentum_eta
    global count_labeling, count_label_path, count_label, count_eval
    global count_predict, count_predict_tmp
    global uncondintionRules, conditionRules
    global gap_dict_combine, coordScale_combine, step_th_remove, extendPara, gap_dict_counting, trackingPara, diameter_th
    global max_split_time

    # hack 200206
    global disable_lower_count, disable_upper_count, disable_in_count, disable_out_count
    disable_lower_count = False
    disable_upper_count = False
    disable_in_count = False
    disable_out_count = False

    global scene_filenames
    # scene_filenames = ['sensor_log_23','sensor_log_24']
    scene_filenames = ['sensor_log_01', 'sensor_log_02', 'sensor_log_03', 'sensor_log_04', 'sensor_log_05',
                       'sensor_log_06', 'sensor_log_07', 'sensor_log_08', 'sensor_log_09', 'sensor_log_10',
                       'sensor_log_11', 'sensor_log_12', 'sensor_log_13', 'sensor_log_14', 'sensor_log_15',
                       'sensor_log_16', 'sensor_log_17', 'sensor_log_18', 'sensor_log_19', 'sensor_log_20',
                       'sensor_log_21',
                       'sensor_log_22', 'sensor_log_23', 'sensor_log_24']

    global printAuxFlag
    printAuxFlag = True

    global data_recording
    data_recording = False

    global data_queue
    data_queue = deque([])

    global adaptive_min_max, window_num_frame, frame_interval, init_minV, init_maxV
    global flag_mat_updated_mm, flag_mat_updated_bg
    flag_mat_updated_mm = flag_mat_updated_bg = False
    global mat_raw
    mat_raw = []
    global flag_detection_mm, flag_detection_bg
    flag_detection_mm = flag_detection_bg = False
    global flag_long_trace_mm, flag_long_trace_bg
    flag_long_trace_mm = flag_long_trace_bg = False
    global flag_valid_trace_mm, flag_valid_trace_bg
    flag_valid_trace_mm = flag_valid_trace_bg = False
    global flag_min_max_updated
    flag_min_max_updated = False

    global is_exiting
    is_exiting = False

    global adaptive_bg, reset_frames, bg_frame_num
    global bg_mat, bg_slope_mat
    # bg_mat = np.zeros((8, 8))
    # bg_slope_mat = np.zeros((8, 8))
    # rz test
    global bg_slope_mat_raw
    # bg_slope_mat_raw = np.zeros((8, 8))
    # global detect_th_auto_flag, detect_th_auto_value
    # detect_th_auto_flag, detect_th_auto_value = False, None

    global data_start_time_epoch, data_start_time_str, data_end_time_epoch, data_end_time_str, playback_slider_value, read_restart
    read_restart = False

    global printing_run_time, debug_run_time
    printing_run_time = False

    global play_speed

    global bg_flag_list, bg_mean_normalized_list
    bg_flag_list = []
    bg_mean_normalized_list = []

    global use_saved_bg, saved_bg
    use_saved_bg = False
    saved_bg = None

    # argument parser
    global mqtt_id, viz, publish, publish_for_activity_map, data_scale, long_trace_num, valid_trace_num, norm_scale, delay_prevention, alt_hr, debug_run_time
    global mqtt_topic_in, mqtt_topic_out, mqtt_topic_trace, mqtt_topic_activity_map
    global mqtt_address, mqtt_port, mqtt_qos, mqtt_keepalive, mqtt_topic_status, mqtt_username, mqtt_password, batch_eval, min_max_delta_limit
    global waking_filtering, play_speed_lower_bound, play_speed_upper_bound, allow_reverse_time, msg_convert
    global detection_write_file, detection_write_path, detection_to_mqtt, mqtt_topic_detection, detection_send_empty_list, world_coord, spatial_config_path, detection_fp_filter
    global world_coord_rotation, world_coord_flip_x, world_coord_flip_y, enable_tracking, zero_noise_filter
    global waking_frame_num, horizontal_weights, buffer_secs, bg_by_flag, waking_trigger, waking_trigger_sec
    global bg_initiation, bg_ini_file_path, prev_frame_num, delay_frame_num
    global sensitivity, sensitivity_bounds
    global save_trace_file_path, saved_activity_map_file_path, AM_FROM_MQTT, AM_START_TIME, activity_map_wait_time, activity_change_status_threshold
    global enable_black_list_map
    global bb_save
    AM_FROM_MQTT = None
    global CHECK_FUTON_METHOD, ENABLE_FUTON_EFFECT_DETECTION


    # ===================================================
    # Loading Parameters from Config File
    # ===================================================
    config = config_butlr_team.config
    rules_str = rules.rules

    if 'run_mode' not in config:
        run_mode = 'saved_data'
    else:
        run_mode = config['run_mode']

    if 'eval_data_info' not in config:
        eval_data_info = None
    else:
        eval_data_info = config['eval_data_info']

    if 'data_folder' not in config or run_mode == 'batch_evaluate':
        data_folder = None
    else:
        data_folder = config['data_folder']

    if 'com_name' not in config:
        # com_name = input('Please input com name: ')
        com_name = 'COM3'
    else:
        com_name = config['com_name']

    if 'videoPath' not in config:
        videoPath = None
    else:
        videoPath = config['videoPath']

    if 'viz_label' not in config:
        viz_label = False
    else:
        viz_label = config['viz_label']
        if viz_label:
            viz_detection = False
            viz_tracking = False
            viz_counting = False

    if 'label_folder' not in config:
        label_folder = None
    else:
        label_folder = config['label_folder']

    if 'count_labeling' not in config:
        count_labeling = False
    else:
        count_labeling = config['count_labeling']

    if 'count_eval' not in config:
        count_eval = False
    else:
        count_eval = config['count_eval']

    if 'count_label_path' not in config:
        count_label_path = None
    else:
        count_label_path = config['count_label_path']

    if 'display_trace' not in config:
        display_trace = True
    else:
        display_trace = config['display_trace']

    if 'display_count' not in config:
        display_count = True
    else:
        display_count = config['display_count']

    if 'auto_disp_trace' not in config:
        auto_disp_trace = False
    else:
        auto_disp_trace = config['auto_disp_trace']

    if 'cell_size' not in config:
        cell_size = 30
    else:
        cell_size = config['cell_size']

    if 'door_line_start' not in config:
        door_line_start = (0.501, 0.001)
        # print('door_line_start not in config file, use default value: (0.001, 0.401)' )
    else:
        door_line_start = config['door_line_start']

    if 'door_line_end' not in config:
        door_line_end = (0.501, 0.999)
        # print('door_line_end not in config file, use default value: (0.999, 0.401)' )
    else:
        door_line_end = config['door_line_end']

    if 'counting_line_key' not in config:
        counting_line_key = 'h_m'
    else:
        counting_line_key = config['counting_line_key']
    if not counting_line_key == None:
        reset_counting_line(counting_line_key)

    if 'tracking_disappear_delay' not in config:
        tracking_disappear_delay = 30  # frames
        # print('tracking_disappear_delay not in config file, use default value: 30' )
    else:
        tracking_disappear_delay = config['tracking_disappear_delay']

    if 'center_r2' not in config:
        center_r2 = 0.09
    else:
        center_r2 = config['center_r2']

    if 'border_r2' not in config:
        border_r2 = 0.03
    else:
        border_r2 = config['border_r2']

    if 'center_r2_round2' not in config:
        center_r2_round2 = 0.15
    else:
        center_r2_round2 = config['center_r2_round2']

    if 'border_r2_round2' not in config:
        border_r2_round2 = 0.05
    else:
        border_r2_round2 = config['border_r2_round2']

    if 'center_slope' not in config:
        center_slope = -1.2
    else:
        center_slope = config['center_slope']

    if 'border_slope' not in config:
        border_slope = -1.0
    else:
        border_slope = config['border_slope']

    if 'logo' not in config:
        logo = 'logo/butlr.logo_32x32.png'
    else:
        logo = config['logo']

    if 'font' not in config:
        font = 'font/Roboto-Regular.ttf'
    else:
        font = config['font']

    # if 'rules' not in config:
    #     rules = 'rules.txt'
    # else:
    #     rules = config['rules']

    if 'detection_method' not in config:
        detection_method = 'regress'
    else:
        detection_method = config['detection_method']

    if 'drange' not in config:
        drange = {'center': 3, 'edge': 2.5}
    else:
        drange = config['drange']

    if 'delta_r2_th' not in config:
        delta_r2_th = 0.0
    else:
        delta_r2_th = config['delta_r2_th']

    if 'round2' not in config:
        round2 = False
    else:
        round2 = config['round2']

    if 'bg_index_list' not in config:
        bg_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        bg_index_list = config['bg_index_list']

    if 'drange2' not in config:
        drange2 = 1.5
    else:
        drange2 = config['drange2']

    if 'delta_r2_th2' not in config:
        delta_r2_th2 = 0.0
    else:
        delta_r2_th2 = config['delta_r2_th2']

    if 'fps' not in config:
        fps = 0
    else:
        fps = config['fps']

    if 'fullscreen' not in config:
        fullscreen = False
    else:
        fullscreen = config['fullscreen']

    if 'gap_dict_combine' not in config:
        gap_dict_combine = {1: 2 * cell_size, 2: 2 * cell_size, 3: 3.5 * cell_size, 4: 4 * cell_size}
    else:
        gap_dict_combine = config['gap_dict_combine']
        gap_dict_combine = {int(key): value for key, value in gap_dict_combine.items()}

    if 'coordScale_combine' not in config:
        coordScale_combine = 8 * cell_size
    else:
        coordScale_combine = config['coordScale_combine']

    if 'step_th_remove' not in config:
        step_th_remove = 3
    else:
        step_th_remove = config['step_th_remove']

    if 'extendPara' not in config:
        extendPara = {'lastN': 18, 'max_th': 0.5, 'min_th': 0}
    else:
        extendPara = config['extendPara']

    if 'momentum_eta' not in config:
        momentum_eta = 0.9
    else:
        momentum_eta = config['momentum_eta']

    if 'gap_dict_counting' not in config:
        gap_dict_counting = {1: 2 * cell_size, 2: 2 * cell_size, 3: 3 * cell_size}
    else:
        gap_dict_counting = config['gap_dict_counting']
        gap_dict_counting = {int(key): value for key, value in gap_dict_counting.items()}

    if 'trackingPara' not in config:
        trackingPara = {'b_d': -0.8, 'b_lr': -0.2}
    else:
        trackingPara = config['trackingPara']

    if 'time_out_serial' not in config:
        time_out_serial = 2
    else:
        time_out_serial = config['time_out_serial']

    if 'min_interval_serial' not in config:
        min_interval_serial = 0.01
    else:
        min_interval_serial = config['min_interval_serial']

    if 'bg_frame_num_serial' not in config:
        bg_frame_num_serial = 10
    else:
        bg_frame_num_serial = config['bg_frame_num_serial']

    if 'minV' not in config:
        # minV = float(input('Please input the minimum value of typical temperature range for normalization: '))
        minV = None
    else:
        minV = config['minV']

    if 'maxV' not in config:
        # maxV = float(input('Please input the maximum value of typical temperature range for normalization: '))
        maxV = None
    else:
        maxV = config['maxV']

    if 'remove_hot_object' not in config:
        remove_hot_object = False
    else:
        remove_hot_object = config['remove_hot_object']

    if 'hot_obj_temp_th' not in config:
        hot_obj_temp_th = 13.5
    else:
        hot_obj_temp_th = config['hot_obj_temp_th']

    if 'hot_obj_slope_th' not in config:
        hot_obj_slope_th = 13.5
    else:
        hot_obj_slope_th = config['hot_obj_slope_th']

    if 'auto_memo_clear' not in config:
        mem_clr_num_frames = 300
    else:
        mem_clr_num_frames = config['auto_memo_clear']

    if 'diameter_th' not in config:
        diameter_th = {2: 4.5, 3: 8.5}
    else:
        diameter_th = config['diameter_th']
        diameter_th = {int(key): value for key, value in diameter_th.items()}

    if 'normalize' not in config:
        normalize = True
    else:
        normalize = config['normalize']

    if 'start_id' not in config:
        startID = 0
    else:
        startID = config['start_id']

    if 'step' not in config:
        step = False
    else:
        step = config['step']

    if 'max_split_time' not in config:
        max_split_time = np.inf
    else:
        max_split_time = config['max_split_time']

    if 'mqtt_id' not in config:
        mqtt_id = '000000000000'
    else:
        mqtt_id = config['mqtt_id']

    if 'mqtt_address' not in config:
        mqtt_address = 'ec2-54-245-187-200.us-west-2.compute.amazonaws.com'
    else:
        mqtt_address = config['mqtt_address']

    if 'mqtt_port' not in config:
        mqtt_port = 1883
    else:
        mqtt_port = config['mqtt_port']

    # ===================================================
    # Loading Parameters from Argument Parser
    # ===================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='mqtt', help='''run mode, "saved_data", "mqtt" , "serial", or "synthetic_data", default="mqtt"''')
    parser.add_argument('-dp', default=r"C:\Users\RYAN\Google Drive\01_Work\Butlr\_Data\200326_Weiduomei_Onsite_Data\20200326083211-20200326090210_data.txt", help='''saved data path, extension name needed, default="1.log"''')
    parser.add_argument('-se', default='0000000000', help='''start epoch in seconds for reading a data file, default="0000000000"''')
    parser.add_argument('-cn', default='COM3', help='''serial com name, default="COM3"''')
    parser.add_argument('-ds', default='0.501,0.001', help='''door line start, specify "x,y", seperated by comma, no space! default="0.501,0.001"''')
    parser.add_argument('-de', default='0.501,0.999', help='''door line end, specify "x,y", seperated by comma, no space! default="0.501,0.999"''')
    parser.add_argument('-sc', default='0.25', help='''data scale, default="0.25"''')
    parser.add_argument('-cr2', default='0.03', help='''center r2, default="0.09"''')
    parser.add_argument('-br2', default='0.09', help='''border r2, default="0.03"''')
    parser.add_argument('-csh', default='-1.0', help='''center slope for hot person, default="-1.2"''')
    parser.add_argument('-bsh', default='-1.0', help='''border slope for hot person, default="-1.0"''')
    parser.add_argument('-csc', default='0.6', help='''center slope for cold person, default="0.8"''')
    parser.add_argument('-bsc', default='0.6', help='''border slope for cold person, default="0.8"''')
    parser.add_argument('-dmh', default='regress', help='''detecting method for hot person, "regress" or "th", default="regress"''')
    parser.add_argument('-dmc', default='th', help='''detecting method for hot person, regress/th, default=th''')
    parser.add_argument('-dr', default='3,2.5', help='''drange, specify "center_drange,border_drange", seperated by comma, no space! default="3,2.5"''')
    parser.add_argument('-acm', default='300', help='''auto clear memory, default=300''')
    parser.add_argument('-dth', default='4.5,8.5', help='''diameter threshold for polygon split, specify "threshold for 2 subpolygons, threshold for 3 subpolygons", seperated by comma, no space! default="4,8.5"''')
    parser.add_argument('-mqid', default='0242E33552C8', help='''mqtt IDs, default="000000000000"''')
    parser.add_argument('-mqti', default='test', help='''mqtt topic in, default="test"''')
    parser.add_argument('-mqto', default='butlr/mall.thermalout', help='''mqtt topic out, default="butlr/mall.thermalout"''')
    parser.add_argument('-mqts', default='butlr/status', help='''mqtt topic status, default="butlr/status"''')
    parser.add_argument('-mqad', default='ec2-54-245-187-200.us-west-2.compute.amazonaws.com', help='''mqtt address, default="ec2-54-245-187-200.us-west-2.compute.amazonaws.com"''')
    parser.add_argument('-mqpt', default='1883', help='''mqtt port, default="1883"''')
    parser.add_argument('-mqos', default='1', help='''mqtt qos, default="1"''')
    parser.add_argument('-mqka', default='60', help='''mqtt keepalive, default="60"''')
    parser.add_argument('-mqus', default='none', help='''mqtt username, default="none"''')
    parser.add_argument('-mqps', default='none', help='''mqtt password, default="none"''')
    parser.add_argument('-viz', default='t', help='''viz or not, t for local debugging, f for server, default="t"''')
    parser.add_argument('-pub', default='f', help='''publish to mqtt server or not, default="f"''')
    parser.add_argument('-n', default='t', help='''normalization, t for True, other for False, default="t"''')
    parser.add_argument('-ns', default='60', help='''normalization scale, default="20"''')
    parser.add_argument('-imin', default='18', help='''init min, default="5"''')
    parser.add_argument('-imax', default='28', help='''init max, default="25"''')
    parser.add_argument('-mmdl', default='10', help='''min max delta limit, default="10"''')
    parser.add_argument('-amm', default='t', help='''enable adaptive min max, default="t"''')
    parser.add_argument('-wamm', default='100', help='''window frame num of adaptive min max, default="100"''')
    parser.add_argument('-famm', default='5', help='''frame interval of adaptive min max, default="5"''')
    parser.add_argument('-abg', default='t', help='''0: off; 1: adaptive background, 10 frames every 5000 frames, and reset when long trace detected; 2: simply mean of normalized and wake remapped 5 minutes of frames; 3: update regularly every -rabg frames, if detection in previous -pabg frames, skip the update, default="1"''')
    parser.add_argument('-rabg', default='5000', help='''reset frame num of adaptive background, default="5000"''')
    parser.add_argument('-fabg', default='10', help='''frame interval of adaptive background, default="10"''')
    parser.add_argument('-pabg', default='5', help='''check -pabg number of previous frames if detection to determine if delay the reset bg or not, default="5"''')
    # parser.add_argument('-dabg', default='1500', help='''delay -dabg number of frames to reset the adaptive background if detection was found in previous frames, default="1500"''')
    parser.add_argument('-lt', default='100', help='''frame number to define as a long trace, default="100"''')
    parser.add_argument('-vt', default='10,50', help='''frame number range to define as a valid trace, seperated by comma, no space! default="10,50"''')
    parser.add_argument('-thh', default='auto', help='''threshold value for hot person detecting, when detect_method=th, default="auto"''')
    parser.add_argument('-thc', default='auto', help='''threshold value for cold person detecting, when detect_method=th, default="auto"''')
    parser.add_argument('-thstdh', default='7', help='''scale for std when calculating auto detecting threshold value, for hot detection, default="7"''')
    parser.add_argument('-thstdc', default='7', help='''scale for std when calculating auto detecting threshold value, for cold detection, default="7"''')
    parser.add_argument('-dprv', default='0', help='''delay prevention method, "0":no skip; "1":adaptive skip; "2": skip all, default="0"''')
    parser.add_argument('-ahr', default='0', help='''alternate hour data receiving method, "0":all hours; "1":odd hours; "2": even hours, default="0"''')
    parser.add_argument('-drt', default='f', help='''if debug runtime or not, default="f"''')
    parser.add_argument('-be', default='f', help='''batch evaluation run on saved data, then true, exit instead of loop when finished, default="f"''')
    parser.add_argument('-tshc', default='t', help='''tracking separately for hot and code objects, default="t"''')
    parser.add_argument('-cfo', default='f', help='''only use the first intersection of a trace when counting, default="f"''')
    parser.add_argument('-wk', default='0', help='''0: off; 1: apply simple filter, for sensor waking period, img temp shift, without flag; 2: apply complete filter, roll temp shift, with flag, default="0"''')
    parser.add_argument('-wktrgr', default='0', help='''0: off; 1: use delta time to trigger waking filter; 2: use flag to trigger waking filter, default="0"''')
    parser.add_argument('-wtsec', default='0.5', help='''delta time in seconds to trigger waking filter, default="0.5"''')
    parser.add_argument('-wfn', default='25', help='''waking frame number, default="25"''')
    parser.add_argument('-ps', default='1.0', help='''initial play speed, default="1.0"''')
    parser.add_argument('-psl', default='0.25', help='''play speed lower bound, default="0.25"''')
    parser.add_argument('-psu', default='5.0', help='''play speed upper bound, default="5.0"''')
    parser.add_argument('-art', default='t', help='''allow reverse time or not, default="f"''')
    parser.add_argument('-cmsg', default='none', help='''convert msg from different project to standard format, e.g.: wiwideOper, seamless, default="none"''')
    parser.add_argument('-dtcwf', default='f', help='''detection coordinates write to txt file, default="f"''')
    parser.add_argument('-dtcwp', default='detections_sensor_id.txt', help='''txt file full path to write detection coordinates, default="detections_sensor_id.txt"''')
    parser.add_argument('-dtcmq', default='f', help='''detection coordinates publish to mqtt, default="f"''')
    parser.add_argument('-dtcmqt', default='butlr/detection', help='''mqtt topic for publishing detection coordinates, default="butlr/detection"''')
    parser.add_argument('-dtcsel', default='f', help='''detection send empty list, default="f"''')
    parser.add_argument('-wldcd', default='f', help='''converting local frame coord to world coord, spatial config file needed, default="f"''')
    parser.add_argument('-scfgp', default='cfg/sensor_spatial_config.json', help='''sensor spatial config file path for local to world transfer, default="cfg/sensor_spatial_config.json"''')
    parser.add_argument('-dtcfpf', default='0', help='''detection false positive filter. 0:off; 1:simple(pervious frame not none); 2:normal(has a nearby detection in any of previous N frames), default="0"''')
    parser.add_argument('-wldrot', default='0.0', help='''rotate the world coordinates N degrees around coverage center, default="0.0"''')
    parser.add_argument('-wldflx', default='f', help='''flip the world coordinates along x axis through coverage center, default="f"''')
    parser.add_argument('-wldfly', default='f', help='''flip the world coordinates along y axis through coverage center, default="f"''')
    parser.add_argument('-trk', default='t', help='''enable tracking or not, default="t"''')
    parser.add_argument('-znf', default='f', help='''enable zero-noise filter, which replace zero-value pixels to median of all pixels, default="f"''')
    parser.add_argument('-hws', nargs='+', default=[1,1,1,1,1,1,1,1], type=float, help='''horizantal weights, default="1 1 1 1 1 1 1 1"''')
    parser.add_argument('-bfs', default=0.0, type=float, help='''buffer and sort frames for X seconds, default=0.0''')
    parser.add_argument('-dc', default='t', help='''display count, default="f"''')
    parser.add_argument('-bgflg', default='f', help='''update background when flag==0, default="f"''')
    parser.add_argument('-bgini', default='0', help='''background initiation. 0:use first fabg frames when start, 1:use saved n frames from file, default="0"''')
    parser.add_argument('-bgifp', default='data/bg_ini.txt', help='''initial background file path, default="data/bg_ini.txt"''')
    parser.add_argument('-sens', default='-1', help='''overall sensitivity. "-1": off; "0.0"-"1.0": overides -cr2, -csh, and -bsh, default="-1"''')
    parser.add_argument('-sensb', default='[0.6,0.1,1.0,0.2,-1.0,-0.2,-1.6,-0.6]', help='''sensitivity bounds. In the list: cr2_l, cr2_h, br2_l, br2_h, csh_l, csh_h, bsh_l, bsh_h, default="[0.6,0.1,1.0,0.2,-1.0,-0.2,-1.6,-0.6]"''')
    parser.add_argument("-sres", default='8*8', help='''grid resolution given by m*n, default="8*8"''')
    parser.add_argument('-eres', default='2*2', help='''edge resolution given by m*n, default="2*2"''')
    parser.add_argument('-tetb', default='t', help='''tracking extend to border, default="t"''')
    parser.add_argument('-am', default='f', help='''enable activity map, default="t"''')
    parser.add_argument('-blm', default='f', help='''enable black list map, default="f"''')
    parser.add_argument('-pubam', default='f', help='''publish to mqtt server or not, just for activity map, independent from -pub, default=t''')
    parser.add_argument('-mqttr', default=None, help='''if -pubam=True, specify mqtt topic of publishing trace for activity map''')
    # parser.add_argument('-stam', default='f', help='''whether or not to save traces to a file for activity map, default=f''')
    parser.add_argument('-stfam', default=None, help='''specify the path for saving traces for activity map, default=None, do not save''')
    parser.add_argument('-mqtam', default=None, help='''if load saved activity map from mqtt, specify the corresponding topic''')
    parser.add_argument('-samp', default=None, help='''specify the path for saved activity map, default=None''')
    parser.add_argument('-sblmp', default=None, help='''specify the path for saved black list map, default=None''')
    parser.add_argument('-amwt', default='0', help='''activity map wait time (in minutes) before applying in new detections, default=0''')
    parser.add_argument('-blmwt', default='0', help='''black list map wait time (in minutes) before applying in new detections, default=0''')
    parser.add_argument('-amcst', default='0.5', help='''activity map change status (to only applying) threshold, default=0.5''')
    parser.add_argument('-owpc', default='3', help='''output waiting for post combine steps, when a finished trace is not post-combined after these steps, it would be outputed, default=3''')
    parser.add_argument('-efed', default='f', help='''enable futon effect detection, default="t"''')
    parser.add_argument('-fecm', default='fc', help='''futon effect checking method''')
    parser.add_argument('-bbp', default='E:/work/week8/bb_data.txt', help='''the data path of saved bounding box data''')
    
    args = parser.parse_args()
    # using parser to specify parameters:
    print(f'version = {version}')
    run_mode = args.m
    print(f'run_mode = {run_mode}')
    data_path = args.dp
    data_folder = data_path
    print(f'data_path = {data_path}')
    start_epoch = int(args.se)
    print(f'start_epoch = {start_epoch}')
    com_name = args.cn
    print(f'com_name = {com_name}')
    door_line_start = tuple([float(x) for x in args.ds.split(',')])
    print(f'door_line_start = {door_line_start}')
    door_line_end = tuple([float(x) for x in args.de.split(',')])
    print(f'door_line_end = {door_line_end}')
    # cell_size = int(args.ce)
    # step = args.st == 't'
    # startID = int(args.si)
    mem_clr_num_frames = int(args.acm)
    print(f'mem_clr_num_frames = {mem_clr_num_frames}')
    center_r2, border_r2 = float(args.cr2), float(args.br2)
    print(f'center_r2 = {center_r2}')
    print(f'border_r2 = {border_r2}')
    center_slope_hot, border_slope_hot = float(args.csh), float(args.bsh)
    center_slope_cold, border_slope_cold = float(args.csc), float(args.bsc)
    print(f'center_slope_hot = {center_slope_hot}')
    print(f'border_slope_hot = {border_slope_hot}')
    print(f'center_slope_cold = {center_slope_cold}')
    print(f'border_slope_cold = {border_slope_cold}')
    detect_method_hot = args.dmh
    detect_method_cold = args.dmc
    print('Detect method: hot={}, cold={}'.format(detect_method_hot, detect_method_cold))
    dr = [float(x) for x in args.dr.split(',')]
    drange = {'center': dr[0], 'edge': dr[1]}
    print(f'drange = {drange}')
    dth = [float(x) for x in args.dth.split(',')]
    diameter_th = {2: dth[0], 3: dth[1]}
    print(f'diameter_th = {diameter_th}')
    data_scale = float(args.sc)
    print(f'data_scale = {data_scale}')
    mqtt_id = args.mqid
    print(f'mqtt_id = {mqtt_id}')
    mqtt_topic_in = args.mqti
    print(f'mqtt_topic_in = {mqtt_topic_in}')
    mqtt_topic_out = args.mqto
    print(f'mqtt_topic_out = {mqtt_topic_out}')
    mqtt_topic_status = args.mqts
    print(f'mqtt_topic_status = {mqtt_topic_status}')
    mqtt_address = args.mqad
    print(f'mqtt_address = {mqtt_address}')
    mqtt_port = int(args.mqpt)
    print(f'mqtt_port = {mqtt_port}')
    mqtt_qos = int(args.mqos)
    print(f'mqtt_qos = {mqtt_qos}')
    mqtt_keepalive = float(args.mqka)
    print(f'mqtt_keepalive = {mqtt_keepalive}')
    mqtt_username = args.mqus
    print(f'mqtt_username = {mqtt_username}')
    mqtt_password = args.mqps
    print(f'mqtt_password = {mqtt_password}')
    viz = args.viz == 't'
    print(f'viz = {viz}')
    publish = args.pub == 't'
    print(f'publish = {publish}')
    normalize = args.n == 't'
    print(f'normalize = {normalize}')
    norm_scale = float(args.ns)
    print(f'norm_scale = {norm_scale}')
    adaptive_min_max = args.amm == 't'
    print(f'adaptive_min_max = {adaptive_min_max}')
    minV = init_minV = float(args.imin)
    print(f'init_minV = {init_minV}')
    maxV = init_maxV = float(args.imax)
    print(f'init_maxV = {init_maxV}')
    min_max_delta_limit = float(args.mmdl)
    print(f'min_max_delta_limit = {min_max_delta_limit}')
    window_num_frame = int(args.wamm)
    print(f'window_num_frame = {window_num_frame}')
    frame_interval = int(args.famm)
    print(f'frame_interval = {frame_interval}')
    if args.abg == 'f': 
        adaptive_bg = 0
    elif args.abg == 't':
        adaptive_bg = 1
    else:
        adaptive_bg = int(args.abg)
    print(f'adaptive_bg = {adaptive_bg}')
    reset_frames = int(args.rabg)
    print(f'reset_frames = {reset_frames}')
    bg_frame_num = int(args.fabg)
    print(f'bg_frame_num = {bg_frame_num}')
    prev_frame_num = int(args.pabg)
    print(f'prev_frame_num = {prev_frame_num}')
    # delay_frame_num = int(args.dabg)
    # print(f'delay_frame_num = {delay_frame_num}')
    long_trace_num = int(args.lt)
    print(f'long_trace_num = {long_trace_num}')
    valid_trace_num = [float(x) for x in args.vt.split(',')]
    print(f'valid_trace_num = {valid_trace_num}')
    if args.thh == 'auto':
        th_value_hot = 'auto'
        # detect_th_auto_flag = True
    else:
        th_value_hot = float(args.thh)
    print(f'th_value_hot = {th_value_hot}')
    if args.thc == 'auto':
        th_value_cold = 'auto'
        # detect_th_auto_flag = True
    else:
        th_value_cold = float(args.thc)
    print(f'th_value_cold = {th_value_cold}')
    th_auto_std_scale_hot = float(args.thstdh)
    print(f'th_auto_std_scale_hot = {th_auto_std_scale_hot}')
    th_auto_std_scale_cold = float(args.thstdc)
    print(f'th_auto_std_scale_cold = {th_auto_std_scale_cold}')
    delay_prevention = int(args.dprv)
    print(f'delay_prevention = {delay_prevention}')
    alt_hr = int(args.ahr)
    print(f'alt_hr = {alt_hr}')
    debug_run_time = args.drt == 't'
    print(f'debug_run_time = {debug_run_time}')
    batch_eval = args.be == 't'
    print(f'batch_eval = {batch_eval}')
    tracking_separate = args.tshc == 't'
    print(f'tracking_separate = {tracking_separate}')
    counting_first_only = args.cfo == 't'
    print(f'counting_first_only = {counting_first_only}')
    if args.wk == 'f': 
        waking_filtering = 0
    elif args.wk == 't':
        waking_filtering = 1
    else:
        waking_filtering = int(args.wk)
    print(f'waking_filtering = {waking_filtering}')
    waking_trigger = int(args.wktrgr)
    print(f'waking_trigger = {waking_trigger}')
    waking_trigger_sec = float(args.wtsec)
    print(f'waking_trigger_sec = {waking_trigger_sec}')
    waking_frame_num = int(args.wfn)
    print(f'waking_frame_num = {waking_frame_num}')
    play_speed = float(args.ps)
    print(f'play_speed = {play_speed}')
    play_speed_lower_bound = float(args.psl)
    print(f'play_speed_lower_bound = {play_speed_lower_bound}')
    play_speed_upper_bound = float(args.psu)
    print(f'play_speed_upper_bound = {play_speed_upper_bound}')
    allow_reverse_time = args.art == 't'
    print(f'allow_reverse_time = {allow_reverse_time}')
    msg_convert = args.cmsg
    print(f'msg_convert = {msg_convert}')
    detection_write_file = args.dtcwf == 't'
    print(f'detection_write_file = {detection_write_file}')
    detection_write_path = args.dtcwp
    print(f'detection_write_path = {detection_write_path}')
    detection_to_mqtt = args.dtcmq == 't'
    print(f'detection_to_mqtt = {detection_to_mqtt}')
    mqtt_topic_detection = args.dtcmqt
    print(f'mqtt_topic_detection = {mqtt_topic_detection}')
    detection_send_empty_list = args.dtcsel == 't'
    print(f'detection_send_empty_list = {detection_send_empty_list}')
    world_coord = args.wldcd == 't'
    print(f'world_coord = {world_coord}')
    spatial_config_path = args.scfgp
    print(f'spatial_config_path = {spatial_config_path}')
    detection_fp_filter = int(args.dtcfpf)
    print(f'detection_fp_filter = {detection_fp_filter}')
    world_coord_rotation = float(args.wldrot)
    print(f'world_coord_rotation = {world_coord_rotation}')
    world_coord_flip_x = args.wldflx == 't'
    print(f'world_coord_flip_x = {world_coord_flip_x}')
    world_coord_flip_y = args.wldfly == 't'
    print(f'world_coord_flip_y = {world_coord_flip_y}')
    enable_tracking = args.trk == 't'
    print(f'enable_tracking = {enable_tracking}')
    zero_noise_filter = args.znf == 't'
    print(f'zero_noise_filter = {zero_noise_filter}')
    horizontal_weights = args.hws
    print(f'horizontal_weights = {horizontal_weights}')
    buffer_secs = args.bfs
    print(f'buffer_secs = {buffer_secs}')
    display_count = args.dc == 't'
    print(f'display_count = {display_count}')
    bg_by_flag = args.bgflg == 't'
    print(f'bg_by_flag = {bg_by_flag}')
    bg_initiation = int(args.bgini)
    print(f'bg_initiation = {bg_initiation}')
    bg_ini_file_path = args.bgifp
    print(f'bg_ini_file_path = {bg_ini_file_path}')
    sensitivity = float(args.sens)
    print(f'sensitivity = {sensitivity}')
    sensitivity_bounds = ast.literal_eval(args.sensb)
    print(f'sensitivity_bounds = {sensitivity_bounds}')
    bb_save = args.bbp
    print(f'bb_save = {bb_save}')
    try:
        sensor_res_m, sensor_res_n = [int(x) for x in args.sres.split('*')]
        sensor_res = (sensor_res_m, sensor_res_n)
    except:
        print("Invalid sensor resolution, using (8,8) instead")
        sensor_res = (8,8)
    print(f'sensor_res = {sensor_res}')
    try:
        edge_res_m, edge_res_n = [int(x) for x in args.eres.split('*')]
        edge_res = (edge_res_m, edge_res_n)
    except:
        print("Invalid edge resolution, using (2,2) instead")
        edge_res = (2,2)
    print(f'edge_res = {edge_res}')
    if args.tetb=='t':
        tracking_extend_to_border = True
    else:
        tracking_extend_to_border = False
    print('tracking_extend_to_border = {}'.format(tracking_extend_to_border))
    enable_activity_map = args.am=='t'
    print('enable_activity_map = {}'.format(enable_activity_map))
    enable_black_list_map = args.blm == 't'
    print('enable_black_list_map = {}'.format(enable_black_list_map))
    publish_for_activity_map = args.pubam == 't'
    print('publish_for_activity_map = {}'.format(publish_for_activity_map))
    mqtt_topic_trace = args.mqttr   # mqtt_topic_traces_for_am_out before
    print('mqtt_topic_trace = {}'.format(mqtt_topic_trace))
    save_trace_file_path = args.stfam
    print('save_trace_file_path = {}'.format(save_trace_file_path))
    mqtt_topic_activity_map = args.mqtam # mqtt_topic_activity_map_in before
    print('mqtt_topic_activity_map = {}'.format(mqtt_topic_activity_map))
    saved_activity_map_file_path = args.samp
    print('saved_activity_map_file_path = {}'.format(saved_activity_map_file_path))
    saved_black_list_map_file_path = args.sblmp
    print('saved_black_list_map_file_path = {}'.format(saved_black_list_map_file_path))
    try:
        activity_map_wait_time = int(float(args.amwt) * 60)
    except:
        activity_map_wait_time = 0
        print('Error: invalid amwt')
    print('activity_map_wait_time = {}'.format(activity_map_wait_time))
    try:
        activity_change_status_threshold = float(args.amcst)
    except:
        activity_change_status_threshold = 0.5
        print('Error: Invalid amcst')
    print('activity_change_status_threshold = {}'.format(activity_change_status_threshold))
    try:
        output_wait_for_post_combine_steps = int(args.owpc)
    except:
        output_wait_for_post_combine_steps = 3
        print('Error: Invalid owpc')
    print('output_wait_for_post_combine_steps = {}'.format(output_wait_for_post_combine_steps))
    ENABLE_FUTON_EFFECT_DETECTION = (args.efed == 't')
    print('ENABLE_FUTON_EFFECT_DETECTION = {}'.format(ENABLE_FUTON_EFFECT_DETECTION))
    if args.fecm == 'fc':
        CHECK_FUTON_METHOD = 'fixed_cell_temp_raw'
    elif args.fecm == 'str':
        CHECK_FUTON_METHOD = 'saved_temp_raw'
    elif args.fecm == 'sdt':
        CHECK_FUTON_METHOD = 'saved_delta_temp'
    print('CHECK_FUTON_METHOD = {}'.format(CHECK_FUTON_METHOD))

    bg_mat = np.zeros(sensor_res)
    bg_slope_mat = np.zeros(sensor_res)
    bg_slope_mat_raw = np.zeros(sensor_res)

    # ===================================================
    # Run !!!
    # ===================================================

    # # to disable viz for server
    # if viz:
    #     import os
    #     os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # center_border_mapping
    bg_r2_mat = center_border_mapping(center_r2, border_r2, sensor_resolution=sensor_res, edge_resolution=edge_res)
    delta_r2_th = center_border_mapping(center_r2, border_r2, sensor_resolution=sensor_res, edge_resolution=edge_res)
    bg_r2_mat_round2 = center_border_mapping(center_r2_round2, border_r2_round2, sensor_resolution=sensor_res, edge_resolution=edge_res)
    delta_slope_th_hot = center_border_mapping(center_slope_hot, border_slope_hot, sensor_resolution=sensor_res, edge_resolution=edge_res)
    delta_slope_th_cold = center_border_mapping(center_slope_cold, border_slope_cold, sensor_resolution=sensor_res, edge_resolution=edge_res)

    # initiate pygame
    title = 'Butlr PoC'
    if viz:
        pg = pygame_init(screen_width=1280, screen_height=720, fullscreen=fullscreen, title=title, logo_path=logo,
                         font_path=font)
    uncondintionRules, conditionRules = read_rules(rules_str)

    # nn
    if detection_method == 'nn':
        nn_model_path = "../Detection/nn_models/two_person_model_big_standardized.h5"
        # nn_model = load_model(nn_model_path)

    if run_mode == 'mqtt' or run_mode == 'saved_data' or run_mode == 'serial' or run_mode == 'synthetic_data' or run_mode == 'synthetic_data_one':
        # print(f'run_mode = mqtt')
        # mqtt_receiver thread
        try:
            if run_mode == 'mqtt':
                # real mqtt receiver
                _thread.start_new_thread(mqtt_receiver, (),
                                         {'device_name': mqtt_id, 'address': mqtt_address, 'port': mqtt_port})
            elif run_mode == 'saved_data':
                if batch_eval:
                    sleep = 0.0
                else:
                    sleep = 0.0975
                # hack file reader as mqtt receiver
                _thread.start_new_thread(saved_data_reader, (),
                                         {'data_path': data_path, 'device_name': mqtt_id, 'start_epoch': start_epoch,
                                          'sleep': sleep})
            elif run_mode == 'serial':
                # serial receiver
                _thread.start_new_thread(serial_receiver, (), {'com_name': com_name, 'time_out': time_out_serial})

            elif run_mode == 'synthetic_data':
                sleep = 0.0
                dtype = 'm'
                _thread.start_new_thread(synthetic_data_reader, (),
                                         {'data_folder':data_path, 'sleep':sleep, 'dtype':dtype})
            elif run_mode == 'synthetic_data_one':
                sleep = 1.0
                _thread.start_new_thread(synthetic_data_reader_one, (),
                                         {'data_path':data_path, 'sleep':sleep})
        except:
            print("Error: unable to start thread: mqtt_receiver")

        # if run_mode in ['saved_data', 'synthetic_data'] and publish_for_activity_map is True:
        #     ## in this case mqtt_receiver is not run, but we still need a client to publish tracking results
        #     _thread.start_new_thread(mqtt_pure_publisher, (), {})

        """
        In this script, we may need to publish traces and retrieve the saved latest activity map,
        but we never publish activity map or retrieve saved traces
        For publishing traces: we do nothing here, the client will do publishing in pygame_run_mqtt function, directly using .publish method;
        Since we do not need retrieve saved traces, we set 'mqtt_topic_trace_in' argument in mqtt_processes_for_am to None;
        For retriving activity map: we set 'mqtt_topic_am_in' argument in mqtt_processes_for_am to mqtt_topic_activity_map, when it is not None,
            then the client can subscribe to this topic, and my_activity_map will read the saved activity map at the beginning to init, we also 
            set 'activity_map' argument in mqtt_processes_for_am to my_activity_map, so that when we publish the latest activity map to mqtt,
            the callback function (on_message) will ask my_activity_map instance to update its activity map.
        """

        # initilize an activity map instance, it would be used as a argument in the following mqtt_processes_for_am function
        if enable_activity_map:
            # arbitrarily instantiate an activity_map instance, and then reconstruct it with params loaded from mqtt
            my_activity_map = Activity_Map(sensor_resolution=(1,1))
            my_activity_map.change_status('only updating')
            AM_START_TIME = int(time.time())
        else:
            my_activity_map = None
            AM_START_TIME = float('inf')

        # if we need to publish traces for acitivty map to mqtt, or we need to retrieve saved activity map from mqtt,
        # lauch a new mqtt client for activity map
        if publish_for_activity_map or (enable_activity_map and mqtt_topic_activity_map is not None):
            global client_for_am
            client_for_am = paho.Client()
            _thread.start_new_thread(mqtt_processes_for_am, (), {
                'client': client_for_am, 'usn': mqtt_username, 'pw': mqtt_password,
                'address': mqtt_address, 'mqtt_topic_trace_in': None,
                'mqtt_topic_am_in': mqtt_topic_activity_map,
                'mqtt_qos':mqtt_qos, 'activity_map': my_activity_map
            })
        else:
            client_for_am = None

        # Reconstrut the activity map
        if enable_activity_map:
            if mqtt_topic_activity_map is not None:
                # we do not need to explicitly ask my_activity_map to read activtity map from mqtt, since the latest activity
                # map is published to mqtt with retain=True, so it will be automatically sent to my_acitivty_map at the moment that
                # the topic is subscribed
                pass
            elif saved_activity_map_file_path is not None and os.path.exists(saved_activity_map_file_path + '_params.p'):
                my_activity_map.read_activity_map(read_from='saved_file', data_type='params', saved_file_path=saved_activity_map_file_path+'_params.p')
            else:
                print('Error: can not retrieve any saved activity map')
                return

        # # adaptive_min_max thread
        # if adaptive_min_max:
        #     try:
        #         _thread.start_new_thread(thread_adaptive_min_max, (), {'window_num_frame':window_num_frame, 'frame_interval':frame_interval, 'min_delta':min_max_delta_limit, 'init_minV':init_minV, 'init_maxV':init_maxV})
        #     except:
        #         print ("Error: unable to start thread: thread_adaptive_min_max")
        # # adaptive bg
        # if adaptive_bg:
        #     try:
        #         _thread.start_new_thread(thread_adaptive_bg, (), {'reset_frames':reset_frames, 'bg_frame_num':bg_frame_num})
        #     except:
        #         print ("Error: unable to start thread: thread_adaptive_bg")
        pygame_run_mqtt(min_interval=min_interval_serial,
                        detect_method_hot=detect_method_hot, detect_method_cold=detect_method_cold,
                        step=False, bg_frame_num=bg_frame_num, drange=drange,
                        delta_slope_th_hot=delta_slope_th_hot, delta_slope_th_cold=delta_slope_th_cold,
                        delta_r2_th=delta_r2_th, th_value_hot=th_value_hot, th_value_cold=th_value_cold,
                        th_auto_std_scale_hot=th_auto_std_scale_hot, th_auto_std_scale_cold=th_auto_std_scale_cold,
                        diff=False, bg_r2_mat=bg_r2_mat,
                        round2=False, drange2=1.5, delta_slope_th2=delta_slope_th_hot,
                        delta_r2_th2=0.0, bg_r2_mat2=bg_r2_mat_round2,
                        mem_clr_num_frames=mem_clr_num_frames,
                        fps=0, bg=pygame.Color('black'),
                        tracking_separate=tracking_separate,
                        counting_first_only=counting_first_only,
                        sensor_resolution=sensor_res, edge_resolution=edge_res,
                        activity_map=my_activity_map,
                        output_wait_for_post_combine_steps=output_wait_for_post_combine_steps)
    else:
        print('please specify correct \"run_mode\" in config json: \"serial\", \"saved_data\", or \"batch_evaluate\". ')

if __name__ == '__main__':
    main()