import re
import numpy as np
import sys
import os
from os import path, chdir
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString, Point
from scipy.spatial import ConvexHull
from shapely.ops import split
import copy
import time
from scipy import stats
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# chdir(path.dirname(sys.argv[0]))   # use relative path

global version
version = 'v020012_200516'

def run_time_utilities(func):
    def wrapper(*args, **kw):
        t1 = time.time()
        res = func(*args, **kw)
        t2 = time.time()
        print('{:4.4f} seconds elapsed for {}'.format(t2-t1, func.__name__))
        return res
    return wrapper

def tuple2list(trace_tuple):
    return [list(xy) for xy in trace_tuple]

def euclidean_distance(x, y):
    """
    Calculate euclidean distance
    Arguments:
    -------------------
    x, y: two points

    Return:
    -------------------
    a scaler: euclidean distance
    """
    x, y = np.asarray(x), np.asarray(y)
    return np.sqrt(np.sum((x - y) ** 2))

def length_ratio_angle(a, b, c):
    """
    Caclulate length ratio and turning angle
    Arguments:
    -------------------------------------
    a/b/c: (x,y) coordinations of 3 points

    Return:
    lr: length ratio 0~1, low value suggests straight way
    ang: turning angle , low value suggests straight way
    """
    ab = euclidean_distance(a, b)
    bc = euclidean_distance(b, c)
    ac = euclidean_distance(a, c)
    # print('ab={}, bc={}, ac={}'.format(ab,bc,ac))
    if ac == 0:
        ac = 1e-10
    # length ratio
    lr = (ab + bc) / ac
    # numerical problem: when ab+bc == ac, sometimes it appears that ab+bc<ac and lr=0.9999, it must be equal or greater than 1.0
    if lr < 1 and abs((ab+bc)-ac) < 0.001:
        lr = 1
        ac = ab+bc
    if lr > 0:
        lr = np.sqrt(1 - 1 / (lr ** 2))
    # angle
    if ab != 0 and bc != 0:
        # print('ab={}, bc={}, ac={}, value={}'.format(ab, bc, ac, (ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc)))
        tmp = (ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc)
        if tmp > 1: tmp = 1
        if tmp < -1: tmp = -1
        ang = np.pi - np.arccos(tmp)
        ang = ang * 180 / np.pi
    else:
        ang = 0
    return lr, ang

def std_ellipse(x, y, n_std=2):
    if len(x) == 1:
        return {'area': 0, 'ob': 0, 'main_axis_length': 0, 'minor_axis_length': 0}
    try:
        x, y = np.asarray(x), np.asarray(y)
        mx, my = np.mean(x), np.mean(y)
        x0, y0 = x-mx, y-my
        scale_dict = {1: 2.2958, 2:6.1801, 3:11.8290}
        scale = scale_dict.get(n_std, 6.1801)
        Cov = np.cov(x0, y0) * scale
        eig_result = np.linalg.eig(Cov)
        eig_values, eig_vectors = eig_result
        a, b = np.sqrt(eig_values.min()), np.sqrt(eig_values.max())
        area = a*b*np.pi
        ob = (b-a)/b
        return {'area':area, 'ob':ob, 'main_axis_length':b, 'minor_axis_length':a}
    except:
        return {'area': 0, 'ob': 0, 'main_axis_length': 0, 'minor_axis_length': 0}
        
def most_common(lst):
    return max(set(lst), key=lst.count)
    
def check_futon_effect(trace, ell_area_th=0.3, ell_ob_th=0.4, slope_th=-0.009, r2_th=0.6, 
        printFlag=False, recent_mats_raw=None, method='fixed_cell_temp_raw', window=None):
    x_coords = [x[0] for x in trace['cood']]
    y_coords = [x[1] for x in trace['cood']]
    ellipse_result = std_ellipse(x_coords, y_coords, n_std=2)
    ell_area, ell_ob = ellipse_result['area'], ellipse_result['ob']  #area and obesity of the std ellipse
    if ell_area >= ell_area_th or ell_ob >= ell_ob_th:
        # this is not a static trace, so no futon effect
        if printFlag: print('\nMoving trace, no futon effect: ell_area={}, ell_ob={}'.format(ell_area, ell_ob))
        return False
        
    if method == 'saved_temp_raw':
        value = trace['temp_raw']
        t = [x for x in range(len(value))]
    elif method == 'saved_delta_temp':
        value = trace['delta_temp']
        t = [x for x in range(len(value))]
    elif method == 'fixed_cell_temp_raw' :
        ok = True
        most_common_cell = most_common(trace['cood_grid'])
        if trace['img_names_first_last'][0] in recent_mats_raw:
            start_img_name_idx = list(recent_mats_raw.keys()).index(trace['img_names_first_last'][0])
        else:
            ok = False
        if trace['img_names_first_last'][1] in recent_mats_raw:
            end_img_name_idx = list(recent_mats_raw.keys()).index(trace['img_names_first_last'][1])
        else:
            ok = False
        if not ok:
            print('Can not fetch data, use recorded temp_raw instead')
            method = 'saved_temp_raw'
            value = trace['temp_raw']
            t = [x for x in range(len(value))]
        elif window is None:
            value = [mat_raw[most_common_cell] for mat_raw in list(recent_mats_raw.values())[start_img_name_idx : end_img_name_idx+1]]
            t = [x for x in range(len(value))]
        else:
            value, t = [], []
            
    if window is not None and type(window)==int:
        if method != 'fixed_cell_temp_raw':
            length = len(value)
            start_idx, end_idx = 0, length-1
        else:
            value = [mat_raw[most_common_cell] for mat_raw in list(recent_mats_raw.values())]
            t = [x for x in range(len(value))]
            length = len(value)
            start_idx, end_idx = start_img_name_idx, end_img_name_idx
        start_window = value[::-1][length-start_idx: length-start_idx+window][::-1] + value[start_idx: start_idx+window+1]
        end_window = value[::-1][length-end_idx: length-end_idx+window][::-1] + value[end_idx: end_idx+window+1]
        start_window_left_idx = start_idx - len(value[::-1][length-start_idx: length-start_idx+window])
        end_window_left_idx = end_idx - len(value[::-1][length-end_idx: length-end_idx+window])  
        start_max_idx = np.argmax(start_window)
        end_min_idx = np.argmin(end_window)
        # max_value, min_value = np.max(start_window), np.min(end_window)
        start_idx_use = start_window_left_idx + start_max_idx
        end_idx_use = end_window_left_idx + end_min_idx
        t, value = t[start_idx_use: end_idx_use+1], value[start_idx_use: end_idx_use+1]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, value)
    # plt.plot(t, value)
    # plt.show()
    if slope <= slope_th and r_value**2 >= r2_th:
        if printFlag: print('\nStatic trace with futon effect: ell_area={}, ell_ob={}, slope={}. r2={}'.format(ell_area, ell_ob, slope, r_value**2))
        return True
    else:
        if printFlag: print('\nStatic trace without futon effect: ell_area={}, ell_ob={}, slope={}. r2={}'.format(ell_area, ell_ob, slope, r_value**2))
        return False
        
    


def V2P(V, axis=0):
    if axis == 0:
        V = V - np.mean(V, axis=0).reshape(1,-1)
        expV = np.exp(V)
        sumExpV = np.sum(expV, axis=0).reshape(1,-1)
        P = expV / sumExpV
    elif axis == 1:
        V = V - np.mean(V,axis=1).reshape(-1,1)
        expV = np.exp(V)
        sumExpV = np.sum(expV, axis=1).reshape(-1,1)
        P = expV / sumExpV
    return P

def accept_reject_matrix(indicesMatrix, uncondintionRules, conditionRules, whyFlag=False):
    if whyFlag:
        already_explained = []
    result = np.ones_like(indicesMatrix['ED'])
    for key, value in indicesMatrix.items():
        code = '{} = value'.format(key)
        exec(code)
    for rule in uncondintionRules:
        crtResult = eval(rule)
        if whyFlag:
            crtWhere = np.where(crtResult == False)
            to_be_explained = [(row, col) for row, col in zip(crtWhere[0], crtWhere[1])]
            to_be_explained = [x for x in to_be_explained if x not in already_explained]
            for x in to_be_explained:
                # print('Pair ({}, {}) is denied for: {}'.format(x[0], x[1], rule))
                already_explained.append(x)
        result = result * crtResult

    for condition, rule in conditionRules:
        conditionList = condition.split('and')
        if len(conditionList) == 1:
            crtCondition = 1 - eval(condition)
        else:
            crtCondition = np.ones_like(indicesMatrix['ED'])
            for subCondition in conditionList:
                crtCondition = crtCondition * eval(subCondition)
            crtCondition = 1 - crtCondition
        tmp = eval(rule)
        crtResult = crtCondition + tmp > 0
        if whyFlag:
            crtWhere = np.where(crtResult == False)
            to_be_explained = [(row, col) for row, col in zip(crtWhere[0], crtWhere[1])]
            to_be_explained = [x for x in to_be_explained if x not in already_explained]
            for x in to_be_explained:
                # print('Pair ({}, {}) is denied for: {} => {}'.format(x[0], x[1], condition, rule))
                already_explained.append(x)
        result = result * crtResult
    return result

def beem_search(P, B, noneP=0.04):
    nrow, ncol = P.shape
    sequences = [[[], 1, P, False]]
    fin = False

    while fin is False:
        # a = input()
        newSequences = []
        for seq in sequences:
            if seq[-1] is True:
                seq[1] *= noneP
                newSequences.append(seq)
                continue
            beforePairs, beforePScalr, beforeP2, beforeStop = seq
            indexMaxP = np.argmax(beforeP2)
            indexCrtRow = int(indexMaxP / ncol)
            crtRow = beforeP2[indexCrtRow, :]
            indexSortCrtRow = np.argsort(crtRow)[::-1]
            for i in range(ncol):
                pairs = beforePairs.copy()
                indexCrtCol = indexSortCrtRow[i]
                if crtRow[indexCrtCol] == 0:
                    break
                pScalar = crtRow[indexCrtCol] * beforePScalr
                P2 = beforeP2.copy()
                P2[indexCrtRow, :] = 0
                P2[:, indexCrtCol] = 0
                tmp = np.unique(np.squeeze(P2.reshape(1, -1)))
                if len(tmp) == 1 and tmp[0] == 0:
                    stop = True
                else:
                    stop = False

                pairs.append((indexCrtRow, indexCrtCol))
                newSequences.append([pairs, pScalar, P2, stop])
        PScalrList = [seq[1] for seq in newSequences]
        indexSortedPScalrList = np.argsort(np.array(PScalrList))[::-1]
        crtB = min(B, len(indexSortedPScalrList))
        sequences = [newSequences[indexSortedPScalrList[i]] for i in range(crtB)]
        fin = True
        for seq in sequences:
            if seq[-1] is True:
                continue
            else:
                fin = False
                break
    if len(sequences) > 0:
        return sequences[0][0]
    else:
        return []

def read_rules(pathOrStr):
    uncondintionRules = []
    conditionRules = []
    generalPattern = re.compile('\[[0-9]+\] (.*)')
    conditionPattern = re.compile('\[[0-9]+\] if(.*)then(.*)')
    if not os.path.exists(pathOrStr):
        lines = pathOrStr.split('\n')
    else:
        with open(pathOrStr, 'r+') as f:
            lines = f.readlines()
    for line in lines:
        if not generalPattern.match(line):
            continue
        if conditionPattern.match(line):
            # condition rules
            condition, judge = conditionPattern.match(line).groups()
            conditionRules.append([condition, judge])
        else:
            # uncondintion rules
            judge = generalPattern.match(line).groups()[0]
            uncondintionRules.append(judge)
    return uncondintionRules, conditionRules

def indices_person_with_person_in_three_frames(meanPosition_a, meanPosition_b, meanPosition_c):
    """
    to calculate a number of association indices between two persons in three consecutive frames: a person in the current frame (meanPosition_b)
    with a person in the next frame (meanPosition_c), assuming his position in the last frame is known (meanPosition_a).

    Return:
    ---------------------------------------------------------------------
    indices: a dict including a number indices about association between meanPosition_b in current frame and meanPosition_c in the next frame,
             given the previous postion of meanPosition_b is given at meanPosition_a in the previous frame.
    """

    distBorder_a = min(meanPosition_a[0], 8 - meanPosition_a[0], meanPosition_a[1], 8 - meanPosition_a[1])
    distBorder_b = min(meanPosition_b[0], 8 - meanPosition_b[0], meanPosition_b[1], 8 - meanPosition_b[1])
    distBorder_c = min(meanPosition_c[0], 8 - meanPosition_c[0], meanPosition_c[1], 8 - meanPosition_c[1])

    lr, ang = length_ratio_angle(meanPosition_a, meanPosition_b, meanPosition_c)
    dist = euclidean_distance(meanPosition_b, meanPosition_c)
    distLast = euclidean_distance(meanPosition_b, meanPosition_a)

    indices = {'ED': dist, 'EDL': distLast, 'LR': lr, 'TA': ang, 'DBA': distBorder_a,
               'DBB': distBorder_b, 'DBC': distBorder_c}
    return indices

# @run_time_utilities
def indices_allpeople_with_allpeople_in_two_frames(meanPositions_b, meanPositions_c, meanPositions_a, printAuxFlag=False, distScale=8*30, b_d=-0.8, b_lr=-0.2):
    """
    to calculate association index matrices among all people in current frame and all people in the next frame.
    Returns:
    ---------------------------------------------------------------
    P: a m*n matrix measuing association with each person in the current frame and each person in the next frame. m is the number of persons
       in the current frame, and n is the number of persons in the next frame.
    indicesMatrix: a dict of matrices (all in shape m*n) of different association indices
    """
    nb, nc = len(meanPositions_b), len(meanPositions_c)
    EDMatrix, EDLMatrix, LRMatrix, TAMatrix = np.ones((nb, nc)), np.ones((nb, nc)), np.ones((nb, nc)), np.ones((nb, nc))
    DBAMatrix, DBBMatrix, DBCMatrix = np.ones((nb, nc)), np.ones((nb, nc)), np.ones((nb, nc))
    for i, meanPosition_b in enumerate(meanPositions_b):
        meanPosition_a = meanPositions_a[i]
        for j, meanPosition_c in enumerate(meanPositions_c):
            indices = indices_person_with_person_in_three_frames(meanPosition_a, meanPosition_b, meanPosition_c)
            EDMatrix[i, j] = indices['ED']
            EDLMatrix[i, j] = indices['EDL']
            LRMatrix[i, j] = indices['LR']
            TAMatrix[i, j] = indices['TA']
            DBAMatrix[i, j] = indices['DBA']
            DBBMatrix[i, j] = indices['DBB']
            DBCMatrix[i, j] = indices['DBC']
    # when a person is new-coming, he has no poseVector_a information, and would lead to a constant-zero LR and angel
    for nrow in range(LRMatrix.shape[0]):
        if np.all(LRMatrix[nrow, :] == 0) and np.all(TAMatrix[nrow, :] == 0):
            for ncol in range(LRMatrix.shape[1]):
                if np.any(LRMatrix[:, ncol] > 0):
                    LRMatrix[nrow, ncol] = np.min([x for x in LRMatrix[:, ncol] if x > 0])
                if np.any(TAMatrix[:, ncol] > 0):
                    TAMatrix[nrow, ncol] = np.min([x for x in TAMatrix[:, ncol] if x > 0])
    # if EDMatrix.shape[1] == 1:
        # NED1Matrix = np.ones((EDMatrix.shape[0], 1))
    # else:
        # NED1Matrix = (EDMatrix - np.min(EDMatrix, axis=1).reshape(-1, 1)) / (0.00000000000000000000001 + (
                    # np.max(EDMatrix, axis=1).reshape(-1, 1) - np.min(EDMatrix, axis=1).reshape(-1, 1)))
    # if EDMatrix.shape[0] == 1:
        # NED2Matrix = np.ones((1, EDMatrix.shape[1]))
    # else:
        # NED2Matrix = (EDMatrix - np.min(EDMatrix, axis=0).reshape(1, -1)) / (0.00000000000000000000001 + (
                    # np.max(EDMatrix, axis=0).reshape(1, -1) - np.min(EDMatrix, axis=0).reshape(1, -1)))

    NED1Matrix = EDMatrix / distScale
    NED2Matrix = NED1Matrix
    
    R_ED1Matrix = np.argsort(np.argsort(EDMatrix))
    R_ED2Matrix = np.argsort(np.argsort(EDMatrix, axis=0), axis=0)
    indicesMatrix = {'ED': EDMatrix, 'EDL': EDLMatrix, 'LR': LRMatrix, 'TA': TAMatrix,
                     'DBA': DBAMatrix, 'DBB': DBBMatrix, 'DBC': DBCMatrix, 'R_ED1': R_ED1Matrix, 'R_ED2': R_ED2Matrix}

    VMatrix_toChooseNext = NED1Matrix * b_d + LRMatrix * b_lr
    VMatrix_toChoosePrevious = NED2Matrix * b_d + LRMatrix * b_lr

    # these V are the min the best, so we need to use -V to calculate P
    P_toChooseNext = V2P(VMatrix_toChooseNext, axis=1)
    P_toChoosePrevious = V2P(VMatrix_toChoosePrevious, axis=0)

    # P_toChooseNext = P_toChooseNext[:, :P_toChooseNext.shape[1] - 1]
    # P_toChoosePrevious = P_toChoosePrevious[:P_toChoosePrevious.shape[0] - 1, :]

    P = P_toChooseNext * P_toChoosePrevious
    if printAuxFlag:
        print('\nDist: \n{}'.format(EDMatrix))
        print('\nNormarlized Dist: \n{}'.format(NED1Matrix))
        print('\nLength Ratio: \n{}'.format(LRMatrix))
        print('\nAngle: \n{}'.format(TAMatrix))
    return P, indicesMatrix

# @run_time_utilities
def choose_and_apply_rules(P, indicesMatrix=None, uncondintionRules=None, conditionRules=None, whyFlag=False):
    """
    to get association relations according to P matrix and rules
    Arguments:
    -----------------------------------------------------------------------
    P: association matrix
    indicesMatrix: a dict of matrices of a number of association indices
    unconditionRules, conditionRules: unconditional and conditional rules

    Returns
    ------------------------------------------------------------------------
    pairs: a list of 2-element tuples, eaching representing a person in current frame is found in the next frame.
    """
    # apply the rules
    if indicesMatrix is not None:
        satisfiyRules = accept_reject_matrix(indicesMatrix, uncondintionRules, conditionRules, whyFlag=whyFlag)
        P_afterRules = P * satisfiyRules
    else:
        P_afterRules = P
    # preprocessing P: values below row cutoff is ommited
    if P.shape[1] >= 4:
        PCut = []
        for row in range(P.shape[0]):
            crtRow = P_afterRules[row, :]
            if np.any(crtRow > 0):
                crtRowGT0 = crtRow[crtRow > 0]
                if len(crtRowGT0) == 1:
                    PCut.append(0)
                else:
                    cutoff = 0
                    sortedCrtRowGT0 = np.sort(crtRowGT0)[::-1]
                    for x in range(len(sortedCrtRowGT0) - 1):
                        if sortedCrtRowGT0[x] / sortedCrtRowGT0[x + 1] >= 2:
                            cutoff = sortedCrtRowGT0[x]
                            break
                    PCut.append(cutoff)
            else:
                PCut.append(0)
        PCut = np.array(PCut).reshape(-1, 1)
        P_filter = P_afterRules >= PCut
        P_afterRules = P_afterRules * P_filter
    pairs = beem_search(P_afterRules, 3)
    nrow, ncol = P.shape
    pairFromLastFrame = [x[0] for x in pairs]
    pairToNextFrame = [x[1] for x in pairs]
    outFromLastFrame = [x for x in range(nrow) if x not in pairFromLastFrame]
    inToNextFrame = [x for x in range(ncol) if x not in pairToNextFrame]
    return pairs, outFromLastFrame, inToNextFrame, pairFromLastFrame, pairToNextFrame, P_afterRules

def getExtrapoledLine(p1, p2, both=False):
    """
    creates a line extrapoled in p1->p2 direction
    if both=True: extrapoled to both direction
    """
    EXTRAPOL_RATIO = 100
    a = p1
    b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]))
    if both:
        a = (p1[0]+EXTRAPOL_RATIO*(p1[0]-p2[0]), p1[1]+EXTRAPOL_RATIO*(p1[1]-p2[1]))
    return LineString([a, b])

def get_perpendicular_line(pointA, pointB, splitPoint):
    # slopeAB = (pointA[1]-pointB[1]) / (pointA[0]-pointB[0])
    # find a point C with (0, ?) to make the product to 0
    AB = pointA - pointB
    if AB[1] != 0:
        deltaY = -AB[0] / AB[1]
        splitPoint2 = [splitPoint[0]+1, splitPoint[1]+deltaY]
    elif AB[1] == 0:
        splitPoint2 = [splitPoint[0], splitPoint[1]+1]
    perpendicularLine = getExtrapoledLine(splitPoint, splitPoint2, both=True)
    return perpendicularLine

def polygon_major_axis(polygon):
    """
    """
    cx, cy = polygon.centroid.x, polygon.centroid.y
    vertices_x, vertices_y = polygon.exterior.xy
    nv = len(vertices_x)
    majorAxisLength = -np.inf
    majorAxisPoint = None
    for i in range(nv-1):
        middlePoint_x, middlePoint_y = (vertices_x[i] + vertices_x[i+1])/2, (vertices_y[i] + vertices_y[i+1])/2
        line = getExtrapoledLine([middlePoint_x, middlePoint_y], [cx, cy])
        aaa = [cx, cy]
        bbb = [middlePoint_x, middlePoint_y]
        # print('\n\nLine: \n--------------centroid: {}, middle: {}\nline: {}'.format(aaa, bbb, line))
        intersection = polygon.exterior.intersection(line)
        if intersection.is_empty:
            print("[Error] shapes don't intersect")
            return
        elif intersection.geom_type.startswith('Multi') or intersection.geom_type == 'GeometryCollection':
            intersectionPoints = []
            for shp in intersection:
                intersectionPoints.append(shp)
            if len(intersectionPoints) != 2:
                print("[Error] the number of intersection points is not 2")
                return
        else:
            # plt.plot(vertices_x, vertices_y, 'r-')
            # plt.plot([line.xy[0][0], line.xy[0][1]], [line.xy[1][0], line.xy[1][1]], 'k-')
            # plt.plot([cx, intersection.x], [cy, intersection.y], 'b-')
            # plt.show()
            # print(intersection)
            intersectionPoints = [Point(middlePoint_x, middlePoint_y), Point(intersection.x, intersection.y)]
        intersectionLineString = LineString(intersectionPoints)
        dist = intersectionLineString.length
        if dist > majorAxisLength:
            majorAxisLength = dist
            majorAxisPoints = intersectionPoints
    return majorAxisPoints, majorAxisLength

def split_polygon(polygon, majorAxisPoints, npoly=2):
    """
    split a ploygon into np parts along long axis and return np polygons as well as np centroids
    """
    newPolygons = [polygon]
    splitPoints = []
    splitPerpendicularLines = []
    point1, point2 = majorAxisPoints
    point1 = np.asarray([point1.x, point1.y])
    point2 = np.asarray([point2.x, point2.y])
    direction = (point2-point1) / npoly
    for i in range(1, npoly):
        splitPoint = point1 + direction*i
        splitPoints.append(splitPoint)
        splitPerpendicularLines.append(get_perpendicular_line(point1, point2, splitPoint))
    # print(splitPoints)
    # for line in splitPerpendicularLines:
    #     print(line.xy)
    for splitLine in splitPerpendicularLines:
        oldPolygons = newPolygons
        newPolygons = []
        for oldPolygon in oldPolygons:
            result = split(oldPolygon, splitLine)
            for newPolygon in result:
                newPolygons.append(newPolygon)
    polygons = newPolygons
    centroids = [(poly.centroid.x, poly.centroid.y) for poly in polygons]
    # for x, y in zip(ROIs, centroids):
    #     print(x)
    #     print(y)
    #     print('\n')
    return polygons, centroids

def num_of_split_polygons(diameter, diameter_th={2:4, 3:8}):
    if diameter >= diameter_th[2] and  diameter < diameter_th[3]:
        return 2
    if diameter >= diameter_th[3]:
        return 3
    return 1

def split_polygon_recurrent(polygon, scale=1, diameter_th={2:4, 3:8}, max_split_time=np.inf):
    split_time = 0 
    polygons = [polygon]
    diameters = []
    majorAxisPoitsList = []
    for polygon in polygons:
        majorAxisPoints, majorAxisLength = polygon_major_axis(polygon)
        majorAxisLength *= scale
        diameters.append(majorAxisLength)
        majorAxisPoitsList.append(majorAxisPoints)
    finish = all([num_of_split_polygons(diameter, diameter_th)==1 for diameter in diameters])
    while finish is False and split_time < max_split_time:
        newPolygons = []
        newDiameters = []
        newMajorAxisPoitsList = []
        for polygon, diameter, majorAxisPoints in zip(polygons, diameters, majorAxisPoitsList):
            npoly = num_of_split_polygons(diameter, diameter_th)
            if npoly == 1:
                newPolygons.append(polygon)
                newDiameters.append(diameter)
                newMajorAxisPoitsList.append(majorAxisPoints)
            else:
                polygons_split, centroids_split = split_polygon(polygon, majorAxisPoints, npoly=npoly)
                split_time += 1
                for polygon_new in polygons_split:
                    majorAxisPoints, majorAxisLength = polygon_major_axis(polygon_new)
                    majorAxisLength *= scale
                    newPolygons.append(polygon_new)
                    newDiameters.append(majorAxisLength)
                    newMajorAxisPoitsList.append(majorAxisPoints)
        polygons, diameters, majorAxisPoitsList = newPolygons, newDiameters, newMajorAxisPoitsList
        finish = all([num_of_split_polygons(diameter, diameter_th) == 1 for diameter in diameters])
    centroids = [(poly.centroid.x, poly.centroid.y) for poly in polygons]
    return polygons, diameters, majorAxisPoitsList, centroids

def scale(coords, x_start, y_start, cell_size=30, sensor_resolution=(8,8)):
    # return ((coords[0]-x_start)/(cell_size*sensor_resolution[0]), (coords[1]-y_start)/(cell_size*sensor_resolution[1]))
    grid_coord_x = (coords[1] - y_start) / cell_size
    grid_coord_y = (coords[0] - x_start) / cell_size
    return (grid_coord_y/sensor_resolution[1], grid_coord_x/sensor_resolution[0])

def exmaple():
    vertices = [[0, 0], [4, 0], [4, 1], [1, 2], [0, 2], [0, 0]]
    hull = ConvexHull(vertices)
    v = np.asarray([vertices[idx] for idx in hull.vertices])
    polygon = Polygon(vertices)
    cx, cy = polygon.centroid.x, polygon.centroid.y
    p = [2.5, 1.5]
    c = [cx, cy]
    line = getExtrapoledLine(p, c)
    intersection = polygon.exterior.intersection(line)
    if intersection.is_empty:
        print("shapes don't intersect")
    elif intersection.geom_type.startswith('Multi') or intersection.geom_type == 'GeometryCollection':
        interPoints = []
        for shp in intersection:
            print(shp)
            interPoints.append(shp)
    else:
        print(intersection)

    splitLine = LineString(interPoints)
    # splitLine = LineString([(100,200), (100, -200)])
    result = split(polygon, splitLine)
    print(len(result))
    for x in result:
        print(x)
        print(x.centroid.x, x.centroid.y)

def combine_two_traces(trace_i, trace_j, gap_dict, uncondintionRules, conditionRules, coodScale=1, printWhy=False):
    # make sure the trace-i start earlier than trace-j
    # if coodScale != 1:
        # trace_i['cood'] = [cood*coodScale for point in trace_i['cood'] for cood in point]
        # trace_j['cood'] = [cood*coodScale for point in trace_j['cood'] for cood in point]
    flag = True
    if trace_i['t'] > trace_j['t']:
        trace_i, trace_j = trace_j, trace_i
    start_i, end_i = trace_i['t'], trace_i['t']+len(trace_i['cood'])-1
    start_j, end_j = trace_j['t'], trace_j['t']+len(trace_j['cood'])-1
    # uncombinable because of overlapping
    if start_j < end_i:
        why = 'uncombinable because of overlapping'
        if printWhy:
            print(why)
        return False, [trace_i, trace_j]

    time_gap = start_j - end_i
    # uncombinable because time gap is unacceptable
    if time_gap not in gap_dict:
        why = 'uncombinable because time gap is unacceptable, time_gap={}'.format(time_gap)
        if printWhy:
            print(why)
        return False, [trace_i, trace_j]
        
    dist = euclidean_distance(trace_j['cood'][0], trace_i['cood'][-1]) * coodScale
    # uncombinable because distance gap is unacceptable
    if dist > gap_dict[time_gap]:
        why = 'uncombinable because distance gap is unacceptable, time_gap={}, distance={}'.format(time_gap, dist)
        if printWhy:
            print(why)
        return False, [trace_i, trace_j]
        
    # processing rules
    if len(trace_i['cood']) > 1:
        meanPosition_b = trace_i['cood'][-1]
        meanPosition_c = trace_j['cood'][0]
        meanPosition_a = trace_i['cood'][-2]
    elif len(trace_j['cood']) > 1:
        meanPosition_b = trace_j['cood'][0]
        meanPosition_a = trace_j['cood'][1]
        meanPosition_c = trace_i['cood'][-1]
    else:
        meanPosition_a = trace_i['cood'][0]
        meanPosition_b = meanPosition_a
        meanPosition_c = meanPosition_c = trace_j['cood'][0]
    # print(trace_i)
    # print(trace_j)
    # print(meanPosition_a, meanPosition_b, meanPosition_c)
    indices = indices_person_with_person_in_three_frames(meanPosition_a, meanPosition_b, meanPosition_c)
    indicesMatrix = {'ED': np.asarray([[indices['ED']]])*coodScale, 'EDL': np.asarray([[indices['EDL']]]*coodScale), 
                     'LR': np.asarray([[indices['LR']]]), 'TA': np.asarray([[indices['TA']]]),
                     'DBA': np.asarray([[indices['DBA']]])*coodScale, 'DBB': np.asarray([[indices['DBB']]])*coodScale, 
                     'DBC': np.asarray([[indices['DBC']]])*coodScale, 'R_ED1': np.asarray([[1]]), 'R_ED2': np.asarray([[1]])}
    # print(indicesMatrix)
    rule_result = accept_reject_matrix(indicesMatrix, uncondintionRules, conditionRules, whyFlag=True)
    # print(rule_result)
    # uncombinable because of rules
    if rule_result[0][0] == 0:
        why = 'uncombinable because of rules'
        if printWhy:
            print(why)
        return False, [trace_i, trace_j]
           
    new_trace_coords = trace_i['cood'] + [trace_i['cood'][-1] for i in range(time_gap-1)] + trace_j['cood']
    new_trace_coords_grid = trace_i['cood_grid'] + [trace_i['cood_grid'][-1] for i in range(time_gap-1)] + trace_j['cood_grid']
    new_trace_temp_raw = trace_i['temp_raw'] + [trace_i['temp_raw'][-1] for i in range(time_gap-1)] + trace_j['temp_raw']
    new_trace_delta_temp = trace_i['delta_temp'] + [trace_i['delta_temp'][-1] for i in range(time_gap-1)] + trace_j['delta_temp']
    if trace_j['fnsh'] is False:
        new_trace_waiting_for_combine = [True]
    else:
        new_trace_waiting_for_combine = trace_j['waiting_for_combine']
    new_trace = {'t':start_i, 'cood': new_trace_coords, 'cood_grid': new_trace_coords_grid, 'fnsh': trace_j['fnsh'], 
        'reflect': trace_j['reflect'], 'show':trace_j['show'], 'id': trace_i['id'], 'combined_ids': trace_i['combined_ids']+trace_j['combined_ids'],
        'waiting_for_combine': new_trace_waiting_for_combine, 'temp_raw': new_trace_temp_raw, 'delta_temp': new_trace_delta_temp,
        'img_names_first_last': [trace_i['img_names_first_last'][0], trace_j['img_names_first_last'][1]]}
    if trace_i['show'] or trace_j['show']:
        new_trace['show'] = True
    if 'to_activity_map' in trace_i or 'to_activity_map' in trace_j:
        new_trace['to_activity_map'] = False
        if trace_i['to_activity_map'] is True or trace_j['to_activity_map'] is True:
            new_trace['to_activity_map'] = True
    return True, [new_trace]

# @run_time_utilities
def post_combine(traces, uncondintionRules, conditionRules, 
    gap_dict={2:2*30, 3:3.5*30, 4:4*30, 5:4*30, 6:4*30}, coodScale=8*30, log=True, debug=False, uncombinableFlag=True):
    if debug: print('\n' + '='*20 + 'Post Combine Debug' + '='*20)
    processTraces = copy.deepcopy(traces)
    finishedTraces = []
    # # a more aggressive way: if a trace is judged as uncombinable before (waiting_for_combine=False), just skip it,
    # # but it may bring some subtle risks that a trace is uncombinable before but combinable now after other traces are combined.
    # finishedTraces = [t for t in processTraces if t['waiting_for_combine'] = False]
    # processTraces = [t for t in processTraces if t['waiting_for_combine'] = True]
    numCombine = 0
    combine_log = []
    while len(processTraces) > 1:
        trace_i = processTraces[0]
        if debug: print('post_combine: trace_i = {}'.format(trace_i))
        thisTrace_uncombinable = True
        for trace_j_idx in range(1, len(processTraces)):
            trace_j = processTraces[trace_j_idx]
            combineFlag, combineRst = combine_two_traces(trace_i, trace_j, gap_dict, uncondintionRules, conditionRules, coodScale=8*30)
            if combineFlag:
                thisTrace_uncombinable = False
                processTraces.remove(trace_i)
                processTraces.remove(trace_j)
                processTraces = combineRst + processTraces
                combine_log.append((trace_i['id'], trace_j['id']))
                break
        if debug: print('post_combine: trace_i uncombinable = {}'.format(thisTrace_uncombinable))
        if thisTrace_uncombinable:
            processTraces.remove(trace_i)
            if trace_i['fnsh'] and uncombinableFlag: trace_i['waiting_for_combine'].append(False)
            if debug: print('This uncombinable trace: {}'.format(trace_i))
            finishedTraces.append(trace_i)
            
    # for consistency, set the "waiting_for_combine" attr of the last trace in processTraces to True if it is also finished
    if len(processTraces) == 1:
        if processTraces[0]['fnsh'] and uncombinableFlag: processTraces[0]['waiting_for_combine'].append(False)
    finishedTraces += processTraces
    if debug: 
        print('Finally: ')
        for t in finishedTraces: print(t)
    if debug: print('='*20 + 'Post Combine Debug' + '='*20 + '\n')
    if log:
        return finishedTraces, combine_log
    else:
        return finishedTraces

# @run_time_utilities
def post_remove(traces, step_th=2):
    finishedTraces = []
    # for trace in traces:
    #     if trace['fnsh'] is True:
    #         if len(trace['cood']) <= step_th:
    #             print('utilities.py: skipped a trace that is too short')
    #             continue
    #         if len(trace['cood']) >= 3:
    #             area = Polygon(np.array(trace['cood'])).area
    #             if area <= 0:
    #                 print('utilities.py: skipped a trace that is a straight line')
    #                 continue
    #     finishedTraces.append(trace)
    for trace in traces:
        if trace['fnsh'] == False:
            finishedTraces.append(trace)
        else:
            if len(trace['cood']) <= step_th:
                # print('utilities.py: skipped a trace that is too short')
                pass
            elif len(trace['cood']) >= 3:
                area = Polygon(np.array(trace['cood'])).area
                if area > 0:
                    finishedTraces.append(trace)
                else:
                    # print('utilities.py: skipped a trace that is a straight line')
                    pass
    return finishedTraces

# @run_time_utilities
def counting_remove(points, gap_dict={1:2*30, 2:2*30, 3:3*30}):
    if len(points) < 2:
        return points
    finishedPoints = points[0:1] # the first point is always regarded as valid
    processPoints = points[1:]   # the 2nd and following points are to be tested
    while len(processPoints) > 0:
        crtTestPoint = processPoints[0]
        crtValidPoint = finishedPoints[-1]
        time_gap = crtTestPoint['t'] - crtValidPoint['t']
        if time_gap in gap_dict.keys():
            dist = euclidean_distance(crtTestPoint['int_pt_screen'], crtValidPoint['int_pt_screen'])
            if dist > gap_dict[time_gap]:
                finishedPoints.append(crtTestPoint)
            else:
                # print('Point be removed:\n{}'.format(crtTestPoint))
                pass
        else:
            finishedPoints.append(crtTestPoint)
        processPoints.remove(crtTestPoint)
    return finishedPoints

# @run_time_utilities
def extent_traces_to_border(traces, lastN=5 ,max_th=1.41421, min_th=0):
    borders = [LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)]), 
               LineString([(1, 1), (1, 0)]), LineString([(1, 0), (0, 0)])]
    finishedTraces = []
    for trace in traces:
        if not trace['fnsh']:
            finishedTraces.append(trace)
            continue
        if len(trace['cood']) == 1:
            finishedTraces.append(trace)
            continue
        
        lastN2 = min(lastN, len(trace['cood']))
        # a -> b
        abLine = getExtrapoledLine(trace['cood'][-lastN2], trace['cood'][-1], both=False)
        for border in borders:
            border_intersection = border.intersection(abLine)
            if not border_intersection.is_empty:
                border_dist = euclidean_distance(trace['cood'][-1], (border_intersection.x, border_intersection.y))
                if border_dist >= min_th and border_dist <= max_th:
                    trace['cood'].append((border_intersection.x, border_intersection.y))
                break   
        # b -> a
        baLine = getExtrapoledLine(trace['cood'][lastN2-1], trace['cood'][0], both=False)
        for border in borders:
            border_intersection = border.intersection(baLine)
            if not border_intersection.is_empty:
                border_dist = euclidean_distance(trace['cood'][0], (border_intersection.x, border_intersection.y))
                if border_dist >= min_th and border_dist <= max_th:
                    trace['cood'] = [(border_intersection.x, border_intersection.y)] + trace['cood']
                break   
        finishedTraces.append(trace)
    return finishedTraces

def momentum(crt_point, last_point, v=[0,0], eta=0.9, alpha='auto'):
    if alpha == 'auto':
        alpha = 1-eta
        # special case
        if np.linalg.norm(v) == 0:
            alpha=1
    
    crt_direction = [crt_point[i]-last_point[i] for i in [0,1]]
    v = [eta*v[i] + alpha*crt_direction[i] for i in [0,1]]
    return v

# @run_time_utilities
def reflecting_distance(meanPositions_a, meanPositions_b, meanPositions_c, vs, eta=0.9, alpha='auto'):
    nrow, ncol = len(meanPositions_b), len(meanPositions_c)
    dm = []
    vs_new = []
    reflecting_points = []
    for meanPosition_a, meanPosition_b, v in zip(meanPositions_a, meanPositions_b, vs):
        v = momentum(meanPosition_b, meanPosition_a, v, eta=eta, alpha=alpha)
        vs_new.append(v)
        expected_meanPosition_c = [meanPosition_b[i]+v[i] for i in [0,1]]
        reflecting_points.append(expected_meanPosition_c) 
        dist = [euclidean_distance(expected_meanPosition_c, meanPositions_c[i]) for i in range(len(meanPositions_c))]
        dm.append(dist)
    dm = np.asarray(dm)
    return dm, vs_new, reflecting_points

# @run_time_utilities
def dist_to_p(dm, b_d=-1, dist_scale=8*30):
    """
    dm: distance matrix
    """
    dm /= dist_scale
    P_toChooseNext = V2P(dm*b_d, axis=1)
    P_toChoosePrevious = V2P(dm*b_d, axis=0)
    P = P_toChooseNext * P_toChoosePrevious
    return P

if __name__ == "__main__":
    """

    a = [[0,0], [1,1], [4,3]]
    b = [[2,4], [3,7], [6,2]]
    c = [[7,5], [9,4]]
    v = [[0,0], [0,0], [0,0]]
    dm = reflecting_distance(a,b,c,v)
    print(dm)
    exit()
    
    uncondintionRules, conditionRules = read_rules(r'C:\Friends\butlr\git20200110\Butlr_POC-master\Application\rules.txt')
    gap_dict = {0:85, 1:85, 2:85, 3:3.5*30, 4:4*30, 5:4*30, 6:4*30}
    trace_a = {'t': 7, 'cood': [(0.3125, 0.9375), (0.4375, 0.9375), (0.4375, 0.9375), (0.4375, 0.9375), (0.4375, 0.8125), (0.4375, 0.8125), (0.4375, 0.6875), (0.3125, 0.9375)], 'fnsh': True}
    trace_b = {'t': 15, 'cood': [(0.5, 0.4375), (0.5625, 0.1875)], 'fnsh': False}
    a, b = combine_two_traces(trace_a, trace_b, gap_dict, uncondintionRules, conditionRules, coodScale=8*30, printWhy=True)
    
    print(a)
    print(b)
    exit()
    
    
    
    trace_a = {'t': 13, 'cood': [(0.5625, 0.0625), (0.5625, 0.0625), (0.5625, 0.0625), (0.4375, 0.0625), (0.5625, 0.1875), (0.5625, 0.3125), (0.5625, 0.4375)], 'fnsh': True}
    trace_b = {'t': 19, 'cood': [(0.3125, 0.6875), (0.4375, 0.9375)], 'fnsh': False}
    trace_a = {'t': 2, 'cood': [(0.3125, 0.9375), (0.4375, 0.9375), (0.4375, 0.9375), (0.4375, 0.9375), (0.4375, 0.8125), (0.4375, 0.8125), (0.4375, 0.6875)], 'fnsh': True}
    trace_b = {'t': 10, 'cood': [(0.5, 0.4375), (0.5625, 0.1875), (0.5625, 0.1875), (0.4375, 0.0625)], 'fnsh': False}
    trace_a = {'t': 7, 'cood': [(0.3125, 0.9375), (0.4375, 0.9375), (0.4375, 0.9375), (0.4375, 0.9375), (0.4375, 0.8125), (0.4375, 0.8125), (0.4375, 0.6875)], 'fnsh': True}
    trace_b = {'t': 13, 'cood': [(0.3125, 0.9375)], 'fnsh': True}
    trace_c = {'t': 15, 'cood': [(0.5, 0.4375), (0.5625, 0.1875)], 'fnsh': False}
    traces = [trace_a, trace_b, trace_c]
    
    rst = post_combine(traces, uncondintionRules, conditionRules, gap_dict=gap_dict)
    for x in rst:
        print(x)
    # a, b = combine_two_traces(trace_a, trace_b, gap_dict, uncondintionRules, conditionRules, coodScale=8*30, printWhy=True)
    # print(a)
    # print(b)
    exit()
    
    points = [{'t': 50, 'int_pt_screen': (277, 162), 'mark_color': (255, 0, 0, 255)}, {'t': 51, 'int_pt_screen': (277, 164), 'mark_color': (255, 0, 0, 255)}, {'t': 52, 'int_pt_screen': (277, 164), 'mark_color': (255, 0, 0, 255)}]
    x = counting_remove(points)
    print(points)
    print(x)
    
    exit()
    
    # trace_a = {'t': -1, 'cood': [(0.8125, 0.5), (0.7380952380952382, 0.4880952380952381), (0.5625, 0.4375), (0.375, 0.4375), (0.3125, 0.375), (0.1875, 0.375), (0.11309523809523808, 0.3869047619047619), (0.0625, 0.4375), (0.0625, 0.4375)], 'fnsh': True}
    # trace_b = {'t': 2, 'cood': [(0.8125, 0.0625), (0.6875, 0.125), (0.625, 0.0625), (0.4375, 0.0625)], 'fnsh': True}
    # trace_c = {'t': 7, 'cood': [(0.3125, 0.0625), (0.1875, 0.125), (0.0625, 0.1875)], 'fnsh': False}
    trace_a = {'t': 2, 'cood': [(0.9375, 0.625), (0.8630952380952382, 0.636904761904762), (0.8125, 0.75), (0.7619047619047618, 0.761904761904762), (0.6875, 0.75), (0.4375, 0.8125), (0.375, 0.8125), (0.25, 0.8125), (0.3125, 0.8125)], 'fnsh': True}
    trace_b = {'t': 11, 'cood': [(0.0625, 0.8125), (0.0625, 0.75), (0.0625, 0.6875)], 'fnsh': False}
    gap_dict = {1:2*30, 2:2*30, 3:3.5*30, 4:4*30, 5:4*30, 6:4*30}
    uncondintionRules, conditionRules = read_rules(r'C:\Friends\butlr\tracking\rules.txt')
    # traces = [trace_a, trace_b, trace_c]
    # rst = post_combine(traces)
    # for x in rst:
        # print(x)
    a, b = combine_two_traces(trace_b, trace_a, gap_dict, uncondintionRules, conditionRules, coodScale=8*30, printWhy=True)
    print(a)
    print(b)

    exit()
    """
    import matplotlib.pyplot as plt
    # exmaple()
    # vertices = [[0, 0], [4, 0], [4, 1], [1, 2], [0, 2], [0, 0]]
    vertices = [[0, 0], [6, 0], [6, 6], [0, 6]]
    vertices = [[0, 0], [6, 0], [6, 3], [3, 3], [3, 6], [0, 6]]
    hull = ConvexHull(vertices)
    vertices = np.asarray([vertices[idx] for idx in hull.vertices])
    polygon = Polygon(vertices)

    # majorAxisPoints, majorAxisLength = polygon_major_axis(polygon)
    # ROIs, centroids = split_polygon(polygon, majorAxisPoints, 3)
    # for ROI, centroid in zip(ROIs, centroids):
    #     ROIx, ROIy = ROI.exterior.xy
    #     plt.plot(ROIx, ROIy, 'k-')
    #     plt.plot(centroid[0], centroid[1], '*')
    # plt.show()
    polygons, diameters, majorAxisPoitsList, centroids = split_polygon_recurrent(polygon)
    print('Number of split polygons: {}'.format(len(polygons)))
    for polygon, centroid in zip(polygons, centroids):
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'k-')
        plt.plot(centroid[0], centroid[1], 'k*')
    plt.axis('equal')
    plt.grid('on')
    plt.show()



