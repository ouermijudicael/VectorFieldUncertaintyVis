# 
#
# 
import sys
import numpy as np
import scipy as sp
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA

import dash_vtk 
from dash_vtk.utils import to_mesh_state

import time

from numba import jit, njit


def insideConvexHull(conv_hull, points):
    """
    purpose: check if a point is inside a convex hull
    input:
    conv_hull: 2D array of shape (n, d) where n is the number of points and d is the dimension of the points
    points: 2D array of shape (m, d) where m is the number of points and d is the dimension of the points
    output:
    res: 1D array of shape (m) where res[i] = 1 if points[i] is inside the convex hull, otherwise res[i] = 0


    """        
    hull = conv_hull
    # print("point = ", points)
    distinct_points = True

    is_a_point = True

    # check if the convex hull is a point
    for i in range(conv_hull.shape[0]):
        if(np.absolute(conv_hull[i][0]) > 1.0e-10):
            is_a_point = False
            break

    if(is_a_point):
        res = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                if(np.absolute(points[i][j] - conv_hull[0][j]) < 1.0e-8):
                    res[i] = 1
        return res
    
    # check if the convex hull is a line
    is_a_line = True
    for i in range(conv_hull.shape[0]-1):
        for j in range(1, conv_hull.shape[1]):
            if( np.absolute(conv_hull[i][j] - conv_hull[i+1][j]) > 1.0e-8 ):
                is_a_line = False
                break
        if(not is_a_line):
            break

    if(is_a_line):
        bbx = vf_utils.getBoundingBox(conv_hull)
        res = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                if( bbx[j][0] <= points[i][j] and points[i][j] <= bbx[j][1]):
                    res[i] = 1
        return res
    
    for i in range(conv_hull.shape[0]-1):
        for j in range(conv_hull.shape[1]):
            if( np.absolute(conv_hull[i][j] - conv_hull[i+1][j]) > 1.0e-8 and
                np.absolute(conv_hull[i][0]) > 1.0e-10):
                distinct_points = True
                break
        if(distinct_points):
            break


    if(not distinct_points):
        res = np.ones(points.shape[0])
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                if(np.absolute(points[i][j] - conv_hull[0][j]) > 1.0e-8):
                    res[i] = 0
        return res
    
    dimensions = conv_hull.shape[1]
    # print("dimensions = ", dimensions)
    if( dimensions == 2):
        is_coplanar = False
    else: 
        is_coplanar = True

    if(is_coplanar):
        for i_dim in range(1,dimensions):
            # print("i_dim = ", i_dim, "is_co_planar = ", is_coplanar)
            is_coplanar = True
            for i in range(conv_hull.shape[0]-1):
                if(np.absolute(conv_hull[i][i_dim] - conv_hull[i+1][i_dim]) > 1.0e-8):
                    is_coplanar = False
                    break
            if(is_coplanar):
                break
    if(is_coplanar):
        res = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            res[i] = 1
        return res
    
    # print("hull = ", hull)
    if (not isinstance(hull, sp.spatial.Delaunay)):
        # print("is_coplanar = ", is_coplanar)
        # print("hull = ", hull)
        hull = sp.spatial.Delaunay(hull)
    return hull.find_simplex(points) >= 0
    
def getSelectedPointsIndices(num_ensemble_members, num_selected_points):
    """
    purpose: compute the indices of the selected points
    input:
      num_ensemble_members: number of ensemble members
      num_selected_points: number of points used to define the convex hull
    output:
      selected_points_indices: 2D array of shape (num_convex_hulls, num_selected_points) where each row contains the indices of the selected points
    
    """
    indices = np.linspace(0, num_ensemble_members-1, num_ensemble_members, dtype=int)
    selected_points_indices = np.array(list(itertools.combinations(indices, num_selected_points)))
    return selected_points_indices

@jit(nopython=True)
def insideTriangle(triangle, points):
    
    n = 3 # must be 3
    num_points = points.shape[0]
    res = np.zeros(num_points)
    for  i_pt in range(num_points):
        p2x = 0.0
        p2y = 0.0
        xints = 0.0
        p1x= triangle[0]
        p1y= triangle[2]
        inside = False
        x = points[i_pt][0]
        y = points[i_pt][2]
        for i in range(n+1):
            p2x= triangle[i % n][0]
            p2y= triangle[i % n][2]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y
        if(inside):
            res[i_pt] = 1
        else:
            res[i_pt] = 0
    return res


@njit
def insideThetahedron(thetahedron, points):
    """
        purpose: check if a point is inside a thetahedron
    input:
    thetahedron: 2D array of shape (4, 3) where each row contains the x, y, and z coordinates of the thetahedron vertices
    points: 2D array of shape (m, 3) where m is the number of points and the last dimension contains the x, y, and z coordinates of the points
    output:
    res: 1D array of shape (m) where res[i] = 1 if points[i] is inside the thetahedron, otherwise res[i] = 0

    """
    res = np.zeros(points.shape[0])
    for i_pt in range(points.shape[0]):
        for i in range(4):
            normal = np.cross(thetahedron[(i+1)%4] - thetahedron[i], thetahedron[(i+2)%4] - thetahedron[i])
            dotV4 = np.dot(normal, thetahedron[(i+3)%4] - thetahedron[i])
            dotP = np.dot(normal, points[i_pt] - thetahedron[i])
            if( dotP * dotV4 < 0):
                res[i_pt] = 0
                break
            else:
                res[i_pt] = 1
    return res


@njit
def getDatasetDepthGPU2D( vectors, selected_points_indices):
    """
    purpose: compute the data depth of a set of vectors
    input:
      vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components
      num_selected_points: number of points used to define the convex hull
    output:
      depths: 3D array of shape (num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members
    
    """
    num_selected_points = 3 # must be 3
    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    num_convex_hulls = selected_points_indices.shape[0]
    depths = np.zeros((num_points, num_ensemble_members))
    selected_points = np.zeros((num_selected_points, 3))
    for i_pt in range(num_points):
        for i_cvx in range(num_convex_hulls):
            # get selected points
            for j in range(num_selected_points):
                for k in range(3):
                    # print("i_cvx = ", i_cvx, "j = ", j, "k = ", k)
                    selected_points[j][k] = vectors[i_pt][selected_points_indices[i_cvx][j]][k]
            for i_m in range(num_ensemble_members):
                inside = False
                p2x = 0.0
                p2y = 0.0
                xints = 0.0
                p1x= selected_points[0][0]
                p1y= selected_points[0][2]
                x = vectors[i_pt][i_m][0]
                y = vectors[i_pt][i_m][2]

                for i in range(num_selected_points):
                    p2x= selected_points[i % num_selected_points][0]
                    p2y= selected_points[i % num_selected_points][2]
                    if y > min(p1y,p2y):
                        if y <= max(p1y,p2y):
                            if x <= max(p1x,p2x):
                                if p1y != p2y:
                                    xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                                if p1x == p2x or x <= xints:
                                    inside = not inside
                    p1x,p1y = p2x,p2y
                if(inside):
                    depths[i_pt][i_m] = depths[i_pt][i_m] + 1
                
    depths = depths/num_convex_hulls
    #         tmp = insideThetahedron(selected_points, vectors[i_pt])
    #         depths[i_pt] = depths[i_pt] + tmp
    # depths = depths/num_convex_hulls
    return depths


@njit
def getDatasetDepthGPU( vectors, selected_points_indices):
    """
    purpose: compute the data depth of a set of vectors
    input:
      vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components
      num_selected_points: number of points used to define the convex hull
    output:
      depths: 3D array of shape (num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members
    
    """
    num_selected_points = 4 # must be 4
    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    num_convex_hulls = selected_points_indices.shape[0]
    depths = np.zeros((num_points, num_ensemble_members))
    selected_points = np.zeros((num_selected_points, 3))
    for i_pt in range(num_points):
        for i_cvx in range(num_convex_hulls):
            # get selected points
            for j in range(num_selected_points):
                for k in range(3):
                    # print("i_cvx = ", i_cvx, "j = ", j, "k = ", k)
                    selected_points[j][k] = vectors[i_pt][selected_points_indices[i_cvx][j]][k]
            for i_m in range(num_ensemble_members):
                for i in range(num_selected_points):
                    normal = np.cross(selected_points[(i+1)%4] - selected_points[i], selected_points[(i+2)%4] - selected_points[i])
                    dotV4 = np.dot(normal, selected_points[(i+3)%4] - selected_points[i])
                    dotP = np.dot(normal, vectors[i_pt][i_m] - selected_points[i])
                    if( dotP * dotV4 < 0):
                        depths[i_pt][i_m] = depths[i_pt][i_m] + 1
    depths = depths/num_convex_hulls
    #         tmp = insideThetahedron(selected_points, vectors[i_pt])
    #         depths[i_pt] = depths[i_pt] + tmp
    # depths = depths/num_convex_hulls
    return depths

@njit
def getHyperCubeDatasetDepthGPU( vectors, selected_points_indices, num_selected_points):
    """
    purpose: compute the data depth of a set of vectors
    input:
      vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components
      num_selected_points: number of points used to define the convex hull
    output:
      depths: 3D array of shape (num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members
    
    """

    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    num_convex_hulls = selected_points_indices.shape[0]
    depths = np.zeros((num_points, num_ensemble_members))
    # selected_points = np.zeros((num_selected_points, 3))
    bbx = np.zeros((3, 2))
    for i_pt in range(num_points):
        for i_cvx in range(num_convex_hulls):
            # get selected points
            bbx[0:3, 0] = 1.0e+10
            bbx[0:3, 1] = -1.0e+10
            for j in range(num_selected_points):
                for k in range(3):
                    if(vectors[i_pt][selected_points_indices[i_cvx][j]][k] < bbx[k][0]):
                        bbx[k][0] = vectors[i_pt][selected_points_indices[i_cvx][j]][k]
                    if(vectors[i_pt][selected_points_indices[i_cvx][j]][k] > bbx[k][1]):
                        bbx[k][1] = vectors[i_pt][selected_points_indices[i_cvx][j]][k]
                    # print("i_cvx = ", i_cvx, "j = ", j, "k = ", k)
                    # selected_points[j][k] = vectors[i_pt][selected_points_indices[i_cvx][j]][k]
            for i_m in range(num_ensemble_members):
               
                if (vectors[i_pt][i_m][0] >= bbx[0][0]- 1e-16 and vectors[i_pt][i_m][0] <= bbx[0][1] + 1e-16 and
                    vectors[i_pt][i_m][1] >= bbx[1][0]- 1e-16 and vectors[i_pt][i_m][1] <= bbx[1][1] + 1e-16 and
                    vectors[i_pt][i_m][2] >= bbx[2][0]- 1e-16 and vectors[i_pt][i_m][2] <= bbx[2][1] + 1e-16):# and 
                    # i_m != selected_points_indices[i_cvx][0] and i_m != selected_points_indices[i_cvx][1] and 
                    # i_m != selected_points_indices[i_cvx][2] and i_m != selected_points_indices[i_cvx][3%num_selected_points]):
                    # if(i_m > 8):
                    #     print("i_m = ", i_m, "i_cvx = ", i_cvx, "i_pt = ", i_pt, "bbx = ", bbx)
                    #     print("selected_points_indices = ", selected_points_indices[i_cvx])
                    #     print("vectors = ", vectors[i_pt][i_m], depths[i_pt][i_m])
                    depths[i_pt][i_m] = depths[i_pt][i_m] + 1
                # for i in range(num_selected_points):
                #     normal = np.cross(selected_points[(i+1)%4] - selected_points[i], selected_points[(i+2)%4] - selected_points[i])
                #     dotV4 = np.dot(normal, selected_points[(i+3)%4] - selected_points[i])
                #     dotP = np.dot(normal, vectors[i_pt][i_m] - selected_points[i])
                #     if( dotP * dotV4 < 0):
                #         depths[i_pt][i_m] = depths[i_pt][i_m] + 1
    depths = depths/num_convex_hulls
    depths = depths/np.max(depths)
    #         tmp = insideThetahedron(selected_points, vectors[i_pt])
    #         depths[i_pt] = depths[i_pt] + tmp
    # depths = depths/num_convex_hulls
    return depths


def hyperCubeDataDepth(data, num_selected_points):
    """
    purpose: compute the data depth of a set of vectors
    input:
      data: dictionary containing the following
        num_time_steps: number of time steps
        num_points: number of positions
        dimensions: dimension of the vectors
        num_ensemble_members: number of ensemble members
        positions: 3D array of shape (1, num_time_steps, num_points, 2) where the last dimension contains the x and y coordinates of the positions
        vectors: 4D array of shape (1, num_time_steps, num_points, num_ensemble_members, dimensions) where the last dimension contains the vector components
      num_selected_points: number of points used to define the convex hull
    output:
      depths: 3D array of shape (num_time_steps, num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members
    
    """
    num_time_steps = int(data["num_time_steps"][0])
    num_positions = int(data["num_points"][0])
    dimensions = int(data["dimensions"][0])
    # print('dimensions =', dimensions, 'num_selected_points =', num_selected_points)
    num_ensemble_members = int(data["num_ensemble_members"][0])
    # indices = np.linspace(0, num_ensemble_members-1, num_ensemble_members, dtype=int)
    # depths = np.zeros([num_time_steps, num_positions, num_ensemble_members])
    # compute indices choose num_selected_points
    selected_points_indices = getSelectedPointsIndices(num_ensemble_members, num_selected_points)
    # print('selected_points_indices.shape =', selected_points_indices.shape)
    # selected_points_indices = np.array(list(itertools.combinations(indices, num_selected_points)))
    # num_convex_hulls = selected_points_indices.shape[0] 
    # print("num_convex_hulls = ", num_convex_hulls)
    # print("dimensions = ", dimensions)
    # print("selected_points_indices = ", selected_points_indices)

    ## Compute convex hulls
    # selected_points = np.zeros([num_selected_points, dimensions])
    for t in range(num_time_steps):
        tmp_depths = getHyperCubeDatasetDepthGPU(data["vectors"][0][t], selected_points_indices, num_selected_points)
        
        for i_pt in range(num_positions):
            for i_m in range(num_ensemble_members):
                data["depths"][0][t][i_pt][i_m] = tmp_depths[i_pt][i_m]

@njit
def hyperCubeVectorDepthGPU(vectors, selected_points_indices, num_selected_points):
    """
    
    """
    num_time_steps = vectors.shape[0]
    num_points = vectors.shape[1]
    num_ensemble_members = vectors.shape[2]
    num_convex_hulls = selected_points_indices.shape[0]
    depths = np.zeros((num_time_steps, num_points, num_ensemble_members))
    # selected_points = np.zeros((num_selected_points, 3))
    bbx = np.zeros((3, 2))
    for i_t in range(num_time_steps):
        for i_pt in range(num_points):
            for i_cvx in range(num_convex_hulls):
                # get selected points
                bbx[0:3, 0] = 1.0e+10
                bbx[0:3, 1] = -1.0e+10
                for j in range(num_selected_points):
                    for k in range(3):
                        if(vectors[i_t][i_pt][selected_points_indices[i_cvx][j]][k] < bbx[k][0]):
                            bbx[k][0] = vectors[i_t][i_pt][selected_points_indices[i_cvx][j]][k]
                        if(vectors[i_t][i_pt][selected_points_indices[i_cvx][j]][k] > bbx[k][1]):
                            bbx[k][1] = vectors[i_t][i_pt][selected_points_indices[i_cvx][j]][k]
                        # print("i_cvx = ", i_cvx, "j = ", j, "k = ", k)
                        # selected_points[j][k] = vectors[i_pt][selected_points_indices[i_cvx][j]][k]
                for i_m in range(num_ensemble_members):
                
                    if (vectors[i_t][i_pt][i_m][0] >= bbx[0][0]- 1e-16 and vectors[i_t][i_pt][i_m][0] <= bbx[0][1] + 1e-16 and
                        vectors[i_t][i_pt][i_m][1] >= bbx[1][0]- 1e-16 and vectors[i_t][i_pt][i_m][1] <= bbx[1][1] + 1e-16 and
                        vectors[i_t][i_pt][i_m][2] >= bbx[2][0]- 1e-16 and vectors[i_t][i_pt][i_m][2] <= bbx[2][1] + 1e-16):# and 
                        # i_m != selected_points_indices[i_cvx][0] and i_m != selected_points_indices[i_cvx][1] and 
                        # i_m != selected_points_indices[i_cvx][2] and i_m != selected_points_indices[i_cvx][3%num_selected_points]):
                        # if(i_m > 8):
                        #     print("i_m = ", i_m, "i_cvx = ", i_cvx, "i_pt = ", i_pt, "bbx = ", bbx)
                        #     print("selected_points_indices = ", selected_points_indices[i_cvx])
                        #     print("vectors = ", vectors[i_pt][i_m], depths[i_pt][i_m])
                        depths[i_t][i_pt][i_m] += 1
                    # for i in range(num_selected_points):
                    #     normal = np.cross(selected_points[(i+1)%4] - selected_points[i], selected_points[(i+2)%4] - selected_points[i])
                    #     dotV4 = np.dot(normal, selected_points[(i+3)%4] - selected_points[i])
                    #     dotP = np.dot(normal, vectors[i_pt][i_m] - selected_points[i])
                    #     if( dotP * dotV4 < 0):
                    #         depths[i_pt][i_m] = depths[i_pt][i_m] + 1
    depths = depths/num_convex_hulls
    # depths = depths/np.max(depths)
    #         tmp = insideThetahedron(selected_points, vectors[i_pt])
    #         depths[i_pt] = depths[i_pt] + tmp
    # depths = depths/num_convex_hulls
    return depths

def hyperCubeVectorDepth(vectors, num_selected_points):
    """
    Compute the data depth of a set of vectors
    
    Parameters:
    vectors: 4D array of shape (num_time_steps, num_points, num_ensemble_members, 3) where the last dimension contains the vector components
    num_selected_points: number of points used to define the convex hull
    Returns:
    vector_depths: 3D array of shape (num_time_steps, num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members

    """

    vectors_shape = vectors.shape
    if(len(vectors_shape) != 4):
        raise ValueError("Invalid vector shape")

    num_members = vectors_shape[2]
    # time function
    # print("Starting hyperCubeVectorDepth")
    # start = time.time()
    selected_points_indices = getSelectedPointsIndices(num_members, num_selected_points)
    # end = time.time()
    # print("getSelectedPointsIndices time = ", end - start)

    # print("starting getHyperCubeDatasetDepthGPU")
    # start = time.time()
    vector_depths = hyperCubeVectorDepthGPU(vectors, selected_points_indices, num_selected_points)
    # end = time.time()
    # print("getHyperCubeDatasetDepthGPU time = ", end - start)

    return vector_depths


def dataDepth(data, num_selected_points):
    """
    purpose: compute the data depth of a set of vectors
    input:
    data: dictionary containing the following
        num_time_steps: number of time steps
        num_points: number of positions
        dimensions: dimension of the vectors
        num_ensemble_members: number of ensemble members
        positions: 3D array of shape (1, num_time_steps, num_points, 2) where the last dimension contains the x and y coordinates of the positions
        vectors: 4D array of shape (1, num_time_steps, num_points, num_ensemble_members, dimensions) where the last dimension contains the vector components
    num_selected_points: number of points used to define the convex hull
    output:
    depths: 3D array of shape (num_time_steps, num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members

    """
    num_time_steps = int(data["num_time_steps"][0])
    num_positions = int(data["num_points"][0])
    dimensions = int(data["dimensions"][0])
    # print('dimensions =', dimensions, 'num_selected_points =', num_selected_points)
    num_ensemble_members = int(data["num_ensemble_members"][0])
    # indices = np.linspace(0, num_ensemble_members-1, num_ensemble_members, dtype=int)
    # depths = np.zeros([num_time_steps, num_positions, num_ensemble_members])
    # compute indices choose num_selected_points
    selected_points_indices = getSelectedPointsIndices(num_ensemble_members, num_selected_points)
    # print('selected_points_indices.shape =', selected_points_indices.shape)
    # selected_points_indices = np.array(list(itertools.combinations(indices, num_selected_points)))
    # num_convex_hulls = selected_points_indices.shape[0] 
    # print("num_convex_hulls = ", num_convex_hulls)
    # print("dimensions = ", dimensions)
    # print("selected_points_indices = ", selected_points_indices)

    ## Compute convex hulls
    # selected_points = np.zeros([num_selected_points, dimensions])
    for t in range(num_time_steps):
        if (dimensions == 3):
            tmp_depths = getDatasetDepthGPU(data["vectors"][0][t], selected_points_indices)
        elif (dimensions == 2):
            tmp_depths = getDatasetDepthGPU2D(data["vectors"][0][t], selected_points_indices)
        else:
            raise ValueError("Invalid dimensions")
        
        for i_pt in range(num_positions):
            for i_m in range(num_ensemble_members):
                data["depths"][0][t][i_pt][i_m] = tmp_depths[i_pt][i_m]



def getDirectionalVariations(positions, vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors, domain, time_range):
    """
    purpose: compute the directional variation of the vectors
    input:
    positions: 2D array of shape (num_points, 3) where the last dimension contains the x, y, and z coordinates of the positions
    vectors: 3D array of shape (num_time_steps, num_points, num_ensemble_members, 3) where the last 
            dimension contains the vector components
    median_vectors: 3D array of shape (num_time_steps, num_points, 3) where the last dimension contains 
                    the median vector components
    max_vectors: 3D array of shape (num_time_steps, num_points, 3) where the last dimension contains
                the max vector components
    min_vectors: 3D array of shape (num_time_steps, num_points, 3) where the last dimension contains
                the min vector components
    domain: 2D array of shape (3, 2) where the first dimension contains the x, y, and z domain limits
    time_range: 1D array of shape (2) where time_range[0] is the selected start time step and time_range[1] 
                    is the selected end time step
    output:
    directional_variation: 2D array of shape (num_time_steps, num_points, 4, 2) where the 4 represents the
                            (pca variance, pca firts component, pca second component, pca mean) and the 2 represents
                            the x and y components of the pca components
    """

    num_time_steps = vectors.shape[0]
    num_points = vectors.shape[1]
    num_ensemble_members = vectors.shape[2]
    local_median_vector = np.zeros(3)
    local_XX = np.zeros([num_ensemble_members, 2])
    directional_variation = np.zeros([time_range[1]-time_range[0]+1, num_points, 3, 2])

    ii_t = -1
    for i_t in range(time_range[0], time_range[1]+1):
        ii_t = ii_t + 1
        for i_p in range(num_points):
            if( domain[0][0] - 1.0e-10 <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] + 1.0e-10 and
                domain[1][0] - 1.0e-10 <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] + 1.0e-10 and
                domain[2][0] - 1.0e-10 <= positions[i_p][2] and positions[i_p][2] <= domain[2][1] + 1.0e-10 and 
                max_vectors[i_t][i_p][0] > 0.001 ):
                r = median_vectors[i_t][i_p][0]
                theta = median_vectors[i_t][i_p][1]
                phi = median_vectors[i_t][i_p][2]
                local_median_vector[0] = r*np.sin(theta)*np.cos(phi)
                local_median_vector[1] = r*np.sin(theta)*np.sin(phi)
                local_median_vector[2] = r*np.cos(theta)
                phi = -phi
                theta = -theta

                # ## get mean vector
                # sp_mean_vector = np.mean(vectors[i_t][i_p], axis=0)
                # r = sp_mean_vector[0]
                # theta = sp_mean_vector[1]
                # phi = sp_mean_vector[2]
                # mean_vector = np.zeros(3)
                # mean_vector[0] = r*np.sin(theta)*np.cos(phi)
                # mean_vector[1] = r*np.sin(theta)*np.sin(phi)
                # mean_vector[2] = r*np.cos(theta)
                # phi = -phi
                # theta = -theta

                Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
                ii_m = 0
                for i_m in range(num_ensemble_members):
                    if(depths[i_t][i_p][i_m] >= depth_threshold):
                        tmp = vectors[i_t][i_p][i_m]
                        vec = np.zeros(3)
                        vec[0] = tmp[0]*np.sin(tmp[1])*np.cos(tmp[2])
                        vec[1] = tmp[0]*np.sin(tmp[1])*np.sin(tmp[2])
                        vec[2] = tmp[0]*np.cos(tmp[1])
                        # # debugging
                        # if( i_t == 0 and i_p ==179):
                        #     print('sp_vec[', i_m, '] = ', tmp)
                        #     # print('phi = ', phi, 'theta = ', theta)
                        #     # print('median_vector = ', median_vectors[i_t][i_p])
                        #     # print('local_median_vector = ', local_median_vector)
                        #     print('vec: ', vec)
                        #     if( i_m == 20):
                        #         # get angle between median and vec
                        #         print('angle = ', np.arccos(np.dot(local_median_vector, vec)/(np.linalg.norm(local_median_vector)*np.linalg.norm(vec))))
                                
                        # print("vec[", i_t, ", ", i_p, ", ", i_m, "] = ", vec)
                        vec = np.dot(Ry, np.dot(Rz, vec))
                        # Debugging
                        # if( i_t == 0 and i_p ==179):
                        #     print('vec1: ', vec)
                        vec =  vec * (max_vectors[i_t][i_p][0]/(np.absolute(vec[2]) + 1.0e-10))
                        # Deubugin
                        # if( i_t == 0 and i_p ==179):
                        #     print('vec2: ', vec)
                        #     if( i_m == 20):
                        #         ## angle between [0, 0, 1] and vec
                        #         print('angle1 = ', np.arccos(np.dot([0, 0, 1], vec)/(np.linalg.norm([0, 0, 1])*np.linalg.norm(vec))))
                        #     sp_vec = vf_utils.cartesian2Spherical(vec)
                        #     print('sp_vec2: ', sp_vec)
                        local_XX[ii_m][0] = vec[0]
                        local_XX[ii_m][1] = vec[1]
                        ii_m = ii_m + 1
                local_X = local_XX[0:ii_m]
                pca = PCA(n_components=2)
                pca.fit(local_X)
                pca_components = pca.components_
                # pca_local_X = pca.transform(local_X)
                # mins = np.min(pca_local_X, axis=0)
                # maxs = np.max(pca_local_X, axis=0)
                # v0_scale = np.maximum(np.absolute(mins[0]), np.absolute(maxs[0]))
                # v1_scale = np.maximum(np.absolute(mins[1]), np.absolute(maxs[1]))
                pca_mean = pca.mean_
                pca_variance = pca.explained_variance_
                v0_scale = pca_variance[0]
                v1_scale = pca_variance[1]
                if(np.absolute(v0_scale) < 1.0e-20):
                # if(ii_t == 0 and i_p == 14):
                    # print('v0_scale = ', v0_scale, 'v1_scale = ', v1_scale)
                    # print('local_X:', local_X)
                    # print('local_median_vector = ', local_median_vector)
                    # print('max_vectors = ', max_vectors[i_t][i_p])
                    v0_scale = 1.0
                    v1_scale = 1.0

                    # raise ValueError("Invalid v0_scale inside getDirectionalVariations")
                    # print('mins = ', mins, 'maxs = ', maxs)
                directional_variation[ii_t][i_p][0][0] = v0_scale
                directional_variation[ii_t][i_p][0][1] = np.maximum(v1_scale, 0.1*v0_scale)
                directional_variation[ii_t][i_p][1][0] = pca_components[0][0]
                directional_variation[ii_t][i_p][1][1] = pca_components[0][1]
                directional_variation[ii_t][i_p][2][0] = pca_components[1][0]
                directional_variation[ii_t][i_p][2][1] = pca_components[1][1]
                # directional_variation[ii_t][i_p][3][0] = pca_mean[0]
                # directional_variation[ii_t][i_p][3][1] = pca_mean[1]

                # # Debugging
                # if(v0_scale > 50 and  np.absolute(max_vectors[i_t][i_p][1]) < np.pi * 3*0.25):
                #     print("i_t = ", i_t, "i_p = ", i_p)
                #     print("v0_scale = ", v0_scale)
                #     print("v1_scale = ", v1_scale)
                #     print("pca_components = ", pca_components)
                #     print("pca_mean = ", pca_mean)
                #     print('pca variance = ', pca.explained_variance_)
                #     print( 'min_vectors = ', min_vectors[i_t][i_p])
                #     print( 'median_vectors = ', median_vectors[i_t][i_p])
                #     print( 'max_vectors = ', max_vectors[i_t][i_p])
                #     for i_m in range(num_ensemble_members):
                #         print('local_X[', i_m, '] = ', local_X[i_m], 'pca_local_X[', i_m, '] = ', pca_local_X[i_m])
                #     raise ValueError("Invalid scale")   


                
    return directional_variation
                

def getBasesInfo(data, time_range, domain):
    """
    purpose: compute the bases information of the vectors
    input:    
    """
    points = []
    polygons = []
    resolution = 10
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    scale = 1
    num_ensemble_members = int(data["num_ensemble_members"][0])
    num_points = int(data['positions'][0].shape[0])
    print("num_points = ", num_points, "num_ensemble_members = ", num_ensemble_members, 'time_range =', time_range)
    print("domain = ", domain)
    local_projected_vectors = np.zeros([num_ensemble_members, 2])
    getMinMedianMaxVectors_old(data, 0.0, time_range, domain, mean_flag = False)
    for i_t in range(time_range[0], time_range[1]+1):
        for i_pt in range(num_points):
            print('data[position] =', data['positions'][0][i_pt])
            print('data[max_vectors] =', data['max_vectors'][0][i_t][i_pt])
            print('data[min_vectors] =', data['min_vectors'][0][i_t][i_pt])
            if(domain[0][0] - 1.0e-10 <= data['positions'][0][i_pt][0] and data['positions'][0][i_pt][0] <= domain[0][1] + 1.0e-10 and
                domain[1][0] - 1.0e-10 <= data['positions'][0][i_pt][1] and data['positions'][0][i_pt][1] <= domain[1][1] + 1.0e-10 and
                domain[2][0] - 1.0e-10 <= data['positions'][0][i_pt][2] and data['positions'][0][i_pt][2] <= domain[2][1] + 1.0e-10 and 
                data["max_vectors"][0][i_t][i_pt][0] > 0.001 and data["min_vectors"][0][i_t][i_pt][0] > 0.001):

                print ("i_t = ", i_t, "i_pt = ", i_pt)
                median_vector = np.zeros(3)
                
                max_vec = data["max_vectors"][0][i_t][i_pt]
                # tmp_max_vec = np.zeros(3)
                # tmp_max_vec[0] = max_vec[0]*np.sin(max_vec[1])*np.cos(max_vec[2])
                # tmp_max_vec[1] = max_vec[0]*np.sin(max_vec[1])*np.sin(max_vec[2])
                # tmp_max_vec[2] = max_vec[0]*np.cos(max_vec[1])
                # min_vec = data["min_vectors"][0][i_t][i_pt]
                # tmp_min_vec = np.zeros(3)
                # tmp_min_vec[0] = min_vec[0]*np.sin(min_vec[1])*np.cos(min_vec[2])
                # tmp_min_vec[1] = min_vec[0]*np.sin(min_vec[1])*np.sin(min_vec[2])
                # tmp_min_vec[2] = min_vec[0]*np.cos(min_vec[1])
                # # angle betwen max and min vectors
                # angle_min_max = np.arctan2(np.linalg.norm(np.cross(tmp_min_vec, tmp_max_vec)), np.dot(tmp_min_vec, tmp_max_vec))

                r = data["median_vectors"][0][i_t][i_pt][0]
                theta = data["median_vectors"][0][i_t][i_pt][1]
                phi = data["median_vectors"][0][i_t][i_pt][2]
                median_vector[0] = r*np.sin(theta)*np.cos(phi)
                median_vector[1] = r*np.sin(theta)*np.sin(phi)
                median_vector[2] = r*np.cos(theta)
                theta = -theta
                phi = -phi
                Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
                for i_m in range(num_ensemble_members):
                    tmp = data["vectors"][0][i_t][i_pt][i_m]
                    print('tmp = ', tmp)
                    vec = np.zeros(3)
                    vec[0] = tmp[0]*np.sin(tmp[1])*np.cos(tmp[2])
                    vec[1] = tmp[0]*np.sin(tmp[1])*np.sin(tmp[2])
                    vec[2] = tmp[0]*np.cos(tmp[1])
                    print('vec[', i_m, '] = ', vec)
                    vec = np.dot(Ry, np.dot(Rz, vec))
                    print('vec1[', i_m, '] = ', vec)
                    print('max_vector = ', max_vec)
                    vec =  vec * (max_vec[0]/(vec[2] + 1.0e-10))
                    print('vec2[', i_m, '] = ', vec)
                    local_projected_vectors[i_m][0] = vec[0]
                    local_projected_vectors[i_m][1] = vec[1]

                pca = PCA(n_components=2)
                pca.fit(local_projected_vectors)
                print("local_projected_vectors = ", local_projected_vectors)
                pca_components = pca.components_
                PCA_vectors = pca.transform(local_projected_vectors)
                mins = np.min(PCA_vectors, axis=0)
                maxs = np.max(PCA_vectors, axis=0)
                v0_scale = np.maximum(np.absolute(mins[0]), np.absolute(maxs[0]))
                v1_scale = np.maximum(np.absolute(mins[1]), np.absolute(maxs[1]))
                v0 = pca_components[0]#*v0_scale
                v1 = pca_components[1]#*v1_scale
                angle = np.arctan2(v0[1], v0[0])

                phi = -phi
                theta = -theta
                print("theta = ", theta, "phi = ", phi)
                v0_scale = np.maximum(v0_scale, v1_scale) 
                v1_scale = np.maximum(0.05*v0_scale, v1_scale)
                print("v0_scale = ", v0_scale, "v1_scale = ", v1_scale)

                Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
                phi_vals = np.linspace(0, 2*np.pi, resolution)
                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* v0_scale 
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* v1_scale

               
                x = x0*np.cos(angle) - y0*np.sin(angle) + pca.mean_[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) + pca.mean_[1]
                # plot super elipse
                # fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', name='Super Elipse'))
                # fig.show()
                # print("x = ", x)
                # print("y = ", y)
                for i in range (resolution):
                    pt = np.dot(Rz, np.dot(Ry, np.array([x[i], y[i], 0])))
                    
                    pt = pt*scale + data["positions"][0][i_pt]
                    
                    points.append([pt[0], pt[1], pt[2]])
                
                polygons.append(resolution)
                for i in range (resolution):
                    polygons.append(points_id + i)
                old_points_id = points_id
                points_id = points_id + resolution

                pt = np.dot(Rz, np.dot(Ry, np.array([0, 0, max_vec[0]])))
                pt = pt + data["positions"][0][i_pt]
                points.append([pt[0], pt[1], pt[2]])
                tip_id = points_id
                points_id = points_id + 1

                for i in range (resolution-1):
                    polygons.append(3)
                    polygons.append(old_points_id + i)
                    polygons.append(tip_id)
                    polygons.append(old_points_id + i + 1)

    print("points_id = ", points_id, 'len(points) =', len(points), len(polygons))
    vector_glyph = dash_vtk.GeometryRepresentation(
                children=[
                    dash_vtk.PolyData(
                        id="vtk-polydata",
                        points=np.array(points).ravel(),
                        polys=np.array(polygons),
                        # children=[
                        #     dash_vtk.PointData(
                        #         [
                        #             dash_vtk.DataArray(
                        #                 id="vtk-array",
                        #                 registration="setScalars",
                        #                 name="time_step",
                        #                 values=scalars_selected_numpy,
                        #             )
                        #         ]
                        #     )
                        # ],
                    )],
                property={'opacity': 1.0},
                # colorMapPreset='coolwarm',
                # colorDataRange=[time_range[0], time_range[1]]
            )
    return vector_glyph
            

def dataDepth_1(data, num_selected_points):
    """
    purpose: compute the data depth of a set of vectors
    input:
      data: dictionary containing the following
        num_time_steps: number of time steps
        num_points: number of positions
        dimensions: dimension of the vectors
        num_ensemble_members: number of ensemble members
        positions: 3D array of shape (1, num_time_steps, num_points, 2) where the last dimension contains the x and y coordinates of the positions
        vectors: 4D array of shape (1, num_time_steps, num_points, num_ensemble_members, dimensions) where the last dimension contains the vector components
      num_selected_points: number of points used to define the convex hull
    output:
      depths: 3D array of shape (num_time_steps, num_points, num_ensemble_members) where the last dimension contains the data depth of the ensemble members
    """
    num_time_steps = int(data["num_time_steps"][0])
    num_positions = int(data["num_points"][0])
    dimensions = int(data["dimensions"][0])
    print('dimensions =', dimensions, 'num_selected_points =', num_selected_points)
    num_ensemble_members = int(data["num_ensemble_members"][0])
    indices = np.linspace(0, num_ensemble_members-1, num_ensemble_members, dtype=int)
    depths = np.zeros([num_time_steps, num_positions, num_ensemble_members])
    # compute indices choose num_selected_points
    selected_points_indices = np.array(list(itertools.combinations(indices, num_selected_points)))
    num_convex_hulls = selected_points_indices.shape[0] 
    # print("num_convex_hulls = ", num_convex_hulls)
    # print("dimensions = ", dimensions)
    # print("selected_points_indices = ", selected_points_indices)

    ## Compute convex hulls
    selected_points = np.zeros([num_selected_points, dimensions])
    for t in range(num_time_steps):
        # print("t = ", t)
        for i_pt in range(num_positions):
            # print("i_pt = ", i_pt)
            count = np.zeros(num_ensemble_members)
            for i in range(num_convex_hulls):
                # print("i = ", i)
                if(dimensions == 3):
                    for j in range(num_selected_points):
                        # print("j = ", j)
                        # print(data["vectors"].shape)
                        # print(data["vectors"][0][t][i_pt][selected_points_indices[i][j]])
                        for k in range(dimensions):
                            # print(data["vectors"][0][t][i_pt][selected_points_indices[i][j]][k])
                            selected_points[j][k] = data["vectors"][0][t][i_pt][selected_points_indices[i][j]][k]
                    tmp = insideConvexHull(selected_points,data["vectors"][0][t][i_pt])
                    count = count + tmp
                elif(dimensions == 2):
                    for j in range(num_selected_points):
                        selected_points[j][0] = data["vectors"][0][t][i_pt][selected_points_indices[i][j]][0]
                        selected_points[j][1] = data["vectors"][0][t][i_pt][selected_points_indices[i][j]][2]
                    tmp_pt  = np.zeros([num_ensemble_members, 2])
                    for j in range(num_ensemble_members):
                        tmp_pt[j][0] = data["vectors"][0][t][i_pt][j][0]
                        tmp_pt[j][1] = data["vectors"][0][t][i_pt][j][2]
                    # print("tmp_pt = ", tmp_pt)
                    tmp = insideConvexHull(selected_points, tmp_pt)
                    count = count + tmp

            depths[t][i_pt] = count/num_convex_hulls
            for i_m in range(num_ensemble_members):
                data["depths"][0][t][i_pt][i_m] = count[i_m]/num_convex_hulls
            # print("depths = ", depths[t][i_pt])
            # print("sum depths:", sum(depths[t][i_pt]))
                # conv_hulls.append(sp.spatial.ConvexHull(selected_points))
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(selected_points[:,0], selected_points[:,1], selected_points[:,2], 'ko')
                # for simplex in conv_hulls[i].simplices:
                #     simplex = np.append(simplex, simplex[0])
                #     ax.plot(selected_points[simplex, 0], selected_points[simplex, 1], selected_points[simplex, 2], 'r-')
                # plt.show()
    return depths


def getVariability(data, time_range, domain):
    """
    purpose: compute the variability of the vectors
    input:
      data: dictionary containing the following
    """

    for i_t in range(time_range[0], time_range[1]+1):
        for i_pt in range(int(data["num_points"][0])):
            for i_dim in range(int(data["dimensions"][0])):
                data["variability"][0][i_t][i_pt][i_dim] = data["max_vectors"][0][i_t][i_pt][i_dim] - data["min_vectors"][0][i_t][i_pt][i_dim]

    return 1


@njit
def getMinVectors(vectors, depths, depth_threshold):
    """    
    purpose: compute the minimum vectors
    input:
    vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components
    
    output:
    min_vectors: 2D array of shape (num_points, 3) where the last dimension contains the minimum vector components
    """
    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    min_vectors = np.zeros((num_points, 3))
    for i_pt in range(num_points):
        for i_dim in range(3):
            min_vectors[i_pt][i_dim] = 1.0e+16
            for i_memeber in range(num_ensemble_members):
                if(depths[i_pt][i_memeber] >= depth_threshold):
                    if(vectors[i_pt][i_memeber][i_dim] < min_vectors[i_pt][i_dim]):
                        min_vectors[i_pt][i_dim] = vectors[i_pt][i_memeber][i_dim]
            if(min_vectors[i_pt][i_dim] == 1.0e+16):
                min_vectors[i_pt][i_dim] = 0.0
    return min_vectors


@njit
def getMaxVectors(vectors, depths, depth_threshold):
    """
        purpose: compute the maximum vectors
    input:
    vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components

    output:
    max_vectors: 2D array of shape (num_points, 3) where the last dimension contains the maximum vector components

    """
    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    max_vectors = np.zeros((num_points, 3))
    angle = 0.0

    # for i_pt in range(num_points):
    #         for i_memeber in range(num_ensemble_members):
    #             for j_memeber in range(num_ensemble_members):
    #                 if(depths[i_pt][i_memeber] >= depth_threshold and 
    #                     depths[i_pt][j_memeber] >= depth_threshold):
    #                     sp_coord0 = vectors[i_pt][i_memeber]
    #                     sp_coord1 = vectors[i_pt][j_memeber]
    #                     vec0 = np.zeros(3)
    #                     vec1 = np.zeros(3)
    #                     vec0[0] = sp_coord0[0]*np.sin(sp_coord0[1])*np.cos(sp_coord0[2])
    #                     vec0[1] = sp_coord0[0]*np.sin(sp_coord0[1])*np.sin(sp_coord0[2])
    #                     vec0[2] = sp_coord0[0]*np.cos(sp_coord0[1])
    #                     vec1[0] = sp_coord1[0]*np.sin(sp_coord1[1])*np.cos(sp_coord1[2])
    #                     vec1[1] = sp_coord1[0]*np.sin(sp_coord1[1])*np.sin(sp_coord1[2])
    #                     vec1[2] = sp_coord1[0]*np.cos(sp_coord1[1])
    #                     angle_vec0_vec1 = np.arctan2(np.linalg.norm(np.cross(vec0, vec1)), np.dot(vec0, vec1))
    #                     angle_vec0_vec1 = np.abs(angle_vec0_vec1)
    #                     if(angle_vec0_vec1 > angle):
    #                         angle = angle_vec0_vec1
    #                         max_vectors[i_pt][1] = angle_vec0_vec1
    #                     if(vectors[i_pt][i_memeber][0] > max_vectors[i_pt][0]):
    #                         max_vectors[i_pt][0] = vectors[i_pt][i_memeber][0]

    #         for i_dim in range(3):
    #             if(max_vectors[i_pt][i_dim] == -1.0e+16):
    #                 max_vectors[i_pt][i_dim] = 0.0

    for i_pt in range(num_points):
        for i_dim in range(3):
            max_vectors[i_pt][i_dim] = -1.0e+16
            for i_memeber in range(num_ensemble_members):
                if(depths[i_pt][i_memeber] >= depth_threshold):
                    if(vectors[i_pt][i_memeber][i_dim] > max_vectors[i_pt][i_dim]):
                        max_vectors[i_pt][i_dim] = vectors[i_pt][i_memeber][i_dim]
            if(max_vectors[i_pt][i_dim] == -1.0e+16):
                max_vectors[i_pt][i_dim] = 0.0
    return max_vectors

@njit
def getMedianVectors(vectors, depths, depth_threshold):
    """
    purpose: compute the median vectors
    input:
    vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components
    
    output:
    median_vectors: 2D array of shape (num_points, 3) where the last dimension contains the median vector components

    """

    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    median_vectors = np.zeros((num_points, 3))
    for i_pt in range(num_points):
        median_prob = depth_threshold
        for i_memeber in range(num_ensemble_members):
            if(depths[i_pt][i_memeber] >= median_prob):
                for i_dim in range(3):
                    median_vectors[i_pt][i_dim] = vectors[i_pt][i_memeber][i_dim]
                median_prob = depths[i_pt][i_memeber]   

    return median_vectors

@njit
def getMeanVectors(vectors, depths, depth_threshold):
    """
    purpose: compute the mean vectors
    input:
    vectors: 3D array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components
    
    output:
    mean_vectors: 2D array of shape (num_points, 3) where the last dimension contains the mean vector components
    
    """
    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    mean_vectors = np.zeros((num_points, 3))
    for i_pt in range(num_points):
        count = 0
        for i_memeber in range(num_ensemble_members):
            if(depths[i_pt][i_memeber] >= depth_threshold):
                count = count + 1
                for i_dim in range(3):
                    mean_vectors[i_pt][i_dim] = mean_vectors[i_pt][i_dim] + vectors[i_pt][i_memeber][i_dim]
        if(count > 0):
            mean_vectors[i_pt] = mean_vectors[i_pt]/count
    return mean_vectors

@njit
def getMinMedianMaxVectors2(positions, vectors, vector_depths, depth_threshold, time_range, 
                            domain, mean_flag = False, mag_threshold_min = -1.0, mag_threshold_max = 1.0e+16):
    """
    purpose: compute the minimum, median, and maximum vectors
    input: positions: 3D s
    """
    
    # print('enter getMinMedianMaxVectors2() ...')
    # print('mag_threshold_min =', mag_threshold_min, 'mag_threshold_max =', mag_threshold_max)

    # print('median_flag =', mean_flag)
    num_points = positions.shape[0]
    num_ensembles = vectors.shape[2]
    dimensions = vectors.shape[3]
    num_time_steps = time_range[1] - time_range[0] + 1
    min_vectors = np.ones((num_time_steps, num_points, dimensions)) * 1.0e+16
    max_vectors = np.ones((num_time_steps, num_points, dimensions)) * -1.0e+16
    median_vectors = np.zeros((num_time_steps, num_points, dimensions))
    variability = np.zeros((num_time_steps, num_points, dimensions))
    # for i_t in range(time_range[0], time_range[1]+1):
    #     ii_t = i_t - time_range[0]
    #     for i_pt in range(num_points):
    #         for i_m in range(num_ensembles):
    #             for i_dim in range(dimensions):
    #                 if(vectors[i_t][i_pt][i_m][i_dim] < min_vectors[ii_t][i_pt][i_dim]):
    #                     min_vectors[ii_t][i_pt][i_dim] = vectors[i_t][i_pt][i_m][i_dim]

    # ## min vectors
    # print('computing min vectors ...')
    # print('time_range =', time_range)
    # print('num_points =', num_points)
    # print('num_ensembles =', num_ensembles)
    # print('dimensions =', dimensions)
    # print('depth_threshold =', depth_threshold)

    for i_t in range(time_range[0], time_range[1]+1):
        # print('i_t =', i_t)
        ii_t = i_t - time_range[0]
        for i_pt in range(num_points):
                # print('i_pt =', i_pt)
                min_vectors[ii_t][i_pt][0] = 1.0e+16
                for i_m in range(num_ensembles):
                    if(vector_depths[i_t][i_pt][i_m] >= depth_threshold):
                        # print('i_t =', i_t, 'i_pt =', i_pt, 'i_m =', i_m)
                        # print('vector_depths[', i_t, '][', i_pt, '][', i_m, '] =', vector_depths[i_t][i_pt][i_m])
                        if(vectors[i_t][i_pt][i_m][0] < min_vectors[ii_t][i_pt][0] and 
                            vectors[i_t][i_pt][i_m][0] >= mag_threshold_min and vectors[i_t][i_pt][i_m][0] <= mag_threshold_max):
                            # print('mag_threshold_min =', mag_threshold_min, 'vetor =', vectors[i_t][i_pt][i_m][0], 'mag_threshold_max =', mag_threshold_max)
                            min_vectors[ii_t][i_pt][0] = vectors[i_t][i_pt][i_m][0]
                        min_vectors[ii_t][i_pt][1] = 0.0
                        min_vectors[ii_t][i_pt][2] = 0.0

    # ## max vectors
    # for i_t in range(time_range[0], time_range[1]+1):
    #     ii_t = i_t - time_range[0]
    #     for i_pt in range(num_points):
    #         for i_m in range(num_ensembles):
    #             for i_dim in range(dimensions):
    #                 if(vectors[i_t][i_pt][i_m][i_dim] > max_vectors[ii_t][i_pt][i_dim]):
    #                     max_vectors[ii_t][i_pt][i_dim] = vectors[i_t][i_pt][i_m][i_dim]

    ## max vectors
    # print('computing max vectors ...')
    for i_t in range(time_range[0], time_range[1]+1):
        ii_t = i_t - time_range[0]
        for i_pt in range(num_points):
            angle = 0.0
            for i_m in range(num_ensembles):
                    # print('i_m =', i_m, 'i_t =', i_t, 'i_pt =', i_pt, 'depth_threshold =', depth_threshold)
                    # print('vector_depths[', i_t, '][', i_pt, '][', i_m, '] =', vector_depths[i_t][i_pt][i_m])
                    if(vector_depths[i_t][i_pt][i_m] >= depth_threshold):
                        if(vectors[i_t][i_pt][i_m][0] > max_vectors[ii_t][i_pt][0] and
                            vectors[i_t][i_pt][i_m][0] >= mag_threshold_min and vectors[i_t][i_pt][i_m][0] <= mag_threshold_max):
                            max_vectors[ii_t][i_pt][0] = vectors[i_t][i_pt][i_m][0]
                    
                        for j_m in range(num_ensembles):
                            if(vector_depths[i_t][i_pt][j_m] >= depth_threshold and 
                                vectors[i_t][i_pt][j_m][0] >= mag_threshold_min and vectors[i_t][i_pt][j_m][0] <= mag_threshold_max):
                                # print('i_m =', i_m, 'j_m =', j_m, 'i_t =', i_t, 'i_pt =', i_pt)
                                sp_coord0 = vectors[i_t][i_pt][i_m]
                                sp_coord1 = vectors[i_t][i_pt][j_m]
                                vec0 = np.zeros(3)
                                vec1 = np.zeros(3)
                                vec0[0] = sp_coord0[0]*np.sin(sp_coord0[1])*np.cos(sp_coord0[2])
                                vec0[1] = sp_coord0[0]*np.sin(sp_coord0[1])*np.sin(sp_coord0[2])
                                vec0[2] = sp_coord0[0]*np.cos(sp_coord0[1])
                                vec1[0] = sp_coord1[0]*np.sin(sp_coord1[1])*np.cos(sp_coord1[2])
                                vec1[1] = sp_coord1[0]*np.sin(sp_coord1[1])*np.sin(sp_coord1[2])
                                vec1[2] = sp_coord1[0]*np.cos(sp_coord1[1])
                                tmp = np.cross(vec0, vec1)
                                tmp2 = np.sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2])
                                angle_vec0_vec1 = np.arctan2(tmp2, np.dot(vec0, vec1))
                                # angle_vec0_vec1 = np.arctan2(np.linalg.norm(np.cross(vec0, vec1)), np.dot(vec0, vec1))
                                angle_vec0_vec1 = np.absolute(angle_vec0_vec1)
                                # print('angle_vec0_vec1('    , i_m, ',', j_m, ') =', angle_vec0_vec1)
                                ## debugging
                                # if (i_t == 0 and i_pt == 643):
                                #     print('angle_vec0_vec1(', i_m, ',', j_m, ') =', angle_vec0_vec1)
                                if(angle_vec0_vec1 > angle):
                                    max_vectors[ii_t][i_pt][1] = angle_vec0_vec1
                                    angle = angle_vec0_vec1
            for i_dim in range(3):
                if(max_vectors[ii_t][i_pt][i_dim] == -1.0e+16):
                    max_vectors[ii_t][i_pt][i_dim] = 0.0

    ## median vectors
    if(mean_flag):
        for i_t in range(time_range[0], time_range[1]+1):
            ii_t = i_t - time_range[0]
            for i_pt in range(num_points):
                for i_dim in range(dimensions):
                    median_prob = depth_threshold
                    count = 0
                    for i_m in range(num_ensembles):
                        if(vector_depths[i_t][i_pt][i_m] >= median_prob):
                            median_vectors[ii_t][i_pt][i_dim] = median_vectors[ii_t][i_pt][i_dim] + vectors[i_t][i_pt][i_m][i_dim]
                            count = count + 1
                    if(count > 0):
                        median_vectors[ii_t][i_pt][i_dim] = median_vectors[ii_t][i_pt][i_dim]/count
    else:
        for i_t in range(time_range[0], time_range[1]+1):
            ii_t = i_t - time_range[0]
            for i_pt in range(num_points):
                for i_dim in range(dimensions):
                    median_prob = depth_threshold
                    for i_m in range(num_ensembles):
                        if(vector_depths[i_t][i_pt][i_m] >= median_prob):
                            median_vectors[ii_t][i_pt][i_dim] = vectors[i_t][i_pt][i_m][i_dim]
                            median_prob = vector_depths[i_t][i_pt][i_m]
 
    ## variability
    for i_t in range(time_range[0], time_range[1]+1):
        ii_t = i_t - time_range[0]
        for i_pt in range(num_points):
            for i_dim in range(dimensions):
                variability[ii_t][i_pt][i_dim] = max_vectors[ii_t][i_pt][i_dim] - min_vectors[ii_t][i_pt][i_dim]

    # print('leaving getMinMedianMaxVectors2() ...')

    return min_vectors, median_vectors, max_vectors, variability


def getMinMedianMaxVectors(data, depth_threshold, time_range, domain, mean_flag = False, mag_threshold_min = -1.0, mag_threshold_max = 1.0e+16):
    """
    purpose: compute the minimum, median, and maximum vectors
    input:
    data: dictionary containing the following
        num_time_steps: number of time steps
        num_points: number of positions
        dimensions: dimension of the vectors
        num_ensemble_members: number of ensemble members

    """
    
    # print(' entering getMinMedianMax(...)')
    tmp_min_vectors, tmp_median_vectors, tmp_max_vectors, tmp_variability = getMinMedianMaxVectors2(data["positions"][0], 
                                                                                   data["vectors"][0], 
                                                                                   data["depths"][0], 
                                                                                   depth_threshold, 
                                                                                   time_range, 
                                                                                   domain, mean_flag, mag_threshold_min, mag_threshold_max)
    data["min_vectors"][0][time_range[0]:time_range[1]+1] = tmp_min_vectors
    data["median_vectors"][0][time_range[0]:time_range[1]+1] = tmp_median_vectors
    data["max_vectors"][0][time_range[0]:time_range[1]+1] = tmp_max_vectors
    data["variability"][0][time_range[0]:time_range[1]+1] = tmp_variability
    # for i_t in range(time_range[0], time_range[1]+1):
    #     for i_pt in range(int(data["num_points"][0])):
    #         if(data['max_vectors'][0][i_t][i_pt][0] > 0.001 and 
    #            (data['variability'][0][i_t][i_pt][0] < 0.001 or data['variability'][0][i_t][i_pt][1] > 3.00 or data['variability'][0][i_t][i_pt][2] > 3.00)):
    #             print('i_t =', i_t, 'i_pt =', i_pt)
    #             print('position:', data['positions'][0][i_pt])
    #             print('min vec:', data['min_vectors'][0][i_t][i_pt])
    #             print('max vec:', data['max_vectors'][0][i_t][i_pt])
    #             print('median vec:', data['median_vectors'][0][i_t][i_pt])
    #             print('variability:', data['variability'][0][i_t][i_pt])
    #             print('vectors:', data['vectors'][0][i_t][i_pt])
    # getVariability(data, time_range, domain)

    # print('positions:', data['positions'][0][0:5])
    # print('Max vecs:', data['max_vectors'][0][0][0:5])
    # print('leaving getMinMedianMax(...)')

    # print(' leaving getMinMedianMax(...)' )
    return 1


def getMinMedianMaxVectors_old(data, depth_threshold, time_range, domain, mean_flag = False):
    """
    purpose: compute the minimum, median, and maximum vectors
    input:
    data: dictionary containing the following
        num_time_steps: number of time steps
        num_points: number of positions
        dimensions: dimension of the vectors
        num_ensemble_members: number of ensemble members

    """
    # print('depth_threshold =', depth_threshold)
    # initialize min median and max
    for t in range(time_range[0], time_range[1]+1):
        for i_pt in range(int(data["num_points"][0])):
            for i_dim in range(int(data["dimensions"][0])):
                data["min_vectors"][0][t][i_pt][i_dim] = 1.0e+16
                data["max_vectors"][0][t][i_pt][i_dim] = -1.0e+16
                data["median_vectors"][0][t][i_pt][i_dim] = 0.0

    if domain is None:
        domain = data["domain"][0]

    num_time_steps = int(data["num_time_steps"][0])
    num_positions = int(data["num_points"][0])
    dimensions = int(data["dimensions"][0])
    num_ensemble_members = int(data["num_ensemble_members"][0])
    # print ('time_range =', time_range)
    if( len(time_range) != 2):
        raise NameError('Incorrect time range')
    for t in range(time_range[0], time_range[1]+1):
        data["max_vectors"][0][t] = getMaxVectors(data["vectors"][0][t], data["depths"][0][t], depth_threshold)
        data["min_vectors"][0][t] = getMinVectors(data["vectors"][0][t], data["depths"][0][t], depth_threshold)
        data["median_vectors"][0][t] = getMedianVectors(data["vectors"][0][t], data["depths"][0][t], depth_threshold)
        if(mean_flag):
            data["median_vectors"][0][t] = getMeanVectors(data["vectors"][0][t], data["depths"][0][t], depth_threshold)

        ### old with no gpu ###
        # for i_pt in range(num_positions):
        #     # check if data is inside domain
        #     inside_domain = True
        #     for i_dim in range(dimensions):
        #             # print("domain[i_dim][0] = ", domain[i_dim][0])
        #             # print("data = ", data["positions"][0][i_pt][i_dim])
        #         if(domain[i_dim][0] - np.finfo(float).eps > data["positions"][0][i_pt][i_dim] or 
        #              data["positions"][0][i_pt][i_dim] > domain[i_dim][1] + np.finfo(float).eps):
        #             inside_domain = False
        #             break
        #     # print('t:', t, 'i_pt:', i_pt, 'inside_domain:', inside_domain)
        #     # print('position:', data['positions'][0][i_pt])
        #     # print('domain:', domain)
                
        #     if( inside_domain):
        #         # print('inside domain')
        #         median_prob = depth_threshold
        #         for i_memeber in range(num_ensemble_members):
        #             # print('t:', t, 'i_pt:', i_pt, 'i_member:', i_memeber, 'depth_threshold:', depth_threshold, 'depth:', data["depths"][0][t][i_pt][i_memeber] )
        #             if (data["depths"][0][t][i_pt][i_memeber] >= depth_threshold):
        #                 ## update max vector
        #                 for i_dim in range(3):
        #                     if(data["vectors"][0][t][i_pt][i_memeber][i_dim] > data["max_vectors"][0][t][i_pt][i_dim]):
        #                         data["max_vectors"][0][t][i_pt][i_dim] = data["vectors"][0][t][i_pt][i_memeber][i_dim]
        #                 ## update min vector
        #                 for i_dim in range(3):
        #                     if(data["vectors"][0][t][i_pt][i_memeber][i_dim] < data["min_vectors"][0][t][i_pt][i_dim]):
        #                         data["min_vectors"][0][t][i_pt][i_dim] = data["vectors"][0][t][i_pt][i_memeber][i_dim]
                   

        #             # print('max vec:', data["max_vectors"][0][t][i_pt])
        #             ## update median vector
        #             if(data["depths"][0][t][i_pt][i_memeber] >= median_prob):
        #                 for i_dim in range(3):
        #                     data["median_vectors"][0][t][i_pt][i_dim] = data["vectors"][0][t][i_pt][i_memeber][i_dim]
        #                     median_prob = data["depths"][0][t][i_pt][i_memeber]
                
        #         ## update min and max vectors where no vectors has a probability greater than the threshold
        #         for i_dim in range(3):
        #             if(data["min_vectors"][0][t][i_pt][i_dim] == 1.0e+16):
        #                 data["min_vectors"][0][t][i_pt][i_dim] = 0.0
        #             if(data["max_vectors"][0][t][i_pt][i_dim] == -1.0e+16):
        #                 data["max_vectors"][0][t][i_pt][i_dim] = 0.0
                        
        #         # if(t==0 and i_pt==1):
        #         #     print('min =', data["min_vectors"][0][t][i_pt])
        #         #     print('median =', data["median_vectors"][0][t][i_pt])
        #         #     print('max =', data["max_vectors"][0][t][i_pt])
        ### end old with no gpu ###
    # compute verctor variability
    getVariability(data, time_range, domain)

    # print('positions:', data['positions'][0][0:5])
    # print('Max vecs:', data['max_vectors'][0][0][0:5])
    # print('leaving getMinMedianMax(...)')

    return 1
