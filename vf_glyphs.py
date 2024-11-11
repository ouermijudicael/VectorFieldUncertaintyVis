import sys
import os
current_path = os.getcwd()
sys.path.insert(0, current_path)
import time
import numpy as np 
import vtk
# import vf_utils
import vf_statistics

import plotly.graph_objects as go
from sklearn.decomposition import PCA

import dash_vtk 
from dash_vtk.utils import to_mesh_state

from numba import njit, jit


def getBoxPolyData(domain):
    """
    purpose: get a box polydata
    input: domain - domain of the box
    """
    points =[ [domain[0][0], domain[1][0], domain[2][0]], 
                [domain[0][1], domain[1][0], domain[2][0]],
                [domain[0][0], domain[1][1], domain[2][0]],
                [domain[0][1], domain[1][1], domain[2][0]],
                [domain[0][0], domain[1][0], domain[2][1]],
                [domain[0][1], domain[1][0], domain[2][1]],
                [domain[0][0], domain[1][1], domain[2][1]],
                [domain[0][1], domain[1][1], domain[2][1]]]
    polygons = [4, 0, 1, 3, 2, 
                4, 4, 5, 7, 6, 
                4, 0, 1, 5, 4,
                4, 2, 3, 7, 6, 
                4, 0, 2, 6, 4,
                4, 1, 3, 7, 5]
    return points, polygons




def mapPointTo(point_in, position, vector, scale=1.0):
    """
    purpose: map a point to a position and direction
    input: point - 3D point to be mapped

    """
    
    
    point = np.array(point_in)
    # rotate the point about y axis
    theta = vector[1]
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    point = np.dot(Ry, point)
    # rotate the point about z axis
    phi = vector[2]
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    point = np.dot(Rz, point)
    # scale the point
    point = scale*point
    # translate the point
    point = point + position

    return point


# @njit
def buildArrowGlyphs(positions, vectors, scale=1.0, resolution=10, magnitude_threshold = 0.001):
    """
    
    """

    # get number of glyphys
    num_glyphs = 0
    for i in range(len(vectors)):
        # print('vectors[i]:', vectors[i])
        if(np.absolute(vectors[i][0]) > magnitude_threshold):
            num_glyphs += 1
            # print('vectors[i]:', vectors[i])

    points = np.zeros((num_glyphs*(3*resolution + 1)*3, ))
    scalars = np.zeros((num_glyphs*(3*resolution + 1), ))
    polygons = np.zeros((num_glyphs*((2*(resolution + 1) + 5 * resolution) + 4*(resolution-1))), dtype=np.int32)
    # polygons = np.zeros((num_of_glyphs*((2*(resolution + 1) + 5*resolution + 4*(resolution-1)) )), dtype=np.int32)
    vec = np.zeros((3, ))
    g_id = 0
    point_id = 0
    polygon_id = 0
    phi0 = np.linspace(0, 2*np.pi, resolution)
    for i in range(len(vectors)):
        if(np.absolute(vectors[i][0]) > magnitude_threshold):
            vec[0] = vectors[i][0]*np.sin(vectors[i][1])*np.cos(vectors[i][2])
            vec[1] = vectors[i][0]*np.sin(vectors[i][1])*np.sin(vectors[i][2])
            vec[2] = vectors[i][0]*np.cos(vectors[i][1])

            theta = vectors[i][1]
            phi = vectors[i][2]

            Ry = np.zeros((3, 3))
            Ry[0][0] = np.cos(theta)
            Ry[0][2] = np.sin(theta)
            Ry[1][1] = 1
            Ry[2][0] = -np.sin(theta)
            Ry[2][2] = np.cos(theta)
            Rz = np.zeros((3, 3))
            Rz[0][0] = np.cos(phi)
            Rz[0][1] = -np.sin(phi)
            Rz[1][0] = np.sin(phi)
            Rz[1][1] = np.cos(phi)
            Rz[2][2] = 1
            
            # base
            base_radius = 0.05*vec[0]
            x = base_radius*np.cos(phi0)
            y = base_radius*np.sin(phi0)
            z = np.zeros(resolution)
            polygons[polygon_id] = resolution
            polygon_id += 1
            for j in range(resolution):
                polygons[polygon_id] = point_id + j
                polygon_id += 1

            for j in range(resolution):
                pt = np.dot(Ry, np.array([x[j], y[j], z[j]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + positions[i]
                points[3*(point_id + j) + 0] = pt[0]
                points[3*(point_id + j) + 1] = pt[1]
                points[3*(point_id + j) + 2] = pt[2]
                # points[3*(point_id + j) + 0] = x[j] + positions[i][0]
                # points[3*(point_id + j) + 1] = y[j] + positions[i][1]
                # points[3*(point_id + j) + 2] = z[j] + positions[i][2]
            old_point_id = point_id
            point_id += resolution

            # shaft
            z_shaft = 0.8*vec[0]*np.ones(resolution)
            for j in range(resolution):
                polygons[polygon_id] = 4
                polygon_id += 1
                polygons[polygon_id] = old_point_id + j
                polygon_id += 1
                polygons[polygon_id] = point_id + j
                polygon_id += 1
                polygons[polygon_id] = point_id + (j+1)%resolution
                polygon_id += 1
                polygons[polygon_id] = old_point_id + (j+1)%resolution
                polygon_id += 1
                
            for j in range(resolution): 
                pt = np.dot(Ry, np.array([x[j], y[j], z_shaft[j]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + positions[i]
                points[3*(point_id + j) + 0] = pt[0]
                points[3*(point_id + j) + 1] = pt[1]
                points[3*(point_id + j) + 2] = pt[2]

                # points[3*(point_id + j) + 0] = x[j] + positions[i][0]
                # points[3*(point_id + j) + 1] = y[j] + positions[i][1]
                # points[3*(point_id + j) + 2] = z_shaft[j] + positions[i][2]
            old_point_id = point_id
            point_id += resolution

            # head base
            head_radius = base_radius*1.5
            x_head = head_radius*np.cos(phi0)
            y_head = head_radius*np.sin(phi0)
            z_head = z_shaft
            polygons[polygon_id] = resolution
            polygon_id += 1
            for j in range(resolution):
                polygons[polygon_id] = point_id + j
                polygon_id += 1

            for j in range(resolution):
                pt = np.dot(Ry, np.array([x_head[j], y_head[j], z_head[j]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + positions[i]
                points[3*(point_id + j) + 0] = pt[0]
                points[3*(point_id + j) + 1] = pt[1]
                points[3*(point_id + j) + 2] = pt[2]
            old_point_id = point_id
            point_id += resolution      

            # head top
            tip = np.array([0, 0, vec[0]])
            pt = np.dot(Ry, tip)
            pt = np.dot(Rz, pt)
            pt = pt*scale + positions[i]
            points[3*point_id + 0] = pt[0]  
            points[3*point_id + 1] = pt[1]
            points[3*point_id + 2] = pt[2]
            tip_id = point_id
            point_id += 1
            for j in range(resolution-1):
                polygons[polygon_id] = 3
                polygon_id += 1
                polygons[polygon_id] = old_point_id + j
                polygon_id += 1
                polygons[polygon_id] = tip_id
                polygon_id += 1
                polygons[polygon_id] = old_point_id + (j+1)%resolution
                polygon_id += 1


    return points, polygons



@njit
def getGlyphsMarkers(min_vectors, max_vectors, time_range, num_points, exclude_time_steps):
    # print('Entering getGlyphsMarkers(...)')
    # print('max_vectors:', max_vectors.shape)
    angle_threshold = np.pi*3*0.25
    magnitude_threshold = 1.0e-3
    ii_t = -1
    if( time_range[0] <= exclude_time_steps and exclude_time_steps <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        
        if(i_t != exclude_time_steps):
            indices[ii_t] = i_t  
            ii_t += 1  
    num_time_steps = len(indices)
    glyph_marker = np.zeros((num_time_steps, num_points), dtype=np.int32)
    num_glyphs0 = 0
    num_glyphs1 = 0
    
    ii_t = -1
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            if( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold and
                (np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or
                np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold)):
                glyph_marker[ii_t][i_p] = 2
            elif( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold):
                glyph_marker[ii_t][i_p] = 1


    for i_t in range(num_time_steps):
        for i_p in range(num_points):
            if(glyph_marker[i_t][i_p] ==1):
                num_glyphs0 += 1
            elif(glyph_marker[i_t][i_p] ==2):
                num_glyphs1 += 1

    # if(time_range[0] == exclude_time_steps):
    #     num_time_steps = time_range[1] - time_range[0]
    #     glyph_marker = np.zeros((num_time_steps, num_points), dtype=np.int32)
    #     ii_t = -1
    #     for i_t in range(time_range[0]+1, time_range[1]+1):
    #         ii_t += 1
    #         for i_p in range(num_points):
    #             if( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold and 
    #                (np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or 
    #                 np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold)):
    #                 glyph_marker[ii_t][i_p] = 2
    #             elif( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold):
    #                 if( np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or 
    #                     np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold):
    #                     print ('i_t:', i_t, ' i_p:', i_p, ' max_vectors[i_t][i_p]:', max_vectors[i_t][i_p])
    #                     exit()
    #                 glyph_marker[ii_t][i_p] = 1
    #             # DEBUG
    #             if(i_t ==  1 and i_p == 179):
    #                 print('max_vectors[1][179]:', max_vectors[i_t][179])
    #                 print('glyph_marker:', glyph_marker[ii_t][i_p])
    #                 print('ii_t:', ii_t)
    #     print('max_vectors[1][179]:', max_vectors[1][179])
    #     print('glyph_marker:', glyph_marker[1][i_p])
    # elif(time_range[1] == exclude_time_steps):
    #     num_time_steps = time_range[1] - time_range[0]
    #     glyph_marker = np.zeros((num_time_steps, num_points), dtype=np.int32)
    #     ii_t = -1
    #     for i_t in range(time_range[0], time_range[1]):
    #         ii_t += 1
    #         for i_p in range(num_points):
    #             if( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold and
    #                  (np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or
    #                     np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold)):
    #                 glyph_marker[ii_t][i_p] = 2
    #             elif( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold):
    #                 glyph_marker[ii_t][i_p] = 1
    # elif(time_range[0] <  exclude_time_steps and exclude_time_steps < time_range[1]):
    #     num_time_steps = time_range[1] - time_range[0] 
    #     glyph_marker = np.zeros((num_time_steps, num_points), dtype=np.int32)
    #     ii_t = -1
    #     for i_t in range(time_range[0], exclude_time_steps):
    #         ii_t += 1
    #         for i_p in range(num_points):
    #             if( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold and
    #                 (np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or
    #                 np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold)):
    #                 glyph_marker[ii_t][i_p] = 2
    #             elif( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold):
    #                 glyph_marker[ii_t][i_p] = 1
    #     for i_t in range(exclude_time_steps+1, time_range[1]+1):
    #         ii_t += 1
    #         for i_p in range(num_points):
    #             if( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold and
    #                 (np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or
    #                 np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold)):
    #                 glyph_marker[ii_t][i_p] = 2
    #             elif( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold):
    #                 glyph_marker[ii_t][i_p] = 1
    
    # else:
    #     num_time_steps = time_range[1] - time_range[0] + 1
    #     glyph_marker = np.zeros((num_time_steps, num_points), dtype=np.int32)
    #     ii_t = -1
    #     for i_t in range(time_range[0], time_range[1]+1):
    #         ii_t += 1
    #         for i_p in range(num_points):
    #             if( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold and
    #                 (np.absolute(max_vectors[i_t][i_p][1]) > angle_threshold or
    #                 np.absolute(max_vectors[i_t][i_p][2]) > angle_threshold)):
    #                 glyph_marker[ii_t][i_p] = 2
    #             elif( np.absolute(max_vectors[i_t][i_p][0]) > magnitude_threshold):
    #                 glyph_marker[ii_t][i_p] = 1
   
    return glyph_marker, num_glyphs0, num_glyphs1


@njit
def buildDiscArrowGlyphNP(positions, min_vectors, median_vectors, max_vectors, domain, glyph_markers, time_range, scale, resolution, num_of_glyphs, exclude_time_step):
    """
    purpose: build glyphs for the given data

    """

    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(4*resolution +1)*3, ))
    scalars = np.zeros((num_of_glyphs*(4*resolution +1), ))
    polygons = np.zeros((num_of_glyphs*((2*(resolution + 1) + 5*resolution + 4*(resolution-1)) )), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t 
            ii_t += 1       
    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            # print('i_t:', i_t, ' i_p:', i_p, ' glyph_markers[i_t][i_p]:', glyph_markers[i_t][i_p])
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
               domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
               domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]

                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                # print('min_vector:', min_vector, ' median_vector:', median_vector, ' max_vector:', max_vector)
                # print('position:', position)
                # glyphs_id += 1
                theta = median_vector[1]
                phi = median_vector[2]
                # Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                # Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1

                ## map min, median and max vectors cartesian coordinates
                vec_min = np.zeros((3,))
                vec_min[0] = min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2])
                vec_min[1] = min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2])
                vec_min[2] = min_vector[0]*np.cos(min_vector[1])
                vec_max = np.zeros((3,))
                vec_max[0] = max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2])
                vec_max[1] = max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2])
                vec_max[2] = max_vector[0]*np.cos(max_vector[1])
                vec_median = np.zeros((3,))
                vec_median[0] = median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2])
                vec_median[1] = median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2])
                vec_median[2] = median_vector[0]*np.cos(median_vector[1])
                    
             
                angle_min_max = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_max)), np.dot(vec_min, vec_max))
                # angle_min_median = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_median)), np.dot(vec_min, vec_median))
                # angle_median_max = np.arctan2(np.linalg.norm(np.cross(vec_median, vec_max)), np.dot(vec_median, vec_max))
    
                base_radius = np.absolute(max_vector[0]* np.tan(angle_min_max*0.5))
                base_radius = np.maximum(base_radius, 0.01* max_vector[0])
                phi_vals = np.linspace(0, 2*np.pi, resolution)

                ## mappoints to the position
                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i]) # points[3*(points_id + i) + 0]
                    y = base_radius*np.sin(phi_vals[i]) #points[3*(points_id + i) + 1]
                    z = 0.0 #points[3*(points_id + i) + 2]
                    pt = np.dot(Ry, np.array([x, y, z])) 
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                shaft_radius = 0.025*max_vector[0]
                for i in range(resolution):
                    x = shaft_radius*np.cos(phi_vals[i])
                    y = shaft_radius*np.sin(phi_vals[i])
                    z = 0.0
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t
                old_points_id = points_id
                points_id += resolution
                for i in range(resolution):
                    x = shaft_radius*np.cos(phi_vals[i])
                    y = shaft_radius*np.sin(phi_vals[i])
                    z = min_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id+0] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id += points_id
                points_id += resolution

                head_radius = 2.0*shaft_radius
                for i in range(resolution):
                    x = head_radius*np.cos(phi_vals[i])
                    y = head_radius*np.sin(phi_vals[i])
                    z = min_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                tip_id = points_id
                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id + 0] = pt[0]
                points[3*points_id + 1] = pt[1]
                points[3*points_id + 2] = pt[2]
                scalars[points_id] = i_t
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id+0] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i+1)%resolution
                    polygons_id += 4
                    

    return points, polygons, scalars


@njit
def buildConeGlyphNP(positions, min_vectors, median_vectors, max_vectors, domain, glyph_markers, time_range, scale, resolution, num_of_glyphs0, num_glyphs1, exclude_time_steps):
    """
        purpose: build glyphs for the given data
    input: data - pendas dataframe
           scale - scaling factor for the glyphs
           resolution - number of sides for the cylinder
           time_range - time range for the data
           data_depth - depth of the data 
           domain - domain of the data
           glyph_type - type of the glyph
    
    """

    # print('Entering buildConeGlyphNP(...)')
    # print('num_of_glyphs:', num_of_glyphs, 'min_vectors.shape:', min_vectors.shape, 'median_vectors.shape:', median_vectors.shape, 'max_vectors.shape:', max_vectors.shape)
    # print('positions.shape:', positions.shape)
    
    num_points = positions.shape[0]
    num0_pt  = num_of_glyphs0 * (resolution+1)
    num0_glyphs = num_of_glyphs0*(resolution +1 + 4*(resolution-1))
    num1_pt  = num_glyphs1 * (2*resolution + 1 + resolution + resolution*(resolution-2))
    num1_glyphs = num_glyphs1*(resolution + 1 + 5*resolution + 4*(resolution-1) + 5*resolution*(resolution-2) )

    points = np.zeros((num0_pt*3 + num1_pt*3, ))
    scalars = np.zeros((num0_pt + num1_pt, ))
    polygons = np.zeros((num0_glyphs + num1_glyphs), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_steps and exclude_time_steps <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        
        if(i_t != exclude_time_steps):
            indices[ii_t] = i_t  
            ii_t += 1      
    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            # print('i_t:', i_t, ' i_p:', i_p, ' glyph_markers[i_t][i_p]:', glyph_markers[i_t][i_p])
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_steps and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
               domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
               domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]

                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                # print('min_vector:', min_vector, ' median_vector:', median_vector, ' max_vector:', max_vector)
                # print('position:', position)
                # glyphs_id += 1
                theta = median_vector[1]
                phi = median_vector[2]
                # Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                # Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1

                ## map min, median and max vectors cartesian coordinates
                vec_min = np.zeros((3,))
                vec_min[0] = min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2])
                vec_min[1] = min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2])
                vec_min[2] = min_vector[0]*np.cos(min_vector[1])
                vec_max = np.zeros((3,))
                vec_max[0] = max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2])
                vec_max[1] = max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2])
                vec_max[2] = max_vector[0]*np.cos(max_vector[1])
                vec_median = np.zeros((3,))
                vec_median[0] = median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2])
                vec_median[1] = median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2])
                vec_median[2] = median_vector[0]*np.cos(median_vector[1])
                    
                # vec_min = np.array([min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2]),
                #                     min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2]),
                #                     min_vector[0]*np.cos(min_vector[1])])
                # vec_max = np.array([max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2]),
                #                     max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2]),
                #                     max_vector[0]*np.cos(max_vector[1])])
                # vec_median = np.array([median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2]),
                #                     median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2]),
                #                     median_vector[0]*np.cos(median_vector[1])]) 
                angle_min_max = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_max)), np.dot(vec_min, vec_max))
                # angle_min_median = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_median)), np.dot(vec_min, vec_median))
                # angle_median_max = np.arctan2(np.linalg.norm(np.cross(vec_median, vec_max)), np.dot(vec_median, vec_max))
    
                base_radius = np.absolute(max_vector[0]* np.tan(angle_min_max*0.5))
                if(base_radius > 100):
                    print('base_radius:', base_radius, ' angle_min_max:', angle_min_max, 
                          'i_t:', i_t, ' i_p:', i_p, 'ii_t:', ii_t)
                    print('max_vector:', max_vector, ' min_vector:', min_vector)
                    print('marker:', glyph_markers[ii_t][i_p])
                    raise ValueError('base_radius is too large')

                base_radius = np.maximum(base_radius, 0.01* max_vector[0])
                phi_vals = np.linspace(0, 2*np.pi, resolution)
                # print('resolution:', resolution, ' base_radius:', base_radius)
                # print('3*points_id:', 3*points_id, 'points.shape:', points.shape)
                # print('phi_vals:', phi_vals)

                # points[3*points_id:3*points_id + resolution] = base_radius*np.cos(phi_vals)
                # points[3*points_id+resolution:3*points_id + 2*resolution] = base_radius*np.sin(phi_vals)
                # points[3*points_id+2*resolution:3*points_id + 3*resolution] = np.zeros(resolution)

                ## mappoints to the position
                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i]) # points[3*(points_id + i) + 0]
                    y = base_radius*np.sin(phi_vals[i]) #points[3*(points_id + i) + 1]
                    z = 0.0 #points[3*(points_id + i) + 2]
                    pt = np.dot(Ry, np.array([x, y, z])) 
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                # points[3*points_id] = 0
                # points[3*points_id+1] = 0
                # points[3*points_id+2] = max_vector[0]
                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt* scale + position
                points[3*points_id] = pt[0]
                points[3*points_id+1] = pt[1]
                points[3*points_id+2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i + 1)%resolution
                    polygons_id += 4

            ## large angle 
            if(glyph_markers[ii_t][i_p] == 2 and i_t != exclude_time_steps and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
               domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
               domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]

                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                theta = median_vector[1]
                phi = median_vector[2]

                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1

                ## map min, median and max vectors cartesian coordinates
                vec_min = np.zeros((3,))
                vec_min[0] = min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2])
                vec_min[1] = min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2])
                vec_min[2] = min_vector[0]*np.cos(min_vector[1])
                vec_max = np.zeros((3,))
                vec_max[0] = max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2])
                vec_max[1] = max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2])
                vec_max[2] = max_vector[0]*np.cos(max_vector[1])
                vec_median = np.zeros((3,))
                vec_median[0] = median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2])
                vec_median[1] = median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2])
                vec_median[2] = median_vector[0]*np.cos(median_vector[1])
                    
                
                angle_min_max = np.absolute(max_vector[1])
                base_radius = 0.1*max_vector[0]
                phi_vals = np.linspace(0, 2*np.pi, resolution)

                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i])
                    y = base_radius*np.sin(phi_vals[i])
                    z = 0.0
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                shaft_height = min_vector[0] + 0.5*(max_vector[0] - min_vector[0])
                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i])
                    y = base_radius*np.sin(phi_vals[i])
                    z = shaft_height
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id+0] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id = points_id
                points_id += resolution
                
                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id + 0] = pt[0]
                points[3*points_id + 1] = pt[1]
                points[3*points_id + 2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                head_radius = 0.5* (max_vector[0] - min_vector[0])
                theta_vals = np.linspace(0, angle_min_max*0.5, resolution)
                for i in range(resolution):
                    x = head_radius*np.sin(theta_vals[1])*np.cos(phi_vals[i])
                    y = head_radius*np.sin(theta_vals[1])*np.sin(phi_vals[i])
                    z = head_radius*np.cos(theta_vals[1]) + min_vector[0] + head_radius
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution-1):
                    polygons[polygons_id] = 3
                    polygons[polygons_id+1] = points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons_id += 4
                old_points_id = points_id
                points_id += resolution

                for i in range(2, resolution):
                    for j in range(resolution):
                        x = head_radius*np.sin(theta_vals[i])*np.cos(phi_vals[j])
                        y = head_radius*np.sin(theta_vals[i])*np.sin(phi_vals[j])
                        z = head_radius*np.cos(theta_vals[i])  + min_vector[0] + head_radius
                        pt = np.dot(Ry, np.array([x, y, z]))
                        pt = np.dot(Rz, pt)
                        pt = pt*scale + position
                        points[3*(points_id + j) + 0] = pt[0]
                        points[3*(points_id + j) + 1] = pt[1]
                        points[3*(points_id + j) + 2] = pt[2]
                    scalars[points_id:points_id+resolution] = i_t

                    for j in range(resolution):
                        polygons[polygons_id] = 4
                        polygons[polygons_id+1] = old_points_id + j
                        polygons[polygons_id+2] = points_id + j
                        polygons[polygons_id+3] = points_id + (j+1)%resolution
                        polygons[polygons_id+4] = old_points_id + (j+1)%resolution
                        polygons_id += 5
                    old_points_id = points_id
                    points_id += resolution

               

    return points, polygons, scalars


@njit
def buildCylinderArrowGlyphNP(positions, min_vectors, median_vectors, max_vectors, domain, glyph_markers, time_range, scale, resolution, num_of_glyphs, exclude_time_step):

    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(2*resolution+1)*3, ))
    scalars = np.zeros((num_of_glyphs*(2*resolution+1), ))
    polygons = np.zeros((num_of_glyphs*(resolution +1 +  5*resolution + 4*(resolution-1))), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t  
            ii_t += 1  

    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
               domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
               domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1] ):
                position = positions[i_p]

                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                theta = median_vector[1]
                phi = median_vector[2]

                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1

                ## map min, median and max vectors cartesian coordinates
                vec_min = np.zeros((3,))
                vec_min[0] = min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2])
                vec_min[1] = min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2])
                vec_min[2] = min_vector[0]*np.cos(min_vector[1])
                vec_max = np.zeros((3,))
                vec_max[0] = max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2])
                vec_max[1] = max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2])
                vec_max[2] = max_vector[0]*np.cos(max_vector[1])
                vec_median = np.zeros((3,))
                vec_median[0] = median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2])
                vec_median[1] = median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2])
                vec_median[2] = median_vector[0]*np.cos(median_vector[1])
                    
                angle_min_max = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_max)), np.dot(vec_min, vec_max))
                # angle_min_median = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_median)), np.dot(vec_min, vec_median))
                # angle_median_max = np.arctan2(np.linalg.norm(np.cross(vec_median, vec_max)), np.dot(vec_median, vec_max))
    
                base_radius = np.absolute(max_vector[0]* np.tan(angle_min_max*0.5))
                base_radius = np.maximum(base_radius, 0.01* max_vector[0])
                phi_vals = np.linspace(0, 2*np.pi, resolution)

                ## mappoints to the position
                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i]) # points[3*(points_id + i) + 0]
                    y = base_radius*np.sin(phi_vals[i]) #points[3*(points_id + i) + 1]
                    z = 0.0 #points[3*(points_id + i) + 2]
                    pt = np.dot(Ry, np.array([x, y, z])) 
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i])
                    y = base_radius*np.sin(phi_vals[i])
                    z = min_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id = points_id
                points_id += resolution

                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt* scale + position
                points[3*points_id+0] = pt[0]
                points[3*points_id+1] = pt[1]
                points[3*points_id+2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i + 1)%resolution
                    polygons_id += 4

    return points, polygons, scalars


@njit
def buildSuperElipticalConeNP(directional_variations, positions, vectors, min_vectors, 
                              median_vectors, max_vectors, domain, glyph_markers, 
                              time_range, scale, resolution, num_of_glyphs, exclude_time_step):
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(resolution+1)*3, ))
    scalars = np.zeros((num_of_glyphs*(resolution+1), ))
    polygons = np.zeros((num_of_glyphs*(resolution +1 + 4*(resolution-1))), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t   
            ii_t += 1     
    ii_t = -1 

    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            # print('i_t:', i_t, ' i_p:', i_p, ' glyph_markers[i_t][i_p]:', glyph_markers[i_t][i_p])
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
                domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
                domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]
                v0_scale = directional_variations[ii_t][i_p][0][0]
                v1_scale = directional_variations[ii_t][i_p][0][1]
                v0 = vectors[i_t][i_p][1]
                angle = np.arctan2( v0[1], v0[0])
                mean_vals = directional_variations[ii_t][i_p][3]
                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                phi_vals = np.linspace(0, 2*np.pi, resolution)
                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* v0_scale 
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* v1_scale

                x = x0*np.cos(angle) - y0*np.sin(angle) + mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) + mean_vals[1]
                phi = median_vector[2]
                theta = median_vector[1]
                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1
                # print('resolution:', resolution, ' points_id:', points_id)
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], 0]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution
               
                # cone tip
                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id] = pt[0]
                points[3*points_id+1] = pt[1]
                points[3*points_id+2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id+0] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i + 1)%resolution
                    polygons_id += 4

    # print('num_points:', num_points, ' points_id:', points_id, ' polygons_id:', polygons_id)
    return points, polygons, scalars
    

@njit
def buildDoubleConeGlyphNP(positions, min_vectors, median_vectors, max_vectors, domain, glyph_markers, time_range, scale, resolution, num_of_glyphs, exclude_time_step):
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(3*resolution +1)*3, ))
    scalars = np.zeros((num_of_glyphs*(3*resolution +1), ))
    polygons = np.zeros((num_of_glyphs*((2*(resolution + 1) + 5*resolution + 4*(resolution-1)) )), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t     
            ii_t += 1

    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            # print('i_t:', i_t, ' i_p:', i_p, ' glyph_markers[i_t][i_p]:', glyph_markers[i_t][i_p])
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
                domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
                domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]

                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                # print('min_vector:', min_vector, ' median_vector:', median_vector, ' max_vector:', max_vector)
                # print('position:', position)
                # glyphs_id += 1
                theta = median_vector[1]
                phi = median_vector[2]
                # Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                # Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1

                ## map min, median and max vectors cartesian coordinates
                vec_min = np.zeros((3,))
                vec_min[0] = min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2])
                vec_min[1] = min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2])
                vec_min[2] = min_vector[0]*np.cos(min_vector[1])
                vec_max = np.zeros((3,))
                vec_max[0] = max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2])
                vec_max[1] = max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2])
                vec_max[2] = max_vector[0]*np.cos(max_vector[1])
                vec_median = np.zeros((3,))
                vec_median[0] = median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2])
                vec_median[1] = median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2])
                vec_median[2] = median_vector[0]*np.cos(median_vector[1])
                    
             
                angle_min_max = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_max)), np.dot(vec_min, vec_max))
                # angle_min_median = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_median)), np.dot(vec_min, vec_median))
                # angle_median_max = np.arctan2(np.linalg.norm(np.cross(vec_median, vec_max)), np.dot(vec_median, vec_max))
    
                base_radius = np.absolute(max_vector[0]* np.tan(angle_min_max*0.5))
                base_radius = np.maximum(base_radius, 0.01* max_vector[0])
                phi_vals = np.linspace(0, 2*np.pi, resolution)

                ## mappoints to the position
                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i]) # points[3*(points_id + i) + 0]
                    y = base_radius*np.sin(phi_vals[i]) #points[3*(points_id + i) + 1]
                    z = 0.0 #points[3*(points_id + i) + 2]
                    pt = np.dot(Ry, np.array([x, y, z])) 
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                top_radius = (max_vector[0] - min_vector[0])*np.tan(angle_min_max*0.5)
                for i in range(resolution):
                    x = top_radius*np.cos(phi_vals[i])
                    y = top_radius*np.sin(phi_vals[i])
                    z = min_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id += points_id
                points_id += resolution

                head_radius = base_radius
                for i in range(resolution):
                    x = head_radius*np.cos(phi_vals[i])
                    y = head_radius*np.sin(phi_vals[i])
                    z = min_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id + 0] = pt[0]
                points[3*points_id + 1] = pt[1]
                points[3*points_id + 2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id+0] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i+1)%resolution
                    polygons_id += 4


    return points, polygons, scalars


@njit
def buildSuperElipticalDoubleConeNP(directional_variations, positions, vectors, min_vectors, 
                                    median_vectors, max_vectors, domain, glyph_markers, 
                                    time_range, scale, resolution, num_of_glyphs, exclude_time_step):
    
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(3*resolution + 1)*3, ))
    scalars = np.zeros((num_of_glyphs*(3*resolution+1), ))
    polygons = np.zeros((num_of_glyphs*(2*(resolution +1) + 5*resolution + 4*(resolution-1))), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t   
            ii_t += 1     

    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
                domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
                domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]
                v0_scale = directional_variations[ii_t][i_p][0][0]
                v1_scale = directional_variations[ii_t][i_p][0][1]
                
                # # elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
                # if(np.absolute(v0_scale) < 1.e-20):
                #     print('i_t:', i_t, ' i_p:', i_p, 'ii_t:', ii_t)
                #     print('v0_scale:', v0_scale, ' v1_scale:', v1_scale)
                #     print('max_vector:', max_vectors[i_t][i_p], ' min_vector:', min_vectors[i_t][i_p])
                #     raise ValueError('v0_scale is zero')
                
                elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
                
                v0 = vectors[i_t][i_p][1]
                angle = np.arctan2( v0[1], v0[0])
                mean_vals = directional_variations[ii_t][i_p][3]
                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
               
                r0 = max_vector[0]*np.tan(max_vector[1]*0.5)
                r0 = np.maximum(r0, 0.01*max_vector[0])
                r1 = r0 * elipse_scale

                phi_vals = np.linspace(0, 2*np.pi, resolution)
                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

                x = x0*np.cos(angle) - y0*np.sin(angle) + mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) + mean_vals[1]
                phi = median_vector[2]
                theta = median_vector[1]
                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1
                # print('resolution:', resolution, ' points_id:', points_id)
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], 0]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution
               
               

                # # angle0 = np.arctan2(v0_scale, max_vector[0])
                # top_base_v0_scale = v0_scale/max_vector[0]*(max_vector[0]-min_vector[0])
                # top_base_v1_scale = v1_scale/max_vector[0]*(max_vector[0]-min_vector[0])
                top_base_r0 = (max_vector[0]-min_vector[0])*np.tan(np.absolute(max_vector[1])*0.5)
                top_base_r0 = np.maximum(top_base_r0, 0.01*(max_vector[0]-min_vector[0]))   
                top_base_r1 = top_base_r0 * elipse_scale

                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* top_base_r0
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* top_base_r1
                x = x0*np.cos(angle) - y0*np.sin(angle) + mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) + mean_vals[1]
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], min_vector[0]]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id += points_id
                points_id += resolution

                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0 
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

                x = x0*np.cos(angle) - y0*np.sin(angle) + mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) + mean_vals[1]

                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], min_vector[0]]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                # cone tip
                pt = np.dot(Ry, np.array([mean_vals[0], mean_vals[1], max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id] = pt[0]
                points[3*points_id+1] = pt[1]
                points[3*points_id+2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id+0] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i + 1)%resolution
                    polygons_id += 4




    return points, polygons, scalars


@njit
def buildSuperElipticalSquidNP(directional_variations, positions, vectors, min_vectors, 
                                    median_vectors, max_vectors, domain, glyph_markers, 
                                    time_range, scale, resolution, num_of_glyphs, exclude_time_step):
    
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(5*resolution + 1)*3, ))
    scalars = np.zeros((num_of_glyphs*(5*resolution+1), ))
    polygons = np.zeros((num_of_glyphs*(3*(resolution +1) + 10*resolution + 4*(resolution-1))), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
        # indices = indices + time_range[0]
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)
        # indices = indices + time_range[0]   

    ii_t = 0
    for i_t in range(time_range[0], time_range[1]+1):
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t 
            ii_t += 1       

    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] - 1.0e-10 <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] + 1.0e-10 and
                domain[1][0]- 1.0e-10 <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] + 1.0e-10 and
                domain[2][0]- 1.0e-10 <= positions[i_p][2] and positions[i_p][2] <= domain[2][1] + 1.0e-10 and 
                 max_vectors[i_t][i_p][0] > 0.001):
                position = positions[i_p]
                v0_scale = directional_variations[ii_t][i_p][0][0]
                v1_scale = directional_variations[ii_t][i_p][0][1]
                # # elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
                if(np.absolute(v0_scale) < 1.e-20):
                    # print('i_t:', i_t, ' i_p:', i_p, 'ii_t:', ii_t)
                    # print('indices:', indices)
                    # print('v0_scale:', v0_scale, ' v1_scale:', v1_scale)
                    # print('max_vector:', max_vectors[i_t][i_p], ' min_vector:', min_vectors[i_t][i_p])
                    # print ensemble memebers
                    # for i_en in range(0, 20):
                    #     print('vector[', i_en, ']:', vectors[i_t][i_p][i_en])
                    elipse_scale = 1.0
                    angle = 0.001 # min angle paramter
                    # mean directions vals 
                else:

                    elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
                    v0 = vectors[i_t][i_p][1]
                    v0 = directional_variations[ii_t][i_p][1]
                    angle = np.arctan2( v0[1], v0[0])
                    # mean_vals = directional_variations[ii_t][i_p][3]
                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]

                # if( v0_scale > 100):
                #     print('v0_scale:', v0_scale, ' v1_scale:', v1_scale)
                #     print('min_vector:', min_vector, ' median_vector:', median_vector, ' max_vector:', max_vector)
                #     raise ValueError('v0_scale > 100')
                
                r0 = max_vector[0]*np.tan(max_vector[1]*0.5)
                r0 = np.maximum(r0, 0.01*max_vector[0])
                r1 = r0 * elipse_scale
                phi_vals = np.linspace(0, 2*np.pi, resolution)
                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

                x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
                phi = median_vector[2]
                theta = median_vector[1]
                
                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1
                # print('resolution:', resolution, ' points_id:', points_id)
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], 0]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution
               
                base_height = max_vector[0] - min_vector[0]
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], base_height]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                
                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id += points_id
                points_id += resolution



                # # # angle0 = np.arctan2(v0_scale, max_vector[0])
                # shaft_base_v0_scale = v0_scale/max_vector[0]*min_vector[0]
                # shaft_base_v1_scale = v1_scale/max_vector[0]*min_vector[0]
                shaft_base_r0 = min_vector[0]*np.tan(np.absolute(max_vector[1])*0.5)
                shaft_base_r0 = np.maximum(shaft_base_r0, 0.01*min_vector[0])
                shaft_base_r1 = shaft_base_r0 * elipse_scale

                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* shaft_base_r0
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* shaft_base_r1
                x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], max_vector[0] - min_vector[0]]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t
                old_points_id = points_id
                points_id += resolution
                # shaft_top_v0_scale = v0_scale/max_vector[0]*(0.2*max_vector[0])
                # shaft_top_v1_scale = v1_scale/max_vector[0]*(0.2*max_vector[0])
                shaft_top_r0 = 0.2*max_vector[0]*np.tan(np.absolute(max_vector[1])*0.5)
                shaft_top_r0 = np.maximum(shaft_top_r0, 0.01*0.2*max_vector[0])
                shaft_top_r1 = shaft_top_r0 * elipse_scale

                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* shaft_top_r0
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* shaft_top_r1
                x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], 0.8*max_vector[0]]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id += points_id
                points_id += resolution

                x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0 
                y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

                x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
                y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
                for i in range(resolution):
                    pt = np.dot(Ry, np.array([x[i], y[i], 0.8*max_vector[0]]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                # cone tip
                # pt = np.dot(Ry, np.array([mean_vals[0], mean_vals[1], max_vector[0]]))
                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id] = pt[0]
                points[3*points_id+1] = pt[1]
                points[3*points_id+2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id+0] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i + 1)%resolution
                    polygons_id += 4

    return points, polygons, scalars



@njit
def buildSquidGlyphNP(positions, min_vectors, median_vectors, max_vectors, domain, glyph_markers, time_range, scale, resolution, num_of_glyphs, exclude_time_step):
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*(5*resolution +1)*3, ))
    scalars = np.zeros((num_of_glyphs*(5*resolution +1), ))
    polygons = np.zeros((num_of_glyphs*((3*(resolution + 1) + 10*resolution + 4*(resolution-1)) )), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    ii_t = -1
    if( time_range[0] <= exclude_time_step and exclude_time_step <= time_range[1]):
        indices = np.zeros((time_range[1] - time_range[0]), dtype=np.int32)
    else:
        indices = np.zeros((time_range[1] - time_range[0] + 1), dtype=np.int32)

    for i_t in range(time_range[0], time_range[1]+1):
        
        if(i_t != exclude_time_step):
            indices[ii_t] = i_t        
    ii_t = -1        
    for i_t in indices:
        ii_t += 1
        for i_p in range(num_points):
            # print('i_t:', i_t, ' i_p:', i_p, ' glyph_markers[i_t][i_p]:', glyph_markers[i_t][i_p])
            if(glyph_markers[ii_t][i_p] == 1 and i_t != exclude_time_step and 
               domain[0][0] <= positions[i_p][0] and positions[i_p][0] <= domain[0][1] and
                domain[1][0] <= positions[i_p][1] and positions[i_p][1] <= domain[1][1] and
                domain[2][0] <= positions[i_p][2] and positions[i_p][2] <= domain[2][1]):
                position = positions[i_p]

                min_vector = min_vectors[i_t][i_p]
                median_vector = median_vectors[i_t][i_p]
                max_vector = max_vectors[i_t][i_p]
                theta = median_vector[1]
                phi = median_vector[2]

                Ry = np.zeros((3, 3))
                Ry[0][0] = np.cos(theta)
                Ry[0][2] = np.sin(theta)
                Ry[1][1] = 1
                Ry[2][0] = -np.sin(theta)
                Ry[2][2] = np.cos(theta)
                Rz = np.zeros((3, 3))
                Rz[0][0] = np.cos(phi)
                Rz[0][1] = -np.sin(phi)
                Rz[1][0] = np.sin(phi)
                Rz[1][1] = np.cos(phi)
                Rz[2][2] = 1

                ## map min, median and max vectors cartesian coordinates
                vec_min = np.zeros((3,))
                vec_min[0] = min_vector[0]*np.sin(min_vector[1])*np.cos(min_vector[2])
                vec_min[1] = min_vector[0]*np.sin(min_vector[1])*np.sin(min_vector[2])
                vec_min[2] = min_vector[0]*np.cos(min_vector[1])
                vec_max = np.zeros((3,))
                vec_max[0] = max_vector[0]*np.sin(max_vector[1])*np.cos(max_vector[2])
                vec_max[1] = max_vector[0]*np.sin(max_vector[1])*np.sin(max_vector[2])
                vec_max[2] = max_vector[0]*np.cos(max_vector[1])
                vec_median = np.zeros((3,))
                vec_median[0] = median_vector[0]*np.sin(median_vector[1])*np.cos(median_vector[2])
                vec_median[1] = median_vector[0]*np.sin(median_vector[1])*np.sin(median_vector[2])
                vec_median[2] = median_vector[0]*np.cos(median_vector[1])
                    
             
                angle_min_max = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_max)), np.dot(vec_min, vec_max))
                # angle_min_median = np.arctan2(np.linalg.norm(np.cross(vec_min, vec_median)), np.dot(vec_min, vec_median))
                # angle_median_max = np.arctan2(np.linalg.norm(np.cross(vec_median, vec_max)), np.dot(vec_median, vec_max))
    
                base_radius = np.absolute(max_vector[0]* np.tan(angle_min_max*0.5))
                base_radius = np.maximum(base_radius, 0.01* max_vector[0])
                phi_vals = np.linspace(0, 2*np.pi, resolution)

                ## mappoints to the position
                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i]) # points[3*(points_id + i) + 0]
                    y = base_radius*np.sin(phi_vals[i]) #points[3*(points_id + i) + 1]
                    z = 0.0 #points[3*(points_id + i) + 2]
                    pt = np.dot(Ry, np.array([x, y, z])) 
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                polygons[polygons_id] = resolution
                polygons_id += 1
                for i in range(resolution):
                    polygons[polygons_id] = points_id + i
                    polygons_id += 1
                old_points_id = points_id
                points_id += resolution

                for i in range(resolution):
                    x = base_radius*np.cos(phi_vals[i]) # points[3*(points_id + i) + 0]
                    y = base_radius*np.sin(phi_vals[i]) #points[3*(points_id + i) + 1]
                    z = (max_vector[0]-min_vector[0]) #points[3*(points_id + i) + 2]
                    pt = np.dot(Ry, np.array([x, y, z])) 
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t

                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id += points_id
                points_id += resolution

                shaft_radius =  min_vector[0]* np.tan(angle_min_max*0.5)
                for i in range(resolution):
                    x = shaft_radius*np.cos(phi_vals[i])
                    y = shaft_radius*np.sin(phi_vals[i])
                    z = max_vector[0] - min_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t
                old_points_id = points_id
                points_id += resolution
                shaft_radius2 = 0.2*max_vector[0]*np.tan(angle_min_max*0.5)
                for i in range(resolution):
                    x = shaft_radius2*np.cos(phi_vals[i])
                    y = shaft_radius2*np.sin(phi_vals[i])
                    z = 0.8*max_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t
                
                for i in range(resolution):
                    polygons[polygons_id] = 4
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = points_id + i
                    polygons[polygons_id+3] = points_id + (i+1)%resolution
                    polygons[polygons_id+4] = old_points_id + (i+1)%resolution
                    polygons_id += 5
                old_points_id += points_id
                points_id += resolution
                head_radius = 0.8*max_vector[0]*np.tan(angle_min_max*0.5)
                for i in range(resolution):
                    x = head_radius*np.cos(phi_vals[i])
                    y = head_radius*np.sin(phi_vals[i])
                    z = 0.8*max_vector[0]
                    pt = np.dot(Ry, np.array([x, y, z]))
                    pt = np.dot(Rz, pt)
                    pt = pt*scale + position
                    points[3*(points_id + i) + 0] = pt[0]
                    points[3*(points_id + i) + 1] = pt[1]
                    points[3*(points_id + i) + 2] = pt[2]
                scalars[points_id:points_id+resolution] = i_t
                old_points_id = points_id
                points_id += resolution

                pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[3*points_id + 0] = pt[0]
                points[3*points_id + 1] = pt[1]
                points[3*points_id + 2] = pt[2]
                scalars[points_id] = i_t
                tip_id = points_id
                points_id += 1

                for i in range(resolution-1):
                    polygons[polygons_id+0] = 3
                    polygons[polygons_id+1] = old_points_id + i
                    polygons[polygons_id+2] = tip_id
                    polygons[polygons_id+3] = old_points_id + (i+1)%resolution
                    polygons_id += 4

    return points, polygons, scalars    



def buildGlyphsForDash(data, scale=1.0, resolution=10, time_range =[0,0], selected_time= None, data_depth = 0.0,
                        domain=None,  glyph_type='cone', mean_flag = False, compute_min_max_flag = False, 
                        mag_threshold = [-1.0, 1.0e+16]):
    """
    purpose: build glyphs for the given data
    input: data - pendas dataframe
           scale - scaling factor for the glyphs
           resolution - number of sides for the cylinder
           time_range - time range for the data
           data_depth - depth of the data
    """    
    final_glyphs = []
    if(domain is None):
        domain = data['domain'][0]

    if(selected_time is None):
        selected_time = time_range[0]

    adjusted_scaling_factor = scale * data['cell_diag'][0]*1.0 / data['max_magnitude'][0]   
    
    # start = time.time()
    if(compute_min_max_flag):
        print('Computing min, median and max vectors')
        vf_statistics.getMinMedianMaxVectors(data, data_depth, time_range, domain, mean_flag, mag_threshold[0], mag_threshold[1])
    # end = time.time()
    # print("Time to compute min, median and max vectors:", end-start)

    # start = time.time()
    glyph_markers, num_glyphs0, num_glyphs1 = getGlyphsMarkers(data['min_vectors'][0], data['max_vectors'][0], 
                                                                [selected_time, selected_time], data['num_points'][0], 
                                                                time_range[1]+1)
    selected_time_range = [selected_time, selected_time]
    direction_variations = vf_statistics.getDirectionalVariations(  data['positions'][0],
                                                                    data['vectors'][0], 
                                                                    data['depths'][0],
                                                                    data_depth,
                                                                    data['min_vectors'][0],
                                                                    data['median_vectors'][0],
                                                                    data['max_vectors'][0],
                                                                    domain, 
                                                                    selected_time_range)
    if(glyph_type == 'cone'):
        
        print('direction_variations:', direction_variations.shape)
        # points, polygons, scalars = buildSuperElipticalConeNP(direction_variations, data['positions'][0],
        #                                                         data['vectors'][0], data['min_vectors'][0],
        #                                                         data['median_vectors'][0], data['max_vectors'][0],
        #                                                         domain, glyph_markers, selected_time_range, 
        #                                                         adjusted_scaling_factor, resolution, num_glyphs, time_range[1]+1)

        points, polygons, scalars = buildConeGlyphNP(data['positions'][0], data['min_vectors'][0], 
                                                     data['median_vectors'][0], data['max_vectors'][0], domain,
                                                     glyph_markers, selected_time_range, adjusted_scaling_factor, 
                                                     resolution, num_glyphs0, num_glyphs1, time_range[1]+1)
    elif(glyph_type == 'disc-arrow'):
        points, polygons, scalars = buildDiscArrowGlyphNP(data['positions'][0], data['min_vectors'][0], 
                                                     data['median_vectors'][0], data['max_vectors'][0], domain,
                                                     glyph_markers, selected_time_range, adjusted_scaling_factor, 
                                                     resolution, num_glyphs0, time_range[1]+1)
    elif(glyph_type == 'cylinder-arrow' or glyph_type == 'comet'):
        points, polygons, scalars = buildCylinderArrowGlyphNP(data['positions'][0], data['min_vectors'][0], 
                                                     data['median_vectors'][0], data['max_vectors'][0], domain,
                                                     glyph_markers, selected_time_range, adjusted_scaling_factor, 
                                                     resolution, num_glyphs0, time_range[1]+1)
    elif(glyph_type == 'new-cone'):
        points, polygons, scalars = buildSuperElipticalDoubleConeNP(direction_variations, data['positions'][0],
                                                                data['vectors'][0], data['min_vectors'][0],
                                                                data['median_vectors'][0], data['max_vectors'][0],
                                                                domain, glyph_markers, selected_time_range, 
                                                                adjusted_scaling_factor, resolution, num_glyphs0, time_range[1]+1)
        # points, polygons, scalars = buildDoubleConeGlyphNP(data['positions'][0], data['min_vectors'][0], 
        #                                             data['median_vectors'][0], data['max_vectors'][0], domain,
        #                                             glyph_markers, selected_time_range, adjusted_scaling_factor, 
        #                                             resolution, num_glyphs0, time_range[1]+1)
    elif(glyph_type == 'squid' or glyph_type == 'squid2'):
        # print('glyph_type:', glyph_type)
        points, polygons, scalars = buildSuperElipticalSquidNP(direction_variations, data['positions'][0],
                                                                data['vectors'][0], data['min_vectors'][0],
                                                                data['median_vectors'][0], data['max_vectors'][0],
                                                                domain, glyph_markers, selected_time_range, 
                                                                adjusted_scaling_factor, resolution, num_glyphs0, time_range[1]+1)
        # points, polygons, scalars = buildSquidGlyphNP(data['positions'][0], data['min_vectors'][0], 
        #                                             data['median_vectors'][0], data['max_vectors'][0], domain,
        #                                             glyph_markers, selected_time_range, adjusted_scaling_factor, 
        #                                             resolution, num_glyphs0, time_range[1]+1)

    vector_glyph = dash_vtk.GeometryRepresentation(
        children=[
            dash_vtk.PolyData(
                id="vtk-polydata",
                points=points,
                polys=polygons,
                children=[
                    dash_vtk.PointData(
                        [
                            dash_vtk.DataArray(
                                id="vtk-array",
                                registration="setScalars",
                                name="time_step",
                                values=scalars,
                            )
                        ]
                    )
                ],
            )],
        colorMapPreset='coolwarm',
        colorDataRange=[time_range[0], np.maximum(time_range[1], time_range[0]+1)],
        property={'opacity':1.0},
        showCubeAxes=True,
    )
   
    final_glyphs.append(vector_glyph)

    if(selected_time != time_range[0] or selected_time != time_range[1]):
        glyph_markers, num_glyphs0, num_glyphs1 = getGlyphsMarkers(data['min_vectors'][0], data['max_vectors'][0], time_range, data['num_points'][0], selected_time)
        direction_variations = vf_statistics.getDirectionalVariations(  data['positions'][0],
                                                                    data['vectors'][0], 
                                                                    data['depths'][0],
                                                                    data_depth,
                                                                    data['min_vectors'][0],
                                                                    data['median_vectors'][0],
                                                                    data['max_vectors'][0],
                                                                    domain, 
                                                                    time_range)
        if(glyph_type == 'cone'):
            points, polygons, scalars = buildConeGlyphNP(data['positions'][0], data['min_vectors'][0], 
                                                     data['median_vectors'][0], data['max_vectors'][0], domain,
                                                     glyph_markers, time_range, adjusted_scaling_factor, 
                                                     resolution, num_glyphs0, num_glyphs1, selected_time)
        elif(glyph_type == 'disc-arrow'):
            points, polygons, scalars = buildDiscArrowGlyphNP(data['positions'][0], data['min_vectors'][0], 
                                                     data['median_vectors'][0], data['max_vectors'][0], domain,
                                                     glyph_markers, time_range, adjusted_scaling_factor, 
                                                     resolution, num_glyphs0, selected_time)
        elif(glyph_type == 'cylinder-arrow' or glyph_type == 'comet'):
            points, polygons, scalars = buildCylinderArrowGlyphNP(data['positions'][0], data['min_vectors'][0], 
                                                     data['median_vectors'][0], data['max_vectors'][0], domain,
                                                     glyph_markers, time_range, adjusted_scaling_factor, 
                                                     resolution, num_glyphs0, selected_time)
        elif(glyph_type == 'new-cone'):
            points, polygons, scalars = buildSuperElipticalDoubleConeNP(direction_variations, data['positions'][0],
                                                                data['vectors'][0], data['min_vectors'][0],
                                                                data['median_vectors'][0], data['max_vectors'][0],
                                                                domain, glyph_markers, time_range, 
                                                                adjusted_scaling_factor, resolution, num_glyphs0, selected_time)
            # points, polygons, scalars = buildDoubleConeGlyphNP(data['positions'][0], data['min_vectors'][0], 
            #                                          data['median_vectors'][0], data['max_vectors'][0], domain,
            #                                          glyph_markers, time_range, adjusted_scaling_factor, 
            #                                          resolution, num_glyphs0, selected_time)
        elif(glyph_type == 'squid' or glyph_type == 'squid2'):
            points, polygons, scalars = buildSuperElipticalSquidNP(direction_variations, data['positions'][0],
                                                                data['vectors'][0], data['min_vectors'][0],
                                                                data['median_vectors'][0], data['max_vectors'][0],
                                                                domain, glyph_markers, time_range, 
                                                                adjusted_scaling_factor, resolution, num_glyphs0, selected_time)
            # points, polygons, scalars = buildSquidGlyphNP(data['positions'][0], data['min_vectors'][0], 
            #                                         data['median_vectors'][0], data['max_vectors'][0], domain,
            #                                         glyph_markers, time_range, adjusted_scaling_factor, 
            #                                         resolution, num_glyphs0, selected_time)


        vector_glyph = dash_vtk.GeometryRepresentation(
            children=[
                dash_vtk.PolyData(
                    id="vtk-polydata",
                    points=points,
                    polys=polygons,
                    children=[
                        dash_vtk.PointData(
                            [
                                dash_vtk.DataArray(
                                    id="vtk-array",
                                    registration="setScalars",
                                    name="time_step",
                                    values=scalars,
                                )
                            ]
                        )
                    ],
                )],
            colorMapPreset='coolwarm',
            colorDataRange=[time_range[0], time_range[1]],
            property={'opacity':0.1},
            # showCubeAxes=True,
        )
        
        final_glyphs.append(vector_glyph)
    
    # end = time.time()
    # print("Time to build glyphs:", end-start)
    return final_glyphs



def buildArrowGlyphsForDash(data, scale=1.0, resolution=10, time_range =[0,0], selected_time= None, data_depth = 0.0, 
                            domain=None, point_ids=[0]):
    # purpose: build glyphs for the given data
    # input: data - pendas dataframe
    #        scale - scaling factor for the glyphs
    #        resolution - number of sides for the cylinder
    #        time_range - time range for the data

    points = []
    polygons = []
    scalars = []
    final_glyphs = []

    adjusted_scaling_factor = scale * data['cell_diag'][0]*1.0 / data['max_magnitude'][0]   

    for i_t in range(time_range[0], time_range[1]+1):
        for i_p in range(data['num_points'][0]): # number of points
            for i_e in range(data['num_ensemble_members'][0]): # number of ensemble members
                if( data['vectors'][0][i_t][i_p][i_e][0] > 1.0e-4 and 
                   vf_utils.insideBoundingBox(data['positions'][0][i_p], domain) ):
                    position = data['positions'][0][i_p]
                    vector = data['vectors'][0][i_t][i_p][i_e]
                    g_id = len(points)
                    old_gid = g_id
                    local_points, local_polygons = buildArrowGlyphNumPyGPU(position, vector, adjusted_scaling_factor,
                                                                            resolution, g_id)

                    for i in range(local_points.shape[0]):
                        points.append([local_points[i][0], local_points[i][1], local_points[i][2]])
                        scalars.append(i_t)
                    for i in range(local_polygons.shape[0]):
                        polygons.append(local_polygons[i])

               
    vector_glyph = dash_vtk.GeometryRepresentation(
        children=[
            dash_vtk.PolyData(
                id="vtk-polydata",
                points=np.array(points).ravel(),
                polys=np.array(polygons),
                children=[
                    dash_vtk.PointData(
                        [
                            dash_vtk.DataArray(
                                id="vtk-array",
                                registration="setScalars",
                                name="time_step",
                                values=np.array(scalars),
                            )
                        ]
                    )
                ],
            )],
        colorMapPreset='coolwarm',
        colorDataRange=[time_range[0], time_range[1]+1],
        property={'opacity':1.0},
        # showCubeAxes=True,
    )

    final_glyphs.append(vector_glyph)
    return final_glyphs
