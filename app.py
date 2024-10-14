# adding Folder_2 to the system path
import sys
import os
current_path = os.getcwd()
sys.path.insert(0, current_path)
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
# VTKImage

import scipy as sp

import vf_statistics as vf_stats
import vf_utils as vf_utils
import vf_glyphs as vf_glyphs

from dash import Dash, dcc, html, Input, Output, callback, ctx

import dash_daq as daq

import dash_bootstrap_components as dbc

#from dash import Dash, dcc, html, Input, Output, callback

# Use helper to get a mesh structure that can be passed as-is to a Mesh
from dash_vtk.utils import to_mesh_state
import dash_vtk

import netCDF4 as nc

import time




fs = 20

from math import sin, cos, pi



def getWindData():

    num_ensemble_members = 30
    num_time_steps = 361 
    num_x = 1024
    num_y = 1024
    num_sampled_ensemble_members = 10
    num_sampled_time_steps = 10
    num_sampled_x = 20
    num_sampled_y = 20
    ## get indices for sampled data
    x_indices = getSampleIndices([0, num_x-1], num_sampled_x)
    y_indices = getSampleIndices([0, num_y-1], num_sampled_y)
    ensemble_indices = getSampleIndices([0, num_ensemble_members-1], num_sampled_ensemble_members)
    time_indices = getSampleIndices([0, num_time_steps-1], num_sampled_time_steps)
    num_points = num_sampled_x * num_sampled_y
    positions = np.zeros((num_points, 3))
    
    vectors = np.zeros((num_sampled_ensemble_members, num_sampled_time_steps, num_points, 3))
    
    for i_ens in range(0, num_sampled_ensemble_members):
        ii_ens = ensemble_indices[i_ens]
        file_name = '/mnt/Storage1/controlled_rand_wind/data/controlled_rand_wind_64_128_16_' + str(ii_ens) + '.nc'
        data = nc.Dataset(file_name, 'r')
        print('data.variables: ', data.variables)
        print('data.variables.keys(): ', data.variables.keys())
        print('UF.shape: ', data.variables['UF'].shape)
        print('VF.shape: ', data.variables['VF'].shape)
        print('ZSF.shape: ', data.variables['ZSF'].shape)

        for i_t in range(0, num_sampled_time_steps):
            ii_t = time_indices[i_t]
            for i_y in range(0, num_sampled_y):
                ii_y = y_indices[i_y]
                for i_x in range(0, num_sampled_x):
                    ii_x = x_indices[i_x]
                    if i_ens == 0 and i_t == 0:
                        positions[i_x*num_sampled_y + i_y] = [ii_x, ii_y, data.variables['ZSF'][ii_t, ii_x, ii_y]]
                    vec = [data.variables['UF'][ii_t, ii_x, ii_y], data.variables['VF'][ii_t, ii_x, ii_y], 0]
                    sp_coords = vf_utils.getSphericalCoordinates(vec)
                    # print('positions:', positions[i_x*num_sampled_y + i_y])
                    # print('sp_coords:', sp_coords)
                    vectors[i_ens, i_t, i_x*num_sampled_y + i_y, 0] = sp_coords[0]
                    vectors[i_ens, i_t, i_x*num_sampled_y + i_y, 1] = sp_coords[1]
                    vectors[i_ens, i_t, i_x*num_sampled_y + i_y, 2] = sp_coords[2]
    ## transpose vectors to match the shape [num_time_steps, num_points, num_ensemble_members, 3]
    vectors = np.transpose(vectors, (1, 2, 0, 3))
    cell_diag = np.sqrt((x_indices[1]-x_indices[0])**2 + (y_indices[1]-y_indices[0])**2)
    max_magnitude = getMaxMagnitude(vectors)
    dd = {'num_time_steps': num_sampled_time_steps,
            'num_points': num_points,
            'dimensions': 3,
            'num_ensemble_members':num_sampled_ensemble_members,
            'positions': [positions],
            'vectors': [vectors],
            'domain': [np.array([[0, num_x], [0, num_y], [0, 2000]])],
            'depths': [np.zeros([num_sampled_time_steps, num_points, num_sampled_ensemble_members])],
            'min_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
            'median_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
            'max_vectors': [-1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
            'variability': [np.zeros([num_sampled_time_steps, num_points, 3])],
            'max_magnitude': [max_magnitude],
            'cell_diag': [cell_diag]}

    return pd.DataFrame(data=dd)


### get wind data Fire exampe ###
def getFireWindData():
    file_name = '/mnt/Storage1/fire_data/wind_data4.npy'
    num_ensemble_members = 100 #1000
    num_time_steps = 109 #37 
    num_x = 69 #17
    num_y = 53 #13
    num_sampled_ensemble_members = 25
    num_sampled_time_steps = 5
    num_sampled_x = 69#34#17
    num_sampled_y = 53#26#13
    ## get indices for sampled data
    x_indices = getSampleIndices([0, num_x-1], num_sampled_x)
    y_indices = getSampleIndices([0, num_y-1], num_sampled_y)
    # ensemble_indices = getSampleIndices([0, num_ensemble_members-1], num_sampled_ensemble_members)
    ensemble_indices = getSampleIndices([0, 99-1], num_sampled_ensemble_members)
    time_indices = [30, 31, 32, 33, 34] #getSampleIndices([40, 49], num_sampled_time_steps)
    num_points = num_sampled_x * num_sampled_y
    positions = np.zeros((num_points, 3))
    
    for i in range(0, num_sampled_x):
        ii = x_indices[i]
        for j in range(0, num_sampled_y):
                jj = y_indices[j]
                positions[i*num_sampled_y + j] = [ii, jj, 0]
    
    data = np.load(file_name)
    print('data.shape:', data.shape)
    # reshape from (17, 13, 37, 2, 1000) to (37, 17, 13, 37, 1000, 2)
    data = np.transpose(data, (2, 0, 1, 4, 3))

   
    ## vectors
    vectors = np.zeros((num_sampled_time_steps, num_points, num_sampled_ensemble_members, 3))
    for i in range(0, num_sampled_time_steps):
        ii = time_indices[i]
        for j in range(0, num_sampled_x):
            jj = x_indices[j]
            for k in range(0, num_sampled_y):
                    kk = y_indices[k]
                    for m in range(0, num_sampled_ensemble_members):
                        mm = ensemble_indices[m]
                        vec = [data[ii, jj, kk, mm, 0], data[ii, jj, kk, mm, 1], data[ii, jj, kk, mm, 2]]
                        sp_coords = vf_utils.getSphericalCoordinates(vec)
                        vectors[i, j*num_sampled_y + k, m][0] = sp_coords[0]
                        vectors[i, j*num_sampled_y + k, m][1] = sp_coords[1]
                        vectors[i, j*num_sampled_y + k, m][2] = sp_coords[2] 
                        # [data[i, j, k, l, m, 0], data[i, j, k, l, m, 1], 0]
    cell_diag = np.sqrt( (x_indices[1]-x_indices[0])**2 + (y_indices[1]-y_indices[0])**2)
    max_magnitude = getMaxMagnitude(vectors)
    dd = {'num_time_steps': num_sampled_time_steps,
            'num_points': num_sampled_x*num_sampled_y,
            'dimensions': 3,
            'num_ensemble_members':num_sampled_ensemble_members,
            'selected_depth': [0.0],
            'positions': [positions],
            'vectors': [vectors],
            'domain': [np.array([[0, num_x], [0, num_y], [-1, 1]])],
            'depths': [np.zeros([num_sampled_time_steps, num_sampled_x*num_sampled_y, num_sampled_ensemble_members])],
            'min_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_sampled_x*num_sampled_y, 3])],
            'median_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_sampled_x*num_sampled_y, 3])],
            'max_vectors': [-1.0e+10*np.ones([num_sampled_time_steps, num_sampled_x*num_sampled_y, 3])],
            'variability': [np.zeros([num_sampled_time_steps, num_sampled_x*num_sampled_y, 3])],
            'max_magnitude': [max_magnitude],
            'cell_diag': [cell_diag]}
    return pd.DataFrame(data=dd)

### End of get wind data Fire exampe ###

### Write hurrican data in format that is easy to read
def writeHurricaneData():
    print('Writing hurricane data ...')
    num_ensemble_members = 1
    num_time_steps = 48
    num_x = 500
    num_y = 500
    num_z = 100
    for i_t in range(10,  num_time_steps):
        if i_t < 10:
            file_name = '/mnt/Storage1/hurrican_isabel/isabel0'+str(i_t+1)+'.vti'
        else:
            file_name = '/mnt/Storage1/hurrican_isabel/isabel'+str(i_t)+'.vti'
        # Read the source file.
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        # Get the output of the reader
        output = dsa.WrapDataObject(reader.GetOutput())
        # get the points
        Uf = output.PointData['Uf']
        Vf = output.PointData['Vf']
        Wf = output.PointData['Wf']
        for i_f in range(Uf.shape[0]):
            if(Uf[i_f] > 1.0e+10):
                Uf[i_f] = 0.0
        for i_f in range(Vf.shape[0]):
            if(Vf[i_f] > 1.0e+10):
                Vf[i_f] = 0.0
        for i_f in range(Wf.shape[0]):
            if(Wf[i_f] > 1.0e+10):
                Wf[i_f] = 0.0
        print( 'Uf.shape:', Uf.shape)
        # print( 'Vf.shape:', Vf.shape)
        # print( 'Wf.shape:', Wf.shape) 
        vectors = np.zeros((num_x*num_y*num_z, 3))
        for i_z in range(0, num_z):
            for i_y in range(0, num_y):
                for i_x in range(0, num_x):
                    vec = [Uf[i_x + i_y*num_x + i_z*num_x*num_y], Vf[i_x + i_y*num_x + i_z*num_x*num_y], Wf[i_x + i_y*num_x + i_z*num_x*num_y]]
                    vectors[i_x + i_y*num_x + i_z*num_x*num_y] = vec
        if(i_t < 10):
            np.save('/mnt/Storage1/hurrican_isabel/isabel0'+str(i_t+1)+'.npy', vectors)
        else:
            np.save('/mnt/Storage1/hurrican_isabel/isabel'+str(i_t)+'.npy', vectors)
    print('Done writing hurricane data ...')

write_hurricane_data = False
if(write_hurricane_data):
    writeHurricaneData()
    exit()


def readHurricaneData2():
    num_ensemble_members = 1
    num_time_steps = 48
    num_x = 500
    num_y = 500
    num_z = 100
    num_sampled_ensemble_members = 5 * 5 * 1
    num_sampled_time_steps = 5
    num_sampled_x = 40
    num_sampled_y = 40
    num_sampled_z = 1
    delta_x_idx = num_x / num_sampled_x
    delta_y_idx = num_y / num_sampled_y
    delta_z_idx = num_z / num_sampled_z
    ## get indices for sampled data
    # x_indices = getSampleIndices([0, num_x-1], num_sampled_x)
    # y_indices = getSampleIndices([0, num_y-1], num_sampled_y)
    # z_indices = getSampleIndices([0, num_z-1], num_sampled_z)
    # ensemble_indices = np.linspace(0, num_ensemble_members-1, num_sampled_ensemble_members)
    # time_indices = getSampleIndices([0, num_time_steps-1], num_sampled_time_steps)
    # # time_indices = np.array([0, 2, 4, 6, 8]) 
    # # getSampleIndices([0, num_time_steps-1], num_sampled_time_steps)
    num_points = num_sampled_x * num_sampled_y * num_sampled_z
    positions = np.zeros((num_points, 3))
    vectors = np.zeros((num_sampled_time_steps, num_points, num_sampled_ensemble_members, 3))
    terrain_points = np.zeros((num_sampled_x*num_sampled_y, 3))
    terrain = file_name = '/mnt/Storage1/hurrican_isabel/isabel_terrain.vti'
    # read the terrain data
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(terrain)
    reader.Update()
    # Get the output of the reader
    # print('reader.GetOutput():', reader.GetOutput())
    terrain_output = dsa.WrapDataObject(reader.GetOutput())
    terrain_height = terrain_output.PointData['HGTdata']

    for i_t in range(0,  num_sampled_time_steps):
        delta_t_idx  = np.floor (num_time_steps/num_sampled_time_steps)
        ii_t = i_t +  0 #int(delta_t_idx) + 1
        print ('ii_t:', ii_t)
        if ii_t < 9:
            local_vectors = np.load('/mnt/Storage1/hurrican_isabel/isabel0'+str(ii_t+1)+'.npy')
        else:
            local_vectors = np.load('/mnt/Storage1/hurrican_isabel/isabel'+str(ii_t +1)+'.npy')
        for i_z in range(0, num_sampled_z):
            ii_zs = int(i_z * delta_z_idx)   #int(np.maximum(i_z*delta_z_idx -1, 0))
            ii_ze =  int(i_z * delta_z_idx)  #int(np.minimum(i_z*delta_z_idx + 1, num_z-1))
            for i_y in range(0, num_sampled_y):
                ii_ys = int(np.maximum(i_y*delta_y_idx -2, 0))
                ii_ye = int(np.minimum(i_y*delta_y_idx + 2, num_y-1))
                for i_x in range(0, num_sampled_x):
                    ii_xs = int(np.maximum(i_x*delta_x_idx -2, 0))
                    ii_xe = int((np.minimum(i_x*delta_x_idx + 2, num_x-1)))
                    idx = int(i_x + i_y*num_sampled_x + i_z*num_sampled_x*num_sampled_y)

                    if (i_t == 0):
                        positions[idx] = [ii_xs, ii_ys, ii_zs]

                    if i_t == 0 and i_z == 0:
                        terrain_points[i_x + i_y*num_sampled_x] = [ii_xs, ii_ys, terrain_height[ii_xs + ii_ys*num_x]]

                    i_m = 0
                    # print ('ii_xs:', ii_xs, 'ii_xe:', ii_xe)
                    # print ('ii_ys:', ii_ys, 'ii_ye:', ii_ye)
                    # print ('ii_zs:', ii_zs, 'ii_ze:', ii_ze)
                    for iii_z in  range(ii_zs, ii_ze+1):
                        for iii_y in range(ii_ys, ii_ye+1):
                            for iii_x in range(ii_xs, ii_xe+1):
                                vec = local_vectors[iii_x + iii_y*num_x + iii_z*num_x*num_y]
                                sp_coords = vf_utils.getSphericalCoordinates(vec)
                                # print('i_t:', i_t, 'idx:', idx, 'i_m:', i_m, 'sp_coords:', sp_coords)
                                # print('vectors.shape:', vectors.shape)
                                vectors[i_t, idx, i_m, 0] = sp_coords[0]
                                vectors[i_t, idx, i_m, 1] = sp_coords[1]
                                vectors[i_t, idx, i_m, 2] = sp_coords[2]
                                i_m = i_m + 1

                    # vec = local_vectors[ii_x + ii_y*num_x + ii_z*num_x*num_y]
                    # for i_m in range(0, num_sampled_ensemble_members):
                    #     vec_tmp = vec
                    #     vec_tmp[0] = vec_tmp[0]*( 1 + 0.2*np.random.rand())
                    #     vec_tmp[1] = vec_tmp[1]*( 1 + 0.2*np.random.rand())
                    #     vec_tmp[2] = vec_tmp[2]*( 1 + 0.2*np.random.rand())
                    #     sp_coords = vf_utils.getSphericalCoordinates(vec_tmp)
                    #     vectors[i_t, idx, i_m, 0] = sp_coords[0]
                    #     vectors[i_t, idx, i_m, 1] = sp_coords[1]
                    #     vectors[i_t, idx, i_m, 2] = sp_coords[2]

                    first_quadrant_flag = False
                    second_quadrant_flag = False
                    third_quadrant_flag = False
                    fourth_quadrant_flag = False
                   
                    for i_m in range(num_sampled_ensemble_members):
                        if(i_t == 0 and idx == 643):
                            print('vectors[i_time][idx][i_m][2]:', vectors[i_t][idx][i_m][2], i_m, idx, i_t)
                        if( 0 <= vectors[i_t][idx][i_m][2] and vectors[i_t][idx][i_m][2] <= np.pi*0.5):
                            first_quadrant_flag = True
                        elif(np.pi*0.5 < vectors[i_t][idx][i_m][2] and vectors[i_t][idx][i_m][2] <= np.pi):
                            second_quadrant_flag = True
                        elif(-np.pi < vectors[i_t][idx][i_m][2] and vectors[i_t][idx][i_m][2] <= -np.pi*0.5):
                            third_quadrant_flag = True
                        elif(-np.pi*0.5 < vectors[i_t][idx][i_m][2] and vectors[i_t][idx][i_m][2] <= 0):
                            fourth_quadrant_flag = True
                    if( i_t == 0 and idx == 643):
                        print('vectors[i_time][idx]:', vectors[i_t][idx])
                        print('first_quadrant_flag:', first_quadrant_flag, second_quadrant_flag, third_quadrant_flag, fourth_quadrant_flag)
                        
                        if( vectors[i_t][idx][i_m][2]> np.pi or vectors[i_t][idx][i_m][2] < -np.pi or 
                            vectors[i_t][idx][i_m][0] < 0 or vectors[i_t][idx][i_m][1] < 0):
                            print('vectors[i_time][idx][i_m]:', vectors[i_t][idx][i_m])
                            raise ValueError('Invalid spherical coordinates')
                    if((second_quadrant_flag == True and third_quadrant_flag == True) or 
                       (first_quadrant_flag == True and third_quadrant_flag == True) or
                       (second_quadrant_flag == True and fourth_quadrant_flag == True) ):
                        for i_m in range(num_sampled_ensemble_members):
                            if(vectors[i_t][idx][i_m][2] < 0):
                                vectors[i_t][idx][i_m][2] =  vectors[i_t][idx][i_m][2] + 2*np.pi


    cell_diag = np.sqrt( delta_x_idx**2 + delta_y_idx**2 + delta_z_idx**2)
    max_magnitude = getMaxMagnitude(vectors)
    dd = {  'num_time_steps': num_sampled_time_steps,
            'num_points': num_points,
            'dimensions': 3,   
            'selected_depth': [0.0],
            'num_ensemble_members':num_sampled_ensemble_members,
            'positions': [positions],
            'vectors': [vectors],
            'domain': [np.array([[0, num_x], [0, num_y], [0, num_z]])],
            'depths': [np.zeros([num_sampled_time_steps, num_points, num_sampled_ensemble_members])],
            'min_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
            'median_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
            'max_vectors': [-1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
            'variability': [np.zeros([num_sampled_time_steps, num_points, 3])],
            'max_magnitude': [max_magnitude],
            'cell_diag': [cell_diag]}
    return pd.DataFrame(data=dd), terrain_points



def readHurricaneData():
    num_ensemble_members = 1
    num_time_steps = 48
    num_x = 500
    num_y = 500
    num_z = 100
    num_sampled_ensemble_members = 10
    num_sampled_time_steps = 5
    num_sampled_x = 20
    num_sampled_y = 20
    num_sampled_z = 10
    ## get indices for sampled data
    x_indices = getSampleIndices([0, num_x-1], num_sampled_x)
    y_indices = getSampleIndices([0, num_y-1], num_sampled_y)
    z_indices = getSampleIndices([0, num_z-1], num_sampled_z)
    ensemble_indices = np.linspace(0, num_ensemble_members-1, num_sampled_ensemble_members)
    time_indices = getSampleIndices([0, num_time_steps-1], num_sampled_time_steps)
    # time_indices = np.array([0, 2, 4, 6, 8]) 
    # getSampleIndices([0, num_time_steps-1], num_sampled_time_steps)
    num_points = num_sampled_x * num_sampled_y * num_sampled_z
    positions = np.zeros((num_points, 3))
    vectors = np.zeros((num_sampled_time_steps, num_points, num_sampled_ensemble_members, 3))
    terrain_points = np.zeros((num_sampled_x*num_sampled_y, 3))
    terrain = file_name = '/mnt/Storage1/hurrican_isabel/isabel_terrain.vti'
    # read the terrain data
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(terrain)
    reader.Update()
    # Get the output of the reader
    # print('reader.GetOutput():', reader.GetOutput())
    terrain_output = dsa.WrapDataObject(reader.GetOutput())
    terrain_height = terrain_output.PointData['HGTdata']

    for i_t in range(0,  num_sampled_time_steps):
        ii_t = time_indices[i_t]
        if ii_t < 10:
            local_vectors = np.load('/mnt/Storage1/hurrican_isabel/isabel0'+str(ii_t+1)+'.npy')
        else:
            local_vectors = np.load('/mnt/Storage1/hurrican_isabel/isabel'+str(ii_t)+'.npy')
        for i_z in range(0, num_sampled_z):
            ii_z = z_indices[i_z]
            for i_y in range(0, num_sampled_y):
                ii_y = y_indices[i_y]
                for i_x in range(0, num_sampled_x):
                    ii_x = x_indices[i_x]
                    idx = i_x + i_y*num_sampled_x + i_z*num_sampled_x*num_sampled_y

                    if (i_t == 0):
                        positions[idx] = [ii_x, ii_y, ii_z]

                    if i_t == 0 and i_z == 0:
                        terrain_points[i_x + i_y*num_sampled_x] = [ii_x, ii_y, terrain_height[ii_x + ii_y*num_x]]
                    vec = local_vectors[ii_x + ii_y*num_x + ii_z*num_x*num_y]
                    for i_m in range(0, num_sampled_ensemble_members):
                        vec_tmp = vec
                        vec_tmp[0] = vec_tmp[0]*( 1 + 0.2*np.random.rand())
                        vec_tmp[1] = vec_tmp[1]*( 1 + 0.2*np.random.rand())
                        vec_tmp[2] = vec_tmp[2]*( 1 + 0.2*np.random.rand())
                        sp_coords = vf_utils.getSphericalCoordinates(vec_tmp)
                        vectors[i_t, idx, i_m, 0] = sp_coords[0]
                        vectors[i_t, idx, i_m, 1] = sp_coords[1]
                        vectors[i_t, idx, i_m, 2] = sp_coords[2]

    cell_diag = np.sqrt( (x_indices[1]-x_indices[0])**2 + (y_indices[1]-y_indices[0])**2 + (z_indices[1]-z_indices[0])**2)
    max_magnitude = getMaxMagnitude(vectors)
    dd = {'num_time_steps': num_sampled_time_steps,
              'num_points': num_points,
                'dimensions': 3,    
                'num_ensemble_members':num_sampled_ensemble_members,
                'positions': [positions],
                'vectors': [vectors],
                'domain': [np.array([[0, num_x], [0, num_y], [0, num_z]])],
                'depths': [np.zeros([num_sampled_time_steps, num_points, num_sampled_ensemble_members])],
                'min_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
                'median_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
                'max_vectors': [-1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
                'variability': [np.zeros([num_sampled_time_steps, num_points, 3])],
                'max_magnitude': [max_magnitude],
                'cell_diag': [cell_diag]}
    return pd.DataFrame(data=dd), terrain_points


## Hurricane isabel data ##
def getHurricaneIsabelWindData():
    num_ensemble_members = 1
    num_time_steps = 48
    num_x = 500
    num_y = 500
    num_z = 100
    num_sampled_ensemble_members = 10
    num_sampled_time_steps = 2
    num_sampled_x = 20
    num_sampled_y = 20
    num_sampled_z = 5
    ## get indices for sampled data
    x_indices = getSampleIndices([0, num_x-1], num_sampled_x)
    y_indices = getSampleIndices([0, num_y-1], num_sampled_y)
    z_indices = getSampleIndices([0, num_z-1], num_sampled_z)
    ensemble_indices = np.linspace(0, num_ensemble_members-1, num_sampled_ensemble_members)
    # time_indices = getSampleIndices([0, num_time_steps-1], num_sampled_time_steps)
    time_indices = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9])
    num_points = num_sampled_x * num_sampled_y * num_sampled_z
    positions = np.zeros((num_points, 3))
    terrain_points = np.zeros((num_sampled_x*num_sampled_y, 3))
    vectors = np.zeros((num_sampled_time_steps, num_points, num_sampled_ensemble_members, 3))
    terrain = file_name = '/mnt/Storage1/hurrican_isabel/isabel_terrain.vti'
    # read the terrain data
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(terrain)
    reader.Update()
    # Get the output of the reader
    # print('reader.GetOutput():', reader.GetOutput())
    terrain_output = dsa.WrapDataObject(reader.GetOutput())
    terrain_height = terrain_output.PointData['HGTdata']
    # print('terrain_height.shape:', terrain_height.shape)
    # print('terrain_height:', terrain_height)


    for i_t in range(0,  num_sampled_time_steps):
        ii_t = time_indices[i_t]
        if ii_t < 10:
            file_name = '/mnt/Storage1/hurrican_isabel/isabel0'+str(ii_t+1)+'.vti'
        else:
            file_name = '/mnt/Storage1/hurrican_isabel/isabel'+str(ii_t)+'.vti'
        # Read the source file.
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        # Get the output of the reader
        output = dsa.WrapDataObject(reader.GetOutput())
        # get the points
        Uf = output.PointData['Uf']
        Vf = output.PointData['Vf']
        Wf = output.PointData['Wf']
        for i_f in range(Uf.shape[0]):
            if(Uf[i_f] > 1.0e+10):
                Uf[i_f] = 0.0
        for i_f in range(Vf.shape[0]):
            if(Vf[i_f] > 1.0e+10):
                Vf[i_f] = 0.0
        for i_f in range(Wf.shape[0]):
            if(Wf[i_f] > 1.0e+10):
                Wf[i_f] = 0.0
        print( 'Uf.shape:', Uf.shape)
        # print( 'Vf.shape:', Vf.shape)
        # print( 'Wf.shape:', Wf.shape)  
        vectors = np.zeros(())         

        for i_z in range(0, num_sampled_z):
            ii_z = z_indices[i_z]
            for i_y in range(0, num_sampled_y):
                ii_y = y_indices[i_y]
                for i_x in range(0, num_sampled_x):
                    ii_x = x_indices[i_x]
                    if i_t == 0 and i_z == 0:
                        terrain_points[i_x + i_y*num_sampled_x] = [ii_x, ii_y, terrain_height[ii_x + ii_y*num_x]]
                    if i_t == 0:
                        positions[i_x*num_sampled_y*num_sampled_z + i_y*num_sampled_z + i_z] = [ii_x, ii_y, ii_z]
                    vec = [Uf[ii_x + ii_y*num_x + ii_z*num_x*num_y], Vf[ii_x + ii_y*num_x + ii_z*num_x*num_y], Wf[ii_x + ii_y*num_x + ii_z*num_x*num_y]]
                    for i_ens in range(0, num_sampled_ensemble_members):
                        vec_tmp = vec
                        vec_tmp[0] = vec_tmp[0]*( 1 + 0.01*np.random.rand())
                        vec_tmp[1] = vec_tmp[1]*( 1 + 0.2*np.random.rand())
                        vec_tmp[2] = vec_tmp[2]*( 1 + 0.2*np.random.rand())
                        sp_coords = vf_utils.getSphericalCoordinates(vec_tmp)
                        vectors[i_t, i_x*num_sampled_y*num_sampled_z + i_y*num_sampled_z + i_z, i_ens, 0] = sp_coords[0]
                        vectors[i_t, i_x*num_sampled_y*num_sampled_z + i_y*num_sampled_z + i_z, i_ens, 1] = sp_coords[1]
                        vectors[i_t, i_x*num_sampled_y*num_sampled_z + i_y*num_sampled_z + i_z, i_ens, 2] = sp_coords[2]
    cell_diag = np.sqrt( (x_indices[1]-x_indices[0])**2 + (y_indices[1]-y_indices[0])**2 + (z_indices[1]-z_indices[0])**2)
    max_magnitude = getMaxMagnitude(vectors) 
    dd = {'num_time_steps': num_sampled_time_steps,
              'num_points': num_points,
                'dimensions': 3,
                'num_ensemble_members':num_sampled_ensemble_members,
                'positions': [positions],
                'vectors': [vectors],
                'domain': [np.array([[0, num_x], [0, num_y], [0, num_z]])],
                'depths': [np.zeros([num_sampled_time_steps, num_points, num_sampled_ensemble_members])],
                'min_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
                'median_vectors': [1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
                'max_vectors': [-1.0e+10*np.ones([num_sampled_time_steps, num_points, 3])],
                'variability': [np.zeros([num_sampled_time_steps, num_points, 3])],
                'max_magnitude': [max_magnitude],
                'cell_diag': [cell_diag]}
    return pd.DataFrame(data=dd), terrain_points       



def generateExampleData(num_time_steps, n, dimensions, num_ensemble_members, name='example0'):
    if(name == 'example0'):
        
        num_points = n*n
        # print('n =', n)
        x = np.linspace(-np.pi, np.pi, n)
        positions  = np.zeros([n*n,  3])
        vectors = np.zeros( [num_time_steps, n*n, num_ensemble_members, 3])
        vec = np.zeros([3])
        # angles = np.linspace(0.0, np.pi*0.25, num_time_steps) # small angles variavtion
        angles = np.linspace(0.0, np.pi*0.75, num_time_steps) # small angles variavtion

        i_time = 0
        count = -1
        for theta in angles:
            count = count + 1
            idx = 0
            for y_val in x:
                for x_val in x:
                    positions[idx][0] = x_val   # x coodinates
                    positions[idx][1] = y_val   # y coodinates
                    positions[idx][2] = 0.0    # z coodinates
                    tmp = [np.sin(y_val+x_val), np.sin(x_val-y_val)]
                    tmp = [np.sin(x_val), np.sin(y_val)]
                    # rotate vector by angle theta
                    vec[0] = tmp[0]*np.cos(theta) - tmp[1]*np.sin(theta) # x component of vector 
                    vec[1] = tmp[0]*np.sin(theta) + tmp[1]*np.cos(theta) # y component of vector
                    if(vec[0] >0 or vec[1] > 0):
                        vec[2] = 0.5 # z component of vector
                    else:
                        vec[2] = 0.0
                    #
                    # print('position:', positions[idx])
                    # sp_coords = vf_utils.getSphericalCoordinates(vec)
                    sp_coords = vf_utils.cartesian2Spherical(vec)


                    sp_coords[1] = np.pi*0.5
                    # if(sp_coords[1] < 0.5):
                    #     print('vec:', vec)
                    #     print('sp_coords:', sp_coords)
                    #     print('catt2sp:', vf_utils.cartesian2Spherical(vec))
                    #     raise ValueError('Invalid spherical coordinates')
                    # print('sp_coords:', sp_coords)
                    scaling_factor = 0.01
                    for i_m in range(num_ensemble_members):
                        vec_tmp = vec
                        if (x_val > 2.14 and y_val > 2.14 and (i_m == 0 or i_m == 0)):
                            vec_tmp[0] = vec_tmp[0]*( 1 + 0.05*np.random.rand())
                            vec_tmp[1] = vec_tmp[1]*( 1 + 0.05*np.random.rand())
                            vec_tmp[2] = vec_tmp[2]*( 1 + 0.05*np.random.rand())
                            # vec_tmp[0] = vec[0]*np.cos(np.pi/3) - vec[1]*np.sin(np.pi/3)
                            # vec_tmp[1] = vec[0]*np.sin(np.pi/3) + vec[1]*np.cos(np.pi/3)
                            # vec_tmp[2] = vec[2]
                        else:
                            vec_tmp[0] = vec_tmp[0]*( 1 + 0.05*np.random.rand())
                            vec_tmp[1] = vec_tmp[1]*( 1 + 0.05*np.random.rand())
                            vec_tmp[2] = vec_tmp[2]*( 1 + 0.05*np.random.rand())
                        # if(theta == np.pi*0.25):
                            # print('vec_tmp:', vec_tmp)
                            # print('sp_coords:', vf_utils.getSphericalCoordinates(vec_tmp))
                        # sp_coord = vf_utils.getSphericalCoordinates(vec_tmp)
                        sp_coord = vf_utils.cartesian2Spherical(vec_tmp)
                        vectors[i_time][idx][i_m][0] =  sp_coord[0]
                        vectors[i_time][idx][i_m][1] =  sp_coord[1]
                        vectors[i_time][idx][i_m][2] =  sp_coord[2]
                        
                        # vectors[i_time][idx][i_m][0] =  2*(1.0 + 1*scaling_factor*np.random.rand() )*sp_coords[0]
                        # vectors[i_time][idx][i_m][1] =  (1.0 + 1*scaling_factor*np.random.rand() )*sp_coords[1]
                        # vectors[i_time][idx][i_m][2] =  (1.0 + 1*scaling_factor*np.random.rand() )*sp_coords[2]
                        # # print('vector:', vectors[i_time][idx][i_m])

                    first_quadrant_flag = False
                    second_quadrant_flag = False
                    for i_m in range(num_ensemble_members):
                        if(np.pi*0.5 <= vectors[i_time][idx][i_m][2] and vectors[i_time][idx][i_m][2] <= np.pi):
                            first_quadrant_flag = True
                        if(-np.pi <= vectors[i_time][idx][i_m][2] and vectors[i_time][idx][i_m][2] <= -np.pi*0.5):
                            second_quadrant_flag = True
                        if( vectors[i_time][idx][i_m][2]> np.pi or vectors[i_time][idx][i_m][2] < -np.pi or 
                            vectors[i_time][idx][i_m][0] < 0 or vectors[i_time][idx][i_m][1] < 0):
                            print('vectors[i_time][idx][i_m]:', vectors[i_time][idx][i_m])
                            raise ValueError('Invalid spherical coordinates')
                    if(first_quadrant_flag and second_quadrant_flag):
                        for i_m in range(num_ensemble_members):
                            if(vectors[i_time][idx][i_m][2] < 0):
                                vectors[i_time][idx][i_m][2] =  vectors[i_time][idx][i_m][2] + 2*np.pi

                    for i_m in range(num_ensemble_members):    
                        if (x_val > 2.14 and y_val > 2.14 and count < 2):
                            # print('vector:[', i_m, ']', vectors[i_time][idx][i_m])
                            if( i_m == 0):
                                vectors[i_time][idx][i_m][0] = vectors[i_time][idx][i_m][0]*2.5
                                vectors[i_time][idx][i_m][1] = vectors[i_time][idx][i_m][1]*1.2
                                vectors[i_time][idx][i_m][2] = vectors[i_time][idx][i_m][2]*1.2

                    idx += 1
            i_time += 1
            # print('i_time =', i_time)   
        cell_diag = np.sqrt( (x[1]-x[0])**2 + (x[1]-x[0])**2)
        max_magnitude = getMaxMagnitude(vectors)
        # print('vector_data =', vector_data.shape)
        dd = {'num_time_steps': num_time_steps, 
              'num_points': n*n, 
              'dimensions': 3, 
              'selected_depth': [0.0], 
              'num_ensemble_members':num_ensemble_members, 
              'positions': [positions], 
              'vectors': [vectors],
              'domain': [np.array([[-np.pi-1.0e-10, np.pi+1.0e-10], [-np.pi-1.0e-10, np.pi+1.0e-10], [-1.0e-10, 1.0e-10]])],
              'depths': [np.zeros([num_time_steps, num_points, num_ensemble_members])],
              'min_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
              'median_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
              'max_vectors': [-1.0e+10*np.ones([num_time_steps, num_points, 3])], 
              'variability': [np.zeros([num_time_steps, num_points, 3])],
              'max_magnitude': [max_magnitude],
              'cell_diag': [cell_diag]}
        return pd.DataFrame(data=dd)
    elif(name == 'example1'):
        # Domain [-1,1]x[-1,1]x[-1,1]x[0,10]
        num_points = n*n*n
        positions  = np.zeros([n*n*n,  3])
        vectors = np.zeros( [num_time_steps, n*n*n, num_ensemble_members, 3])
        vec = np.zeros([3])
        x = np.linspace(-1.0, 1.0, n)
        y = np.linspace(-1.0, 1.0, n)
        z = np.linspace(-1.0, 1.0, n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    positions[i*n*n+j*n+k][0] = x[i]
                    positions[i*n*n+j*n+k][1] = y[j]
                    positions[i*n*n+j*n+k][2] = z[k]
                    phi = np.linspace(0.0, np.pi*0.8, num_time_steps)
                    for l in range(num_time_steps):
                        vec[0] = x[i]*x[i]*x[i] #y[j] +2*z[k]
                        vec[1] = y[j]*y[j]*y[j] #x[i] - z[k]
                        vec[2] = z[k]*z[k]*z[k] #2*x[i] - y[j]
                        #
                        Ry = np.array([[np.cos(phi[l]), 0, np.sin(phi[l])], [0, 1, 0], [-np.sin(phi[l]), 0, np.cos(phi[l])]])
                        
                        for i_m in range(num_ensemble_members):
                            vec_tmp = np.dot(Ry, vec)
                            vec_tmp[0] = vec_tmp[0]*( 1 + 0.1*np.random.rand())
                            vec_tmp[1] = vec_tmp[1]*( 1 + 0.1*np.random.rand())
                            vec_tmp[2] = vec_tmp[2]*( 1 + 0.1*np.random.rand())
                            sp_coord = vf_utils.getSphericalCoordinates(vec_tmp)
                            vectors[l][i*n*n+j*n+k][i_m][0] =  sp_coord[0]
                            vectors[l][i*n*n+j*n+k][i_m][1] =  sp_coord[1]
                            vectors[l][i*n*n+j*n+k][i_m][2] =  sp_coord[2]
                        # sp_coords = vf_utils.getSphericalCoordinates(vec)
                        # for i_m in range(num_ensemble_members):
                        #     vectors[l][i*n*n+j*n+k][i_m][0] = 2*(1.0 + 10*np.random.rand() )*sp_coords[0]
                        #     vectors[l][i*n*n+j*n+k][i_m][1] = (1.0 + 100*np.random.rand() )*sp_coords[1]
                        #     vectors[l][i*n*n+j*n+k][i_m][2] = (1.0 + 100*np.random.rand() )*sp_coords[2]

        cell_diag = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2 + (z[1]-z[0])**2)
        max_magnitude = getMaxMagnitude(vectors)
        dd = {'num_time_steps': num_time_steps,
                'num_points': n*n*n,
                'dimensions': 3,
                'num_ensemble_members':num_ensemble_members,
                'positions': [positions],
                'vectors': [vectors],
                'domain': [np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])],
                'depths': [np.zeros([num_time_steps, num_points, num_ensemble_members])],
                'min_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
                'median_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
                'max_vectors': [-1.0e+10*np.ones([num_time_steps, num_points, 3])],
                'variability': [np.zeros([num_time_steps, num_points, 3])],
                'max_magnitude': [max_magnitude],
                'cell_diag': [cell_diag]}
        return pd.DataFrame(data=dd)
     
    elif(name == 'doublegyre'):
        # Domain x, y, t [0, 2]x[0, 1]x[0, 10] 
        num_points = n*n
        # print('n =', n)
        positions  = np.zeros([n*n,  3])
        vectors = np.zeros( [num_time_steps, n*n, num_ensemble_members, 3])
        vec = np.zeros([3])
        x = np.linspace(0.0, 2.0, n)
        y = np.linspace(0.0, 1.0, n)
        t = np.linspace(0.0, 10.0, num_time_steps)
        for i in range(n):
            for j in range(n):
                positions[i*n+j][0] = x[i]
                positions[i*n+j][1] = y[j]
                positions[i*n+j][2] = 0.0
                for k in range(num_time_steps):
                    u, v = doublegyre2d().sample(x[i], y[j], t[k])
                    vec[0] = u
                    vec[1] = v
                    vec[2] = 0.0
                    #
                    # sp_coords = vf_utils.getSphericalCoordinates(vec)

                    # print('point id:', i*n + j, 'u =', u, 'v =', v)
                    for i_m in range(num_ensemble_members):
                        vec_tmp = vec
                        vec_tmp[0] = vec_tmp[0]*( 1 + 0.25*np.random.rand())
                        vec_tmp[1] = vec_tmp[1]*( 1 + 0.25*np.random.rand())
                        vec_tmp[2] = vec_tmp[2]*( 1 + 0.25*np.random.rand())
                        sp_coord = vf_utils.getSphericalCoordinates(vec_tmp)
                        vectors[k][i*n+j][i_m][0] =  sp_coord[0]
                        vectors[k][i*n+j][i_m][1] =  sp_coord[1]
                        vectors[k][i*n+j][i_m][2] =  sp_coord[2]

                        # vectors[k][i*n+j][i_m][0] = 2*(1.0 + 10*np.random.rand() )*sp_coords[0]
                        # vectors[k][i*n+j][i_m][1] = (1.0 + 1*np.random.rand() )*sp_coords[1]
                        # vectors[k][i*n+j][i_m][2] = (1.0 + 1*np.random.rand() )*sp_coords[2]

                       
        max_magnitude = getMaxMagnitude(vectors)
        cell_diag = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2)
        dd = {'num_time_steps': num_time_steps,
              'num_points': n*n,
              'dimensions': 2,
              'num_ensemble_members':num_ensemble_members,
              'positions': [positions],
              'vectors': [vectors],
              'domain': [np.array([[0.0, 2.0], [0.0, 1.0], [-1.0e-10, 1.0e-10]])],
              'depths': [np.zeros([num_time_steps, num_points, num_ensemble_members])],
              'min_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
              'median_vectors': [1.0e+10*np.ones([num_time_steps, num_points, 3])],
              'max_vectors': [-1.0e+10*np.ones([num_time_steps, num_points, 3])],
              'variability': [np.zeros([num_time_steps, num_points, 3])],
              'max_magnitude': [max_magnitude],
              'cell_diag': [cell_diag]}
        return pd.DataFrame(data=dd)


def getMaxMagnitude(vectors):
    """
    purpose: compute the maximum magnitude of the vectors
    input: vectors: numpy array of shape (num_time_steps, num_points, num_ensemble_members, 3)
    output: max_magnitude
    
    """
    max_magnitude = 0.0
    for t in range(vectors.shape[0]):
        for i in range(vectors.shape[1]):
            for j in range(vectors.shape[2]):

                if( max_magnitude < np.absolute(vectors[t][i][j][0])):
                    max_magnitude = np.absolute(vectors[t][i][j][0])

    if(max_magnitude == 0.0):
        raise ValueError('Invalid max magnitude')
    
    return max_magnitude



class doublegyre2d(object):
    """The Double Gyre vector field."""

    def __init__(self, a=0.1, eps=0.25, omega=pi/5):
        self.A = a
        self.EPS = eps
        self.OMEGA = omega

    # Samples the Double Gyre vector field
    def sample(self, x, y, t):
        u = -self.A * pi * sin(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*cos(pi*y)
        v = self.A * pi*(2 * self.EPS*sin(self.OMEGA*t)*x - 2 * self.EPS*sin(self.OMEGA*t) + 1)* cos(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*sin(pi*y)
        return u, v;

    # Samples the x-partial of the Double Gyre vector field
    def sample_dx(self, x, y, t):
        u = -self.A * pi*pi * (2 * self.EPS*sin(self.OMEGA*t)*x - 2 * self.EPS*sin(self.OMEGA*t) + 1)*cos(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*cos(pi*y)
        v = 2 * self.A*self.EPS*pi* sin(self.OMEGA*t)*cos(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*sin(pi*y) - self.A * pi*pi * (2 * self.EPS*sin(self.OMEGA*t)*x - 2 * self.EPS*sin(self.OMEGA*t) + 1) * (2 * self.EPS*sin(self.OMEGA*t)*x - 2 * self.EPS*sin(self.OMEGA*t) + 1) * sin(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*sin(pi*y)
        return u, v;

    # Samples the y-partial of the Double Gyre vector field
    def sample_dy(self, x, y, t):
        u = self.A * pi*pi * sin(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*sin(pi*y)
        v = self.A * pi*pi * (2 * self.EPS*sin(self.OMEGA*t)*x - 2 * self.EPS*sin(self.OMEGA*t) + 1)*cos(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*cos(pi*y)
        return u, v;

    # Samples the t-partial of the Double Gyre vector field
    def sample_dt(self, x, y, t):
        u = -self.A * pi*pi * (self.EPS*self.OMEGA*cos(self.OMEGA*t)*x*x - 2 * self.EPS*self.OMEGA*cos(self.OMEGA*t)*x)*cos(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*cos(pi*y)
        v = self.A * pi *	(2 * self.EPS*self.OMEGA*cos(self.OMEGA*t)*x - 2 * self.EPS*self.OMEGA*cos(self.OMEGA*t))*cos(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*sin(pi*y) - self.A * pi*pi * (2 * self.EPS*sin(self.OMEGA*t)*x - 2 * self.EPS*sin(self.OMEGA*t) + 1)*(self.EPS*self.OMEGA*cos(self.OMEGA*t)*x*x - 2 * self.EPS*self.OMEGA*cos(self.OMEGA*t)*x) * sin(pi*(self.EPS*sin(self.OMEGA*t)*x*x + (1 - 2 * self.EPS*sin(self.OMEGA*t))*x))*sin(pi*y)
        return u, v;


def loadData(name):
    if(name == 'example0' or name == 'doublegyre' or name == 'example1'):
        num_time_steps = 5
        num_points = 10
        dimensions = 2
        if(name == 'example1' or name == 'example0'):
            dimensions = 3
        num_ensemble_members = 20
        num_selected_points=dimensions+1
        # depth_threshold = 0.0
        # domain = np.array([[-np.pi-1.0e-10, np.pi+1.0e-10], [-np.pi, np.pi+1.0e-10]])
        time_range = np.array([0, num_time_steps-1])
        print(' Generating example data ...')
        data  = generateExampleData(num_time_steps, num_points, dimensions, num_ensemble_members, name=name)
        # print("domain:", data['domain'][0])
        # print(' Computing statistics ...')
        # vf_stats.dataDepth(data, num_selected_points)
        print(' Computing statistics ...')
        vf_stats.hyperCubeDataDepth(data, num_selected_points)
        return data
  

print(' Loading data  and computing statistics ...')
example_name = 'example0' #'hurricane_isabel' #'fire_wind' #'hurricane_isabel' #'example0' #'wind' #'hurricane_isabel' #'doublegyre'
data = loadData(example_name)
# data, hurricane_isabel_terrain = loadData('hurricane_isabel')


terrain = False
# print('data[domain]:', data['domain'][0])
global_polygones =  vf_glyphs.buildGlyphsForDash(data, scale=0.5, resolution=10, time_range =[0,0], 
                                                 data_depth = 0.0, domain= data['domain'][0], glyph_type='cone', 
                                                 mean_flag= False, compute_min_max_flag= True)

## Construct box 
box = dash_vtk.GeometryRepresentation(
    property={"edgeVisibility": True, "edgeColor":(0,0,0), "opacity": 0.1},
    actor={"position": (0, 0, 0), "scale": (2, 2, 2)},
    children=[dash_vtk.Algorithm(id="box-selection-id", vtkClass="vtkCubeSource")],
)
global_polygones.append(box)
content = dash_vtk.View(global_polygones, background=[1,1,1])
print(' Building glyphs ...')

## Construct local view
local_polygons = vf_glyphs.buildGlyphsForDash(data, scale=0.1, resolution=10, time_range =[0,0], data_depth = 0.0, 
                                              domain= data['domain'][0], glyph_type='cone', mean_flag= False, compute_min_max_flag= False)

print('Done building glyphs ...')
local_content = dash_vtk.View(local_polygons, background=[1,1,1])
# Figure
fig0 = go.Figure( data = [go.Heatmap(z=data['depths'][0][0], colorscale='Cividis')])
fig0.update_layout(margin=dict(l=0, r=0, t=0, b=0), font=dict(size=fs))

# Figure line plots
fig1 = make_subplots(rows=2, cols=1)
y_data = []
for i in range(0, 1):
    fig1.add_trace(go.Scatter(x=np.linspace(0, data['num_points'][0], data['num_points'][0]), y=data['variability'][0][i][:,0], mode='lines', name='min'), row=1, col=1)
    y_data.append(np.max(data['variability'][0][i][:,0]))
fig1.add_trace(go.Scatter(x=np.linspace(0, len(y_data), len(y_data)), y=y_data, mode='lines', name='max'), row=2, col=1)
#
fig1.update_layout(
    margin=dict(l=0, r=0, t=0, b=0), 
    showlegend=False,
    font=dict(size=20))


### App design for comparing glyphs ###
comapare_glyphs_flag = False
if(comapare_glyphs_flag):
    comparison_view_cards = []
    scale = 0.9
    if (example_name == 'doublegyre'):
        local_domain = np.array([[0.01, 0.35], [0.01, 0.35], [-0.01, 0.01]])
    if (example_name == 'example1'):
        local_domain = np.array([[0, 1], [0, 1], [0, 1]])
    if (example_name == 'example0'):
        local_domain = np.array([[2, 3.5], [2, 3.5], [-0.1, 0.1]])
    global_domain = data['domain'][0]
    time_range = np.array([0, 4])
    vector_depth = 0.0
    glyph_types = ['arrows', 'cone',  'disc-arrow', 'comet', 'squid', 'new-cone', 'new-cone']

    for glyph_type in glyph_types:
        if(glyph_type == 'arrows'):
            global_polygones = vf_glyphs.buildArrowGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range,
                                                                data_depth = vector_depth, domain=global_domain, point_ids=[0])
        else:
            global_polygones =  vf_glyphs.buildGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
                                                        data_depth = vector_depth, glyph_type=glyph_type)

        box_points, box_polygons = vf_glyphs.getBoxPolyData(local_domain)
        box = dash_vtk.GeometryRepresentation(
            children=[
                dash_vtk.PolyData(
                    id="vtk-polydata",
                    points=np.array(box_points).ravel(),
                    polys=np.array(box_polygons),
                )],
            property={'opacity':0.8, 'color': [0, 0, 0], 'representation':1},
        )
        global_polygones.append(box)
        global_content = dash_vtk.View(global_polygones, 
                                background=[1,1,1])

        ## Construct local view
        if(glyph_type == 'arrows'):
            local_polygons = vf_glyphs.buildArrowGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
                                                            data_depth = vector_depth, domain=local_domain, point_ids=[0])
        else:
            local_polygons = vf_glyphs.buildGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
                                                        data_depth = vector_depth, domain=local_domain, glyph_type=glyph_type)

        local_content = dash_vtk.View(local_polygons, background=[1,1,1])
        # Dash setup
        ##
        # view card
        comparison_view_card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([ html.H3("Global View")], width=6),
                    dbc.Col([ html.H3("Local View")], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        # html.Label('Global View'),
                        dbc.Card([
                            html.Div(
                                id= 'global-view-id',
                                children = global_content, 
                                style={"width": "98%", "margin-left": "1%", "margin-right": "1%", "margin-top": "10px", "margin-bottom": "10px", "height": "calc(45vh - 15px)", 'float': 'center', 'display': 'inline-block'},
                            ),
                        ]),
                    ], width=6),
                    #
                    dbc.Col([
                        # html.Label('Local View'),
                        dbc.Card([
                            html.Div(
                                id = 'local-view-id',
                                children = local_content,
                                style={"width": "98%", "margin-left": "1%", "margin-right": "1%", "margin-top": "10px", "margin-bottom": "10px", "height": "calc(45vh - 15px)", 'float': 'center', 'display': 'inline-block'},
                            )
                        ]),
                    ], width=6),
                ]),
                html.Br(),
            ]),
        ])
        
        comparison_view_cards.append(comparison_view_card)
        

    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server
    
    app.layout = dbc.Container([
        html.H1("Glyphs comparison"),
        comparison_view_cards[0],
        comparison_view_cards[1],
        comparison_view_cards[2],
        comparison_view_cards[3],
        comparison_view_cards[4],
        comparison_view_cards[5],
        comparison_view_cards[6],
        ])

    if __name__ == "__main__":
        app.run(debug=True)



run_main_app = True
if(run_main_app):
    ### App design ###

    # Dash setup
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server

    fs = 35

    # glyph_type 
    glyph_type = 'squid'#'squid'#'cone'#'squid' #'disc-arrow' #'new-cone'

    # data = loadData('example0')
    # data = load_data('exmaple0')
    # drop down
    x_min_val = data['domain'][0][0][0]
    x_max_val = data['domain'][0][0][1]
    str_x_min_val = "{:.2f}".format(x_min_val)
    str_x_max_val = "{:.2f}".format(x_max_val)
    min_mag_val  = 0.0
    max_mag_val = data['max_magnitude'][0]
    str_min_mag_val = "{:.2f}".format(min_mag_val)
    str_max_mag_val = "{:.2f}".format(max_mag_val)
    #
    y_min_val = data['domain'][0][1][0]
    y_max_val = data['domain'][0][1][1]
    str_y_min_val = "{:.2f}".format(y_min_val)
    str_y_max_val = "{:.2f}".format(y_max_val)
    #
    z_min_val = data['domain'][0][2][0]
    z_max_val = data['domain'][0][2][1]
    str_z_min_val = "{:.2f}".format(z_min_val)
    str_z_max_val = "{:.2f}".format(z_max_val)
    t_min = 0
    t_max = data['num_time_steps'][0]-1

    drop_down = dcc.Dropdown(
        id='drop-down',
        options=[
            {'label': 'Example0', 'value': 0},
        ],
        value=0
    )
    # view card
    view_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([ html.H2("Global View")], width=6),
                # dbc.Col([
                #     html.Div([
                #         dbc.RadioItems(
                #             options=[
                #                 {"label": "Glyphs", "value": 0},
                #                 {"label": "Variability", "value": 1},
                #             ],
                #             value=0,
                #             id="global-view-radioitems",
                #             inline=True,
                            
                #         ),
                #     ]),
                # ], width=4),
                dbc.Col([ html.H2("Local View")], width=6),
                # dbc.Col([
                #     html.Div([
                #         dbc.RadioItems(
                #             options=[
                #                 {"label": "Glyphs", "value": 0},
                #                 {"label": "Variability", "value": 1},
                #             ],
                #             value=0,
                #             id="local-view-radioitems",
                #             inline=True,
                #         ),
                #     ]),
                # ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    # html.Label('Global View'),
                    dbc.Card([
                        html.Div(
                            id= 'global-view-id',
                            children = local_content, 
                            style={"width": "98%", "margin-left": "1%", "margin-right": "1%", "margin-top": "10px", "margin-bottom": "10px", "height": "calc(40vh - 15px)", 'float': 'center', 'display': 'inline-block'},
                        ),
                    ]),
                ], width=6),
                #
                dbc.Col([
                    # html.Label('Local View'),
                    dbc.Card([
                        html.Div(
                            id = 'local-view-id',
                            children = local_content,
                            style={"width": "98%", "margin-left": "1%", "margin-right": "1%", "margin-top": "10px", "margin-bottom": "10px", "height": "calc(40vh - 15px)", 'float': 'center', 'display': 'inline-block'},
                        )
                    ]),
                ], width=6),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.Label('Time selector'),
                        ], width=6),
                        dbc.Col([
                            dbc.RadioItems(
                                options=[
                                    {"label": "Max", "value": 0},
                                    {"label": "RMS", "value": 1},
                                ],
                                value=0,
                                id="time-selector-radioitems",
                                inline=True,
                            ),

                        ], width=6),
                    ]),
                    # dbc.Row([
                    #     dbc.Col([
                    #         daq.GraduatedBar(
                    #             color={"ranges":{"green":[0,4],"yellow":[4,7],"red":[7,100]}},
                    #             size = 300,
                    #             showCurrentValue=True,
                    #             value=100
                    #         ),
                    #     ], width=12),
                    # ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Slider(0, 5, step=1, value=0.0, marks={0: '0', 5: '5'}, tooltip={'placement':'bottom', 'always_visible':True}, id='time-slider-id'),
                            dcc.Graph(figure=fig1, id= 'time-depedent-variability-id'),
                        ], width=12, ),
                    ]),
                    #
                    dbc.Row([
                        dbc.Col([
                            # dcc.Slider(0, 5, step=1, value=0.0, marks={0: '0', 5: '5'}, tooltip={'placement':'bottom', 'always_visible':True}, id='time-slider-id'),
                            dcc.Graph(figure=fig1, id= 'variability-at-time-t-id'),
                        ], width=12),
                    ]),
                    
                ], width=6),
                #
                dbc.Col([
                    dbc.Row([
                        dbc.Col(
                            id = 'depth-distribution-slider-container',
                            children = [
                                dcc.Slider(min=0, 
                                        max=10, 
                                        step=1, 
                                        value=0.0, 
                                        marks={0: '0', 10: '10'}, 
                                        tooltip={'placement':'left', 'always_visible':True}, 
                                        vertical=True,
                                        id='depth-distribution-slider-id'),
                        ], width=3),
                        dbc.Col([ dcc.Graph(figure=fig0, id= 'depth-id-t0'),
                        ], width=9, ),
                    ]),
                    ##
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H5('Magnitude and Angular Distribution'), 
                        ], width=9),
                        dbc.Col([ 
                            dbc.RadioItems(
                                options=[
                                    {"label": "2D", "value": 0},
                                    {"label": "3D", "value": 1},
                                ],
                                value=0,
                                id="vector-distribution-radioitems-id",
                                inline=True,
                            ),
                        ], width=3),
                    ]),
                    dbc.Row([
                        dbc.Col([ dcc.Graph(figure=fig0, id= 'vectors-distribution-id-ip'),
                        ], width=12),
                    ]),
                    
                ], width=6),
            ]),

        ]),
    ])

    # tools card
    tools_card = dbc.Card([
        # dbc.CardHeader("Tools"),
        dbc.CardBody([
            # html.Div("Aprox. Error CDF"),
            # html.Div([
            #     dbc.Switch(value=False, id="error-cdf-switch", label="Off"),
            # ]),
            dbc.Row([
                dbc.Col([ 
                    html.H5('Time Range'),
                    dcc.RangeSlider(
                        t_min, 
                        t_max, 
                        step=1, 
                        value=[0, 1], 
                        marks=None, 
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='time-range-slider-id'
                    ),
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5('Vector Depth'),
                    dcc.Slider(
                        0, 
                        1, 
                        step=None, 
                        value=0.0, 
                        marks=None, 
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='data-depth-slider-id'
                    ),
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5('Scale Glyphs'),
                    dcc.Slider(
                        0, 
                        2, 
                        step=None, 
                        value=1.0, 
                        marks=None, 
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='scale-glyphs-slider-id'
                    ),
                ]),
            ]),
            html.Br(),
             dbc.Row([
                dbc.Col([ 
                    html.H5('Magnitude Threshold'),
                    dcc.RangeSlider(
                        min=min_mag_val, 
                        max=max_mag_val,
                        step=0.01, 
                        value=[min_mag_val, max_mag_val], 
                        marks=None, 
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='magnitude-threshold-slider-id'
                    ),
                ]),
            ]),
            html.Br(),
            html.Br(),
            html.H5("Selection Box"),
            html.Div([
                dbc.Switch(value=False, id="box-selection-switch", label="Off"),
            ]),
        
            ## box center
            dbc.Row([
                dbc.Col([html.Div(html.H6("Center"))], width=12),
                # dbc.Col([html.Div(html.H6("Dimensions"))], width=6)
            ]),
            #             
            dbc.Row([
                dbc.Col([html.Div("x")], width=2),
                dbc.Col(
                    id = 'cx-slider-container', 
                    children=[
                    dcc.Slider(min=x_min_val, max=x_max_val,
                        value= 0.5*(x_min_val + x_max_val),
                        step = 0.01,
                        tooltip={'placement':'bottom', 'always_visible':True},
                        marks={x_min_val+1e-6: {'label':''+str_x_min_val}, x_max_val-1e-6: {'label': ''+str_x_max_val}},
                        id='cx-slider',
                    ),
                    ], width=10),
            ]), 
            html.Br(),
            dbc.Row([
                dbc.Col([html.Div("y")], width=2),
                dbc.Col(
                    id = 'cy-slider-container',
                    children = [dcc.Slider(min=y_min_val, max=y_max_val,
                        value=0.5*(y_min_val + y_max_val),
                        step = 0.01,
                        tooltip={'placement':'bottom', 'always_visible':True},
                        marks={y_min_val+1e-6: {'label': ''+str_y_min_val}, y_max_val-1e-6: {'label': ''+str_y_max_val}},
                        id='cy-slider',
                    ),
                    ], width=10),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Div("z")], width=2),
                dbc.Col(
                    id = 'cz-slider-container', 
                    children=[
                    dcc.Slider(min=z_min_val, max=z_max_val,
                        value=0.5*(z_min_val + z_max_val),
                        step = 0.01,
                        tooltip={'placement':'bottom', 'always_visible':True},
                        marks={z_min_val+1e-6: {'label': ''+str_z_min_val}, z_max_val-1e-6: {'label': ''+str_z_max_val}},
                        id='cz-slider',
                    ),
                    ], width=10),
            ]),
            html.Br(),
            ## box dimension
            dbc.Row([
                dbc.Col([html.Div(html.H6("Dimensions"))], width=12)
            ]),
            #             
            dbc.Row([
                dbc.Col([html.Div("dx")], width=2),
                dbc.Col( 
                    id = 'dx-slider-container',
                    children=[
                    dcc.Slider(min=0, max=0.5*(x_max_val - x_min_val),
                        value=0.1*(x_max_val - x_min_val),
                        step = 0.01,
                        tooltip={'placement':'bottom', 'always_visible':True},
                        marks={0: '0', 0.5*(x_max_val - x_min_val)-1e-6: "{:.2f}".format(0.5*(x_max_val - x_min_val))},
                        id='dx-slider',
                    ),
                ], width=10),
            ]), 
            html.Br(),
            dbc.Row([
                dbc.Col([html.Div("dy")], width=2),
                dbc.Col(
                    id = 'dy-slider-container',
                    children =[
                    dcc.Slider(min=0, max=0.5*(y_max_val - y_min_val),
                        value=0.1*(y_max_val - y_min_val),
                        step = 0.01,
                        tooltip={'placement':'bottom', 'always_visible':True},
                        marks={0: '0', 0.5*(y_max_val - y_min_val)-1e-6: "{:.2f}".format(0.5*(y_max_val - y_min_val))},
                        id='dy-slider',
                    ),
                ], width=10),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Div("dz")], width=2),
                dbc.Col(
                    id = 'dz-slider-container',
                    children=[
                    dcc.Slider(min=0, max=0.5*(z_max_val - z_min_val),
                        value=0.1*(z_max_val - z_min_val),
                        step = 0.01,
                        tooltip={'placement':'bottom', 'always_visible':True},
                        marks={0: '0', 0.5*(z_max_val - z_min_val)-1e-6: "{:.2f}".format(0.5*(z_max_val - z_min_val))},
                        id='dz-slider',
                    ),
                ], width=10),
            ]),
            html.Br(),
        ]),
    ])

    # Summary card
    summary_card = dbc.Card([
        dbc.CardBody([
            dbc.Row(html.H5("Summary", style={"text-align": "center"})),
            dbc.Row(html.Hr()),
            dbc.Row([
                dbc.Col(children =[html.Div("")], width=4),
                dbc.Col(children =[html.Div("Global")], width=4),
                dbc.Col(children =[html.Div("Local")], width=4),
            ]),
            #
            dbc.Row([
                dbc.Col(html.Div('Num. points.'), width=4),
                dbc.Col(children =[html.Div(data['num_points'][0])], id='gobal-num-points', width=4),
                dbc.Col(children =[html.Div(data['num_points'][0])], id='local-num-points', width=4),
                # dbc.Col(children =[html.Div(('%.2E' % Decimal(np.mean(data["vertices_error"][0]))))], id='mean-error', width=3),
                # dbc.Col(children =[html.Div(('%.2E' % Decimal(np.sqrt(np.mean(data["vertices_error"][0])))))], id='rms-error', width=3),
            ]),
            #
            dbc.Row([
                dbc.Col(children =[html.Div('Num. vectors.')], width=4),
                dbc.Col(children =[html.Div(data['num_points'][0] * data['num_ensemble_members'][0])], id='gobal-num-vectors', width=4),
                dbc.Col(children =[html.Div(data['num_points'][0] * data['num_ensemble_members'][0])], id='local-num-vectors', width=4),
            ]),
        ]),  
            
    ])

    ## page layout
    page_layout = dbc.Row([
        dbc.Col([
            dbc.Row(tools_card), 
            dbc.Row(summary_card)
        ], width=3),
        dbc.Col(view_card, width=9), 
    ])
    
    ## App layout
    app.layout = html.Div([
        dbc.Row([
            dbc.Col(html.H1("Vector Field Uncertainty Vis."), width=4),
            dbc.Col(drop_down, width=4),

        ]),
        html.Div([page_layout]),
    ])


    @app.callback(
        Output('local-num-points', 'children'),
        Output('local-num-vectors', 'children'),
        #
        Input('box-selection-switch', 'value'),
        Input('cx-slider', 'value'),
        Input('cy-slider', 'value'),
        Input('cz-slider', 'value'),
        Input('dx-slider', 'value'),
        Input('dy-slider', 'value'),
        Input('dz-slider', 'value'),
        prevent_initial_call=True)
    def update_local_summary( box_selection, cx, cy, cz, dx, dy, dz):
        if(box_selection):
            domain = np.array([[cx-dx*0.5, cx+dx*0.5], [cy-dy*0.5, cy+dy*0.5], [cz-dz*0.5, cz+dz*0.5]])
            num_points = 0
            num_vectors = 0
            for i_pt in range(data['num_points'][0]):
                if(vf_utils.insideBoundingBox(data['positions'][0][i_pt], domain)):
                    num_points += 1
                    num_vectors += data['num_ensemble_members']
            return html.Div(num_points), html.Div(num_vectors)

        else:
            num_points = data['num_points'][0]
            num_vectors = data['num_points'][0]*data['num_ensemble_members']
            return html.Div(num_points), html.Div(num_vectors)

    @app.callback(
        Output('time-slider-id', 'max'),
        Output('time-slider-id', 'marks'),

        Input('time-range-slider-id', 'value'),
        prevent_initial_call=True)
    def update_time_slider_max(time_range):
        return time_range[1], {time_range[0]: str(time_range[0]), time_range[1]: str(time_range[1])}
    
    @app.callback( 
        Output('cx-slider-container', 'children'),
        Output('cy-slider-container', 'children'),
        Output('cz-slider-container', 'children'),
        Output('dx-slider-container', 'children'),
        Output('dy-slider-container', 'children'),
        Output('dz-slider-container', 'children'),
        #
        Input('drop-down', 'value'),
        prevent_initial_call=True)
    def update_sliders(value):
        if(value == 0):
            return [dcc.Slider(-3, 3, value=0.0, step = 0.01, tooltip={'placement':'bottom', 'always_visible':True}, marks={-3: '-3', 3: '3'}, id='cx-slider')], \
                [dcc.Slider(-3, 3, value=0.0, step = 0.01, tooltip={'placement':'bottom', 'always_visible':True}, marks={-3: '-3', 3: '3'}, id='cy-slider')], \
                [dcc.Slider(-3, 3, value=0.0, step = 0.01, tooltip={'placement':'bottom', 'always_visible':True}, marks={-3: '-3', 3: '3'}, id='cz-slider')], \
                [dcc.Slider(0, 6, value=1.0, step = 0.01, tooltip={'placement':'bottom', 'always_visible':True}, marks={0: '0', 6: '6'}, id='dx-slider')], \
                [dcc.Slider(0, 6, value=1.0, step = 0.01, tooltip={'placement':'bottom', 'always_visible':True}, marks={0: '0', 6: '6'}, id='dy-slider')], \
                [dcc.Slider(0, 6, value=1.0, step = 0.01, tooltip={'placement':'bottom', 'always_visible':True}, marks={0: '0', 6: '6'}, id='dz-slider')]

    @app.callback(
        Output('depth-distribution-slider-container', 'children'),

        Input('time-slider-id', 'value'),
        Input('cx-slider', 'value'),
        Input('cy-slider', 'value'),
        Input('cz-slider', 'value'),
        Input('dx-slider', 'value'),
        Input('dy-slider', 'value'),
        Input('dz-slider', 'value'),
        prevent_initial_call=True)
    def update_depth_distribution_slider(time, cx, cy, cz, dx, dy, dz):
        xmin = cx - dx*0.5
        xmax = cx + dx*0.5
        ymin = cy - dy*0.5
        ymax = cy + dy*0.5
        zmin = cz - dz*0.5
        zmax = cz + dz*0.5
        num_points = data['positions'][0].shape[0]
        max_val = -1
        for i_p in range(num_points):
            if(vf_utils.insideBoundingBox(data['positions'][0][i_p], np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]]))):
                max_val += 1
        return dcc.Slider(min=0, max=max_val, step=1, value=0.0, marks={0: '0', max_val: str(max_val)}, tooltip={'placement':'left', 'always_visible':True}, vertical=True, id='depth-distribution-slider-id')

    @app.callback(
        Output('global-view-id', 'children'),
        Output('local-view-id', 'children' ),
        #
        Input('global-view-id', 'children'),
        Input('local-view-id', 'children'),
        Input('time-range-slider-id', 'value'),
        Input('time-slider-id', 'value'),
        Input('scale-glyphs-slider-id', 'value'),
        Input('magnitude-threshold-slider-id', 'value'),
        Input('data-depth-slider-id', 'value'),
        Input('box-selection-switch', 'value'),
        Input('cx-slider', 'value'),
        Input('cy-slider', 'value'),
        Input('cz-slider', 'value'),
        Input('dx-slider', 'value'),
        Input('dy-slider', 'value'),
        Input('dz-slider', 'value'),
        # Input('global-view-radioitems', 'value'),
        # Input('local-view-radioitems', 'value'),
        prevent_initial_call=True
    )
    def update_global_view( global_view_children, 
                            local_view_children,
                            time_range, 
                            selected_time,
                            scale, 
                            magnitude_threshold,
                            depth_threshold,
                            box_selection,
                            cx, cy, cz,
                            dx, dy, dz): 
                            # global_view_radioitems_value, 
                            # local_view_radioitems_value):
        
        print('Updating glyphs ...')
        start = time.time()

        # gobal domain bounding box
        domain = data['domain'][0]
        
        if(depth_threshold - data['selected_depth'][0] > 1e-4):
             comp_min_max_flag = True
             data['selected_depth'][0] = depth_threshold
        else:
            comp_min_max_flag = False
        # if( global_view_radioitems_value == 0):
        # print('glyph_type:', glyph_type)
        global_polygones = vf_glyphs.buildGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
                                                        selected_time = selected_time, data_depth = depth_threshold, 
                                                        domain = domain, glyph_type=glyph_type, mean_flag= False, compute_min_max_flag= True)

        # elif(global_view_radioitems_value == 1):
        #     global_polygones = vf_glyphs.buildGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
        #                                                     selected_time = selected_time, data_depth = depth_threshold, 
        #                                                     domain= domain, glyph_type='variability', mean_flag= False, compute_min_max_flag= True)

        # if(local_view_radioitems_value == 0):
        if(box_selection):
            domain = np.array([[cx-dx*0.5, cx+dx*0.5], [cy-dy*0.5, cy+dy*0.5], [cz-dz*0.5, cz+dz*0.5]])
        else:
            domain = data['domain'][0]

        local_polygons = vf_glyphs.buildGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
                                                    selected_time= selected_time, data_depth = depth_threshold, 
                                                    domain= domain, glyph_type=glyph_type, mean_flag= False, compute_min_max_flag= False)
        # elif(local_view_radioitems_value == 1):
        #     if(box_selection):
        #         domain = np.array([[cx-dx*0.5, cx+dx*0.5], [cy-dy*0.5, cy+dy*0.5], [cz-dz*0.5, cz+dz*0.5]])
        #     else:
        #         domain = data['domain'][0]
        #         # print('Here domain:', domain)
        #     local_polygons = vf_glyphs.buildGlyphsForDash(data, scale=scale, resolution=10, time_range =time_range, 
        #                                                   selected_time= selected_time, data_depth = depth_threshold, 
        #                                                   domain= domain, glyph_type='variability', mean_flag= False, compute_min_max_flag= False)



        if(box_selection):
            # box = dash_vtk.GeometryRepresentation(
            #     property={"edgeVisibility": True, "edgeColor":(0,0,0), "opacity": 0.5, "representation": 2},
            #     actor={"position": (cx, cy, cz), "scale": (dx, dy, dz)},
            #     children=[dash_vtk.Algorithm(id="box-selection-id", vtkClass="vtkCubeSource")],
            # )
            # global_polygones.append(box)
            # global_polygones.append(domain_box)
            # local_polygons.append(domain_box)

            box_points, box_polygons = vf_glyphs.getBoxPolyData(domain)
            box = dash_vtk.GeometryRepresentation(
            children=[
                dash_vtk.PolyData(
                    id="vtk-polydata",
                    points=np.array(box_points).ravel(),
                    polys=np.array(box_polygons),
                )],
            property={'opacity':1.0, 'color': [0, 0, 0], 'representation':1},
            )
            global_polygones.append(box)

            if(terrain == True):
                nx = int(np.sqrt(data['num_points'][0]))
                ny = int(np.sqrt(data['num_points'][0]))
                if(example_name == 'hurricane_isabel'):
                    global_surface = getsurfaceGeometry(hurricane_isabel_terrain, nx, ny, data['domain'][0])
                else:
                    global_surface = getsurfaceGeometry(data['positions'][0], nx, ny, data['domain'][0])
                global_polygones.append(global_surface)
           
            print('Rendering glyphs ...')
            if(terrain == True):
                if(example_name == 'hurricane_isabel'):
                    local_surface = getsurfaceGeometry(hurricane_isabel_terrain, nx, ny, domain)
                else:
                    local_surface = getsurfaceGeometry(data['positions'][0], nx, ny, domain)
                local_polygons.append(local_surface)
            content = dash_vtk.View(global_polygones, background=[1,1,1]) #, interactorSettings=[{'button': 1, 'action': 'Rotate'}, {'button': 2, 'action': 'Pan'}, {'button': 3, 'action': 'Zoom', 'scrollEnabled': True},  {'button': 1, 'action': 'Pan', 'shift': True}])
            local_content = dash_vtk.View(local_polygons, background=[1,1,1]) #, interactorSettings=[{'button': 1, 'action': 'Rotate'}, {'button': 2, 'action': 'Pan'}, {'button': 3, 'action': 'Zoom', 'scrollEnabled': True}, {'button': 1, 'action': 'Pan', 'shift': True}])
            print(' Done rendering glyphs ...')
        else:
            # global_polygones.append(domain_box)
            # local_polygons.append(domain_box)
            print('Rendering glyphs ...')
            if(terrain == True):
                nx = int(np.sqrt(data['num_points'][0]))
                ny = int(np.sqrt(data['num_points'][0]))
                print('nx =', nx, 'ny =', ny)
                global_surface = getsurfaceGeometry(data['positions'][0], nx, ny, data['domain'][0])
                global_polygones.append(global_surface)
            content = dash_vtk.View(global_polygones, background=[1,1,1])#, triggerRender=0) #, interactorSettings=[{'button': 1, 'action': 'Rotate'}, {'button': 2, 'action': 'Pan'}, {'button': 3, 'action': 'Zoom', 'scrollEnabled': True},  {'button': 1, 'action': 'Pan', 'shift': True}])
            if(terrain == True):
                local_surface = getsurfaceGeometry(data['positions'][0], nx, ny, data['domain'][0])
                local_polygons.append(local_surface)
            local_content = dash_vtk.View(local_polygons, background=[1,1,1]) #, interactorSettings=[{'button': 1, 'action': 'Rotate'}, {'button': 2, 'action': 'Pan'}, {'button': 3, 'action': 'Zoom', 'scrollEnabled': True},  {'button': 1, 'action': 'Pan', 'shift': True}])
            print(' Done rendering glyphs ...')
        # get context id 
        context = ctx.triggered_id
        end_time = time.time()
        print('Time taken:', end_time - start)
        if(context == 'global-view-radioitems' or context == 'time-range-slider-id' or context == 'time-slider-id' or 
        context == 'scale-glyphs-slider-id' or context == 'data-depth-slider-id' or context == 'box-selection-switch' or 
        context == 'cx-slider' or context == 'cy-slider' or context == 'cz-slider' or context == 'dx-slider' or 
        context == 'dy-slider' or context == 'dz-slider' or context == 'global-view-radioitems'):
            return content, local_content
        else:
            return global_view_children, local_view_children


    @app.callback(
        Output('time-depedent-variability-id', 'figure'),
        Output('variability-at-time-t-id', 'figure'),
        #
        Input('time-range-slider-id', 'value'),
        Input('data-depth-slider-id', 'value'),
        Input('time-slider-id', 'value'),
        Input('time-selector-radioitems', 'value'),
        prevent_initial_call=True
    )
    def update_time_dependent_variability( time_range,
                            depth_threshold,
                            selected_time, 
                            time_selector_radioitems_value):
        # fig = make_subplots(rows=2, cols=1)
        fig0 = go.Figure()
        fig1 = go.Figure()
        y_data = []
        for i_t in range(time_range[0], time_range[1]+1):
            yy = data['max_vectors'][0][i_t][:, 0]-data['min_vectors'][0][i_t][:, 0]
            xx = np.linspace(0, data['num_points'][0], data['num_points'][0])
            if(i_t == selected_time):
                fig0.add_trace(go.Scatter(x=xx, y=yy, mode='lines', line_color='red', opacity=1.0 ))
            else:
                fig0.add_trace(go.Scatter(x=xx, y=yy, mode='lines', line_color='blue', opacity=0.05))
            if(time_selector_radioitems_value == 0):
                y_data.append(np.max(yy))
            else:       
                y_data.append(np.sqrt(np.mean(yy**2)))
        fig1.add_trace(go.Scatter(x=np.linspace(time_range[0], time_range[1], len(y_data)), y=y_data, mode='lines'))
        # update axis label
        fig1.update_xaxes(title_text="time")
        if(time_selector_radioitems_value == 0):
            fig1.update_yaxes(title_text="max variability", range=[0, np.max(y_data)])
        else:
            fig1.update_yaxes(title_text="rms variability", range=[0, np.max(y_data)])
        fig1.update_layout( margin=dict(l=1, r=1, t=1, b=1), font=dict(size=fs))
        fig0.update_xaxes(title_text="global point index")
        fig0.update_yaxes(title_text="variability", range = [0, np.max(y_data)])
        fig0.update_layout( margin=dict(l=1, r=1, t=1, b=1), showlegend=False, font=dict(size=fs))
        return fig1, fig0


    @app.callback(
        Output('depth-id-t0', 'figure'),
        #
        Input('box-selection-switch', 'value'),
        Input('cx-slider', 'value'),
        Input('cy-slider', 'value'),
        Input('cz-slider', 'value'),
        Input('dx-slider', 'value'),
        Input('dy-slider', 'value'),
        Input('dz-slider', 'value'),
        Input('time-slider-id', 'value'),
        Input('depth-distribution-slider-id', 'value'),
        prevent_initial_call=True
    )
    def update_depth( box_selection, cx, cy, cz, dx, dy, dz, selected_time, selected_depth_distribution_id):
        z_data = []
        global_point_ids = []
        num_members = data['depths'][0][selected_time].shape[1]
        if(box_selection):
            domain = np.array([[cx-dx*0.5, cx+dx*0.5], [cy-dy*0.5, cy+dy*0.5], [cz-dz*0.5, cz+dz*0.5]])
        else:
            domain = data['domain'][0]
        for i_pt in range(data['num_points'][0]):
            if(vf_utils.insideBoundingBox(data['positions'][0][i_pt], domain)):
                z_data.append(data['depths'][0][selected_time][i_pt])
                global_point_ids.append(i_pt)
        fig = go.Figure( data = [go.Heatmap(z=z_data, colorscale='Cividis', zmin=0.0, zmax=1.0, colorbar=dict(title='Depth'))])
        # update axis label
        fig.add_shape(type="rect", x0=-0.5, y0=selected_depth_distribution_id -0.5, x1=num_members-0.5, y1=selected_depth_distribution_id+0.5, line=dict(color="lightblue", width=2))
        fig.update_xaxes(title_text="ensemble members")
        fig.update_yaxes(title_text="Local point index")
        fig.update_layout(margin=dict(l=1, r=1, t=1, b=1), font=dict(size=fs))
        
        return fig


    @app.callback(
        Output('vectors-distribution-id-ip', 'figure'),
        #
        Input('vectors-distribution-id-ip', 'figure'),
        Input('vector-distribution-radioitems-id', 'value'),
        Input('box-selection-switch', 'value'),
        Input('cx-slider', 'value'),
        Input('cy-slider', 'value'),
        Input('cz-slider', 'value'),
        Input('dx-slider', 'value'),
        Input('dy-slider', 'value'),
        Input('dz-slider', 'value'),
        Input('depth-distribution-slider-id', 'value'),
        Input('time-slider-id', 'value'),
        Input('time-range-slider-id', 'value'),
        prevent_initial_call=True
    )
    def update_vectors_distribution(in_fig, vectors_distribution_flag, box_selection, cx, cy, cz, dx, dy, dz, 
                                    selected_depth_distribution_id, 
                                    selected_time, 
                                    time_range):
        if(box_selection):
            matching_global_point_ids = 0
            local_id = 0
            domain = np.array([[cx-dx*0.5, cx+dx*0.5], [cy-dy*0.5, cy+dy*0.5], [cz-dz*0.5, cz+dz*0.5]])
            for i_pt in range(data['num_points'][0]):
                if(vf_utils.insideBoundingBox(data['positions'][0][i_pt], domain)):
                    if( local_id == selected_depth_distribution_id):
                        matching_global_point_ids = i_pt
                        break
                    local_id += 1

            if(vectors_distribution_flag == 1):
                i_t = selected_time
                fig0 = go.Figure()
                x0 = []
                y0 = []
                z0 = []
                vectors = data['vectors'][0][i_t][matching_global_point_ids]
                num_ensemble_members = data['vectors'][0][selected_time][matching_global_point_ids].shape[0]
                scale_mag = data['max_vectors'][0][selected_time][matching_global_point_ids][0]
                for i_m in range(num_ensemble_members):
                    vec = vf_utils.spherical2Cartesian(vectors[i_m])
                    vec = vec / scale_mag
                    x0.append(0)
                    y0.append(0)
                    z0.append(0)
                    x0.append(vec[0])
                    y0.append(vec[1])
                    z0.append(vec[2])
                    x0.append(None)
                    y0.append(None)
                    z0.append(None)
                fig0.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode='lines', line=dict(color='blue', width=2), name=str(selected_time)))
                fig0.update_layout(scene=dict(aspectmode='cube', xaxis_title='x', yaxis_title='y', zaxis_title='z'), margin=dict(l=1, r=1, t=1, b=1))
                
                return fig0
            else:
                # get vectors
                ii_t = -1
                fig0 = go.Figure()
                fig1 = go.Figure()
                fig0.add_trace(go.Scatter(x=[0, 0, None, -1, 1], y=[-1, 1, None, 0, 0], mode='lines', name='axes', line=dict(color='black')))
                

                for i_t in range(time_range[0], time_range[1]+1):
                    x0 = []
                    y0 = []
                    x1 = []
                    y1 = []
                    ii_t += 1
                    vectors = data['vectors'][0][i_t][matching_global_point_ids]
                    # get median vector
                    median_vector = data['median_vectors'][0][i_t][matching_global_point_ids]
                    scale_mag = data['max_vectors'][0][i_t][matching_global_point_ids][0]
                    median_vec = vf_utils.spherical2Cartesian(median_vector)
                    if(ii_t >0):
                        angle_current_previous_median = np.arctan2(np.linalg.norm(np.cross(median_vec, median_vec_previous)), np.dot(median_vec, median_vec_previous))
                    else:
                        angle_current_previous_median = 0.0
                    angle_current_previous_median = - np.absolute(angle_current_previous_median) + np.pi*0.5
                    median_vec_previous = median_vec
                    # GET NUMBER OF ENSEMBLE MEMBERS
                    num_ensemble_members = data['vectors'][0][selected_time][matching_global_point_ids].shape[0]
                    
                    for i_m in range(num_ensemble_members):
                        vec = vf_utils.spherical2Cartesian(vectors[i_m])
                        angle_vec_median_vec = np.arctan2(np.linalg.norm(np.cross(median_vec, vec)), np.dot(median_vec, vec))
                        tmp= [vectors[i_m][0]*np.cos(angle_vec_median_vec), vectors[i_m][0]*np.sin(angle_vec_median_vec)]
                        tmp = tmp / scale_mag
                        vec_2d = np.zeros(2)    
                        vec_2d[0] = tmp[0]*np.cos(angle_current_previous_median) - tmp[1]*np.sin(angle_current_previous_median)
                        vec_2d[1] = tmp[0]*np.sin(angle_current_previous_median) + tmp[1]*np.cos(angle_current_previous_median)
                        x0 
                        if(i_t == selected_time):
                            x0.append(0)
                            y0.append(0)

                            x0.append(vec_2d[0])
                            y0.append(vec_2d[1])
                            x0.append(None)
                            y0.append(None)
                            # fig0.add_trace(go.Scatter(x=[0, vec_2d[0]], y=[0, vec_2d[1]], mode='lines', line=dict(color='blue')))
                        else:
                            x1.append(0)
                            y1.append(0)
                            x1.append(vec_2d[0])
                            y1.append(vec_2d[1])
                            x1.append(None)
                            y1.append(None)
                            # fig1.add_trace(go.Scatter(x=[0, vec_2d[0]], y=[0, vec_2d[1]], mode='lines', line=dict(color='red')))
                    # scale figure axis
                    # combine figures
                    if(i_t == selected_time):
                        fig0.add_trace(go.Scatter(x=x0, y=y0, mode='lines', line=dict(color='blue'), name=str(i_t)))
                    else:
                        fig1.add_trace(go.Scatter(x=x1, y=y1, mode='lines', line=dict(color='red'), name=str(i_t)))

                fig1.update_traces(opacity=0.05)

                fig0.add_traces(fig1.data)
                # fig0.update_layout(scene=dict(aspectmode='cube', xaxis_title='x', yaxis_title='y', zaxis_title='z'), margin=dict(l=1, r=1, t=1, b=1))
                fig0.update_yaxes(scaleanchor = "x", scaleratio = 1)
                fig0.update_layout(plot_bgcolor='white', margin=dict(l=1, r=1, t=1, b=1), font=dict(size=fs))
                ## set range
                fig0.update_xaxes(range=[-1, 1])
                fig0.update_yaxes(range=[-1, 1])
                # fig0.show()
                return fig0
        else:
            return in_fig

    if __name__ == "__main__":
        # app.run(debug=True)
        app.run(host='0.0.0.0', port=8080, debug=True)
