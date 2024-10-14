import numpy as np
from numba import njit

@njit
def getBoundingBox(vectors):
    """    
    The function getBoundingBox(...) calculates the bounding box of the vector field
     data in the array vectors
    
    INPUT
     vectors: 2D array of dimensions [number_of_vectors x dim] containing vector field data
    
    OUTPUT
     bounding_box: 2D array of dimensions [dim x 2] containing the minimum and maximum
                   coordinates of the bounding box
    """
    dimensions = np.shape(vectors)
    if(len(dimensions) < 1 and len(dimensions) > 2):
        raise NameError('Incorrect array dimensions. Expected a 2D numpy array where the number of' 
              ' rows represents the number of vectors and the number of columns represents', 
              ' the dimension')
    elif(len(dimensions) == 1):
        bbox = np.zeros(2)
        bbox[0] = np.min(vectors)
        bbox[1] = np.max(vectors)
    elif(len(dimensions) == 2):
        number_of_vectors = dimensions[0]
        dim = dimensions[1]
        bbox = np.zeros([dim, 2])
        for i in range(dim):
            bbox[i][0] = np.min(vectors[:,i])
            bbox[i][1] = np.max(vectors[:,i])
    return bbox


def insideBoundingBox(point, bbox):
    """    
     The function insideBoundingBox(...) checks if a point is inside the bounding box
    
     INPUT
     point: 1D array of dimensions [dim] containing the coordinates of the point
     bbox: 2D array of dimensions [dim x 2] containing the minimum and maximum
           coordinates of the bounding box
    
     OUTPUT
     inside: boolean indicating if the point is inside the bounding box
    """
    if(len(point.shape)<1 or len(point.shape)>2):
        raise NameError('Incorrect array dimensions. Expected a 1D numpy array')
    
    if(len(point.shape) == 1):
        for i in range(bbox.shape[0]):
            if(point[i] < bbox[i][0] or point[i] > bbox[i][1]):
                return False

    if(len(point.shape) == 2):
        for i in range(point.shape[0]):
            for j in range(bbox.shape[0]):
                if(point[i][j] < bbox[j][0] or point[i][j] > bbox[j][1]):
                    return False
    return True


def cartesian2Spherical(vectors):
    """
     This function converts the cartesian coordinates of the vectors
     in the array vectors to spherical coordinates
    
     INPUT:
     vectors: numpy 2D array with each row corresponding to a vector
              cartesian coordinates
    
     OUTPUT:
     spherical_coordinates: numpy 2D array with each row corresponding to a vector
                        spherical coordinates
    """
    if(len(vectors.shape) == 1):
        spherical_vectors = np.zeros(3)
        spherical_vectors[0] = np.sqrt(vectors[0]*vectors[0] + vectors[1]*vectors[1] + vectors[2]*vectors[2])
        # spherical_vectors[1] = np.arctan(np.sqrt(vectors[0]*vectors[0] + vectors[1]*vectors[1])/(vectors[2] + np.finfo(np.float32).eps))
        # spherical_vectors[2] = np.arctan(vectors[1]/(vectors[0] + np.finfo(np.float32).eps))
        spherical_vectors[2] = np.arctan2(vectors[1], vectors[0] + np.finfo(np.float32).eps)
        spherical_vectors[1] = np.arctan2(np.sqrt(vectors[0]*vectors[0] + vectors[1]*vectors[1]), vectors[2] + np.finfo(np.float32).eps)
        return spherical_vectors
    spherical_vectors = np.zeros([vectors.shape[0], 3])
    for i in range(vectors.shape[0]):
        spherical_vectors[i][0] = np.sqrt(vectors[i][0]*vectors[i][0] + vectors[i][1]*vectors[i][1] + vectors[i][2]*vectors[i][2])
        spherical_vectors[i][1] = np.arctan2(np.sqrt(vectors[i][0]*vectors[i][0] + vectors[i][1]*vectors[i][1]), (vectors[i][2] + np.finfo(np.float32).eps))
        spherical_vectors[i][2] = np.arctan2(vectors[i][1],(vectors[i][0] + np.finfo(np.float32).eps))

    return spherical_vectors


def spherical2Cartesian(vectors):
    """
     This function converts the spherical coordinates of the vectors
     in the array vectors to cartesian coordinates
    
     INPUT:
     vectors: numpy 2D array with each row corresponding to a vector
              spherical coordinates
    
     OUTPUT:
     cartesian_coordinates: numpy 2D array with each row corresponding to a vector
                        cartesian coordinates
    """
    if (len(vectors.shape) == 1):
        cartesian_vectors = np.zeros(3)
        cartesian_vectors[0] = vectors[0]*np.sin(vectors[1])*np.cos(vectors[2])
        cartesian_vectors[1] = vectors[0]*np.sin(vectors[1])*np.sin(vectors[2])
        cartesian_vectors[2] = vectors[0]*np.cos(vectors[1])

        return cartesian_vectors
    
    if(len(vectors.shape) == 2):
        cartesian_vectors = np.zeros([vectors.shape[0], 3])
        for i in range(vectors.shape[0]):
            cartesian_vectors[i][0] = vectors[i][0]*np.sin(vectors[i][1])*np.cos(vectors[i][2])
            cartesian_vectors[i][1] = vectors[i][0]*np.sin(vectors[i][1])*np.sin(vectors[i][2])
            cartesian_vectors[i][2] = vectors[i][0]*np.cos(vectors[i][1])

        return cartesian_vectors
    
    raise NameError('Incorrect array dimensions. Expected a 1D or 2D numpy array')


def getSphericalCoordinates(vectors):
    """
     This function computes the spherical coordinates
     for each vector in vetors.
     
     INPUT:
     vector: numpy 2D array with each row corresponding to a vector
             cartesian coordinates
    
     OUTPUT:
     coordinates: numpy 2D array with each corresponding to a vector 
                  spherical coordinates
    """

    dimensions = np.array(np.shape(vectors))
    eps = np.absolute(np.finfo(np.float32).eps)
    if(len(dimensions) ==1):
        theta = 1.0e+10
        phi = 1.0e+10
        x = vectors[0]
        y = vectors[1]
        z = vectors[2]

        # get vector magnitude/radius
        r = np.sqrt(x*x + y*y + z*z)

        # polar angle
        # print('x=', x, 'y=', y, 'z=', z, 'eps=', eps)
        if(np.absolute(z) <= eps):
            theta = np.pi * 0.5
        elif(z > 0.0):
            theta = np.arctan(np.sqrt(x*x + y*y) / (z+eps))
        elif(z < 0.0):
            theta = np.pi + np.arctan(np.sqrt(x*x + y*y) / (z+eps))
        elif(np.absolute(z) <= eps and np.abs(x*y) > eps):
            theta = np.pi * 0.5
        #elif(z == 0 and y == 0 and x ==0 ):
        #    print('x=', x, 'y=', y, 'z=', z)
        #    raise NameError("theta z=y=x=0")
        else:
            theta = 0.0
        # print('theta =', theta)
        # get azimuthal angle
        if(np.absolute(x) < eps and np.absolute(y) < eps):
            phi = 0.0
        elif( np.absolute(x) < eps and y > 0.0):
            phi = 0.5*np.pi
        elif( np.absolute(x) < eps and y < 0.0):
            phi = -0.5*np.pi
        elif(x > 0.0):
            phi = np.arctan( y/(x + eps))
        elif(x < 0.0 and y >= 0.0):
            phi = np.arctan( y/(x + eps)) + np.pi
        elif(x < 0.0 and y < 0.0):
            phi = np.arctan( y/(x + eps)) - np.pi
        elif(np.absolute(x) <= eps and y > 0.0):
            phi = 0.5*np.pi
        elif(np.absolute(x) <= eps and y < 0.0):
            phi = -0.5*np.pi
        #elif(z == 0 and y == 0 and x ==0 ):
        #    raise NameError(' phi z=y=x=0')
        else:
            phi = 0.0
        # print('phi =', phi)

        

        coordinates = np.array([r, theta, phi])
        xx = r*np.sin(theta)*np.cos(phi)
        yy = r*np.sin(theta)*np.sin(phi)
        zz = r*np.cos(theta)
        if(np.abs(xx - x) > 1.0e-5 or np.abs(yy - y) > 1.0e-5 or np.abs(zz - z) > 1.0e-5):
            print('x=', x, 'y=', y, 'z=', z)
            print('xx=', xx, 'yy=', yy, 'zz=', zz)
            print('r=', r, 'theta=', theta, 'phi=', phi)
            raise NameError('Spherical coordinates conversion error')
        
    # 2D array
    elif(len(dimensions) == 2):
        number_of_vectors = dimensions[0] #len(vectors)
        
        coordinates = np.zeros([number_of_vectors, 3])
        for i in range(number_of_vectors):
            x = vectors[i][0]
            y = vectors[i][1]
            z = vectors[i][2]

            # get vector magnitude/radius
            r = np.sqrt(x*x + y*y + z*z)

            # get azimuthal angle
            if(np.absolute(z) < eps):
                theta = np.pi * 0.5
            elif(z > 0.0):
                theta = np.arctan(np.sqrt(x*x + y*y) / (z+eps))
            elif(z < 0.0):
                theta = np.pi + np.arctan(np.sqrt(x*x + y*y) / (z+eps))
            elif(np.absolute(z)<= eps and np.abs(x*y) > eps):
                theta = np.pi * 0.5
            #elif(z == 0 and y == 0 and x ==0 ):
            #    print('x=', x, 'y=', y, 'z=', z)
            #    raise NameError("theta z=y=x=0")
            else:
                theta = 0.0

            # get theta the polar angle 
            if(x > 0.0):
                phi = np.arctan( y/(x + eps))
            elif(x < 0.0 and y >= 0.0):
                phi = np.arctan( y/(x + eps)) + np.pi
            elif(x < 0.0 and y < 0.0):
                phi = np.arctan( y/(x + eps)) - np.pi
            elif(np.absolute(x) <= eps and y > 0.0):
                phi = 0.5*np.pi
            elif( np.absolute(x) <= eps and y < 0.0):
                phi = -0.5*np.pi
            #elif(z == 0 and y == 0 and x ==0 ):
            #    raise NameError(' phi z=y=x=0')
            else:
                phi = 0.0

            coordinates[i][0] = r
            coordinates[i][1] = theta
            coordinates[i][2] = phi
    # print('coordinates =', coordinates)
    return coordinates

