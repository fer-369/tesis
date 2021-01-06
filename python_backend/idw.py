import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def main():
   

   
    # Setup: Generate data...
    n = 10
    nx, ny = 100, 100
    #x, y, z = map(np.random.random, [n, n, n])
    y = np.array([-78.4771628, -78.4768582, -78.4768690, -78.4761548 , -78.4763388])
    x = np.array([0.0183425 ,0.0197352, 0.0197663, 0.0181757, 0.0181647])
    z = np.array([79.23, 86.77, 87.43, 85.50, 72.20])

    

    #print(x)
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    # Calculate IDW
    grid1 = simple_idw(x,y,z,xi,yi)
    print(type(grid1))
    grid1 = grid1.reshape((ny, nx))

    # Calculate scipy's RBF
    grid2 = scipy_idw(x,y,z,xi,yi)
    #print(grid2)
    grid2 = grid2.reshape((ny, nx))

    grid3 = linear_rbf(x,y,z,xi,yi)
    #print grid3.shape
    grid3 = grid3.reshape((ny, nx))


    # Comparisons...
    plot(x,y,z,grid1)
    plt.title('Homemade IDW')

    plot(x,y,z,grid2)
    plt.title("Scipy's Rbf with function=linear")

    plot(x,y,z,grid3)
    plt.title('Homemade linear Rbf')

    plt.show()

    

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi


def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)


def plot(x,y,z,grid):
    #plt.figure()
    #plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    #plt.hold(True)
    #plt.scatter(x,y,c=z)
    #plt.colorbar()

    df = pd.read_csv('python_backend/coords.csv')

    map = plt.imread('python_backend/map.png')
    BBox = (-78.48439,   -78.47253, 0.01479, 0.02180)
    fig, ax = plt.subplots(figsize = (8,7))

    print(BBox)
    ax.scatter(y, x, zorder=1, alpha= 1, c=z, s=30)
    
    ax.set_title('Pululahua')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    
    ax.imshow(map,  extent = BBox, aspect= 'equal')
    ax.imshow(grid, alpha= .5, extent=(x.min(), x.max(), y.min(), y.max()))
    


if __name__ == '__main__':
    main()