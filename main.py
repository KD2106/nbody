import numpy as np
import matplotlib.pyplot as plt

"""
Inspired by Philip Mosz https://github.com/pmocz
"""

def getAcc(pos, mass, G, softening):
    """Calculates acceleration of each particle in an N-body system under Gravitational force."""
    
    # Get coordinate-wise components of position
    x, y, z = pos[:,0:1], pos[:,1:2], pos[:,2:3]

    # Get pairwise separations between particles
    dx, dy, dz = x.T -x, y.T -y, z.T -z #.T is the transpose method

    # Get inverse cube of separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2) 
    inv_r3[inv_r3 > 0 ] = inv_r3[inv_r3 > 0]**(-1.5)

    # Calculate components of acceleration in each direction for all elements
    ax = G* dx* inv_r3 @ mass
    ay = G* dy* inv_r3 @ mass
    az = G* dz* inv_r3 @ mass

    # Pack the components together
    a = np.hstack((ax,ay,az))

    return a

def main():
    # Simulation parameters
    N = 100
    G = 6.67e-11
    softening = 0.1
    t = 0
    tEnd = 10
    dt = 0.1
    
    #np.random.seed(9)

    mass = 20000*np.ones((N,1))/N
    pos = np.random.randn(N,3)
    vel = np.random.randn(N,3)

    # Number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    
    # Set up grid
    fig = plt.figure()

    # Make the figure 3d
    ax = fig.add_subplot(projection='3d')
#    ax.set_aspect('equal', 'box')
#    ax.set_xticks([-2,-1,0,1,2])
#    ax.set_yticks([-2,-1,0,1,2])

    acc = getAcc(pos, mass, G, softening)

    # Simulation loop
    for i in range(Nt):
        #Use Leaf-frog method (Kick-Drift-Kick scheme) to update position

        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc( pos, mass, G, softening )

        # (1/2) kick
        vel += acc * dt/2.0

        # update time
        t += dt
        
        #Plot in real time
        if t < tEnd:
            plt.sca(ax)
            plt.cla() 

            """ xx = pos_save[:,0,max(i-50,0):i+1]
            yy = pos_save[:,1,max(i-50,0):i+1]
            plt.scatter(xx,yy,s=1,color=[.7,.7,1])
            ax.set(xlim=(-5,5), ylim=(-5,5))
            ax.set_aspect('equal', 'box')
            ax.set_xticks([-5,-4,-3-2,-1,0,1,3,4,5])
            ax.set_yticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])"""

            ax.set(xlim=(-5,5), ylim=(-5,5), zlim=(-5,5))
            ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=10, color='blue')
            plt.pause(0.001)

    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    main()





"""for i in range(100):
    x.append(i)
    y.append(i**2)
    plt.scatter(x,y,s=10)
    plt.pause(0.001)
    
plt.show()"""