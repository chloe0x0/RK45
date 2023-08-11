import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import numpy as np 
from rk45 import rk45

if __name__ == "__main__":
    def lorenz(t,y):
        ''' The chaotic lorenz 1963 attractor
        '''
        sigma = 10
        beta = 8/3
        rho = 28

        y_prime = [sigma*(y[1] - y[0]), y[0]*(rho - y[2]) - y[1], (y[0] * y[1]) - (beta * y[2])]
        return np.array(y_prime)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))

    ax.view_init(30, 0)

    # Initial Conditions
    num_ivps = 10
    ivps = np.random.randint(size = (num_ivps,3), low = -5, high = 10)
    trajectories = [None]*num_ivps
    for idx,ivp in enumerate(ivps):
        trajectory = rk45(lorenz,0,ivp,25)
        trajectories[idx] = trajectory
 
    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, num_ivps))

    # set up lines and points
    lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        i = (2*i) % trajectory.shape[1]

        for line, pt, xi in zip(lines, pts, trajectories):
            x, y, z = xi[0,:i], xi[1,:i], xi[2,:i]
            line.set_data(x, y)
            line.set_3d_properties(z)

            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()

        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=150, interval=30, blit=True)
    plt.show()