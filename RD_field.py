import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

from numba import njit, stencil


def diffuse_2d(density, diff_coefficient, dt, dx):
    x, y = density.shape
    new_density = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            # finite difference method
            diff = diff_coefficient * (
                (density[(i + 1) % x][j] + density[i - 1][j] - 2 * density[i][j]) / dx**2 +
                (density[i][(j + 1) % y] + density[i][j - 1] - 2 * density[i][j]) / dx**2
            )
            new_density[i][j] = density[i][j] + diff * dt
    # Refelctive Boundary conditions:
    # x-boundaries
    new_density[0,:] = new_density[1,:]
    new_density[-1,:] = new_density[-2,:]
    # y-boundaries
    new_density[:,0] = new_density[:,1,]
    new_density[:,-1] = new_density[:,-2]
    return new_density


def diffuse(density, diff_coefficient, dt, dx):
    '''
    Diffusion with Neumann Reflective Boundary Conditions
    :param density:
    :param diff_coefficient:
    :param dt:
    :param dx:
    :return:
    '''
    x, y, z = density.shape
    new_density = np.zeros((x, y, z))

    for i in range(x):
        for j in range(y):
            for k in range(z):
                # finite difference method
                diff = diff_coefficient * (
                    (density[(i + 1) % x][j][k] + density[i - 1][j][k] - 2 * density[i][j][k]) / dx**2 +
                    (density[i][(j + 1) % y][k] + density[i][j - 1][k] - 2 * density[i][j][k]) / dx**2 +
                    (density[i][j][(k + 1) % z] + density[i][j][k - 1] - 2 * density[i][j][k]) / dx**2
                )

                new_density[i][j][k] = density[i][j][k] + diff * dt
    # Reflective Boundary conditions:
    # x-boundaries
    new_density[0,:,:] = new_density[1,:,:]
    new_density[-1,:,:] = new_density[-2,:,:]
    # y-boundaries
    new_density[:,0,:] = new_density[:,1,:]
    new_density[:,-1,:] = new_density[:,-2,:]
    # z-boundaries
    new_density[:,:,0] = new_density[:,:,1]
    new_density[:,:,-1] = new_density[:,:,-2]
    return new_density


def diffuse_morphogen_to_agent(agent_position, morphogen_gradient, agent_morphogen_concentration, diff, ts):
    """
    Agent diffuses a morphogen into a morphogen gradient following Fick's law.

    Parameters
    ----------
    agent_position : numpy array
        The position of the agent.
    morphogen_gradient : numpy array
        The morphogen gradient represented as a 2D grid.
    agent_morphogen_concentration : float
        The current morphogen concentration in the agent.
    diff : float
        The diffusivity of the morphogen.
    ts : float
        The time step for the simulation.

    Returns
    -------
    numpy array, float
        The updated morphogen gradient after the agent has diffused the morphogen and the updated morphogen
        concentration in the agent.
    """
    ROW, COL, HEI = morphogen_gradient.shape

    # Interpolate the morphogen concentration at the agent's position
    x, y, z = agent_position
    x0, x1 = int(x), int(x + 1)
    y0, y1 = int(y), int(y + 1)
    z0, z1 = int(z), int(z + 1)
    morphogen = (x - x0) * (y - y0) * (z - z0) * morphogen_gradient[x0 % ROW, y0 % COL, z0 % HEI] + \
                (x1 - x) * (y - y0) * (z - z0) * morphogen_gradient[x1 % ROW, y0 % COL, z0 % HEI] + \
                (x - x0) * (y1 - y) * (z - z0) * morphogen_gradient[x0 % ROW, y1 % COL, z0 % HEI] + \
                (x1 - x) * (y1 - y) * (z - z0) * morphogen_gradient[x1 % ROW, y1 % COL, z0 % HEI] + \
                (x - x0) * (y - y0) * (z1 - z) * morphogen_gradient[x0 % ROW, y0 % COL, z1 % HEI] + \
                (x1 - x) * (y - y0) * (z1 - z) * morphogen_gradient[x1 % ROW, y0 % COL, z1 % HEI] + \
                (x - x0) * (y1 - y) * (z1 - z) * morphogen_gradient[x0 % ROW, y1 % COL, z1 % HEI] + \
                (x1 - x) * (y1 - y) * (z1 - z) * morphogen_gradient[x1 % ROW, y1 % COL, z1 % HEI]
        # Calculate the morphogen flux at the four neighbors of the agent's position
    neighbors = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    flux = np.zeros((8,))
    for i, (x, y, z) in enumerate(neighbors):
        x = x % ROW
        y = y % COL
        z = z % HEI
        flux[i] = diff * (morphogen - agent_morphogen_concentration) / 8
        # Update the morphogen gradient field based on the morphogen flux
        morphogen_gradient[x, y, z] -= flux[i] * ts

    # Update the morphogen concentration in the agent
    agent_morphogen_concentration += np.sum(flux) * ts

    return morphogen_gradient, agent_morphogen_concentration


def diffuse_morphogen_to_agent_2d(agent_position, morphogen_gradient, agent_morphogen_concentration, diff, ts):
    """
    Agent diffuses a morphogen into a morphogen gradient following Fick's law.

    Parameters
    ----------
    agent_position : numpy array
        The position of the agent.
    morphogen_gradient : numpy array
        The morphogen gradient represented as a 2D grid.
    agent_morphogen_concentration : float
        The current morphogen concentration in the agent.
    diff : float
        The diffusivity of the morphogen.
    ts : float
        The time step for the simulation.

    Returns
    -------
    numpy array, float
        The updated morphogen gradient after the agent has diffused the morphogen and the updated morphogen
        concentration in the agent.
    """
    ROW, COL, HEI = morphogen_gradient.shape

    # Interpolate the morphogen concentration at the agent's position
    x, y = agent_position
    x0, x1 = int(x), int(x + 1)
    y0, y1 = int(y), int(y + 1)
    morphogen = (x - x0) * (y - y0) * morphogen_gradient[x0 % ROW, y0 % COL] + \
                (x1 - x) * (y - y0) * morphogen_gradient[x1 % ROW, y0 % COL] + \
                (x - x0) * (y1 - y) * morphogen_gradient[x0 % ROW, y1 % COL] + \
                (x1 - x) * (y1 - y) * morphogen_gradient[x1 % ROW, y1 % COL]
        # Calculate the morphogen flux at the four neighbors of the agent's position
    neighbors = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    flux = np.zeros((4,))
    for i, (x, y) in enumerate(neighbors):
        x = x % ROW
        y = y % COL
        flux[i] = diff * (morphogen - agent_morphogen_concentration) / 4
        # Update the morphogen gradient field based on the morphogen flux
        morphogen_gradient[x, y] -= flux[i] * ts

    # Update the morphogen concentration in the agent
    agent_morphogen_concentration += np.sum(flux) * ts

    return morphogen_gradient, agent_morphogen_concentration


def diffuse_morphogen_to_agents(agent_positions, morphogen_gradient, agent_morphogen_concentrations, diff, type, ts):
    """
    Agents diffuse a morphogen into a morphogen gradient following Fick's law.

    Parameters
    ----------
    agent_positions : numpy array
        A matrix of agent positions.
    morphogen_gradient : numpy array
        The morphogen gradient represented as a 2D/3D grid.
    agent_morphogen_concentrations : numpy array
        The current morphogen concentrations in the agents.
    diff : float
        The diffusivity of the morphogen.
    ts : float
        The time step for the simulation.

    Returns
    -------
    numpy array, numpy array
        The updated morphogen gradient after the agents have diffused the morphogen
        and the updated morphogen concentrations in the agents.
    """
    num_agents = agent_positions.shape[0]

    for i in range(num_agents):
        agent_position = agent_positions[i, :]
        agent_morphogen_concentration = agent_morphogen_concentrations[i]
        if type == 3:
            morphogen_gradient, agent_morphogen_concentration = \
                diffuse_morphogen_to_agent(agent_position, morphogen_gradient, agent_morphogen_concentration, diff, ts)
        else:
            morphogen_gradient, agent_morphogen_concentration = \
                diffuse_morphogen_to_agent_2d(agent_position, morphogen_gradient, agent_morphogen_concentration, diff, ts)
        agent_morphogen_concentrations[i] = agent_morphogen_concentration

    return morphogen_gradient, agent_morphogen_concentrations


def plot_field_2d(field, path, name, current_step, max_field, min_field, field_num):
    """ Creates an image of the specified field.
    """
    # only continue if outputting images
    field_types = ["activator", "inhibitor", "modulator"]
    plt.figure()
    plt.imshow(field, interpolation='None', vmin=min_field, vmax=max_field)
    plt.colorbar()
    file_name = f"{name}_{field_types[field_num]}_field_{current_step}.png"
    plt.savefig(path + file_name)
    plt.close()

def plot_field(field, path, name, current_step, field_min, field_max, cmap, field_num):
    """ Creates an image of the specified field.
    """
    # only continue if outputting images
    field_types = ["activator", "inhibitor", "modulator"]
    x, y, z = np.indices(field.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=np.flip(field,0)[y, x, z], marker='s', vmin=field_min, vmax=field_max, cmap=cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    file_name = f"{name}_{field_types[field_num]}_field3d_{current_step}.png"
    plt.savefig(path + file_name)
    plt.close()