import numpy as np
from numba import jit
from simulation import Simulation, record_time
import backend
import RD_field
import RD1_with_mRNA


@jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, types, radius, alpha=10, r_e=1.01,
                        u_bb=5, u_rb=1, u_yb=1, u_rr=20, u_ry=12, u_yy=30, u_sb=1, u_rs=1, u_ys=1, u_ss=30,
                        u_repulsion=10000):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]
        adhesion_values = np.reshape(np.array([u_bb, u_rb, u_yb, u_sb,
                                               u_yb, u_rr, u_ry, u_rs,
                                               u_rb, u_ry, u_yy, u_ys,
                                               u_sb, u_rs, u_ys, u_ss]), (4, 4))
        # get cell positions
        cell_1_loc = locations[cell_1] - center
        cell_2_loc = locations[cell_2] - center

        # get new location position
        vec = cell_2_loc - cell_1_loc
        dist2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
        # based on the distance apply force differently
        if dist2 == 0:
            edge_forces[index][0] = 0
            edge_forces[index][1] = 0
        else:
            dist = dist2 ** (1/2)
            if 0 < dist2 < (2 * radius) ** 2:
                edge_forces[index][0] = -1 * u_repulsion * (vec / dist)
                edge_forces[index][1] = 1 * u_repulsion * (vec / dist)
            else:
                # get the cell type
                cell_1_type = types[cell_1]
                cell_2_type = types[cell_2]
                u = adhesion_values[cell_1_type, cell_2_type]
                # get value prior to applying type specific adhesion const
                value = (dist - r_e) * (vec / dist)
                edge_forces[index][0] = u * value
                edge_forces[index][1] = -1 * u * value
    return edge_forces


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces, grav=1):
    for index in range(number_cells):
        new_loc = locations[index] - center
        new_loc_sum = new_loc[0] ** 2 + new_loc[1] ** 2 + new_loc[2] ** 2
        net_forces[index] = -grav * (new_loc / well_rad) * new_loc_sum ** (1/2)
    return net_forces


@jit(nopython=True)
def convert_edge_forces(number_edges, edges, edge_forces, neighbor_forces):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        neighbor_forces[cell_1] += edge_forces[index][0]
        neighbor_forces[cell_2] += edge_forces[index][1]

    return neighbor_forces

def seed_cells(num_agents, center, radius):
    theta = np.random.uniform(0, 2 * np.pi, num_agents).reshape(num_agents, 1)
    phi = np.random.uniform(0, np.pi/2, num_agents).reshape(num_agents, 1)
    r = np.random.uniform(0, radius, num_agents).reshape(num_agents, 1)

    # convert to cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta) + center[0]
    y = r * np.sin(phi) * np.sin(theta) + center[1]
    z = r * np.cos(phi) + center[2]
    locations = np.hstack((x, y, z))
    return locations

def calculate_rate(combined_percent, end_step):
    return 1 - np.power(1-combined_percent, 1/end_step)

def calculate_transition_rate(combined_percent, t):
    transition_rate = 1 - (1-combined_percent) ** t
    return transition_rate

def activator_inhibitor_equal(activator, inhibitor, buffer):
    return (np.abs(activator - inhibitor)) < buffer

class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, model_params):
        # initialize the Simulation object
        Simulation.__init__(self)

        self.default_parameters = {
            "cuda": False,
            "PACE": False,

            # Well Dimensions
            "size": [1, 1, 1],
            "dimension": 3,
            "well_rad": 30,
            "output_values": True,
            "output_images": False,
            "image_quality": 1000,
            "video_quality": 1000,
            "fps": 5,
            "cell_rad": 0.5,
            "velocity": 0.3,
            "initial_seed_ratio": 0.5,
            "cell_interaction_rad": 3.2,
            "replication_type": 'Default',

            # Adhesion and other Biophysical Parameters
            "u_bb": 1,
            "u_rb": 1,
            "u_yb": 1,
            "u_rr": 30,
            "u_yy": 40,
            "u_ry": 1,
            "u_repulsion": 10000,
            "alpha": 10,
            "gravity": .2,
        }
        self.model_parameters(self.default_parameters)
        self.model_parameters(model_params)
        self.model_params = model_params

        # aba/dox/cho ratio
        self.M_color = np.array([255, 255, 0], dtype=int) #yellow
        self.E_color = np.array([255, 50, 50], dtype=int) #red
        self.ME_color = np.array([50, 50, 255], dtype=int)
        self.S_color = np.array([50, 200, 50], dtype=int)
        self.diff_color = np.array([[255, 255, 255]], dtype=int)


        self.initial_seed_rad = self.well_rad * self.initial_seed_ratio
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad
        self.center = np.array([self.size[0] / 2, self.size[1] / 2, 0])
        #self.dt = self.dx2 * self.dy2 / (2 * self.inducer_D * (self.dx2 + self.dy2))
        self.extracellular_prot_degradation = self.protein_degradation
        self.dt = 1
        self.dx2 = 3


    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """

        # initial seeded cell populations (M=mesoderm, E=endoderm, ME=mesendoderm)
        num_M = 0
        num_E = 0
        num_S = self.initial_S
        num_ME = int(self.num_to_start) - num_S

        # add agents to the simulation
        self.add_agents(num_M, agent_type="M")
        self.add_agents(num_E, agent_type="E")
        self.add_agents(num_S, agent_type="S")
        self.add_agents(num_ME, agent_type="ME")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "division_set", "div_thresh")
        #morphogen concentrations within each cell.
        self.indicate_arrays('modulator_conc', 'activator_conc', 'inhibitor_conc',
                             'mod_prot', 'act_prot', 'inh_prot',
                             'diff_timer', 'diff_set', 'diff_threshold')

        # generate random locations for cells
        self.locations = seed_cells(self.number_agents, self.center, self.initial_seed_rad)
        self.radii = self.agent_array(initial=lambda: self.cell_rad)

        # Define cell types, S is source cells, 2 is mesoderm, 1 is definitive endoderm, 0 is mesendoderm.
        self.cell_type = self.agent_array(dtype=int, initial={"S": lambda: 3,
                                                              "M": lambda: 2,
                                                              "E": lambda: 1,
                                                              "ME": lambda: 0})

        self.colors = self.agent_array(dtype=int, vector=3, initial={"S": lambda: self.S_color,
                                                                     "M": lambda: self.M_color,
                                                                     "E": lambda: self.E_color,
                                                                     "ME": lambda: self.ME_color})

        # setting division times (in seconds):
        self.div_thresh = self.agent_array(initial={"S": lambda: 96,
                                                    "M": lambda: 16,
                                                    "E": lambda: 16,
                                                    "ME": lambda: 16})
        self.division_set = self.agent_array(initial={"M": lambda: np.random.uniform(0, 16, 1),
                                                      "E": lambda: np.random.uniform(0, 16, 1),
                                                      "S": lambda: np.random.uniform(0, 0, 1),
                                                      "ME": lambda: np.random.uniform(0, 16, 1)})

        #indicate and create graphs for identifying neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        #matrices for concentrations of morphogens inside cells
        self.modulator_conc = self.agent_array(initial=0)
        self.activator_conc = self.agent_array(initial=0)
        self.inhibitor_conc = self.agent_array(initial=0)
        self.act_prot = self.agent_array(initial=0)
        self.inh_prot = self.agent_array(initial=0)
        self.mod_prot = self.agent_array(initial=0)

        #Differentiation threshold:
        self.diff_timer = self.agent_array(initial=12)
        self.diff_set = self.agent_array(initial=0)
        self.diff_threshold = np.random.uniform(self.thresh - self.noise, self.thresh + self.noise, self.number_agents)
        self.diff_threshold_M = np.random.uniform(0, self.buffer, self.number_agents)

        #Specify and initiate RD Fields.
        self.fields = {"activator": 10, "inhibitor": 10, "modulator": 10} #not sure what this does...
        self.activator_field = np.zeros((self.size[0], self.size[1], self.size[2]))
        self.inhibitor_field = np.zeros((self.size[0], self.size[1], self.size[2]))
        self.modulator_field = np.zeros((self.size[0], self.size[1], self.size[2]))

        #self.modulator_field = define_field.sphere(self.modulator_field, origin, 5, 10)

        # self.modulator_field[0:10,:] = 2
        # self.min_field = np.amin(self.modulator_field)
        # self.max_field = 10 # np.amax(self.modulator_field)

        # colors = [(0, 0, 1, c) for c in np.linspace(0, 1, 100)]
        # self.cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

        # save parameters to text file
        self.save_params(self.model_params)

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform subset force and RD calculations
        for i in range(self.sub_ts):
            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()

            # Update morphogens.
            self.update_morphogens()

        self.cell_fate(1)
        self.reproduce(1)
        # add/remove agents from the simulation
        self.update_populations()
        print(f'Num_M: {len(np.argwhere(self.cell_type == 2))}, Num_E: {len(np.argwhere(self.cell_type == 1))}')
        print(f'Num_S: {len(np.argwhere(self.cell_type == 3))}')

        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.step_values()
        self.step_image()
        #self.create_video()

    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)
        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add indices to the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

            # if locations array
            if name == "locations":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # move distance of radius in random direction
                    vec = self.radii[i] * self.random_vector()
                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # reset division time
            if name == "division_set":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # set division counter to zero
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "division_threshold":
                # go through the number of cells added
                for i in range(num_added):
                    # get daughter index
                    daughter = self.number_agents + i

                    # set division threshold based on cell type
                    self.__dict__[name][daughter] = 1

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    def update_morphogens(self):
        active = self.current_step > self.inducer_timer
        self.activator_conc, self.inhibitor_conc, self.modulator_conc, \
        self.act_prot, self.inh_prot, self.mod_prot = RD1_with_mRNA.dudt(self.activator_conc,
                                                                         self.inhibitor_conc,
                                                                         self.modulator_conc,
                                                                         self.act_prot,
                                                                         self.inh_prot,
                                                                         self.mod_prot,
                                                                         active,
                                                                         self.dt,
                                                                         self.cell_type,
                                                                         self.mRNA_production,
                                                                         self.mRNA_degradation,
                                                                         self.protein_translation,
                                                                         self.protein_degradation,
                                                                         self.hill_km,
                                                                         self.hill_n,
                                                                         self.hill_interaction_terms)

        if active:
            self.activator_field, self.act_prot = RD_field.diffuse_morphogen_to_agents(self.locations,
                                                                                             self.activator_field,
                                                                                             self.act_prot,
                                                                                             self.diffusivity[0],
                                                                                             self.dimension,
                                                                                             self.dt)
            self.inhibitor_field, self.inh_prot = RD_field.diffuse_morphogen_to_agents(self.locations,
                                                                                             self.inhibitor_field,
                                                                                             self.inh_prot,
                                                                                             self.diffusivity[1],
                                                                                             self.dimension,
                                                                                             self.dt)

            self.modulator_field, self.mod_prot = RD_field.diffuse_morphogen_to_agents(self.locations,
                                                                                       self.modulator_field,
                                                                                       self.mod_prot,
                                                                                       self.diffusivity[2],
                                                                                       self.dimension,
                                                                                       self.dt)
            self.modulator_field = RD_field.diffuse(self.modulator_field, self.diffusivity[2], self.dt, self.dx2) - self.modulator_field * self.extracellular_prot_degradation[2]
            self.activator_field = RD_field.diffuse(self.activator_field, self.diffusivity[0], self.dt, self.dx2) - self.activator_field * self.extracellular_prot_degradation[0]
            self.inhibitor_field = RD_field.diffuse(self.inhibitor_field, self.diffusivity[1], self.dt, self.dx2) - self.inhibitor_field * self.extracellular_prot_degradation[1]



    def cell_fate(self, ts):
        self.diff_set += ts * (self.act_prot > self.diff_threshold)
        differentiating = (self.diff_set > self.diff_timer) * (self.cell_type == 0)
        for i in range(len(differentiating)):
            if differentiating[i]:
                if activator_inhibitor_equal(self.act_prot[i], self.inh_prot[i], self.buffer):
                    self.cell_type[i] = 1
                else:
                    self.cell_type[i] = 2

        self.update_colors()

    def update_colors(self):
        ref = np.array([self.ME_color, self.E_color, self.M_color, self.S_color])
        self.colors = ref[self.cell_type]

    @record_time
    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, self.dimension))
        neighbor_forces = np.zeros((self.number_agents, self.dimension))
        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, self.center, self.cell_type,
                                          self.cell_rad, u_bb=self.u_bb, u_rb=self.u_rb, u_rr=self.u_rr, u_yb=self.u_yb,
                                          u_ry=self.u_ry, u_yy=self.u_yy, alpha=0, u_repulsion=self.u_repulsion)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        noise_vector = np.ones((self.number_agents, self.dimension)) * self.alpha * (2 * np.random.rand(self.number_agents, self.dimension) - 1)
        neighbor_forces = neighbor_forces + noise_vector
        if self.gravity > 0:
            net_forces = np.zeros((self.number_agents, self.dimension))
            gravity_forces = get_gravity_forces(self.number_agents, self.locations, self.center,
                                                self.well_rad, net_forces, grav=self.gravity)
            neighbor_forces = neighbor_forces + gravity_forces
        for i in range(self.number_agents):
            vec = neighbor_forces[i]
            sum = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
            if sum != 0:
                neighbor_forces[i] = neighbor_forces[i] / (sum ** (1/2))
            else:
                neighbor_forces[i] = 0
        # update locations based on forces
        self.locations += 2 * self.velocity * self.cell_rad * neighbor_forces
        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

    #Unused..
    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        if self.current_step > self.growth_timer:
            self.division_set += ts
            if self.replication_type == 'Default':
                for index in range(self.number_agents):
                    if self.division_set[index] > self.div_thresh[index]:
                        self.mark_to_hatch(index)
            if self.replication_type == 'None':
                return

    @classmethod
    def simulation_mode_0(cls, name, output_dir, model_params):
        """ Creates a new brand new simulation and runs it through
            all defined steps.
        """
        # make simulation instance, update name, and add paths
        sim = cls(model_params)
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()

    def save_params(self, params):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.
        """

        # iterate through the keys adding each instance variable
        with open(self.main_path + "parameters.txt", "w") as parameters:
            for key in list(params.keys()):
                parameters.write(f"{key}: {params[key]}\n")
        parameters.close()

    @classmethod
    def start_sweep(cls, output_dir, model_params, name):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the name/mode for the simulation
        output_dir = backend.check_output_dir(output_dir)

        name = backend.check_existing(name, output_dir, new_simulation=True)
        cls.simulation_mode_0(name, output_dir, model_params)


def parameter_sweep_abm(par,
                        directory,
                        transcriptions,
                        mrna_degradations,
                        translations,
                        protein_degradations,
                        hill_km,
                        hill_n,
                        hill_interaction_parameters,
                        diffusivity,
                        noise,
                        buffer,
                        threshold,
                        inducer_timer,
                        growth_timer,
                        final_ts=60,
                        sub_ts=10):
    """
    Run model with specified parameters
    :param par: parameter number
    :param directory: directory of simulation folder
    :param transcriptions: a vector of transcription rates of [mRNA_A, mRNA_I, mRNA_M]
    :param mrna_degradations: a vector of degradation rates of [mRNA_A, mRNA_I, mRNA_M]
    :param translations: a vector of translation rates of [A, I, M]
    :param protein_degradations: a vector of degradation rates of [A, I, M]
    :param hill_km: vector of apparent dissociation constants (K_d) of [A, I, M]
    :param hill_n: vector of hill coefficients of [A, I, M]
    :param hill_interaction_parameters: vector of hill interaction parameters in the order: [a1, b1, a2, a3]
    :param diffusivity: a vector of the diffusivity for the [A, I, M]
    :param noise: noise range of cell state transition
    :param buffer: range of M differentiation
    :param threshold: threshold of E differentiation
    :param inducer_timer: time at which inducer begins
    :param growth_timer: time at which accelerated growth begins
    :param final_ts: final timestep of simulation
    :param sub_ts: number of "sub timesteps"
    :return:
    """

    model_params = {
        # ABM state parameters
        "num_to_start": 500,
        "initial_S": 400,
        "end_step": final_ts,
        "sub_ts": sub_ts,
        "inducer_timer": inducer_timer,
        "growth_timer": growth_timer,

        # morphogen system parameters
        # ode parameters
        "mRNA_production": transcriptions,
        "mRNA_degradation": mrna_degradations,
        "protein_translation": translations,
        "protein_degradation": protein_degradations,
        "hill_km": hill_km,
        "hill_n": hill_n,
        "hill_interaction_parameters": hill_interaction_parameters,
        # pde parameters
        "diffusivity": diffusivity,

        # cell state transition parameters
        "thresh": threshold,
        "buffer": buffer,
        "noise": noise
    }
    name = f'033023_RD1_60%green_40%blue_{par}'
    sim = TestSimulation(model_params)
    sim.start_sweep(directory + '/outputs', model_params, name)
    return par, sim.image_quality, sim.image_quality, 3, final_ts/sim.sub_ts

if __name__ == "__main__":
    mRNA_transcription = [1, 0.1, 15]
    mRNA_degradation = [0.5, 0.5, 0.5]
    protein_translation = [0.125, 0.125, 0.125]
    protein_degradation = [0.167, 0.167, 0.167]
    diffusion = [0.01, 0.2, 0.4]
    hill_km = [1, 1, 5]
    hill_interaction_parameters = [4, 10, 4, 4]
    hill_n = [4, 2, 4]
    directory = "/Users/andrew/PycharmProjects/AIM_RD_D+all_Modulator_Source_cells/"

    a = parameter_sweep_abm(12,
                            directory,
                            mRNA_transcription,
                            mRNA_degradation,
                            protein_translation,
                            protein_degradation,
                            diffusion,
                            0.5,
                            1,
                            2.5,
                            inducer_timer=96,
                            growth_timer=128,
                            final_ts=200,
                            sub_ts=10)
    # a = parameter_sweep_abm(0, "/Users/andrew/PycharmProjects/AIM_RD_D+all_Modulator_Source_cells/",
    #                         [0.01, 0.4, 0.4], 0.5, 1, 2.5, '3D', final_ts=72, sub_ts=10)
    # diffA = [0.01, 0.1, 0.2, 0.4]
    # diffI = [0.01, 0.1, 0.2, 0.4]
    # diffM = [0.01, 0.1, 0.2, 0.4]
    # for i in diffA:
    #     for j in diffI:
    #         for k in diffM:
    #             a = parameter_sweep_abm(0, "/Users/andrew/PycharmProjects/AIM_RD_D+all_Modulator_Source_cells/",
    #                                     [i, j, k], 0.5, 1, 2.5, '3D', final_ts=72, sub_ts=10)

    # noise = [0, 0.1, 0.25, 0.5, 1, 2]
    # buffer = [0, 0.1, 0.25, 0.5, 1]
    # for i in noise:
    #     for j in buffer:
    #         a = parameter_sweep_abm(0, "/Users/andrew/PycharmProjects/AIM_RD_D+all_Modulator_Source_cells/", [0.01, 0.2, 0.4],
    #                                 i, j, 1.9, 10, '3D', final_ts=60, sub_ts=10)
    print(a)

