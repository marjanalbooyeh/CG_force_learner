import itertools
import math


import gsd.hoomd
import hoomd
import matplotlib
import numpy

import h5py
import numpy as np
import numpy as np
import rowan
import os
import pandas as pd
import pickle

import warnings
warnings.filterwarnings('ignore')




def create_sim_obj(q1, q2, rot_freedom=False, log_name=None, log_path=None, T=1.5):
    dimer_positions = [[-0.6, 0, 0], [0.6, 0, 0]]


    gpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=gpu, seed=1)
    sim.create_state_from_gsd(filename='../../lattice_init.gsd')


    rigid = hoomd.md.constrain.Rigid()

    rigid.body['dimer'] = {
        "constituent_types": ['A', 'B'],""
        "positions": dimer_positions,
        "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        "charges": [0.0, 0.0],
        "diameters": [1., 1.]
    }

    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    R1_filter = hoomd.filter.Tags([0])
    _filter= hoomd.filter.SetDifference(rigid_centers_and_free, R1_filter)

    integrator = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=rot_freedom)
    sim.operations.integrator = integrator


    integrator.rigid = rigid

    kT = T
    nvt = hoomd.md.methods.NVT(kT=kT, tau=1., filter=_filter)
    integrator.methods.append(nvt)
#     cap = hoomd.md.methods.DisplacementCapped(filter=_filter, maximum_displacement=1e-1)
#     integrator.methods.append(cap)
    
    
    cell = hoomd.md.nlist.Cell(buffer=0, exclusions=['body'])

    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'A')] = 2.8

    lj.params[('B', 'B')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('B', 'B')] = 2.8

    lj.params[('A', 'B')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'B')] = 2.8

    lj.params[('dimer', ['dimer', 'A', 'B'])] = dict(epsilon=0, sigma=0)
    lj.r_cut[('dimer', ['dimer', 'A', 'B'])] = 0

    integrator.forces.append(lj)

    thermodynamic_quantities = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All())

    sim.operations.computes.append(thermodynamic_quantities)

    sim.state.thermalize_particle_momenta(filter=_filter, kT=kT)
    
    with sim._state.cpu_local_snapshot as data:
        rtag = data.particles.rtag
        idx_0 = rtag[0]
        idx_1 = rtag[1]

        data.particles.orientation[idx_0] = q1

        data.particles.orientation[idx_1] = q2
        

#     sim.run(0)


#     nvt.thermalize_thermostat_dof()
    if log_name:

        # Logging 
        log_quantities = [
                    "kinetic_temperature",
                    "potential_energy",
                    "kinetic_energy",
                    "volume",
                    "pressure",
                    "pressure_tensor",
                ]
        logger = hoomd.logging.Logger(categories=["scalar", "string", "particle"])
        logger.add(sim, quantities=["timestep", "tps"])
        thermo_props = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        sim.operations.computes.append(thermo_props)
        logger.add(thermo_props, quantities=log_quantities)

        for f in integrator.forces:

            logger.add(f, quantities=["energy", "forces", "energies"])

        gsd_writer = hoomd.write.GSD(
            filename=os.path.join(log_path, f"{log_name}_trajectory.gsd"),
            trigger=hoomd.trigger.Periodic(int(1)),
            mode="wb",
            logger=logger,
            dynamic=["momentum"]
            )

        sim.operations.writers.append(gsd_writer)

    
    
    return sim


def create_radius_grid_positions(init_position, init_radius, final_radius, n_circles=10, 
                          circle_slice=1, circle_coverage=2*np.pi, z_init_last=(-1, 1), z_slice=10):



    # make sure number of z slices is odd to include z=0
    if z_slice % 2:
        z_slice = z_slice + 1
        
    # going from -z to z
    z_positions = np.linspace(z_init_last[0], z_init_last[1], z_slice)
    
    # angle between slices of the circle
    dtheta = circle_coverage / circle_slice
    
    grid_positions = []
    for z in z_positions:
        for radius in np.linspace(init_radius, final_radius, n_circles):
            for i in range(circle_slice):
                grid_positions.append((init_position[0] + (radius* np.cos(i*dtheta)), 
                                       init_position[1] +  (radius* np.sin(i*dtheta)),
                                      init_position[2] + z))
    print("positions length: ", len(grid_positions))
    return grid_positions


def rot_z(A, angle):
    return A @ np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

def rot_y(A, angle):
    return A @ np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )

def get_fixed_quaternions():
    A0 = np.eye(3)
    A1 = rot_z(A0, np.pi / 2)
    A2 = rot_y(A0, np.pi/2)
    A3 = rot_y(A1, np.pi/2)
    A4 = rot_z(A0, np.pi/4)

    fixed_quaternions = np.asarray([rowan.from_matrix(A0),
                        rowan.from_matrix(A1),
                        rowan.from_matrix(A2),
                        rowan.from_matrix(A3),
                        rowan.from_matrix(A4)
                        ])
    return fixed_quaternions



def create_random_orientations(n=100, fixed=True):
    grid_orientations= rowan.random.rand(n)
    if fixed:
        grid_orientations = np.concatenate((grid_orientations, get_fixed_quaternions()))
    print("orientations: ", grid_orientations.shape)
    return grid_orientations



def position_constants(x_start=-2, x_finish=0.8):
    x0, y0, z0 = (-3, 0, 0)
    init_radius = np.abs(x0 - x_start)
    final_radius = np.abs(x0 - x_finish)
    return (x0, y0, z0), init_radius, final_radius



def run_batch(orientation_list,grid_positions):
    columns = [
    "position",
    "orientation",
    "net_force",
    "net_torque",
    "energy"]
    positions = []
    orientations = []
    energies = []
    forces = []
    torques = []
    for i, (q1, q2) in enumerate(orientation_list):
        if i % 10 == 0:
            print("*********************************")
            print("orientation: ", (q1, q2))
        sim_obj = create_sim_obj(q1=q1, q2=q2, rot_freedom=False)
        for (x, y, z) in grid_positions:
            with sim_obj.state.cpu_local_snapshot as data:
                idx = data.particles.rtag[1]
                data.particles.position[idx] = (x, y, z)
            sim_obj.run(0)
            energies.append(sim_obj.operations.integrator.forces[0].energy)
            with sim_obj.state.cpu_local_snapshot as data:
                com_idx = data.particles.rtag[[0, 1]]
                positions.append(data.particles.position[com_idx])
                orientations.append(data.particles.orientation[com_idx])
                forces.append(data.particles.net_force[com_idx])
                torques.append(data.particles.net_torque[com_idx])
            try:
                sim_obj.run(1)
            except:
                print("exception")
    new_traj_df = pd.DataFrame(columns=columns)
    new_traj_df["position"] = positions
    new_traj_df["orientation"] = orientations
    new_traj_df["net_force"] = forces
    new_traj_df["net_torque"] = torques
    new_traj_df["energy"] = energies
    print(len(positions))
   
    new_traj_df.to_pickle("raw_data.pkl")

