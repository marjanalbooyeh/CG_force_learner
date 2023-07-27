#!/usr/bin/env python
"""Initialize the project's data space.
Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.
"""

import logging
from collections import OrderedDict
from itertools import product

import signac
import numpy as np

def get_parameters():
    parameters = OrderedDict()

    # dataset parameters
    parameters["x_start"] = [-2]
    parameters["x_finish"] = [0.8]
    parameters["n_circles"] = [10]
    parameters["circle_slice"] = [1]
    parameters["circle_coverage"] = [2*np.pi]
    parameters["z_init_last"] = [(-1, 1)]
    parameters["z_slice"] = [10]
    
    
    parameters["j_id"] = list(range(10))
    parameters["orient_slice"] = [154]


    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("Dimer-Sampling", root=root)  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)

    project.write_statepoints()
    return project


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
