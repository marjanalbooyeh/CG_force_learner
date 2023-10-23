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


def get_parameters():
    parameters = OrderedDict()

    # project parameters
    parameters["project"] = ["Dimer-overfit-Oct10"]
    parameters["group"] = ["small"]
    parameters["notes"] = ["Learning dimer forces"]
    parameters["tags"] = ["NN"]
    parameters["target_type"] = ["force"]

    # dataset parameters
    parameters["data_path"] = ["/home/marjanalbooyeh/code/CG_force_learner/dimer/force-flow-Oct9/small-overfit-scaled.pkl"] 
    # parameters["data_path"] =["/home/erjank_project/marjan/CG_force_learner/datasets/dimer/limited_pair_orientation/3_pair_orientation_overfit_dimer.pkl"]
    # supported input modes: "append", "stack"
    parameters["inp_mode"] = ["append"]
    # supported augmentations for rel. positions: "r" (center-to-center distance)
    parameters["augment_pos"] = ["r"]
    # supported augmentations for rel. orientations: "a" (relative angle between two orientations)
    parameters["augment_orient"] = ["q1q2"]
    parameters["batch_size"] = [2]
    parameters["shrink"] = [True]
    parameters["overfit"] = [True]

    # model parameters
    # supported model types: "NN", "NNSkipShared", "NNGrow"
    parameters["model_type"] = ["NNSkipShared", "NN"]
    parameters["hidden_dim"] = [32]
    parameters["n_layer"] = [3]
    parameters["act_fn"] = ["ReLU"]
    parameters["dropout"] = [0.001]
    # supported pooling operations (only works when inp_mode="stack"): "mean", "max", "sum"
    parameters["pool"] = ["mean"]
    parameters["batch_norm"] = [False]

    # optimizer parameters
    parameters["optim"] = ["SGD"]
    parameters["lr"] = [0.01]
    parameters["use_scheduler"] = [True]
    parameters["scheduler_type"] = ["StepLR"]
    parameters["decay"] = [0.0001]
    # supported loss types: "mse" (mean squared error), "mae" (means absolute error)
    parameters["loss_type"] = ["mse"]

    # run parameters
    parameters["epochs"] = [100000]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("Dimer-Overfit", root=root)  # Set the signac project name
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
