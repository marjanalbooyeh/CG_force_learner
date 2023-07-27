"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import sys
import os
import itertools

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from data_sampler import create_radius_grid_positions, position_constants, run_batch
import numpy as np

class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="short",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )
        parser.add_argument(
            "--cpus",
            default=5,
            help="Specify cpu-cores per task."
        )



# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")


@directives(executable="python -u")
# @directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("generating grid positions and orientations")
        print("----------------------")
        init_position, init_radius, final_radius = position_constants(x_start=job.sp.x_start, x_finish=job.sp.x_finish)
        grid_positions = create_radius_grid_positions(init_position=init_position,
                                                      init_radius = init_radius, final_radius = final_radius,
                                                     n_circles=job.sp.n_circles, circle_slice=job.sp.circle_slice, 
                                                     circle_coverage = job.sp.circle_coverage, z_init_last=job.sp.z_init_last,
                                                      z_slice=job.sp.z_slice)
        
        
        orientation_pairs = np.load("../../orientation_pairs.npy")
        
        job_orientations = orientation_pairs[job.sp.j_id * job.sp.orient_slice : (job.sp.j_id + 1) * job.sp.orient_slice]
        print("orientations from: {} | {} ".format(job.sp.j_id * job.sp.orient_slice,  (job.sp.j_id + 1) * job.sp.orient_slice))
        job.doc["job_orientations"] = job_orientations
    
        
        print("Sampling...")
        print("----------------------")
        run_batch(job_orientations, grid_positions)
        job.doc["done"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")


def submit_project():
    MyProject().run()


if __name__ == "__main__":
    MyProject().main()