import numpy as np
from mppi_torch.mppi_torch.mppi import MPPIPlanner
from typing import Callable, Optional
import io
import math
import os
import yaml
from yaml.loader import SafeLoader
import torch

torch.set_printoptions(precision=2, sci_mode=False)


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)


class MPPI_Wrapper(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, prior: Optional[Callable] = None):
        self.cfg = cfg
        self.objective = objective
        self.nx = 4  # number of states
        self.nu = 2  # number of control variables

        if prior:
            self.prior = lambda state, t: prior.compute_command(self.sim)
        else:
            self.prior = None

        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
        )

        # Note: place_holder variable to pass to mppi, so it doesn't complain, while the real state is actually the
        # isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))

    def update_objective(self, objective):
        self.objective = objective

    def dynamics(self, old_state, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the
        # state is already saved in the simulator itself, so we ignore it. Note: t is an unused step dependent
        # dynamics variable

        t = 0.2
        v = u[:, 0]
        w = u[:, 1]
        psi = old_state[:, 2]
        
        trans = torch.stack([
            t * v * torch.cos(psi),
            t * v * torch.sin(psi),
            t * w,
            t * v
        ], dim=-1)
        state = torch.add(old_state, trans)
        return state, u
        

    def running_cost(self, state):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the
        # state is already saved and accessible in the simulator itself, so we ignore it and pass a handle to the
        # simulator.
        return self.objective.compute_cost(state)

    def compute_action(self, q, qdot, obst=None, obst_tensor=None):
        self.state_place_holder = torch.tensor(q * self.cfg.mppi.num_samples).view(self.cfg.mppi.num_samples, -1)
        actions, states = self.mppi.command(self.state_place_holder)
        
        actions = actions.cpu()
        # loop over the actions and forward propagate the dynamics to get the trajectory
        old_state = self.state_place_holder[0, :].unsqueeze(0)
        traj = []
        for i in range(actions.size(0)):
            element = actions[i, :]
            new_state, _ = self.dynamics(old_state, element.unsqueeze(0))
            traj.append(old_state)
            old_state = new_state

        traj = torch.cat(traj, dim=0)
        return actions[0], traj, states

    def command(self):
        return torch_to_bytes(self.mppi.command(self.state_place_holder))
