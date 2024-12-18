"""Main simulation module for physics-based string simulation using Hooke's Law."""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from Mass import SimulationObject
from Physics_Engine import PhysicsEngine
from Visualization import SimulationVisualizer

t = 0

# Lets you define the beginning and end points for the string
# I have this set to be along the x-axis, scale doesn't really matter because it's dimensionless rn.
def default_start_point() -> np.ndarray:
    """Default start point."""
    return np.array([-1.0, 0.0, 0.0])
def default_end_point() -> np.ndarray:
    """Default end point."""
    return np.array([1.0, 0.0, 0.0])
def default_force_applied() -> np.ndarray:
    """Default applied force."""
    global t
    current_state = np.array([0.0, 0.0, 1.0*np.sin(t)])
    return current_state


class StringSimulation:
    """Class to handle the string simulation setup and execution."""

    def __init__(self, params):
        """
        Initializes string simulation with given parameters.

        Args:
            params: SimulationParameters object containing configuration
        """
        self.params = params

        # Create string elements
        self.masses, self.equilibrium_length = self.create_string()
        self.physics = self.setup_physics()
        self.visualizer = self.setup_visualization()

    def create_string(self):
        """Creates string of masses connected by springs."""
        masses = []

        # Calculate total length and equilibrium length between masses
        total_length = np.linalg.norm(self.params.end_point - self.params.start_point)

        # The equilibrium length which can be manipulated by the user, however the default will be based
        # off a calculation of the end points
        equilibrium_length = total_length / self.params.num_segments
        #print("Equilibium: ",  equilibrium_length)

        # Calculate positions of masses along the string
        for i in range(self.params.num_segments+1):
            # Calculate position as linear interpolation between start and end points
            t = i / self.params.num_segments
            position = (1 - t) * self.params.start_point + t * self.params.end_point

            # Reverses the object ID assignment for the sake of lining up better with the slider.
            reversed_id = (self.params.num_segments) - i

            # Create mass object with initial conditions
            mass = SimulationObject(
                x=position[0],
                y=position[1],
                z=position[2],
                vx=0.0, vy=0.0, vz=0.0,
                ax=0.0, ay=0.0, az=0.0,
                mass=self.params.mass,
                obj_id=reversed_id,  # Use the reversed ID
                pinned=(i == 0 or i == self.params.num_segments)  # Pin the ends still determined by position
            )
            masses.append(mass)

        # Sorts the masses by their object ID to maintain consistent ordering since I flipped them
        # This ensures that the physics engine will receive them in the correct order
        masses.sort(key=lambda m: m.obj_id)

        return masses, equilibrium_length

    def setup_physics(self) -> PhysicsEngine:
        """Initialize physics engine with simulation parameters."""
        # If we are in string_params mode, use computed_k. Otherwise use the regular spring_constant.
        if self.params.parameter_mode == 'string_params':
            k_value = self.params.computed_k
        else:
            k_value = self.params.spring_constant

        return PhysicsEngine(
            objects=self.masses,
            k=k_value,
            equilibrium_length=self.params.equilibrium_length,
            time=0.0,
            dt=self.params.dt,
            start_point=self.params.start_point,
            end_point=self.params.end_point
        )

    def setup_visualization(self) -> SimulationVisualizer:
        """Initialize visualization system."""
        return SimulationVisualizer(
            physics_model=self.physics,
            objects=self.masses,
            dark_mode=self.params.dark_mode,
            integration_method=self.params.integration_method  # Pass the integration method
        )

    def run(self):
        """
        Run the simulation.

        Returns:
            bool: True if the simulation should restart, False otherwise
        """

        def on_key(event):
            """Handle keyboard input."""
            if event.key == ' ':  # Spacebar applies vertical force to middle mass
                middle_index = len(self.masses) // 2
                self.physics.apply_force(middle_index, self.params.applied_force)
            elif event.key == 'p':  # 'p' toggles pause
                self.visualizer.toggle_pause(None)

        # Connect keyboard handler
        self.visualizer.fig.canvas.mpl_connect('key_press_event', on_key)

        # Start animation and return restart flag
        return self.visualizer.animate()