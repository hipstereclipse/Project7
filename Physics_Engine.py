import numpy as np
from typing import List, Optional, Dict
from Mass import SimulationObject
from Integrator import INTEGRATORS

class PhysicsEngine:
    """Physics engine for string simulation using Hooke's Law."""

    def __init__(self, objects: List[SimulationObject], k: float, equilibrium_length: float, time: float,
                 dt: float):
        """
        Initialize the physics engine with simulation parameters.

        Args:
            objects: List of masses to simulate, represented by SimulationObject instances.
            k: Spring constant governing the stiffness of the springs.
            equilibrium_length: Natural (unstretched) length of the springs.
            time: Initial simulation time.
            dt: Simulation timestep for numerical integration.
        """
        self.objects = objects  # List of masses (each a SimulationObject instance)
        self.k = k  # Spring constant
        self.equilibrium_length = equilibrium_length  # Natural spring length
        self.time = time  # Current simulation time
        self.dt = dt  # Timestep for simulation

        # Track external forces applied to objects
        self.external_forces = {obj.obj_id: np.zeros(3) for obj in objects}  # Map of object ID to force vector
        self.force_form = {obj.obj_id: np.array([0,0,0]) for obj in objects} # Used to force the form of the string
        self.force_end_times = {obj.obj_id: 0.0 for obj in objects}  # When forces should end
        self.continuous_forces = {obj.obj_id: False for obj in objects}  # Whether forces are continuous

        # Add data recorder
        from data_handling import SimulationDataRecorder
        self.data_recorder = SimulationDataRecorder(objects)

    def apply_force(self, obj_id: int, force: np.ndarray, duration: Optional[float] = None):
        """
        Applies an external force to a specific mass for a given duration.

        Args:
            obj_id: The ID of the mass to apply the force to.
            force: The force vector (e.g., [fx, fy, fz]).
            duration: Duration in seconds for the force to be applied. If None, the force is continuous.
        """
        if not self.objects[obj_id].pinned:  # Skips pinned objects
            self.external_forces[obj_id] = force  # Updates force vector
            if duration is None:
                # Continuous force
                self.continuous_forces[obj_id] = True
                self.force_end_times[obj_id] = float('inf')  # Infinite duration
            else:
                # Time-limited force
                self.continuous_forces[obj_id] = False
                self.force_end_times[obj_id] = self.time + duration  # Set end time

    def check_and_clear_expired_forces(self):
        """
        Check for expired forces and clear them if their duration has elapsed.
        """
        for obj_id in self.external_forces:
            if not self.continuous_forces[obj_id] and self.time >= self.force_end_times[obj_id]:
                self.clear_force(obj_id)

    def clear_force(self, obj_id: int):
        """
        Clear all forces applied to a specific object.

        Args:
            obj_id: ID of the object to clear forces for.
        """
        self.external_forces[obj_id] = np.zeros(3)  # Reset force vector
        self.force_end_times[obj_id] = 0.0  # Reset end time
        self.continuous_forces[obj_id] = False  # Mark force as inactive

    def compute_spring_force(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute the spring force between two connected masses using Hooke's Law.

        Args:
            pos1: Position of the first mass.
            pos2: Position of the second mass.

        Returns:
            The force vector acting on the first mass due to the spring.
        """
        separation = pos2 - pos1  # Vector from mass 1 to mass 2
        current_length = np.linalg.norm(separation)  # Distance between masses
        if current_length < 1e-10:
            return np.zeros(3)  # Avoid division by zero for overlapping masses

        unit_vector = separation / current_length  # Direction of the spring force
        force_magnitude = self.k * (current_length - self.equilibrium_length)  # Hooke's Law
        return force_magnitude * unit_vector

    def compute_acceleration(self, obj_index: int) -> np.ndarray:
        """
        Compute the net acceleration of a mass due to spring and external forces.

        Args:
            obj_index: Index of the mass in the simulation.

        Returns:
            Acceleration vector for the mass.
        """
        obj = self.objects[obj_index]  # Get the mass object
        if obj.pinned:
            return np.zeros(3)  # Pinned objects don't accelerate

        total_force = np.zeros(3)  # Accumulate forces acting on the object

        # Spring force from the left neighbor
        if obj_index > 0:
            total_force += self.compute_spring_force(obj.position, self.objects[obj_index - 1].position)

        # Spring force from the right neighbor
        if obj_index < len(self.objects) - 1:
            total_force += self.compute_spring_force(obj.position, self.objects[obj_index + 1].position)

        # Add external forces if active
        if self.time <= self.force_end_times[obj.obj_id] or self.continuous_forces[obj.obj_id]:
            total_force += self.external_forces[obj.obj_id]

        return total_force / obj.mass  # Acceleration = Force / Mass

    def step(self, integration_method: str = 'leapfrog') -> None:
        """
        Advance the simulation by one timestep using the specified integration method.

        Args:
            integration_method: Name of the integration method (e.g., 'leapfrog', 'rk2').
        """
        # Remove any forces whose duration has expired
        self.check_and_clear_expired_forces()

        # Gather current simulation data
        positions = np.array([obj.position for obj in self.objects])  # Current positions
        velocities = np.array([obj.velocity for obj in self.objects])  # Current velocities
        fixed_masses = [obj.pinned for obj in self.objects]  # Pinned status of masses
        accelerations = np.array([self.compute_acceleration(i) for i in range(len(self.objects))])  # Accelerations

        # Prepare for integration
        if integration_method in ['leapfrog', 'euler_croner', 'euler', 'rk2', 'rk4']:
            # A function to recompute accelerations during integration
            def get_accelerations(pos):
                original_positions = positions.copy()
                for i, obj in enumerate(self.objects):
                    obj.position = pos[i]
                acceleration = np.array([self.compute_acceleration(i) for i in range(len(self.objects))])
                for i, obj in enumerate(self.objects):
                    obj.position = original_positions[i]
                return acceleration

            # Perform integration with custom acceleration function
            new_positions, new_velocities = INTEGRATORS[integration_method](
                positions, velocities, accelerations, fixed_masses, self.dt, get_accelerations)
        else:
            # Perform integration without custom acceleration function
            new_positions, new_velocities = INTEGRATORS[integration_method](
                positions, velocities, accelerations, fixed_masses, self.dt)

        # Update the state of each object
        for i, obj in enumerate(self.objects):
            if not obj.pinned:
                obj.position = new_positions[i]
                obj.velocity = new_velocities[i]
                obj.acceleration = self.compute_acceleration(i)

        # Advance simulation time
        self.time += self.dt

        # Record data after updating positions
        self.data_recorder.record_step(self.time, self.objects)