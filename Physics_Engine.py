import numpy as np
from typing import List, Optional, Dict
from Mass import SimulationObject
from Integrator import INTEGRATORS
from data_handling import SimulationDataRecorder

class PhysicsEngine:
    """Physics engine for string simulation using Hooke's Law."""

    def __init__(self, objects: List[SimulationObject], k: float, equilibrium_length: float, time: float,
                 dt: float, start_point: np.ndarray, end_point: np.ndarray):
        """
        Initialize the physics engine with simulation parameters.

        Args:
            objects: List of masses to simulate, represented by SimulationObject instances.
            k: Spring constant governing the stiffness of the springs.
            equilibrium_length: Natural (unstretched) length of the springs.
            time: Initial simulation time.
            dt: Simulation timestep for numerical integration.
        """
        # Store the core simulation parameters
        print("Physics engine initialized with r0: ", equilibrium_length)
        self.objects = objects  # List of masses (each a SimulationObject instance)
        print(len(objects))
        self.k = k  # Spring constant
        self.equilibrium_length = equilibrium_length  # Natural spring length
        self.start_point = start_point
        self.end_point = end_point
        self.tension = self.calculate_tension()
        self.time = time  # Current simulation time
        self.dt = dt  # Timestep for simulation
        self.simulation_started = False  # Flag to track if simulation has started
        self.start_time = None  # Will store the actual start time when simulation begins

        # Initialize tracking of external forces
        # Create dictionaries to store force-related information for each object
        self.external_forces = {obj.obj_id: np.zeros(3) for obj in objects}
        self.force_form = {obj.obj_id: np.array([0, 0, 0]) for obj in objects}
        self.force_end_times = {obj.obj_id: 0.0 for obj in objects}
        self.continuous_forces = {obj.obj_id: False for obj in objects}

        # Initialize the data recorder
        self.data_recorder = SimulationDataRecorder(len(objects))


    def apply_force(self, obj_id: int, force: np.ndarray, duration: float = None):
        """
        Applies an external force to a specific mass for a given duration.

        Args:
            obj_id: The ID of the mass to apply the force to.
            force: The force vector (e.g., [fx, fy, fz]).
            duration: Duration in seconds for the force to be applied. If None, the force is continuous.
        """
        if not self.objects[obj_id].pinned:  # Skip pinned objects
            self.external_forces[obj_id] = force
            if duration is None:
                self.continuous_forces[obj_id] = True
                self.force_end_times[obj_id] = float('inf')
            else:
                self.continuous_forces[obj_id] = False
                self.force_end_times[obj_id] = self.time + duration

    def get_forces_on_object(self, obj_index: int) -> tuple:
        """
        Calculate all forces acting on a specific object.

        Args:
            obj_index: Index of the object to analyze

        Returns:
            tuple: (external_force, spring_forces, total_force)
        """
        obj = self.objects[obj_index]
        external_force = self.external_forces[obj.obj_id]
        spring_forces = np.zeros(3)

        # Calculate spring forces from neighbors
        if obj_index > 0:
            spring_forces += self.compute_spring_force(obj.position, self.objects[obj_index - 1].position)
        if obj_index < len(self.objects) - 1:
            spring_forces += self.compute_spring_force(obj.position, self.objects[obj_index + 1].position)

        total_force = external_force + spring_forces
        return external_force, spring_forces, total_force

    def check_and_clear_expired_forces(self):
        """Check for and clear any expired forces."""
        for obj_id in self.external_forces:
            if not self.continuous_forces[obj_id] and self.time >= self.force_end_times[obj_id]:
                self.clear_force(obj_id)

    def clear_force(self, obj_id: int):
        """Clear all forces applied to a specific object."""
        self.external_forces[obj_id] = np.zeros(3)
        self.force_end_times[obj_id] = 0.0
        self.continuous_forces[obj_id] = False

    def compute_spring_force(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute the spring force between two connected masses using Hooke's Law.

        Args:
            pos1: Position of the first mass.
            pos2: Position of the second mass.

        Returns:
            The force vector acting on the first mass due to the spring.

        Compute the spring force between two connected masses using Hooke's Law.
        F = -k(|x2-x1| - L0)(x2-x1)/|x2-x1| where L0 is equilibrium length
        """
        separation = pos2 - pos1
        current_length = np.linalg.norm(separation)

        if current_length < 1e-10:  # Avoid division by zero
            return np.zeros(3)

        unit_vector = separation / current_length
        # How much the spring is stretched/compressed from equilibrium
        displacement = current_length - self.equilibrium_length
        force_magnitude = self.k * displacement

        # Used to debug info and verify force calculation
        # if hasattr(self, 'debug_counter'):
        #     self.debug_counter += 1
        #     if self.debug_counter % 1000 == 0:  # Print every 1000th calculation
        #         print(f"Spring force calculation:")
        #         print(f"Current length: {current_length:.6f}")
        #         print(f"Equilibrium length: {self.equilibrium_length:.6f}")
        #         print(f"Displacement from equilibrium: {displacement:.6f}")
        #         print(f"Force magnitude: {force_magnitude:.2f}")
        # else:
        #     self.debug_counter = 0

        return force_magnitude * unit_vector

    def compute_acceleration(self, obj_index: int) -> np.ndarray:
        """
        Compute the net acceleration of a mass due to spring and external forces.

        Args:
            obj_index: Index of the mass in the simulation.

        Returns:
            Acceleration vector for the mass.
        """
        obj = self.objects[obj_index]
        if obj.pinned:
            return np.zeros(3)

        total_force = np.zeros(3)

        if obj_index > 0:
            total_force += self.compute_spring_force(obj.position, self.objects[obj_index - 1].position)
        if obj_index < len(self.objects) - 1:
            total_force += self.compute_spring_force(obj.position, self.objects[obj_index + 1].position)

        if self.time <= self.force_end_times[obj.obj_id] or self.continuous_forces[obj.obj_id]:
            total_force += self.external_forces[obj.obj_id]

        return total_force / obj.mass

    def calculate_tension(self):
        """
        Calculates the tension on the string based on the equilibrium length, spring constant, and custom equilibrium length.

        Returns:
            float: The calculated tension on the string.
        """
        # Calculate total length based on start and end positions
        total_length = np.linalg.norm(self.end_point - self.start_point)

        # Determine the equilibrium length
        if self.equilibrium_length is not None:
            equilibrium_length = self.equilibrium_length
        else:
            equilibrium_length = total_length / (len(self.objects) - 1)
            print(equilibrium_length)

        # Calculate displacement from equilibrium length
        displacement = total_length - (equilibrium_length * (len(self.objects) - 1))

        # Apply Hooke's law to calculate tension
        tension = self.k * displacement

        return tension

    def start_simulation(self):
        """Start the simulation timer if not already started."""
        if not self.simulation_started:
            self.simulation_started = True
            self.start_time = self.time
            print("Simulation started at time:", self.time)

    def step(self, integration_method: str = 'leapfrog') -> None:
        """Advance the simulation by one timestep."""
        if self.simulation_started:
            self.check_and_clear_expired_forces()

            positions = np.array([obj.position for obj in self.objects])
            velocities = np.array([obj.velocity for obj in self.objects])
            fixed_masses = [obj.pinned for obj in self.objects]
            accelerations = np.array([self.compute_acceleration(i) for i in range(len(self.objects))])

            if integration_method in ['leapfrog', 'euler_cromer', 'rk2', 'rk4']:
                def get_accelerations(pos):
                    original_positions = positions.copy()
                    for i, obj in enumerate(self.objects):
                        obj.position = pos[i]
                    acceleration = np.array([self.compute_acceleration(i) for i in range(len(self.objects))])
                    for i, obj in enumerate(self.objects):
                        obj.position = original_positions[i]
                    return acceleration

                new_positions, new_velocities = INTEGRATORS[integration_method](
                    positions, velocities, accelerations, fixed_masses, self.dt, get_accelerations)
            else:
                new_positions, new_velocities = INTEGRATORS[integration_method](
                    positions, velocities, accelerations, fixed_masses, self.dt)

            for i, obj in enumerate(self.objects):
                if not obj.pinned:
                    obj.position = new_positions[i]
                    obj.velocity = new_velocities[i]
                    obj.acceleration = self.compute_acceleration(i)

            self.time += self.dt
