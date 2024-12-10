"""
Data handling module for physics-based string simulation.

This module provides functionality for:
1. Recording simulation state data over time
2. Saving simulation data to CSV files
3. Loading simulation data from CSV files
4. Extracting specific object data from recorded history

The module contains two main components:
- SimulationDataRecorder class for managing simulation history
- Utility functions for direct file I/O operations
"""
import time
import csv
import os
from typing import List, Optional, Tuple
from Mass import SimulationObject
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from Mass import SimulationObject


class SimulationDataRecorder:
    """
    Records and manages simulation data over time.

    This class maintains the history of object states throughout the simulation,
    including positions, velocities, and accelerations. It also provides
    functionality to save this data to CSV files and manage the recording process.

    Attributes:
        objects (List[SimulationObject]): List of simulation objects being tracked
        time_history (List[float]): History of simulation timestamps
        position_history (List[List[float]]): History of object positions
        velocity_history (List[List[float]]): History of object velocities
        acceleration_history (List[List[float]]): History of object accelerations
        frame_count (int): Number of recorded frames
        simulation_time (float): Current simulation time
    """

    def __init__(self, objects: List[SimulationObject]):
        """
        Initialize the data recorder with a list of simulation objects.

        Args:
            objects: List of SimulationObject instances to record data for
        """
        # Store reference to simulation objects
        self.objects = objects

        # Initialize history lists for storing time series data
        self.time_history = []          # Stores timestamps
        self.position_history = []      # Stores position vectors
        self.velocity_history = []      # Stores velocity vectors
        self.acceleration_history = []   # Stores acceleration vectors

        # Initialize counters
        self.frame_count = 0            # Tracks number of recorded frames
        self.simulation_time = 0.0      # Tracks current simulation time

        self.objects = objects
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.frame_count = 0
        self.simulation_time = 0.0

        # Add new attributes for enhanced functionality
        self.metadata = {
            'num_objects': len(objects),
            'pinned_objects': [i for i, obj in enumerate(objects) if obj.pinned],
            'creation_time': time.time(),
            'last_modified': time.time()
        }

        # Add force tracking
        self.force_history = []

    def record_step(self, time: float, objects: List[SimulationObject]) -> None:
        """
        Record the state of all objects at the current simulation step.

        This method captures the complete state of all simulation objects,
        including their positions, velocities, and accelerations, along
        with the current simulation time.

        Args:
            time: Current simulation time
            objects: List of objects whose state should be recorded
        """
        # Update timing information
        self.simulation_time = time
        self.frame_count += 1
        self.time_history.append(time)

        # Initialize lists for current frame's data
        positions = []
        velocities = []
        accelerations = []

        for i, obj in enumerate(objects):
            positions.extend(obj.position)
            velocities.extend(obj.velocity)
            accelerations.extend(obj.acceleration)
            if forces:
                current_forces.extend(forces.get(obj.obj_id, np.zeros(3)))

        # Store the current frame's data
        self.position_history.append(positions)
        self.velocity_history.append(velocities)
        self.acceleration_history.append(accelerations)
        self.force_history.append(current_forces)


    def clear_history(self) -> None:
        """
        Clear all recorded history and reset counters.

        This method is typically called when resetting the simulation
        or when needing to clear memory.
        """
        # Clear all history lists
        self.time_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()

        # Reset counters
        self.frame_count = 0
        self.simulation_time = 0.0

    def get_time_info(self) -> dict:
        """
        Get current timing information.

        Returns:
            Dictionary containing:
                - frame: Current frame number
                - simulation_time: Current simulation time in seconds
        """
        return {
            'frame': self.frame_count,
            'simulation_time': self.simulation_time
        }

    def save_to_csv(self, file_name: str, start_time: Optional[float] = None,
                    end_time: Optional[float] = None, start_frame: Optional[int] = None,
                    end_frame: Optional[int] = None) -> None:
        """
        Save recorded data to a CSV file with optional time/frame range selection.

        This method allows saving either the complete simulation history or a
        specific portion of it based on either time range or frame range.

        Args:
            file_name: Path to the CSV file to create
            start_time: Start time for data selection (if using time range)
            end_time: End time for data selection (if using time range)
            start_frame: Start frame for data selection (if using frame range)
            end_frame: End frame for data selection (if using frame range)

        Note:
            If both time and frame ranges are specified, time range takes precedence.
        """
        # Determine which frames to save based on provided ranges
        if start_time is not None or end_time is not None:
            # Use time-based selection
            start_idx = 0
            end_idx = len(self.time_history)

            if start_time is not None:
                # Find first frame at or after start_time
                start_idx = next((i for i, t in enumerate(self.time_history)
                                if t >= start_time), 0)
            if end_time is not None:
                # Find first frame after end_time
                end_idx = next((i for i, t in enumerate(self.time_history)
                              if t > end_time), len(self.time_history))

        elif start_frame is not None or end_frame is not None:
            # Use frame-based selection
            start_idx = max(0, start_frame if start_frame is not None else 0)
            end_idx = min(len(self.time_history),
                         end_frame if end_frame is not None else len(self.time_history))
        else:
            # No range specified - save all frames
            start_idx = 0
            end_idx = len(self.time_history)

        # Write data to CSV file
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Create and write headers
            headers = ['Time', 'Frame']
            for i, obj in enumerate(self.objects):
                # Add headers for each object's data
                headers.extend([
                    f'obj{obj.obj_id}_pos_x',   # Position components
                    f'obj{obj.obj_id}_pos_y',
                    f'obj{obj.obj_id}_pos_z'
                ])
                headers.extend([
                    f'obj{obj.obj_id}_vel_x',   # Velocity components
                    f'obj{obj.obj_id}_vel_y',
                    f'obj{obj.obj_id}_vel_z'
                ])
                headers.extend([
                    f'obj{obj.obj_id}_acc_x',   # Acceleration components
                    f'obj{obj.obj_id}_acc_y',
                    f'obj{obj.obj_id}_acc_z'
                ])
            writer.writerow(headers)

            # Write data rows for selected frame range
            for i in range(start_idx, end_idx):
                row = [self.time_history[i], i + 1]  # Time and frame number

                # Add data for each object
                for j in range(len(self.objects)):
                    # Add position vector components
                    row.extend(self.position_history[i][j*3:(j+1)*3])
                    # Add velocity vector components
                    row.extend(self.velocity_history[i][j*3:(j+1)*3])
                    # Add acceleration vector components
                    row.extend(self.acceleration_history[i][j*3:(j+1)*3])

                writer.writerow(row)


def save_simulation_data(file_name: str, time: float, objects: List[SimulationObject]) -> None:
    """
    Save current simulation state to a CSV file.

    This function provides a simpler alternative to the SimulationDataRecorder
    for cases where only the current state needs to be saved.

    Args:
        file_name: Path to the CSV file
        time: Current simulation time
        objects: List of simulation objects to record
    """
    # Check if file exists to determine if headers are needed
    file_exists = os.path.isfile(file_name)

    # Open file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers for new file
        if not file_exists:
            headers = ['Time']
            for obj in objects:
                headers.extend([
                    f'obj{obj.obj_id}_x',
                    f'obj{obj.obj_id}_y',
                    f'obj{obj.obj_id}_z'
                ])
            writer.writerow(headers)

        # Write current state data
        data_row = [time]
        for obj in objects:
            data_row.extend([
                obj.position[0],
                obj.position[1],
                obj.position[2]
            ])
        writer.writerow(data_row)


def load_simulation_data(file_name: str) -> Tuple[List[str], List[List[float]]]:
    """
    Load simulation data from a CSV file.

    Args:
        file_name: Path to the CSV file to read

    Returns:
        Tuple containing:
            - List of column headers
            - List of data columns (each column is a list of float values)

    Note:
        This function assumes all data values can be converted to float.
    """
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Read headers from first row
        headers = next(reader)

        # Initialize lists for each column
        columns = [[] for _ in headers]

        # Read and parse all data rows
        for row in reader:
            for i, value in enumerate(row):
                columns[i].append(float(value))

    return headers, columns


def get_object_positions(headers: List[str], columns: List[List[float]], obj_id: int) -> Tuple[
    List[float], List[float], List[float]]:
    """
    Extract position data for a specific object from loaded simulation data.

    Args:
        headers: Column headers from loaded data
        columns: Data columns from loaded data
        obj_id: ID of the object to get positions for

    Returns:
        Tuple containing:
            - List of x positions
            - List of y positions
            - List of z positions

    Raises:
        ValueError: If object ID not found in data
    """
    # Find column indices for object's position data
    try:
        x_idx = headers.index(f'obj{obj_id}_x')
        y_idx = headers.index(f'obj{obj_id}_y')
        z_idx = headers.index(f'obj{obj_id}_z')
    except ValueError:
        raise ValueError(f"Object ID {obj_id} not found in data")

    return columns[x_idx], columns[y_idx], columns[z_idx]


class DataAnalysis:
    """
    Analyzes and compares data from multiple string simulation runs.

    This enhanced class can load multiple simulation files and provide comparative
    analysis between them, helping identify patterns and differences in node behavior
    across different simulation parameters.
    """

    def __init__(self):
        """
        Initialize the data analysis system to handle multiple simulations.

        The class maintains a dictionary of loaded simulations, each with a unique
        identifier for easy reference in comparisons.
        """
        # Dictionary to store loaded simulation data
        self.simulations = {}

        # Track metadata about each simulation
        self.simulation_metadata = {}

    def load_simulation(self, file_path: str, simulation_id: Optional[str] = None) -> str:
        """
        Load a simulation data file and assign it an identifier.

        Args:
            file_path: Path to the CSV file containing simulation data
            simulation_id: Optional identifier for this simulation. If None,
                         uses the filename without extension

        Returns:
            The simulation identifier used to reference this data

        Raises:
            ValueError: If the CSV file format is invalid
            FileNotFoundError: If the CSV file doesn't exist
        """
        # Create Path object for reliable file handling
        path = Path(file_path)

        # Generate simulation ID if none provided
        if simulation_id is None:
            simulation_id = path.stem

        # Ensure unique identifier
        if simulation_id in self.simulations:
            base_id = simulation_id
            counter = 1
            while simulation_id in self.simulations:
                simulation_id = f"{base_id}_{counter}"
                counter += 1

        try:
            # Load the CSV file
            data = pd.read_csv(file_path)

            # Validate data format
            if not self._validate_data_format(data):
                raise ValueError(f"CSV file {file_path} does not match expected format")

            # Store the data and extract basic metadata
            self.simulations[simulation_id] = data
            self.simulation_metadata[simulation_id] = {
                'file_path': str(path),
                'num_frames': len(data),
                'time_range': (data['Time'].min(), data['Time'].max()),
                'num_objects': len([col for col in data.columns if 'pos_x' in col])
            }

            return simulation_id

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find CSV file: {file_path}")

    def _validate_data_format(self, data: pd.DataFrame) -> bool:
        """
        Validate that a dataframe matches the expected simulation data format.

        Args:
            data: Pandas DataFrame to validate

        Returns:
            bool: True if format is valid, False otherwise
        """
        expected_columns = ['Time', 'Frame']
        num_objects = len([col for col in data.columns if 'pos_x' in col])

        for i in range(num_objects):
            expected_columns.extend([
                f'obj{i}_pos_x', f'obj{i}_pos_y', f'obj{i}_pos_z',
                f'obj{i}_vel_x', f'obj{i}_vel_y', f'obj{i}_vel_z',
                f'obj{i}_acc_x', f'obj{i}_acc_y', f'obj{i}_acc_z'
            ])

        return all(col in data.columns for col in expected_columns)

    def get_object_trajectory(self, simulation_id: str, obj_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get position data for a specific object in a specific simulation.

        Args:
            obj_id: ID of the object to analyze

        Returns:
            Tuple containing arrays of x, y, and z positions

        Raises:
            ValueError: If object ID is invalid
        """
        if simulation_id not in self.simulations:
            raise KeyError(f"Simulation '{simulation_id}' not found")

        data = self.simulations[simulation_id]
        num_objects = self.simulation_metadata[simulation_id]['num_objects']

        if obj_id < 0 or obj_id >= num_objects:
            raise ValueError(f"Invalid object ID. Must be between 0 and {num_objects - 1}")

        x = data[f'obj{obj_id}_pos_x'].values
        y = data[f'obj{obj_id}_pos_y'].values
        z = data[f'obj{obj_id}_pos_z'].values

        return x, y, z

    def find_stationary_nodes(self, simulation_id: str, displacement_threshold: float = 0.01) -> Dict[int, np.ndarray]:
        """
        Find nodes that remain nearly stationary during a specific simulation.

        Args:
            simulation_id: Identifier for the simulation to analyze
            displacement_threshold: Maximum allowed displacement from mean position

        Returns:
            Dictionary mapping object IDs to their mean positions for stationary nodes
        """
        stationary_nodes = {}
        num_objects = self.simulation_metadata[simulation_id]['num_objects']

        for obj_id in range(num_objects):
            x, y, z = self.get_object_trajectory(simulation_id, obj_id)
            positions = np.column_stack([x, y, z])

            mean_pos = np.mean(positions, axis=0)
            max_displacement = np.max(np.linalg.norm(positions - mean_pos, axis=1))

            if max_displacement <= displacement_threshold:
                stationary_nodes[obj_id] = mean_pos

        return stationary_nodes

    def compare_stationary_nodes(self, simulation_ids: List[str],
                                 displacement_threshold: float = 0.01) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Compare stationary nodes across multiple simulations.

        Args:
            simulation_ids: List of simulation identifiers to compare
            displacement_threshold: Maximum allowed displacement for stationary nodes

        Returns:
            Dictionary mapping simulation IDs to their stationary node dictionaries
        """
        return {
            sim_id: self.find_stationary_nodes(sim_id, displacement_threshold)
            for sim_id in simulation_ids
        }

    def plot_comparative_movement(self, simulation_ids: List[str], obj_ids: Optional[List[int]] = None,
                                  highlight_stationary: bool = True, displacement_threshold: float = 0.01):
        """
        Create a comparative 3D visualization of node movements across multiple simulations.

        Args:
            simulation_ids: List of simulation identifiers to compare
            obj_ids: Optional list of object IDs to plot. If None, plots all objects
            highlight_stationary: Whether to highlight stationary nodes
            displacement_threshold: Threshold for identifying stationary nodes
        """
        # Create figure with subplots for each simulation
        n_sims = len(simulation_ids)
        fig = plt.figure(figsize=(6 * n_sims, 8))

        # Create a color map for consistent object colors across plots
        colors = plt.cm.rainbow(np.linspace(0, 1, max(
            self.simulation_metadata[sim_id]['num_objects']
            for sim_id in simulation_ids
        )))

        for i, sim_id in enumerate(simulation_ids):
            ax = fig.add_subplot(1, n_sims, i + 1, projection='3d')

            # Get object IDs to plot
            if obj_ids is None:
                plot_obj_ids = range(self.simulation_metadata[sim_id]['num_objects'])
            else:
                plot_obj_ids = obj_ids

            # Plot movement paths
            for obj_id in plot_obj_ids:
                x, y, z = self.get_object_trajectory(sim_id, obj_id)
                ax.plot(x, y, z, color=colors[obj_id], alpha=0.5,
                        label=f'Node {obj_id}')

            # Highlight stationary nodes if requested
            if highlight_stationary:
                stationary_nodes = self.find_stationary_nodes(sim_id, displacement_threshold)
                for obj_id, pos in stationary_nodes.items():
                    if obj_ids is None or obj_id in obj_ids:
                        ax.scatter(pos[0], pos[1], pos[2],
                                   color='red', s=100, marker='*',
                                   label=f'Stationary {obj_id}')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Simulation: {sim_id}')

            # Add legend if not too many objects
            if len(plot_obj_ids) <= 10:
                ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_displacement_comparison(self, simulation_ids: List[str], obj_id: int):
        """
        Create a comparative plot of a specific node's displacement over time
        across multiple simulations.

        Args:
            simulation_ids: List of simulation identifiers to compare
            obj_id: ID of the node to analyze
        """
        plt.figure(figsize=(12, 6))

        for sim_id in simulation_ids:
            # Get position data
            x, y, z = self.get_object_trajectory(sim_id, obj_id)
            positions = np.column_stack([x, y, z])

            # Calculate displacement from initial position
            initial_pos = positions[0]
            displacements = np.linalg.norm(positions - initial_pos, axis=1)

            # Plot displacement over time
            time = self.simulations[sim_id]['Time'].values
            plt.plot(time, displacements, label=sim_id)

        plt.xlabel('Time (s)')
        plt.ylabel('Displacement from Initial Position')
        plt.title(f'Node {obj_id} Displacement Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_simulation_summary(self, simulation_id: str) -> Dict:
        """
        Generate a summary of key statistics for a simulation.

        Args:
            simulation_id: Identifier for the simulation to analyze

        Returns:
            Dictionary containing summary statistics
        """
        if simulation_id not in self.simulations:
            raise KeyError(f"Simulation '{simulation_id}' not found")

        data = self.simulations[simulation_id]

        # Calculate various statistics
        stationary_nodes = self.find_stationary_nodes(simulation_id)
        num_objects = self.simulation_metadata[simulation_id]['num_objects']

        return {
            'num_frames': len(data),
            'simulation_time': data['Time'].max() - data['Time'].min(),
            'num_objects': num_objects,
            'num_stationary_nodes': len(stationary_nodes),
            'stationary_node_positions': stationary_nodes
        }