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

import csv
import os
from typing import List, Optional, Tuple
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

        # Record state data for each object
        for obj in objects:
            # Extend lists with x, y, z components
            positions.extend(obj.position)       # Add position vector components
            velocities.extend(obj.velocity)      # Add velocity vector components
            accelerations.extend(obj.acceleration)  # Add acceleration vector components

        # Store the current frame's data
        self.position_history.append(positions)
        self.velocity_history.append(velocities)
        self.acceleration_history.append(accelerations)

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