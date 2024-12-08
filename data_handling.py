"""Data handling module for saving string simulation data to CSV files.
Handles saving state data for simulation objects, including positions over time."""

import csv
import os
from typing import List
from Mass import SimulationObject


def save_simulation_data(file_name: str, time: float, objects: List[SimulationObject]) -> None:
    """
    Saves or appends the current simulation state to a CSV file.

    Creates a CSV with columns for time and x,y,z positions of each object.
    If the file exists, appends the new data. If not, creates it with headers.

    Args:
        file_name (str): Name of the CSV file to write to
        time (float): Current simulation time
        objects (List[SimulationObject]): List of simulation objects to record
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_name)

    # Open file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers if new file
        if not file_exists:
            # Start with time column
            headers = ['Time']

            # Add position columns for each object
            for i, obj in enumerate(objects):
                headers.extend([
                    f'obj{obj.obj_id}_x',
                    f'obj{obj.obj_id}_y',
                    f'obj{obj.obj_id}_z'
                ])

            writer.writerow(headers)

        # Prepare data row
        data_row = [time]

        # Add position data for each object
        for obj in objects:
            data_row.extend([
                obj.position[0],  # x position
                obj.position[1],  # y position
                obj.position[2]  # z position
            ])

        writer.writerow(data_row)


def load_simulation_data(file_name: str) -> tuple[List[str], List[List[float]]]:
    """
    Loads simulation data from a CSV file.

    Args:
        file_name (str): Name of the CSV file to read from

    Returns:
        Tuple containing:
            - List of column headers
            - List of data columns (each column is a list of float values)
    """
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Read headers
        headers = next(reader)

        # Initialize columns list
        columns = [[] for _ in headers]

        # Read data rows
        for row in reader:
            for i, value in enumerate(row):
                columns[i].append(float(value))

    return headers, columns


def get_object_positions(headers: List[str], columns: List[List[float]], obj_id: int) -> tuple[
    List[float], List[float], List[float]]:
    """
    Extracts position data for a specific object from loaded simulation data.

    Args:
        headers (List[str]): Column headers from loaded data
        columns (List[List[float]]): Data columns from loaded data
        obj_id (int): ID of the object to get positions for

    Returns:
        Tuple containing:
            - List of x positions
            - List of y positions
            - List of z positions

    Raises:
        ValueError: If object ID not found in data
    """
    # Find indices for object's position columns
    x_idx = headers.index(f'obj{obj_id}_x')
    y_idx = headers.index(f'obj{obj_id}_y')
    z_idx = headers.index(f'obj{obj_id}_z')

    return columns[x_idx], columns[y_idx], columns[z_idx]