import os
import pandas as pd
import numpy as np


class SimulationDataRecorder:
    """
    This class is responsible for recording simulation data (time, positions,
    velocities, and accelerations) of all objects at each timestep, and
    provides functionality to save the recorded data to a CSV file.
    """

    def __init__(self, num_objects: int):
        """
        Initialize the data recorder with the given number of objects.

        Args:
            num_objects: Number of objects in the simulation whose data we will record.
        """
        # Store the number of objects
        self.num_objects = num_objects
        # Lists to hold the recorded time, position, velocity, and acceleration data for all timesteps
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []

    def record_step(self, time: float, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray):
        """
        Record one timestep of simulation data: time, positions, velocities, and accelerations.

        Args:
            time: Current simulation time (float)
            positions: np.ndarray of shape (num_objects, 3) - positions of all objects at this timestep
            velocities: np.ndarray of shape (num_objects, 3) - velocities of all objects at this timestep
            accelerations: np.ndarray of shape (num_objects, 3) - accelerations of all objects at this timestep

        Raises:
            ValueError: If the provided arrays do not match the expected number of objects.
        """
        # Validate that positions, velocities, and accelerations match the expected number of objects
        if (len(positions) != self.num_objects or
                len(velocities) != self.num_objects or
                len(accelerations) != self.num_objects):
            raise ValueError("Data arrays do not match the number of objects.")

        # Store the current time and copies of the arrays so we don't modify the original data later
        self.time_history.append(time)
        self.position_history.append(positions.copy())
        self.velocity_history.append(velocities.copy())
        self.acceleration_history.append(accelerations.copy())

    def clear_history(self):
        """
        Clear all recorded data. Useful if we want to reset the recorder before a new simulation run.
        """
        # Clear all stored lists
        self.time_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()

    def save_to_csv(self, file_path: str):
        """
        Save the recorded simulation data to a CSV file. The CSV includes columns for time, and for each object:
        Xn, Yn, Zn, Xvn, Yvn, Zvn, Xan, Yan, Zan.

        Args:
            file_path: Path where the CSV file will be saved.

        Raises:
            ValueError: If no data has been recorded yet.
        """
        # Ensure that there is data to save
        if not self.time_history:
            raise ValueError("No data recorded. Nothing to save.")

        num_steps = len(self.time_history)
        num_objects = self.num_objects

        # Prepare column names: Time plus 9 columns per object (Position: X,Y,Z; Velocity: Xv,Yv,Zv; Acceleration: Xa,Ya,Za)
        columns = ["Time"]
        for n in range(num_objects):
            columns += [f"X{n}", f"Y{n}", f"Z{n}", f"Xv{n}", f"Yv{n}", f"Zv{n}", f"Xa{n}", f"Ya{n}", f"Za{n}"]

        # Create a data array to hold all info. Each row is one timestep, each column is one data field.
        # 1 column for Time + 9 columns per object
        data = np.zeros((num_steps, 1 + 9 * num_objects))
        # Fill in the time column
        data[:, 0] = self.time_history

        # Loop through each recorded step and fill in position, velocity, and acceleration data for each object
        for i in range(num_steps):
            pos = self.position_history[i]
            vel = self.velocity_history[i]
            acc = self.acceleration_history[i]
            for n in range(num_objects):
                # Calculate the starting column index for this object's data in this row
                start_col = 1 + n * 9
                # Assign position (X, Y, Z)
                data[i, start_col] = pos[n, 0]
                data[i, start_col + 1] = pos[n, 1]
                data[i, start_col + 2] = pos[n, 2]
                # Assign velocity (Xv, Yv, Zv)
                data[i, start_col + 3] = vel[n, 0]
                data[i, start_col + 4] = vel[n, 1]
                data[i, start_col + 5] = vel[n, 2]
                # Assign acceleration (Xa, Ya, Za)
                data[i, start_col + 6] = acc[n, 0]
                data[i, start_col + 7] = acc[n, 1]
                data[i, start_col + 8] = acc[n, 2]

        # Convert the array to a DataFrame for convenient CSV output
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_path, index=False)

    def get_time_info(self):
        """
        Return basic time-related info about the simulation, specifically the total simulated time.

        Returns:
            A dictionary with 'simulation_time': the difference between last and first recorded time.
        """
        # If fewer than 2 steps, we can't compute a meaningful duration (returns 0.0)
        if len(self.time_history) < 2:
            sim_time = 0.0
        else:
            sim_time = self.time_history[-1] - self.time_history[0]
        return {
            'simulation_time': sim_time
        }


class DataAnalysis:
    """
    This class loads simulation CSV files produced by SimulationDataRecorder
    and provides methods to analyze them: summarizing simulations, extracting
    trajectories, velocities, accelerations, identifying stationary nodes,
    computing average displacements, and performing harmonic correlation analysis.
    """

    def __init__(self):
        # Dictionary to store DataFrames of loaded simulations keyed by file path
        self.simulations = {}

    def load_simulation(self, file_path):
        """
        Load a simulation CSV file into memory for analysis.

        Args:
            file_path: The path to the CSV file.

        Returns:
            file_path: The same input file path if successful.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file doesn't have a 'Time' column.
        """
        # Check that file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read CSV into a DataFrame
        df = pd.read_csv(file_path)
        # Ensure 'Time' column exists
        if "Time" not in df.columns:
            raise ValueError("CSV file must have a 'Time' column.")

        # Store DataFrame in the dictionary
        self.simulations[file_path] = df
        return file_path

    def _detect_objects(self, file_path):
        """
        Determine how many objects are in the simulation by checking for sets
        of columns that correspond to each object's position, velocity, and acceleration.

        Args:
            file_path: Key to the loaded simulation DataFrame.

        Returns:
            num_objects: The number of objects detected.

        Raises:
            ValueError: If no valid object columns are found.
        """
        df = self.simulations[file_path]

        # Check if for a given index n, we have all required columns:
        # (X{n}, Y{n}, Z{n}, Xv{n}, Yv{n}, Zv{n}, Xa{n}, Ya{n}, Za{n})
        def have_full_set(n):
            pos_cols = [f"X{n}", f"Y{n}", f"Z{n}"]
            vel_cols = [f"Xv{n}", f"Yv{n}", f"Zv{n}"]
            acc_cols = [f"Xa{n}", f"Ya{n}", f"Za{n}"]
            for c in pos_cols + vel_cols + acc_cols:
                if c not in df.columns:
                    return False
            return True

        node_ids = []
        i = 0
        # Attempt to find consecutive sets of columns for n = 0, 1, 2, ...
        while True:
            if have_full_set(i):
                node_ids.append(i)
                i += 1
            else:
                break

        num_objects = len(node_ids)
        if num_objects == 0:
            raise ValueError(
                "No valid object columns found. "
                "Ensure each node has Xn, Yn, Zn, Xv{n}, Yv{n}, Zv{n}, Xa{n}, Ya{n}, Za{n} columns."
            )

        return num_objects

    def get_simulation_summary(self, file_path):
        """
        Provide a summary of the simulation, including:
        - Number of frames
        - Total simulation time
        - Number of objects
        - Number of stationary nodes
        - Positions of stationary nodes

        Args:
            file_path: The path (key) to the simulation data.

        Returns:
            A dictionary containing summary information.
        """
        df = self.simulations[file_path]
        time_col = df["Time"]
        num_frames = len(df)
        simulation_time = time_col.iloc[-1] - time_col.iloc[0]
        num_objects = self._detect_objects(file_path)
        # Use 5% as default threshold for summary
        stationary_nodes = self.find_stationary_nodes(file_path, threshold_percentage=10)
        num_stationary_nodes = len(stationary_nodes)

        return {
            "num_frames": num_frames,
            "simulation_time": simulation_time,
            "num_objects": num_objects,
            "num_stationary_nodes": num_stationary_nodes,
            "stationary_node_positions": stationary_nodes
        }

    def get_object_trajectory(self, file_path, node_id):
        """
        Get the trajectory (X, Y, Z over time) for a specific object/node.

        Args:
            file_path: Key to the simulation data.
            node_id: The index of the object/node.

        Returns:
            x, y, z: Arrays of the object's positions over time.

        Raises:
            ValueError: If the required position columns are missing.
        """
        df = self.simulations[file_path]
        x_col = f"X{node_id}"
        y_col = f"Y{node_id}"
        z_col = f"Z{node_id}"
        # Check columns exist
        for col in [x_col, y_col, z_col]:
            if col not in df.columns:
                raise ValueError(f"Position column {col} missing for node {node_id}.")
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        return x, y, z

    def get_object_velocities(self, file_path, node_id):
        """
        Get the velocity (Vx, Vy, Vz over time) for a specific object/node.

        Args:
            file_path: Key to the simulation data.
            node_id: The index of the object/node.

        Returns:
            vx, vy, vz: Arrays of the object's velocities over time.

        Raises:
            ValueError: If the required velocity columns are missing.
        """
        df = self.simulations[file_path]
        vx_col = f"Xv{node_id}"
        vy_col = f"Yv{node_id}"
        vz_col = f"Zv{node_id}"
        # Check columns exist
        for col in [vx_col, vy_col, vz_col]:
            if col not in df.columns:
                raise ValueError(f"Velocity column {col} missing for node {node_id}.")
        vx = df[vx_col].values
        vy = df[vy_col].values
        vz = df[vz_col].values
        return vx, vy, vz

    def get_object_accelerations(self, file_path, node_id):
        """
        Get the acceleration (Ax, Ay, Az over time) for a specific object/node.

        Args:
            file_path: Key to the simulation data.
            node_id: The index of the object/node.

        Returns:
            ax, ay, az: Arrays of the object's accelerations over time.

        Raises:
            ValueError: If the required acceleration columns are missing.
        """
        df = self.simulations[file_path]
        ax_col = f"Xa{node_id}"
        ay_col = f"Ya{node_id}"
        az_col = f"Za{node_id}"
        # Check columns exist
        for col in [ax_col, ay_col, az_col]:
            if col not in df.columns:
                raise ValueError(f"Acceleration column {col} missing for node {node_id}.")
        ax = df[ax_col].values
        ay = df[ay_col].values
        az = df[az_col].values
        return ax, ay, az

    def find_stationary_nodes(self, file_path, threshold_percentage=10.0):
        """
        Identify nodes that have negligible displacement from their initial position,
        using a threshold that's a percentage of the maximum displacement observed.

        Args:
            file_path: Key to the simulation data.
            threshold_percentage: The percentage of maximum displacement to use as threshold (default: 5.0)

        Returns:
            A dictionary of node_id -> initial position for all stationary nodes.

        Raises:
            ValueError: If threshold_percentage is not between 0 and 100.
        """
        if not 0 <= threshold_percentage <= 100:
            raise ValueError("Threshold percentage must be between 0 and 100")

        num_objects = self._detect_objects(file_path)
        stationary_nodes = {}

        # First pass: calculate all displacements and find maximum
        max_displacement = 0
        node_displacements = []
        node_initial_positions = []

        for node_id in range(num_objects):
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            avg_disp = np.mean(displacements)

            max_displacement = max(max_displacement, avg_disp)
            node_displacements.append(avg_disp)
            node_initial_positions.append(initial_pos)

        # Calculate threshold based on percentage of maximum displacement
        threshold = (threshold_percentage / 100.0) * max_displacement

        # Second pass: identify stationary nodes using normalized threshold
        for node_id in range(num_objects):
            if node_displacements[node_id] < threshold:
                stationary_nodes[node_id] = node_initial_positions[node_id]

        return stationary_nodes

    def get_average_displacements(self, file_path):
        """
        Compute the average displacement of each node from its initial position.

        Args:
            file_path: Key to the simulation data.

        Returns:
            node_ids: Array of node indices
            avg_displacements: Array of average displacements for each node
        """
        num_objects = self._detect_objects(file_path)
        node_ids = np.arange(num_objects)
        avg_displacements = []
        for node_id in node_ids:
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            avg_displacements.append(np.mean(displacements))
        return node_ids, np.array(avg_displacements)

    def get_harmonic_correlation(self, file_path, num_harmonics=10):
        """
        Analyze how well the node displacement pattern matches simple harmonic modes.

        Args:
            file_path: Key to the simulation data.
            num_harmonics: Number of harmonics to consider.

        Returns:
            harmonics: Array of harmonic indices (1 to num_harmonics).
            correlations: Array of correlation values (in %) for each harmonic.
        """

        def generate_harmonic(x, n):
            # Generate a sine wave pattern for harmonic number n
            return np.sin(n * np.pi * x)

        def compute_correlation(x, y):
            # Compute correlation between two signals x and y
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
            return numerator / denominator if denominator != 0 else 0

        # Get basic simulation info and number of objects
        summary = self.get_simulation_summary(file_path)
        num_objects = summary['num_objects']

        node_ids = np.arange(num_objects)
        pattern = []
        # Compute RMS displacement pattern across all nodes
        for node_id in node_ids:
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            rms_disp = np.sqrt(np.mean(displacements ** 2))
            pattern.append(rms_disp)

        pattern = np.array(pattern)
        # Normalizes pattern by maximum displacement for consistent comparison
        max_disp = np.max(np.abs(pattern))
        if max_disp > 0:
            pattern /= max_disp
        else:
            # If no displacement, return zero correlations
            return np.arange(1, num_harmonics + 1), np.zeros(num_harmonics)

        # Generate a uniform set of node positions from 0 to 1 for harmonic analysis
        x_positions = np.linspace(0, 1, num_objects)
        harmonics = np.arange(1, num_harmonics + 1)
        correlations = []

        # Compare pattern against each harmonic shape
        for n in harmonics:
            harmonic_pattern = generate_harmonic(x_positions, n)
            corr = compute_correlation(pattern, harmonic_pattern)
            correlations.append(abs(corr) * 100)

        return harmonics, np.array(correlations)
