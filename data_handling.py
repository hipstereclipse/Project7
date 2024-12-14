import os
import pandas as pd
import numpy as np


class SimulationDataRecorder:
    """
    I use this class to keep track of simulation data (time, positions, velocities,
    and accelerations) for each timestep. After I'm done running the simulation,
    I can save all this recorded data to a CSV file if I want.
    """

    def __init__(self, num_objects: int):
        """
        I initialize the recorder by specifying how many objects I'm tracking.
        I then create empty lists to store time and the arrays of positions, velocities,
        and accelerations at each recorded step.
        """
        self.num_objects = num_objects
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []

    def record_step(self, time: float, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray):
        """
        Here I record a single timestep. I expect positions, velocities, and accelerations
        to be arrays with shapes like (num_objects, 3).

        I make sure these arrays have the correct length to match the number of objects.
        If they do, I append the current time and copies of these arrays to my histories.
        """
        if (len(positions) != self.num_objects or
                len(velocities) != self.num_objects or
                len(accelerations) != self.num_objects):
            raise ValueError("The provided data doesn't match the number of objects.")

        self.time_history.append(time)
        self.position_history.append(positions.copy())
        self.velocity_history.append(velocities.copy())
        self.acceleration_history.append(accelerations.copy())

    def clear_history(self):
        """
        If I want to start fresh, I can call this to clear all previously recorded data.
        Maybe I ran a simulation and now want to run a new one without mixing the data.
        """
        self.time_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()

    def save_to_csv(self, file_path: str):
        """
        When I'm done with a simulation, I can call this to write out all the recorded data to a CSV file.
        The CSV will have a 'Time' column and then, for each object, columns like:
        Xn, Yn, Zn, Xvn, Yvn, Zvn, Xan, Yan, Zan (position, velocity, acceleration).

        I need to have recorded some data first; if I haven't, I'll raise a ValueError.
        """
        if not self.time_history:
            raise ValueError("No data recorded, so I have nothing to save.")

        num_steps = len(self.time_history)
        num_objects = self.num_objects

        columns = ["Time"]
        for n in range(num_objects):
            # For each object, I store X, Y, Z, Xv, Yv, Zv, Xa, Ya, Za
            columns += [f"X{n}", f"Y{n}", f"Z{n}", f"Xv{n}", f"Yv{n}", f"Zv{n}", f"Xa{n}", f"Ya{n}", f"Za{n}"]

        data = np.zeros((num_steps, 1 + 9 * num_objects))
        data[:, 0] = self.time_history

        for i in range(num_steps):
            pos = self.position_history[i]
            vel = self.velocity_history[i]
            acc = self.acceleration_history[i]
            for n in range(num_objects):
                start_col = 1 + n * 9
                # Positions
                data[i, start_col] = pos[n, 0]
                data[i, start_col + 1] = pos[n, 1]
                data[i, start_col + 2] = pos[n, 2]
                # Velocities
                data[i, start_col + 3] = vel[n, 0]
                data[i, start_col + 4] = vel[n, 1]
                data[i, start_col + 5] = vel[n, 2]
                # Accelerations
                data[i, start_col + 6] = acc[n, 0]
                data[i, start_col + 7] = acc[n, 1]
                data[i, start_col + 8] = acc[n, 2]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_path, index=False)

    def get_time_info(self):
        """
        I just want to know how long the simulation lasted, so I return the total simulated time.
        If there's fewer than 2 steps, I'll consider the simulation time as 0.0 since I can't measure a duration.
        """
        if len(self.time_history) < 2:
            sim_time = 0.0
        else:
            sim_time = self.time_history[-1] - self.time_history[0]
        return {
            'simulation_time': sim_time
        }


class DataAnalysis:
    """
    I use this class after the simulation is done and I have CSV files.
    I load those files into memory and then I can do various forms of analysis:
    - Check how many objects there are
    - Compute trajectories, velocities, accelerations for each node
    - Find stationary nodes
    - Compute average and normalized displacements
    - Do harmonic analysis
    and so forth.
    """

    def __init__(self):
        # I'll store each loaded simulation's DataFrame in this dictionary, keyed by the file path.
        self.simulations = {}

    def load_simulation(self, file_path):
        """
        I load a CSV file that should have 'Time' and then columns for each object's data.
        If the file doesn't exist or doesn't have a Time column, I'll raise an error.
        If it works, I return the file_path so I know which key to use later.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        if "Time" not in df.columns:
            raise ValueError("CSV file must have a 'Time' column.")

        self.simulations[file_path] = df
        return file_path

    def _detect_objects(self, file_path):
        """
        I figure out how many objects there are by checking for columns named X0,Y0,Z0,...Xa0, etc.
        I start at n=0 and go up until I can't find a full set of required columns for an object.

        If I don't find any objects at all, I raise an error.
        """
        df = self.simulations[file_path]

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
        while True:
            if have_full_set(i):
                node_ids.append(i)
                i += 1
            else:
                break

        num_objects = len(node_ids)
        if num_objects == 0:
            raise ValueError(
                "I couldn't find valid object columns. "
                "I expect columns like Xn, Yn, Zn, Xvn, Yvn, Zv n, Xan, Yan, Zan."
            )

        return num_objects

    def get_simulation_summary(self, file_path):
        """
        I create a quick summary of the simulation, including:
        - How many frames?
        - Total simulation time?
        - How many objects?
        - How many nodes ended up stationary (and what are their positions)?

        I return this info as a dictionary.
        """
        df = self.simulations[file_path]
        time_col = df["Time"]
        num_frames = len(df)
        simulation_time = time_col.iloc[-1] - time_col.iloc[0]
        num_objects = self._detect_objects(file_path)

        # I'm calling find_stationary_nodes with a threshold_percentage=15 by default.
        # That means if a node doesn't move at more than 15% of max displacement, I consider it stationary.
        stationary_nodes = self.find_stationary_nodes(file_path, threshold_percentage=15)
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
        I want the X, Y, Z position over time for a given node_id.
        I check columns Xn, Yn, Zn and return their values as arrays.
        If these columns are missing, I raise a ValueError.
        """
        df = self.simulations[file_path]
        x_col = f"X{node_id}"
        y_col = f"Y{node_id}"
        z_col = f"Z{node_id}"
        for col in [x_col, y_col, z_col]:
            if col not in df.columns:
                raise ValueError(f"Missing position column {col} for node {node_id}.")
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        return x, y, z

    def get_object_velocities(self, file_path, node_id):
        """
        Similar to get_object_trajectory, but for velocity columns Xv, Yv, Zv.
        I return vx, vy, vz as arrays.
        """
        df = self.simulations[file_path]
        vx_col = f"Xv{node_id}"
        vy_col = f"Yv{node_id}"
        vz_col = f"Zv{node_id}"
        for col in [vx_col, vy_col, vz_col]:
            if col not in df.columns:
                raise ValueError(f"Missing velocity column {col} for node {node_id}.")
        vx = df[vx_col].values
        vy = df[vy_col].values
        vz = df[vz_col].values
        return vx, vy, vz

    def get_object_accelerations(self, file_path, node_id):
        """
        Again, similar approach, but now for acceleration columns Xa, Ya, Za.
        I return ax, ay, az.
        """
        df = self.simulations[file_path]
        ax_col = f"Xa{node_id}"
        ay_col = f"Ya{node_id}"
        az_col = f"Za{node_id}"
        for col in [ax_col, ay_col, az_col]:
            if col not in df.columns:
                raise ValueError(f"Missing acceleration column {col} for node {node_id}.")
        ax = df[ax_col].values
        ay = df[ay_col].values
        az = df[az_col].values
        return ax, ay, az

    def find_stationary_nodes(self, file_path, threshold_percentage=15.0):
        """
        I want to find nodes that barely moved compared to others. I define a threshold based on a percentage
        of the max average displacement. If a node's average displacement is below that threshold,
        I consider it stationary.

        I return a dictionary {node_id: initial_position}.
        """
        if not 0 <= threshold_percentage <= 100:
            raise ValueError("Threshold percentage must be between 0 and 100")

        num_objects = self._detect_objects(file_path)
        stationary_nodes = []

        max_displacement = 0
        node_displacements = []
        node_initial_positions = []

        for node_id in range(num_objects):
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            avg_disp = np.mean(displacements)

            if avg_disp > max_displacement:
                max_displacement = avg_disp
            node_displacements.append(avg_disp)
            node_initial_positions.append(initial_pos)

        threshold = (threshold_percentage / 100.0) * max_displacement

        stationary_dict = {}
        for node_id in range(num_objects):
            if node_displacements[node_id] < threshold:
                stationary_dict[node_id] = node_initial_positions[node_id]

        return stationary_dict

    def get_average_displacements(self, file_path):
        """
        I want to know how much, on average, each node moved away from its initial position.
        I compute the displacement at each timestep, average it, and return these averages.

        returns arrays: node_ids and their corresponding average displacements.
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

    def get_normalized_average_displacements(self, file_path):
        """
        I want to compute something similar to average displacements, but now I want to normalize each node's displacement
        based on that node's own maximum displacement. That means for each node, I:
        1. Compute its displacement time-series.
        2. Find the maximum displacement for that node.
        3. Divide all of its displacements by that maximum, resulting in values between 0 and 1.
        4. Compute the average of these normalized values to represent the node's normalized displacement.

        returns arrays: node_ids and their corresponding normalized average displacements.
        """
        num_objects = self._detect_objects(file_path)
        node_ids = np.arange(num_objects)
        normalized_averages = np.zeros(num_objects)

        # Iterate over each node and compute normalized average displacement
        for i, node_id in enumerate(node_ids):
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)

            # Avoid division by zero if max displacement is zero (stationary node)
            max_disp = np.max(displacements) if np.max(displacements) != 0 else 1.0
            normalized = displacements / max_disp

            # Use the average of normalized displacements as the node's normalized value
            normalized_averages[i] = np.mean(normalized)

        return node_ids, normalized_averages

    def get_normalized_amplitudes(self, file_path):
        """
        Computes normalized maximum amplitudes for each node to make them comparable
        regardless of their absolute displacement values. For each node:
        1. Compute its total displacement from initial position over time
        2. Find the maximum displacement value (peak amplitude)
        3. Compare this to the largest displacement seen in any node
        4. Normalize to get values between 0 and 1

        Returns:
        - node_ids: Array of node indices
        - normalized_amplitudes: Array of normalized maximum amplitudes for each node
        """
        # First, detect how many nodes we have in the simulation
        num_objects = self._detect_objects(file_path)
        node_ids = np.arange(num_objects)

        # We'll store the maximum displacement (amplitude) for each node
        max_amplitudes = np.zeros(num_objects)

        # Calculate maximum displacement for each node
        for i, node_id in enumerate(node_ids):
            # Get trajectory data
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])

            # Calculate displacement from initial position at each timestep
            displacements = np.linalg.norm(positions - initial_pos, axis=1)

            # Store the maximum displacement this node experienced
            max_amplitudes[i] = np.max(displacements)

        # Find the overall maximum displacement across all nodes
        overall_max = np.max(max_amplitudes)

        # Normalize all amplitudes by the overall maximum
        # Add small epsilon to avoid division by zero
        normalized_amplitudes = max_amplitudes / (overall_max + 1e-10)

        return node_ids, normalized_amplitudes

    def get_harmonic_correlation(self, file_path, num_harmonics=10):
        """
        I want to see how the displacement pattern across nodes relates to harmonic modes.
        I generate simple sine wave patterns (harmonics) and compute correlation with the actual pattern.

        I first compute an RMS displacement pattern for each node, normalize it, and then compare it
        to harmonic patterns. I return arrays of harmonic indices and their correlation (in %) with the pattern.
        """

        def generate_harmonic(x, n):
            return np.sin(n * np.pi * x)

        def compute_correlation(x, y):
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
            return numerator / denominator if denominator != 0 else 0

        summary = self.get_simulation_summary(file_path)
        num_objects = summary['num_objects']

        node_ids = np.arange(num_objects)
        pattern = []
        for node_id in node_ids:
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            rms_disp = np.sqrt(np.mean(displacements ** 2))
            pattern.append(rms_disp)

        pattern = np.array(pattern)
        max_disp = np.max(np.abs(pattern))
        if max_disp > 0:
            pattern /= max_disp
        else:
            # If there's no displacement at all, just return zero correlations
            return np.arange(1, num_harmonics + 1), np.zeros(num_harmonics)

        x_positions = np.linspace(0, 1, num_objects)
        harmonics = np.arange(1, num_harmonics + 1)
        correlations = []

        for n in harmonics:
            harmonic_pattern = generate_harmonic(x_positions, n)
            corr = compute_correlation(pattern, harmonic_pattern)
            correlations.append(abs(corr) * 100)

        return harmonics, np.array(correlations)
