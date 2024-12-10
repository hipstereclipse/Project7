import os
import pandas as pd
import numpy as np

class SimulationDataRecorder:
    """
    Records time, positions, velocities, and accelerations of all objects at each time step.
    This class can then save the recorded data to a CSV file and provide basic info about the simulation.
    """

    def __init__(self, num_objects: int):
        """
        Initialize the data recorder.

        Args:
            num_objects: Number of objects in the simulation.
        """
        self.num_objects = num_objects
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []

    def record_step(self, time: float, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray):
        """
        Record a single time step of simulation data.

        Args:
            time: Current simulation time (float).
            positions: np.ndarray of shape (num_objects, 3)
            velocities: np.ndarray of shape (num_objects, 3)
            accelerations: np.ndarray of shape (num_objects, 3)
        """
        if (len(positions) != self.num_objects or
            len(velocities) != self.num_objects or
            len(accelerations) != self.num_objects):
            raise ValueError("Data arrays do not match the number of objects.")

        self.time_history.append(time)
        self.position_history.append(positions.copy())
        self.velocity_history.append(velocities.copy())
        self.acceleration_history.append(accelerations.copy())

    def clear_history(self):
        """Clear all recorded data."""
        self.time_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()

    def save_to_csv(self, file_path: str):
        """
        Save the recorded simulation data to a CSV file.

        The CSV will have columns:
        Time, X0, Y0, Z0, Xv0, Yv0, Zv0, Xa0, Ya0, Za0, X1, Y1, Z1, Xv1, Yv1, Zv1, Xa1, Ya1, Za1, ...
        """
        if not self.time_history:
            raise ValueError("No data recorded. Nothing to save.")

        num_steps = len(self.time_history)
        num_objects = self.num_objects

        # Prepare columns
        columns = ["Time"]
        for n in range(num_objects):
            columns += [f"X{n}", f"Y{n}", f"Z{n}", f"Xv{n}", f"Yv{n}", f"Zv{n}", f"Xa{n}", f"Ya{n}", f"Za{n}"]

        # Create data array
        data = np.zeros((num_steps, 1 + 9 * num_objects))
        data[:, 0] = self.time_history

        for i in range(num_steps):
            pos = self.position_history[i]
            vel = self.velocity_history[i]
            acc = self.acceleration_history[i]
            for n in range(num_objects):
                start_col = 1 + n * 9
                data[i, start_col] = pos[n, 0]    # Xn
                data[i, start_col+1] = pos[n, 1]  # Yn
                data[i, start_col+2] = pos[n, 2]  # Zn
                data[i, start_col+3] = vel[n, 0]  # Xv{n}
                data[i, start_col+4] = vel[n, 1]  # Yv{n}
                data[i, start_col+5] = vel[n, 2]  # Zv{n}
                data[i, start_col+6] = acc[n, 0]  # Xa{n}
                data[i, start_col+7] = acc[n, 1]  # Ya{n}
                data[i, start_col+8] = acc[n, 2]  # Za{n}

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_path, index=False)

    def get_time_info(self):
        """
        Return a dictionary with basic time-related info about the simulation.
        Currently, this includes 'simulation_time', which is the difference between the last and first recorded time.
        """
        if len(self.time_history) < 2:
            sim_time = 0.0
        else:
            sim_time = self.time_history[-1] - self.time_history[0]
        return {
            'simulation_time': sim_time
        }

class DataAnalysis:
    def __init__(self):
        self.simulations = {}

    def load_simulation(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        if "Time" not in df.columns:
            raise ValueError("CSV file must have a 'Time' column.")

        self.simulations[file_path] = df
        return file_path

    def _detect_objects(self, file_path):
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
                "No valid object columns found. "
                "Ensure each node has Xn, Yn, Zn, Xv{n}, Yv{n}, Zv{n}, Xa{n}, Ya{n}, Za{n} columns."
            )

        return num_objects

    def get_simulation_summary(self, file_path):
        df = self.simulations[file_path]
        time_col = df["Time"]
        num_frames = len(df)
        simulation_time = time_col.iloc[-1] - time_col.iloc[0]
        num_objects = self._detect_objects(file_path)
        stationary_nodes = self.find_stationary_nodes(file_path)
        num_stationary_nodes = len(stationary_nodes)

        return {
            "num_frames": num_frames,
            "simulation_time": simulation_time,
            "num_objects": num_objects,
            "num_stationary_nodes": num_stationary_nodes,
            "stationary_node_positions": stationary_nodes
        }

    def get_object_trajectory(self, file_path, node_id):
        df = self.simulations[file_path]
        x_col = f"X{node_id}"
        y_col = f"Y{node_id}"
        z_col = f"Z{node_id}"
        for col in [x_col, y_col, z_col]:
            if col not in df.columns:
                raise ValueError(f"Position column {col} missing for node {node_id}.")
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        return x, y, z

    def get_object_velocities(self, file_path, node_id):
        df = self.simulations[file_path]
        vx_col = f"Xv{node_id}"
        vy_col = f"Yv{node_id}"
        vz_col = f"Zv{node_id}"
        for col in [vx_col, vy_col, vz_col]:
            if col not in df.columns:
                raise ValueError(f"Velocity column {col} missing for node {node_id}.")
        vx = df[vx_col].values
        vy = df[vy_col].values
        vz = df[vz_col].values
        return vx, vy, vz

    def get_object_accelerations(self, file_path, node_id):
        df = self.simulations[file_path]
        ax_col = f"Xa{node_id}"
        ay_col = f"Ya{node_id}"
        az_col = f"Za{node_id}"
        for col in [ax_col, ay_col, az_col]:
            if col not in df.columns:
                raise ValueError(f"Acceleration column {col} missing for node {node_id}.")
        ax = df[ax_col].values
        ay = df[ay_col].values
        az = df[az_col].values
        return ax, ay, az

    def find_stationary_nodes(self, file_path, threshold=1e-4):
        num_objects = self._detect_objects(file_path)
        stationary_nodes = {}
        for node_id in range(num_objects):
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            avg_disp = np.mean(displacements)
            if avg_disp < threshold:
                stationary_nodes[node_id] = initial_pos
        return stationary_nodes

    def get_average_displacements(self, file_path):
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
        def generate_harmonic(x, n):
            return np.sin(n * np.pi * x)

        def compute_correlation(x, y):
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
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
            rms_disp = np.sqrt(np.mean(displacements**2))
            pattern.append(rms_disp)

        pattern = np.array(pattern)
        max_disp = np.max(np.abs(pattern))
        if max_disp > 0:
            pattern /= max_disp
        else:
            return np.arange(1, num_harmonics+1), np.zeros(num_harmonics)

        x_positions = np.linspace(0, 1, num_objects)
        harmonics = np.arange(1, num_harmonics+1)
        correlations = []
        for n in harmonics:
            harmonic_pattern = generate_harmonic(x_positions, n)
            corr = compute_correlation(pattern, harmonic_pattern)
            correlations.append(abs(corr) * 100)

        return harmonics, np.array(correlations)
