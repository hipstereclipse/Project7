import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimulationDataRecorder:
    """
    This class is used to keep track of simulation data (time, positions, velocities,
    and accelerations) for each timestep. After user is done running the simulation,
    they can save all this recorded data to a CSV file if I want.
    """

    def __init__(self, num_objects: int):
        self.num_objects = num_objects
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []

    def record_step(self, time: float, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray):
        if (len(positions) != self.num_objects or
                len(velocities) != self.num_objects or
                len(accelerations) != self.num_objects):
            raise ValueError("The provided data doesn't match the number of objects.")

        self.time_history.append(time)
        self.position_history.append(positions.copy())
        self.velocity_history.append(velocities.copy())
        self.acceleration_history.append(accelerations.copy())

    def clear_history(self):
        self.time_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()

    def save_to_csv(self, file_path: str):
        if not self.time_history:
            raise ValueError("No data recorded, so I have nothing to save.")

        num_steps = len(self.time_history)
        num_objects = self.num_objects

        columns = ["Time"]
        for n in range(num_objects):
            columns += [f"X{n}", f"Y{n}", f"Z{n}", f"Xv{n}", f"Yv{n}", f"Zv{n}", f"Xa{n}", f"Ya{n}", f"Za{n}"]

        data = np.zeros((num_steps, 1 + 9 * num_objects))
        data[:, 0] = self.time_history

        for i in range(num_steps):
            pos = self.position_history[i]
            vel = self.velocity_history[i]
            acc = self.acceleration_history[i]
            for n in range(num_objects):
                start_col = 1 + n * 9
                data[i, start_col] = pos[n, 0]
                data[i, start_col + 1] = pos[n, 1]
                data[i, start_col + 2] = pos[n, 2]
                data[i, start_col + 3] = vel[n, 0]
                data[i, start_col + 4] = vel[n, 1]
                data[i, start_col + 5] = vel[n, 2]
                data[i, start_col + 6] = acc[n, 0]
                data[i, start_col + 7] = acc[n, 1]
                data[i, start_col + 8] = acc[n, 2]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_path, index=False)

    def get_time_info(self):
        if len(self.time_history) < 2:
            sim_time = 0.0
        else:
            sim_time = self.time_history[-1] - self.time_history[0]
        return {
            'simulation_time': sim_time
        }


class DataAnalysis:
    """
    I use this class after the simulation is done and have CSV files.
    User can load those files into memory, then do various forms of analysis.
    """

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
                "I couldn't find valid object columns."
            )

        return num_objects

    def get_simulation_summary(self, file_path):
        df = self.simulations[file_path]
        time_col = df["Time"]
        num_frames = len(df)
        simulation_time = time_col.iloc[-1] - time_col.iloc[0]
        num_objects = self._detect_objects(file_path)
        stationary_nodes = self.find_stationary_nodes(file_path, threshold_percentage=15)

        return {
            "num_frames": num_frames,
            "simulation_time": simulation_time,
            "num_objects": num_objects,
            "num_stationary_nodes": len(stationary_nodes),
            "stationary_node_positions": stationary_nodes
        }

    def get_object_trajectory(self, file_path, node_id):
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
        if not 0 <= threshold_percentage <= 100:
            raise ValueError("Threshold percentage must be between 0 and 100")

        num_objects = self._detect_objects(file_path)
        stationary_dict = {}

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

        for node_id in range(num_objects):
            if node_displacements[node_id] < threshold:
                stationary_dict[node_id] = node_initial_positions[node_id]

        return stationary_dict

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

    def get_normalized_average_displacements(self, file_path):
        num_objects = self._detect_objects(file_path)
        node_ids = np.arange(num_objects)
        normalized_averages = np.zeros(num_objects)

        for i, node_id in enumerate(node_ids):
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)

            max_disp = np.max(displacements) if np.max(displacements) != 0 else 1.0
            normalized = displacements / max_disp
            normalized_averages[i] = np.mean(normalized)

        return node_ids, normalized_averages

    def get_normalized_amplitudes(self, file_path):
        num_objects = self._detect_objects(file_path)
        node_ids = np.arange(num_objects)
        max_amplitudes = np.zeros(num_objects)

        for i, node_id in enumerate(node_ids):
            x, y, z = self.get_object_trajectory(file_path, node_id)
            initial_pos = np.array([x[0], y[0], z[0]])
            positions = np.column_stack([x, y, z])
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            max_amplitudes[i] = np.max(displacements)

        overall_max = np.max(max_amplitudes)
        normalized_amplitudes = max_amplitudes / (overall_max + 1e-10)

        return node_ids, normalized_amplitudes

    def get_harmonic_correlation(self, file_path, num_harmonics=10):
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
            return np.arange(1, num_harmonics + 1), np.zeros(num_harmonics)

        x_positions = np.linspace(0, 1, num_objects)
        harmonics = np.arange(1, num_harmonics + 1)
        correlations = []

        for n in harmonics:
            harmonic_pattern = generate_harmonic(x_positions, n)
            corr = compute_correlation(pattern, harmonic_pattern)
            correlations.append(abs(corr) * 100)

        return harmonics, np.array(correlations)

    def get_mode_period(self, file_path, node_id, use_velocity=False):
        """
        Measures the period of oscillation of a mode by finding zero-crossings in displacement or velocity
        at a chosen node. This helps estimate the fundamental frequency or a chosen mode's period.

        Args:
            file_path: Path to the simulation data file.
            node_id: The node at which we measure the oscillation (preferably an antinode).
            use_velocity: If True, uses the velocity data to find zero-crossings; else uses displacement.

        Returns:
            float: Estimated period of oscillation.
        """
        df = self.simulations[file_path]
        time = df["Time"].values
        x, y, z = self.get_object_trajectory(file_path, node_id)
        initial_pos = np.array([x[0], y[0], z[0]])
        positions = np.column_stack([x, y, z])
        displacements = np.linalg.norm(positions - initial_pos, axis=1)

        if use_velocity:
            vx, vy, vz = self.get_object_velocities(file_path, node_id)
            # Compute speed along displacement direction or just magnitude
            # For zero cross detection, we can just pick vx or use magnitude per profs instructions:
            speeds = np.sqrt(vx**2 + vy**2 + vz**2)
            data_to_analyze = speeds - np.mean(speeds)  # Center it
        else:
            data_to_analyze = displacements - np.mean(displacements)

        # Finds zero crossings and store them in the zero crossing array
        zero_crossings = []
        for i in range(1, len(data_to_analyze)):
            if data_to_analyze[i-1] < 0 and data_to_analyze[i] >= 0: # found a crossing
                ratio = -data_to_analyze[i-1] / (data_to_analyze[i] - data_to_analyze[i-1]) # uses a linear interpolation for better accuracy
                t_cross = time[i-1] + ratio * (time[i] - time[i-1])
                zero_crossings.append(t_cross)

        # If we find multiple zero crossings, we can measure the period by the average difference
        # between consecutive zero crossings of the same sign pattern.
        # Usually every second zero crossing might represent a full period since it goes up then down.
        if len(zero_crossings) < 2: # Not enough data to determine period
            return None

        # Typically the period is the difference between every second crossing
        # because one zero cross is going up, the next going down, then next going up again.
        # We take the difference between every other zero crossing:
        if len(zero_crossings) > 2:
            intervals = []
            for i in range(2, len(zero_crossings)):
                intervals.append(zero_crossings[i] - zero_crossings[i-2])
            if intervals:
                period = np.mean(intervals)
            else: # fallback if not enough
                period = zero_crossings[-1] - zero_crossings[0]
        else:
            period = 2 * (zero_crossings[1] - zero_crossings[0]) # Just two zero crossings means half a period

        return period

    def plot_mode_period(self, file_path, node_id, use_velocity=False):
        """
        Plots displacement or velocity over time at the given node and highlights one period of oscillation.

        Args:
            file_path: CSV file path of the simulation data.
            node_id: The node to analyze.
            use_velocity: If True, plot velocity data, otherwise displacement data.
        """
        df = self.simulations[file_path]
        time = df["Time"].values
        x, y, z = self.get_object_trajectory(file_path, node_id)
        initial_pos = np.array([x[0], y[0], z[0]])
        positions = np.column_stack([x, y, z])
        displacements = np.linalg.norm(positions - initial_pos, axis=1)

        if use_velocity:
            vx, vy, vz = self.get_object_velocities(file_path, node_id)
            data_to_analyze = np.sqrt(vx**2 + vy**2 + vz**2)
            ylabel = "Speed"
        else:
            data_to_analyze = displacements
            ylabel = "Displacement"

        period = self.get_mode_period(file_path, node_id, use_velocity=use_velocity)

        plt.figure(figsize=(10,6))
        plt.title(f"Oscillation at Node {node_id}{' (Velocity)' if use_velocity else ''}")
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)

        plt.plot(time, data_to_analyze, label="Oscillation")

        if period is not None:
            # Mark one period interval
            start_t = time[0]
            end_t = start_t + period
            plt.axvspan(start_t, end_t, color='yellow', alpha=0.3, label=f"One period ~ {period:.4f}s")

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
