"""Enhanced visualization system with integrated force handling."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
from enum import Enum
from typing import List
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
import numpy as np
from typing import Optional

class ForceHandler:
    """Handles force application and state management."""

    def __init__(self, physics, objects, dark_mode=True):
        """
        Initialize the force handler with default settings.

        Args:
            physics: The physics engine or model handling forces and motion.
            objects: List of objects in the simulation.
            dark_mode: Whether to use dark mode for UI elements.
        """
        self.physics = physics  # Reference to the physics model.
        self.objects = objects  # List of simulation objects.
        self.dark_mode = dark_mode  # Flag for UI theme.

        # Force application state
        self.active = False  # Whether a force is currently active.
        self.continuous = False  # Whether the force is continuous or single-shot.
        self.duration = 1.0  # Duration of force application in seconds.
        self.duration_remaining = 1.0 # Duration remaining in force application
        self.amplitude = 10.0  # Magnitude of the force.
        self.start_time = None  # Timestamp of when the force started.
        self.selected_object = len(objects) // 2  # Default target object (middle one).

        # Available force types and directions
        self.types = {
            'Single Mass': lambda t, x: np.array([0.0, 0.0, 1.0]),
            'Sinusoidal': lambda t, x: np.array([0.0, 0.0, np.sin(2 * np.pi * t)]),
            'Gaussian': lambda t, x: np.array(
                [0.0, 0.0, np.exp(-(x - len(self.objects) // 2) ** 2 / (len(self.objects) / 8) ** 2)]
            )
        }
        self.directions = {
            'Up/Down (Z)': np.array([0.0, 0.0, 1.0]),
            'Left/Right (X)': np.array([1.0, 0.0, 0.0]),
            'Front/Back (Y)': np.array([0.0, 1.0, 0.0])
        }
        self.selected_type = 'Single Mass'  # Default force type.
        self.selected_direction = 'Up/Down (Z)'  # Default force direction.

    def toggle(self):
        """
        Toggle the force state between active and inactive.

        Returns:
            A tuple with the new label and color for the force button, or None.
        """
        import time

        if self.continuous:
            # Continuous force toggles between locked and unlocked states.
            self.active = not self.active
            if self.active:
                self.start_time = time.time()
            return ('Force Locked', 'red') if self.active else ('Apply Force', 'darkgray')
        elif not self.active:
            # Single force: activate if not already active.
            self.active = True
            self.start_time = time.time()
            return ('Force Active', 'lightgreen')
        return None

    def check_duration(self):
        """
        Check if the force duration has expired.

        Uses the system clock to determine if the elapsed time since force activation
        exceeds the set duration. Deactivates the force if necessary.

        Returns:
            True if the force was deactivated, False otherwise.
        """
        import time

        if not self.continuous and self.active:
            elapsed = time.time() - self.start_time
            if (self.duration - elapsed) >= 0:
                self.duration_remaining = self.duration - elapsed
            else:
                self.duration_remaining = 0
            if elapsed >= self.duration:
                self.deactivate()
                return True
        return False
        self.duration_remaining = self.duration

    def apply(self, simulation_time):
        """
        Applies the selected force to the simulation.

        Args:
            simulation_time: Current time in the simulation (used for time-dependent forces).
        """
        if not self.active:
            return  # Do nothing if the force is inactive.

        # Determine the direction and magnitude of the force.
        direction = self.directions[self.selected_direction]
        duration = None if self.continuous else self.duration

        if self.selected_type == 'Single Mass':
            # Constant force applied to a single object.
            force = direction * self.amplitude
            self.physics.apply_force(self.selected_object, force, duration)
        elif self.selected_type == 'Sinusoidal':
            # Sinusoidal force varying with simulation time.
            magnitude = np.sin(2 * np.pi * simulation_time) * self.amplitude
            self.physics.apply_force(self.selected_object, direction * magnitude, duration)
        elif self.selected_type == 'Gaussian':
            # Gaussian-distributed forces applied to all objects.
            for i in range(1, len(self.objects) - 1):
                magnitude = np.exp(-(i - self.selected_object) ** 2 / (len(self.objects) / 8) ** 2) * self.amplitude
                self.physics.apply_force(i, direction * magnitude, duration)

    def deactivate(self):
        """
        Deactivate all forces and reset the state.
        """
        self.active = False
        self.start_time = None
        for obj_id in range(len(self.objects)):
            self.physics.external_forces[obj_id] = np.zeros(3)

class ViewPreset(Enum):
    """Predefined view angles for the simulation."""
    DEFAULT = {"name": "Default", "azim": 45, "elev": 15}
    TOP = {"name": "Top (XY)", "azim": 0, "elev": 90}
    FRONT = {"name": "Front (XZ)", "azim": 0, "elev": 0}
    SIDE = {"name": "Side (YZ)", "azim": 90, "elev": 0}
    ISOMETRIC = {"name": "Isometric", "azim": 45, "elev": 35}
    FREE = {"name": "Free", "azim": None, "elev": None}

@dataclass
class SimulationParameters:
    """Parameters for string simulation configuration."""
    num_segments: int = 50  # Number of segments in the string
    spring_constant: float = 1000.0  # Spring constant (k)
    mass: float = 0.01  # Mass of each point
    dt: float = 0.0001  # Time step
    start_point: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 0.0]))
    end_point: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    integration_method: str = 'leapfrog'
    applied_force: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    dark_mode: bool = True

class StringSimulationSetup:
    """Setup GUI for string simulation configuration."""

    def __init__(self):
        """Initialize the setup GUI."""
        self.root = tk.Tk()
        self.root.title("String Simulation Setup")

        # Calculate window size - 40% width, 60% height of screen
        window_width = int(self.root.winfo_screenwidth() * 0.4)
        window_height = int(self.root.winfo_screenheight() * 0.6)
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Initialize variables
        self.num_segments_var = tk.IntVar(value=25)
        self.spring_constant_var = tk.DoubleVar(value=1000.0)
        self.mass_var = tk.DoubleVar(value=0.01)
        self.dt_var = tk.DoubleVar(value=0.0001)
        self.integration_var = tk.StringVar(value='leapfrog')
        self.force_magnitude_var = tk.DoubleVar(value=1.0)
        self.dark_mode_var = tk.BooleanVar(value=True)

        # Store the final parameters
        self.simulation_params = None

        # Setup the GUI elements
        self.setup_gui()

    def setup_gui(self):
        """Setup the main GUI elements."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(sticky="nsew")

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(header_frame, text="String Simulation Setup",
                  font=("Arial", 14, "bold")).pack()
        ttk.Label(header_frame,
                  text="Configure simulation parameters",
                  font=("Arial", 10)).pack()

        # Parameters notebook
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky="nsew", pady=5)

        # Basic parameters tab
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Parameters")
        self.setup_basic_parameters(basic_frame)

        # Advanced parameters tab
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text="Advanced Parameters")
        self.setup_advanced_parameters(advanced_frame)

        # Start button
        start_button = ttk.Button(
            main_frame,
            text="Start Simulation",
            command=self.start_simulation,
            style="Accent.TButton"
        )
        start_button.grid(row=2, column=0, pady=10)

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def setup_basic_parameters(self, parent):
        """Setup basic simulation parameters."""
        # String properties
        props_frame = ttk.LabelFrame(parent, text="String Properties", padding="10")
        props_frame.pack(fill="x", padx=5, pady=5)

        # Number of segments
        ttk.Label(props_frame, text="Number of segments:").grid(row=0, column=0, padx=5, pady=5)
        segments_entry = ttk.Entry(props_frame, textvariable=self.num_segments_var, width=10)
        segments_entry.grid(row=0, column=1, padx=5, pady=5)

        # Spring constant
        ttk.Label(props_frame, text="Spring constant (N/m):").grid(row=1, column=0, padx=5, pady=5)
        spring_entry = ttk.Entry(props_frame, textvariable=self.spring_constant_var, width=10)
        spring_entry.grid(row=1, column=1, padx=5, pady=5)

        # Mass per point
        ttk.Label(props_frame, text="Mass per point (kg):").grid(row=2, column=0, padx=5, pady=5)
        mass_entry = ttk.Entry(props_frame, textvariable=self.mass_var, width=10)
        mass_entry.grid(row=2, column=1, padx=5, pady=5)

        # Integration method
        method_frame = ttk.LabelFrame(parent, text="Integration Method", padding="10")
        method_frame.pack(fill="x", padx=5, pady=5)

        methods = ['euler', 'euler_cromer', 'rk2', 'leapfrog', 'rk4']
        for i, method in enumerate(methods):
            ttk.Radiobutton(
                method_frame,
                text=method.replace('_', ' ').title(),
                value=method,
                variable=self.integration_var
            ).grid(row=i, column=0, padx=5, pady=2, sticky="w")

        # Add method descriptions
        descriptions = {
            'euler': "Simple first-order method (fastest but least accurate)",
            'euler_cromer': "Modified Euler method with better energy conservation",
            'rk2': "Second-order Runge-Kutta method",
            'leapfrog': "Symplectic method with good energy conservation",
            'rk4': "Fourth-order Runge-Kutta method (most accurate but slowest)"
        }

        for i, method in enumerate(methods):
            ttk.Label(
                method_frame,
                text=descriptions[method],
                font=("Arial", 8),
                foreground="gray"
            ).grid(row=i, column=1, padx=5, pady=2, sticky="w")

    def setup_advanced_parameters(self, parent):
        """Setup advanced simulation parameters."""
        # Time step configuration
        time_frame = ttk.LabelFrame(parent, text="Time Settings", padding="10")
        time_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(time_frame, text="Time step (dt):").grid(row=0, column=0, padx=5, pady=5)
        dt_entry = ttk.Entry(time_frame, textvariable=self.dt_var, width=10)
        dt_entry.grid(row=0, column=1, padx=5, pady=5)

        # Force configuration
        force_frame = ttk.LabelFrame(parent, text="Force Settings", padding="10")
        force_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(force_frame, text="Force magnitude (N):").grid(row=0, column=0, padx=5, pady=5)
        force_entry = ttk.Entry(force_frame, textvariable=self.force_magnitude_var, width=10)
        force_entry.grid(row=0, column=1, padx=5, pady=5)

        # Display settings
        display_frame = ttk.LabelFrame(parent, text="Display Settings", padding="10")
        display_frame.pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(
            display_frame,
            text="Dark Mode",
            variable=self.dark_mode_var
        ).pack(padx=5, pady=5)

        # Add tooltips and help text
        help_frame = ttk.Frame(parent)
        help_frame.pack(fill="x", padx=5, pady=10)
        ttk.Label(
            help_frame,
            text="Tip: Smaller time steps give more accurate results but run slower",
            font=("Arial", 8),
            foreground="gray"
        ).pack(pady=2)

    def validate_parameters(self):
        """Validate all simulation parameters."""
        try:
            num_segments = self.num_segments_var.get()
            if num_segments < 2:
                raise ValueError("Number of segments must be at least 2")

            spring_constant = self.spring_constant_var.get()
            if spring_constant <= 0:
                raise ValueError("Spring constant must be positive")

            mass = self.mass_var.get()
            if mass <= 0:
                raise ValueError("Mass must be positive")

            dt = self.dt_var.get()
            if dt <= 0:
                raise ValueError("Time step must be positive")

            force_magnitude = self.force_magnitude_var.get()
            if force_magnitude < 0:
                raise ValueError("Force magnitude cannot be negative")

            return True

        except tk.TkError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values")
            return False
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return False

    def start_simulation(self):
        """Start the simulation with configured parameters."""
        if not self.validate_parameters():
            return

        # Create simulation parameters
        self.simulation_params = SimulationParameters(
            num_segments=self.num_segments_var.get(),
            spring_constant=self.spring_constant_var.get(),
            mass=self.mass_var.get(),
            dt=self.dt_var.get(),
            integration_method=self.integration_var.get(),
            applied_force=np.array([0.0, 0.0, self.force_magnitude_var.get()]),
            dark_mode=self.dark_mode_var.get()
        )

        # Close the setup window
        self.root.quit()
        self.root.destroy()

    def get_parameters(self):
        """Run the GUI and return the simulation parameters."""
        self.root.mainloop()
        return self.simulation_params

def setup_simulation():
    """Create and run the simulation setup GUI."""
    setup = StringSimulationSetup()
    return setup.get_parameters()

class SimulationVisualizer:
    """Main visualization class handling display and interaction."""

    def __init__(self, physics_model, objects: List, dark_mode: bool = True, integration_method: str = 'leapfrog'):
        """
        Initialize the visualizer with default settings and store simulation parameters.

        Args:
            physics_model: The physics engine or model managing object dynamics.
            objects: List of objects to visualize.
            dark_mode: Whether to use dark mode for the UI.
            integration_method: The integration method being used (e.g., 'leapfrog', 'rk2').
        """
        self.physics = physics_model
        self.objects = objects
        self.dark_mode = dark_mode
        self.paused = False
        self.rotating = False
        self.panning = False
        self.plots = {}
        self.simulation_time = 0.0
        self.iteration_count = 100

        # Store the integration method name directly
        self.integration_method = integration_method

        # Save initial object states
        self.original_positions = [obj.position.copy() for obj in objects]
        self.original_velocities = [obj.velocity.copy() for obj in objects]
        self.initial_physics_time = physics_model.time

        # Initialize the force handler
        self.force_handler = ForceHandler(physics_model, objects, dark_mode)

        # Camera settings
        self.camera = {
            'distance': 2.0,
            'azimuth': 45,
            'elevation': 15,
            'rotation_speed': 0.3,
            'zoom_speed': 0.1,
            'target': np.zeros(3)
        }

        # Set up the visualization environment
        self.setup_visualization()

    def setup_visualization(self):
        """Set up visualization window and controls."""
        # Apply the appropriate matplotlib style (dark or light mode)
        plt.style.use('dark_background' if self.dark_mode else 'default')

        # Create the figure for the 3D plot
        self.fig = plt.figure(figsize=(12, 10))

        # Add a 3D subplot and adjust its position on the canvas
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.20, 0.15, 0.6, 0.8])

        # Initialize plot elements, camera settings, and control panels
        self.setup_plots()
        self.setup_camera()
        self.setup_enhanced_controls()

        # Connect user interactions (mouse and keyboard events) to appropriate methods
        self._connect_events()

    def setup_enhanced_controls(self):
        """Set up control panel with force controls and buttons."""
        btn_color = 'darkgray' if not self.dark_mode else 'gray'
        text_color = 'white' if self.dark_mode else 'black'

        # Return to Setup button in top left corner with padding
        self.setup_button = Button(
            plt.axes([0.02, 0.94, 0.12, 0.04]),  # [left, bottom, width, height]
            'Return to Setup',
            color=btn_color
        )
        self.setup_button.on_clicked(self.return_to_setup)

        # Left side panel for force controls (keep existing position)
        left_panel_start = 0.07
        panel_width = 0.12

        # Add simulation speed slider at the bottom (keep existing position)
        self.speed_slider = Slider(
            plt.axes([0.24, 0.02, 0.44, 0.02]),
            'Simulation Speed',
            1, 1000,
            valinit=self.iteration_count,
            valfmt='%d steps/frame'
        )
        self.speed_slider.on_changed(self.set_simulation_speed)

        # Move the remaining control buttons up slightly
        button_configs = [
            ('play_button', 'Pause', 0.24),
            ('reset_button', 'Reset', 0.35),
            ('view_button', 'View: Default', 0.46),
            ('zoom_button', 'Zoom: Fit All', 0.57),
            ('theme_button', 'Theme', 0.68)
            # Removed setup_button from here since it's now in top left
        ]

        for btn_name, label, x_pos in button_configs:
            btn = Button(plt.axes([x_pos, 0.06, 0.1, 0.04]), label, color=btn_color)
            setattr(self, btn_name, btn)

        # Connect button callbacks
        self.play_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset_simulation)
        self.view_button.on_clicked(self.cycle_view)
        self.zoom_button.on_clicked(self.cycle_zoom)
        self.theme_button.on_clicked(self.toggle_theme)

        # Force controls header (adjust position to avoid overlap with setup button)
        self.fig.text(left_panel_start, 0.9, 'Force Controls', color=text_color, fontsize=10)

        # Rest of the controls remain the same
        self.force_radio = RadioButtons(
            plt.axes([left_panel_start, 0.72, panel_width, 0.15]),
            list(self.force_handler.types.keys())
        )
        self.force_radio.on_clicked(self.set_force_type)

        self.fig.text(left_panel_start, 0.65, 'Direction:', color=text_color, fontsize=10)
        self.direction_radio = RadioButtons(
            plt.axes([left_panel_start, 0.47, panel_width, 0.15]),
            list(self.force_handler.directions.keys())
        )
        self.direction_radio.on_clicked(self.set_force_direction)

        self.object_slider = Slider(
            plt.axes([left_panel_start, 0.40, panel_width, 0.02]),
            'Object',
            1, len(self.objects) - 2,
            valinit=self.force_handler.selected_object,
            valfmt='%d'
        )
        self.object_slider.on_changed(self.set_selected_object)

        self.amplitude_slider = Slider(
            plt.axes([left_panel_start, 0.35, panel_width, 0.02]),
            'Amplitude',
            0.1, 20.0,
            valinit=self.force_handler.amplitude
        )
        self.amplitude_slider.on_changed(self.set_force_amplitude)

        self.duration_slider = Slider(
            plt.axes([left_panel_start, 0.30, panel_width, 0.02]),
            'Duration',
            0.1, 10.0,
            valinit=self.force_handler.duration,
            valfmt='%.1f s'
        )
        self.duration_slider.on_changed(self.set_force_duration)
        self.duration_slider.on_changed(self.set_force_duration_remaining)

        self.force_button = Button(
            plt.axes([left_panel_start, 0.20, panel_width, 0.04]),
            'Apply Force',
            color=btn_color
        )
        self.force_button.on_clicked(self.toggle_force)

        self.continuous_force_button = Button(
            plt.axes([left_panel_start, 0.15, panel_width, 0.04]),
            'Continuous: Off',
            color=btn_color
        )
        self.continuous_force_button.on_clicked(self.toggle_continuous_force)

    def return_to_setup(self, event):
        """Handle the return to setup button click."""
        self.should_restart = True  # Set the restart flag
        plt.close(self.fig)  # Close the current simulation window

    def setup_plots(self):
        """Initialize plot elements for visualization."""
        for i, obj in enumerate(self.objects):
            scatter = self.ax.scatter([], [], [],
                                      c='red' if obj.pinned else 'blue',
                                      s=50)
            line, = self.ax.plot([], [], [],
                                 color='green' if i < len(self.objects) - 1 else 'none',
                                 linewidth=1)
            self.plots[i] = {'scatter': scatter, 'line': line}

        # Info display
        text_color = 'white' if self.dark_mode else 'black'
        self.info_text = self.ax.text2D(
            1.15, 0.95,
            '',
            transform=self.ax.transAxes,
            color=text_color,
            fontsize=10,
            verticalalignment='top'
        )

    def setup_camera(self):
        """Initialize camera settings."""
        if self.objects:
            positions = np.array([obj.position for obj in self.objects])
            max_dist = np.max(np.abs(positions))
            self.camera['distance'] = max(max_dist * 2.0, 1.0)

        self.ax.view_init(
            elev=self.camera['elevation'],
            azim=self.camera['azimuth']
        )

        dist = self.camera['distance']
        self.ax.set_xlim(-dist, dist)
        self.ax.set_ylim(-dist, dist)
        self.ax.set_zlim(-dist, dist)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True, alpha=0.3)

        self.fig.canvas.draw_idle()

    def set_simulation_speed(self, val):
        """Set the number of physics iterations per frame and update related values."""
        self.iteration_count = int(val)
        # Force an immediate update of the info display when speed changes
        self.update_info()

    def update_frame(self, frame):
        """Update animation frame."""
        if not self.paused:
            # Use the iteration count from the slider
            i = self.iteration_count
            while i > 0:
                self.physics.step()
                i -= 1

            if self.force_handler.check_duration():
                self.force_button.label.set_text('Apply Force')
                self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'
                self.fig.canvas.draw_idle()

            if self.force_handler.continuous and self.force_handler.active:
                self.force_handler.apply(self.simulation_time)

            self.simulation_time += 1

        self.update_plots()
        self.update_info()
        self.highlight_selected_object()

        return self.plots.values()

    def update_plots(self):
        """Update positions of all plot elements."""
        for i, obj in enumerate(self.objects):
            plots = self.plots[i]
            plots['scatter']._offsets3d = ([obj.position[0]],
                                           [obj.position[1]],
                                           [obj.position[2]])

            if i < len(self.objects) - 1:
                next_obj = self.objects[i + 1]
                plots['line'].set_data([obj.position[0], next_obj.position[0]],
                                       [obj.position[1], next_obj.position[1]])
                plots['line'].set_3d_properties([obj.position[2], next_obj.position[2]])

    def update_info(self):
        """Update information display with enhanced simulation details."""
        # Calculate effective dt per frame
        dt_per_frame = self.physics.dt * self.iteration_count

        # Format the integration method name for display
        method_name = self.integration_method.replace('_', ' ').title()

        info_text = (
            f"View: {self.view_button.label.get_text().split(': ')[1]}\n"
            f"{'PAUSED' if self.paused else 'RUNNING'}\n"
            f"Integration: {method_name}\n"
            f"dt/step: {self.physics.dt:.6f}s\n"
            f"dt/frame: {dt_per_frame:.6f}s\n"
            f"Steps per Frame: {self.iteration_count}\n"
            f"Selected Object: {self.force_handler.selected_object}\n"
            f"Force Mode: {'Continuous' if self.force_handler.continuous else 'Single'}\n"
            f"Force Status: {'Active' if self.force_handler.active else 'Inactive'}\n"
            f"Force Type: {self.force_handler.selected_type}\n"
            f"Duration: {self.force_handler.duration_remaining:.1f}s"
        )
        self.info_text.set_text(info_text)

    def highlight_selected_object(self):
        """Update object highlighting based on selection."""
        for i, obj in enumerate(self.objects):
            if 0 < i < len(self.objects) - 1:
                scatter = self.plots[i]['scatter']
                if i == self.force_handler.selected_object:
                    scatter._facecolors[0] = [1.0, 1.0, 0.0, 1.0]  # Yellow
                    scatter._sizes = [100]
                else:
                    scatter._facecolors[0] = [0.0, 0.0, 1.0, 1.0]  # Blue
                    scatter._sizes = [50]
            else:
                self.plots[i]['scatter']._facecolors[0] = [1.0, 0.0, 0.0, 1.0]  # Red
                self.plots[i]['scatter']._sizes = [50]

    def reset_simulation(self, event):
        """Reset simulation to initial state."""
        self.simulation_time = 0.0
        self.force_handler.deactivate()
        self.force_handler.last_application_time = 0.0

        for i, obj in enumerate(self.objects):
            obj.position = self.original_positions[i].copy()
            obj.velocity = self.original_velocities[i].copy()
            if hasattr(obj, 'acceleration'):
                obj.acceleration = np.zeros(3)

        self.force_button.label.set_text('Apply Force')
        self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'
        self.fig.canvas.draw_idle()

    def toggle_force(self, event):
        """Toggle force application."""
        result = self.force_handler.toggle()  # No longer passing simulation_time
        if result:
            label, color = result
            self.force_button.label.set_text(label)
            self.force_button.color = color
            if self.force_handler.active:
                self.force_handler.apply(self.simulation_time)
            self.fig.canvas.draw_idle()

    def toggle_continuous_force(self, event):
        """Toggle continuous force mode."""
        self.force_handler.continuous = not self.force_handler.continuous

        if self.force_handler.continuous:
            self.continuous_force_button.label.set_text('Continuous: On')
            self.continuous_force_button.color = 'lightgreen'
        else:
            self.continuous_force_button.label.set_text('Continuous: Off')
            self.continuous_force_button.color = 'darkgray' if not self.dark_mode else 'gray'
            self.force_handler.deactivate()
            self.force_button.label.set_text('Apply Force')
            self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'

        self.fig.canvas.draw_idle()

    # Control update callbacks
    def set_force_type(self, label):
        """Set the type of force to apply."""
        self.force_handler.selected_type = label

    def set_force_direction(self, label):
        """Set the direction of force application."""
        self.force_handler.selected_direction = label

    def set_force_amplitude(self, val):
        """Set force amplitude."""
        self.force_handler.amplitude = val

    def set_force_duration(self, val):
        """Set force duration."""
        self.force_handler.duration = val

    def set_force_duration_remaining(self, val):
        """Set force duration remaining for top right corner."""
        self.force_handler.duration_remaining = val

    def set_selected_object(self, val):
        """Set selected object and update highlighting."""
        self.force_handler.selected_object = int(val)
        self.highlight_selected_object()
        self.fig.canvas.draw_idle()

    # Camera and view controls
    def toggle_pause(self, event):
        """Toggle simulation pause state."""
        self.paused = not self.paused
        self.play_button.label.set_text('Resume' if self.paused else 'Pause')

    def cycle_view(self, event):
        """Cycle through predefined views."""
        current = self.view_button.label.get_text().split(': ')[1]
        views = list(ViewPreset)
        current_idx = next((i for i, v in enumerate(views)
                            if v.value["name"] == current), 0)
        next_idx = (current_idx + 1) % len(views)
        next_view = views[next_idx].value

        self.camera['azimuth'] = next_view["azim"]
        self.camera['elevation'] = next_view["elev"]
        self.view_button.label.set_text(f'View: {next_view["name"]}')
        self.update_camera()

    def cycle_zoom(self, event):
        """Cycle through zoom levels."""
        zoom_levels = {
            "Fit All": self.calculate_fit_distance(),
            "Close": 0.5,
            "Medium": 2.0,
            "Far": 5.0
        }

        current = self.zoom_button.label.get_text().split(': ')[1]
        levels = list(zoom_levels.keys())
        current_idx = levels.index(current) if current in levels else 0
        next_level = levels[(current_idx + 1) % len(levels)]

        self.camera['distance'] = zoom_levels[next_level]
        self.zoom_button.label.set_text(f'Zoom: {next_level}')
        self.update_camera()

    def calculate_fit_distance(self):
        """Calculate distance needed to fit all objects in view."""
        if not self.objects:
            return 2.0
        positions = np.array([obj.position for obj in self.objects])
        max_dist = np.max(np.abs(positions))
        return max_dist * 2.0

    def update_camera(self):
        """Update camera view settings."""
        self.ax.view_init(
            elev=self.camera['elevation'],
            azim=self.camera['azimuth']
        )

        dist = self.camera['distance']
        self.ax.set_xlim(-dist, dist)
        self.ax.set_ylim(-dist, dist)
        self.ax.set_zlim(-dist, dist)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True, alpha=0.3)

    def toggle_theme(self, event):
        """Toggle between light and dark mode."""
        self.dark_mode = not self.dark_mode
        plt.style.use('dark_background' if self.dark_mode else 'default')

        text_color = 'white' if self.dark_mode else 'black'
        self.info_text.set_color(text_color)
        self.fig.set_facecolor('black' if self.dark_mode else 'white')
        self.ax.set_facecolor('black' if self.dark_mode else 'white')

        btn_color = 'darkgray' if not self.dark_mode else 'gray'
        for btn in [self.play_button, self.view_button,
                    self.zoom_button, self.theme_button]:
            btn.color = btn_color

        self.theme_button.label.set_text(
            'Dark Mode' if not self.dark_mode else 'Light Mode'
        )

        self.fig.canvas.draw_idle()

    # Mouse event handlers
    def _connect_events(self):
        """Connect mouse and keyboard event handlers."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                self.rotating = True
            elif event.button == 3:  # Right click
                self.panning = True
            self.last_x = event.xdata
            self.last_y = event.ydata

            if self.view_button.label.get_text() != 'View: Free':
                self.view_button.label.set_text('View: Free')

    def on_mouse_release(self, event):
        """Handle mouse release events."""
        self.rotating = False
        self.panning = False

    def on_mouse_move(self, event):
        """Handle mouse movement for camera control."""
        if event.inaxes == self.ax and hasattr(self, 'last_x'):
            if self.rotating:
                dx = event.xdata - self.last_x
                dy = event.ydata - self.last_y

                self.camera['azimuth'] = (self.camera['azimuth'] +
                                          dx * self.camera['rotation_speed']) % 360
                self.camera['elevation'] = np.clip(
                    self.camera['elevation'] + dy * self.camera['rotation_speed'],
                    -89,
                    89
                )

                self.last_x = event.xdata
                self.last_y = event.ydata
                self.update_camera()

    def on_scroll(self, event):
        """Handle mouse scroll for zooming."""
        if event.inaxes == self.ax:
            factor = 0.9 if event.button == 'up' else 1.1
            self.camera['distance'] *= factor
            self.zoom_button.label.set_text('Zoom: Custom')
            self.update_camera()


    def animate(self, interval: int = 20):
        """Start the animation."""
        self.anim = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=interval,
            blit=False
        )
        plt.show()
        return self.should_restart  # Return whether we should restart setup