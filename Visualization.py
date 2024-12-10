"""Enhanced visualization system with integrated force handling."""
import os
import time
import tkinter as tk
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from tkinter import ttk, messagebox
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons

from data_handling import DataAnalysis


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
        self.physics = physics  # Reference to the physics model
        self.objects = objects  # List of simulation objects
        self.dark_mode = dark_mode  # Flag for UI theme

        # Force application state
        self.active = False  # Whether a force is currently active
        self.continuous = False  # Whether the force is continuous or single-shot
        self.duration = 0.01  # Duration of force application in seconds
        self.duration_remaining = 0.01  # Duration remaining in force application
        self.amplitude = 10.0  # Magnitude of the force
        self.selected_object = len(objects) // 2  # Default target object (middle one)

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
        self.selected_type = 'Single Mass'  # Default force type
        self.selected_direction = 'Up/Down (Z)'  # Default force direction

    def check_duration(self, iterations_per_frame):
        """
        Check if the force duration has expired and update remaining duration.

        Args:
            iterations_per_frame: Number of physics steps per animation frame

        Returns:
            True if the force was deactivated, False otherwise.
        """
        if not self.continuous and self.active:
            # Calculate actual time elapsed in this frame
            time_elapsed = self.physics.dt * iterations_per_frame

            # Update remaining duration
            self.duration_remaining = max(0.0, self.duration_remaining - time_elapsed)

            # Check if duration has expired
            if self.duration_remaining <= 0:
                self.deactivate()
                return True

            # Apply force with remaining duration
            self.apply(self.duration_remaining)
        return False

    def toggle(self):
        """
        Toggle the force state between active and inactive.
        If force is already active, cancels it immediately.

        Returns:
            A tuple with the new label and color for the force button, or None.
        """
        if self.active:
            # If force is active, cancel it immediately
            self.deactivate()
            return ('Apply Force', 'darkgray')
        elif self.continuous:
            # Toggle continuous force state
            self.active = True
            return ('Force Locked', 'red')
        else:
            # Initialize a new single-shot force
            self.active = True
            self.duration_remaining = self.duration  # Reset duration for new force
            self.apply(self.duration)
            return ('Force Active', 'lightgreen')

        return None

    def apply(self, duration=None):
        """
        Apply force based on current settings.

        Args:
            duration: Duration for the force application, or None for continuous forces.
            Defaults to None which is appropriate for continuous forces.
        """
        direction = self.directions[self.selected_direction]

        if self.selected_type == 'Single Mass':
            # Apply constant force to selected mass
            force = direction * self.amplitude
            self.physics.apply_force(self.selected_object, force, duration)

        elif self.selected_type == 'Sinusoidal':
            # Apply time-varying sinusoidal force
            magnitude = np.sin(2 * np.pi * self.physics.time) * self.amplitude
            self.physics.apply_force(self.selected_object, direction * magnitude, duration)

        elif self.selected_type == 'Gaussian':
            # Apply spatially distributed Gaussian force across multiple masses
            for i in range(1, len(self.objects) - 1):
                # Calculate Gaussian distribution centered on selected object
                magnitude = np.exp(-(i - self.selected_object) ** 2 / (len(self.objects) / 8) ** 2) * self.amplitude
                self.physics.apply_force(i, direction * magnitude, duration)

    def deactivate(self):
        """
        Deactivate all forces and reset the force state.
        Also resets the duration remaining to prevent any lingering effects.
        """
        self.active = False
        self.duration_remaining = 0.0  # Immediately end any ongoing force

        # Clear forces from all objects
        for obj_id in range(len(self.objects)):
            self.physics.clear_force(obj_id)

class StringSimulationSetup:
    """Setup GUI for string simulation configuration."""

    def __init__(self, main_root):
        """
        Initialize the setup GUI with proper variable handling.

        Args:
            main_root: Reference to the main menu window
        """
        self.main_root = main_root
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Setup")

        # Create SimulationParameters instance first
        self.default_params = SimulationParameters()

        # Initialize variables
        self.init_variables()

        # Set minimum window size
        self.root.minsize(600, 500)

        # Default to 40% of screen width, 60% of screen height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.4)
        window_height = int(screen_height * 0.6)

        # Center the window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 4) - (window_height // 4)

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create responsive styles
        self.create_responsive_styles()

        # Setup the GUI elements
        self.setup_gui()

        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)

    def init_variables(self):
        """Initialize GUI variables using defaults from SimulationParameters."""
        # Create tkinter variables and set their values from default_params
        self.num_segments_var = tk.IntVar(self.root)
        self.num_segments_var.set(self.default_params.num_segments)

        self.spring_constant_var = tk.DoubleVar(self.root)
        self.spring_constant_var.set(self.default_params.spring_constant)

        self.mass_var = tk.DoubleVar(self.root)
        self.mass_var.set(self.default_params.mass)

        self.dt_var = tk.DoubleVar(self.root)
        self.dt_var.set(self.default_params.dt)

        # Extract force magnitude from the applied_force vector
        force_magnitude = float(np.linalg.norm(self.default_params.applied_force))
        self.force_magnitude_var = tk.DoubleVar(self.root)
        self.force_magnitude_var.set(force_magnitude)

        self.integration_var = tk.StringVar(self.root)
        self.integration_var.set(self.default_params.integration_method)

        self.dark_mode_var = tk.BooleanVar(self.root)
        self.dark_mode_var.set(self.default_params.dark_mode)

        # Initialize simulation_params as None - will be set when starting simulation
        self.simulation_params = None

    def create_responsive_styles(self):
        """Create TTK styles that adapt to window size."""
        style = ttk.Style()

        # Base font sizes
        self.base_header_size = 14
        self.base_text_size = 10
        self.base_button_size = 11

        # Create initial styles
        style.configure(
            "Header.TLabel",
            font=("Arial", self.base_header_size, "bold"),
            anchor="center"
        )
        style.configure(
            "Normal.TLabel",
            font=("Arial", self.base_text_size),
            anchor="w"
        )
        style.configure(
            "Setup.TButton",
            font=("Arial", self.base_button_size),
            padding=10
        )

    def on_window_resize(self, event):
        """Update styles based on window size."""
        if event.widget == self.root:
            # Calculate scale factor
            width_scale = event.width / (self.root.winfo_screenwidth() * 0.4)
            height_scale = event.height / (self.root.winfo_screenheight() * 0.6)
            scale = min(width_scale, height_scale)

            # Update font sizes
            style = ttk.Style()
            style.configure("Header.TLabel",
                            font=("Arial", int(self.base_header_size * scale), "bold"))
            style.configure("Normal.TLabel",
                            font=("Arial", int(self.base_text_size * scale)))
            style.configure("Setup.TButton",
                            font=("Arial", int(self.base_button_size * scale)))

            # Update padding
            base_padding = 10
            scaled_padding = int(base_padding * scale)
            style.configure("Setup.TButton", padding=scaled_padding)

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
        ttk.Entry(props_frame, textvariable=self.num_segments_var, width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        # Spring constant
        ttk.Label(props_frame, text="Spring constant (N/m):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(props_frame, textvariable=self.spring_constant_var, width=10).grid(
            row=1, column=1, padx=5, pady=5
        )

        # Mass per point
        ttk.Label(props_frame, text="Mass per point (kg):").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(props_frame, textvariable=self.mass_var, width=10).grid(
            row=2, column=1, padx=5, pady=5
        )

        # Display settings
        display_frame = ttk.LabelFrame(parent, text="Display Settings", padding="10")
        display_frame.pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(
            display_frame,
            text="Dark Mode",
            variable=self.dark_mode_var
        ).pack(padx=5, pady=5)

    def setup_advanced_parameters(self, parent):
        """Setup advanced simulation parameters."""

        # Force configuration
        force_frame = ttk.LabelFrame(parent, text="Force Settings", padding="10")
        force_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(force_frame, text="Force magnitude (N):").grid(row=0, column=0, padx=5, pady=5)
        force_entry = ttk.Entry(force_frame, textvariable=self.force_magnitude_var, width=10)
        force_entry.grid(row=0, column=1, padx=5, pady=5)

        # Integration method
        method_frame = ttk.LabelFrame(parent, text="Integration Method", padding="10")
        method_frame.pack(fill="x", padx=5, pady=5)

        methods = ['euler', 'euler_cromer', 'rk2', 'leapfrog', 'rk4']
        descriptions = {
            'euler': "Simple first-order method (fastest but least accurate)",
            'euler_cromer': "Modified Euler method with better energy conservation",
            'rk2': "Second-order Runge-Kutta method",
            'leapfrog': "Symplectic method with good energy conservation",
            'rk4': "Fourth-order Runge-Kutta method (most accurate but slowest)"
        }

        for i, method in enumerate(methods):
            frame = ttk.Frame(method_frame)
            frame.grid(row=i, column=0, sticky="w", padx=5, pady=2)

            ttk.Radiobutton(
                frame,
                text=method.replace('_', ' ').title(),
                value=method,
                variable=self.integration_var
            ).pack(side=tk.LEFT)

            ttk.Label(
                frame,
                text=descriptions[method],
                font=("Arial", 8),
                foreground="gray"
            ).pack(side=tk.LEFT, padx=5)

        # Time step configuration
        time_frame = ttk.LabelFrame(parent, text="Time Settings", padding="10")
        time_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(time_frame, text="Time step (dt):").grid(row=0, column=0, padx=5, pady=5)
        dt_entry = ttk.Entry(time_frame, textvariable=self.dt_var, width=10)
        dt_entry.grid(row=0, column=1, padx=5, pady=5)

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

        except ValueError as e:
            # Show the error as a popup
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
        self.paused = True  # Start paused by default for data collection purposes
        self.simulation_started = False  # Tracks if simulation has begun
        self.rotating = False
        self.panning = False
        self.plots = {}
        self.simulation_time = 0.0
        self.iteration_count = 100

        # Store the integration method name directly
        self.integration_method = integration_method

        # Add the should_restart flag initialization
        self.should_restart = False

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

        # Add force info text display
        self.force_info_text = self.ax.text2D(
            1.15, 0.55,  # Position it below the regular info text
            '',
            transform=self.ax.transAxes,
            color='white' if self.dark_mode else 'black',
            fontsize=10,
            verticalalignment='top'
        )

        # Add FPS tracking variables
        self.last_frame_time = time.time()
        self.fps = 0
        self.fps_update_interval = 0.5  # Update FPS every half second
        self.frame_times = []  # Store recent frame times for averaging
        self.max_frame_times = 30  # Number of frames to average
        self.animation_frame_count = 0
        self.main_root = None  # Will store reference to main menu window

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
        """Set up control panel with force controls, buttons, and data saving."""
        btn_color = 'darkgray' if not self.dark_mode else 'gray'
        text_color = 'white' if self.dark_mode else 'black'

        # Return to Setup button in top left corner with padding
        self.setup_button = Button(
            plt.axes([0.02, 0.94, 0.12, 0.04]),
            'Return to Setup',
            color=btn_color
        )
        self.setup_button.on_clicked(self.return_to_setup)

        # Left side panel for force controls
        left_panel_start = 0.07
        panel_width = 0.12

        # Simulation speed slider at the bottom
        self.speed_slider = Slider(
            plt.axes([0.24, 0.02, 0.44, 0.02]),
            'Simulation Speed',
            1, 1000,
            valinit=self.iteration_count,
            valfmt='%d steps/frame'
        )
        self.speed_slider.on_changed(self.set_simulation_speed)

        # Control buttons
        button_configs = [
            ('play_button', 'Pause', 0.24),
            ('reset_button', 'Reset', 0.35),
            ('view_button', 'View: Default', 0.46),
            ('zoom_button', 'Zoom: Fit All', 0.57),
            ('theme_button', 'Theme', 0.68),
            ('save_button', 'Save Data', 0.79)  # Added save button
        ]

        # Create buttons
        for btn_name, label, x_pos in button_configs:
            btn = Button(plt.axes([x_pos, 0.06, 0.1, 0.04]), label, color=btn_color)
            setattr(self, btn_name, btn)

        # Connect button callbacks
        self.play_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset_simulation)
        self.view_button.on_clicked(self.cycle_view)
        self.zoom_button.on_clicked(self.cycle_zoom)
        self.theme_button.on_clicked(self.toggle_theme)
        self.save_button.on_clicked(self.save_simulation_data)

        # Force controls header
        self.fig.text(left_panel_start, 0.9, 'Force Controls', color=text_color, fontsize=10)

        # Force type radio buttons
        self.force_radio = RadioButtons(
            plt.axes([left_panel_start, 0.72, panel_width, 0.15]),
            list(self.force_handler.types.keys())
        )
        self.force_radio.on_clicked(self.set_force_type)

        # Force direction controls
        self.fig.text(left_panel_start, 0.65, 'Direction:', color=text_color, fontsize=10)
        self.direction_radio = RadioButtons(
            plt.axes([left_panel_start, 0.47, panel_width, 0.15]),
            list(self.force_handler.directions.keys())
        )
        self.direction_radio.on_clicked(self.set_force_direction)

        # Object selection slider
        self.object_slider = Slider(
            plt.axes([left_panel_start, 0.40, panel_width, 0.02]),
            'Object',
            1, len(self.objects) - 2,
            valinit=self.force_handler.selected_object,
            valfmt='%d'
        )
        self.object_slider.on_changed(self.set_selected_object)

        # Force amplitude slider
        self.amplitude_slider = Slider(
            plt.axes([left_panel_start, 0.35, panel_width, 0.02]),
            'Amplitude',
            0.1, 50.0,
            valinit=self.force_handler.amplitude
        )
        self.amplitude_slider.on_changed(self.set_force_amplitude)

        # Force duration slider
        self.duration_slider = Slider(
            plt.axes([left_panel_start, 0.30, panel_width, 0.02]),
            'Duration',
            0.01, 10.0,
            valinit=self.force_handler.duration,
            valfmt='%.2f s'
        )
        self.duration_slider.on_changed(self.set_force_duration)
        self.duration_slider.on_changed(self.set_force_duration_remaining)

        # Force control buttons
        self.force_button = Button(
            plt.axes([left_panel_start, 0.20, panel_width, 0.04]),
            'Apply Force',
            color=btn_color
        )
        self.force_button.on_clicked(self.toggle_force)

        # Continuous force toggle button
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
        if self.main_root:
            self.main_root.deiconify()  # Show main menu

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

    def save_simulation_data(self, event):
        """Handle saving simulation data when save button is clicked."""
        from tkinter import filedialog, messagebox
        import tkinter as tk

        # Pause the simulation first
        if not self.paused:
            self.toggle_pause(None)
            self.play_button.label.set_text('Resume')
            self.fig.canvas.draw_idle()

        # Create temporary root window for file dialog
        root = tk.Tk()
        root.withdraw()

        # Get file name from user
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Simulation Data History"
        )

        if file_path:
            try:
                # Save the recorded data history
                self.physics.data_recorder.save_to_csv(file_path)

                # Show success popup
                messagebox.showinfo(
                    "Success",
                    f"Simulation history has been saved successfully to:\n{file_path}\n\n"
                    f"Total time steps saved: {len(self.physics.data_recorder.time_history)}"
                )

            except Exception as e:
                # Show error popup
                messagebox.showerror(
                    "Error",
                    f"An error occurred while saving the data:\n{str(e)}"
                )

        root.destroy()

    def update_frame(self, frame):
        """Update animation frame with proper timing tracking."""
        # Only update when not paused and simulation has started
        if not self.paused and self.simulation_started:
            self.animation_frame_count += 1

            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)

            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)

            if current_time - self.last_frame_time >= self.fps_update_interval:
                if self.frame_times:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                self.last_frame_time = current_time

            # Run physics steps
            i = self.iteration_count
            while i > 0:
                self.physics.step(self.integration_method)
                i -= 1

            # Check and handle force duration
            if self.force_handler.check_duration(self.iteration_count):
                self.force_button.label.set_text('Apply Force')
                self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'
                self.fig.canvas.draw_idle()

            # Apply continuous force if active
            if self.force_handler.continuous and self.force_handler.active:
                self.force_handler.apply()

        # Always update visualization
        self.update_plots()
        self.update_info()
        self.highlight_selected_object()

        return list(self.plots.values())

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
        # Get time info from data recorder
        time_info = self.physics.data_recorder.get_time_info()

        # Calculate timing information
        dt_per_frame = self.physics.dt * self.iteration_count
        method_name = self.integration_method.replace('_', ' ').title()

        info_text = (
            f"View: {self.view_button.label.get_text().split(': ')[1]}\n"
            f"{'PAUSED' if self.paused else 'RUNNING'}\n"
            f"Animation Frame: {self.animation_frame_count}\n"
            f"FPS: {self.fps:.1f}\n"
            f"Simulation Time: {time_info['simulation_time']:.3f}s\n"
            f"Integration: {method_name}\n"
            f"dt/step: {self.physics.dt:.6f}s\n"
            f"dt/frame: {dt_per_frame:.6f}s\n"
            f"Physics Steps/Frame: {self.iteration_count}\n"
            f"Selected Object: {self.force_handler.selected_object}\n"
            f"Force Mode: {'Continuous' if self.force_handler.continuous else 'Single'}\n"
            f"Force Status: {'Active' if self.force_handler.active else 'Inactive'}\n"
            f"Force Type: {self.force_handler.selected_type}\n"
            f"Duration: {self.force_handler.duration_remaining:.2f}s"
        )
        self.info_text.set_text(info_text)

        # Update force information display
        selected_obj = self.objects[self.force_handler.selected_object]

        # Get external force from physics engine
        external_force = self.physics.external_forces[selected_obj.obj_id]

        # Get spring forces
        spring_forces = np.zeros(3)
        if self.force_handler.selected_object > 0:
            spring_forces += self.physics.compute_spring_force(
                selected_obj.position,
                self.objects[self.force_handler.selected_object - 1].position
            )
        if self.force_handler.selected_object < len(self.objects) - 1:
            spring_forces += self.physics.compute_spring_force(
                selected_obj.position,
                self.objects[self.force_handler.selected_object + 1].position
            )

        # Calculate total force
        total_force = external_force + spring_forces

        # Format force information
        force_text = (
            f"Forces on Mass {self.force_handler.selected_object}:\n"
            f"─────────────────────────\n"
            f"External Force:\n"
            f"  X: {external_force[0]:8.3f} N\n"
            f"  Y: {external_force[1]:8.3f} N\n"
            f"  Z: {external_force[2]:8.3f} N\n"
            f"Spring Forces:\n"
            f"  X: {spring_forces[0]:8.3f} N\n"
            f"  Y: {spring_forces[1]:8.3f} N\n"
            f"  Z: {spring_forces[2]:8.3f} N\n"
            f"Total Force:\n"
            f"  X: {total_force[0]:8.3f} N\n"
            f"  Y: {total_force[1]:8.3f} N\n"
            f"  Z: {total_force[2]:8.3f} N\n"
        )
        self.force_info_text.set_text(force_text)

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
        self.animation_frame_count = 0
        self.fps = 0
        self.frame_times.clear()
        self.last_frame_time = time.time()

        # Reset simulation state flags
        self.simulation_started = False
        self.paused = True
        self.play_button.label.set_text('Start')

        # Reset physics state
        self.physics.time = self.initial_physics_time
        self.physics.simulation_started = False
        self.force_handler.deactivate()
        self.force_handler.last_application_time = 0.0

        # Reset object states
        for i, obj in enumerate(self.objects):
            obj.position = self.original_positions[i].copy()
            obj.velocity = self.original_velocities[i].copy()
            if hasattr(obj, 'acceleration'):
                obj.acceleration = np.zeros(3)

        # Reset force button state
        self.force_button.label.set_text('Apply Force')
        self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'

        # Clear data history
        self.physics.data_recorder.clear_history()

        self.fig.canvas.draw_idle()

    def toggle_force(self, event):
        """Toggle force application and handle simulation start."""
        # If applying force and simulation hasn't started, start it
        if not self.simulation_started and not self.force_handler.active:
            self.simulation_started = True
            self.physics.start_simulation()
            self.paused = False  # Unpause when force is applied
            self.play_button.label.set_text('Pause')

        result = self.force_handler.toggle()
        if result:
            label, color = result
            self.force_button.label.set_text(label)
            self.force_button.color = color

            if self.force_handler.active:
                self.force_handler.apply()

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
        """Toggle simulation pause state and handle simulation start."""
        self.paused = not self.paused

        # If unpausing and simulation hasn't started, start it
        if not self.paused and not self.simulation_started:
            self.simulation_started = True
            self.physics.start_simulation()

        self.play_button.label.set_text('Resume' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()

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

        # Update text colors
        text_color = 'white' if self.dark_mode else 'black'
        self.info_text.set_color(text_color)
        self.force_info_text.set_color(text_color)
        self.fig.texts[0].set_color(text_color)  # Force Controls header
        self.fig.texts[1].set_color(text_color)  # Direction label

        # Updates UI elements
        btn_color = 'darkgray' if not self.dark_mode else 'gray'
        for btn in [self.play_button, self.reset_button, self.view_button,
                    self.zoom_button, self.theme_button, self.setup_button,
                    self.force_button, self.continuous_force_button, self.save_button]:  # Added save_button
            btn.color = btn_color
            if hasattr(btn, 'label'):
                btn.label.set_color(text_color)

        # Updates sliders
        for slider in [self.speed_slider, self.object_slider,
                      self.amplitude_slider, self.duration_slider]:
            slider.label.set_color(text_color)
            slider.valtext.set_color(text_color)

        # Updates figure colors
        self.fig.set_facecolor('black' if self.dark_mode else 'white')
        self.ax.set_facecolor('black' if self.dark_mode else 'white')

        # Theme button text
        self.theme_button.label.set_text('Dark Mode' if not self.dark_mode else 'Light Mode')

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
        """
        Start the animation with proper frame caching settings.

        Args:
            interval: Time interval between frames in milliseconds

        Returns:
            Boolean indicating whether to restart the simulation
        """
        self.anim = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=interval,
            blit=False,
            cache_frame_data=False  # Prevent the warning about unbounded cache
        )
        plt.show()
        return self.should_restart

class AnalysisVisualizer:
    """
    Interactive visualization system for analyzing string simulation data.
    Provides flexible analysis capabilities for any number of simulation files.
    """

    def __init__(self, main_root):
        """Initialize the analysis visualization system with enhanced flexibility."""
        from data_handling import DataAnalysis
        import tkinter as tk
        from tkinter import ttk

        # Initialize core components
        self.analyzer = DataAnalysis()  # Data analysis engine
        self.loaded_files = {}  # Maps file paths to their simulation data

        # Set up main window
        self.main_root = main_root  # Store reference to main menu window
        self.root = tk.Toplevel(main_root)  # Create as Toplevel instead of Tk
        self.root.title("String Simulation Analysis")

        # Calculate and set window dimensions
        window_width = int(self.root.winfo_screenwidth() * 0.55)
        window_height = int(self.root.winfo_screenheight() * 0.40)
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Define color palette for visualization
        self.colors = ["red", "green", "blue", "yellow", "orange", "purple"]

        # Set up the GUI elements
        self.setup_gui()

        # Add protocol for window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        """Set up the main GUI elements with streamlined layout and improved organization."""
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # File management section
        file_frame = ttk.LabelFrame(self.main_frame, text="File Management", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", pady=5)

        # File management buttons with improved layout
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(button_frame, text="Load Files", command=self.load_files).grid(
            row=0, column=0, padx=5, sticky="w"
        )

        delete_frame = ttk.Frame(button_frame)
        delete_frame.grid(row=0, column=2, sticky="e")

        ttk.Button(delete_frame, text="Delete Selected", command=self.delete_selected).pack(
            side="left", padx=5
        )
        ttk.Button(delete_frame, text="Clear All", command=self.clear_files).pack(
            side="left", padx=5
        )

        # Create treeview for file list
        self.file_tree = ttk.Treeview(file_frame, show="headings", height=3, selectmode="extended")
        self.file_tree.grid(row=1, column=0, sticky="ew", pady=5)

        # Configure treeview columns
        self.file_tree["columns"] = ("filename", "nodes", "frames", "time", "color")
        columns = [
            ("filename", "Filename", 200, "w"),
            ("nodes", "Nodes", 100, "center"),
            ("frames", "Frames", 120, "center"),
            ("time", "Simulation Time", 150, "center"),
            ("color", "Color", 100, "center")
        ]

        for col, heading, width, anchor in columns:
            self.file_tree.heading(col, text=heading)
            self.file_tree.column(col, width=width, anchor=anchor)

        self.file_tree.bind("<Double-1>", self.cycle_color)

        # Analysis options frame
        analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="5")
        analysis_frame.grid(row=1, column=0, sticky="ew", pady=5)

        # Define analysis buttons in two rows
        row1_buttons = [
            ("View Summary", self.show_summary),
            ("Find Stationary Nodes", self.compare_stationary),
            ("Node Displacement", self.compare_displacement),
            ("Average Displacements", self.plot_nodal_average_displacement),
            ("Movement Patterns", self.compare_movement)
        ]

        row2_buttons = [
            ("Harmonic Analysis", self.analyze_harmonics)
        ]

        # Create buttons for the first row
        for i, (text, command) in enumerate(row1_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=0, column=i, padx=5, pady=5, sticky="ew"
            )
            analysis_frame.columnconfigure(i, weight=1)

        # Create buttons for the second row
        # Calculate the starting column for centering the button
        start_col = (len(row1_buttons) - len(row2_buttons)) // 2
        for i, (text, command) in enumerate(row2_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=1, column=start_col + i, padx=5, pady=5, sticky="ew"
            )

    def load_files(self):
        """Load and automatically select simulation data files."""
        from tkinter import filedialog, messagebox

        # Open file dialog for multiple file selection
        files = filedialog.askopenfilenames(
            title="Select Simulation Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        # Process each selected file
        for file_path in files:
            try:
                # Load the file data using the analyzer
                data = self.analyzer.load_simulation(file_path)
                self.loaded_files[file_path] = data

                # Get file summary information
                summary = self.analyzer.get_simulation_summary(data)

                # Insert into treeview with default color
                item = self.file_tree.insert("", "end", values=(
                    os.path.basename(file_path),
                    summary['num_objects'],
                    summary['num_frames'],
                    f"{summary['simulation_time']:.3f} s",
                    self.colors[len(self.loaded_files) % len(self.colors)]  # Cycle through colors
                ))

                # Automatically select the newly added item
                self.file_tree.selection_add(item)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file_path}:\n{str(e)}")

    def delete_selected(self):
        """Delete selected files from the analysis system."""
        selected_items = self.file_tree.selection()
        if not selected_items:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please select files to delete.")
            return

        files_to_remove = []
        # Build list of files to remove
        for item in selected_items:
            filename = self.file_tree.item(item)["values"][0]
            # Find and store the full path
            for path in self.loaded_files:
                if os.path.basename(path) == filename:
                    files_to_remove.append(path)
                    break
            # Remove from treeview
            self.file_tree.delete(item)

        # Remove from loaded files dictionary
        for file_path in files_to_remove:
            if file_path in self.loaded_files:
                del self.loaded_files[file_path]

    def get_selected_files(self):
        """Get the file paths for visualization, defaulting to all loaded files if none selected."""
        selected_items = self.file_tree.selection()

        if not selected_items:  # If no files are selected, select all files
            selected_items = self.file_tree.get_children()

        selected_files = []
        for item in selected_items:
            filename = self.file_tree.item(item)["values"][0]
            # Find the full path matching this filename
            for path in self.loaded_files:
                if os.path.basename(path) == filename:
                    selected_files.append(path)
                    break

        return selected_files

    def cycle_color(self, event):
        """Cycle through visualization colors for the selected file."""
        item = self.file_tree.identify_row(event.y)
        column = self.file_tree.identify_column(event.x)

        if column == "#5" and item:  # Color column
            values = list(self.file_tree.item(item)["values"])
            if len(values) >= 5:
                current_color = values[4]
                # Find next color in cycle
                try:
                    next_index = (self.colors.index(current_color) + 1) % len(self.colors)
                except ValueError:
                    next_index = 0
                values[4] = self.colors[next_index]
                self.file_tree.item(item, values=values)

    def clear_files(self):
        """Clear all loaded files and reset the analyzer."""
        self.loaded_files.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self.analyzer = DataAnalysis()

    def compare_movement(self):
        """Visualize movement patterns for loaded simulations."""
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories for each simulation
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]

            # Find matching file path
            file_path = next((p for p in files if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]

                # Plot trajectories for each node
                num_nodes = self.analyzer.get_simulation_summary(data)['num_objects']
                for node in range(num_nodes):
                    x, y, z = self.analyzer.get_object_trajectory(data, node)
                    ax.plot(x, y, z, color=color, alpha=0.5)

                # Add to legend
                ax.plot([], [], color=color, label=filename)

        # Configure plot appearance
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Movement Pattern Comparison')
        ax.legend()

        plt.show()

    def compare_displacement(self):
        """Compare node displacements across simulations."""
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Create node selection dialog
        self.create_node_selection_dialog(files[0])

    def compare_displacement(self):
        """Compare node displacements across simulations."""
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Create node selection dialog
        self.create_node_selection_dialog(files[0])

    def analyze_harmonics(self):
        """
        Compare simulation data to harmonic patterns and visualize the matches.
        This analysis helps identify which harmonic modes are most present in the simulation.
        """
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Hide the main window while showing the analysis
        self.root.withdraw()

        try:
            # Generate the harmonic analysis
            self.plot_harmonic_correlation(files)
        finally:
            # Ensure the main window is shown again
            self.root.deiconify()

    def plot_harmonic_correlation(self, files):
        """
        Calculate and plot the correlation between simulation data and harmonic patterns.
        Uses numpy for calculations instead of scipy.

        Args:
            files: List of simulation file paths to analyze
        """
        import matplotlib.pyplot as plt
        import numpy as np

        def generate_harmonic(x, n):
            """
            Generate nth harmonic pattern.

            Args:
                x: Position values along string (0 to 1)
                n: Harmonic number (1 for fundamental, 2 for first overtone, etc.)

            Returns:
                Harmonic displacement pattern
            """
            return np.sin(n * np.pi * x)

        def compute_correlation(x, y):
            """
            Compute the Pearson correlation coefficient between two arrays.

            Args:
                x, y: Arrays to compare

            Returns:
                Correlation coefficient between -1 and 1
            """
            # Remove mean
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)

            # Compute correlation
            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

            if denominator == 0:
                return 0

            return numerator / denominator

        # Create figure for the analysis
        fig, axes = plt.subplots(len(files), 1, figsize=(12, 4 * len(files)))
        if len(files) == 1:
            axes = [axes]  # Make axes iterable if only one file

        # Number of harmonics to analyze
        num_harmonics = 10

        for file_idx, file_path in enumerate(files):
            # Get the data for this simulation
            data = self.loaded_files[file_path]
            filename = os.path.basename(file_path)

            # Get number of nodes
            num_nodes = self.analyzer.get_simulation_summary(data)['num_objects']

            # Create normalized position array (0 to 1)
            x_positions = np.linspace(0, 1, num_nodes)

            # Calculate average displacement pattern
            avg_displacement = np.zeros(num_nodes)
            max_displacement = 0

            # Calculate time-averaged displacement pattern
            for node in range(num_nodes):
                _, _, z = self.analyzer.get_object_trajectory(data, node)
                # Remove initial position to get pure displacement
                z = z - z[0]
                # Use RMS of displacement to capture motion amplitude
                avg_displacement[node] = np.sqrt(np.mean(z ** 2))
                max_displacement = max(max_displacement, abs(avg_displacement[node]))

            # Normalize the displacement pattern
            if max_displacement > 0:
                avg_displacement /= max_displacement

            # Calculate correlation with each harmonic
            correlations = []
            for n in range(1, num_harmonics + 1):
                harmonic_pattern = generate_harmonic(x_positions, n)
                # Take absolute value of correlation since phase doesn't matter
                correlation = abs(compute_correlation(avg_displacement, harmonic_pattern))
                correlations.append(correlation * 100)  # Convert to percentage

            # Create the bar plot for this file
            ax = axes[file_idx]
            bars = ax.bar(range(1, num_harmonics + 1), correlations)

            # Configure the plot
            ax.set_xlabel('Harmonic Number')
            ax.set_ylabel('Correlation (%)')
            ax.set_title(f'Harmonic Analysis - {filename}')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, num_harmonics + 1))
            ax.set_ylim(0, 100)

            # Add percentage labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom')

            # Add explanatory text for strongest harmonics
            sorted_harmonics = np.argsort(correlations)[::-1]  # Sort in descending order
            top_3_text = "Dominant harmonics:\n"
            for i in range(min(3, len(sorted_harmonics))):
                harmonic_num = sorted_harmonics[i] + 1
                correlation = correlations[sorted_harmonics[i]]
                top_3_text += f"{harmonic_num}th: {correlation:.1f}%\n"

            ax.text(0.98, 0.98, top_3_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

        return correlations

    def create_node_selection_dialog(self, file_path):
        """
        Create a dialog window for selecting nodes to analyze.

        Args:
            file_path: Path to the simulation file being analyzed
        """
        import tkinter as tk
        from tkinter import ttk

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Node to Analyze")

        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Get number of nodes from the simulation data
        data = self.loaded_files[file_path]
        num_nodes = self.analyzer.get_simulation_summary(data)['num_objects']

        # Create node selection list
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Select a node to analyze:").pack(pady=(0, 5))

        # Create listbox for node selection
        listbox = tk.Listbox(frame, selectmode=tk.SINGLE, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=scrollbar.set)

        # Add node numbers to listbox
        for i in range(num_nodes):
            listbox.insert(tk.END, f"Node {i}")

        def on_ok():
            """Handle OK button click."""
            selections = listbox.curselection()
            if selections:
                node_id = int(listbox.get(selections[0]).split()[1])
                dialog.destroy()
                # Hide the main window while showing the plot
                self.root.withdraw()
                # Show the plot
                self.plot_displacement_comparison(node_id)
                # Show the main window again
                self.root.deiconify()
            else:
                from tkinter import messagebox
                messagebox.showwarning("Warning", "Please select a node.")

        def on_cancel():
            """Handle cancel button click."""
            dialog.destroy()

        # Add buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

        # Center dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')

    def plot_displacement_comparison(self, node_id):
        """
        Plot displacement comparison for a selected node across all active simulations.

        Args:
            node_id: ID of the node to compare across simulations
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a new figure with a larger size
        plt.figure(figsize=(12, 8))

        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Plot data for each simulation
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]

            # Find matching file path
            file_path = next((p for p in files if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]

                try:
                    # Get position data for the node
                    x, y, z = self.analyzer.get_object_trajectory(data, node_id)
                    positions = np.column_stack([x, y, z])

                    # Calculate displacement from initial position
                    initial_pos = positions[0]
                    displacements = np.linalg.norm(positions - initial_pos, axis=1)

                    # Calculate time values based on simulation data
                    time_step = self.analyzer.simulations[data]['Time'].diff().mean()
                    time = np.arange(len(displacements)) * time_step

                    # Plot with the color from the file list
                    plt.plot(time, displacements, color=color, label=filename)

                except Exception as e:
                    from tkinter import messagebox
                    messagebox.showerror("Error", f"Error plotting {filename}: {str(e)}")
                    continue

        # Configure the plot
        plt.xlabel('Time (s)')
        plt.ylabel(f'Node {node_id} Displacement')
        plt.title(f'Displacement Comparison for Node {node_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Show the plot - this blocks until the plot window is closed
        plt.show()

    def plot_nodal_average_displacement(self):
        """Compare average node displacements across simulations."""
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(12, 6))

        # Calculate and plot average displacements for each simulation
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]

            file_path = next((p for p in files if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                num_nodes = self.analyzer.get_simulation_summary(data)['num_objects']

                # Calculate average displacement for each node
                avg_displacements = []
                for node_id in range(num_nodes):
                    x, y, z = self.analyzer.get_object_trajectory(data, node_id)
                    positions = np.column_stack([x, y, z])
                    initial_pos = positions[0]
                    displacements = np.linalg.norm(positions - initial_pos, axis=1)
                    avg_displacements.append(np.mean(displacements))

                plt.plot(range(num_nodes), avg_displacements,
                         color=color, label=filename, marker='o')

        # Configure plot appearance
        plt.xlabel('Node ID')
        plt.ylabel('Average Displacement')
        plt.title('Average Node Displacement Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set integer ticks for node IDs
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.show()

    def compare_stationary(self):
        """Analyze and compare stationary nodes across simulations."""
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Compare stationary nodes across simulations
        comparison = {}
        for file_path in files:
            data = self.loaded_files[file_path]
            comparison[os.path.basename(file_path)] = self.analyzer.find_stationary_nodes(data)

        # Display results in a new window
        import tkinter as tk
        window = tk.Toplevel(self.root)
        window.title("Stationary Nodes Comparison")

        text = tk.Text(window, wrap=tk.WORD, width=60, height=30)
        text.pack(padx=10, pady=10)

        # Format and display comparison results
        for filename, nodes in comparison.items():
            text.insert("end", f"\nSimulation: {filename}\n{'=' * 50}\n")
            if nodes:
                text.insert("end", f"Found {len(nodes)} stationary nodes:\n")
                for node_id, pos in nodes.items():
                    text.insert("end", f"\nNode {node_id}:\n")
                    text.insert("end", f"  Position: {pos}\n")
                    text.insert("end", f"  Displacement: {np.linalg.norm(pos):.6f}\n")
            else:
                text.insert("end", "No stationary nodes found\n")

        text.config(state=tk.DISABLED)

    def show_summary(self):
        """Display summary information for selected simulations."""
        files = self.get_selected_files()
        if not files:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Create summary window
        import tkinter as tk
        window = tk.Toplevel(self.root)
        window.title("Simulation Summary")

        text = tk.Text(window, wrap=tk.WORD, width=60, height=30)
        text.pack(padx=10, pady=10)

        # Display summary for each selected file
        for file_path in files:
            summary = self.analyzer.get_simulation_summary(self.loaded_files[file_path])
            filename = os.path.basename(file_path)

            text.insert("end", f"\nFile: {filename}\n{'=' * 50}\n")
            text.insert("end", f"Number of frames: {summary['num_frames']}\n")
            text.insert("end", f"Simulation time: {summary['simulation_time']:.3f} s\n")
            text.insert("end", f"Number of objects: {summary['num_objects']}\n")
            text.insert("end", f"Stationary nodes: {summary['num_stationary_nodes']}\n\n")

            for node_id, pos in summary['stationary_node_positions'].items():
                text.insert("end", f"Node {node_id} position: {pos}\n")

        text.config(state=tk.DISABLED)

    def run(self):
        """Start the analysis visualization system."""
        self.root.mainloop()

    def on_closing(self):
        """Handle window closing event."""
        self.root.destroy()
        if self.main_root:
            self.main_root.deiconify()  # Show main menu
