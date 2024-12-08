"""Enhanced visualization system with integrated force handling."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
from enum import Enum
from typing import List
from mpl_toolkits.mplot3d import Axes3D

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

class SimulationVisualizer:
    """Main visualization class handling display and interaction."""

    def __init__(self, physics_model, objects: List, dark_mode: bool = True):
        """
        Initialize the visualizer with default settings.

        Args:
            physics_model: The physics engine or model managing object dynamics.
            objects: List of objects to visualize.
            dark_mode: Whether to use dark mode for the UI.
        """
        self.physics = physics_model  # Reference to the physics model
        self.objects = objects  # List of objects in the simulation
        self.dark_mode = dark_mode  # Flag for dark mode
        self.paused = False  # Whether the simulation is paused
        self.rotating = False  # Flag for camera rotation
        self.panning = False  # Flag for camera panning
        self.plots = {}  # Storage for plot elements (scatter, lines)
        self.simulation_time = 0.0  # Current simulation time
        self.iteration_count = 100  # Default number of physics iterations per frame

        # Save initial object states (positions and velocities).
        self.original_positions = [obj.position.copy() for obj in objects]
        self.original_velocities = [obj.velocity.copy() for obj in objects]
        self.initial_physics_time = physics_model.time  # Initial time in the physics engine.

        # Initialize the force handler.
        self.force_handler = ForceHandler(physics_model, objects, dark_mode)

        # Camera settings for visualization.
        self.camera = {
            'distance': 2.0,  # Default camera distance from origin.
            'azimuth': 45,  # Azimuth angle for the camera view.
            'elevation': 15,  # Elevation angle for the camera view.
            'rotation_speed': 0.3,  # Speed of camera rotation.
            'zoom_speed': 0.1,  # Speed of camera zoom.
            'target': np.zeros(3)  # Point to center the camera on.
        }

        # Set up the visualization environment.
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

        # Left side panel for force controls
        left_panel_start = 0.07
        panel_width = 0.12

        # Add simulation speed slider at the bottom
        self.speed_slider = Slider(
            plt.axes([0.24, 0.02, 0.44, 0.02]),  # Position below other controls
            'Simulation Speed',
            1, 200,  # Range from 1 to 200 iterations per frame
            valinit=self.iteration_count,
            valfmt='%d steps/frame'
        )
        self.speed_slider.on_changed(self.set_simulation_speed)

        # Move the basic control buttons up slightly to make room for the speed slider
        button_configs = [
            ('play_button', 'Pause', 0.24),
            ('reset_button', 'Reset', 0.35),
            ('view_button', 'View: Default', 0.46),
            ('zoom_button', 'Zoom: Fit All', 0.57),
            ('theme_button', 'Theme', 0.68)
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

        # Force controls header
        self.fig.text(left_panel_start, 0.9, 'Force Controls', color=text_color, fontsize=10)

        # Force type selector
        self.force_radio = RadioButtons(
            plt.axes([left_panel_start, 0.72, panel_width, 0.15]),
            list(self.force_handler.types.keys())
        )
        self.force_radio.on_clicked(self.set_force_type)

        # Force direction selector
        self.fig.text(left_panel_start, 0.65, 'Direction:', color=text_color, fontsize=10)
        self.direction_radio = RadioButtons(
            plt.axes([left_panel_start, 0.47, panel_width, 0.15]),
            list(self.force_handler.directions.keys())
        )
        self.direction_radio.on_clicked(self.set_force_direction)

        # Object selector slider
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
            0.1, 20.0,
            valinit=self.force_handler.amplitude
        )
        self.amplitude_slider.on_changed(self.set_force_amplitude)

        # Duration slider
        self.duration_slider = Slider(
            plt.axes([left_panel_start, 0.30, panel_width, 0.02]),
            'Duration',
            0.1, 10.0,
            valinit=self.force_handler.duration,
            valfmt='%.1f s'
        )
        self.duration_slider.on_changed(self.set_force_duration)
        self.duration_slider.on_changed(self.set_force_duration_remaining)

        # Force buttons
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
        """Set the number of physics iterations per frame."""
        self.iteration_count = int(val)

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
        """Update information display."""
        info_text = (
            f"View: {self.view_button.label.get_text().split(': ')[1]}\n"
            f"{'PAUSED' if self.paused else 'RUNNING'}\n"
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