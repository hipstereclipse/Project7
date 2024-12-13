import os
import time
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons

from data_handling import DataAnalysis


class ViewPreset(Enum):
    """Predefined camera view angle presets for the 3D simulation plot."""
    DEFAULT = {"name": "Default", "azim": 45, "elev": 15}
    TOP = {"name": "Top (XY)", "azim": 0, "elev": 90}
    FRONT = {"name": "Front (XZ)", "azim": 0, "elev": 0}
    SIDE = {"name": "Side (YZ)", "azim": 90, "elev": 0}
    ISOMETRIC = {"name": "Isometric", "azim": 45, "elev": 35}
    FREE = {"name": "Free", "azim": None, "elev": None}


@dataclass
class PlotConfig:
    """
    Configuration settings for customizing plots, including padding, axis labels,
    figure size, and whether the plot is 2D or 3D. Also includes optional view presets
    and grid/legend settings.
    """
    # Basic parameters for labels and scales
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    y2label: Optional[str] = None
    xscale: str = 'linear'
    yscale: str = 'linear'
    y2scale: str = 'linear'
    grid: bool = True
    legend_position: str = 'right'
    figure_size: Tuple[float, float] = (10, 6)
    style: str = 'default'
    colors: Optional[List[str]] = None

    # 3D-specific parameters
    is_3d: bool = False
    view_preset: Optional[ViewPreset] = None
    interactive_3d: bool = True
    show_view_controls: bool = False
    show_animation_controls: bool = False
    auto_rotate: bool = False
    rotation_speed: float = 1.0

    # Padding and spacing controls for titles, labels, and subplots
    title_pad: float = 20
    xlabel_pad: float = 10
    ylabel_pad: float = 10
    zlabel_pad: float = 10
    legend_pad: float = 0.1
    tick_pad: float = 5
    subplot_top_pad: float = 0.9
    subplot_bottom_pad: float = 0.1
    subplot_left_pad: float = 0.1
    subplot_right_pad: float = 0.9


class PlottingToolkit:
    """
    A toolkit class for creating and managing 2D/3D matplotlib plots,
    including UI elements like buttons, sliders, and radio buttons.
    It provides methods to customize views, themes, and plot data.
    """
    DEFAULT_CONFIG = {
        'enable_view_cycling': True,
        'enable_zoom_controls': True,
        'enable_theme_toggle': True,
        'enable_camera_controls': True,
        'dark_mode': True,
        'show_grid': True,
        'show_axes': True,
        'enable_animation': False,
        'control_panel_width': 0.2,
        'side_panel_start': 0.85,
        'views': {
            'Default': {'azim': 45, 'elev': 15},
            'Top': {'azim': 0, 'elev': 90},
            'Front': {'azim': 0, 'elev': 0},
            'Side': {'azim': 90, 'elev': 0},
            'Isometric': {'azim': 45, 'elev': 35}
        },
        'colors': {
            'dark': {
                'background': 'black',
                'text': 'white',
                'button': 'gray',
                'hover': '#404040',
                'active': 'lightblue'
            },
            'light': {
                'background': 'white',
                'text': 'black',
                'button': 'lightgray',
                'hover': '#d0d0d0',
                'active': 'skyblue'
            }
        }
    }

    def __init__(self):
        # Initialize default colors and states
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.current_rotation = 0
        self.animation_running = False
        self.fig = None
        self.ax = None
        self.ui_elements = {k: {} for k in ['buttons','sliders','radio_buttons','text','legends']}
        self._mouse_button = None
        self._mouse_x = None
        self._mouse_y = None
        # Camera settings for 3D view manipulation
        self.camera = {'distance': 2.0, 'azimuth': 45, 'elevation': 15, 'rotation_speed': 0.3, 'zoom_speed': 0.1}
        self.config = dict(self.DEFAULT_CONFIG)
        self.dark_mode = self.config['dark_mode']
        self.current_theme = 'dark' if self.dark_mode else 'light'

    def create_figure(self, config: PlotConfig) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure and axes object based on the provided configuration.
        Applies styles, padding, and axis labels as requested.

        Args:
            config: A PlotConfig instance containing plot setup preferences.

        Returns:
            (Figure, Axes) from matplotlib.
        """
        # Close any existing figure to avoid conflicts
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        plt.style.use(config.style)
        self.fig = plt.figure(figsize=config.figure_size)

        # Create a 3D axes if requested, else 2D axes
        if config.is_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
            if config.zlabel:
                self.ax.set_zlabel(config.zlabel, labelpad=config.zlabel_pad)
            if config.view_preset:
                self.ax.view_init(
                    elev=config.view_preset.value["elev"],
                    azim=config.view_preset.value["azim"]
                )
        else:
            self.ax = self.fig.add_subplot(111)

        # Apply subplot padding
        plt.subplots_adjust(
            top=config.subplot_top_pad,
            bottom=config.subplot_bottom_pad,
            left=config.subplot_left_pad,
            right=config.subplot_right_pad
        )

        # Set title and axes labels with padding
        if config.title:
            self.ax.set_title(config.title, pad=config.title_pad)
        if config.xlabel:
            self.ax.set_xlabel(config.xlabel, labelpad=config.xlabel_pad)
        if config.ylabel:
            self.ax.set_ylabel(config.ylabel, labelpad=config.ylabel_pad)

        # Set axes scales for 2D plots
        if not config.is_3d:
            self.ax.set_xscale(config.xscale)
            self.ax.set_yscale(config.yscale)

        # Apply tick padding
        self.ax.tick_params(pad=config.tick_pad)

        # Configure grid
        self.ax.grid(config.grid, alpha=0.3)
        self.fig.canvas.draw_idle()

        return self.fig, self.ax

    def plot(self, x, y, z=None, new_figure=True, legend_kwargs=None, **kwargs):
        """
        Plot data (2D or 3D) using various plot types (line, scatter, bar),
        with optional new figure creation and custom legend.

        Args:
            x, y: Data coordinates for 2D or 3D plotting
            z: Optional, if provided and is_3d is True, plots in 3D
            new_figure: Whether to create a new figure for this plot
            legend_kwargs: Parameters for legend customization
            **kwargs: Additional plot configuration arguments.
        """
        # Extract PlotConfig-compatible arguments
        config = PlotConfig(**{k: v for k, v in kwargs.items() if k in PlotConfig.__annotations__})

        # If requested, create a new figure or if no figure exists
        if new_figure or self.fig is None:
            self.fig, self.ax = self.create_figure(config)

        # Set default legend parameters and merge with user-provided ones
        default_legend_kwargs = {
            'loc': 'best',
            'bbox_to_anchor': (1 + config.legend_pad, 1),
            'frameon': True,
            'fancybox': True
        }
        legend_kwargs = legend_kwargs or {}
        default_legend_kwargs.update(legend_kwargs)

        plot_type = kwargs.get('plot_type', 'line')
        color = kwargs.get('color', self.default_colors[0])
        alpha = kwargs.get('alpha', 1.0)

        # Decide whether to plot in 2D or 3D
        if z is not None and config.is_3d:
            self._plot_3d(x, y, z, plot_type, color, alpha, kwargs)
        else:
            self._plot_2d(x, y, plot_type, color, alpha, kwargs)

        # If a label is provided, add a legend
        if 'label' in kwargs:
            self.ax.legend(**default_legend_kwargs)

    def _plot_2d(self, x, y, plot_type, color, alpha, kwargs):
        """Internal helper for 2D plotting with different styles (line, bar, scatter)."""
        if plot_type == 'line':
            self.ax.plot(x, y, color=color, linestyle=kwargs.get('line_style', '-'),
                         marker=kwargs.get('marker_style', None),
                         alpha=alpha, linewidth=kwargs.get('line_width', 1.5),
                         label=kwargs.get('label'))
        elif plot_type == 'bar':
            self.ax.bar(x, y, color=color, alpha=alpha, label=kwargs.get('label'))
        elif plot_type == 'scatter':
            self.ax.scatter(x, y, color=color, marker=kwargs.get('marker_style', 'o'),
                            alpha=alpha, label=kwargs.get('label'))

    def _plot_3d(self, x, y, z, plot_type, color, alpha, kwargs):
        """Internal helper for 3D plotting with line, scatter, or surface plots."""
        if plot_type == 'scatter':
            self.ax.scatter(x, y, z, marker=kwargs.get('marker_style', 'o'),
                            color=color, alpha=alpha, label=kwargs.get('label'))
        elif plot_type == 'line':
            self.ax.plot(x, y, z, linestyle=kwargs.get('line_style', '-'),
                         linewidth=kwargs.get('line_width', 1.5),
                         color=color, alpha=alpha, label=kwargs.get('label'))
        elif plot_type == 'surface':
            surf = self.ax.plot_surface(x, y, z, cmap=kwargs.get('color_map', 'viridis'),
                                        alpha=alpha, linewidth=0)
            if kwargs.get('colorbar', True):
                self.fig.colorbar(surf)

    def update_data(self, plot_object, x, y, z=None):
        """
        Update an existing plot object's data points dynamically.
        Useful for animations or real-time updates.
        """
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(self.ax, Axes3D):
            # If it's a 3D scatter, offsets3d can be updated directly
            if hasattr(plot_object, '_offsets3d'):
                plot_object._offsets3d = (np.array(x), np.array(y), np.array(z))
            else:
                # Otherwise, update data for line plots
                plot_object.set_data(x, y)
                plot_object.set_3d_properties(z)
        else:
            # For 2D objects, either set_offsets for scatter or set_data for line
            if hasattr(plot_object, 'set_offsets'):
                plot_object.set_offsets(np.column_stack([x, y]))
            else:
                plot_object.set_data(x, y)

    def add_text(self, x, y, text, **kwargs):
        """
        Add text annotations to the plot, either in 2D or 3D.
        Positions are given in Axes coordinates, not data coordinates.

        Args:
            x, y: float - Position to place the text in axes fraction
            text: str - Text to display
            **kwargs: Additional text formatting parameters
        """
        text_id = kwargs.pop('text_id', None)
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(self.ax, Axes3D):
            txt = self.ax.text2D(x, y, text, transform=self.ax.transAxes, **kwargs)
        else:
            txt = self.ax.text(x, y, text, transform=self.ax.transAxes, **kwargs)

        if text_id:
            self.ui_elements['text'][text_id] = txt
        return txt

    def add_button(self, position, label, callback, color='gray', text_color='white',
                   hover_color=None, button_id=None):
        """
        Add an interactive button to the plot figure.

        Args:
            position: (left, bottom, width, height) in figure coordinates
            label: str - Button label text
            callback: Function to call when button is clicked
        """
        from matplotlib.widgets import Button
        ax_button = plt.axes(position)
        hover_color = hover_color or self._adjust_color_brightness(color, 1.2)
        button = Button(ax_button, label, color=color, hovercolor=hover_color)
        button.label.set_color(text_color)
        button.on_clicked(callback)
        button_id = button_id or f"button_{len(self.ui_elements['buttons'])}"
        self.ui_elements['buttons'][button_id] = {
            'object': button
        }
        return button

    def add_slider(self, position, label, valmin, valmax, valinit=None,
                   callback=None, color='gray', text_color='white', slider_id=None):
        """
        Add a slider UI element to control parameters interactively.

        Args:
            position: (left, bottom, width, height) for slider axes
            label: Slider label text
            valmin, valmax: Range for slider
            valinit: Initial slider value
            callback: Optional function to call on slider value change
        """
        from matplotlib.widgets import Slider
        ax_slider = plt.axes(position)
        valinit = valinit if valinit is not None else valmin
        slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit, color=color)
        slider.label.set_color(text_color)
        slider.valtext.set_color(text_color)
        if callback:
            slider.on_changed(callback)
        slider_id = slider_id or f"slider_{len(self.ui_elements['sliders'])}"
        self.ui_elements['sliders'][slider_id] = {
            'object': slider
        }
        return slider

    def add_radio_buttons(self, position, labels, callback=None, active=0,
                          color='gray', text_color='white', active_color='lightblue', radio_id=None):
        """
        Add a group of radio buttons for selecting one of several options.

        Args:
            position: (left, bottom, width, height) for radio box
            labels: List of label strings for each radio button
            callback: Function to call when a radio button is selected
        """
        from matplotlib.widgets import RadioButtons
        ax_radio = plt.axes(position)
        radio = RadioButtons(ax_radio, labels, active=active, activecolor=active_color)

        # Customize appearance of radio buttons and their labels
        if hasattr(radio, 'circles'):
            for circle in radio.circles:
                circle.set_facecolor('none')
                circle.set_edgecolor(text_color)

        for label in radio.labels:
            label.set_color(text_color)
            label.set_fontweight('bold')

        if callback:
            radio.on_clicked(callback)

        radio_id = radio_id or f"radio_{len(self.ui_elements['radio_buttons'])}"
        self.ui_elements['radio_buttons'][radio_id] = {
            'object': radio
        }

        return radio

    def add_labeled_section(self, title_position, title, description=None,
                            text_color='white', title_size=10, desc_size=8):
        """
        Add a titled section to the plot, possibly with a subtitle or description.
        Useful for grouping UI elements on the figure.
        """
        texts = {}
        texts['title'] = self.add_text(
            title_position[0], title_position[1],
            title,
            color=text_color,
            fontsize=title_size,
            fontweight='bold'
        )
        if description:
            desc_y = title_position[1] - 0.03
            texts['description'] = self.add_text(
                title_position[0] + 0.01, desc_y,
                description,
                color=text_color,
                fontsize=desc_size
            )

        return texts

    def _adjust_color_brightness(self, color, factor):
        """Utility to lighten or darken a given color by a factor."""
        import matplotlib.colors as mcolors
        try:
            rgb = mcolors.to_rgb(color)
            adjusted = tuple(min(1.0, c * factor) for c in rgb)
            return adjusted
        except ValueError:
            return color

    def update_ui_theme(self, dark_mode=True):
        """
        Update the plot theme between dark and light mode, adjusting colors
        of all UI elements and text accordingly.
        """
        self.dark_mode = dark_mode
        self.current_theme = 'dark' if dark_mode else 'light'
        background = 'black' if dark_mode else 'white'
        text_color = 'white' if dark_mode else 'black'
        btn_color = 'gray' if dark_mode else 'lightgray'

        # Update figure and axes background and text colors
        if self.fig and self.ax:
            self.fig.set_facecolor(background)
            self.ax.set_facecolor(background)
            for spine in self.ax.spines.values():
                spine.set_color(text_color)
            self.ax.tick_params(colors=text_color)
            self.ax.xaxis.label.set_color(text_color)
            self.ax.yaxis.label.set_color(text_color)
            if hasattr(self.ax, 'zaxis'):
                self.ax.zaxis.label.set_color(text_color)
            self.ax.grid(self.config.get('show_grid', True), alpha=0.3)

        # Update button, slider, and radio button colors
        for button_info in self.ui_elements['buttons'].values():
            button = button_info['object']
            button.color = btn_color
            button.hovercolor = self._adjust_color_brightness(btn_color, 1.2)
            button.label.set_color(text_color)

        for slider_info in self.ui_elements['sliders'].values():
            slider = slider_info['object']
            slider.label.set_color(text_color)
            slider.valtext.set_color(text_color)

        for radio_info in self.ui_elements['radio_buttons'].values():
            radio = radio_info['object']
            if hasattr(radio, 'circles'):
                for circle in radio.circles:
                    circle.set_edgecolor(text_color)
            for label in radio.labels:
                label.set_color(text_color)

        for text_id, text_obj in self.ui_elements['text'].items():
            text_obj.set_color(text_color)

        if self.fig:
            self.fig.canvas.draw_idle()

    def clear_ui(self):
        """Remove all UI elements (buttons, sliders, text) from the current figure."""
        for element_type in self.ui_elements:
            for element_info in self.ui_elements[element_type].values():
                if hasattr(element_info['object'], 'ax'):
                    element_info['object'].ax.remove()
        self.ui_elements = {key: {} for key in self.ui_elements}

    def remove_element(self, element_type, element_id):
        """Remove a specific UI element by its type and ID."""
        if element_id in self.ui_elements[element_type]:
            element = self.ui_elements[element_type][element_id]
            if hasattr(element['object'], 'ax'):
                element['object'].ax.remove()
            del self.ui_elements[element_type][element_id]

    def update_view(self, camera):
        """Update the 3D view (camera angles and distance) of the plot."""
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(self.ax, Axes3D):
            self.ax.view_init(elev=camera['elevation'], azim=camera['azimuth'])
            self.ax.dist = camera['distance']
            if self.fig:
                self.fig.canvas.draw_idle()

    def show(self):
        """Display the figure window."""
        if self.fig is not None:
            plt.show()
            self.fig = None
            self.ax = None

    def clear(self):
        """Close the current figure window and clear the axes."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def update_text(self, text_id, new_text):
        """Update the text of a previously created text element."""
        text_obj = self.ui_elements['text'].get(text_id, None)
        if text_obj:
            text_obj.set_text(new_text)
            if self.fig:
                self.fig.canvas.draw_idle()

    def update_plot_bounds(self, x_range, y_range, z_range=None):
        """
        Set new axis limits for the plot, ensuring ranges are not zero-width.
        Works for both 2D and 3D plots.
        """
        from mpl_toolkits.mplot3d import Axes3D
        if self.ax is not None:
            def safe_range(rng):
                if rng[0] == rng[1]:
                    return (rng[0] - 0.5, rng[1] + 0.5)
                return rng
            x_range = safe_range(x_range)
            y_range = safe_range(y_range)
            if z_range is not None:
                z_range = safe_range(z_range)

            self.ax.set_xlim(x_range)
            self.ax.set_ylim(y_range)
            if isinstance(self.ax, Axes3D) and z_range is not None:
                self.ax.set_zlim(z_range)
            if self.fig:
                self.fig.canvas.draw_idle()

    def setup_core_controls(self, callbacks):
        # This method can be extended to setup default UI controls.
        pass


@dataclass
class SimulationParameters:
    """
    Dataclass for storing simulation parameters like number of segments,
    spring constant, mass, time step, integration method, and initial conditions.
    """
    num_segments: int = 25
    spring_constant: float = 1000.0
    mass: float = 0.01
    dt: float = 0.0001
    start_point: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 0.0]))
    end_point: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    integration_method: str = 'leapfrog'
    applied_force: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    dark_mode: bool = True


class ForceHandler:
    """
    Handles applying and managing external forces to the simulation objects.
    Supports different force types (single mass, sinusoidal, Gaussian profile),
    as well as continuous or time-limited forces.
    """

    def __init__(self, physics, objects, dark_mode=True):
        self.physics = physics
        self.objects = objects
        self.dark_mode = dark_mode
        self.active = False
        self.continuous = False
        self.duration = 0.01
        self.duration_remaining = 0.01
        self.amplitude = 10.0
        self.selected_object = len(objects) // 2
        self.gaussian_width = len(self.objects) / 8.0
        self.sinusoidal_frequency = 1.0  # Frequency for sinusoidal force

        self.types = {
            'Single Mass': lambda t, x: np.array([0.0, 0.0, 1.0]),
            'Sinusoidal': lambda t, x: np.array([0.0, 0.0, np.sin(2 * np.pi * self.sinusoidal_frequency * t)]),
            'Gaussian': lambda t, x: np.array(
                [0.0, 0.0, np.exp(-(x - len(self.objects) // 2) ** 2 / self.gaussian_width ** 2)]
            )
        }

        self.directions = {
            'Up/Down (Z)': np.array([0.0, 0.0, 1.0]),
            'Left/Right (X)': np.array([1.0, 0.0, 0.0]),
            'Front/Back (Y)': np.array([0.0, 1.0, 0.0])
        }

        self.selected_type = 'Single Mass'
        self.selected_direction = 'Up/Down (Z)'

    def check_duration(self, iterations_per_frame):
        """
        Check if the force duration has expired if it's not continuous.
        Deactivates force if time runs out.
        """
        if not self.continuous and self.active:
            time_elapsed = self.physics.dt * iterations_per_frame
            self.duration_remaining = max(0.0, self.duration_remaining - time_elapsed)
            if self.duration_remaining <= 0:
                self.deactivate()
                return True
            self.apply(self.duration_remaining)
        return False

    def toggle(self):
        """
        Toggle the force on or off. If continuous, remains locked on.
        If not continuous, reset duration and activate/deactivate accordingly.
        """
        if self.active:
            self.deactivate()
            return ('Apply Force', 'darkgray')
        elif self.continuous:
            self.active = True
            return ('Force Locked', 'red')
        else:
            self.active = True
            self.duration_remaining = self.duration
            self.apply(self.duration)
            return ('Force Active', 'lightgreen')

    def apply(self, duration=None):
        direction = self.directions[self.selected_direction]

        if self.selected_type == 'Single Mass':
            force = direction * self.amplitude
            self.physics.apply_force(self.selected_object, force, duration)

        elif self.selected_type == 'Sinusoidal':
            # Use self.sinusoidal_frequency
            magnitude = np.sin(2 * np.pi * self.sinusoidal_frequency * self.physics.time) * self.amplitude
            self.physics.apply_force(self.selected_object, direction * magnitude, duration)

        elif self.selected_type == 'Gaussian':
            # Use self.gaussian_width
            for i in range(1, len(self.objects) - 1):
                magnitude = np.exp(-(i - self.selected_object) ** 2 / self.gaussian_width ** 2) * self.amplitude
                self.physics.apply_force(i, direction * magnitude, duration)



    def deactivate(self):
        """
        Deactivate the force, resetting duration_remaining and clearing forces from all objects.
        """
        self.active = False
        self.duration_remaining = self.duration
        for obj_id in range(len(self.objects)):
            self.physics.clear_force(obj_id)


class StringSimulationSetup:
    """
    A GUI class that allows users to configure simulation parameters before starting the simulation.
    Provides fields for number of segments, spring constant, mass, dt, and force magnitude, and integrates chosen settings.
    """
    def __init__(self, main_root):
        self.main_root = main_root
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Setup")
        self.default_params = SimulationParameters()
        self.init_variables()
        self.root.minsize(600, 500)
        # Position window at a reasonable location on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.4)
        window_height = int(screen_height * 0.6)
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 4) - (window_height // 4)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.create_responsive_styles()
        self.setup_gui()
        self.root.bind('<Configure>', self.on_window_resize)
        self.simulation_params = None

    def init_variables(self):
        """Initialize tkinter variables for each parameter the user can adjust."""
        self.num_segments_var = tk.IntVar(self.root)
        self.num_segments_var.set(self.default_params.num_segments)
        self.spring_constant_var = tk.DoubleVar(self.root)
        self.spring_constant_var.set(self.default_params.spring_constant)
        self.mass_var = tk.DoubleVar(self.root)
        self.mass_var.set(self.default_params.mass)
        self.dt_var = tk.DoubleVar(self.root)
        self.dt_var.set(self.default_params.dt)
        force_magnitude = float(np.linalg.norm(self.default_params.applied_force))
        self.force_magnitude_var = tk.DoubleVar(self.root)
        self.force_magnitude_var.set(force_magnitude)
        self.integration_var = tk.StringVar(self.root)
        self.integration_var.set(self.default_params.integration_method)
        self.dark_mode_var = tk.BooleanVar(self.root)
        self.dark_mode_var.set(self.default_params.dark_mode)
        self.simulation_params = None

    def create_responsive_styles(self):
        """Create ttk styles that adjust with window resize."""
        style = ttk.Style()
        self.base_header_size = 14
        self.base_text_size = 10
        self.base_button_size = 11
        style.configure("Header.TLabel", font=("Arial", self.base_header_size, "bold"), anchor="center")
        style.configure("Normal.TLabel", font=("Arial", self.base_text_size), anchor="w")
        style.configure("Setup.TButton", font=("Arial", self.base_button_size), padding=10)

    def on_window_resize(self, event):
        """Dynamically scale fonts and padding as the window resizes."""
        if event.widget == self.root:
            width_scale = event.width / (self.root.winfo_screenwidth() * 0.4)
            height_scale = event.height / (self.root.winfo_screenheight() * 0.6)
            scale = min(width_scale, height_scale)
            style = ttk.Style()
            style.configure("Header.TLabel", font=("Arial", int(self.base_header_size * scale), "bold"))
            style.configure("Normal.TLabel", font=("Arial", int(self.base_text_size * scale)))
            style.configure("Setup.TButton", font=("Arial", int(self.base_button_size * scale)))
            base_padding = 10
            scaled_padding = int(base_padding * scale)
            style.configure("Setup.TButton", padding=scaled_padding)

    def setup_gui(self):
        """Build the main GUI with tabs for basic and advanced parameters."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(sticky="nsew")
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(header_frame, text="String Simulation Setup",
                  font=("Arial", 14, "bold")).pack()
        ttk.Label(header_frame,
                  text="Configure simulation parameters",
                  font=("Arial", 10)).pack()

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

        start_button = ttk.Button(
            main_frame,
            text="Start Simulation",
            command=self.start_simulation,
            style="Accent.TButton"
        )
        start_button.grid(row=2, column=0, pady=10)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def setup_basic_parameters(self, parent):
        """UI elements for basic string properties: number of segments, k, mass, and display settings."""
        props_frame = ttk.LabelFrame(parent, text="String Properties", padding="10")
        props_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(props_frame, text="Number of Masses:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(props_frame, textvariable=self.num_segments_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(props_frame, text="Spring constant (N/m):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(props_frame, textvariable=self.spring_constant_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(props_frame, text="Mass per point (kg):").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(props_frame, textvariable=self.mass_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        display_frame = ttk.LabelFrame(parent, text="Display Settings", padding="10")
        display_frame.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(
            display_frame,
            text="Dark Mode",
            variable=self.dark_mode_var
        ).pack(padx=5, pady=5)

    def setup_advanced_parameters(self, parent):
        """UI elements for force magnitude, integration method, and time step settings."""
        force_frame = ttk.LabelFrame(parent, text="Force Settings", padding="10")
        force_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(force_frame, text="Force magnitude (N):").grid(row=0, column=0, padx=5, pady=5)
        force_entry = ttk.Entry(force_frame, textvariable=self.force_magnitude_var, width=10)
        force_entry.grid(row=0, column=1, padx=5, pady=5)

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

        time_frame = ttk.LabelFrame(parent, text="Time Settings", padding="10")
        time_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(time_frame, text="Time step (dt):").grid(row=0, column=0, padx=5, pady=5)
        dt_entry = ttk.Entry(time_frame, textvariable=self.dt_var, width=10)
        dt_entry.grid(row=0, column=1, padx=5, pady=5)

        help_frame = ttk.Frame(parent)
        help_frame.pack(fill="x", padx=5, pady=10)
        ttk.Label(
            help_frame,
            text="Tip: Smaller time steps give more accurate results but run slower",
            font=("Arial", 8),
            foreground="gray"
        ).pack(pady=2)

    # Methods like set_simulation_speed, toggle_pause, reset_simulation, etc. are utility methods called from visualization/UI
    # They handle updating simulation parameters and UI state.

    def validate_parameters(self):
        """Check that all user-entered parameters are valid."""
        try:
            num_segments = self.num_segments_var.get()
            if num_segments < 3:
                raise ValueError("Number of masses must be at least 3")

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
            messagebox.showerror("Invalid Parameters", str(e))
            return False

    def get_parameters(self):
        """Assemble and return a SimulationParameters object from user input."""
        if not hasattr(self, 'simulation_params'):
            return None
        try:
            # Re-validate before returning parameters
            if self.num_segments_var.get() < 3:
                raise ValueError("Number of masses must be at least 3")
            if self.spring_constant_var.get() <= 0:
                raise ValueError("Spring constant must be positive")
            if self.mass_var.get() <= 0:
                raise ValueError("Mass must be positive")
            if self.dt_var.get() <= 0:
                raise ValueError("Time step must be positive")
            if self.force_magnitude_var.get() < 0:
                raise ValueError("Force magnitude cannot be negative")

            params = SimulationParameters(
                num_segments=(self.num_segments_var.get() - 1),
                spring_constant=self.spring_constant_var.get(),
                mass=self.mass_var.get(),
                dt=self.dt_var.get(),
                integration_method=self.integration_var.get(),
                applied_force=np.array([0.0, 0.0, self.force_magnitude_var.get()]),
                dark_mode=self.dark_mode_var.get()
            )
            return params

        except Exception as e:
            messagebox.showerror("Parameter Validation Error", str(e))
            return None

    def start_simulation(self):
        """
        Validate parameters and if correct, close the setup window and pass parameters back.
        If invalid, show an error. If valid, store them and exit.
        """
        if not self.validate_parameters():
            return
        self.simulation_params = SimulationParameters(
            num_segments=(self.num_segments_var.get()-1),
            spring_constant=self.spring_constant_var.get(),
            mass=self.mass_var.get(),
            dt=self.dt_var.get(),
            integration_method=self.integration_var.get(),
            applied_force=np.array([0.0, 0.0, self.force_magnitude_var.get()]),
            dark_mode=self.dark_mode_var.get()
        )
        self.root.quit()
        self.root.destroy()


class SimulationVisualizer:
    """
    This class manages the live simulation visualization, handling the main matplotlib figure,
    camera controls, UI elements like start/pause/reset, and shows forces and camera info.
    It integrates with the physics model and objects, updating their states and the display each frame.
    """
    def __init__(self, physics_model, objects: List, dark_mode: bool = True, integration_method: str = 'leapfrog'):
        self.physics = physics_model
        self.objects = objects
        self.dark_mode = dark_mode
        self.paused = True
        self.iteration_count = 100
        self.integration_method = integration_method
        self.should_restart = False
        self.original_positions = [obj.position.copy() for obj in objects]
        self.original_velocities = [obj.velocity.copy() for obj in objects]
        self.initial_physics_time = physics_model.time

        # Initialize force handling
        self.force_handler = ForceHandler(physics_model, objects, dark_mode)

        # Setup plotting toolkit for UI and visualization
        self.plotter = PlottingToolkit()

        # Camera and rendering settings
        self.camera = {
            'distance': 2.0,
            'azimuth': 45,
            'elevation': 15,
            'rotation_speed': 0.3,
            'zoom_speed': 0.1,
            'target': np.zeros(3)
        }

        self.simulation_started = False
        self.rotating = False
        self.panning = False
        self.plots = {}
        self.fps = 0
        self.frame_times = []
        self.max_frame_times = 30
        self.animation_frame_count = 0
        self.fps_update_interval = 0.5
        self.last_frame_time = time.time()
        self.main_root = None

        self.setup_visualization()

    def setup_visualization(self):
        """Initialize figure, axes, initial plots, camera settings, and UI controls for the simulation."""
        plt.style.use('dark_background' if self.dark_mode else 'default')
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.20, 0.15, 0.6, 0.8])

        self.plotter.fig = self.fig
        self.plotter.ax = self.ax

        self.setup_plots()
        self.setup_camera()
        self.setup_enhanced_controls()
        self._connect_events()

        text_color = 'white' if self.dark_mode else 'black'
        # Add text elements to display force info and camera info
        self.force_info_text = self.ax.text2D(
            1.15, 0.55,
            '',
            transform=self.ax.transAxes,
            color=text_color,
            fontsize=10,
            verticalalignment='top'
        )
        self.plotter.ui_elements['text']['force_info'] = self.force_info_text

        self.camera_info_text = self.ax.text2D(
            1.15, 0.40,
            '',
            transform=self.ax.transAxes,
            color=text_color,
            fontsize=10,
            verticalalignment='top'
        )
        self.plotter.ui_elements['text']['camera_info'] = self.camera_info_text

    def setup_plots(self):
        """Create scatter and line objects for each mass and the spring lines between them."""
        for i, obj in enumerate(self.objects):
            scatter = self.ax.scatter(
                [], [], [],
                c='red' if obj.pinned else 'blue',
                s=50
            )

            line = None
            if i < len(self.objects) - 1:
                line, = self.ax.plot([], [], [], color='green', linewidth=1.5)

            self.plots[i] = {'scatter': scatter, 'line': line}

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
        """Configure camera distance and axes limits based on object positions."""
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

    def setup_controls(self):
        # Placeholder if we add more default controls
        pass

    def setup_enhanced_controls(self):
        """Set up UI elements like buttons, sliders, and radio buttons for simulation control and force manipulation."""
        btn_color = 'darkgray' if not self.dark_mode else 'gray'
        text_color = 'white' if self.dark_mode else 'black'

        self.setup_button = Button(
            plt.axes([0.02, 0.94, 0.12, 0.04]),
            'Return to Setup',
            color=btn_color
        )
        self.setup_button.on_clicked(self.return_to_setup)

        left_panel_start = 0.07
        panel_width = 0.12

        # Slider for simulation speed (# steps per frame)
        self.speed_slider = Slider(
            plt.axes([0.24, 0.02, 0.44, 0.02]),
            'Simulation Speed',
            1, 1000,
            valinit=self.iteration_count,
            valfmt='%d steps/frame'
        )
        self.speed_slider.on_changed(self.set_simulation_speed)

        # Create and place multiple buttons for play/pause, reset, view cycles, zoom, theme, and saving data
        button_configs = [
            ('play_button', 'Pause', 0.24),
            ('reset_button', 'Reset', 0.35),
            ('view_button', 'View: Default', 0.46),
            ('zoom_button', 'Zoom: Fit All', 0.57),
            ('theme_button', 'Theme', 0.68),
            ('save_button', 'Save Data', 0.79)
        ]

        for btn_name, label, x_pos in button_configs:
            btn = Button(plt.axes([x_pos, 0.06, 0.1, 0.04]), label, color=btn_color)
            setattr(self, btn_name, btn)

        # Assign button callbacks
        self.play_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset_simulation)
        self.view_button.on_clicked(self.cycle_view)
        self.zoom_button.on_clicked(self.cycle_zoom)
        self.theme_button.on_clicked(self.toggle_theme)
        self.save_button.on_clicked(self.save_simulation_data)

        # Force controls section
        self.fig.text(left_panel_start, 0.9, 'Force Controls', color=text_color, fontsize=10)

        # Radio buttons for selecting force type
        self.force_radio = RadioButtons(
            plt.axes([left_panel_start, 0.72, panel_width, 0.15]),
            list(self.force_handler.types.keys())
        )
        self.force_radio.on_clicked(self.set_force_type)

        # Radio buttons for selecting force direction
        self.fig.text(left_panel_start, 0.65, 'Direction:', color=text_color, fontsize=10)
        self.direction_radio = RadioButtons(
            plt.axes([left_panel_start, 0.47, panel_width, 0.15]),
            list(self.force_handler.directions.keys())
        )
        self.direction_radio.on_clicked(self.set_force_direction)

        # Calculate valid slider range for object selection
        min_object = 1  # Skip first object (index 0) as it's usually fixed
        max_object = max(2, len(self.objects) - 2)  # Ensure at least 2 for slider range, skip last object
        initial_object = min(max_object, max(min_object, self.force_handler.selected_object))

        # Create object selection slider with guaranteed valid range
        self.object_slider = Slider(
            plt.axes([left_panel_start, 0.40, panel_width, 0.02]),
            'Object',
            min_object, max_object,
            valinit=initial_object,
            valfmt='%d'
        )

        # Slider for force amplitude
        self.amplitude_slider = Slider(
            plt.axes([left_panel_start, 0.35, panel_width, 0.02]),
            'Amplitude',
            0.1, 50.0,
            valinit=self.force_handler.amplitude
        )
        self.amplitude_slider.on_changed(self.set_force_amplitude)

        # Slider for force duration
        self.duration_slider = Slider(
            plt.axes([left_panel_start, 0.30, panel_width, 0.02]),
            'Duration',
            0.01, 10.0,
            valinit=self.force_handler.duration,
            valfmt='%.2f s'
        )
        self.duration_slider.on_changed(self.set_force_duration)
        self.duration_slider.on_changed(self.set_force_duration_remaining)

        # Create the Gaussian width slider but initially hide it
        self.gaussian_width_slider_ax = plt.axes([left_panel_start+.02, 0.25, panel_width-.02, 0.02])
        self.gaussian_width_slider = Slider(
            self.gaussian_width_slider_ax,
            'Gaussian Width',
            0.1, len(self.objects) / 2.0,
            valinit=self.force_handler.gaussian_width,
            valfmt='%.2f'
        )
        self.gaussian_width_slider.on_changed(self.set_gaussian_width)

        # Create the sinusoidal frequency slider but initially hide it
        self.frequency_slider_ax = plt.axes([left_panel_start, 0.25, panel_width, 0.02])
        self.frequency_slider = Slider(
            self.frequency_slider_ax,
            'Frequency',
            0.1, 999.9,
            valinit=self.force_handler.sinusoidal_frequency,
            valfmt='%.2f'
        )
        self.frequency_slider.on_changed(self.set_sinusoidal_frequency)

        # Hide both sliders initially (since 'Single Mass' is default)
        self.gaussian_width_slider_ax.set_visible(False)
        self.frequency_slider_ax.set_visible(False)

        # Buttons to apply force and toggle continuous force mode
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
        """Close current figure and return to the setup menu, indicating a restart."""
        self.should_restart = True
        plt.close(self.fig)
        if self.main_root:
            self.main_root.deiconify()

    def _connect_events(self):
        """Connect mouse and scroll events for rotating/panning/zooming the 3D view."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    # The mouse/scroll events handle camera rotation and zoom
    def on_mouse_press(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:
                self.rotating = True
            elif event.button == 3:
                self.panning = True
            self.last_x = event.x
            self.last_y = event.y
            if self.view_button.label.get_text() != 'View: Free':
                self.view_button.label.set_text('View: Free')

    def on_mouse_release(self, event):
        self.rotating = False
        self.panning = False

    def on_mouse_move(self, event):
        if event.inaxes == self.ax and hasattr(self, 'last_x'):
            if self.rotating:
                dx = event.x - self.last_x
                dy = event.y - self.last_y

                self.camera['azimuth'] = (self.camera['azimuth'] + dx * self.camera['rotation_speed']) % 360
                self.camera['elevation'] = np.clip(self.camera['elevation'] + (-dy) * self.camera['rotation_speed'], -89, 89)

                self.last_x = event.x
                self.last_y = event.y
                self.update_camera()

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            factor = 0.9 if event.button == 'up' else 1.1
            self.camera['distance'] *= factor
            self.zoom_button.label.set_text('Zoom: Custom')
            self.update_camera()

    def update_frame(self, frame):
        """
        Update the simulation state each animation frame, record data,
        apply forces if active, and update the displayed plots and info.
        """
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

            # For each frame, run multiple simulation steps
            for _ in range(self.iteration_count):
                positions = np.array([obj.position for obj in self.objects])
                velocities = np.array([obj.velocity for obj in self.objects])
                accelerations = np.array([obj.acceleration for obj in self.objects])

                # Record the current state
                self.physics.data_recorder.record_step(
                    self.physics.time,
                    positions,
                    velocities,
                    accelerations
                )

                # Integrate one step
                self.physics.step(self.integration_method)

            # Check if the force duration ended
            if self.force_handler.check_duration(self.iteration_count):
                self.force_button.label.set_text('Apply Force')
                self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'
                self.fig.canvas.draw_idle()

            # If continuous force is active, apply it every frame
            if self.force_handler.continuous and self.force_handler.active:
                self.force_handler.apply()

        self.update_plots()
        self.update_info()
        self.highlight_selected_object()

        return []

    def update_plots(self):
        """Refresh the position of scatter points and lines connecting them."""
        for i, obj in enumerate(self.objects):
            self.plots[i]['scatter']._offsets3d = ([obj.position[0]], [obj.position[1]], [obj.position[2]])
            if i < len(self.objects) - 1:
                next_obj = self.objects[i + 1]
                self.plots[i]['line'].set_data_3d(
                    [obj.position[0], next_obj.position[0]],
                    [obj.position[1], next_obj.position[1]],
                    [obj.position[2], next_obj.position[2]]
                )

    def update_info(self):
        """
        Update text display showing simulation state: paused/running,
        FPS, time, integration method, dt, and force duration info.
        """
        current_time = self.physics.time if self.simulation_started else 0.0
        dt_per_frame = self.physics.dt * self.iteration_count
        state = 'PAUSED' if self.paused else 'RUNNING'

        if self.force_handler.active:
            duration_text = f"{self.force_handler.duration_remaining:.2f}s remaining"
        elif self.force_handler.continuous:
            duration_text = f"Continuous"
        else:
            duration_text = f"Duration set to {self.force_handler.duration:.2f}s"

        info_text = (
            f"{state}\n"
            f"Frame: {self.animation_frame_count}\n"
            f"FPS: {min(self.fps, 60.0):.1f}\n"
            f"Time: {current_time:.3f}s\n"
            f"Integration: {self.integration_method.title()}\n"
            f"dt/step: {self.physics.dt:.6f}s\n"
            f"dt/frame: {dt_per_frame:.6f}s\n"
            f"Steps/Frame: {self.iteration_count}\n"
            f"Force Duration: {duration_text}"
        )
        self.info_text.set_text(info_text)
        self.update_force_info()

    def update_force_info(self):
        """Update the text box that shows the detailed external, spring, and total forces on the selected mass."""
        external_force, spring_forces, total_force = self.physics.get_forces_on_object(
            self.force_handler.selected_object)

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
        self.plotter.update_text('force_info', force_text)

    def update_camera_info(self):
        """Update a text element with current camera angles and distance."""
        text = (
            f"Camera Azim: {self.camera['azimuth']:.1f}\n"
            f"Camera Elev: {self.camera['elevation']:.1f}\n"
            f"Camera Dist: {self.camera['distance']:.1f}"
        )
        self.plotter.update_text('camera_info', text)

    def highlight_selected_object(self):
        """Visually highlight the currently selected object for force application."""
        for i, obj in enumerate(self.objects):
            scatter = self.plots[i]['scatter']
            if i == self.force_handler.selected_object:
                scatter._facecolors[0] = [1.0, 1.0, 0.0, 1.0]
                scatter._sizes = [100]
            elif obj.pinned:
                scatter._facecolors[0] = [1.0, 0.0, 0.0, 1.0]
                scatter._sizes = [50]
            else:
                scatter._facecolors[0] = [0.0, 0.0, 1.0, 1.0]
                scatter._sizes = [50]

    def animate(self):
        """Start the matplotlib animation loop."""
        self.anim = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=20,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
        return self.should_restart

    # Methods to handle UI callbacks:
    def set_simulation_speed(self, val):
        self.iteration_count = int(float(val))
        self.update_info()

    def toggle_pause(self, event):
        """
        Pause/resume simulation and start simulation if it hasn't started yet.
        On resume from idle, record the initial state for data logging.
        """
        if not self.simulation_started and not self.paused:
            return

        self.paused = not self.paused

        if not self.paused and not self.simulation_started:
            self.simulation_started = True
            self.physics.start_simulation()
            # Record initial state
            positions = np.array([obj.position for obj in self.objects])
            velocities = np.array([obj.velocity for obj in self.objects])
            accelerations = np.array([obj.acceleration for obj in self.objects])
            self.physics.data_recorder.record_step(
                self.physics.time,
                positions,
                velocities,
                accelerations
            )

        button_text = 'Start' if not self.simulation_started else ('Resume' if self.paused else 'Pause')
        self.play_button.label.set_text(button_text)
        self.fig.canvas.draw_idle()

    def reset_simulation(self, event):
        """
        Reset simulation to initial conditions, clear data, and set paused state.
        """
        self.animation_frame_count = 0
        self.fps = 0
        self.frame_times.clear()
        self.last_frame_time = time.time()
        self.simulation_started = False
        self.paused = True

        self.play_button.label.set_text('Start')

        self.physics.time = self.initial_physics_time
        self.physics.simulation_started = False
        self.force_handler.deactivate()

        for i, obj in enumerate(self.objects):
            obj.position = self.original_positions[i].copy()
            obj.velocity = self.original_velocities[i].copy()
            obj.acceleration = np.zeros(3)

        self.force_button.label.set_text('Apply Force')

        self.physics.data_recorder.clear_history()

        self.update_plots()
        self.update_info()
        self.fig.canvas.draw_idle()

    def save_simulation_data(self, event):
        """
        Save current simulation data to a CSV file. If simulation is running, pause it first.
        """
        if not self.paused:
            self.toggle_pause(None)
            self.play_button.label.set_text('Resume')
            self.fig.canvas.draw_idle()

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Simulation Data"
        )

        if file_path:
            try:
                self.physics.data_recorder.save_to_csv(file_path)
                messagebox.showinfo(
                    "Success",
                    f"Simulation data saved to:\n{file_path}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to save data: {str(e)}"
                )

        root.destroy()

    def toggle_force(self, event):
        """
        Toggle the application of forces to the string. If simulation not started yet,
        start it now. Handle updating force button label and color.
        """
        if not self.simulation_started and not self.force_handler.active:
            self.simulation_started = True
            self.physics.start_simulation()
            self.paused = False
            self.play_button.label.set_text('Pause')

            if not hasattr(self, 'anim'):
                self.anim = FuncAnimation(
                    self.fig,
                    self.update_frame,
                    interval=20,
                    blit=False,
                    cache_frame_data=False
                )

        result = self.force_handler.toggle()
        if result:
            label, color = result
            self.force_button.label.set_text(label)
            self.force_button.color = color
            if not self.force_handler.active:
                self.force_handler.duration_remaining = self.force_handler.duration
            self.fig.canvas.draw_idle()

    def toggle_continuous_force(self, event):
        """Toggle continuous force mode on or off and update UI accordingly."""
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

    def set_force_type(self, label):
        """
        Change the currently selected force type based on user radio button selection.
        Also show/hide the Gaussian width or frequency slider depending on selection.
        """
        self.force_handler.selected_type = label

        # Hide both sliders by default
        self.gaussian_width_slider_ax.set_visible(False)
        self.frequency_slider_ax.set_visible(False)

        if label == 'Gaussian':
            # Show the Gaussian width slider
            self.gaussian_width_slider_ax.set_visible(True)
        elif label == 'Sinusoidal':
            # Show the frequency slider
            self.frequency_slider_ax.set_visible(True)

        # Redraw figure to reflect changes
        self.fig.canvas.draw_idle()

    def set_force_direction(self, label):
        """Change the force direction based on user selection."""
        self.force_handler.selected_direction = label

    def set_selected_object(self, val):
        """Change which object/mass is selected for force application."""
        self.force_handler.selected_object = int(float(val))
        self.highlight_selected_object()
        self.fig.canvas.draw_idle()

    def set_force_amplitude(self, val):
        """Update force amplitude from slider input."""
        self.force_handler.amplitude = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_force_duration(self, val):
        """Update the base force duration and reset duration_remaining if force inactive."""
        self.force_handler.duration = float(val)
        if not self.force_handler.active:
            self.force_handler.duration_remaining = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_force_duration_remaining(self, val):
        """Adjust duration_remaining if desired (typically not necessary unless fine-tuning)."""
        self.force_handler.duration_remaining = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_gaussian_width(self, val):
        self.force_handler.gaussian_width = float(val)
        self.update_info()
        if self.fig:
            self.fig.canvas.draw_idle()

    def set_sinusoidal_frequency(self, val):
        self.force_handler.sinusoidal_frequency = float(val)
        self.update_info()
        if self.fig:
            self.fig.canvas.draw_idle()

    def cycle_view(self, event):
        """Cycle through predefined camera views."""
        views = [
            ("Default", 45, 15),
            ("Top", 0, 90),
            ("Front", 0, 0),
            ("Side", 90, 0),
            ("Isometric", 45, 35),
        ]

        current = self.view_button.label.get_text().split(': ')[1]
        try:
            current_idx = next(i for i, (name, _, _) in enumerate(views) if name == current)
            next_idx = (current_idx + 1) % len(views)
        except StopIteration:
            next_idx = 0

        name, azim, elev = views[next_idx]
        self.camera['azimuth'] = azim
        self.camera['elevation'] = elev
        self.view_button.label.set_text(f'View: {name}')

        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def cycle_zoom(self, event):
        """Cycle through predefined zoom levels (Fit, Close, Medium, Far)."""
        zoom_levels = {
            "Fit": self.calculate_fit_distance(),
            "Close": 1.0,
            "Medium": 2.0,
            "Far": 4.0
        }

        current = self.zoom_button.label.get_text().split(': ')[1]
        levels = list(zoom_levels.keys())
        current_idx = levels.index(current) if current in levels else 0
        next_level = levels[(current_idx + 1) % len(levels)]

        self.camera['distance'] = zoom_levels[next_level]
        self.zoom_button.label.set_text(f'Zoom: {next_level}')
        self.update_camera()
        self.fig.canvas.draw_idle()

    def calculate_fit_distance(self):
        """Calculate a camera distance that fits all objects comfortably in view."""
        if not self.objects:
            return 2.0
        positions = np.array([obj.position for obj in self.objects])
        max_dist = np.max(np.abs(positions))
        return max_dist * 2.0

    def update_camera(self):
        """Update camera based on self.camera settings."""
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

    def toggle_theme(self, event):
        """Toggle between dark and light theme for the figure and all UI elements."""
        self.dark_mode = not self.dark_mode
        text_color = 'white' if self.dark_mode else 'black'
        background_color = 'black' if self.dark_mode else 'white'
        button_color = 'gray' if self.dark_mode else 'lightgray'

        # Update figure and axes background and text colors
        self.fig.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)

        for spine in self.ax.spines.values():
            spine.set_color(text_color)
        self.ax.tick_params(colors=text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.zaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)

        self.ax.grid(True, alpha=0.3, color=text_color)

        # Update info texts
        self.info_text.set_color(text_color)
        self.force_info_text.set_color(text_color)
        self.camera_info_text.set_color(text_color)

        # Update buttons and sliders
        buttons = [
            self.setup_button,
            self.play_button,
            self.reset_button,
            self.view_button,
            self.zoom_button,
            self.theme_button,
            self.save_button,
            self.force_button,
            self.continuous_force_button
        ]

        for button in buttons:
            button.color = button_color
            button.hovercolor = self.plotter._adjust_color_brightness(button_color, 1.2)
            button.label.set_color(text_color)

        sliders = [
            self.speed_slider,
            self.object_slider,
            self.amplitude_slider,
            self.duration_slider
        ]

        for slider in sliders:
            slider.label.set_color(text_color)
            slider.valtext.set_color(text_color)
            if hasattr(slider, 'poly'):
                slider.poly.set_color(text_color)
            if hasattr(slider, 'vline'):
                slider.vline.set_color(text_color)
            if hasattr(slider, 'hline'):
                slider.hline.set_color(text_color)

        # Update radio buttons
        radios = [self.force_radio, self.direction_radio]
        for radio in radios:
            circles = getattr(radio, '_circles', getattr(radio, 'circles', []))
            for circle in circles:
                circle.set_edgecolor(text_color)
            for label in radio.labels:
                label.set_color(text_color)

        self.update_plots()
        self.highlight_selected_object()

        self.theme_button.label.set_text('Dark Mode' if not self.dark_mode else 'Light Mode')
        self.fig.canvas.draw_idle()

class AnalysisVisualizer:
    """
    The AnalysisVisualizer class provides a graphical user interface (GUI) for analyzing
    previously recorded simulation data. It uses the DataAnalysis class to extract and
    compute insights about the simulation results.

    By loading multiple simulation CSV files, the user can compare different runs,
    visualize their data, and perform various analyses without re-implementing the logic
    that DataAnalysis already provides.

    All plotting is done using the PlottingToolkit (referred to as `plotter`), which
    provides figure and axes objects for plotting. Direct matplotlib calls are only made
    through the `plotter`'s figure (`plotter.fig`) and axes (`plotter.ax`), or on figures/axes
    created by `plotter.create_figure()`. This ensures all plotting is done through the
    given toolkit.
    """

    def __init__(self, main_root):
        # Initialize the main analysis window with reference to the main_root.
        # This window will allow users to load multiple simulation CSV files and
        # perform various analyses on them.
        self.main_root = main_root  # Store a reference to the main (root) window.
        self.analyzer = DataAnalysis()  # Create a DataAnalysis instance for data handling.
        self.loaded_files = {}  # Dictionary mapping file paths to loaded data references.

        # Create a new window (Toplevel) for analysis separate from the main window.
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Analysis")  # Set the window title.

        # Configure the window's size and position based on screen dimensions.
        window_width = int(self.root.winfo_screenwidth() * 0.55)
        window_height = int(self.root.winfo_screenheight() * 0.40)
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Define a color palette for line colors in multi-file plots.
        self.colors = ["red", "green", "blue", "yellow", "orange", "purple"]

        # Build the main GUI layout, including file management and analysis buttons.
        self.setup_gui()

        # Ensure proper handling of window close (returning to main window if needed).
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        """Set up the main GUI elements with responsive layout."""
        # Configure grid weights for the root window
        self.root.grid_rowconfigure(0, weight=1)  # Allow vertical expansion
        self.root.grid_columnconfigure(0, weight=1)  # Allow horizontal expansion

        # Create main frame with padding and configure its grid
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure main frame grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)  # Make file frame take most space
        self.main_frame.grid_rowconfigure(2, weight=0)  # Analysis frame doesn't need to expand

        # Create and configure file management frame
        file_frame = ttk.LabelFrame(self.main_frame, text="File Management", padding="5")
        file_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Configure file frame grid weights
        file_frame.grid_columnconfigure(0, weight=1)
        file_frame.grid_rowconfigure(1, weight=1)  # Let tree frame expand

        # Button frame for file management controls
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 5))
        button_frame.grid_columnconfigure(1, weight=1)  # Space between left and right buttons

        # Left-side buttons
        ttk.Button(button_frame, text="Load Files", command=self.load_files).grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )

        # Right-side buttons in their own frame
        delete_frame = ttk.Frame(button_frame)
        delete_frame.grid(row=0, column=2, sticky="e")

        ttk.Button(delete_frame, text="Delete Selected", command=self.delete_selected).pack(
            side="left", padx=5
        )
        ttk.Button(delete_frame, text="Clear All", command=self.clear_files).pack(
            side="left", padx=(0, 5)
        )

        # Create and configure the tree frame to expand
        tree_frame = ttk.Frame(file_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        # Create the treeview
        self.file_tree = ttk.Treeview(tree_frame, show="headings", selectmode="extended")
        self.file_tree.grid(row=0, column=0, sticky="nsew")

        # Add vertical scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.file_tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.file_tree.configure(yscrollcommand=vsb.set)

        # Add horizontal scrollbar
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.file_tree.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.file_tree.configure(xscrollcommand=hsb.set)

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
            self.file_tree.column(col, width=width, anchor=anchor, stretch=True)  # Allow columns to stretch

        # Bind double-click on color column
        self.file_tree.bind("<Double-1>", self.cycle_color)

        # Create and configure analysis options frame - no vertical expansion needed
        analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="5")
        analysis_frame.grid(row=2, column=0, sticky="ew", pady=5)

        # Configure the analysis frame grid
        for i in range(5):  # First row
            analysis_frame.grid_columnconfigure(i, weight=1)

        # First row of analysis buttons
        row1_buttons = [
            ("View Summary", self.show_summary),
            ("Find Stationary Nodes", self.compare_stationary),
            ("Node Displacement", self.compare_displacement),
            ("Average Displacements", self.plot_nodal_average_displacement),
            ("Movement Patterns", self.compare_movement)
        ]

        for i, (text, command) in enumerate(row1_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=0, column=i, padx=5, pady=5, sticky="ew"
            )

        # Second row of analysis buttons (centered)
        row2_buttons = [
            ("Harmonic Analysis", self.analyze_harmonics),
            ("Frequency Analysis", self.analyze_frequencies)
        ]

        # Calculate starting column to center the second row
        start_col = (len(row1_buttons) - len(row2_buttons)) // 2

        for i, (text, command) in enumerate(row2_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=1, column=start_col + i, padx=5, pady=5, sticky="ew"
            )

    def load_files(self):
        # Open a file dialog to select CSV files.
        files = filedialog.askopenfilenames(
            title="Select Simulation Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        # Load each selected file into the analyzer and display it in the file tree.
        for file_path in files:
            try:
                data = self.analyzer.load_simulation(file_path)
                self.loaded_files[file_path] = data
                summary = self.analyzer.get_simulation_summary(data)
                item = self.file_tree.insert("", "end", values=(
                    os.path.basename(file_path),
                    summary['num_objects'],
                    summary['num_frames'],
                    f"{summary['simulation_time']:.3f} s",
                    self.colors[len(self.loaded_files) % len(self.colors)]
                ))
                self.file_tree.selection_add(item)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file_path}:\n{str(e)}")

    def delete_selected(self):
        # Delete the selected files from both the tree and memory.
        selected_items = self.file_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select files to delete.")
            return

        files_to_remove = []
        for item in selected_items:
            filename = self.file_tree.item(item)["values"][0]
            for path in self.loaded_files:
                if os.path.basename(path) == filename:
                    files_to_remove.append(path)
                    break
            self.file_tree.delete(item)

        for file_path in files_to_remove:
            if file_path in self.loaded_files:
                del self.loaded_files[file_path]

    def get_selected_files(self):
        # Return a list of currently selected files; if none selected, return all.
        selected_items = self.file_tree.selection()
        if not selected_items:
            selected_items = self.file_tree.get_children()

        selected_files = []
        for item in selected_items:
            filename = self.file_tree.item(item)["values"][0]
            for path in self.loaded_files:
                if os.path.basename(path) == filename:
                    selected_files.append(path)
                    break
        return selected_files

    def cycle_color(self, event):
        # On double-clicking the color column, cycle through the available colors.
        item = self.file_tree.identify_row(event.y)
        column = self.file_tree.identify_column(event.x)
        if column == "#5" and item:
            values = list(self.file_tree.item(item)["values"])
            if len(values) >= 5:
                current_color = values[4]
                try:
                    next_index = (self.colors.index(current_color) + 1) % len(self.colors)
                except ValueError:
                    next_index = 0
                values[4] = self.colors[next_index]
                self.file_tree.item(item, values=values)

    def clear_files(self):
        # Clear all loaded files from memory and the tree.
        self.loaded_files.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self.analyzer = DataAnalysis()

    def compare_movement(self):
        # Plot 3D trajectories of nodes from selected simulations for visual comparison.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Use the plotting toolkit to create a 3D plot.
        plotter = PlottingToolkit()
        plot_config = PlotConfig(
            title="Movement Pattern Comparison",
            xlabel="X Position",
            ylabel="Y Position",
            zlabel="Z Position",
            is_3d=True,
            view_preset=ViewPreset.ISOMETRIC,
            interactive_3d=True,
            show_view_controls=True,
            show_animation_controls=True,
            grid=True,
            figure_size=(12, 8)
        )
        legend_kwargs = {
            'bbox_to_anchor': (1.15, 0.5),
            'loc': 'center left',
            'frameon': True,
            'fancybox': True,
            'shadow': True,
            'fontsize': 10,
            'title': 'Simulations'
        }

        first_file = True
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]
            file_path = next((p for p in files if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                summary = self.analyzer.get_simulation_summary(data)
                num_nodes = summary['num_objects']

                for node in range(num_nodes):
                    x, y, z = self.analyzer.get_object_trajectory(data, node)
                    plotter.plot(
                        x, y, z,
                        new_figure=(first_file and node == 0),
                        plot_type='line',
                        color=color,
                        label=f"{filename} - Node {node}" if node == 0 else None,
                        alpha=0.5,
                        line_width=1.5,
                        legend_kwargs=legend_kwargs if node == num_nodes - 1 else None,
                        **vars(plot_config)
                    )
                first_file = False

        plotter.show()

    def compare_displacement(self):
        # Prompt user to select a node, then plot displacement over time for that node.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return
        self.create_node_selection_dialog(files[0])

    def analyze_frequencies(self):
        # Perform FFT-based frequency analysis on the first selected file.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Hide main window to show plots, then restore it after.
        self.root.withdraw()
        try:
            self.plot_multi_node_frequency_analysis(files[0])
        finally:
            self.root.deiconify()

    def plot_multi_node_frequency_analysis(self, file_path):
        """
        Perform frequency analysis (FFT) on displacement data from each node.
        Exclude the endpoints from the frequency analysis by slicing the frequency
        and FFT arrays to remove the first and last data points.
        """
        # Load data and summarize to get number of nodes, etc.
        data = self.loaded_files[file_path]
        summary = self.analyzer.get_simulation_summary(data)
        num_nodes = summary['num_objects']

        # Compute node displacements over time for each node.
        x0, y0, z0 = self.analyzer.get_object_trajectory(data, 0)
        x1, y1, z1 = self.analyzer.get_object_trajectory(data, 1)
        pos0 = np.array([x0[0], y0[0], z0[0]])
        pos1 = np.array([x1[0], y1[0], z1[0]])
        spacing = np.linalg.norm(pos1 - pos0)

        node_displacements = []
        for node in range(num_nodes):
            x, y, z = self.analyzer.get_object_trajectory(data, node)
            positions = np.column_stack([x, y, z])
            initial_pos = positions[0]
            displacements = np.linalg.norm(positions - initial_pos, axis=1)
            node_displacements.append(displacements)

        # Compute time step from the data's Time column.
        time_step = self.analyzer.simulations[data]['Time'].diff().mean()
        if time_step <= 0:
            messagebox.showerror("Error", "Invalid time step in simulation data")
            return

        n_samples = len(node_displacements[0])

        # Compute frequency bins and take half (positive frequencies).
        # Using fftfreq and slicing to half, then exclude endpoints by slicing [1:-1].
        freqs = np.fft.fftfreq(n_samples, time_step)[:n_samples // 2]
        # Exclude the first and last frequency bins to avoid endpoints.
        freqs = freqs[1:-1]  # exclude endpoints

        # Compute FFT for each node and take magnitude of half spectrum, then exclude endpoints.
        fft_results = []
        for displacements in node_displacements:
            fft_result = np.fft.fft(displacements)
            half_fft = np.abs(fft_result[:n_samples // 2])
            half_fft = half_fft[1:-1]  # exclude endpoints
            fft_results.append(half_fft)

        fft_matrix = np.array(fft_results)
        max_val = np.max(fft_matrix)
        if max_val > 0:
            fft_matrix = fft_matrix / max_val

        # Function to compute theoretical frequencies for string modes.
        def compute_theoretical_frequencies(n_modes, spring_constant, mass, spacing, num_nodes):
            if spacing <= 0 or mass <= 0:
                return np.array([])
            length = spacing * (num_nodes - 1)
            if length <= 0:
                return np.array([])
            tension = spring_constant * spacing
            mass_density = mass / spacing
            v_wave = np.sqrt(tension / mass_density)
            return np.array([
                (n / (2 * length)) * v_wave
                for n in range(1, n_modes + 1)
            ])

        # Calculate some example theoretical mode frequencies (for reference).
        theoretical_freqs = compute_theoretical_frequencies(
            10,
            spring_constant=1000.0,  # Example values (could be taken from simulation params)
            mass=0.01,
            spacing=spacing,
            num_nodes=num_nodes
        )

        # Create a PlottingToolkit instance for the heatmap.
        plotter = PlottingToolkit()
        # Plot config for the heatmap of FFT magnitude across nodes and frequency.
        heatmap_config = PlotConfig(
            title="Frequency Content vs Node Position (Endpoints Excluded)",
            xlabel="Frequency (Hz)",
            ylabel="Node Number",
            grid=True,
            figure_size=(12, 10)
        )

        # Create figure and axes using plotter for the heatmap.
        fig1, ax1 = plotter.create_figure(heatmap_config)

        # Create a heatmap of FFT data.
        # Note: freqs is now shorter due to endpoint exclusion.
        im = ax1.pcolormesh(
            freqs,                   # x-axis: frequency
            np.arange(num_nodes),    # y-axis: node index
            fft_matrix,              # magnitude data
            shading='auto',
            cmap='viridis'
        )

        # Add a colorbar for the heatmap.
        fig1.colorbar(im, ax=ax1, label='Normalized Magnitude')

        # Plot theoretical frequency lines (if they fall within the freq range).
        for i, freq in enumerate(theoretical_freqs):
            if freq <= np.max(freqs):
                ax1.axvline(
                    x=freq,
                    color='red',
                    linestyle='--',
                    alpha=0.5
                )
                ax1.text(
                    freq, num_nodes + 0.5,
                    f'Mode {i + 1}\n{freq:.1f} Hz',
                    rotation=90,
                    verticalalignment='bottom',
                    color='red'
                )

        # Create another PlottingToolkit instance for average spectrum.
        plotter2 = PlottingToolkit()
        spectrum_config = PlotConfig(
            title="Average Frequency Spectrum (Endpoints Excluded)",
            xlabel="Frequency (Hz)",
            ylabel="Normalized Magnitude",
            grid=True,
            figure_size=(12, 6)
        )

        # Compute average spectrum over all nodes.
        average_spectrum = np.mean(fft_matrix, axis=0)
        # Plot the average spectrum using the second plotter.
        plotter2.plot(
            freqs,
            average_spectrum,
            new_figure=True,
            color='blue',
            label='Average Spectrum',
            line_style='-',
            line_width=2,
            **vars(spectrum_config)
        )

        # Mark theoretical frequencies on the average spectrum plot.
        for i, freq in enumerate(theoretical_freqs):
            if freq <= np.max(freqs):
                plotter2.ax.axvline(
                    x=freq,
                    color='red',
                    linestyle='--',
                    alpha=0.5,
                    label=f'Mode {i + 1}' if i == 0 else None
                )

        # Identify top peaks in the average spectrum.
        peak_indices = np.argsort(average_spectrum)[-5:][::-1]
        peak_freqs = freqs[peak_indices]
        peak_mags = average_spectrum[peak_indices]

        summary_text = "Dominant Frequencies:\n─────────────────\n"
        for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)):
            summary_text += f"{i + 1}. {freq:.1f} Hz (mag: {mag:.2f})\n"
            plotter2.ax.plot(freq, mag, 'ro')
            plotter2.ax.text(
                freq, mag + 0.05,
                f'{freq:.1f} Hz',
                horizontalalignment='center'
            )

        # Add a summary textbox to the average spectrum plot.
        plotter2.ax.text(
            1.02, 0.95,
            summary_text,
            transform=plotter2.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.8,
                pad=8,
                boxstyle='round'
            )
        )

        # Add a legend to the average spectrum plot.
        plotter2.ax.legend(
            bbox_to_anchor=(1.15, 0.5),
            loc='center left',
            frameon=True,
            fancybox=True,
            shadow=True
        )

        # Show both the heatmap and the average spectrum plots.
        plotter.show()
        plotter2.show()

    def create_node_selection_dialog(self, file_path):
        # Create a dialog to select a node for displacement analysis.
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Node to Analyze")
        dialog.transient(self.root)
        dialog.grab_set()

        data = self.loaded_files[file_path]
        summary = self.analyzer.get_simulation_summary(data)
        num_nodes = summary['num_objects']

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Select a node to analyze:").pack(pady=(0, 5))

        listbox = tk.Listbox(frame, selectmode=tk.SINGLE, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=scrollbar.set)

        for i in range(num_nodes):
            listbox.insert(tk.END, f"Node {i}")

        def on_ok():
            selections = listbox.curselection()
            if selections:
                node_id = int(listbox.get(selections[0]).split()[1])
                dialog.destroy()
                self.root.withdraw()
                self.plot_displacement_comparison(node_id)
                self.root.deiconify()
            else:
                messagebox.showwarning("Warning", "Please select a node.")

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')

    def plot_displacement_comparison(self, node_id):
        # Plot displacement vs. time for the selected node across chosen simulations.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        plotter = PlottingToolkit()
        plot_config = PlotConfig(
            title=f"Displacement Comparison for Node {node_id}",
            xlabel="Time (s)",
            ylabel="Displacement",
            grid=True,
            figure_size=(12, 8)
        )
        legend_kwargs = {
            'bbox_to_anchor': (1.15, 0.5),
            'loc': 'center left',
            'frameon': True,
            'fancybox': True,
            'shadow': True,
            'fontsize': 10,
            'title': 'Simulations'
        }

        first_file = True
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]

            file_path = next((p for p in files if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                try:
                    x, y, z = self.analyzer.get_object_trajectory(data, node_id)
                    positions = np.column_stack([x, y, z])
                    initial_pos = positions[0]
                    displacements = np.linalg.norm(positions - initial_pos, axis=1)
                    time_step = self.analyzer.simulations[data]['Time'].diff().mean()
                    t = np.arange(len(displacements)) * time_step

                    plotter.plot(
                        t,
                        displacements,
                        new_figure=first_file,
                        color=color,
                        label=filename,
                        line_style='-',
                        line_width=1.5,
                        legend_kwargs=legend_kwargs if not first_file else None,
                        **vars(plot_config)
                    )
                    first_file = False
                except Exception as e:
                    messagebox.showerror("Error", f"Error plotting {filename}: {str(e)}")

        plotter.show()

    def plot_nodal_average_displacement(self):
        # Plot average displacement per node for selected simulations.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        plotter = PlottingToolkit()
        plot_config = PlotConfig(
            title="Average Node Displacement Comparison",
            xlabel="Node ID",
            ylabel="Average Displacement",
            grid=True,
            figure_size=(12, 6)
        )
        legend_kwargs = {
            'bbox_to_anchor': (1.15, 0.5),
            'loc': 'center left',
            'frameon': True,
            'fancybox': True,
            'shadow': True,
            'fontsize': 10,
            'title': 'Simulations'
        }

        first_file = True
        files_to_plot = self.get_selected_files()

        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]

            file_path = next((p for p in files_to_plot if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                node_ids, avg_displacements = self.analyzer.get_average_displacements(data)
                plotter.plot(
                    node_ids,
                    avg_displacements,
                    new_figure=first_file,
                    color=color,
                    label=filename,
                    plot_type='line',
                    marker_style='o',
                    line_style='-',
                    line_width=1.5,
                    legend_kwargs=legend_kwargs if not first_file else None,
                    **vars(plot_config)
                )
                first_file = False

        plotter.show()

    def compare_stationary(self):
        # Show which nodes remain stationary across selected simulations.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        comparison = {}
        for file_path in files:
            data = self.loaded_files[file_path]
            comparison[os.path.basename(file_path)] = self.analyzer.find_stationary_nodes(data)

        window = tk.Toplevel(self.root)
        window.title("Stationary Nodes Comparison")
        text = tk.Text(window, wrap=tk.WORD, width=60, height=30)
        text.pack(padx=10, pady=10)

        for filename, nodes in comparison.items():
            text.insert("end", f"\nSimulation: {filename}\n{'=' * 50}\n")
            if nodes:
                text.insert("end", f"Found {len(nodes)} stationary nodes:\n")
                for node_id, pos in nodes.items():
                    displacement = np.linalg.norm(pos)
                    text.insert("end", f"\nNode {node_id}:\n")
                    text.insert("end", f"  Position: {pos}\n")
                    text.insert("end", f"  Displacement: {displacement:.6f}\n")
            else:
                text.insert("end", "No stationary nodes found\n")

        text.config(state=tk.DISABLED)

    def show_summary(self):
        # Display a summary of the selected simulations in a text window.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        window = tk.Toplevel(self.root)
        window.title("Simulation Summary")
        text = tk.Text(window, wrap=tk.WORD, width=60, height=30)
        text.pack(padx=10, pady=10)

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

    def analyze_harmonics(self):
        # Perform harmonic analysis on the selected files.
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Hide main window while showing harmonic analysis plots.
        self.root.withdraw()
        try:
            self.plot_harmonic_correlation(files)
        finally:
            self.root.deiconify()

    def plot_harmonic_correlation(self, files):
        # Plot correlation of displacement patterns with harmonic modes for each selected file.
        plotter = PlottingToolkit()
        num_files = len(files)

        for idx, file_path in enumerate(files):
            data = self.loaded_files[file_path]
            filename = os.path.basename(file_path)

            harmonics, correlations = self.analyzer.get_harmonic_correlation(data)

            plot_config = PlotConfig(
                title=f"Harmonic Analysis - {filename}",
                xlabel="Harmonic Number",
                ylabel="Correlation (%)",
                grid=True,
                figure_size=(15, 5 * ((num_files + 1) // 2))
            )

            plotter.plot(
                harmonics,
                correlations,
                plot_type='bar',
                color='steelblue',
                alpha=0.7,
                new_figure=(idx == 0),
                **vars(plot_config)
            )

            # Annotate bars.
            for i, correlation in enumerate(correlations):
                plotter.ax.text(
                    i + 1, correlation + 1,
                    f'{correlation:.1f}%',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=9
                )

            # Identify top 3 harmonics.
            sorted_indices = np.argsort(correlations)[::-1]
            summary_text = (
                f"Dominant Harmonics:\n"
                f"1st: {harmonics[sorted_indices[0]]}th ({correlations[sorted_indices[0]]:.1f}%)\n"
                f"2nd: {harmonics[sorted_indices[1]]}th ({correlations[sorted_indices[1]]:.1f}%)\n"
                f"3rd: {harmonics[sorted_indices[2]]}th ({correlations[sorted_indices[2]]:.1f}%)"
            )

            plotter.ax.text(
                1.02, 0.95,
                summary_text,
                transform=plotter.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=8, boxstyle='round')
            )

            plotter.update_plot_bounds(
                x_range=(0.5, len(harmonics) + 0.5),
                y_range=(0, max(correlations) * 1.15)
            )

        if plotter.fig:
            plotter.fig.tight_layout()
        plotter.show()

    def run(self):
        # Run the Tkinter event loop for this analysis window.
        self.root.mainloop()

    def on_closing(self):
        # When the window is closed, destroy it and show the main_root window again if available.
        self.root.destroy()
        if self.main_root:
            self.main_root.deiconify()
