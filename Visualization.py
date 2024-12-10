import os
import time
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum
from tkinter import ttk, messagebox
from typing import List, Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

from data_handling import DataAnalysis

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

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
class PlotConfig:
    """Configuration settings for plot customization."""
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None  # For 3D plots
    y2label: Optional[str] = None  # For dual axis plots
    xscale: str = 'linear'
    yscale: str = 'linear'
    y2scale: str = 'linear'
    grid: bool = True
    legend_position: str = 'right'
    figure_size: Tuple[float, float] = (10, 6)
    style: str = 'default'
    colors: Optional[List[str]] = None

    # 3D specific settings
    is_3d: bool = False
    view_preset: Optional[ViewPreset] = None
    interactive_3d: bool = True
    show_view_controls: bool = False
    show_animation_controls: bool = False
    auto_rotate: bool = False
    rotation_speed: float = 1.0


class PlottingToolkit:
    """Enhanced plotting toolkit with 2D and 3D visualization capabilities."""
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
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.current_rotation = 0
        self.animation_running = False
        self.fig = None
        self.ax = None
        self.ui_elements = {k: {} for k in ['buttons','sliders','radio_buttons','text','legends']}
        self._mouse_button = None
        self._mouse_x = None
        self._mouse_y = None
        self.camera = {'distance': 2.0, 'azimuth': 45, 'elevation': 15, 'rotation_speed': 0.3, 'zoom_speed': 0.1}
        self.config = dict(self.DEFAULT_CONFIG)
        self.dark_mode = self.config['dark_mode']
        self.current_theme = 'dark' if self.dark_mode else 'light'

    def create_figure(self, config: PlotConfig) -> Tuple[plt.Figure, plt.Axes]:
        # Ensure a new figure is created cleanly
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        plt.style.use(config.style)
        self.fig = plt.figure(figsize=config.figure_size)

        if config.is_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
            if config.zlabel:
                self.ax.set_zlabel(config.zlabel)
            if config.view_preset:
                self.ax.view_init(
                    elev=config.view_preset.value["elev"],
                    azim=config.view_preset.value["azim"]
                )
        else:
            self.ax = self.fig.add_subplot(111)

        # Adjust for side panel if needed
        if config.legend_position == 'right' or config.show_view_controls or config.show_animation_controls:
            plt.subplots_adjust(right=0.85)

        if config.title:
            self.ax.set_title(config.title)
        if config.xlabel:
            self.ax.set_xlabel(config.xlabel)
        if config.ylabel:
            self.ax.set_ylabel(config.ylabel)

        if not config.is_3d:
            self.ax.set_xscale(config.xscale)
            self.ax.set_yscale(config.yscale)

        self.ax.grid(config.grid, alpha=0.3)

        # Ensure figure redraw to avoid premature crashing
        self.fig.canvas.draw_idle()

        return self.fig, self.ax

    def plot(self, x, y, z=None, new_figure=True, legend_kwargs=None, **kwargs):
        config = PlotConfig(**{k: v for k, v in kwargs.items() if k in PlotConfig.__annotations__})

        if new_figure or self.fig is None:
            self.fig, self.ax = self.create_figure(config)

        legend_kwargs = legend_kwargs or {}
        legend_kwargs.setdefault('loc', 'best')
        legend_kwargs.setdefault('frameon', True)
        legend_kwargs.setdefault('fancybox', True)

        plot_type = kwargs.get('plot_type', 'line')
        color = kwargs.get('color', self.default_colors[0])
        alpha = kwargs.get('alpha', 1.0)

        if z is not None:
            # 3D plotting
            if plot_type == 'scatter':
                self.ax.scatter(x, y, z, marker=kwargs.get('marker_style', 'o'), color=color, alpha=alpha,
                                label=kwargs.get('label'))
            elif plot_type == 'line':
                self.ax.plot(x, y, z, linestyle=kwargs.get('line_style', '-'), linewidth=kwargs.get('line_width', 1.5),
                             color=color, alpha=alpha, label=kwargs.get('label'))
            elif plot_type == 'surface':
                surf = self.ax.plot_surface(x, y, z, cmap=kwargs.get('color_map', 'viridis'), alpha=alpha, linewidth=0)
                if kwargs.get('colorbar', True):
                    self.fig.colorbar(surf)
        else:
            # 2D plotting
            if plot_type == 'line':
                self.ax.plot(x, y, color=color, linestyle=kwargs.get('line_style', '-'),
                             marker=kwargs.get('marker_style', None),
                             alpha=alpha, linewidth=kwargs.get('line_width', 1.5), label=kwargs.get('label'))
            elif plot_type == 'bar':
                self.ax.bar(x, y, color=color, alpha=alpha, label=kwargs.get('label'))
            elif plot_type == 'scatter':
                self.ax.scatter(x, y, color=color, marker=kwargs.get('marker_style', 'o'), alpha=alpha,
                                label=kwargs.get('label'))

        if 'label' in kwargs:
            self.ax.legend(**legend_kwargs)

    def update_data(self, plot_object, x, y, z=None):
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(self.ax, Axes3D):
            if hasattr(plot_object, '_offsets3d'):
                plot_object._offsets3d = (np.array(x), np.array(y), np.array(z))
            else:
                plot_object.set_data(x, y)
                plot_object.set_3d_properties(z)
        else:
            if hasattr(plot_object, 'set_offsets'):
                plot_object.set_offsets(np.column_stack([x, y]))
            else:
                plot_object.set_data(x, y)

    def add_text(self, x, y, text, **kwargs):
        text_id = kwargs.pop('text_id', None)
        txt = self.ax.text2D(x, y, text, transform=self.ax.transAxes, **kwargs)
        if text_id:
            self.ui_elements['text'][text_id] = txt
        return txt

    def add_button(self, position, label, callback, color='gray', text_color='white',
                   hover_color=None, button_id=None):
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
        from matplotlib.widgets import RadioButtons
        ax_radio = plt.axes(position)
        radio = RadioButtons(ax_radio, labels, active=active, activecolor=active_color)

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
        import matplotlib.colors as mcolors
        try:
            rgb = mcolors.to_rgb(color)
            adjusted = tuple(min(1.0, c * factor) for c in rgb)
            return adjusted
        except ValueError:
            return color

    def update_ui_theme(self, dark_mode=True):
        self.dark_mode = dark_mode
        self.current_theme = 'dark' if dark_mode else 'light'
        background = 'black' if dark_mode else 'white'
        text_color = 'white' if dark_mode else 'black'
        btn_color = 'gray' if dark_mode else 'lightgray'

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
        for element_type in self.ui_elements:
            for element_info in self.ui_elements[element_type].values():
                if hasattr(element_info['object'], 'ax'):
                    element_info['object'].ax.remove()
        self.ui_elements = {key: {} for key in self.ui_elements}

    def remove_element(self, element_type, element_id):
        if element_id in self.ui_elements[element_type]:
            element = self.ui_elements[element_type][element_id]
            if hasattr(element['object'], 'ax'):
                element['object'].ax.remove()
            del self.ui_elements[element_type][element_id]

    def update_view(self, camera):
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(self.ax, Axes3D):
            self.ax.view_init(elev=camera['elevation'], azim=camera['azimuth'])
            self.ax.dist = camera['distance']
            if self.fig:
                self.fig.canvas.draw_idle()

    def show(self):
        if self.fig is not None:
            plt.show()
            self.fig = None
            self.ax = None

    def clear(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def update_text(self, text_id, new_text):
        text_obj = self.ui_elements['text'].get(text_id, None)
        if text_obj:
            text_obj.set_text(new_text)
            if self.fig:
                self.fig.canvas.draw_idle()

    def update_plot_bounds(self, x_range, y_range, z_range=None):
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
        pass


@dataclass
class SimulationParameters:
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
        self.selected_type = 'Single Mass'
        self.selected_direction = 'Up/Down (Z)'

    def check_duration(self, iterations_per_frame):
        if not self.continuous and self.active:
            time_elapsed = self.physics.dt * iterations_per_frame
            self.duration_remaining = max(0.0, self.duration_remaining - time_elapsed)
            if self.duration_remaining <= 0:
                self.deactivate()
                return True
            self.apply(self.duration_remaining)
        return False

    def toggle(self):
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
            magnitude = np.sin(2 * np.pi * self.physics.time) * self.amplitude
            self.physics.apply_force(self.selected_object, direction * magnitude, duration)
        elif self.selected_type == 'Gaussian':
            for i in range(1, len(self.objects) - 1):
                magnitude = np.exp(-(i - self.selected_object) ** 2 / (len(self.objects) / 8) ** 2) * self.amplitude
                self.physics.apply_force(i, direction * magnitude, duration)

    def deactivate(self):
        self.active = False
        self.duration_remaining = 0.0
        for obj_id in range(len(self.objects)):
            self.physics.clear_force(obj_id)


class StringSimulationSetup:
    def __init__(self, main_root):
        self.main_root = main_root
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Setup")
        self.default_params = SimulationParameters()
        self.init_variables()
        self.root.minsize(600, 500)
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
        style = ttk.Style()
        self.base_header_size = 14
        self.base_text_size = 10
        self.base_button_size = 11
        style.configure("Header.TLabel", font=("Arial", self.base_header_size, "bold"), anchor="center")
        style.configure("Normal.TLabel", font=("Arial", self.base_text_size), anchor="w")
        style.configure("Setup.TButton", font=("Arial", self.base_button_size), padding=10)

    def on_window_resize(self, event):
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

        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Parameters")
        self.setup_basic_parameters(basic_frame)

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
        props_frame = ttk.LabelFrame(parent, text="String Properties", padding="10")
        props_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(props_frame, text="Number of segments:").grid(row=0, column=0, padx=5, pady=5)
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

    def set_simulation_speed(self, val):
        self.iteration_count = int(val)

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.play_button.label.set_text('Resume' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()

    def reset_simulation(self, event):
        self.paused = True
        self.play_button.label.set_text('Start')
        self.physics.time = self.initial_physics_time
        for i, obj in enumerate(self.objects):
            obj.position = self.original_positions[i].copy()
            obj.velocity = self.original_velocities[i].copy()
            obj.acceleration = np.zeros(3)
        self.physics.data_recorder.clear_history()
        self.update_plots()
        self.fig.canvas.draw_idle()

    def save_simulation_data(self, event):
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

    def validate_parameters(self):
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
            messagebox.showerror("Invalid Parameters", str(e))
            return False

    def get_parameters(self):
        if not hasattr(self, 'simulation_params'):
            return None
        try:
            if self.num_segments_var.get() < 2:
                raise ValueError("Number of segments must be at least 2")
            if self.spring_constant_var.get() <= 0:
                raise ValueError("Spring constant must be positive")
            if self.mass_var.get() <= 0:
                raise ValueError("Mass must be positive")
            if self.dt_var.get() <= 0:
                raise ValueError("Time step must be positive")
            if self.force_magnitude_var.get() < 0:
                raise ValueError("Force magnitude cannot be negative")

            params = SimulationParameters(
                num_segments=self.num_segments_var.get(),
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
        if not self.validate_parameters():
            return
        self.simulation_params = SimulationParameters(
            num_segments=self.num_segments_var.get(),
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
    """Main visualization class handling display and interaction."""

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

        # Initialize force handler
        self.force_handler = ForceHandler(physics_model, objects, dark_mode)

        # Initialize plotter before setup_visualization
        self.plotter = PlottingToolkit()

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
        plt.style.use('dark_background' if self.dark_mode else 'default')
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.20, 0.15, 0.6, 0.8])

        # Assign fig and ax to the plotter so it can create UI elements
        self.plotter.fig = self.fig
        self.plotter.ax = self.ax

        self.setup_plots()
        self.setup_camera()
        self.setup_enhanced_controls()
        self._connect_events()

        text_color = 'white' if self.dark_mode else 'black'
        # Create force_info and camera_info texts and register them in plotter's ui_elements:
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
        pass

    def setup_enhanced_controls(self):
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

        self.speed_slider = Slider(
            plt.axes([0.24, 0.02, 0.44, 0.02]),
            'Simulation Speed',
            1, 1000,
            valinit=self.iteration_count,
            valfmt='%d steps/frame'
        )
        self.speed_slider.on_changed(self.set_simulation_speed)

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

        self.play_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset_simulation)
        self.view_button.on_clicked(self.cycle_view)
        self.zoom_button.on_clicked(self.cycle_zoom)
        self.theme_button.on_clicked(self.toggle_theme)
        self.save_button.on_clicked(self.save_simulation_data)

        self.fig.text(left_panel_start, 0.9, 'Force Controls', color=text_color, fontsize=10)

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
            0.1, 50.0,
            valinit=self.force_handler.amplitude
        )
        self.amplitude_slider.on_changed(self.set_force_amplitude)

        self.duration_slider = Slider(
            plt.axes([left_panel_start, 0.30, panel_width, 0.02]),
            'Duration',
            0.01, 10.0,
            valinit=self.force_handler.duration,
            valfmt='%.2f s'
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
        self.should_restart = True
        plt.close(self.fig)
        if self.main_root:
            self.main_root.deiconify()

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

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

            for _ in range(self.iteration_count):
                self.physics.step(self.integration_method)

            if self.force_handler.check_duration(self.iteration_count):
                self.force_button.label.set_text('Apply Force')
                self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'
                self.fig.canvas.draw_idle()

            if self.force_handler.continuous and self.force_handler.active:
                self.force_handler.apply()

        self.update_plots()
        self.update_info()
        self.highlight_selected_object()

        # Return no blit artists because 3D + blit is problematic
        return []

    def update_plots(self):
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
        time_info = self.physics.data_recorder.get_time_info()
        dt_per_frame = self.physics.dt * self.iteration_count
        state = 'PAUSED' if self.paused else 'RUNNING'

        info_text = (
            f"{state}\n"
            f"Frame: {self.animation_frame_count}\n"
            f"FPS: {min(self.fps, 60.0):.1f}\n"
            f"Time: {time_info['simulation_time']:.3f}s\n"
            f"Integration: {self.integration_method.title()}\n"
            f"dt/step: {self.physics.dt:.6f}s\n"
            f"dt/frame: {dt_per_frame:.6f}s\n"
            f"Steps/Frame: {self.iteration_count}"
        )
        self.info_text.set_text(info_text)
        self.update_force_info()

    def update_force_info(self):
        # Retrieve forces from the physics model for the selected object
        external_force, spring_forces, total_force = self.physics.get_forces_on_object(
            self.force_handler.selected_object)

        # Format force information as requested
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
        text = (
            f"Camera Azim: {self.camera['azimuth']:.1f}\n"
            f"Camera Elev: {self.camera['elevation']:.1f}\n"
            f"Camera Dist: {self.camera['distance']:.1f}"
        )
        self.plotter.update_text('camera_info', text)

    def highlight_selected_object(self):
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
        self.anim = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=20,
            blit=False,  # Disable blit for proper 3D and text updates
            cache_frame_data=False
        )
        plt.show()
        return self.should_restart

    def set_simulation_speed(self, val):
        self.iteration_count = int(float(val))
        self.update_info()

    def toggle_pause(self, event):
        if not self.simulation_started and not self.paused:
            return
        self.paused = not self.paused
        if not self.paused and not self.simulation_started:
            self.simulation_started = True
            self.physics.start_simulation()
        button_text = 'Start' if not self.simulation_started else ('Resume' if self.paused else 'Pause')
        self.play_button.label.set_text(button_text)
        self.fig.canvas.draw_idle()

    def reset_simulation(self, event):
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
            self.fig.canvas.draw_idle()

    def toggle_continuous_force(self, event):
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
        self.force_handler.selected_type = label

    def set_force_direction(self, label):
        self.force_handler.selected_direction = label

    def set_selected_object(self, val):
        self.force_handler.selected_object = int(float(val))
        self.highlight_selected_object()
        self.fig.canvas.draw_idle()

    def set_force_amplitude(self, val):
        self.force_handler.amplitude = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_force_duration(self, val):
        self.force_handler.duration = float(val)
        self.force_handler.duration_remaining = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_force_duration_remaining(self, val):
        self.force_handler.duration_remaining = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def cycle_view(self, event):
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
        if not self.objects:
            return 2.0
        positions = np.array([obj.position for obj in self.objects])
        max_dist = np.max(np.abs(positions))
        return max_dist * 2.0

    def update_camera(self):
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
        self.dark_mode = not self.dark_mode
        self.plotter.update_ui_theme(self.dark_mode)
        self.update_plots()
        self.fig.canvas.draw_idle()
        self.theme_button.label.set_text('Dark Mode' if not self.dark_mode else 'Light Mode')
        self.highlight_selected_object()


class AnalysisVisualizer:
    """
    Interactive visualization system for analyzing string simulation data.
    Uses the DataAnalysis functions without re-implementing logic that data_handling already provides.
    """

    def __init__(self, main_root):
        self.analyzer = DataAnalysis()  # Data analysis engine
        self.loaded_files = {}  # Maps file paths to their simulation data
        self.main_root = main_root
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Analysis")

        # Window dimensions
        window_width = int(self.root.winfo_screenwidth() * 0.55)
        window_height = int(self.root.winfo_screenheight() * 0.40)
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Color palette for visualization
        self.colors = ["red", "green", "blue", "yellow", "orange", "purple"]

        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        file_frame = ttk.LabelFrame(self.main_frame, text="File Management", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", pady=5)

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

        self.file_tree = ttk.Treeview(file_frame, show="headings", height=3, selectmode="extended")
        self.file_tree.grid(row=1, column=0, sticky="ew", pady=5)

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

        analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="5")
        analysis_frame.grid(row=1, column=0, sticky="ew", pady=5)

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

        for i, (text, command) in enumerate(row1_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=0, column=i, padx=5, pady=5, sticky="ew"
            )
            analysis_frame.columnconfigure(i, weight=1)

        start_col = (len(row1_buttons) - len(row2_buttons)) // 2
        for i, (text, command) in enumerate(row2_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=1, column=start_col + i, padx=5, pady=5, sticky="ew"
            )

    def load_files(self):
        files = filedialog.askopenfilenames(
            title="Select Simulation Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
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
        self.loaded_files.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self.analyzer = DataAnalysis()

    def compare_movement(self):
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

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
                        new_figure=first_file and node == 0,
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
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return
        self.create_node_selection_dialog(files[0])

    def create_node_selection_dialog(self, file_path):
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
                    text.insert("end", f"\nNode {node_id}:\n")
                    text.insert("end", f"  Position: {pos}\n")
                    text.insert("end", f"  Displacement: {np.linalg.norm(pos):.6f}\n")
            else:
                text.insert("end", "No stationary nodes found\n")

        text.config(state=tk.DISABLED)

    def show_summary(self):
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
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        self.root.withdraw()

        try:
            self.plot_harmonic_correlation(files)
        finally:
            self.root.deiconify()

    def plot_harmonic_correlation(self, files):
        fig = plt.figure(figsize=(15, 5))
        plt.title("Harmonic Analysis Placeholder")
        plt.show()

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        self.root.destroy()
        if self.main_root:
            self.main_root.deiconify()
