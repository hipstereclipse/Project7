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
from data_handling import SimulationDataRecorder
from Mass import SimulationObject
from Integrator import INTEGRATORS
from Physics_Engine import PhysicsEngine
from math import sqrt, pi

class ViewPreset(Enum):
    DEFAULT = {"name": "Default", "azim": 45, "elev": 15}
    TOP = {"name": "Top (XY)", "azim": 0, "elev": 90}
    FRONT = {"name": "Front (XZ)", "azim": 0, "elev": 0}
    SIDE = {"name": "Side (YZ)", "azim": 90, "elev": 0}
    ISOMETRIC = {"name": "Isometric", "azim": 45, "elev": 35}
    FREE = {"name": "Free", "azim": None, "elev": None}

@dataclass
class PlotConfig:
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
    is_3d: bool = False
    view_preset: Optional[ViewPreset] = None
    interactive_3d: bool = True
    show_view_controls: bool = False
    show_animation_controls: bool = False
    auto_rotate: bool = False
    rotation_speed: float = 1.0
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
    spacing: dict = field(default_factory=lambda: {
        'margins': {
            'left': None,
            'right': None,
            'top': None,
            'bottom': None,
            'wspace': 0.2,
            'hspace': 0.2
        },
        'elements': {
            'title_spacing': None,
            'xlabel_spacing': None,
            'ylabel_spacing': None,
            'zlabel_spacing': None,
            'legend_spacing': None,
            'tick_spacing': None
        }
    })
    legend_config: dict = field(default_factory=lambda: {
        'position': None,
        'anchor': None,
        'columns': 1,
        'fontsize': 10,
        'framewidth': 1,
        'framealpha': 0.8,
        'spacing': 0.5,
        'title_fontsize': 12
    })

    def get_margin(self, side: str) -> float:
        margin = self.spacing['margins'][side]
        if margin is not None:
            return margin
        legacy_map = {
            'left': self.subplot_left_pad,
            'right': self.subplot_right_pad,
            'top': self.subplot_top_pad,
            'bottom': self.subplot_bottom_pad
        }
        return legacy_map.get(side, 0.1)

    def get_element_spacing(self, element: str) -> float:
        spacing = self.spacing['elements'][f'{element}_spacing']
        if spacing is not None:
            return spacing
        legacy_map = {
            'title': self.title_pad,
            'xlabel': self.xlabel_pad,
            'ylabel': self.ylabel_pad,
            'zlabel': self.zlabel_pad,
            'legend': self.legend_pad,
            'tick': self.tick_pad
        }
        return legacy_map.get(element, 5.0)


class PlottingToolkit:
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
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        plt.style.use(config.style)
        self.fig = plt.figure(figsize=config.figure_size)

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

        self.fig.subplots_adjust(
            left=config.get_margin('left'),
            right=config.get_margin('right'),
            top=config.get_margin('top'),
            bottom=config.get_margin('bottom'),
            wspace=config.spacing['margins']['wspace'],
            hspace=config.spacing['margins']['hspace']
        )

        if config.title:
            self.ax.set_title(config.title, pad=config.get_element_spacing('title'))
        if config.xlabel:
            self.ax.set_xlabel(config.xlabel, labelpad=config.get_element_spacing('xlabel'))
        if config.ylabel:
            self.ax.set_ylabel(config.ylabel, labelpad=config.get_element_spacing('ylabel'))

        self.ax.tick_params(pad=config.get_element_spacing('tick'))

        if not config.is_3d:
            self.ax.set_xscale(config.xscale)
            self.ax.set_yscale(config.yscale)

        self.ax.grid(config.grid, alpha=0.3)

        self.fig.canvas.draw_idle()
        return self.fig, self.ax

    def plot(self, x, y, z=None, new_figure=True, legend_kwargs=None, **kwargs):
        config = PlotConfig(**{k: v for k, v in kwargs.items() if k in PlotConfig.__annotations__})

        if new_figure or self.fig is None:
            self.fig, self.ax = self.create_figure(config)

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

        if z is not None and config.is_3d:
            self._plot_3d(x, y, z, plot_type, color, alpha, kwargs)
        else:
            self._plot_2d(x, y, plot_type, color, alpha, kwargs)

        if 'label' in kwargs:
            self.ax.legend(**default_legend_kwargs)

    def _plot_2d(self, x, y, plot_type, color, alpha, kwargs):
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

    def update_spacing(self, config: PlotConfig):
        if not self.fig or not self.ax:
            return

        self.fig.subplots_adjust(
            left=config.get_margin('left'),
            right=config.get_margin('right'),
            top=config.get_margin('top'),
            bottom=config.get_margin('bottom'),
            wspace=config.spacing['margins']['wspace'],
            hspace=config.spacing['margins']['hspace']
        )

        if self.ax.get_title():
            self.ax.set_title(self.ax.get_title(), pad=config.get_element_spacing('title'))
        if self.ax.get_xlabel():
            self.ax.set_xlabel(self.ax.get_xlabel(), labelpad=config.get_element_spacing('xlabel'))
        if self.ax.get_ylabel():
            self.ax.set_ylabel(self.ax.get_ylabel(), labelpad=config.get_element_spacing('ylabel'))
        if hasattr(self.ax, 'get_zlabel') and self.ax.get_zlabel():
            self.ax.set_zlabel(self.ax.get_zlabel(), labelpad=config.get_element_spacing('zlabel'))

        self.ax.tick_params(pad=config.get_element_spacing('tick'))
        self.fig.canvas.draw_idle()

    def add_text(self, x, y, text, **kwargs):
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


@dataclass
class SimulationParameters:
    num_segments: int = 25
    spring_constant: float = 1000.0
    custom_equilibrium_length: float = field(default=None)
    mass: float = 0.01
    dt: float = 0.0001
    start_point: np.ndarray = field(default_factory=lambda: np.array([-1.00000, 0.00000, 0.00000]))
    end_point: np.ndarray = field(default_factory=lambda: np.array([1.00000, 0.00000, 0.00000]))
    integration_method: str = 'leapfrog'
    applied_force: np.ndarray = field(default_factory=lambda: np.array([0.00000, 0.00000, 1.00000]))
    dark_mode: bool = True

    # New parameters for alternative mode
    parameter_mode: str = 'simulation'  # 'Simulation' or 'string_params'
    L0: float = 2.0      # Unstretched total length
    mu0: float = 0.01     # linear mass density
    alpha: float = 1000.0 # stiffness (EA)
    N: int = 25           # number of masses in string_params mode
    use_L_not_T: bool = True
    L: float = 2.0        # If use_L_not_T is True
    T: float = 0.0        # If use_L_not_T is False

    @property
    def equilibrium_length(self) -> float:
        if self.parameter_mode == 'simulation':
            if self.custom_equilibrium_length is not None:
                return self.custom_equilibrium_length
            total_length = np.linalg.norm(self.end_point - self.start_point)
            return total_length / (self.num_segments-1)
        else:
            # string_params mode
            # Computes equilibrium length from L0, alpha, N, and either L or T
            # k per segment = alpha*(N)/L0, mass per node = (mu0 * L0)/(N+1)
            k_segment = self.alpha * (self.N-1) / self.L0
            if self.use_L_not_T:
                # L given
                eq_length_per_segment = self.L / (self.N-1)
                return eq_length_per_segment
            else:
                # T given
                # Solves L = L0 + T/k => eq_length_per_segment = L/N
                Lfinal = self.L0 + self.T / k_segment
                return Lfinal / (self.N-1)

    @property
    def computed_mass(self) -> float:
        # In string_params mode, mass per node = (mu0 * L0)/(N+1)
        if self.parameter_mode == 'simulation':
            return self.mass
        else:
            return (self.mu0 * self.L0) / (self.N)

    @property
    def computed_k(self) -> float:
        # k per segment = alpha*(N)/L0 in string_params mode
        if self.parameter_mode == 'simulation':
            return self.spring_constant
        else:
            return (self.alpha * (self.N-1)) / self.L0


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
        self.selected_object = ((len(self.objects)+1) // 2)-1
        self.gaussian_width = len(self.objects) / 8.0
        self.sinusoidal_frequency = 10.0

        self.types = {
            'Single Mass': lambda t, x: np.array([0.0, 0.0, 1.0]),
            'Sinusoidal': lambda t, x: np.array([0.0, 0.0, np.sin(2 * np.pi * self.sinusoidal_frequency * t)]),
            'Gaussian': lambda t, x: np.array([0.0, 0.0,
                                               np.exp(-(x - len(self.objects) // 2) ** 2 / self.gaussian_width ** 2)])
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
            return ('Apply Force', 'darkgray' if not self.dark_mode else 'gray')
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
            magnitude = np.sin(2 * np.pi * self.sinusoidal_frequency * self.physics.time) * self.amplitude
            self.physics.apply_force(self.selected_object, direction * magnitude, duration)

        elif self.selected_type == 'Gaussian':
            for i in range(1, len(self.objects) - 1):
                magnitude = np.exp(-(i - self.selected_object) ** 2 / self.gaussian_width ** 2) * self.amplitude
                self.physics.apply_force(i, direction * magnitude, duration)

    def deactivate(self):
        self.active = False
        self.duration_remaining = self.duration
        for obj_id in range(len(self.objects)):
            self.physics.clear_force(obj_id)


class StringSimulationSetup:
    """
    This GUI to lets the user select simulation parameters.
    GUI has "Basic Parameters" and "Advanced Parameters" tabs.
    At the top of the Basic Parameters tab, The user can choose between "Simulation"
    and "String Params", and depending on that choice, it shows the appropriate parameters
    in the "String Properties" frame beneath.
    """

    def __init__(self, main_root):
        self.main_root = main_root
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Setup")
        self.default_params = SimulationParameters()
        self.init_variables()

        # Set window sizing and behavior like originally specified
        self.root.minsize(550, 550)
        self.root.resizable(False, False)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 550
        window_height = 550
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 4) - (window_height // 4)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.create_responsive_styles()
        self.setup_gui()
        self.simulation_params = None

    def init_variables(self):
        # Common variables
        self.num_segments_var = tk.IntVar(self.root)
        self.num_segments_var.set(self.default_params.num_segments)
        self.spring_constant_var = tk.DoubleVar(self.root)
        self.spring_constant_var.set(self.default_params.spring_constant)
        self.mass_var = tk.DoubleVar(self.root)
        self.mass_var.set(self.default_params.mass)
        self.dt_var = tk.DoubleVar(self.root)
        self.dt_var.set(self.default_params.dt)
        self.equilibrium_length_var = tk.DoubleVar(self.root)
        self.equilibrium_length_var.set(self.default_params.equilibrium_length)
        force_magnitude = float(np.linalg.norm(self.default_params.applied_force))
        self.force_magnitude_var = tk.DoubleVar(self.root)
        self.force_magnitude_var.set(force_magnitude)
        self.integration_var = tk.StringVar(self.root)
        self.integration_var.set(self.default_params.integration_method)
        self.dark_mode_var = tk.BooleanVar(self.root)
        self.dark_mode_var.set(self.default_params.dark_mode)

        # Parameter mode
        self.parameter_mode_var = tk.StringVar(self.root)
        self.parameter_mode_var.set(self.default_params.parameter_mode)

        # String parameters
        self.L0_var = tk.DoubleVar(self.root)
        self.L0_var.set(self.default_params.L0)
        self.mu0_var = tk.DoubleVar(self.root)
        self.mu0_var.set(self.default_params.mu0)
        self.alpha_var = tk.DoubleVar(self.root)
        self.alpha_var.set(self.default_params.alpha)
        self.N_var = tk.IntVar(self.root)
        self.N_var.set(self.default_params.N)
        self.use_L_not_T_var = tk.BooleanVar(self.root)
        self.use_L_not_T_var.set(self.default_params.use_L_not_T)
        self.L_var = tk.DoubleVar(self.root)
        self.L_var.set(self.default_params.L)
        self.T_var = tk.DoubleVar(self.root)
        self.T_var.set(self.default_params.T)

    def create_responsive_styles(self):
        style = ttk.Style()
        self.base_header_size = 14
        self.base_text_size = 10
        self.base_button_size = 11
        style.configure("Header.TLabel", font=("Arial", self.base_header_size, "bold"), anchor="center")
        style.configure("Normal.TLabel", font=("Arial", self.base_text_size), anchor="w")
        style.configure("Setup.TButton", font=("Arial", self.base_button_size), padding=10)

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

        # Basic Parameters tab
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Parameters")
        self.setup_basic_parameters(basic_frame)

        # Advanced Parameters tab
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

        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def setup_basic_parameters(self, parent):
        # Frame for selecting parameter mode
        mode_frame = ttk.LabelFrame(parent, text="Parameter Mode", padding="10")
        mode_frame.pack(fill="x", padx=5, pady=5)

        ttk.Radiobutton(mode_frame, text="Simulation Properties", variable=self.parameter_mode_var, value="simulation",
                        command=self.update_parameter_visibility).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="String Properties", variable=self.parameter_mode_var, value="string_params",
                        command=self.update_parameter_visibility).pack(side=tk.LEFT, padx=5)

        # Frame that shows either simulations parameters or string parameters based on selection
        self.props_frame = ttk.LabelFrame(parent, text="String Properties", padding="10")
        self.props_frame.pack(fill="x", padx=5, pady=5)

        # Simulation parameters widgets
        self.simulation_widgets = []

        # Number of Masses entry
        rowi=0
        lbl = ttk.Label(self.props_frame, text="Number of Masses:")
        ent = ttk.Entry(self.props_frame, textvariable=self.num_segments_var, width=10)
        lbl.grid(row=rowi, column=0, padx=5, pady=5)
        ent.grid(row=rowi, column=1, padx=5, pady=5)
        self.simulation_widgets.extend([lbl, ent])
        def update_equilibrium_length(*args):
            try:
                num_segments = self.num_segments_var.get()
                if num_segments > 1:
                    total_length = np.linalg.norm(
                        self.default_params.end_point - self.default_params.start_point
                    )
                    natural_length = total_length / (num_segments-1)
                    self.equilibrium_length_var.set(str(natural_length))
            except tk.TclError:
                pass
        self.num_segments_var.trace_add("write", update_equilibrium_length)

        # Spring constant entry
        rowi+=1
        lbl = ttk.Label(self.props_frame, text="Spring constant (N/m):")
        ent = ttk.Entry(self.props_frame, textvariable=self.spring_constant_var, width=10)
        lbl.grid(row=rowi, column=0, padx=5, pady=5)
        ent.grid(row=rowi, column=1, padx=5, pady=5)
        self.simulation_widgets.extend([lbl, ent])

        # Mass per point entry
        rowi+=1
        lbl = ttk.Label(self.props_frame, text="Mass per point (kg):")
        ent = ttk.Entry(self.props_frame, textvariable=self.mass_var, width=10)
        lbl.grid(row=rowi, column=0, padx=5, pady=5)
        ent.grid(row=rowi, column=1, padx=5, pady=5)
        self.simulation_widgets.extend([lbl, ent])

        # Equilibrium length entry
        rowi+=1
        lbl = ttk.Label(self.props_frame, text="Equilibrium length (m):")
        ent = ttk.Entry(self.props_frame, textvariable=self.equilibrium_length_var, width=10)
        lbl.grid(row=rowi, column=0, padx=5, pady=5)
        ent.grid(row=rowi, column=1, padx=5, pady=5)
        self.simulation_widgets.extend([lbl, ent])
        help_label = ttk.Label(
            self.props_frame,
            text="(Initial natural length of each segment)\nForce: T = k(r - r0)",
            font=("Arial", 8),
            foreground="gray"
        )
        help_label.grid(row=rowi, column=2, padx=5, pady=5, sticky="w")
        self.simulation_widgets.append(help_label)

        # String parameters widgets
        self.string_widgets = []

        # Unstretched length entry
        rowj = 0
        lbl = ttk.Label(self.props_frame, text="L0 (unstretched length):")
        ent = ttk.Entry(self.props_frame, textvariable=self.L0_var, width=10)
        lbl.grid(row=rowj, column=0, padx=5, pady=5)
        ent.grid(row=rowj, column=1, padx=5, pady=5)
        self.string_widgets.extend([lbl, ent])

        # Stiffness entry
        lbl = ttk.Label(self.props_frame, text="α (stiffness = EA):")
        ent = ttk.Entry(self.props_frame, textvariable=self.alpha_var, width=10)
        lbl.grid(row=rowj, column=3, padx=5, pady=5)
        ent.grid(row=rowj, column=4, padx=5, pady=5)
        self.string_widgets.extend([lbl, ent])

        # Linear mass density entry
        rowj+=1
        lbl = ttk.Label(self.props_frame, text="µ0 (linear mass density):")
        ent = ttk.Entry(self.props_frame, textvariable=self.mu0_var, width=10)
        lbl.grid(row=rowj, column=0, padx=5, pady=5)
        ent.grid(row=rowj, column=1, padx=5, pady=5)
        self.string_widgets.extend([lbl, ent])

        # Number of masses entry (my logics screwy with this because of a mistake I made early on in designing the program)
        lbl = ttk.Label(self.props_frame, text="N (Number of masses):")
        ent = ttk.Entry(self.props_frame, textvariable=self.N_var, width=10)
        lbl.grid(row=rowj, column=3, padx=5, pady=5)
        ent.grid(row=rowj, column=4, padx=5, pady=5)
        self.string_widgets.extend([lbl, ent])


        # Lets user choose L or T, and seperates them to let them sit over their respective variables to some extent
        rowj+=1
        choose_L_frame = ttk.Frame(self.props_frame)
        choose_L_frame.grid(row=rowj, column=0, columnspan=2, pady=10)
        ttk.Radiobutton(choose_L_frame, text="Set L", variable=self.use_L_not_T_var, value=True).pack(side=tk.LEFT, padx=5)
        self.string_widgets.append(choose_L_frame)

        choose_T_frame = ttk.Frame(self.props_frame)
        choose_T_frame.grid(row=rowj, column=2, columnspan=2, pady=10)
        ttk.Radiobutton(choose_T_frame, text="Set T", variable=self.use_L_not_T_var, value=False).pack(side=tk.LEFT, padx=5)
        self.string_widgets.append(choose_T_frame)

        rowj+=1
        lbl = ttk.Label(self.props_frame, text="L (if chosen):")
        ent = ttk.Entry(self.props_frame, textvariable=self.L_var, width=10)
        lbl.grid(row=rowj, column=0, padx=5, pady=5)
        ent.grid(row=rowj, column=1, padx=5, pady=5)
        self.string_widgets.extend([lbl, ent])

        lbl = ttk.Label(self.props_frame, text="T (if chosen):")
        ent = ttk.Entry(self.props_frame, textvariable=self.T_var, width=10)
        lbl.grid(row=rowj, column=3, padx=5, pady=5)
        ent.grid(row=rowj, column=4, padx=5, pady=5)
        self.string_widgets.extend([lbl, ent])

        # Initially update visibility
        self.update_parameter_visibility()

        # Display Settings
        display_frame = ttk.LabelFrame(parent, text="Display Settings", padding="10")
        display_frame.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(
            display_frame,
            text="Dark Mode",
            variable=self.dark_mode_var
        ).pack(padx=5, pady=5)

    def update_parameter_visibility(self):
        mode = self.parameter_mode_var.get()
        if mode == 'simulation':
            # Show simulation widgets
            for w in self.simulation_widgets:
                w.grid()
            # Hide string widgets
            for w in self.string_widgets:
                w.grid_remove()
        else:
            # Show string widgets
            for w in self.string_widgets:
                w.grid()
            # Hide simulation widgets
            for w in self.simulation_widgets:
                w.grid_remove()

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
            'euler': "Simple first-order method (fast but less accurate)",
            'euler_cromer': "Modified Euler method with better energy conservation",
            'rk2': "Second-order Runge-Kutta method",
            'leapfrog': "Symplectic method with good energy conservation",
            'rk4': "Fourth-order Runge-Kutta (most accurate but slowest)"
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
            text="Tip: Smaller dt = more accurate but slower",
            font=("Arial", 8),
            foreground="gray"
        ).pack(pady=2)

    def validate_parameters(self):
        try:
            mode = self.parameter_mode_var.get()
            if mode == 'simulation':
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

                equilibrium_length = self.equilibrium_length_var.get()
                if equilibrium_length <= 0:
                    raise ValueError("Equilibrium length must be positive")

                force_magnitude = self.force_magnitude_var.get()
                if force_magnitude < 0:
                    raise ValueError("Force magnitude cannot be negative")

            else:
                # string_params
                L0 = self.L0_var.get()
                mu0 = self.mu0_var.get()
                alpha = self.alpha_var.get()
                N = self.N_var.get()
                dt = self.dt_var.get()

                if N < 3:
                    raise ValueError("N must be >= 3")
                if L0 <= 0:
                    raise ValueError("L0 must be positive")
                if mu0 <= 0:
                    raise ValueError("µ0 must be positive")
                if alpha <= 0:
                    raise ValueError("α must be positive")
                if dt <= 0:
                    raise ValueError("dt must be positive")

                force_magnitude = self.force_magnitude_var.get()
                if force_magnitude < 0:
                    raise ValueError("Force magnitude cannot be negative")

                if self.use_L_not_T_var.get():
                    L = self.L_var.get()
                    if L <= 0:
                        raise ValueError("L must be positive")
                else:
                    T = self.T_var.get()
                    if T < 0:
                        raise ValueError("T cannot be negative")

            return True

        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return False

    def get_parameters(self):
        if not self.validate_parameters():
            return None

        mode = self.parameter_mode_var.get()
        if mode == 'simulation':
            params = SimulationParameters(
                num_segments=(self.num_segments_var.get() - 1),
                spring_constant=self.spring_constant_var.get(),
                mass=self.mass_var.get(),
                dt=self.dt_var.get(),
                integration_method=self.integration_var.get(),
                applied_force=np.array([0.0, 0.0, self.force_magnitude_var.get()]),
                dark_mode=self.dark_mode_var.get(),
                custom_equilibrium_length=self.equilibrium_length_var.get(),
                parameter_mode='simulation'
            )
            return params
        else:
            # In string_params mode, N_var is the number_of_masses directly (not subtracting 1 here)
            N_val = self.N_var.get()

            params = SimulationParameters(
                parameter_mode='string_params',
                integration_method=self.integration_var.get(),
                applied_force=np.array([0.0, 0.0, self.force_magnitude_var.get()]),
                dark_mode=self.dark_mode_var.get(),
                dt=self.dt_var.get(),
                L0=self.L0_var.get(),
                mu0=self.mu0_var.get(),
                alpha=self.alpha_var.get(),
                N=N_val,  # N is number_of_masses directly
                use_L_not_T=self.use_L_not_T_var.get(),
                L=self.L_var.get(),
                T=self.T_var.get(),
                num_segments=N_val - 1  # Since N is masses, segments = N-1
            )
            return params

    def start_simulation(self):
        if not self.validate_parameters():
            return
        self.simulation_params = self.get_parameters()
        self.root.quit()
        self.root.destroy()


class SimulationVisualizer:
    """
    This class is responsible for the live simulation visualization.
    It sets up a 3D scene with the masses, allows me to pause/resume, apply forces, and adjust camera.
    Uses matplotlib animations and the PlottingToolkit for UI.
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

        # Setups a ForceHandler for external forces.
        self.force_handler = ForceHandler(physics_model, objects, dark_mode)

        # Initializes PlottingToolkit for controlling and drawing the scene.
        self.plotter = PlottingToolkit()

        # Some default camera settings so I have a good starting viewpoint.
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
        """
        Actually builds the figure, axes, and initials plots. Adds text elements for force/camera info as well.
        """
        plt.close('all')  # Close any existing figures
        plt.style.use('dark_background' if self.dark_mode else 'default')

        # Gets screen dimensions without creating a figure
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # Creates figure in full screen
        self.fig = plt.figure(figsize=(screen_width / 100, screen_height / 100))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Adjusts the main plot area for full screen while leaving space for controls
        self.ax.set_position([0.20, 0.15, 0.6, 0.8])

        # Makes window full screen
        manager = plt.get_current_fig_manager()
        if plt.get_backend() == 'TkAgg':
            manager.window.state('zoomed')  # For Windows
        elif plt.get_backend() == 'Qt5Agg':
            manager.window.showMaximized()  # For Qt
        else:
            # Tries to make it full screen for other backends
            try:
                manager.full_screen_toggle()
            except:
                manager.resize(*manager.window.maxsize())

        self.setup_plots()
        self.setup_camera()
        self.setup_enhanced_controls()
        self._connect_events()

        text_color = 'white' if self.dark_mode else 'black'
        # Text elements for displaying some simulation info.
        self.force_info_text = self.ax.text2D(
            1.15, 0.55,
            '',
            transform=self.ax.transAxes,
            color=text_color,
            fontsize=10,
            verticalalignment='top'
        )
        self.plotter.ui_elements['text']['force_info'] = self.force_info_text

    def setup_plots(self):
        """
        Creates a scatter point for each mass and a line connecting consecutive masses to represent the string segments.
        """
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
        """
        Set initial camera distance and axis limits based on object positions,
        so the entire string is nicely visible.
        """
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

    def setup_enhanced_controls(self):
        """
        Adds buttons, sliders, and radio buttons directly onto the plot for simulation control.
        For example: a play/pause button, a speed slider, force selection controls, etc.
        """
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
            ('play_button', 'Start', 0.24),
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

        min_object = 2
        max_object = len(self.objects)-1
        initial_object = (len(self.objects)+1) // 2
        #print(round(12.5))
        #print(float(len(self.objects)),"+",initial_object)
        self.object_slider = Slider(
            plt.axes([left_panel_start, 0.40, panel_width, 0.02]),
            'Object',
            min_object, max_object,
            valinit=initial_object,
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

        self.gaussian_width_slider_ax = plt.axes([left_panel_start+.02, 0.25, panel_width-.02, 0.02])
        self.gaussian_width_slider = Slider(
            self.gaussian_width_slider_ax,
            'Gaussian Width',
            0.1, len(self.objects) / 2.0,
            valinit=self.force_handler.gaussian_width,
            valfmt='%.2f'
        )
        self.gaussian_width_slider.on_changed(self.set_gaussian_width)

        self.frequency_slider_ax = plt.axes([left_panel_start, 0.25, panel_width, 0.02])
        self.frequency_slider = Slider(
            self.frequency_slider_ax,
            'Frequency',
            0.1, 99.9,
            valinit=self.force_handler.sinusoidal_frequency,
            valfmt='%.2f'
        )
        self.frequency_slider.on_changed(self.set_sinusoidal_frequency)

        # Hide Gaussian and frequency sliders by default since "Single Mass" is initial.
        self.gaussian_width_slider_ax.set_visible(False)
        self.frequency_slider_ax.set_visible(False)

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
        """
        If I want to go back to the setup window, close the figure and show main_root window.
        Ensures proper cleanup of matplotlib windows and restoration of main window state.
        """
        self.should_restart = True

        # Stop any ongoing animation
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()

        # Restores window state before closing
        manager = plt.get_current_fig_manager()
        if plt.get_backend() == 'TkAgg':
            manager.window.state('normal')  # Un-maximize window
        elif plt.get_backend() == 'Qt5Agg':
            manager.window.showNormal()  # Un-maximize window

        # Close all matplotlib figures
        plt.close('all')

        # Shows the main window if it exists
        if self.main_root:
            self.main_root.deiconify()
            self.main_root.lift()  # Brings to front
            self.main_root.focus_force()  # Forces focus

            # Resets main window state if needed
            self.main_root.state('normal')  # Ensures it's not minimized

            # Centers the main window
            window_width = int(self.main_root.winfo_screenwidth() * 0.27)
            window_height = int(self.main_root.winfo_screenheight() * 0.6)
            x = (self.main_root.winfo_screenwidth() // 2) - (window_width // 2)
            y = (self.main_root.winfo_screenheight() // 2) - (window_height // 4)
            self.main_root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def _connect_events(self):
        """
        Connects mouse and scroll events so I can rotate and zoom the 3D plot with the mouse.
        """
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
        """
        When I move the mouse while holding a button, I can rotate or pan the camera.
        """
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
        """
        Scrolling the mouse wheel adjusts zoom level. Zooming in/out helps focus on details or see the whole scene.
        """
        if event.inaxes == self.ax:
            factor = 0.9 if event.button == 'up' else 1.1
            self.camera['distance'] *= factor
            self.zoom_button.label.set_text('Zoom: Custom')
            self.update_camera()

    def update_frame(self, frame):
        """
        This method is called each animation frame.
        If not paused, I advance the simulation by a number of steps, record data, apply forces, and then update the display.
        Also calculates FPS and checks if forces should stop.
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

            # Runs multiple simulation steps per frame to control simulation speed.
            for _ in range(self.iteration_count):
                positions = np.array([obj.position for obj in self.objects])
                velocities = np.array([obj.velocity for obj in self.objects])
                accelerations = np.array([obj.acceleration for obj in self.objects])

                self.physics.data_recorder.record_step(
                    self.physics.time,
                    positions,
                    velocities,
                    accelerations
                )

                self.physics.step(self.integration_method)

            # Checks if force duration ended.
            if self.force_handler.check_duration(self.iteration_count):
                self.force_button.label.set_text('Apply Force')
                self.force_button.color = 'darkgray' if not self.dark_mode else 'gray'
                self.fig.canvas.draw_idle()

            # If continuous force is active, apply it every frame.
            if self.force_handler.continuous and self.force_handler.active:
                self.force_handler.apply()

        self.update_plots()
        self.update_info()
        self.highlight_selected_object()

        return []

    def update_plots(self):
        """
        Refreshes positions of scatter points and lines to shows the string shape as it evolves.
        """
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
        Updates text info about current state: paused/running, FPS, time, integration method, dt, steps/frame, force duration.
        """
        current_time = self.physics.time if self.simulation_started else 0.0
        dt_per_frame = self.physics.dt * self.iteration_count
        state = 'PAUSED' if self.paused else 'RUNNING'

        if self.force_handler.continuous:
            duration_text = "Continuous"
        elif self.force_handler.active:
            duration_text = f"{self.force_handler.duration_remaining:.2f}s remaining"
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
        """
        Shows detailed force information on the selected mass: external forces, spring forces, total force.
        Also includes spring constant, natural unstretched length, and custom unstretched length.
        """
        external_force, spring_forces, total_force = self.physics.get_forces_on_object(self.force_handler.selected_object)

        # Get spring constant and equilibrium length from physics engine
        spring_constant = self.physics.k
        total_length = np.linalg.norm(self.physics.end_point - self.physics.start_point)
        natural_length = total_length / (len(self.objects) - 1)  # Natural unstretched length
        custom_length = self.physics.equilibrium_length  # Custom equilibrium length

        force_text = (
            f"Forces on Mass {self.force_handler.selected_object + 1}:\n"
            f"K: {spring_constant:.1f} N/m\n"
            f"r: {natural_length:.6f} m\n"
            f"r0: {custom_length:.6f} m\n"
            f"Tension on String: {self.physics.tension:.1f}N\n"
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
        """
        Show the current camera angles and distance in a text box.
        Helps me understand what view I’m currently on.
        """
        text = (
            f"Camera Azim: {self.camera['azimuth']:.1f}\n"
            f"Camera Elev: {self.camera['elevation']:.1f}\n"
            f"Camera Dist: {self.camera['distance']:.1f}"
        )
        self.plotter.update_text('camera_info', text)

    def highlight_selected_object(self):
        """
        Visually highlights the currently selected object where force is being applied,
        so user always know which mass is targeted.
        """
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
        """
        Starts the matplotlib animation loop. Once called, it keeps updating frames until closed.
        """
        self.anim = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=20,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
        return self.should_restart

    # The following methods handle UI callbacks for various buttons and sliders.
    # They let me pause the sim, reset it, save data, change force parameters, switch views, etc.

    def set_simulation_speed(self, val):
        self.iteration_count = int(float(val))
        self.update_info()

    def toggle_pause(self, event):
        """
        Pauses or resumes simulation. If it hasn't started, start it now.
        Also update the button text accordingly.
        """
        if not self.simulation_started and not self.paused:
            return

        self.paused = not self.paused

        if not self.paused and not self.simulation_started:
            self.simulation_started = True
            self.physics.start_simulation()
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
        Resets everything back to the initial state. Clears data, restores original positions/velocities,
        turns off forces, and sets paused.
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
        Save the currently recorded simulation data to a CSV file.
        If simulation is running, pause it first.
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
        Turn on or off the external force application.
        If simulation not started, start now. Update button text and colors accordingly.
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
        """
        If user toggles continuous mode, the force stays active until turned off, instead of timing out.
        """
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
        Change force type based on radio button selection. Also show/hide Gaussian or frequency slider as needed.
        """
        self.force_handler.selected_type = label

        self.gaussian_width_slider_ax.set_visible(False)
        self.frequency_slider_ax.set_visible(False)

        if label == 'Gaussian':
            self.gaussian_width_slider_ax.set_visible(True)
        elif label == 'Sinusoidal':
            self.frequency_slider_ax.set_visible(True)

        self.fig.canvas.draw_idle()

    def set_force_direction(self, label):
        self.force_handler.selected_direction = label

    def set_selected_object(self, val):
        self.force_handler.selected_object = int(val-1)
        self.highlight_selected_object()
        self.fig.canvas.draw_idle()

    def set_force_amplitude(self, val):
        self.force_handler.amplitude = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_force_duration(self, val):
        self.force_handler.duration = float(val)
        if not self.force_handler.active:
            self.force_handler.duration_remaining = float(val)
        self.update_info()
        self.fig.canvas.draw_idle()

    def set_force_duration_remaining(self, val):
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
        """
        Cycle through predefined camera views like Top, Front, Side, etc.
        Makes it quick to get a known viewpoint of the simulation.
        """
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
        """
        Cycle through zoom levels (Fit, Close, Medium, Far).
        This makes it easier to quickly frame the scene nicely.
        """
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
        """
        Calculate a camera distance that fits all objects into view nicely.
        """
        if not self.objects:
            return 2.0
        positions = np.array([obj.position for obj in self.objects])
        max_dist = np.max(np.abs(positions))
        return max_dist * 2.0

    def update_camera(self):
        """
        After changing camera params (e.g., by rotating or zooming), update the axes limits and redraw.
        """
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
        """
        Switch between dark and light themes.
        This updates background, text, and UI elements for better visibility.
        """
        self.dark_mode = not self.dark_mode

        text_color = 'white' if self.dark_mode else 'black'
        background_color = 'black' if self.dark_mode else 'white'
        button_color = 'gray' if self.dark_mode else 'lightgray'

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

        self.info_text.set_color(text_color)
        self.force_info_text.set_color(text_color)

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
            self.duration_slider,
            self.gaussian_width_slider,
            self.frequency_slider
        ]

        for slider in sliders:
            slider.label.set_color(text_color)
            slider.valtext.set_color(text_color)

        self.update_plots()
        self.highlight_selected_object()

        self.theme_button.label.set_text('Dark Mode' if not self.dark_mode else 'Light Mode')

        if self.fig:
            self.fig.canvas.draw_idle()


class AnalysisVisualizer:
    """
    A GUI to analyze previously recorded simulation data. I can load multiple CSV files
    and then compare their results. I can do displacement comparisons, frequency analysis,
    and more.
    """
    def __init__(self, main_root):
        self.main_root = main_root
        self.analyzer = DataAnalysis()  # adds a data analysis instance to handle loaded simulations
        self.loaded_files = {}  # adds a dictionary that maps file paths to loaded simulation references

        # adds a top-level window for the analysis interface
        self.root = tk.Toplevel(main_root)
        self.root.title("String Simulation Analysis")
        self.root.minsize(750, 330)

        # calculates window size and position based on screen dimensions
        window_width = int(self.root.winfo_screenwidth() * 0.55)
        window_height = int(self.root.winfo_screenheight() * 0.40)
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # defines a set of colors I'll use for plotting multiple simulations distinctly
        self.colors = ["red", "green", "blue", "yellow", "orange", "purple"]

        self.setup_gui()  # sets up the main GUI layout and elements

        # makes sure that when this window closes, I handle it properly (e.g., show main window again)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        """
        This method sets up the entire GUI layout for the analysis window.
        I create a main frame, add a file management section and an analysis options section.
        I configure grid weights so the interface resizes nicely.
        I add a treeview to display loaded files, without scroll bars, and I add buttons
        for loading and deleting files.
        I also add buttons in the analysis frame for different analyses.
        """
        # Configure grid weights for the root window
        self.root.grid_rowconfigure(0, weight=1)  # Allows vertical expansion
        self.root.grid_columnconfigure(0, weight=1)  # Allows horizontal expansion

        # Creates main frame with padding and configures its grid
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # allows the main frame to expand horizontally and vertically
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)  # Make file frame take most space
        self.main_frame.grid_rowconfigure(2, weight=0)  # Analysis frame doesn't need to expand

        # adds a labeled frame for file management at the top
        file_frame = ttk.LabelFrame(self.main_frame, text="File Management", padding="5")
        file_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        file_frame.grid_columnconfigure(0, weight=1)
        file_frame.grid_rowconfigure(1, weight=1)

        # adds a frame for file management buttons (load, delete, clear)
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 5))
        button_frame.grid_columnconfigure(1, weight=1)

        # adds a button to load files
        ttk.Button(button_frame, text="Load Files", command=self.load_files).grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )

        # adds a small frame on the right side for delete/clear buttons
        delete_frame = ttk.Frame(button_frame)
        delete_frame.grid(row=0, column=2, sticky="e")

        # adds a "Delete Selected" button
        ttk.Button(delete_frame, text="Delete Selected", command=self.delete_selected).pack(
            side="left", padx=5
        )
        # adds a "Clear All" button
        ttk.Button(delete_frame, text="Clear All", command=self.clear_files).pack(
            side="left", padx=(0, 5)
        )

        # adds a frame to hold the treeview that lists loaded files
        tree_frame = ttk.Frame(file_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        # adds a treeview to display loaded simulations (no scroll bars)
        self.file_tree = ttk.Treeview(tree_frame, show="headings", selectmode="extended")
        self.file_tree.grid(row=0, column=0, sticky="nsew")

        # Configures treeview columns
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
            ("Movement Patterns", self.compare_movement),
            ("Node Displacement", self.compare_displacement),
            ("Average Displacements", self.plot_nodal_average_displacement)

        ]

        # adds each button in the first row of analysis options
        for i, (text, command) in enumerate(row1_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=0, column=i, padx=5, pady=5, sticky="ew"
            )

        # Second row of analysis buttons (centered)
        row2_buttons = [
            ("Measure Period", self.measure_period_dialog),
            ("Harmonic Analysis", self.analyze_harmonics),
            ("Frequency Analysis", self.analyze_frequencies),
            ("Natural Frequencies", self.analyze_natural_frequencies),
            ("Normalized Displacements", self.plot_nodal_normalized_displacement)
        ]

        # adds the second row of buttons
        for i, (text, command) in enumerate(row2_buttons):
            ttk.Button(analysis_frame, text=text, command=command).grid(
                row=1, column=i, padx=5, pady=5, sticky="ew"
            )

    def load_files(self):
        """
        This method shows a file dialog so I can select simulation CSV files.
        For each selected file, it attempts to load it into the analyzer.
        If successful, it inserts an entry into the file_tree with summary info:
        filename, number of objects, number of frames, simulation time, and a color.
        It also stores a reference in loaded_files and selects the newly added item in the tree.
        If an error occurs, it shows a message box.
        """
        files = filedialog.askopenfilenames(
            title="Select Simulation Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        for file_path in files:
            try:
                data = self.analyzer.load_simulation(file_path)  # loads the simulation into analyzer
                summary = self.analyzer.get_simulation_summary(data)  # gets a summary of the loaded sim
                item = self.file_tree.insert("", "end", values=(
                    os.path.basename(file_path),
                    summary['num_objects'],
                    summary['num_frames'],
                    f"{summary['simulation_time']:.3f} s",
                    self.colors[len(self.loaded_files) % len(self.colors)]
                ))
                # adds this simulation reference to loaded_files
                self.loaded_files[file_path] = data
                # selects the newly inserted item in the file tree
                self.file_tree.selection_add(item)
            except Exception as e:
                # shows an error message if something goes wrong
                messagebox.showerror("Error", f"Failed to load {file_path}:\n{str(e)}")

    def delete_selected(self):
        """
        This method deletes the selected simulations from both the treeview and from memory.
        I look up the selected items, find their associated file paths in loaded_files, remove them,
        and delete the treeview entries. If no items selected, shows a warning.
        """
        selected_items = self.file_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select files to delete.")
            return

        files_to_remove = []
        # goes through each selected item in the tree
        for item in selected_items:
            filename = self.file_tree.item(item)["values"][0]
            # finds the corresponding file_path in loaded_files
            for path in self.loaded_files:
                if os.path.basename(path) == filename:
                    files_to_remove.append(path)
                    break
            self.file_tree.delete(item)

        # removes each identified file from loaded_files
        for file_path in files_to_remove:
            if file_path in self.loaded_files:
                del self.loaded_files[file_path]

    def clear_files(self):
        """
        This method removes all files from loaded_files and clears the tree.
        It then reinitializes the analyzer to a fresh state.
        """
        self.loaded_files.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self.analyzer = DataAnalysis()

    def get_selected_files(self):
        """
        This method retrieves the file paths of the currently selected files in the treeview.
        If none are selected, it returns all loaded files.
        It finds them by matching the displayed filename in the tree to the loaded_files dictionary keys.
        """
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

    def show_summary(self):
        """
        This method shows a new window with a summary of each selected simulation.
        It prints the number of frames, total simulation time, number of objects, and stationary nodes info.
        If no files loaded/selected, it warns the user.
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return
        window = tk.Toplevel(self.root)
        window.title("Simulation Summary")
        text = tk.Text(window, wrap=tk.WORD, width=60, height=30)
        text.pack(padx=10, pady=10)

        # adds info for each selected file
        for file_path in files:
            summary = self.analyzer.get_simulation_summary(file_path)
            filename = os.path.basename(file_path)
            text.insert("end", f"\nFile: {filename}\n{'=' * 50}\n")
            text.insert("end", f"Number of frames: {summary['num_frames']}\n")
            text.insert("end", f"Simulation time: {summary['simulation_time']:.3f} s\n")
            text.insert("end", f"Number of objects: {summary['num_objects']}\n")
            text.insert("end", f"Stationary nodes: {summary['num_stationary_nodes']}\n\n")
            for node_id, pos in summary['stationary_node_positions'].items():
                text.insert("end", f"Node {node_id} position: {pos}\n")
        text.config(state=tk.DISABLED)

    def analyze_natural_frequencies(self):
        """
        Compares observed frequencies from the simulation with theoretical natural frequencies
        based on string parameters. For each simulation file, it:

        1. Calculates theoretical natural frequencies using fn = (n/2L) * sqrt(T/μ)
        2. Extracts actual frequency peaks from simulation data
        3. Compares theoretical vs observed frequencies
        4. Visualizes the results with both line and bar plots
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        self.root.withdraw()
        try:
            plotter = PlottingToolkit()
            first_file = True

            for file_path in files:
                data = self.loaded_files[file_path]
                filename = os.path.basename(file_path)

                # Get simulation parameters
                summary = self.analyzer.get_simulation_summary(file_path)
                num_nodes = summary['num_objects']

                # Calculate string properties
                df = self.analyzer.simulations[file_path]
                time_step = df['Time'].diff().mean()

                # Get positions for first and last nodes to calculate length
                x0, y0, z0 = self.analyzer.get_object_trajectory(file_path, 0)
                xn, yn, zn = self.analyzer.get_object_trajectory(file_path, num_nodes - 1)
                start_pos = np.array([x0[0], y0[0], z0[0]])
                end_pos = np.array([xn[0], yn[0], zn[0]])
                string_length = np.linalg.norm(end_pos - start_pos)

                # Calculate total mass and mass per unit length
                mass_per_node = None
                tension = None

                # Try to extract mass and tension from column names
                for col in df.columns:
                    if 'mass' in col.lower():
                        mass_per_node = float(col.split('=')[1]) if '=' in col else None
                    if 'tension' in col.lower():
                        tension = float(col.split('=')[1]) if '=' in col else None

                if mass_per_node is None:
                    mass_per_node = 0.01  # Default from simulation parameters
                if tension is None:
                    tension = 1000.0  # Default spring constant

                total_mass = mass_per_node * num_nodes
                mass_per_length = total_mass / string_length

                # Calculate theoretical natural frequencies for first 10 modes
                modes = np.arange(1, 11)
                theoretical_freqs = (modes / (2 * string_length)) * np.sqrt(tension / mass_per_length)

                # Get actual frequency spectrum from simulation data
                node_displacements = []
                for node_id in range(num_nodes):
                    x, y, z = self.analyzer.get_object_trajectory(file_path, node_id)
                    positions = np.column_stack([x, y, z])
                    initial_pos = positions[0]
                    displacements = np.linalg.norm(positions - initial_pos, axis=1)
                    node_displacements.append(displacements)

                node_displacements = np.array(node_displacements)
                n_samples = node_displacements.shape[1]

                # Compute FFT
                freqs = np.fft.fftfreq(n_samples, time_step)[:n_samples // 2]
                freqs = freqs[1:-1]  # Remove DC and Nyquist

                fft_matrix = []
                for disp in node_displacements:
                    fft_result = np.fft.fft(disp)
                    half_fft = np.abs(fft_result[:n_samples // 2])
                    half_fft = half_fft[1:-1]
                    fft_matrix.append(half_fft)

                fft_matrix = np.array(fft_matrix)
                average_spectrum = np.mean(fft_matrix, axis=0)

                # Custom peak finding implementation
                def find_peaks(spectrum, min_height_ratio=0.1):
                    """Find peaks in spectrum that are above min_height_ratio * max(spectrum)."""
                    peaks = []
                    peak_heights = []
                    threshold = min_height_ratio * np.max(spectrum)

                    # A point is a peak if it's larger than its neighbors and above threshold
                    for i in range(1, len(spectrum) - 1):
                        if (spectrum[i] > spectrum[i - 1] and
                                spectrum[i] > spectrum[i + 1] and
                                spectrum[i] > threshold):
                            peaks.append(i)
                            peak_heights.append(spectrum[i])

                    return np.array(peaks), np.array(peak_heights)

                # Find peaks in the average spectrum
                peak_indices, peak_heights = find_peaks(average_spectrum)
                observed_freqs = freqs[peak_indices]
                observed_mags = peak_heights

                # Sort peaks by magnitude
                sort_idx = np.argsort(observed_mags)[::-1]
                observed_freqs = observed_freqs[sort_idx]
                observed_mags = observed_mags[sort_idx]

                # Keep only top 10 peaks
                observed_freqs = observed_freqs[:10]
                observed_mags = observed_mags[:10]

                # Calculate frequency differences
                freq_differences = []
                matched_observed = []
                matched_theoretical = []

                for theo_freq in theoretical_freqs:
                    if len(observed_freqs) > 0:
                        # Find closest observed frequency
                        idx = np.argmin(np.abs(observed_freqs - theo_freq))
                        obs_freq = observed_freqs[idx]

                        diff_percent = 100 * (obs_freq - theo_freq) / theo_freq
                        freq_differences.append(diff_percent)
                        matched_observed.append(obs_freq)
                        matched_theoretical.append(theo_freq)

                # Create visualization
                plot_config = PlotConfig(
                    title=f"Natural Frequency Analysis - {filename}",
                    xlabel="Mode Number",
                    ylabel="Frequency (Hz)",
                    grid=True,
                    figure_size=(15, 10)
                )

                # Get color from tree
                color = None
                for item in self.file_tree.get_children():
                    values = self.file_tree.item(item)["values"]
                    if values[0] == filename:
                        color = values[4]
                        break
                if color is None:
                    color = 'steelblue'

                # Plot theoretical vs observed frequencies
                plotter.plot(
                    modes,
                    theoretical_freqs,
                    plot_type='line',
                    color=color,
                    label=f'{filename} (Theoretical)',
                    marker_style='o',
                    new_figure=first_file,
                    **vars(plot_config)
                )

                plotter.plot(
                    modes[:len(matched_observed)],
                    matched_observed,
                    plot_type='scatter',
                    color='red',
                    label=f'{filename} (Observed)',
                    marker_style='x',
                    new_figure=False,
                    **vars(plot_config)
                )

                # Add text annotations for differences
                summary_text = "Frequency Differences:\n"
                for i, (theo, obs, diff) in enumerate(zip(matched_theoretical, matched_observed, freq_differences)):
                    summary_text += f"Mode {i + 1}: {diff:+.1f}% ({obs:.1f} vs {theo:.1f} Hz)\n"

                # Place summary text as annotation
                plotter.ax.text(
                    0.95, 0.95,
                    summary_text,
                    transform=plotter.ax.transAxes,
                    fontsize=9,
                    va='top',
                    ha='right',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round')
                )

                first_file = False

            # Show legend and adjust layout
            plotter.ax.legend(loc='upper left', frameon=True)
            plotter.fig.tight_layout()
            plotter.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze natural frequencies: {str(e)}")
        finally:
            self.root.deiconify()

    def compare_movement(self):
        """
        Plots 3D trajectories of selected simulations to compare movement patterns.
        Uses a 3D isometric view and uses fig.suptitle to set the title at the figure level.
        Ensures that a figure and axes are created before trying to adjust spacing or set the title.
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # Creates a 3D plot. We explicitly create the figure before adjusting spacing or setting the suptitle.
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
            figure_size=(12, 8),
            spacing={
                'margins': {
                    'left': 0.1,
                    'right': 0.9,
                    'top': 0.90,
                    'bottom': 0.15,
                    'wspace': 0.2,
                    'hspace': 0.2
                },
                'elements': {
                    'title_spacing': -30,
                    'xlabel_spacing': 10,
                    'ylabel_spacing': 10,
                    'zlabel_spacing': 10,
                    'legend_spacing': 0.1,
                    'tick_spacing': 5
                }
            }
        )

        plotter.create_figure(plot_config)

        # Now that fig and ax exist, we can update spacing.
        plotter.update_spacing(plot_config)

        first_file = True
        files_to_plot = self.get_selected_files()

        # Plots node trajectories. Labels only the first line of each simulation.
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]
            file_path = next((p for p in files_to_plot if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                summary = self.analyzer.get_simulation_summary(file_path)
                num_nodes = summary['num_objects']
                for node in range(num_nodes):
                    x, y, z = self.analyzer.get_object_trajectory(file_path, node)
                    label = filename if node == 0 else None
                    # Now that figure and axes exist, we can plot.
                    plotter.plot(
                        x, y, z,
                        new_figure=False,  # We already created the figure above.
                        plot_type='line',
                        color=color,
                        label=label,
                        alpha=0.5,
                        line_width=1.5,
                        **vars(plot_config)
                    )
                first_file = False

        plotter.ax.legend(loc='right', bbox_to_anchor=(1.6, 0.5), frameon=True)
        plotter.show()

    def compare_stationary(self):
        """
        This method shows a window comparing stationary nodes across selected simulations.
        It prints which nodes are stationary and their positions and displacements.
        If no files selected, it warns the user.
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return
        window = tk.Toplevel(self.root)
        window.title("Stationary Nodes Comparison")
        text = tk.Text(window, wrap=tk.WORD, width=60, height=30)
        text.pack(padx=10, pady=10)

        # adds info for each selected file
        for file_path in files:
            filename = os.path.basename(file_path)
            nodes = self.analyzer.find_stationary_nodes(file_path)
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

    def compare_displacement(self):
        """
        This method prompts me to select a single node from a chosen simulation (if any are loaded),
        and then plots the displacement vs time for that node across all selected simulations.
        If no files selected, it shows a warning.
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return
        self.create_node_selection_dialog(files[0])

    def create_node_selection_dialog(self, file_path):
        """
        This method creates a small dialog window where I can select which node I want to analyze in detail.
        It lists all nodes (0 to num_nodes-1) for the chosen simulation.
        After I pick a node, it calls plot_displacement_comparison to actually show the plot.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Node to Analyze")
        dialog.transient(self.root)
        dialog.grab_set()

        data = self.loaded_files[file_path]
        summary = self.analyzer.get_simulation_summary(file_path)
        num_nodes = summary['num_objects']

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Select a node to analyze:").pack(pady=(0, 5))

        listbox = tk.Listbox(frame, selectmode=tk.SINGLE, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # adds all node IDs to the listbox
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

    def measure_period_dialog(self):
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        # We will pick the first file for simplicity, or let user pick a file if needed.
        file_path = files[0]
        summary = self.analyzer.get_simulation_summary(file_path)
        num_nodes = summary['num_objects']

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Node for Period Measurement")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Select a node to analyze:").pack(pady=(0, 5))

        listbox = tk.Listbox(frame, selectmode=tk.SINGLE, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        for i in range(num_nodes):
            listbox.insert(tk.END, f"Node {i}")

        use_velocity_var = tk.BooleanVar(dialog)
        use_velocity_var.set(False)
        ttk.Checkbutton(frame, text="Use Velocity for Zero-Crossings", variable=use_velocity_var).pack(pady=5)

        def on_ok():
            selections = listbox.curselection()
            if selections:
                node_id = int(listbox.get(selections[0]).split()[1])
                dialog.destroy()
                self.root.withdraw()
                try:
                    self.analyzer.plot_mode_period(file_path, node_id, use_velocity=use_velocity_var.get())
                finally:
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
        """
        Given a node_id, plots displacement over time for that node from all selected simulations.
        Each simulation line is labeled by filename, with chosen color. After plotting,
        shows a legend and adjusts layout.
        """
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

        first_file = True
        files_to_plot = self.get_selected_files()

        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]
            file_path = next((p for p in files_to_plot if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                try:
                    x, y, z = self.analyzer.get_object_trajectory(file_path, node_id)
                    positions = np.column_stack([x, y, z])
                    initial_pos = positions[0]
                    displacements = np.linalg.norm(positions - initial_pos, axis=1)
                    time_step = self.analyzer.simulations[file_path]['Time'].diff().mean()
                    t = np.arange(len(displacements)) * time_step

                    # Plot line for this simulation with filename label and chosen color
                    plotter.plot(
                        t,
                        displacements,
                        new_figure=first_file,
                        color=color,
                        label=filename,
                        line_style='-',
                        line_width=1.5,
                        **vars(plot_config)
                    )
                    first_file = False
                except Exception as e:
                    messagebox.showerror("Error", f"Error plotting {filename}: {str(e)}")

        # Show legend and tighten layout after all lines are plotted
        plotter.ax.legend(loc='best')
        plotter.fig.tight_layout()
        plotter.show()

    def plot_nodal_average_displacement(self):
        """
        Plots the average displacement of each node for selected simulations.
        Each simulation is plotted as a line with a unique color and label.
        After plotting all simulations, a legend is shown on the best location.
        """
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

        # We'll plot all selected simulations on the same figure
        # Each simulation line will have a label for easy identification.
        first_file = True
        files_to_plot = self.get_selected_files()

        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]
            file_path = next((p for p in files_to_plot if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                node_ids, avg_displacements = self.analyzer.get_average_displacements(file_path)

                # Plot each simulation with a label and chosen color
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
                    **vars(plot_config)
                )
                first_file = False

        # After plotting all lines, add a legend and adjust layout
        plotter.ax.legend(loc='best')
        plotter.fig.tight_layout()
        plotter.show()

    def plot_nodal_normalized_displacement(self):
        """
        Plots the normalized displacement of each node for selected simulations.
        Similar to average displacement plot, but node id values are normalized.
        Each simulation line is labeled for easy comparison.
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        plotter = PlottingToolkit()
        plot_config = PlotConfig(
            title="Normalized Node Displacement Comparison",
            xlabel="Normalized Number of Nodes",
            ylabel="Normalized Displacement",
            grid=True,
            figure_size=(12, 6)
        )

        first_file = True
        files_to_plot = self.get_selected_files()

        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)["values"]
            filename = values[0]
            color = values[4]
            file_path = next((p for p in files_to_plot if os.path.basename(p) == filename), None)
            if file_path:
                data = self.loaded_files[file_path]
                node_ids, normalized_displacements = self.analyzer.get_normalized_amplitudes(file_path)
                normalized_node_ids = node_ids / max(node_ids)
                # Plot each simulation line with filename label and chosen color
                plotter.plot(
                    normalized_node_ids,
                    normalized_displacements,
                    new_figure=first_file,
                    color=color,
                    label=filename,
                    plot_type='line',
                    marker_style='o',
                    line_style='-',
                    line_width=1.5,
                    **vars(plot_config)
                )
                first_file = False

        # Add legend and tighten layout after plotting all lines
        plotter.ax.legend(loc='best')
        plotter.fig.tight_layout()
        plotter.show()

    def analyze_frequencies(self):
        """
        This method triggers the frequency analysis of the data,
        but mostly exists just check if I have selected files.
        If I do, I call a function to plot frequency analysis results.
        Otherwise, I show a warning.
        """
        files = self.get_selected_files()
        if not files:
            messagebox.showwarning("Warning", "No simulation files loaded.")
            return

        self.root.withdraw()
        try:
            self.plot_multi_node_frequency_analysis(files[0])
        finally:
            self.root.deiconify()

    def plot_multi_node_frequency_analysis(self, file_path):
        """
        This method performs a frequency analysis of the displacement data for each node
        in the selected simulation. It computes the FFT (Fast Fourier Transform) of the displacement
        time-series for each node, extracts positive frequencies, normalizes them, and creates a heatmap
        that shows how each node responds at different frequencies.

        After generating the heatmap, it also adds a plot of the average spectrum over all nodes
        so I can see which frequencies dominate the system on average.

        I do this by:
        - Retrieving node displacement data over time
        - Computing the FFT for each node
        - Taking the magnitude and normalizing by the maximum value to get a consistent scale
        - Creating a frequency axis using fftfreq
        - Removing endpoints to focus on the main frequency range
        - Plotting a heatmap of node vs frequency
        - Computing an average spectrum across all nodes and plotting it separately

        If time_step or data is not valid, it shows an error. If everything is fine, it shows the results.
        """
        try:
            data = self.loaded_files[file_path]
            summary = self.analyzer.get_simulation_summary(file_path)
            num_nodes = summary['num_objects']

            df = self.analyzer.simulations[file_path]
            time_step = df['Time'].diff().mean()
            if time_step <= 0:
                messagebox.showerror("Error", "Invalid or zero time step in data.")
                return

            node_displacements = []
            for node_id in range(num_nodes):
                x, y, z = self.analyzer.get_object_trajectory(file_path, node_id)
                positions = np.column_stack([x, y, z])
                initial_pos = positions[0]
                displacements = np.linalg.norm(positions - initial_pos, axis=1)
                node_displacements.append(displacements)
            node_displacements = np.array(node_displacements)
            n_samples = node_displacements.shape[1]

            freqs = np.fft.fftfreq(n_samples, time_step)[:n_samples // 2]
            freqs = freqs[1:-1]
            fft_matrix = []
            for disp in node_displacements:
                fft_result = np.fft.fft(disp)
                half_fft = np.abs(fft_result[:n_samples // 2])
                half_fft = half_fft[1:-1]
                fft_matrix.append(half_fft)
            fft_matrix = np.array(fft_matrix)
            max_val = np.max(fft_matrix)
            if max_val > 0:
                fft_matrix = fft_matrix / max_val

            # Heatmap configuration
            plotter = PlottingToolkit()
            heatmap_config = PlotConfig(
                title="Frequency Content vs Node Position (Endpoints Excluded)",
                xlabel="Frequency (Hz)",
                ylabel="Node Number",
                grid=True,
                figure_size=(12, 10)
            )
            fig1, ax1 = plotter.create_figure(heatmap_config)
            im = ax1.pcolormesh(
                freqs,
                np.arange(num_nodes),
                fft_matrix,
                shading='auto',
                cmap='viridis'
            )
            fig1.colorbar(im, ax=ax1, label='Normalized Magnitude')

            # Average spectrum plot
            average_spectrum = np.mean(fft_matrix, axis=0)
            plotter2 = PlottingToolkit()
            spectrum_config = PlotConfig(
                title="Average Frequency Spectrum (Endpoints Excluded)",
                xlabel="Frequency (Hz)",
                ylabel="Normalized Magnitude",
                grid=True,
                figure_size=(12, 6)
            )

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

            # Identifies top peaks
            peak_indices = np.argsort(average_spectrum)[-10:][::-1]
            peak_freqs = freqs[peak_indices]
            peak_mags = average_spectrum[peak_indices]

            for freq, mag in zip(peak_freqs, peak_mags):
                plotter2.ax.plot(freq, mag, 'ro')
                plotter2.ax.text(
                    freq, mag + 0.05,
                    f'{freq:.1f} Hz',
                    ha='center'
                )

            # Summary text for dominant frequencies
            summary_text = "Dominant Frequencies:\n"
            for i, (pf, pm) in enumerate(zip(peak_freqs, peak_mags)):
                summary_text += f"{i + 1}: {pf:.1f} Hz ({pm:.2f})\n"

            # Places text box with better padding and adjust figure margins
            plotter2.ax.text(
                0.95, 0.85,  # Moved text box down to prevent title overlap
                summary_text,
                transform=plotter2.ax.transAxes,
                fontsize=9,
                va='top',
                ha='right',
                bbox=dict(
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.8,
                    boxstyle='round,pad=0.5'
                )
            )

            # Shows legends and adjust layout
            plotter2.ax.legend(loc='upper left', frameon=True)
            fig1.tight_layout()
            plotter2.fig.tight_layout()

            # Adjusts margins to prevent text cutoff
            fig1.subplots_adjust(left=0.12, right=0.88, top=0.85, bottom=0.12)  # Increased top margin
            plotter2.fig.subplots_adjust(left=0.12, right=0.85, top=0.85, bottom=0.12)  # Increased top margin

            plotter.show()
            plotter2.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process frequency analysis: {str(e)}")

    def analyze_harmonics(self):
        """
        This method would perform harmonic analysis if files are selected.
        If not, it shows a warning. If yes, it calls a plotting function.
        """
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
        """
        For each selected simulation, plots a bar chart of harmonic correlations.
        Shows all harmonics. Then determines the top 3 dominant harmonics and adds a small annotation.
        Ensures that legends and annotations are placed neatly without cutting off.
        """
        plotter = PlottingToolkit()
        num_files = len(files)
        first_file = True

        for idx, file_path in enumerate(files):
            data = self.loaded_files[file_path]
            filename = os.path.basename(file_path)

            harmonics, correlations = self.analyzer.get_harmonic_correlation(file_path)

            # Retrieves assigned color
            color = None
            for item in self.file_tree.get_children():
                values = self.file_tree.item(item)["values"]
                if values[0] == filename:
                    color = values[4]
                    break
            if color is None:
                color = 'steelblue'

            # Plots configuration for harmonic analysis
            plot_config = PlotConfig(
                title=f"Harmonic Analysis - {filename}",
                xlabel="Harmonic Number",
                ylabel="Correlation (%)",
                grid=True,
                figure_size=(15, 5)
            )

            # Plots all harmonics as a bar chart with label
            plotter.plot(
                harmonics,
                correlations,
                plot_type='bar',
                color=color,
                label=filename,
                new_figure=first_file,
                **vars(plot_config)
            )
            first_file = False

            # Adds text on each bar
            for i, cval in enumerate(correlations):
                plotter.ax.text(
                    harmonics[i],
                    cval + 1,
                    f'{cval:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            # Identifies top 3 harmonics
            sorted_indices = np.argsort(correlations)[::-1]
            top_indices = sorted_indices[:3]
            summary_text = (
                "Dominant Harmonics:\n"
                f"1: {harmonics[top_indices[0]]}th ({correlations[top_indices[0]]:.1f}%)\n"
                f"2: {harmonics[top_indices[1]]}th ({correlations[top_indices[1]]:.1f}%)\n"
                f"3: {harmonics[top_indices[2]]}th ({correlations[top_indices[2]]:.1f}%)"
            )

            # Places the summary text as a small annotation box
            plotter.ax.text(
                0.95, 0.95,
                summary_text,
                transform=plotter.ax.transAxes,
                fontsize=9,
                va='top',
                ha='right',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round')
            )

        # Show a legend for all files and adjust layout
        plotter.ax.legend(loc='upper left', frameon=True)
        plotter.fig.tight_layout()
        plotter.show()

    def on_closing(self):
        """
        This method runs when the user closes the analysis window.
        It destroys this window and shows the main_root window again if it exists.
        """
        self.root.destroy()
        if self.main_root:
            self.main_root.deiconify()
