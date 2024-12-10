import tkinter as tk
from tkinter import ttk
from Simulation import StringSimulation
from Visualization import StringSimulationSetup, AnalysisVisualizer


class MainMenu:
    """
    Main menu GUI for the String Physics Simulation and Analysis Tool.
    Provides a graphical interface for choosing between simulation and analysis modes.
    """

    def __init__(self):
        """Initialize the main menu window with default settings."""
        # Create the main window
        self.root = tk.Tk()
        self.root.title("String Physics Simulation and Analysis Tool")

        # Calculate window size (30% of screen)
        window_width = int(self.root.winfo_screenwidth() * 0.27)
        window_height = int(self.root.winfo_screenheight() * 0.6)

        # Calculate position to center the window
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)

        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Make window non-resizable for consistent layout
        self.root.resizable(False, False)

        # Configure the grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Setup the GUI elements
        self.setup_gui()

        # Track if we're returning to menu from simulation
        self.returning_from_simulation = False

    def setup_gui(self):
        """Set up the main GUI elements."""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Title label with large font
        title_label = ttk.Label(
            main_frame,
            text="String Physics\nSimulation and Analysis Tool",
            font=("Arial", 16, "bold"),
            justify="center"
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Description text
        description = (
            "This tool allows you to simulate and analyze\n"
            "the physics of vibrating strings using various\n"
            "numerical methods and parameters."
        )
        desc_label = ttk.Label(
            main_frame,
            text=description,
            justify="center",
            font=("Arial", 10)
        )
        desc_label.grid(row=1, column=0, pady=(0, 30))

        # Button frame for consistent button sizing
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=(0, 20))

        # Style configuration for buttons
        button_style = ttk.Style()
        button_style.configure(
            "Action.TButton",
            font=("Arial", 11),
            padding=10
        )

        # Create buttons with consistent width
        button_width = 25

        # New Simulation button
        sim_button = ttk.Button(
            button_frame,
            text="Run New Simulation",
            command=self.start_simulation,
            style="Action.TButton",
            width=button_width
        )
        sim_button.grid(row=0, column=0, pady=5)

        # Analysis button
        analysis_button = ttk.Button(
            button_frame,
            text="Analyze Simulation Data",
            command=self.start_analysis,
            style="Action.TButton",
            width=button_width
        )
        analysis_button.grid(row=1, column=0, pady=5)

        # Exit button
        exit_button = ttk.Button(
            button_frame,
            text="Exit Program",
            command=self.exit_program,
            style="Action.TButton",
            width=button_width
        )
        exit_button.grid(row=2, column=0, pady=5)

        # Version info at bottom
        version_label = ttk.Label(
            main_frame,
            text="Version 1.0",
            font=("Arial", 8),
            foreground="gray"
        )
        version_label.grid(row=3, column=0, pady=(10, 0))

    def start_simulation(self):
        """Launch the simulation setup and handling loop."""
        self.root.withdraw()  # Hide main menu

        while True:
            # Create and run the setup GUI
            setup = StringSimulationSetup()
            params = setup.get_parameters()

            # Check if setup was cancelled
            if params is None:
                break

            # Create and run simulation
            simulation = StringSimulation(params)
            should_restart = simulation.run()

            if not should_restart:
                break

        self.root.deiconify()  # Show main menu again

    def start_analysis(self):
        """Launch the analysis visualization system."""
        self.root.withdraw()  # Hide main menu

        # Create and run analyzer
        analyzer = AnalysisVisualizer()
        analyzer.run()

        self.root.deiconify()  # Show main menu again

    def exit_program(self):
        """Clean up and exit the program."""
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the main menu."""
        self.root.mainloop()


def main():
    """
    Main entry point for the String Physics Simulation and Analysis Tool.
    Creates and runs the main menu GUI.
    """
    menu = MainMenu()
    menu.run()



main()