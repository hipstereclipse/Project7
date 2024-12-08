import numpy as np
from Simulation import StringSimulation, SimulationParameters
from Visualization import StringSimulationSetup

def main():
    """Main function to run the simulation."""
    # Create and run the setup GUI to get simulation parameters
    setup = StringSimulationSetup()
    params = setup.get_parameters()

    # Check if the user closed the window without starting
    if params is None:
        print("Simulation cancelled by user")
        return

    print("\nStarting simulation with parameters:")
    print(f"Number of segments: {params.num_segments}")
    print(f"Spring constant: {params.spring_constant}")
    print(f"Mass per point: {params.mass}")
    print(f"Time step: {params.dt}")
    print(f"Integration method: {params.integration_method}")
    print("\nControls:")
    print("- Left click and drag: Rotate view")
    print("- Scroll wheel: Zoom in/out")
    print("- Space bar: Apply vertical force to middle mass")
    print("- 'p': Pause/Resume simulation")

    # Create and run the simulation with the selected parameters
    simulation = StringSimulation(params)
    simulation.run()

main()