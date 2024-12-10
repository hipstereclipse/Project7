from Simulation import StringSimulation
from Visualization import StringSimulationSetup, AnalysisVisualizer


def main():
    """
    Main function offering choice between running new simulations or analyzing existing data.
    The function provides two primary modes of operation:
    1. Simulation Mode: Run new string physics simulations with configurable parameters
    2. Analysis Mode: Analyze and compare previously saved simulation data
    """
    print("\nString Physics Simulation and Analysis Tool")
    print("==========================================")
    print("\nChoose operation mode:")
    print("1. Run new simulation")
    print("2. Analyze existing simulation data")
    print("3. Exit program")

    while True:
        try:
            choice = input("\nEnter choice (1-3): ")
            if choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nProgram terminated by user")
            return

    if choice == '1':
        # Simulation mode
        while True:
            # Create and run the setup GUI to get simulation parameters
            setup = StringSimulationSetup()
            params = setup.get_parameters()

            # Check if the user closed the window without starting
            if params is None:
                print("Simulation cancelled by user")
                break

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
            print("- 'Return to Setup': Close simulation and restart setup")

            # Create and run the simulation with the selected parameters
            simulation = StringSimulation(params)
            should_restart = simulation.run()

            if not should_restart:  # If we're not restarting, break the loop
                break

            # After simulation ends, offer to switch to analysis mode
            print("\nSimulation completed. Would you like to:")
            print("1. Run another simulation")
            print("2. Switch to analysis mode")
            print("3. Exit program")

            post_sim_choice = input("\nEnter choice (1-3): ")
            if post_sim_choice == '2':
                # Launch analysis mode
                analyzer = AnalysisVisualizer()
                analyzer.run()
                break
            elif post_sim_choice == '3':
                break

    elif choice == '2':
        # Analysis mode for existing data
        print("\nLaunching Analysis Tool...")
        print("You can load and analyze multiple simulation files through the interface.")
        analyzer = AnalysisVisualizer()
        analyzer.run()

    else:  # choice == '3'
        print("\nExiting program. Goodbye!")



main()