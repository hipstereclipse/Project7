import numpy as np
from Simulation import StringSimulation, SimulationParameters

def easy_select():
    easynum = 5
    if easynum == 1:
        return 'euler'
    elif easynum == 2:
        return  'rk2'
    elif easynum == 3:
        return  'euler_cromer'
    elif easynum == 4:
        return  'leapfrog'
    else:
        return 'rk4'

def main():
    """Main function to run the simulation."""
    # Lets me make some different simulation parameters easily
    params = SimulationParameters(
        num_segments=25,  # Number of segments (spaces in between objects)
        spring_constant=1000.0,  # Strong springs for stability
        applied_force = np.array([0.0, 0.0, 10.0]),
        mass=0.001,  # Mass for each point
        dt=0.0001,  # Small time step for simulation stability
        start_point=np.array([-1.0, 0.0, 0.0]),  # Start at -1 on x-axis
        end_point=np.array([1.0, 0.0, 0.0]),  # End at +1 on x-axis
        integration_method= easy_select()
    )

    # Creates and runs the simulation
    simulation = StringSimulation(params)
    simulation.run()

main()