"""Numerical integration methods for physics simulation testing."""

import numpy as np
from typing import List, Callable, Tuple

def euler_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
               fixed_masses: List[bool], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Euler integration step.
    Simplest method: x(t+dt) = x(t) + v(t)dt
                     v(t+dt) = v(t) + a(t)dt

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        accelerations: Nx3 array of accelerations
        fixed_masses: List of booleans indicating which masses are fixed
        dt: Time step

    Returns:
        Updated positions & velocities where the calculation is accurate to the first order but does not conserve energy
    """
    # Our temp then final values
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    for i in range(len(positions)):
        if not fixed_masses[i]: # Skips the fixed masses
            new_positions[i] += velocities[i] * dt # Updates position using original velocity vector
            new_velocities[i] += accelerations[i] * dt # Updates velocity according to acceleration

    return new_positions, new_velocities

def euler_cromer_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
                     fixed_masses: List[bool], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Euler-Cromer integration step.
    Semi-implicit: v(t+dt) = v(t) + a(t)dt
                  x(t+dt) = x(t) + v(t+dt)dt

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        accelerations: Nx3 array of accelerations
        fixed_masses: List of booleans indicating which masses are fixed
        dt: Time step

    Returns:
        Updated positions & velocities where energy is conserved and the calculation is accurate to the first order
    """
    # Our temp then final values
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    # Conserves energy by not doing calculation in stupid order
    for i in range(len(positions)):
        if not fixed_masses[i]: # Skips the fixed masses
            new_velocities[i] += accelerations[i] * dt # Updates velocity first so it is always most appropriate given current position
            new_positions[i] += new_velocities[i] * dt # Kobe dunk

    return new_positions, new_velocities

def rk2_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
             fixed_masses: List[bool], dt: float, get_accelerations: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs RK2 (midpoint method) integration step.
    Second-order accurate:
            1. Calculate midpoint velocities and positions
            2. Use midpoint values to calculate final step

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        accelerations: Nx3 array of accelerations
        fixed_masses: List of booleans indicating which masses are fixed
        dt: Time step

    Returns:
        Updated positions & velocities where the calculation is accurate to the second order but does not conserve energy
    """
    # Our temp then final values
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    # Stores initial values
    init_positions = positions.copy()
    init_velocities = velocities.copy()

    # Moves to midpoint
    for i in range(len(positions)): # Goes through all possible positions
        if not fixed_masses[i]: # Skips the fixed masses
            new_velocities[i] += accelerations[i] * (dt / 2) # Updates velocity first
            new_positions[i] += init_velocities[i] * (dt / 2) # Updates position afterward

    # Gets midpoint accelerations
    mid_accelerations = get_accelerations(new_positions)

    # Resets back to original position and computes the full step using the midpoint acceleration
    for i in range(len(positions)):
        if not fixed_masses[i]: # Skips the fixed masses
            new_positions[i] = init_positions[i] + init_velocities[i] * dt
            new_velocities[i] = init_velocities[i] + mid_accelerations[i] * dt # Using new midpoint acceleration

    return new_positions, new_velocities

def leapfrog_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
                  fixed_masses: List[bool], dt: float,
                  get_accelerations: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs leapfrog integration step.
    Symplectic method that conserves energy better than Euler methods.

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        accelerations: Nx3 array of accelerations
        fixed_masses: List of booleans indicating which masses are fixed
        dt: Time step

    Returns:
        Updated positions & velocities where energy is conserved and the calculation is accurate to the second order
    """
    # Our temp then final values
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    # First half position step
    # pos_half = v * dt/2
    for i in range(len(positions)):
        if not fixed_masses[i]: # Skips the fixed masses
            new_positions[i] += velocities[i] * (dt / 2)

    # Get accelerations at midpoint to get better velocity estimate for the midpoint riemann sum of our program
    mid_accelerations = get_accelerations(new_positions)

    # Full velocity step
    # v = a(midpoint) * dt
    for i in range(len(positions)):
        if not fixed_masses[i]: # Skips the fixed masses
            new_velocities[i] += mid_accelerations[i] * dt

    # Second half position step
    # pos_half = v * dt/2
    for i in range(len(positions)):
        if not fixed_masses[i]: # Skips the fixed masses
            new_positions[i] += new_velocities[i] * (dt / 2) # Effectively our kobe dunk

    return new_positions, new_velocities


def rk4_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
                        fixed_masses: List[bool], dt: float,
                        get_accelerations: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs RK4 symplectic integration step.
    Fourth-order accurate while preserving energy conservation properties.
    Uses a composition of steps to maintain symplectic nature while achieving RK4 accuracy.

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        accelerations: Nx3 array of accelerations
        fixed_masses: List of booleans indicating which masses are fixed
        dt: Time step
        get_accelerations: Function to compute accelerations at given positions

    Returns:
        Updated positions & velocities where energy is conserved and the calculation is accurate to the fourth order
    """
    # Weighting for our symplectic RK4
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = 1.0 - 2.0 * w1

    # Creates arrays for intermediate steps
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    # First symplectic step with weight w1
    for i in range(len(positions)):
        if not fixed_masses[i]:
            # First half position update
            new_positions[i] += 0.5 * w1 * dt * velocities[i]

            # Full velocity update using accelerations at intermediate position
            acc = get_accelerations(new_positions)[i]
            new_velocities[i] += w1 * dt * acc

            # Second half position update with updated velocity
            new_positions[i] += 0.5 * w1 * dt * new_velocities[i]

    # Middle symplectic step with weight w0
    for i in range(len(positions)):
        if not fixed_masses[i]:
            # First half position update
            new_positions[i] += 0.5 * w0 * dt * new_velocities[i]

            # Full velocity update using accelerations at intermediate position
            acc = get_accelerations(new_positions)[i]
            new_velocities[i] += w0 * dt * acc

            # Second half position update with updated velocity
            new_positions[i] += 0.5 * w0 * dt * new_velocities[i]

    # Final symplectic step with weight w1
    for i in range(len(positions)):
        if not fixed_masses[i]:
            # First half position update
            new_positions[i] += 0.5 * w1 * dt * new_velocities[i]

            # Full velocity update using accelerations at intermediate position
            acc = get_accelerations(new_positions)[i]
            new_velocities[i] += w1 * dt * acc

            # Second half position update with updated velocity
            new_positions[i] += 0.5 * w1 * dt * new_velocities[i]

    return new_positions, new_velocities

# Maps integration methods to their functions
INTEGRATORS = {
    'euler': euler_step,
    'euler_cromer': euler_cromer_step,
    'rk2': rk2_step,
    'leapfrog': leapfrog_step,
    'rk4': rk4_step
}