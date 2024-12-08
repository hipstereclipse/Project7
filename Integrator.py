"""Numerical integration methods for physics simulation testing."""

import numpy as np
from typing import List, Callable, Tuple

def euler_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
               fixed_masses: List[bool], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Euler integration step.
    Simple method: x(t+dt) = x(t) + v(t)dt
                  v(t+dt) = v(t) + a(t)dt

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        accelerations: Nx3 array of accelerations
        fixed_masses: List of booleans indicating which masses are fixed
        dt: Time step

    Returns:
        Tuple of updated (positions, velocities)
    """
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_positions[i] += velocities[i] * dt
            new_velocities[i] += accelerations[i] * dt

    return new_positions, new_velocities

def euler_cromer_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
                     fixed_masses: List[bool], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Euler-Cromer integration step.
    Semi-implicit: v(t+dt) = v(t) + a(t)dt
                  x(t+dt) = x(t) + v(t+dt)dt
    """
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_velocities[i] += accelerations[i] * dt
            new_positions[i] += new_velocities[i] * dt

    return new_positions, new_velocities

def rk2_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
             fixed_masses: List[bool], dt: float,
             get_accelerations: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs RK2 (midpoint method) integration step.
    Second-order accurate:
    1. Calculate midpoint velocities and positions
    2. Use midpoint values to calculate final step
    """
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    # Stores initial values
    init_positions = positions.copy()
    init_velocities = velocities.copy()

    # Moves to midpoint
    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_velocities[i] += accelerations[i] * (dt / 2)
            new_positions[i] += init_velocities[i] * (dt / 2)

    # Gets midpoint accelerations
    mid_accelerations = get_accelerations(new_positions)

    # Resets back to original position and computes full step
    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_positions[i] = init_positions[i] + init_velocities[i] * dt
            new_velocities[i] = init_velocities[i] + mid_accelerations[i] * dt

    return new_positions, new_velocities

def leapfrog_step(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
                  fixed_masses: List[bool], dt: float,
                  get_accelerations: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs leapfrog integration step.
    Symplectic method that conserves energy better than Euler methods.
    """
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    # First half position step
    # pos_half = v * dt/2
    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_positions[i] += velocities[i] * (dt / 2)

    # Get accelerations at midpoint
    mid_accelerations = get_accelerations(new_positions)

    # Full velocity step
    # v = a(midpoint) * dt
    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_velocities[i] += mid_accelerations[i] * dt

    # Second half position step
    # pos_half = v * dt/2
    for i in range(len(positions)):
        if not fixed_masses[i]:
            new_positions[i] += new_velocities[i] * (dt / 2)

    return new_positions, new_velocities

# Maps integration methods to their functions
INTEGRATORS = {
    'euler': euler_step,
    'euler_cromer': euler_cromer_step,
    'rk2': rk2_step,
    'leapfrog': leapfrog_step
}