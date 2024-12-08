import numpy as np
class SimulationObject:
    def __init__(self, x, y, z, vx, vy, vz, ax, ay, az, mass, obj_id, pinned=False):
        """
        Initializes a new 3D object that will act as our mass chunk for the matter the string is made up of.

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - z (float): Z-coordinate.
        - pinned (bool): Flag indicating if the object is pinned.
        - mass (float or int): The mass of our mass
        - obj_id (int): The individual identifier number for a mass, used for figuring out what is attached to what
        """
        # Store coordinates as a NumPy array
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.array([vx,vy,vz], dtype=float)
        self.acceleration = np.array([ax,ay,az], dtype=float)
        self.pinned = pinned
        self.mass = mass
        self.obj_id = obj_id

    def pin(self):
        """Pin the object in its current position."""
        self.pinned = True

    def unpin(self):
        """Unpin the object, allowing its position to be changed."""
        self.pinned = False

    def move(self, new_position: np.ndarray, new_velocity: np.ndarray, new_acceleration: np.ndarray):
        """
        Moves the object to a new position if it's not pinned.
        """
        if not self.pinned:
            self.position = new_position
            self.velocity = new_velocity
            self.acceleration = new_acceleration
            print(f"Object {self.obj_id} moved to {self.position}.")
        else:
            print(f"Object {self.obj_id} is pinned and cannot be moved.")

    # A built-in Python function that returns a string representation of an object.
    def __repr__(self):
        return (f"position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}"
                f"mass={self.mass}, object_id={self.obj_id}, pinned={self.pinned})")