"""All available drones for which controller parameters are available."""

from enum import Enum


class Drones(str, Enum):
    """Available drones."""

    cf2x_L250 = "cf2x_L250"
