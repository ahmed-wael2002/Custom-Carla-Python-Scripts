#!/usr/bin/env python

import carla
import math

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world
    world = client.get_world()

    # Get the blueprint for a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.toyota.prius')

    # Get a spawn point
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    try:
        # Get the bounding box
        bounding_box = vehicle.bounding_box

        # Get dimensions
        length = bounding_box.extent.x * 2  # Multiply by 2 because extent is half-length
        width = bounding_box.extent.y * 2
        height = bounding_box.extent.z * 2

        print(f"Vehicle Dimensions:")
        print(f"Length: {length:.2f} meters")
        print(f"Width: {width:.2f} meters")
        print(f"Height: {height:.2f} meters")

    finally:
        # Clean up
        vehicle.destroy()

if __name__ == '__main__':
    main() 