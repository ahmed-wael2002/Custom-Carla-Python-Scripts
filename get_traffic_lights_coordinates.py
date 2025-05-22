#!/usr/bin/env python

import carla
import csv


def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world
    world = client.get_world()

    # Get all traffic lights
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    # Prepare data for CSV
    data = [(tl.id, loc.x, loc.y, loc.z) for tl in traffic_lights for loc in [tl.get_location()]]

    # Write to CSV
    with open('traffic_lights.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'x', 'y', 'z'])
        writer.writerows(data)

    print(f"Exported {len(data)} traffic lights to traffic_lights.csv")


if __name__ == '__main__':
    main() 