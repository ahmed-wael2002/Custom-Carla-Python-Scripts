#!/usr/bin/env python

import carla
import math
import time

def find_nearest_traffic_light(player_location, traffic_lights):
    """Find the nearest traffic light to the player"""
    min_dist = float('inf')
    nearest_tl = None
    
    for tl in traffic_lights:
        dist = player_location.distance(tl.get_location())
        if dist < min_dist:
            min_dist = dist
            nearest_tl = tl
            
    return nearest_tl, min_dist

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Get the world
    world = client.get_world()
    
    try:
        # Get the player vehicle
        player = None
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle'):
                player = actor
                break
                
        if player is None:
            print("No vehicle found in the world")
            return
            
        # Get all traffic lights
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        
        # Find nearest traffic light
        player_location = player.get_location()
        nearest_tl, distance = find_nearest_traffic_light(player_location, traffic_lights)
        
        if nearest_tl is None:
            print("No traffic lights found in the world")
            return
            
        print(f"Nearest traffic light found at distance: {distance:.2f} meters")
        
        # Get current state
        current_state = nearest_tl.get_state()
        print(f"Current traffic light state: {current_state}")
        
        # Change state to next state
        if current_state == carla.TrafficLightState.Red:
            new_state = carla.TrafficLightState.Green
        elif current_state == carla.TrafficLightState.Green:
            new_state = carla.TrafficLightState.Yellow
        else:
            new_state = carla.TrafficLightState.Red
            
        # Set new state
        nearest_tl.set_state(new_state)
        print(f"Changed traffic light state to: {new_state}")
        
        # Wait a bit to see the change
        time.sleep(2)
        
    finally:
        # Clean up
        print("Done!")

if __name__ == '__main__':
    main() 