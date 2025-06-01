#!/usr/bin/env python

import carla
import math
import time

def find_traffic_light_in_front(vehicle, traffic_lights, max_distance=50.0, angle_threshold=120):
    """
    Find the nearest traffic light that is in front of the vehicle
    Args:
        vehicle: The vehicle to check from
        traffic_lights: List of traffic lights to check
        max_distance: Maximum distance to consider (in meters)
        angle_threshold: Angle threshold in degrees (default 120 for wider coverage)
    Returns:
        tuple: (traffic_light, distance) if found, (None, None) otherwise
    """
    if not vehicle.is_alive:
        return None, None
        
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation.yaw
    
    min_dist = float('inf')
    nearest_tl = None
    
    for tl in traffic_lights:
        if not tl.is_alive:
            continue
            
        # Get traffic light location
        tl_location = tl.get_location()
        
        # Calculate distance
        dist = vehicle_location.distance(tl_location)
        
        # Skip if too far
        if dist > max_distance:
            continue
            
        # Calculate angle between vehicle and traffic light
        dx = tl_location.x - vehicle_location.x
        dy = tl_location.y - vehicle_location.y
        angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize angle to be between -180 and 180
        angle = (angle + 180) % 360 - 180
        
        # Check if traffic light is in front of vehicle (within angle_threshold degrees)
        if abs(angle - vehicle_rotation) < angle_threshold/2:
            if dist < min_dist:
                min_dist = dist
                nearest_tl = tl
    
    return nearest_tl, min_dist if nearest_tl else None

def set_traffic_light_state(traffic_light, state):
    """Set the traffic light to the specified state"""
    if not traffic_light.is_alive:
        raise ValueError("Traffic light no longer exists")
        
    if state not in [carla.TrafficLightState.Red, 
                    carla.TrafficLightState.Yellow, 
                    carla.TrafficLightState.Green]:
        raise ValueError("Invalid traffic light state")
    traffic_light.set_state(state)

def find_hero_vehicle(world):
    """Find the vehicle with role_name='hero'"""
    for actor in world.get_actors():
        if (actor.type_id.startswith('vehicle') and 
            actor.attributes.get('role_name') == 'hero' and 
            actor.is_alive):
            return actor
    return None

def print_help():
    """Print available commands"""
    print("\nAvailable commands:")
    print("  r - Set traffic light to RED")
    print("  y - Set traffic light to YELLOW")
    print("  g - Set traffic light to GREEN")
    print("  h - Show this help message")
    print("  q - Quit the program")
    print("  s - Show current traffic light state")
    print("  d - Show distance to current traffic light")

def main():
    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        print("\nTraffic Light Controller Started!")
        print("Connected to CARLA server")
        print_help()
        
        last_traffic_light_id = None
        last_distance = None
        
        while True:
            # Get all traffic lights
            traffic_lights = world.get_actors().filter('traffic.traffic_light')
            
            if not traffic_lights:
                print("\rNo traffic lights found in the world", end='')
                time.sleep(1.0)
                continue
                
            # Find your hero vehicle
            vehicle = find_hero_vehicle(world)
            if vehicle is None:
                print("\rNo hero vehicle found in the world", end='')
                time.sleep(1.0)
                continue

            # Find nearest traffic light in front of vehicle
            nearest_tl, distance = find_traffic_light_in_front(vehicle, traffic_lights)
            
            if nearest_tl is None:
                if last_traffic_light_id is not None:
                    print("\rNo traffic light in front of vehicle within range", end='')
                    last_traffic_light_id = None
                    last_distance = None
                time.sleep(0.2)  # Reduced sleep time for better responsiveness
                continue
                
            # Check if we found a new traffic light
            current_tl_id = nearest_tl.id
            if current_tl_id != last_traffic_light_id:
                print(f"\nFound new traffic light {distance:.2f} meters away")
                last_traffic_light_id = current_tl_id
                last_distance = distance
                
            # Get user input
            command = input("\nEnter command (h for help): ").lower().strip()
            
            if command == 'q':
                print("Quitting...")
                break
            elif command == 'h':
                print_help()
            elif command == 's':
                if nearest_tl.is_alive:
                    print(f"Current traffic light state: {nearest_tl.get_state()}")
                else:
                    print("Traffic light no longer exists")
            elif command == 'd':
                if nearest_tl.is_alive:
                    print(f"Distance to traffic light: {distance:.2f} meters")
                else:
                    print("Traffic light no longer exists")
            elif command in ['r', 'y', 'g']:
                if not nearest_tl.is_alive:
                    print("Traffic light no longer exists")
                    continue
                    
                try:
                    if command == 'r':
                        set_traffic_light_state(nearest_tl, carla.TrafficLightState.Red)
                        print("Changed traffic light to RED")
                    elif command == 'y':
                        set_traffic_light_state(nearest_tl, carla.TrafficLightState.Yellow)
                        print("Changed traffic light to YELLOW")
                    elif command == 'g':
                        set_traffic_light_state(nearest_tl, carla.TrafficLightState.Green)
                        print("Changed traffic light to GREEN")
                except Exception as e:
                    print(f"Failed to change traffic light state: {str(e)}")
            else:
                print("Invalid command. Type 'h' for help.")
        
    except carla.exceptions.ConnectionError:
        print("Failed to connect to CARLA server")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("Done!")

if __name__ == '__main__':
    main() 