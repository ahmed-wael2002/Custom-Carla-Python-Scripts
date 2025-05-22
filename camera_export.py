#!/usr/bin/env python

import carla
import cv2
import numpy as np
import os
import time

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world
    world = client.get_world()

    try:
        # Wait for the player's vehicle to be available
        print("Waiting for player's vehicle...")
        player = None
        while player is None:
            time.sleep(0.1)
            # Get all actors in the world
            actors = world.get_actors()
            # Find the player's vehicle (usually the first vehicle)
            for actor in actors:
                if actor.type_id.startswith('vehicle.'):
                    player = actor
                    print(f"Found player's vehicle: {actor.type_id}")
                    break

        if player is None:
            print("Could not find player's vehicle!")
            return

        # Create a camera blueprint
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')

        # Spawn the camera
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=player)

        # Create output directory if it doesn't exist
        output_dir = 'camera_output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Callback function to save the image
        def save_image(image):
            # Convert the image to a numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            
            # Save the image
            timestamp = int(time.time() * 1000)
            filename = os.path.join(output_dir, f'camera_{timestamp}.jpg')
            cv2.imwrite(filename, array)
            print(f"Saved image: {filename}")

        # Register the callback
        camera.listen(save_image)

        # Keep the script running
        print("Camera is recording. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping the script...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if 'camera' in locals():
            camera.destroy()
        print("Cleanup complete.")

if __name__ == '__main__':
    main() 