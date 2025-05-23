#!/usr/bin/env python

import glob
import os
import sys
import weakref
import numpy as np
import pygame
import carla
from carla import ColorConverter as cc

class TrafficLightClient:
    def __init__(self, world, parent_actor, detector_class, model_path=None, **detector_kwargs):
        self.detector = detector_class(model_path, **detector_kwargs) if model_path else detector_class(**detector_kwargs)
        self.world = world
        self.parent = parent_actor
        self.sensor = None
        self._surface = None
        
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '640')
        blueprint.set_attribute('image_size_y', '480')
        
        self.sensor = self.world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=2.0, z=1.4)),
            attach_to=self.parent
        )
        
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._parse_image(weak_self, image))
    
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        processed_frame = self.detector.process_carla_frame(array)
        # Optionally, you could extract detection results here for further logic (e.g., traffic light state decision)
        # For now, just visualize with confidence overlay
        
        self._surface = pygame.surfarray.make_surface(processed_frame.swapaxes(0, 1))
    
    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))
    
    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        vehicle = None
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle'):
                vehicle = actor
                break
        
        if vehicle is None:
            print("No vehicle found in the world")
            return
        
        # Import your detector class here
        from traffic_light_detector import TrafficLightDetector
        
        # Initialize with your detector class and model path
        traffic_light_client = TrafficLightClient(
            world, 
            vehicle,
            detector_class=TrafficLightDetector,
            model_path='yolov8m.pt'
        )
        
        pygame.init()
        display = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('Traffic Light Detection')
        
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            display.fill((0, 0, 0))
            traffic_light_client.render(display)
            pygame.display.flip()
            clock.tick(60)
        
    finally:
        if 'traffic_light_client' in locals():
            traffic_light_client.destroy()
        pygame.quit()

if __name__ == '__main__':
    main() 