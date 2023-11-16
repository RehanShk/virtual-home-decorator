import cv2
import math
import numpy as np
import websockets
import asyncio
import json
import os
from helping_functions import  overlay_item, get_differences, smooth_frames, create_buttons, determine_mouse_action

# Load the wall image and item image with an alpha channel
image_path = os.path.join(os.getcwd(), "Images", "decor_item1.png")
item_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image_path1 = os.path.join(os.getcwd(), "Images", "wall1.jpg")
wall_image = cv2.imread(image_path1)
original_item_image = item_image.copy()

# Get the dimensions of the wall image
wall_height, wall_width, _ = wall_image.shape

# Initialize item position variables
item_last_x = (wall_width - item_image.shape[1]) // 2
item_last_y = (wall_height - item_image.shape[0]) // 2

stop = False
stop_mouse = True

# Number of frames to average for smoothing
NUM_FRAMES_TO_SMOOTH = 1

# Lists to store hand landmark positions for smoothing
landmark_positions_x = []
landmark_positions_y = []

# Initialize mouse position variables
last_mouse_x, last_mouse_y = item_last_x, item_last_y

# calculate the maximum allowed dimensions for the item image
original_width = item_image.shape[1]
original_height = item_image.shape[0]

max_item_width = int(item_image.shape[1] + 50)
max_item_height = int(item_image.shape[0] + 50)
min_item_width = int(item_image.shape[1] - 50)
min_item_height = int(item_image.shape[0] - 50)

def determine_mouse_action(x, y, item_image, original_item_image, stop_mouse, radius, add, sub, reset, item_last_x, item_last_y):
    add_button_center = add
    sub_button_center = sub
    reset_button_center = reset

    # Check if the click is inside the increase button
    if math.sqrt((x - add_button_center[0])**2 + (y - add_button_center[1])**2) <= radius:
        # Increase button clicked
        new_width = int(original_item_image.shape[1] * 1.02)
        new_height = int(original_item_image.shape[0] * 1.02)

        if new_width < max_item_width:
            original_item_image = cv2.resize(original_item_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            item_image = original_item_image.copy()

    # Check if the click is inside the decrease button
    elif math.sqrt((x - sub_button_center[0])**2 + (y - sub_button_center[1])**2) <= radius:
        # Decrease button clicked
        new_width = int(original_item_image.shape[1] * 0.98)
        new_height = int(original_item_image.shape[0] * 0.98)

        if new_width > min_item_width:
            original_item_image = cv2.resize(original_item_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            item_image = original_item_image.copy()

    elif math.sqrt((x - reset_button_center[0])**2 + (y - reset_button_center[1])**2) <= radius:
        # Reset button clicked
        item_image = cv2.imread(r"E:\Startup Files\App\startup\wall_decorator_app\Images\decor_item1.jpg", cv2.IMREAD_UNCHANGED)
    
    if item_last_x <= x < item_last_x + item_image.shape[1] and item_last_y <= y < item_last_y + item_image.shape[0]:
            # Update the item_last_x and item_last_y to the clicked position
            item_last_x = x - item_image.shape[1] // 2
            item_last_y = y - item_image.shape[0] // 2
            stop_mouse = not stop_mouse
        
    return item_image, stop_mouse

# Mouse click event handler
def mouse_click(event, x, y, flags, param):
    global item_image, original_item_image, stop_mouse
    radius, add, sub, reset= param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        item_image, stop_mouse = determine_mouse_action(x, y, item_image, original_item_image, stop_mouse, radius, add, sub, reset, item_last_x, item_last_y)


async def receive_coordinates(websocket, path):
    async for message in websocket:
        coordinates = json.loads(message)
        yield coordinates['x'], coordinates['y']

async def main(websocket, path):
    
    global item_last_x, item_last_y, landmark_positions_x, landmark_positions_y, NUM_FRAMES_TO_SMOOTH
    async for x, y in receive_coordinates(websocket, path):
        print(x, y)
        # Get a copy of the wall image
        frame = wall_image.copy()
        frame, radius, add, sub, reset= create_buttons(frame.copy())
        
        if stop_mouse == False:
            smoothed_x, smoothed_y = smooth_frames(x, y, landmark_positions_x, landmark_positions_y, NUM_FRAMES_TO_SMOOTH)
            top_left_x = smoothed_x - item_image.shape[1] // 2
            top_left_y = smoothed_y - item_image.shape[0] // 2

            # Ensure the item does not move beyond the image boundaries
            top_left_x = max(0, min(top_left_x, wall_width - item_image.shape[1]))
            top_left_y = max(0, min(top_left_y, wall_height - item_image.shape[0]))

            item_last_x = top_left_x
            item_last_y = top_left_y

        # If Hand not detected, then display item on last detected position of hand or at the center.
        top_left_x = item_last_x
        top_left_y = item_last_y
        frame = overlay_item(frame, top_left_y, top_left_x, item_image)

        cv2.imshow('Image', frame)

        cv2.setMouseCallback('Image', mouse_click, (radius, add, sub, reset))

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

# Run the WebSocket server
if __name__ == "__main__":
    start_server = websockets.serve(main, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
