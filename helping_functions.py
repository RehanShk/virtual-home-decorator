#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import math
import numpy as np

item_image = cv2.imread("Images/decor_item1.png", cv2.IMREAD_UNCHANGED)

max_item_width = int(item_image.shape[1] + 50)
max_item_height = int(item_image.shape[0] + 50)
min_item_width = int(item_image.shape[1] - 50)
min_item_height = int(item_image.shape[0] - 50)

def resize_image_to_fullscreen(image):
    # Load your room image
    image = cv2.imread(image)

    # Get screen size using ctypes (Windows-specific)
    screen_width, screen_height = (1920, 1080)

    # Resize the room image to match the screen resolution
    image = cv2.resize(image, (screen_width, screen_height))
    return image

def overlay_item(frame, top_left_y, top_left_x, item_image):
    # Create a mask of the item
    item_mask = item_image[:, :, 3]

    # Extract the item region from the wall image
    item_region = frame[top_left_y:top_left_y + item_image.shape[0],
                         top_left_x:top_left_x + item_image.shape[1]]

    # Resize the item image to match the dimensions of the item region
    resized_item_image = cv2.resize(item_image[:, :, 0:3], (item_region.shape[1], item_region.shape[0]))

    # Overlay the item onto the item region using the mask
    item_region[:, :, 0:3] = item_region[:, :, 0:3] * (1 - item_mask[:, :, np.newaxis] / 255.0) + \
                              resized_item_image * (item_mask[:, :, np.newaxis] / 255.0)
    return frame

def get_differences(hand_landmarks, wall_width, wall_height):
    # Get hand landmark 8 and 12 positions
    hand_landmark_8_x = int(hand_landmarks.landmark[8].x * wall_width)
    hand_landmark_8_y = int(hand_landmarks.landmark[8].y * wall_height)
    hand_landmark_12_x = int(hand_landmarks.landmark[12].x * wall_width)
    hand_landmark_12_y = int(hand_landmarks.landmark[12].y * wall_height)
    hand_landmark_7_x = int(hand_landmarks.landmark[7].x * wall_width)
    hand_landmark_7_y = int(hand_landmarks.landmark[7].y * wall_height)
    hand_landmark_11_x = int(hand_landmarks.landmark[11].x * wall_width)
    hand_landmark_11_y = int(hand_landmarks.landmark[11].y * wall_height)
    hand_landmark_4_x = int(hand_landmarks.landmark[4].x * wall_width)
    hand_landmark_4_y = int(hand_landmarks.landmark[4].y * wall_height)
    hand_landmark_16_x = int(hand_landmarks.landmark[16].x * wall_width)
    hand_landmark_16_y = int(hand_landmarks.landmark[16].y * wall_height)
    hand_landmark_5_x = int(hand_landmarks.landmark[5].x * wall_width)
    hand_landmark_5_y = int(hand_landmarks.landmark[5].y * wall_height)


    # Calculate the difference between hand landmark 8 and 12
    diff_8_12 = math.sqrt((hand_landmark_12_x - hand_landmark_8_x) ** 2 +
                                         (hand_landmark_12_y - hand_landmark_8_y) ** 2)
    diff_7_11 = math.sqrt((hand_landmark_11_x - hand_landmark_7_x) ** 2 +
                                         (hand_landmark_11_y - hand_landmark_7_y) ** 2)
    diff_12_16 = math.sqrt((hand_landmark_12_x - hand_landmark_16_x) ** 2 +
                                         (hand_landmark_12_y - hand_landmark_16_y) ** 2)
    diff_4_12 = math.sqrt((hand_landmark_12_x - hand_landmark_4_x) ** 2 +
                                         (hand_landmark_12_y - hand_landmark_4_y) ** 2)
    diff_4_8 = math.sqrt((hand_landmark_4_x - hand_landmark_8_x) ** 2 +
                                         (hand_landmark_4_y - hand_landmark_8_y) ** 2)

    diff_4_5 = math.sqrt((hand_landmark_4_x - hand_landmark_5_x) ** 2 +
                                         (hand_landmark_4_y - hand_landmark_5_y) ** 2)
    
    return diff_8_12, diff_7_11, diff_12_16, diff_4_12, diff_4_8, diff_4_5, hand_landmark_8_x, hand_landmark_8_y

def smooth_frames(hand_landmark_8_x, hand_landmark_8_y, landmark_positions_x, landmark_positions_y, NUM_FRAMES_TO_SMOOTH):
    # Store hand landmark 8 positions for smoothing
    landmark_positions_x.append(hand_landmark_8_x)
    landmark_positions_y.append(hand_landmark_8_y)

    # Keep only the last N positions for smoothing
    landmark_positions_x = landmark_positions_x[-NUM_FRAMES_TO_SMOOTH:]
    landmark_positions_y = landmark_positions_y[-NUM_FRAMES_TO_SMOOTH:]

    # Calculate the average hand landmark positions for smoothing
    smoothed_x = sum(landmark_positions_x) // len(landmark_positions_x)
    smoothed_y = sum(landmark_positions_y) // len(landmark_positions_y)

    return smoothed_x, smoothed_y

def is_left_hand(hand_landmarks):
    landmark_14_x = hand_landmarks.landmark[14].x
    landmark_8_x = hand_landmarks.landmark[8].x
    if landmark_14_x > landmark_8_x:
        return True
    else:
        return False

def determine_action(x, y, item_image, original_item_image, radius, add, sub, reset):
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
        item_image = cv2.imread("Images/decor_item1.png", cv2.IMREAD_UNCHANGED)
        
    return item_image

def create_buttons(image):
    add_button_center = (40, 40)
    sub_button_center = (40, 120)
    reset_button_center = (40, 200)
    radius = 30

    # Calculate text size and position for increase button
    add_text = "+"
    add_text_size = cv2.getTextSize(add_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    add_text_position = (add_button_center[0] - add_text_size[0] // 2, add_button_center[1] + add_text_size[1] // 2)

    # Draw increase button (circle)
    cv2.circle(image, add_button_center, radius, (0, 0, 0), -1)
    # Draw increase icon (plus sign) at the center of the circle
    cv2.putText(image, add_text, add_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    # Calculate text size and position for decrease button
    sub_text = "-"
    sub_text_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    sub_text_position = (sub_button_center[0] - sub_text_size[0] // 2, sub_button_center[1] + sub_text_size[1] // 2)

    # Draw decrease button (circle)
    cv2.circle(image, sub_button_center, radius, (0, 0, 0), -1)
    # Draw decrease icon (minus sign) at the center of the circle
    cv2.putText(image, sub_text, sub_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Calculate text size and position for decrease button
    reset_text = "R"
    reset_text_size = cv2.getTextSize(reset_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    reset_text_position = (reset_button_center[0] - reset_text_size[0] // 2, reset_button_center[1] + reset_text_size[1] // 2)

    # Draw decrease button (circle)
    cv2.circle(image, reset_button_center, radius, (0, 0, 0), -1)
    # Draw decrease icon (minus sign) at the center of the circle
    cv2.putText(image, reset_text, reset_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    return image, radius, add_button_center, sub_button_center, reset_button_center

def determine_mouse_action(x, y, item_image, original_item_image, stop_mouse, radius, add, sub, reset):
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
        item_image = cv2.imread("Images/decor_item1.png", cv2.IMREAD_UNCHANGED)
    else:
        stop_mouse = not stop_mouse
        
    return item_image, stop_mouse
