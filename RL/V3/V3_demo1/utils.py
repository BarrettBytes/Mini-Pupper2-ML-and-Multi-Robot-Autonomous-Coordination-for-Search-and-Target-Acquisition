# utils.py
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pygame
from PIL import Image, ImageDraw
import pygame
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2

def initialize_pygame(dog_image_path, cat_image_path, robot_image_path):
    pygame.init()
    dog_image = pygame.image.load(dog_image_path)
    cat_image = pygame.image.load(cat_image_path)
    robot_image = pygame.image.load(robot_image_path)
    return dog_image, cat_image, robot_image

def get_frame(env, dog_image, cat_image, robot_image):
    frame = pygame.Surface((env.grid_width, env.grid_height))
    frame.fill((255, 255, 255))  # White background

    for obs_pos in env.obstacle_positions:
        x0, y0 = obs_pos[1] * env.cell_width, obs_pos[0] * env.cell_height
        pygame.draw.rect(frame, (128, 128, 128), pygame.Rect(x0, y0, env.cell_width, env.cell_height))
    
    for dog_pos in env.dog_positions:
        x0, y0 = dog_pos[1] * env.cell_width, dog_pos[0] * env.cell_height
        frame.blit(pygame.transform.scale(dog_image, (env.cell_width, env.cell_height)), (x0, y0))
    
    for i, cat_pos in enumerate(env.cat_positions):
        if not env.found_cats[i]:
            x0, y0 = cat_pos[1] * env.cell_width, cat_pos[0] * env.cell_height
            frame.blit(pygame.transform.scale(cat_image, (env.cell_width, env.cell_height)), (x0, y0))
    
    x0, y0 = env.position[1] * env.cell_width, env.position[0] * env.cell_height
    frame.blit(pygame.transform.scale(robot_image, (env.cell_width, env.cell_height)), (x0, y0))

    return frame

def save_video(frames, filename, fps=10):
    if not frames:
        print("No frames to save.")
        return
    
    height, width = frames[0].get_height(), frames[0].get_width()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_array = pygame.surfarray.array3d(frame).transpose([1, 0, 2])
        video.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
    video.release()

def save_plot(performance, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(performance)
    plt.xlabel('Episode')
    plt.ylabel('Performance (Cats reached - Dogs reached)')
    plt.title('Policy Performance Over Time')
    plt.savefig(filename)
    plt.close()

def load_and_resize_image(image_path, target_size=(64, 64)):
    image = mpimg.imread(image_path)
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)
    return image

def create_video_writer(video_filename, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

def surface_to_array(surface):
    """ Convert pygame surface to numpy array. """
    return pygame.surfarray.array3d(surface).transpose([1, 0, 2])

def ensure_numpy_image(image):
    """ Ensure the image is in numpy array format. """
    if isinstance(image, pygame.Surface):
        return surface_to_array(image)
    return image

def add_frame_to_video(out, environment, episode):
    if environment.use_pygame:
        frame = environment.get_frame()
        frame_array = pygame.surfarray.array3d(frame).swapaxes(0, 1)
    else:
        frame = Image.new('RGB', (environment.grid_width, environment.grid_height), color='white')
        draw = ImageDraw.Draw(frame)
        
        # Draw obstacles
        for obs_pos in environment.obstacle_positions:
            x0, y0 = obs_pos[1] * environment.cell_width, obs_pos[0] * environment.cell_height
            draw.rectangle([x0, y0, x0 + environment.cell_width, y0 + environment.cell_height], fill='gray')
        
        # Draw dogs
        for dog_pos in environment.dog_positions:
            x0, y0 = dog_pos[1] * environment.cell_width, dog_pos[0] * environment.cell_height
            draw.rectangle([x0, y0, x0 + environment.cell_width, y0 + environment.cell_height], fill='red')
        
        # Draw cats
        for i, cat_pos in enumerate(environment.cat_positions):
            if not environment.found_cats[i]:
                x0, y0 = cat_pos[1] * environment.cell_width, cat_pos[0] * environment.cell_height
                draw.rectangle([x0, y0, x0 + environment.cell_width, y0 + environment.cell_height], fill='green')
        
        # Draw robot
        x0, y0 = environment.position[1] * environment.cell_width, environment.position[0] * environment.cell_height
        draw.rectangle([x0, y0, x0 + environment.cell_width, y0 + environment.cell_height], fill='blue')
        
        frame_array = np.array(frame)

    out.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))

def add_plot_to_video(out, plot_filename):
    plot_img = cv2.imread(plot_filename)
    for _ in range(5):
        out.write(cv2.resize(plot_img, (640, 640)))


def display_environment(environment, title="Environment State"):
    fig, ax = plt.subplots(figsize=(6, 6))
    grid_image = np.ones((environment.grid_height, environment.grid_width, 3))

    # Draw obstacles
    for obs_pos in environment.obstacle_positions:
        x0, y0 = obs_pos[1] * environment.cell_width, obs_pos[0] * environment.cell_height
        x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
        grid_image[y0:y1, x0:x1] = [0.5, 0.5, 0.5]  # Gray color for obstacles

    dog_image_np = ensure_numpy_image(environment.dog_image)
    cat_image_np = ensure_numpy_image(environment.cat_image)
    robot_image_np = ensure_numpy_image(environment.robot_image)

    for dog_pos in environment.dog_positions:
        x0, y0 = dog_pos[1] * environment.cell_width, dog_pos[0] * environment.cell_height
        x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
        grid_image[y0:y1, x0:x1] = cv2.resize(dog_image_np, (environment.cell_width, environment.cell_height))

    for cat_pos in environment.cat_positions:
        if not environment.found_cats[environment.cat_positions.index(cat_pos)]:
            x0, y0 = cat_pos[1] * environment.cell_width, cat_pos[0] * environment.cell_height
            x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
            grid_image[y0:y1, x0:x1] = cv2.resize(cat_image_np, (environment.cell_width, environment.cell_height))

    x0, y0 = environment.position[1] * environment.cell_width, environment.position[0] * environment.cell_height
    x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
    grid_image[y0:y1, x0:x1] = cv2.resize(robot_image_np, (environment.cell_width, environment.cell_height))

    ax.imshow(grid_image, interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.axis('off')

    plt.show()
