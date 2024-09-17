import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

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

def add_frame_to_video(out, environment, episode):
    fig, ax = plt.subplots(figsize=(6, 6))
    grid_image = np.ones((environment.grid_height, environment.grid_width, 3))

    for dog_pos in environment.dog_positions:
        x0, y0 = dog_pos[1] * environment.cell_width, dog_pos[0] * environment.cell_height
        x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
        grid_image[y0:y1, x0:x1] = environment.dog_image

    for cat_pos in environment.cat_positions:
        if not environment.found_cats[environment.cat_positions.index(cat_pos)]:
            x0, y0 = cat_pos[1] * environment.cell_width, cat_pos[0] * environment.cell_height
            x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
            grid_image[y0:y1, x0:x1] = environment.cat_image

    x0, y0 = environment.position[1] * environment.cell_width, environment.position[0] * environment.cell_height
    x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
    grid_image[y0:y1, x0:x1] = environment.robot_image

    ax.imshow(grid_image, interpolation='nearest', aspect='auto')
    ax.set_title(f"Episode {episode + 1}")
    ax.axis('off')

    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_resized = cv2.resize(img, (640, 640))

    out.write(img_resized)
    plt.close(fig)

def add_plot_to_video(out, plot_filename):
    plot_img = cv2.imread(plot_filename)
    for _ in range(5):
        out.write(cv2.resize(plot_img, (640, 640)))

def display_environment(environment, title="Environment State"):
    fig, ax = plt.subplots(figsize=(6, 6))
    grid_image = np.ones((environment.grid_height, environment.grid_width, 3))

    for dog_pos in environment.dog_positions:
        x0, y0 = dog_pos[1] * environment.cell_width, dog_pos[0] * environment.cell_height
        x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
        grid_image[y0:y1, x0:x1] = environment.dog_image

    for cat_pos in environment.cat_positions:
        if not environment.found_cats[environment.cat_positions.index(cat_pos)]:
            x0, y0 = cat_pos[1] * environment.cell_width, cat_pos[0] * environment.cell_height
            x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
            grid_image[y0:y1, x0:x1] = environment.cat_image

    x0, y0 = environment.position[1] * environment.cell_width, environment.position[0] * environment.cell_height
    x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
    grid_image[y0:y1, x0:x1] = environment.robot_image

    ax.imshow(grid_image, interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.axis('off')

    plt.show()
