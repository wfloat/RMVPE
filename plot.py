import matplotlib.pyplot as plt
import numpy as np
import time
import pygame

def read_file_to_numpy_array(file_path):
    with open(file_path, 'r') as file:
        array = np.array([float(line.strip()) for line in file])
    return array

# Replace this with read_file_to_numpy_array("./tensor.txt") for actual data
frequency_data = read_file_to_numpy_array("./tensor.txt") # Simulated data for testing

video_duration = 15  # Replace with actual video duration in seconds
interval = video_duration / len(frequency_data)

# Initialize Pygame mixer
pygame.mixer.init()
# Load and play the audio file (replace 'your_audio_file.mp3' with your audio file)
pygame.mixer.music.load('in.wav')
pygame.mixer.music.play()

plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
x = np.arange(0, len(frequency_data), 1)
line, = ax.plot(x, np.zeros_like(frequency_data))  # Start with an empty plot

# Set y-axis limits
ax.set_ylim(np.min(frequency_data), np.max(frequency_data))

for i in range(1, len(frequency_data)):
    # Gradually reveal the data
    line.set_ydata(np.concatenate([frequency_data[:i], np.zeros(len(frequency_data) - i)]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(interval/2)

plt.ioff()  # Interactive mode off
plt.show()

# Stop the music after the plot is done
pygame.mixer.music.stop()
