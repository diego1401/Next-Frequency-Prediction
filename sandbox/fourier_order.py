from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, hsv_to_rgb
import cv2
from tqdm import tqdm
import numpy as np

def load_image():
    # We load the image
    image_path = 'example_image.jpg'
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if needed
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts to tensor and scales values to [0,1]
    ])

    # Apply the transformation
    tensor_image = transform(image)
    return tensor_image

def save_tensor_as_image(tensor,filename):
    # Convert from (C, H, W) to (H, W, C) for matplotlib
    # Plot the image
    plt.imshow(tensor.clamp(0.0,1.0).permute(1,2,0))

    # Remove axes, ticks, and borders
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the image
    plt.savefig(f'{filename}.png')

def get_order_of_frequencies(h,w):
    center_y, center_x = h // 2, w // 2
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    distances = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    distances = distances.flatten()

    # Step 3: Get sorted indices based on distance
    sorted_indices = torch.argsort(distances)
    return sorted_indices

def get_shifted_fourier(tensor_image):
    fourier_image = torch.fft.fft2(tensor_image)
    shifted_fourier_image = torch.fft.fftshift(fourier_image)
    return shifted_fourier_image

def get_reconstructed_image(clipped_shifted_fourier_image):
    clipped_fourier_image = torch.fft.ifftshift(clipped_shifted_fourier_image)
    reconstructed_image = torch.fft.ifft2(clipped_fourier_image).real
    
    return reconstructed_image

def clip_fourier_space(shifted_fourier,index_order,clip_percentage=80):
    # Step: Determine how many frequencies to clip
    num_to_clip = int(clip_percentage / 100 * len(index_order))

    # Create a flattened Fourier array and zero out the high frequencies
    # flat_shifted = shifted_fourier.view(shifted_fourier.shape[0],-1)
    # flat_shifted[:,index_order[-num_to_clip:]] = 0  # Zero out the farthest frequencies
    shifted_fourier.view(shifted_fourier.shape[0],-1)[:,index_order[-num_to_clip:]] = 0

    # Step: Reshape the flattened array back to 2D
    # reshaped_shifted = flat_shifted.view_as(shifted_fourier)
    return shifted_fourier

def clip_fourier_space_figure():
    tensor_image = load_image()
    
    # Put image into fourier space
    shifted_fourier_image = get_shifted_fourier(tensor_image)
    save_tensor_as_image(shifted_fourier_image,'shifted')

    # Put fourier space into a 1D array
    # Step 2: Compute distances from the center
    _, h, w = shifted_fourier_image.shape
    sorted_indices = get_order_of_frequencies(h,w)

    save_tensor_as_image(clip_fourier_space(shifted_fourier_image,sorted_indices),'clipped_shifted')

def smooth_image_figure():
    tensor_image = load_image()
    save_tensor_as_image(tensor_image,'sanity')
    # Put image into fourier space
    shifted_fourier_image = get_shifted_fourier(tensor_image)

    # Put fourier space into a 1D array
    # Step 2: Compute distances from the center
    _, h, w = shifted_fourier_image.shape
    sorted_indices = get_order_of_frequencies(h,w)

    clipped_shifted_fourier_image = clip_fourier_space(shifted_fourier_image,sorted_indices,clip_percentage=90)
    save_tensor_as_image(clipped_shifted_fourier_image.real,'clipped_shifted')

    print('clipped_shifted_fourier_image',clipped_shifted_fourier_image.shape)
    clipped_fourier_image = torch.fft.ifftshift(clipped_shifted_fourier_image)
    reconstructed_image = torch.fft.ifft2(clipped_fourier_image).real
    print('reconstructed shape',reconstructed_image.shape)
    save_tensor_as_image(reconstructed_image,'reconstructed_image')

def smooth_video():
    tensor_image = load_image()
    save_tensor_as_image(tensor_image,'sanity')
    # Put image into fourier space
    shifted_fourier_image = get_shifted_fourier(tensor_image)

    _, h, w = shifted_fourier_image.shape
    sorted_indices = get_order_of_frequencies(h,w)

    frames = []
    for percentage in tqdm(range(100)):
        clipped_shifted_fourier_image = clip_fourier_space(shifted_fourier_image.clone(),sorted_indices,clip_percentage=percentage)
        frame = get_reconstructed_image(clipped_shifted_fourier_image)
        frame = frame.clamp(0.0,1.0).permute(1, 2, 0).numpy()

        # frame = np.random.rand(h,w,3)
        frame = (frame * 255).astype('uint8')  # Scale to [0, 255] range
        frames.append(frame)
        

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (for mp4)
    fps = 5  # Frames per second
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (h, w))

    # Write each frame to the video
    for frame in frames:
        out.write(frame)

    # Release the video writer object
    out.release()

def separate_fourier_space_in_K(K=3):
    tensor_image = load_image()
    
    # Put image into fourier space
    shifted_fourier_image = get_shifted_fourier(tensor_image)

    # Put fourier space into a 1D array
    # Step : Compute distances from the center
    _, h, w = shifted_fourier_image.shape
    sorted_indices = get_order_of_frequencies(h,w)
    number_of_frequencies = len(sorted_indices)
    group_size = number_of_frequencies // K + 1

    groups = [sorted_indices[i * group_size: (i + 1) * group_size] for i in range(K)]
    print([len(i) for i in groups])
    # Step 4: Create a color gradient
    hue_values = torch.linspace(0.3, 1.0, K)  # Hue gradient from 0 to 1
    hsv_image = torch.zeros((h, w, 3))  # HSV color space

    for i, group in enumerate(groups):
        # Assign hue to the current group
        hsv_image.view(-1,3)[group,:] = hue_values[i]

    # Convert HSV to RGB for visualization
    # rgb_image = hsv_to_rgb(hsv_image.numpy())
    rgb_image = hsv_to_rgb(hsv_image.numpy())
    # Step 5: Plot the Fourier space with coloring
    plt.figure(figsize=(8, 8))
    plt.title("Fourier Space with Gradient Coloring")
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.savefig('groups.png')

def low_to_high_frequency(h,w):
    sorted_indices = get_order_of_frequencies(h,w)
    
    # Step 4: Create a color gradient
    hue_values = torch.linspace(0, 1.0, len(sorted_indices))  # Hue gradient from 0 to 1
    hsv_image = torch.zeros((h, w, 3))  # HSV color space

    hsv_image.view(-1,3)[sorted_indices,:] = hue_values.view(-1,1).repeat(1,3)

    # Convert HSV to RGB for visualization
    # rgb_image = hsv_image.numpy()
    rgb_image = hsv_to_rgb(hsv_image.numpy())
    # Step 5: Plot the Fourier space with coloring
    plt.figure(figsize=(8, 8))
    plt.title("Low_to_high_frequencies.png")
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.savefig('low_to_high.png')
    

if __name__ == '__main__':

    # separate_fourier_space_in_K(K=29)
    # low_to_high_frequency(8,8)
    smooth_video()