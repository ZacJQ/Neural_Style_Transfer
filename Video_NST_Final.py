import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

def crop_center(image: tf.Tensor) -> tf.Tensor:
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

# @functools.lru_cache(maxsize=None)
def load_image(image_path:str, content_img_size=(256, 256), preserve_aspect_ratio=True)-> tf.Tensor:
    """Loads and preprocesses images."""
    if type(image_path) == "<class 'numpy.ndarray'>":
            img = tf.convert_to_tensor(image_path)[tf.newaxis, ...]
    else:
            img = tf.io.decode_image(
                tf.io.read_file(image_path),
                channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, content_img_size, preserve_aspect_ratio=True)
    return img


def load_model(hub_handle:str)->tf.keras.Model:
    """Function to load the model"""
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
    return hub_module

def stylize_image(hub_module:tf.keras.Model, content_image:tf.Tensor, style_image:tf.Tensor)-> np.ndarray:
    """Performs the task of stylizing the input based on the given model , content image and style image"""
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()
    stylized_image = stylized_image[0]
    return stylized_image

def read_video_frames_old(video_path):
    """Do not use this function (old version)"""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return
    frames_list = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB format
        frame_np = frame_np.astype(np.float32)
        frames_list.append(frame_np)
    video_capture.release()    
    frames_list.pop(-1)
    return frames_list

def read_video_frames(video_path: str) -> list:
    """
    Given video path reads the frames as np.ndarray as stores it in a list. Returns a list of np.ndarray
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return
    frames_list = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB format
        frame_np = frame.astype(np.float32)
        frame_np /= 255
        frames_list.append(frame_np)
    video_capture.release()
    
    return frames_list

def load_video_old(video_path):
    """
    Generator function to load video frames. Used during testing.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def save_video_old(frames, output_path, fps=20, size=None):
    """
    Saves processed frames as a video. Used during testing.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()

def make_black(image_array: np.ndarray) -> np.ndarray:
    """
    Dummy fucntion that turns the output image black
    """
    image_array = image_array.astype(np.float32)
    image_array = image_array * 0
    return image_array

def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
    - image: Input image
    - kernel_size: Size of the Gaussian kernel (tuple of two integers)
    - sigma: Standard deviation of the Gaussian kernel
    
    Returns:
    - Blurred image
    """
    kernel_size=(3,3)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def change_saturation(image: np.ndarray, saturation_value: float) -> np.ndarray:
    """
    Changes the saturation of an image.
    
    Args:
      image: The input image.
      saturation_value: The saturation value, where 0 is no saturation and 255 is full saturation.

    Returns:
      The output image with the changed saturation.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s = cv2.addWeighted(s, saturation_value, 0, 0, 0)
    hsv[:, :, 1] = s
    output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return output

def apply_effects(tensor: tf.Tensor, function: any, parameters: any)-> tf.Tensor:
    """
    Applies effects given the parameter
    """
    image_array = tensor.numpy()
    image_array = image_array[0]
    effects_image_array = function(image_array, parameters)
    effects_image_tensor = tf.convert_to_tensor(effects_image_array)[tf.newaxis, ...]
    effects_image_tensor.set_shape(tensor.shape)
    effects_image_tensor = tf.cast(effects_image_tensor, tensor.dtype)
    return effects_image_tensor


def images_to_video(image_list, video_path, fps=24):
  """
  This function converts a list of NumPy ndarrays of images into a video.

  Args:
      image_list: A list of NumPy ndarrays representing images.
      video_path: The path to save the output video.
      fps: The desired frame rate of the video (default: 24).
  """
  height, width, channels = image_list[0].shape

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

  for image in image_list:
    # OpenCV uses BGR format, so convert if necessary
    if image.shape[-1] == 4:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out.write(image)

  out.release()



def stylize(content_image: str,output_image_size: int, output_name: str , style_image: str, style_inp_size: int, gaus_blur_val: int, saturation: int)-> tf.Tensor:
    """Processes images and returns the stylized image. Main function used to connect with gradio"""
    output_path = output_name + ".mp4"
    content_img_size = (output_image_size, output_image_size)
    style_img_size = (style_inp_size, style_inp_size)  # Keep this fixed for best results
    style_image = load_image(style_image, style_img_size)
    style_image_blur = apply_effects(style_image,gaussian_blur,gaus_blur_val)
    style_image_effects = apply_effects(style_image_blur,change_saturation,saturation)
    video = []

    content_image_frames = read_video_frames(content_image)
    for frame in content_image_frames:
        frame_tf = tf.convert_to_tensor(frame)[tf.newaxis, ...]
        frame_tf = crop_center(frame_tf)
        frame_tf = tf.image.resize(frame_tf, content_img_size, preserve_aspect_ratio=True)
        stylized_image = stylize_image(hub_module, frame_tf, style_image_effects)
        video.append(stylized_image) 
    # only for images 
    # temp = gaussian_blur(stylized_image,output_gausian)
    # temp = change_saturation(temp, output_saturation)
    # blur_out = temp
    frame_height, frame_width = video[0].shape[:2]
    new_video = []
    for img in video:
        img *= 255
        img = np.clip(img, 0 , 255)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        new_video.append(img)

    output_video = images_to_video(new_video, output_path, fps=30)
    return output_path

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# Load the model once at startup
hub_module = load_model(hub_handle)



iface = gr.Interface(
  fn=stylize,
  inputs=[
    gr.Video(label="Content Video that has to be stylized"),
    gr.Slider(minimum=124, maximum=1920, value=512 ,label="Output Video Size"),
    gr.Textbox(label="File output name", value="New_video"),
    gr.Image(label="Style Image (Image to copy style of)", type="filepath"),
    gr.Slider(minimum=124, maximum=1920, value=256 ,label="Style Image Input Size"),
    gr.Slider(minimum=0, maximum=30, step=1, value=0.5, label="Gausian blur on style image"),
    gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Saturation on style image")
  ],
  outputs= [gr.Video(label="Stylized Image")],
  title="Image Stylization",
  live=False,
  submit_btn="Submit"
)

iface.launch()