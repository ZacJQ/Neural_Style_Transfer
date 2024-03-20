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
def load_image(image_path:str, image_size=(256, 256), preserve_aspect_ratio=True)-> tf.Tensor:
    """Loads and preprocesses images."""
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def load_model(hub_handle:str)->tf.keras.Model:
    """Loads the given tensorflow model."""
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
    return hub_module

def stylize_image(hub_module:tf.keras.Model, content_image:tf.Tensor, style_image:tf.Tensor)-> np.ndarray:
    """Stylizes the model given the inputs."""
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()
    stylized_image = stylized_image[0]
    return stylized_image



def make_black(image_array: np.ndarray) -> np.ndarray:
    """A dummy fuction used for testing"""
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
  """Changes the saturation of an image.

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

def apply_effects(tensor: tf.Tensor, function, parameters)-> tf.Tensor:
    """Used to apply effects, given the effects function and its parameter"""
    image_array = tensor.numpy()
    image_array = image_array[0]
    effects_image_array = function(image_array, parameters)
    effects_image_tensor = tf.convert_to_tensor(effects_image_array)[tf.newaxis, ...]
    effects_image_tensor.set_shape(tensor.shape)
    effects_image_tensor = tf.cast(effects_image_tensor, tensor.dtype)
    return effects_image_tensor

def stylize(content_image: str,output_image_size: int, style_image: str, style_inp_size: int, gaus_blur_val: int, saturation:int , output_gausian:int, output_saturation: int)-> tf.Tensor:
  """Processes images and returns the stylized image."""

  content_img_size = (output_image_size, output_image_size)
  style_img_size = (style_inp_size, style_inp_size)  # Keep this fixed for best results

  content_image = load_image(content_image, content_img_size)
  style_image = load_image(style_image, style_img_size)
  style_image_blur = apply_effects(style_image,gaussian_blur,gaus_blur_val)
  style_image_effects = apply_effects(style_image_blur,change_saturation,saturation)


  stylized_image = stylize_image(hub_module, content_image, style_image_effects)
  temp = gaussian_blur(stylized_image,output_gausian)
  temp = change_saturation(temp, output_saturation)
  blur_out = temp


  return stylized_image , blur_out

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# Load the model once at startup
hub_module = load_model(hub_handle)


# Create Gradio interface
iface = gr.Interface(
  fn=stylize,
  inputs=[
    gr.Image(label="Content Image" , type="filepath"),
    gr.Slider(minimum=124, maximum=1920, value=512 ,label="Output Image Size"),
    gr.Image(label="Style Image", type="filepath"),
    gr.Slider(minimum=124, maximum=1920, value=256 ,label="Style Image Input Size"),
    gr.Slider(minimum=0, maximum=30, step=1, value=0.5, label="Gausian blur on style image"),
    gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Saturation on style image"),
    gr.Slider(minimum=0, maximum=30, step=1, value=0.5, label="Gausian blur on OUTPUT IMAGE"),
    gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Saturation on OUTPUT IMAGE"),
  ],
  outputs= [gr.Image(label="Stylized Image", type="numpy"),
            gr.Image(label="Edited Filter", type="numpy")],
  title="Image Stylization",
  live=False,
  submit_btn="Submit"
)

iface.launch()