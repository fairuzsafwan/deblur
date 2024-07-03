import zmq
import cv2
import numpy as np
import time
import os
import struct
import tensorflow as tf
import json

# Define a class to deserialize data received via ZeroMQ
class OBC_TlmData:
    def __init__(self, data):
        unpack_format = 'b'*4 + 'H'*2 + 'h'*2 + 'f' + 'i'*4 + 'f'*13 + 'Ii' + 'b'*2 + 'H'*3 + 'I'
        expected_size = struct.calcsize(unpack_format)
        
        if len(data) != expected_size:
            print(f"Received data size: {len(data)}, expected: {expected_size}")
            raise ValueError(f"Expected data of size {expected_size} bytes, but received {len(data)} bytes.")
        
        (
            self.active_sensor_count, self.mission_mode, self.voltage_5v, self.voltage_3v,
            self.current_5v, self.current_3v, self.pi_temperature, self.board_temperature,
            self.al_lux,
            self.red_lux, self.green_lux, self.blue_lux, self.ir_lux,
            self.mag_uT_x, self.mag_uT_y, self.mag_uT_z,
            self.gyro_dps_x, self.gyro_dps_y, self.gyro_dps_z,
            self.accel_ms2_x, self.accel_ms2_y, self.accel_ms2_z,
            self.uv_a, self.uv_b, self.uv_c, self.uv_temp,
            self.ss_lux, self.ss_temperature,
            self.sc_voltage, self.padding, self.sc_ckt_resistance,
            self.sc_current, self.sc_power,
            self.timeepoch
        ) = struct.unpack(unpack_format, data)

# Function to read and preprocess images for inference
def read_inference_image(image, img_size=(256, 256), normalize=False):
    try:
        if image is None:
            raise ValueError(f"Error: Failed to load image")

        # Calculate aspect ratio
        h, w, _ = image.shape
        target_h, target_w = img_size

        if h > w:
            # Crop height
            start_h = (h - w) // 2
            image = image[start_h:start_h + w, :, :]
        else:
            # Crop width
            start_w = (w - h) // 2
            image = image[:, start_w:start_w + h, :]

        # Resize to target size
        image = cv2.resize(image, (target_w, target_h))

        if normalize:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = image.astype(np.float32) / 255.0  # Normalize the images

        return image
    except Exception as e:
        print(f"Exception in reading image: {e}")
        return None

# Function to perform inference on a single image
def inference(interpreter, img, output_path, img_size, index):
    # Save the cropped original image before inference
    original_image_path = os.path.join(output_path, "originalCropped")
    if not os.path.exists(original_image_path):
        os.makedirs(original_image_path)

    cv2.imwrite(os.path.join(original_image_path, f"original_image_{index}.jpg"), img)

    # Preprocess the input image for inference
    img_infer = read_inference_image(img, img_size, normalize=True)
    img_infer_expanded = np.expand_dims(img_infer, axis=0)  # Add batch dimension

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_infer_expanded)

    # Perform inference
    interpreter.invoke()

    # Get the output tensor
    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output tensor if needed
    # For example, convert to RGB and save the image
    output_image = np.squeeze(output_tensor)  # Assuming single output tensor
    output_image = (output_image * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Create output directory for inference results if it does not exist
    result_image_path = os.path.join(output_path, "inferenceResult")
    if not os.path.exists(result_image_path):
        os.makedirs(result_image_path)

    # Save the output image with index
    output_image_path = os.path.join(result_image_path, f"inference_result_{index}.jpg")
    cv2.imwrite(output_image_path, output_image)

    print(f"Inference result {index} saved at {output_image_path}")

# Function to capture an image using OpenCV
def capture_image(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_index}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture image")
        return None

    return frame

if __name__ == "__main__":
    # Load configuration from JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Read configuration values
    ipc_address = config["ZeroMQ"]["ipc_address"]
    lux_min = config["LUX"]["lux_min"]
    lux_max = config["LUX"]["lux_max"]
    model_path = config["Model"]["model_path"]
    output_path = config["Inference"]["output_path"]
    img_size = config["Inference"]["img_size"]
    capture_interval = config.get("capture_interval", 10)  # Default to 10 seconds if not specified
    camera_port = config.get("camera_port", 0)

    # Set up ZeroMQ subscriber socket
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(ipc_address)

    # Set subscription filter to empty string (receive all messages)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    index = 0
    try:
        while True:
            try:
                # Receive message from ZeroMQ
                message = subscriber.recv()

                # Deserialize the received data into OBC_TlmData object
                obc_data = OBC_TlmData(message)

                # Access the lux value
                lux_value = obc_data.al_lux
                
                print(f"Value obtained: {lux_value}")

                # Perform inference or other processing as needed based on lux_value
                if lux_min <= lux_value <= lux_max:
                    # Capture an image
                    frame = capture_image(camera_port)
                    if frame is None:
                        continue
                    # Perform inference and save the result
                    inference(interpreter, frame, output_path, img_size, index)
                    index += 1

            except ValueError as e:
                print(f"Error: {e}")
                continue  # Skip processing this message

            time.sleep(capture_interval)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

