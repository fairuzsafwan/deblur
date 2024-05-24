import zmq
import cv2
import numpy as np
import time
import os
import tensorflow as tf

# Define a class to deserialize data received via ZeroMQ
class OBC_TlmData:
    def __init__(self, data):
        # Unpack the received data into attributes based on the defined format
        (
            self.active_sensor_count, self.mission_mode, self.voltage_5v, self.voltage_3v,
            self.current_5v, self.current_3v, self.pi_temperature, self.board_temperature,
            self.uv_temperature, self.uv_a, self.uv_b, self.uv_c,
            self.mag_uT_x, self.mag_uT_y, self.mag_uT_z,
            self.gyro_dps_x, self.gyro_dps_y, self.gyro_dps_z,
            self.accel_ms2_x, self.accel_ms2_y, self.accel_ms2_z,
            self.gravity_x, self.gravity_y, self.gravity_z,
            self.quater_w, self.quater_x, self.quater_y, self.quater_z,
            self.al_lux,
            self.red_lux, self.green_lux, self.blue_lux, self.ir_lux,
            self.ss_lux, self.ss_temperature,
            self.timeepoch
        ) = struct.unpack('bbbbHHhhHHHfff' + 'fff' * 3 + 'fiiiiII', data)

# Set up ZeroMQ subscriber socket
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("ipc:///tmp/obc_ss")

# Set subscription filter to empty string (receive all messages)
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

# Lux range for filtering
lux_min = 10  # Define minimum lux threshold
lux_max = 100000  # Define maximum lux threshold

# Load inference model (assuming we have already trained the model)
model_path = "saved_model"
model = tf.keras.models.load_model(model_path)

# Function to read and preprocess images for inference
def read_inference_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0  # Normalize the images
    return img

# Function to perform inference on a single image
def inference(model_path, img, output_path, img_size, index):
    print("------------- Inference start -------------")
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the input image
    img_infer = read_inference_image(img)
    infer_input = np.expand_dims(img_infer, axis=0)  # Add batch dimension

    # Perform inference
    output_image = model.predict(infer_input)
    
    # Denormalize the output image
    output_image = output_image.squeeze() * 255.0  

    # Clip and convert image to uint8
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    # Save the output image with index
    output_image_path = os.path.join(output_path, f"inference_result_{index}.jpg")
    cv2.imwrite(output_image_path, output_image)

    # # Print confirmation message
    # print(f"Inference result {index} saved at {output_image_path}")
    # print("------------- Inference completed -------------")

if __name__ == "__main__":
    # Start ZeroMQ loop for inference

    # Initialize VideoCapture outside of the loop
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    index = 0

    while True:
        # Receive message from ZeroMQ
        message = subscriber.recv()

        # Deserialize the received data into OBC_TlmData object
        obc_data = OBC_TlmData(message)

        # Access the lux value
        lux_value = obc_data.al_lux

        # Capture frame from the camera
        ret, frame = cap.read()

        # Check if lux value is within range
        if lux_min <= lux_value <= lux_max:
            # Perform inference
            inference(model_path, frame, "output_path", (256, 256), index)
            index += 1  # Increment index

        # Sleep for a short duration to avoid high CPU usage
        time.sleep(0.1)

    # Release the camera capture object after the loop ends
    cap.release()
