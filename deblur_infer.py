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
def read_inference_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0  # Normalize the images
    return img

# Function to perform inference on a single image
def inference(interpreter, img, output_path, img_size, index):
    # Preprocess the input image
    img_infer = read_inference_image(img)
    img_infer_resized = cv2.resize(img_infer, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img_infer_expanded = np.expand_dims(img_infer_resized, axis=0)  # Add batch dimension

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

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the output image with index
    output_image_path = os.path.join(output_path, f"inference_result_{index}.jpg")
    cv2.imwrite(output_image_path, output_image)

    print(f"Inference result {index} saved at {output_image_path}")

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
    while True:
        # Receive message from ZeroMQ
        message = subscriber.recv()

        try:
            # Deserialize the received data into OBC_TlmData object
            obc_data = OBC_TlmData(message)
        except ValueError as e:
            print(f"Error: {e}")
            continue  # Skip processing this message

        # Access the lux value
        lux_value = obc_data.al_lux
        print(f"Value obtained: {lux_value}")

        # Perform inference or other processing as needed based on lux_value
        # For example, if lux_value is within a certain range, trigger inference
        #if lux_min <= lux_value <= lux_max:
        #    # Assume `img` is available for inference, you would need to load the actual image
        #    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)  # Placeholder for the actual image
        #    inference(interpreter, img, output_path, img_size, index)
        #    index += 1

        time.sleep(0.1)

