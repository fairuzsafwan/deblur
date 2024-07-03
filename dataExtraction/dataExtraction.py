import zmq
import time
import json
import struct
import os

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

# Function to save sensor data to a file
def save_sensor_data(data, output_path, index):
    output_file = os.path.join(output_path, f"sensor_data_{index}.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Sensor data {index} saved at {output_file}")

if __name__ == "__main__":
    # Load configuration from JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Read configuration values
    ipc_address = config["ZeroMQ"]["ipc_address"]
    output_path = config["Data"]["output_path"]
    capture_interval = config.get("capture_interval", 10)  # Default to 10 seconds if not specified

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set up ZeroMQ subscriber socket
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(ipc_address)

    # Set subscription filter to empty string (receive all messages)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    index = 0
    try:
        while True:
            try:
                # Receive message from ZeroMQ
                message = subscriber.recv()

                # Deserialize the received data into OBC_TlmData object
                obc_data = OBC_TlmData(message)

                # Convert the OBC_TlmData object to a dictionary for easy JSON serialization
                sensor_data = {
                    "active_sensor_count": obc_data.active_sensor_count,
                    "mission_mode": obc_data.mission_mode,
                    "voltage_5v": obc_data.voltage_5v,
                    "voltage_3v": obc_data.voltage_3v,
                    "current_5v": obc_data.current_5v,
                    "current_3v": obc_data.current_3v,
                    "pi_temperature": obc_data.pi_temperature,
                    "board_temperature": obc_data.board_temperature,
                    "al_lux": obc_data.al_lux,
                    "red_lux": obc_data.red_lux,
                    "green_lux": obc_data.green_lux,
                    "blue_lux": obc_data.blue_lux,
                    "ir_lux": obc_data.ir_lux,
                    "mag_uT_x": obc_data.mag_uT_x,
                    "mag_uT_y": obc_data.mag_uT_y,
                    "mag_uT_z": obc_data.mag_uT_z,
                    "gyro_dps_x": obc_data.gyro_dps_x,
                    "gyro_dps_y": obc_data.gyro_dps_y,
                    "gyro_dps_z": obc_data.gyro_dps_z,
                    "accel_ms2_x": obc_data.accel_ms2_x,
                    "accel_ms2_y": obc_data.accel_ms2_y,
                    "accel_ms2_z": obc_data.accel_ms2_z,
                    "uv_a": obc_data.uv_a,
                    "uv_b": obc_data.uv_b,
                    "uv_c": obc_data.uv_c,
                    "uv_temp": obc_data.uv_temp,
                    "ss_lux": obc_data.ss_lux,
                    "ss_temperature": obc_data.ss_temperature,
                    "sc_voltage": obc_data.sc_voltage,
                    "sc_ckt_resistance": obc_data.sc_ckt_resistance,
                    "sc_current": obc_data.sc_current,
                    "sc_power": obc_data.sc_power,
                    "timeepoch": obc_data.timeepoch,
                }

                # Save the sensor data to a file
                save_sensor_data(sensor_data, output_path, index)
                index += 1

            except ValueError as e:
                print(f"Error: {e}")
                continue  # Skip processing this message

            time.sleep(capture_interval)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")
