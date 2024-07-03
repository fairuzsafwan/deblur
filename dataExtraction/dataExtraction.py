import zmq
import time
import json
import struct
import os
import csv
from datetime import datetime

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

# Function to save sensor data to a CSV file
def save_sensor_data(data, output_file):
    # CSV header (change as per your data structure)
    fieldnames = [
        "active_sensor_count", "mission_mode", "voltage_5v", "voltage_3v",
        "current_5v", "current_3v", "pi_temperature", "board_temperature",
        "al_lux", "red_lux", "green_lux", "blue_lux", "ir_lux",
        "mag_uT_x", "mag_uT_y", "mag_uT_z",
        "gyro_dps_x", "gyro_dps_y", "gyro_dps_z",
        "accel_ms2_x", "accel_ms2_y", "accel_ms2_z",
        "uv_a", "uv_b", "uv_c", "uv_temp",
        "ss_lux", "ss_temperature",
        "sc_voltage", "sc_ckt_resistance",
        "sc_current", "sc_power",
        "timeepoch",
        "current_time"  # Additional column for current time
    ]

    # Get current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Format timestamp as per your preference

    # Write data to CSV file
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow({
            "active_sensor_count": data.active_sensor_count,
            "mission_mode": data.mission_mode,
            "voltage_5v": data.voltage_5v,
            "voltage_3v": data.voltage_3v,
            "current_5v": data.current_5v,
            "current_3v": data.current_3v,
            "pi_temperature": data.pi_temperature,
            "board_temperature": data.board_temperature,
            "al_lux": data.al_lux,
            "red_lux": data.red_lux,
            "green_lux": data.green_lux,
            "blue_lux": data.blue_lux,
            "ir_lux": data.ir_lux,
            "mag_uT_x": data.mag_uT_x,
            "mag_uT_y": data.mag_uT_y,
            "mag_uT_z": data.mag_uT_z,
            "gyro_dps_x": data.gyro_dps_x,
            "gyro_dps_y": data.gyro_dps_y,
            "gyro_dps_z": data.gyro_dps_z,
            "accel_ms2_x": data.accel_ms2_x,
            "accel_ms2_y": data.accel_ms2_y,
            "accel_ms2_z": data.accel_ms2_z,
            "uv_a": data.uv_a,
            "uv_b": data.uv_b,
            "uv_c": data.uv_c,
            "uv_temp": data.uv_temp,
            "ss_lux": data.ss_lux,
            "ss_temperature": data.ss_temperature,
            "sc_voltage": data.sc_voltage,
            "sc_ckt_resistance": data.sc_ckt_resistance,
            "sc_current": data.sc_current,
            "sc_power": data.sc_power,
            "timeepoch": data.timeepoch,
            "current_time": current_time
        })

    print(f"Sensor data appended to {output_file} at {current_time}")

if __name__ == "__main__":
    # Load configuration from JSON file
    with open('dataConfig.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Read configuration values
    ipc_address = config["ZeroMQ"]["ipc_address"]
    output_file = os.path.join(config["Data"]["output_path"], "sensor_data.csv")
    capture_interval = config.get("capture_interval", 10)  # Default to 10 seconds if not specified

    # Create output directory if it does not exist
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Set up ZeroMQ subscriber socket
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(ipc_address)

    # Set subscription filter to empty string (receive all messages)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    try:
        while True:
            try:
                # Receive message from ZeroMQ
                message = subscriber.recv()

                # Deserialize the received data into OBC_TlmData object
                obc_data = OBC_TlmData(message)

                # Save the sensor data to a CSV file
                save_sensor_data(obc_data, output_file)

            except ValueError as e:
                print(f"Error: {e}")
                continue  # Skip processing this message

            time.sleep(capture_interval)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

