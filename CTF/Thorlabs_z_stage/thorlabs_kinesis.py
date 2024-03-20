"""
lts_pythonnet
==================

An example of using the LTS integrated stages with python via pythonnet
"""
import os
import time
import sys
import clr

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.IntegratedStepperMotorsCLI.dll")
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
from System import Decimal  # necessary for real world units


class ThorlabsZStage:
    def __init__(self, device_sn='49337854', max_velocity=10):
        DeviceManagerCLI.BuildDeviceList()

        # Connect, begin polling, and enable
        device = LabJack.CreateLabJack(device_sn)
        self.device = device
        device.Connect(device_sn)

        # Ensure that the device settings have been initialized
        if not device.IsSettingsInitialized():
            device.WaitForSettingsInitialized(10000)  # 10 second timeout
            assert device.IsSettingsInitialized() is True

        # Start polling and enable
        device.StartPolling(250)  # 250ms polling rate
        # time.sleep(25)
        device.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable

        # Get Device Information and display description
        device_info = device.GetDeviceInfo()
        print(device_info.Description)

        # Load any configuration settings needed by the controller/stage
        motor_config = device.LoadMotorConfiguration(device_sn)

        # Get Velocity Params
        vel_params = device.GetVelocityParams()
        vel_params.MaxVelocity = Decimal(max_velocity)  # This is a bad idea
        device.SetVelocityParams(vel_params)

    def move_to(self, z, timeout=6000, rel=False):
        if rel:
            curr_pos = self.get_position()
            z = z + curr_pos
        # Move the device to a new position
        new_pos = Decimal(z)  # Must be a .NET decimal
        print(f'Moving to {new_pos}')
        self.device.MoveTo(new_pos, timeout)  # 60 second timeout
        print("Done")
    def get_position(self):
        return float(str(self.device.Position))

    def get_velocity(self):
        return self.stage.velocity

    def get_status(self):
        return self.stage.status

    def close(self):
        # Stop Polling and Disconnect
        self.device.StopPolling()
        self.device.Disconnect()

def main():
    """The main entry point for the application"""
    stage = ThorlabsZStage()
    # stage.move_to(0)
    # stage.move_to(2, rel=True)
    stage.move_to(-2, rel=True)
    stage.move_to(1, rel=True)
    stage.move_to(1, rel=True)
    stage.move_to(-1, rel=True)
    stage.move_to(-1, rel=True)
    print(stage.device.Position)




if __name__ == "__main__":
    main()