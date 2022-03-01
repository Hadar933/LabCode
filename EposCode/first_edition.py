"""
specs: 
max speed: 60,000 rpm
max continuous input speed: 40,000 rpm
number of pulses: 256 pulses/revolution 
 max acceleration: 100,000 rpm/s
"""

path = "C:\Program Files (x86)\maxon motor ag\EPOS IDX\EPOS4\\04 Programming\Windows DLL\Microsoft Visual C++\Example VC++/EposCmd64.dll"
import serial
import time
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

# EPOS Command Library path
# Load library
cdll.LoadLibrary(path)
epos = CDLL(path)

# Defining return variables from Library Functions
ret = 0
pErrorCode = c_uint()
pDeviceErrorCode = c_uint()

# Defining a variable NodeID and configuring connection
nodeID = 1
baudrate = 1000000
timeout = 500

# Configure desired motion profile
acceleration = 10000  # rpm/s, up to 1e7 would be possible
deceleration = 10000  # rpm/s


# Query motor position
def GetPositionIs():
    pPositionIs = c_long()
    pErrorCode = c_uint()
    ret = epos.VCS_GetPositionIs(keyHandle, nodeID, byref(pPositionIs), byref(pErrorCode))
    return pPositionIs.value  # motor steps


# Move to position at speed
def MoveToPositionSpeed(target_position, target_speed):
    epos.VCS_SetPositionProfile(keyHandle, nodeID, target_speed, acceleration, deceleration,
                                byref(pErrorCode))  # set profile parameters
    while True:
        if target_speed != 0:
            epos.VCS_MoveToPosition(keyHandle, nodeID, target_position, True, True,
                                    byref(pErrorCode))  # move to position
        elif target_speed == 0:
            epos.VCS_HaltPositionMovement(keyHandle, nodeID, byref(pErrorCode))  # halt motor
        true_position = GetPositionIs()
        if true_position == target_position:
            break


def close_device():
    """
    disables the state and closes the device
    """
    print("Closing Device")
    epos.VCS_SetDisableState(keyHandle, nodeID, byref(pErrorCode))  # disable device
    epos.VCS_CloseDevice(keyHandle, byref(pErrorCode))  # close device


def move_device(position, velocity, return_to_zero=True):
    """
    moves the motor to the given position with the given velocity
    :param position:
    :param velocity:
    :param return_to_zero: true if you'd wish the motor would return to zero at the given velocity
    """
    MoveToPositionSpeed(position, velocity)  # move to position 2000 steps at 10000 rpm/s
    print(f'Motor position: {GetPositionIs()}')
    time.sleep(1)
    if return_to_zero:
        MoveToPositionSpeed(0, velocity)  # move to position 0 steps at 2000 rpm/s
        print(f'Motor position: {GetPositionIs()}')
        time.sleep(1)


def sin_Position(target_rad):
    # Encoder Resolution 2000qc
    target_position = int(target_rad * 2000)
    print(f"{target_position}.")
    epos.VCS_MoveToPosition(keyHandle, nodeID, target_position, True, True, byref(pErrorCode))
    # MoveToPositionSpeed(target_position, 1000)
    # time.sleep(0.02)


if __name__ == "__main__":
    # Initiating connection and setting motion profile
    keyHandle = epos.VCS_OpenDevice(b'EPOS4', b'MAXON SERIAL V2', b'USB', b'USB0',
                                    byref(pErrorCode))  # specify EPOS version and interface
    epos.VCS_SetProtocolStackSettings(keyHandle, baudrate, timeout, byref(pErrorCode))  # set baudrate
    epos.VCS_ClearFault(keyHandle, nodeID, byref(pErrorCode))  # clear all faults
    epos.VCS_ActivateProfilePositionMode(keyHandle, nodeID, byref(pErrorCode))  # activate profile position mode
    epos.VCS_SetEnableState(keyHandle, nodeID, byref(pErrorCode))  # enable device

    freq = 0.1
    sine_wave = np.sin(2 * np.pi * freq * np.linspace(0, 10, num=500))
    print("Homing to 0 inc...")
    MoveToPositionSpeed(0, 1000)
    print("Sin...")
    for radian in sine_wave:
        print(f"{radian:.2f} rad, ", end="")
        sin_Position(radian)
    print("Closing...")
    close_device()
