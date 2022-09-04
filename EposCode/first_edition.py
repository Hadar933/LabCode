import serial
import time
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

"""
specs: 
max speed: 60,000 rpm
max continuous input speed: 40,000 rpm
number of pulses: 256 pulses/revolution -> encoder resolution = 256*4 = 1024 
 max acceleration: 100,000 rpm/s 
"""

path = "C:\Program Files (x86)\maxon motor ag\EPOS IDX\EPOS2\\" \
       "04 Programming\Windows DLL\Microsoft Visual C++\Example VC++/EposCmd64.dll"

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
timewait = 2000  # maxtime for position reach

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


def sin_position(sine_value):
    # Encoder Resolution 1024qc
    target_position = int(sine_value * 2000)
    print(f"{target_position}.")
    epos.VCS_MoveToPosition(keyHandle, nodeID, target_position, True, True, byref(pErrorCode))
    # MoveToPositionSpeed(target_position, 1000)
    # time.sleep(0.02)


def home_device():
    """
    moves to position 0
    """
    print("Homing to 0 inc...")
    epos.VCS_MoveToPosition(keyHandle, nodeID, 0, True, True, byref(pErrorCode))
    epos.VCS_WaitForTargetReached(keyHandle, nodeID, timewait, byref(pErrorCode))


if __name__ == "__main__":
    # ========= Initiating connection and setting motion profile ========= #

    # specify EPOS version and interface:
    keyHandle = epos.VCS_OpenDevice(b'EPOS2', b'MAXON SERIAL V2', b'USB', b'USB1', byref(pErrorCode))
    # set baudrate:
    epos.VCS_SetProtocolStackSettings(keyHandle, baudrate, timeout, byref(pErrorCode))
    # clear all faults:
    epos.VCS_ClearFault(keyHandle, nodeID, byref(pErrorCode))

    #  ======== activate profile position mode: ======= #
    epos.VCS_ActivateProfilePositionMode(keyHandle, nodeID, byref(pErrorCode))
    # enable device:
    epos.VCS_SetEnableState(keyHandle, nodeID, byref(pErrorCode))

    # =========== profile position test: =========== #
    # freq, num_samples, speed = 1, 1024, 5000  # Hz,_,rpm
    # x = np.linspace(0, 1, num=num_samples)
    # angle = 2 * np.pi * freq * x
    # sine_wave = np.sin(angle)
    # plt.plot(x, sine_wave), plt.title(f"f={freq}"), plt.show()

    # set up speed:
    print(GetPositionIs())
    home_device()
    print(GetPositionIs())
    #
    # print("Starting movement in 3 seconds...")
    # time.sleep(3)

    # for elem in sine_wave:
    #     print(f"{elem:.2f}, ", end="")
    #     sin_position(elem)

    # activate interpolated position mode:
    # epos.VCS_ActivateInterpolatedPositionMode(keyHandle, nodeID, byref(pErrorCode))
    # pUnderflowWarningLimit, pOverflowWarningLimit = c_uint(), c_uint()
    # epos.VCS_SetIpmBufferParameter(keyHandle, nodeID, pUnderflowWarningLimit, pOverflowWarningLimit, byref(pErrorCode))
    # pMaxBufferSize = c_uint(0)
    # epos.VCS_GetIpmBufferParameter(keyHandle, nodeID, byref(pUnderflowWarningLimit), byref(pOverflowWarningLimit),
    #                                byref(pMaxBufferSize), byref(pErrorCode))
    # x = 2
    # re initializing buffer and printing init size
    # buff_size = c_uint()
    # epos.VCS_ClearIpmBuffer(keyHandle, nodeID, byref(pErrorCode))
    # epos.VCS_GetFreeIpmBufferSize(keyHandle, nodeID, byref(buff_size), byref(pErrorCode))
    #

    # epos.VCS_AddPvtValueToIpmBuffer(keyHandle, nodeID, 500, 100, 10)
    # epos.VCS_AddPvtValueToIpmBuffer(keyHandle, nodeID, 1000, 100, 10)
    # epos.VCS_AddPvtValueToIpmBuffer(keyHandle, nodeID, 500, 100, 10)
    # epos.VCS_GetFreeIpmBufferSize(keyHandle, nodeID, byref(buff_size), byref(pErrorCode))

    close_device()
