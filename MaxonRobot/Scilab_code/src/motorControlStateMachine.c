#include "motorControlStateMachine.h"

e_motorControlStates motorControlState[NUM_OF_MOTORS];

extern SMotorsData motors[NUM_OF_MOTORS];

//static int32_t count;

void handleMotors(void)
{
	for(EMotor motor = right; motor < endOfMotors; motor++)
	{
	    handleMotor(motor);
	}
	return;
}



void handleMotor(EMotor motor)
{
    counter_read_cntr(motor);
    calcSpeedFromEncoder(motor);
//    calcSpeedFromHulls(motor); // TODO need to change to speed from encoder
    setMotorDriveState(motor);
    setMotorControlState(motor);
    motors[motor].isRunning |= motorControlSatetExct(motor);
#ifdef DEBUG_SPEED_CONTROL
      record_motor_data(motor);
#endif
  return;
}



void setMotorControlState(EMotor motor)
{
	// this function handle and set the needed state machine for the motor control
//	static int32_t startRuningHallCnt[NUM_OF_MOTORS];

	if (IS_ZERO_FLOAT(motors[motor].speedControler.refSpeed))
	{
		motorControlState[motor] = MCS_HALT;
	}

	switch (motorControlState[motor])
	{
	case MCS_HALT:
		if (!IS_ZERO_FLOAT(motors[motor].speedControler.refSpeed))
    {
			motorControlState[motor] = MCS_START_RUNING;
		}
		break;

	case MCS_START_RUNING:
	  if(!IS_ZERO_FLOAT(motors[motor].speedControler.currentSpeed))
    {
			motorControlState[motor] = MCS_RUNING;
    }
		break;

	case MCS_RUNING:
	  if (IS_ZERO_FLOAT(motors[motor].speedControler.currentSpeed))
	  {
	    motorControlState[motor] = MCS_START_RUNING;
	  }
		break;


	case  MCS_MOTOR_FATAL_ERROR:
		motorControlState[motor] = MCS_HALT;
		resetMotorData(motor);
		break;
	}
}


bool motorControlSatetExct(EMotor motor)
{
	bool motorRunning = false;

	// this function execute the need state operation
	switch(motorControlState[motor]) {

	case MCS_HALT  :
		resetMotorData(motor);
		speedControlHandle(motor);
		calcPWMpercent(motor);
		sendCommandToDriver(motor);
		break;

	case MCS_START_RUNING :
		getMotorHulls(motor);
		motorPhaseConfigurationHandle(motor);
//		getHullSequence(motor);
		speedControlHandle(motor);
		setMotorDriveState(motor);
		calcPWMpercent(motor);
		sendCommandToDriver(motor);
		motorRunning = true;
		break;

	case MCS_RUNING :
		motorRunning = true;
		speedControlHandle(motor);
    calcPWMpercent(motor);
    sendCommandToDriver(motor);
		break;


	default : /* Optional */
		break;
	}
	return motorRunning; // calculate Vlim
}


void calcMotorPWMCommand(EMotor motor)
{
  if (motors[motor].motorDriveState != DS_STOP)
  {
    calcPWMpercent(motor);
  }
  else
  {
    motors[motor].PWMCommand = 0.0;
  }
}
