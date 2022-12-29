#ifndef INC_MOTORCONTROLSTATEMACHINE_H_
#define INC_MOTORCONTROLSTATEMACHINE_H_

#include "stdbool.h"
#include "generalDefines.h"
#include <math.h>
#include "motorDriverMain.h"
#include "counter.h"

typedef enum{
	MCS_HALT = 0,
  MCS_START_RUNING,
	MCS_RUNING,
	MCS_MOTOR_FATAL_ERROR
}e_motorControlStates;

// function proto
void handleMotors(void);
void handleMotor(EMotor motor);
void setMotorControlState(EMotor motor);
bool motorControlSatetExct(EMotor motor);
void calcMotorPWMCommand(EMotor motor);

#endif /* INC_MOTORCONTROLSTATEMACHINE_H_ */
