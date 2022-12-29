#include "motorDriverMain.h"

fctPtr pwmSetDutyCycle[] = {GPIO_motorPinPWMoutLow , GPIO_motorPinPWMoutHigh , GPIO_motorPinPWMoutDisable};
S_fullMotorPhaseConfiguration motorPhaseConfiguration;
uint8_t gCommotationState[NUM_OF_MOTORS];
SMotorsData motors[NUM_OF_MOTORS];

extern sl_pwm_instance_t sl_pwm_motor1_ch0;
extern sl_pwm_instance_t sl_pwm_motor1_ch1;
extern sl_pwm_instance_t sl_pwm_motor1_ch2;
extern sl_pwm_instance_t sl_pwm_motor2_ch0;
extern sl_pwm_instance_t sl_pwm_motor2_ch1;
extern sl_pwm_instance_t sl_pwm_motor2_ch2;
extern int8_t hallIrqCntAdevanceMatrix[6][6];

void sendCommandToDriver(EMotor motor){
  static sl_pwm_instance_t *motor_pwm_ch0;
  static sl_pwm_instance_t *motor_pwm_ch1;
  static sl_pwm_instance_t *motor_pwm_ch2;

  switch(motor){
    case left:
      motor_pwm_ch0 = &sl_pwm_motor1_ch0;
      motor_pwm_ch1 = &sl_pwm_motor1_ch1;
      motor_pwm_ch2 = &sl_pwm_motor1_ch2;
      break;

    case right:
      motor_pwm_ch0 = &sl_pwm_motor2_ch0;
      motor_pwm_ch1 = &sl_pwm_motor2_ch1;
      motor_pwm_ch2 = &sl_pwm_motor2_ch2;
      break;

    default:
      ERROR_BREAK
      return;
  }

	(*pwmSetDutyCycle[motors[motor].commutation.Upolarity])(motor_pwm_ch0, motor);
	(*pwmSetDutyCycle[motors[motor].commutation.Vpolarity])(motor_pwm_ch1, motor);
	(*pwmSetDutyCycle[motors[motor].commutation.Wpolarity])(motor_pwm_ch2, motor);
}


void getAllMotorsCommutation(void)
{
  for(EMotor motor = right; motor < endOfMotors; motor++)
  {
      getMotorHulls(motor);
  }
}


void getMotorHulls(EMotor motor){
  switch(motor)
  {
    case left:
      motors[motor].hull.HullU = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_PD8_PORT, SL_EMLIB_GPIO_INIT_PD8_PIN);
      motors[motor].hull.HullV = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_PA6_PORT, SL_EMLIB_GPIO_INIT_PA6_PIN);
      motors[motor].hull.HullW = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_PA7_PORT, SL_EMLIB_GPIO_INIT_PA7_PIN);
      break;

    case right:
      motors[motor].hull.HullV = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_F15_PORT, SL_EMLIB_GPIO_INIT_F15_PIN);
      motors[motor].hull.HullW = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_F14_PORT, SL_EMLIB_GPIO_INIT_F14_PIN);
      motors[motor].hull.HullU = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_F13_PORT, SL_EMLIB_GPIO_INIT_F13_PIN);
      break;

    default:
      ERROR_BREAK
  }
  return;
}

void motorPhaseConfigurationHandle(EMotor motor)
{
  gCommotationState[motor] = (motors[motor].hull.HullU << 2 | motors[motor].hull.HullV << 1 | motors[motor].hull.HullW) & 0x7;

#ifdef DEBUG_SPEED_CONTROL
  record_gCommotationState(motor, gCommotationState[motor]);
#endif


  if ((motors[motor].motorDriveState == DS_CW) || (motors[motor].motorDriveState == DS_STOP))
  {
    motors[motor].commutation = motorPhaseConfiguration.forward[gCommotationState[motor]];
  }
  else if (motors[motor].motorDriveState == DS_CCW)
  {
    motors[motor].commutation = motorPhaseConfiguration.backward[gCommotationState[motor]];
  }
  return;
}

// ==================================== PI speed control algorithm - START ===================================
float PISpeedControl(EMotor motor)
{
// TODO - enable if each motor has different parrams
//  static float ki;
//  static float kp;
//  switch(motor){
//    case left:
//      ki = LEFT_KI;
//      kp = LEFT_KP;
//      break;
//
//    case right:
//      ki = RIGHT_KI;
//      kp = RIGHT_KP;
//      break;
//  }

	float SpeedError = motors[motor].speedControler.refSpeed - motors[motor].speedControler.speedAverage.AverageData; // TODO - speedAverage or speedFromHall

	float Pcorrection = KP * SpeedError;
	float speed_I_correction = KI * SpeedError + motors[motor].speedControler.speed_I_correction; // TODO need to add dt to the integration


	if (speed_I_correction > PI_ANTIWINDUP) // unti-windup
	{
		speed_I_correction = PI_ANTIWINDUP;
	}

	if (speed_I_correction < -1 * PI_ANTIWINDUP) // unti-windup
	{
		speed_I_correction = -1 * PI_ANTIWINDUP;
	}
	motors[motor].speedControler.speed_I_correction = speed_I_correction;

#ifdef DEBUG_SPEED_CONTROL
	record_SpeedError(motor, SpeedError);
	record_Pcorrection(motor, Pcorrection);
	record_speed_I_correction(motor, speed_I_correction);
#endif

	return Pcorrection + speed_I_correction;
}
// ==================================== PI linear speed control algorithm - START ===================================


// ==================================== GPIO motor Pin PWM out Disable - START ===================================
void GPIO_motorPinPWMoutDisable(sl_pwm_instance_t *motor_pwm_ch, EMotor motor)
{
  (void) motor; // unused argument
  sl_pwm_set_duty_cycle(motor_pwm_ch, 50);
  sl_pwm_start(motor_pwm_ch);
}
// ==================================== GPIO motor Pin PWM out Disable - END ===================================


// ==================================== GPIO motor Pin PWM out High - START ===================================
void GPIO_motorPinPWMoutHigh(sl_pwm_instance_t *motor_pwm_ch, EMotor motor){
  sl_pwm_set_duty_cycle(motor_pwm_ch, motors[motor].PWMCommand);
  sl_pwm_start(motor_pwm_ch);
}
// ==================================== GPIO motor Pin PWM out High - END ===================================


// ==================================== GPIO motor Pin PWM out Low - START ===================================
void GPIO_motorPinPWMoutLow(sl_pwm_instance_t *motor_pwm_ch, EMotor motor){
  (void) motor; // unused argument
//  sl_pwm_set_duty_cycle(motor_pwm_ch, 0);
//  sl_pwm_start(motor_pwm_ch);
  sl_pwm_stop(motor_pwm_ch);
}
// ==================================== GPIO motor Pin PWM out Low - END ===================================


// ==================================== set Motor Drive State - START ===================================
void setMotorDriveState(EMotor motor)
{

  if(motors[motor].motorDriveState == DS_STOP)
  {
    if(motors[motor].speedControler.refSpeed < 0)
    {
        motors[motor].motorDriveState = DS_CCW;
    }
    else
    {
        motors[motor].motorDriveState = DS_CW;
    }
  }
  else // motor at drive mode
  {
      if(IS_ZERO_FLOAT(motors[motor].speedControler.refSpeed))
      {
          motors[motor].motorDriveState = DS_STOP;
      }
      else
      {
        if (motors[motor].speedControler.correctedSpeed < 0)
        {
            motors[motor].motorDriveState = DS_CCW;
        }
        else
        {
            motors[motor].motorDriveState = DS_CW;
        }
      }
  }
}
// ==================================== set Motor Drive State - END ===================================


// ==================================== set All Motors Drive State - START ===================================
void setAllMotorsDriveState(void)
{
  for(EMotor motor = right; motor < endOfMotors; motor++)
  {
      setMotorDriveState(motor);
  }
}
// ==================================== set all Motors Drive State - START ===================================


// ==================================== motor Driver Phase Configuration Initialize - START ===================================
void motorDriverPhaseConfigurationInit(void){

	// Xpolarity = 0 - LOW
	// Xpolarity = 1 - HIGH
	// Xpolarity = 2 - 50% duty cycle

	// ======================= BACKWARD CONFIGURATION - START =============================
	motorPhaseConfiguration.backward[0].Upolarity = NA;
	motorPhaseConfiguration.backward[0].Vpolarity = NA;
	motorPhaseConfiguration.backward[0].Wpolarity = NA;


	motorPhaseConfiguration.backward[1].Upolarity = D;
	motorPhaseConfiguration.backward[1].Vpolarity = H;
	motorPhaseConfiguration.backward[1].Wpolarity = L;


	motorPhaseConfiguration.backward[2].Upolarity = H;
	motorPhaseConfiguration.backward[2].Vpolarity = L;
	motorPhaseConfiguration.backward[2].Wpolarity = D;


	motorPhaseConfiguration.backward[3].Upolarity = H;
	motorPhaseConfiguration.backward[3].Vpolarity = D;
	motorPhaseConfiguration.backward[3].Wpolarity = L;


	motorPhaseConfiguration.backward[4].Upolarity = L;
	motorPhaseConfiguration.backward[4].Vpolarity = D;
	motorPhaseConfiguration.backward[4].Wpolarity = H;


	motorPhaseConfiguration.backward[5].Upolarity = L;
	motorPhaseConfiguration.backward[5].Vpolarity = H;
	motorPhaseConfiguration.backward[5].Wpolarity = D;


	motorPhaseConfiguration.backward[6].Upolarity = D;
	motorPhaseConfiguration.backward[6].Vpolarity = L;
	motorPhaseConfiguration.backward[6].Wpolarity = H;


	motorPhaseConfiguration.backward[7].Upolarity = NA;
	motorPhaseConfiguration.backward[7].Vpolarity = NA;
	motorPhaseConfiguration.backward[7].Wpolarity = NA;
	// ======================= BACKWARD CONFIGURATION - END =============================


	// ======================= FORWARD CONFIGURATION - START =============================
	motorPhaseConfiguration.forward[0].Upolarity = NA;
	motorPhaseConfiguration.forward[0].Vpolarity = NA;
	motorPhaseConfiguration.forward[0].Wpolarity = NA;


	motorPhaseConfiguration.forward[1].Upolarity = D;
	motorPhaseConfiguration.forward[1].Vpolarity = L;
	motorPhaseConfiguration.forward[1].Wpolarity = H;


	motorPhaseConfiguration.forward[2].Upolarity = L;
	motorPhaseConfiguration.forward[2].Vpolarity = H;
	motorPhaseConfiguration.forward[2].Wpolarity = D;


	motorPhaseConfiguration.forward[3].Upolarity = L;
	motorPhaseConfiguration.forward[3].Vpolarity = D;
	motorPhaseConfiguration.forward[3].Wpolarity = H;


	motorPhaseConfiguration.forward[4].Upolarity = H;
	motorPhaseConfiguration.forward[4].Vpolarity = D;
	motorPhaseConfiguration.forward[4].Wpolarity = L;


	motorPhaseConfiguration.forward[5].Upolarity = H;
	motorPhaseConfiguration.forward[5].Vpolarity = L;
	motorPhaseConfiguration.forward[5].Wpolarity = D;


	motorPhaseConfiguration.forward[6].Upolarity = D;
	motorPhaseConfiguration.forward[6].Vpolarity = H;
	motorPhaseConfiguration.forward[6].Wpolarity = L;


	motorPhaseConfiguration.forward[7].Upolarity = NA;
	motorPhaseConfiguration.forward[7].Vpolarity = NA;
	motorPhaseConfiguration.forward[7].Wpolarity = NA;
	// ======================= FORWARD CONFIGURATION - END =============================
}
// ==================================== motor Driver Phase Configuration Initialize - END ===================================


void getHullSequence(EMotor motor)
{
	uint8_t sequence = ((motors[motor].hull.HullU << 2) | (motors[motor].hull.HullV << 1) | (motors[motor].hull.HullW)) & 0x7;
	motors[motor].hull.currentSequence = sequence-1;
}


void calcHullAdd(EMotor motor)
{
  motors[motor].hull.hullAdd =  hallIrqCntAdevanceMatrix[motors[motor].hull.prevSequence][motors[motor].hull.currentSequence];
  motors[motor].hull.prevSequence = motors[motor].hull.currentSequence;
}

void calcSpeedFromHulls(EMotor motor)
{
  // to prevent changes in current time and hall counts during calculation we copy them to local variables
  uint32_t curentTimeuSec = motors[motor].hull.cnt_last_time_uSec;
  uint32_t currentHallCnt = motors[motor].hull.cnt;

	int currentDeltaCount = currentHallCnt - motors[motor].speedControler.lastHullCnt;
	uint32_t currentDt = curentTimeuSec - motors[motor].speedControler.lastHullCalcTimeuSec;

	if (currentDt == 0)
	{
	  motors[motor].speedControler.speedFromHull = 0;
	  return;
	}

  float motorAngleRotated = currentDeltaCount * RAD_PER_HULL_INT;
  float dtSec = currentDt/1000000.0;
  float motorSpeedBeforeGearRadSec = motorAngleRotated/dtSec;
  motors[motor].speedControler.speedFromHull = motorSpeedBeforeGearRadSec * INV_GEAR_RATIO;

	motors[motor].speedControler.lastHullCalcTimeuSec = curentTimeuSec;
	motors[motor].speedControler.lastHullCnt = currentHallCnt;

	motors[motor].speedControler.currentSpeed = motors[motor].speedControler.speedFromHull;
	motors[motor].speedControler.speedAverage.currentData = motors[motor].speedControler.currentSpeed;
  movingAverage(&motors[motor].speedControler.speedAverage);
	return;
}


void calcSpeedFromEncoder(EMotor motor)
{
  // to prevent changes in current time and hall counts during calculation we copy them to local variables
  uint32_t currentEncoderCnt = motors[motor].encoder.cnt;
  uint32_t curentTimeuSec = motors[motor].encoder.cnt_last_time_uSec;

  int currentDeltaCount = currentEncoderCnt - motors[motor].speedControler.lastEncoderCnt;
  uint32_t currentDt = curentTimeuSec - motors[motor].speedControler.lastEncoderCalcTimeuSec;

  if (currentDt == 0)
  {
    motors[motor].speedControler.speedFromEncoder = 0; //TODO - zero order hold the last speed
    return;
  }

  float motorAngleRotated = COUNTER_TO_RAD(currentDeltaCount);
  float dtSec = currentDt/1000000.0;
  float motorSpeedBeforeGearRadSec = motorAngleRotated/dtSec;

  motors[motor].speedControler.speedFromEncoder = motorSpeedBeforeGearRadSec * INV_GEAR_RATIO;

  motors[motor].speedControler.lastEncoderCalcTimeuSec = curentTimeuSec;
  motors[motor].speedControler.lastEncoderCnt = currentEncoderCnt;

  motors[motor].speedControler.currentSpeed = motors[motor].speedControler.speedFromEncoder;
  motors[motor].speedControler.speedAverage.currentData = motors[motor].speedControler.currentSpeed;
  movingAverage(&motors[motor].speedControler.speedAverage);
  return;
}


void speedControlHandle(EMotor motor)
{
#ifdef SPEED_CONTROL_ON
  float speedCorrection = PISpeedControl(motor);
  motors[motor].speedControler.correctedSpeed = motors[motor].speedControler.refSpeed + speedCorrection;
#else
  motors[motor].speedControler.correctedSpeed = motors[motor].speedControler.refSpeed;
#endif
}


void resetMotorData(EMotor motor)
{
	motors[motor].speedControler.speed_I_correction = 0;
	motors[motor].speedControler.correctedSpeed = 0;
	motors[motor].speedControler.speedFromHull = 0;
	motors[motor].speedControler.speedFromEncoder = 0;
	motors[motor].speedControler.currentSpeed = 0;
	motors[motor].speedControler.refSpeed = 0;
	motors[motor].speedControler.correctedSpeed = 0;
	motors[motor].speedControler.speedAverage.reset = true;
	setMotorDriveState(motor);
}


void resetAllDriveMotorsData(void)
{
  for(EMotor motor = right; motor < endOfMotors; motor++)
  {
    resetMotorData(motor);
  }
}


// ==================================== calc PWM command - START ===================================
void calcPWMpercent(EMotor motor)
{
  // calculate the motor needed voltage after PI controller
  float motorSpeedCommandRPM = fabs(motors[motor].speedControler.correctedSpeed) * GEAR_RATIO * RadPS_TO_RPM;
  float commandVrms = (motorSpeedCommandRPM / MOTOR_SPEED_CONSTANT) / MOTOR_EFFICIENCY + STARTING_VOLTAGE;
  uint32_t tmp = (commandVrms/POWER_SUPPLY_VOLTAGE)*100.0;
  if(tmp > 100.0)
  {
      motors[motor].PWMCommand = 100.0;
  }
  else
  {
      motors[motor].PWMCommand = tmp;
  }

#ifdef DEBUG_SPEED_CONTROL
  record_motorSpeedCommandRPM(motor, motorSpeedCommandRPM);
  record_commandVrms(motor, commandVrms);
#endif
  return;
}
// ==================================== calc PWM command - END ===================================


void sendPWMCommadToAllMotors(void)
{
  for(EMotor motor = right; motor < endOfMotors; motor++)
  {
      sendCommandToDriver(motor);
  }
}
