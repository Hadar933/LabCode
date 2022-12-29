#ifndef INC_MOTORDRIVERMAIN_H_
#define INC_MOTORDRIVERMAIN_H_

#include "generalDefines.h"
#include "stdbool.h"
#include "sl_pwm.h"
#include "stdint.h"
#include "generalPurposeFunctions.h"
#include "motorsDB.h"
#include "motorDriverMain.h"
#include "sl_pwm.h"
#include "sl_emlib_gpio_init_F13_config.h"
#include "sl_emlib_gpio_init_F14_config.h"
#include "sl_emlib_gpio_init_F15_config.h"
#include "sl_emlib_gpio_init_PA6_config.h"
#include "sl_emlib_gpio_init_PA7_config.h"
#include "sl_emlib_gpio_init_PD8_config.h"
#include "em_gpio.h"
#include <math.h>

#define MOTOR_PHASE_CONFIG_SIZE (8)


typedef enum{
  right = 0,
  left,
  endOfMotors
}EMotor;


typedef enum{
  L = 0,
  H,
  D,
  NA = 0
}EPolarity;

typedef struct
{
  uint8_t Upolarity;
  uint8_t Vpolarity;
  uint8_t Wpolarity;
}S_motorPhaseConfiguration;

typedef struct
{
  S_motorPhaseConfiguration forward[MOTOR_PHASE_CONFIG_SIZE];
  S_motorPhaseConfiguration backward[MOTOR_PHASE_CONFIG_SIZE];
}S_fullMotorPhaseConfiguration;

typedef struct
{
  uint8_t   currentSequence;
  uint8_t   prevSequence;
  uint8_t   HullU;
  uint8_t   HullV;
  uint8_t   HullW;
  int32_t   cnt;
  int8_t    prevHullAdded;
  uint32_t  cnt_last_time_uSec;
  int8_t    hullAdd;
}SGDComutation;


typedef struct
{
  int32_t   cnt;
  uint64_t  cnt_last_time_uSec;
}SGDEncoder;


typedef enum{
  DS_STOP = 0,
  DS_CW,
  DS_CCW
}e_driveState;


typedef struct {
  float               speed_I_correction;
  float               correctedSpeed;     // [Rad/sec]
  float               speedFromHull;      // [Rad/sec]
  float               speedFromEncoder;   // [rad/sec]
  float               currentSpeed;       // [rad/sec]
  uint64_t            lastHullCalcTimeuSec;
  uint64_t            lastEncoderCalcTimeuSec;
  int32_t             lastHullCnt;
  int32_t             lastEncoderCnt;
  float               refSpeed;
  SMovingAverage  speedAverage;
}SPIspeedContorl;

typedef struct{
  SGDComutation                 hull;
  SGDEncoder                 encoder;
  S_motorPhaseConfiguration     commutation;
  float                         payloadAngle;
  e_driveState                  motorDriveState;
  uint8_t                       PWMCommand;
  SPIspeedContorl               speedControler;
  bool                          isRunning;
}SMotorsData;



// ================= function prototypes ===========================
void sendCommandToDriver(EMotor motor);
void getAllMotorsCommutation(void);
void getMotorHulls(EMotor motor);
float PISpeedControl(EMotor motor);
void GPIO_motorPinPWMoutDisable(sl_pwm_instance_t *motor_ch, EMotor motor);
void GPIO_motorPinPWMoutHigh(sl_pwm_instance_t *motor_ch, EMotor motor);
void GPIO_motorPinPWMoutLow(sl_pwm_instance_t *motor_ch, EMotor motor);
void setMotorDriveState(EMotor motor);
void setAllMotorsDriveState(void);
void motorDriverPhaseConfigurationInit(void);
void getHullSequence(EMotor motor);
void calcSpeedFromHulls(EMotor motor);
void calcSpeedFromEncoder(EMotor motor);
void speedControlHandle(EMotor motor);
void resetMotorData(EMotor motor);
void resetAllDriveMotorsData(void);
void calcPWMpercent(EMotor motor);
void sendPWMCommadToAllMotors(void);
void motorPhaseConfigurationHandle(EMotor motor);
void calcHullAdd(EMotor motor);

// ============================= Type def =======================
typedef void(*fctPtr)(sl_pwm_instance_t *, EMotor);

#ifdef DEBUG_SPEED_CONTROL
  #include "debugFunctions.h"
#endif

#endif /* INC_MOTORDRIVERMAIN_H_ */
