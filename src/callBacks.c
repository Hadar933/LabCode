#include "callBacks.h"

#ifdef DEBUG_HULLS
  #include "debugFunctions.h"
#endif

//uint8_t prevHullSequence[NUM_OF_MOTORS];

// the correct hull sequence Hall State (Hall a, Hall b, Hall c)
//4 (100) 00  10  01
//6 (110) 01  10  00
//2 (010) 01  00  10
//3 (011) 00  01  10
//1 (001) 10  01  00
//5 (101) 10  00  01
// using this sequence we create a look up table:
int8_t hallIrqCntAdevanceMatrix[6][6] = { {0,0,-1,0,1,0},\
                                          {0,0,1,0,0,-1},\
                                          {1,-1,0,0,0,0},\
                                          {0,0,0,0,-1,1},\
                                          {-1,0,0,1,0,0},\
                                          {0,1,0,-1,0,0}  };

static STimedCallBacks timedCallBacksDB[endOfTimedCallbacksFuncList];
static SIntCallBacks intCallBacksDB[endOfIntCallbacksFuncList];

extern SMotorsData motors[NUM_OF_MOTORS];

//uint32_t pinCounter[16];

void setTimedCallBacksDB(void)
{
  timedCallBacksDB[0].func = handleMotors;
  timedCallBacksDB[0].us = CALLBACK_uS(MOTOR_SPEED_CONTROLLER_HZ);
  timedCallBacksDB[0].prevTimeCall = getuSec();

  timedCallBacksDB[1].func = callback_change_dir;
  timedCallBacksDB[1].us = CALLBACK_uS(CHANGE_DIR_HZ);
  timedCallBacksDB[1].prevTimeCall = getuSec();

}

void setIntCallBacksDB(void)
{
  // ====================== Hull motor 1 =================================
  intCallBacksDB[motor1U].func = callback_pin8;
  intCallBacksDB[motor1U].pin = SL_EMLIB_GPIO_INIT_PD8_PIN;
  intCallBacksDB[motor1U].port = SL_EMLIB_GPIO_INIT_PD8_PORT;

  intCallBacksDB[motor1V].func = callback_pin6;
  intCallBacksDB[motor1V].pin = SL_EMLIB_GPIO_INIT_PA6_PIN;
  intCallBacksDB[motor1V].port = SL_EMLIB_GPIO_INIT_PA6_PORT;

  intCallBacksDB[motor1W].func = callback_pin7;
  intCallBacksDB[motor1W].pin = SL_EMLIB_GPIO_INIT_PA7_PIN;
  intCallBacksDB[motor1W].port = SL_EMLIB_GPIO_INIT_PA7_PORT;

  // ====================== Hull motor 2 =================================
  intCallBacksDB[motor2U].func = callback_pin13;
  intCallBacksDB[motor2U].pin = SL_EMLIB_GPIO_INIT_F13_PIN;
  intCallBacksDB[motor2U].port = SL_EMLIB_GPIO_INIT_F13_PORT;

  intCallBacksDB[motor2V].func = callback_pin15;
  intCallBacksDB[motor2V].pin = SL_EMLIB_GPIO_INIT_F15_PIN;
  intCallBacksDB[motor2V].port = SL_EMLIB_GPIO_INIT_F15_PORT;

  intCallBacksDB[motor2W].func = callback_pin14;
  intCallBacksDB[motor2W].pin = SL_EMLIB_GPIO_INIT_F14_PIN;
  intCallBacksDB[motor2W].port = SL_EMLIB_GPIO_INIT_F14_PORT;

  // ====================== Encoder motor 1 =================================
  intCallBacksDB[motor1EncA].func = callback_pin1;
  intCallBacksDB[motor1EncA].pin = SL_EMLIB_GPIO_INIT_I1_PIN;
  intCallBacksDB[motor1EncA].port = SL_EMLIB_GPIO_INIT_I1_PORT;

  intCallBacksDB[motor1EncB].func = callback_pin2;
  intCallBacksDB[motor1EncB].pin = SL_EMLIB_GPIO_INIT_I2_PIN;
  intCallBacksDB[motor1EncB].port = SL_EMLIB_GPIO_INIT_I2_PORT;

  intCallBacksDB[motor1EncI].func = callback_pin3;
  intCallBacksDB[motor1EncI].pin = SL_EMLIB_GPIO_INIT_I3_PIN;
  intCallBacksDB[motor1EncI].port = SL_EMLIB_GPIO_INIT_I3_PORT;

  // ====================== Encoder motor 2 =================================
  intCallBacksDB[motor2EncA].func = callback_pin9;
  intCallBacksDB[motor2EncA].pin = SL_EMLIB_GPIO_INIT_PB9_PIN;
  intCallBacksDB[motor2EncA].port = SL_EMLIB_GPIO_INIT_PB9_PORT;

  intCallBacksDB[motor2EncB].func = callback_pin4;
  intCallBacksDB[motor2EncB].pin = SL_EMLIB_GPIO_INIT_PC4_PIN;
  intCallBacksDB[motor2EncB].port = SL_EMLIB_GPIO_INIT_PC4_PORT;

  intCallBacksDB[motor2EncI].func = callback_pin5;
  intCallBacksDB[motor2EncI].pin = SL_EMLIB_GPIO_INIT_PC5_PIN;
  intCallBacksDB[motor2EncI].port = SL_EMLIB_GPIO_INIT_PC5_PORT;
}


void executeTimedFunctions(void)
{
  for(ETimedCallBacksdFunctions ind = 0; ind < endOfTimedCallbacksFuncList; ind++)
    {
      uint64_t currentTime = getuSec();
      if(currentTime - timedCallBacksDB[ind].prevTimeCall >= timedCallBacksDB[ind].us)
      {
          timedCallBacksDB[ind].prevTimeCall = currentTime;
          (*timedCallBacksDB[ind].func)();
      }
    }
}


void init_callbacks_GPIO(void)
{
  setIntCallBacksDB();
  // Initialization of the GPIOINT driver.
  GPIOINT_Init();

  for(uint8_t ind = 0; ind < endOfIntCallbacksFuncList; ind++)
  {
      GPIOINT_CallbackRegister(intCallBacksDB[ind].pin, intCallBacksDB[ind].func);
      GPIO_ExtIntConfig(intCallBacksDB[ind].port, intCallBacksDB[ind].pin, intCallBacksDB[ind].pin, true, true, true);
  }
}


void callback_pin1(uint8_t intNo) // pin I1
{
  (void) intNo; // not in use
//  pinCounter[1]++;
}

void callback_pin2(uint8_t intNo) // pin I2
{
  (void) intNo; // not in use
//  pinCounter[2]++;
}

void callback_pin3(uint8_t intNo) // pin I3
{
  (void) intNo; // not in use
//  pinCounter[3]++;
}

void callback_pin4(uint8_t intNo) // pin C4
{
  (void) intNo; // not in use
//  pinCounter[4]++;
}

void callback_pin5(uint8_t intNo) // pin C5
{
  (void) intNo; // not in use
//  pinCounter[5]++;
}

void callback_pin6(uint8_t intNo) // pin A6
{
  (void) intNo; // not in use
//  pinCounter[6]++;
//  motors[left].hull.HullV ^= 1;
  hullHandle(left);
}

void callback_pin7(uint8_t intNo) // pin A7
{
  (void) intNo; // not in use
//  pinCounter[7]++;
//  motors[left].hull.HullW ^= 1;
  hullHandle(left);
}

void callback_pin8(uint8_t intNo) // pin D8
{
  (void) intNo; // not in use
//  pinCounter[8]++;
//  motors[left].hull.HullU ^= 1;
  hullHandle(left);
}

void callback_pin9(uint8_t intNo) // pin B9
{
  (void) intNo; // not in use
//  pinCounter[9]++;
}

void callback_pin13(uint8_t intNo) // pin F13
{
  (void) intNo; // not in use
//  pinCounter[13]++;
//  motors[right].hull.HullU ^= 1;
  hullHandle(right);
}

void callback_pin14(uint8_t intNo) // pin F14
{
  (void) intNo; // not in use
//  pinCounter[14]++;
//  motors[right].hull.HullW ^= 1;
  hullHandle(right);
}

void callback_pin15(uint8_t intNo) // pin F15
{
  (void) intNo; // not in use
  hullHandle(right);
}

void hullHandle(EMotor motor)
{
  getMotorHulls(motor);
  motorPhaseConfigurationHandle(motor);
  getHullSequence(motor);
  calcHullAdd(motor);
  motors[motor].hull.cnt_last_time_uSec = getuSec();
  motors[motor].hull.cnt += motors[motor].hull.hullAdd;
  sendCommandToDriver(motor);
#ifdef DEBUG_HULLS
//  getHullsDBG(motor);
  record_hull(motor);
#endif
}

void callback_change_dir(EMotor motor)
{

  motors[left].speedControler.refSpeed *= 1;
}
