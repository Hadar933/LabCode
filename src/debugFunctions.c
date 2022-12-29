#include "debugFunctions.h"
#include "motorDriverMain.h"

extern SMotorsData motors[NUM_OF_MOTORS];
//extern uint32_t pinCounter[16];

SDBGMainDBG mainDBG;


void record_motor_data(EMotor motor)
{
  // ========================================================== DEBUG - START ========================================================================
  mainDBG.speedControl[motor].counter++;
  uint32_t ind = mainDBG.speedControl[motor].counter;
  mainDBG.speedControl[motor].commutation[ind] = motors[motor].commutation;
  mainDBG.speedControl[motor].PI_correction[ind] = motors[motor].speedControler.speed_I_correction;
  mainDBG.speedControl[motor].PWMpercent[ind] =  motors[motor].PWMCommand;
  mainDBG.speedControl[motor].speedFromHull[ind] = motors[motor].speedControler.speedFromHull;
  mainDBG.speedControl[motor].speedAverage[ind] = motors[motor].speedControler.speedAverage.AverageData;
  mainDBG.speedControl[motor].correcteddSpeed[ind] = motors[motor].speedControler.correctedSpeed;
  if (mainDBG.speedControl[motor].counter > (speedControlDbgArraySize - 1))
  {
    DEBUG_BREAK;
    while (1);
//    mainDBG.speedControl[motor].counter = 0;
  }
  // ========================================================== DEBUG - END ==========================================================================
}

void record_gCommotationState(EMotor motor, uint8_t gCommotationState)
{
  mainDBG.speedControl[motor].gCommotationState[mainDBG.speedControl[motor].counter] = gCommotationState;
}

void record_SpeedError(EMotor motor, float SpeedError)
{
  mainDBG.speedControl[motor].speedError[mainDBG.speedControl[motor].counter] = SpeedError;
}

void record_Pcorrection(EMotor motor, float Pcorrection)
{
  mainDBG.speedControl[motor].Pcorrection[mainDBG.speedControl[motor].counter] = Pcorrection;
}

void record_speed_I_correction(EMotor motor, float speed_I_correction)
{
  mainDBG.speedControl[motor].speed_I_correction[mainDBG.speedControl[motor].counter] = speed_I_correction;
}

void record_motorSpeedCommandRPM(EMotor motor, float refMotorSpeedRPM)
{
  mainDBG.speedControl[motor].motorSpeedCommandRPM[mainDBG.speedControl[motor].counter] = refMotorSpeedRPM;
}

void record_commandVrms(EMotor motor, float commandVrms)
{
  mainDBG.speedControl[motor].commandVrms[mainDBG.speedControl[motor].counter] = commandVrms;
}


void record_hull(EMotor motor)
{
  mainDBG.hullCounter[motor].hull[mainDBG.hullCounter[motor].counter] = motors[motor].hull;
  mainDBG.hullCounter[motor].counter++;
  if (mainDBG.hullCounter[motor].counter > (hullCounterDbgArraySize - 1))
  {
    DEBUG_BREAK;
    while (1);
  }
}

void getHullsDBG(EMotor motor)
{
  uint8_t ind = mainDBG.hullCounter[motor].counter;
  switch(motor)
  {
    case left:
      mainDBG.hullCounter[motor].motorsRealHulls[ind].HullU = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_PD8_PORT, SL_EMLIB_GPIO_INIT_PD8_PIN);
      mainDBG.hullCounter[motor].motorsRealHulls[ind].HullV = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_PA6_PORT, SL_EMLIB_GPIO_INIT_PA6_PIN);
      mainDBG.hullCounter[motor].motorsRealHulls[ind].HullW = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_PA7_PORT, SL_EMLIB_GPIO_INIT_PA7_PIN);
      break;

    case right:
      mainDBG.hullCounter[motor].motorsRealHulls[ind].HullU = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_F13_PORT, SL_EMLIB_GPIO_INIT_F13_PIN);
      mainDBG.hullCounter[motor].motorsRealHulls[ind].HullV = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_F15_PORT, SL_EMLIB_GPIO_INIT_F15_PIN);
      mainDBG.hullCounter[motor].motorsRealHulls[ind].HullW = GPIO_PinInGet(SL_EMLIB_GPIO_INIT_F14_PORT, SL_EMLIB_GPIO_INIT_F14_PIN);
      break;

    default:
      ERROR_BREAK
  }
  mainDBG.hullCounter[motor].motorsRealHulls[ind].currentSequence = ((mainDBG.hullCounter[motor].motorsRealHulls[ind].HullU << 2) | (mainDBG.hullCounter[motor].motorsRealHulls[ind].HullV << 1) | (mainDBG.hullCounter[motor].motorsRealHulls[ind].HullW)) & 0x7;
  mainDBG.hullCounter[motor].motorsRealHulls[ind].currentSequence -= 1;
  mainDBG.hullCounter[motor].counter++;
  if (mainDBG.hullCounter[motor].counter > (hullCounterDbgArraySize - 1))
  {
    DEBUG_BREAK;
    while (1);
  }
  return;
}
