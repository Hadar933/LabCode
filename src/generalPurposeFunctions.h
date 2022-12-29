#ifndef INC_GENERALPURPOSEFUNCTIONS_H_
#define INC_GENERALPURPOSEFUNCTIONS_H_

#include "stdbool.h"
#include <stddef.h>
#include "rail.h"
#include "generalDefines.h"

typedef struct SContinuousAverage
{
  float AverageData;
  float currentData;
  float N;
  bool reset;
}SContinuousAverage;

typedef struct SMovingAverage
{
  float AverageData;
  float bufferData[WINDOW_SIZE];
  float popData;
  float currentData;
  uint8_t counter;
  bool arrayFull;
  bool reset;
}SMovingAverage;

// functions protos
void delay_ms(uint32_t mDelay);
void delay_us(uint32_t uDelay);
void continuousAverage(SContinuousAverage * dataIn);
void movingAverage(SMovingAverage * dataIn);
uint32_t getMillis(void);
uint32_t getSec(void);
RAIL_Time_t getuSec(void);

#define CALLBACK_uS(hz) (1.0/((float)hz)*1000000.0)

#endif /* INC_GENERALPURPOSEFUNCTIONS_H_ */
