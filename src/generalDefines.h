#ifndef SRC_GENERALDEFINES_H_
#define SRC_GENERALDEFINES_H_

//Debug
//#define DEBUG_SPEED_CONTROL
//#define DEBUG_HULLS

#define SPEED_CONTROL_ON


// Units conversion
#define PI (3.1415926535897932384626433832795)
#define RAD_PER_HULL_INT (PI/3.0)
#define RAD_PER_ENCODER_INT (PI/512.0) // TODO - need to check that the encoder gives 1024 interapt per rev
#define PI_ANTIWINDUP (1000.0)
#define IS_ZERO_FLOAT(dataIn) ((dataIn < 0.00001 && dataIn > -0.00001)?true:false)
#define RadPS_TO_RPM (60.0/(2.0*PI))
#define COUNTER_TO_RAD(x) (((float)x)*2*PI/1024.0f)

#define WINDOW_SIZE  (5) // moving average window size

#define NUM_OF_MOTORS (2)


#define ERROR_BREAK           __asm__("BKPT #0");
#define DEBUG_BREAK           __asm__("");

#ifdef DEBUG_MODE
#define DEBUG_BREAK           ERROR_BREAK;
#else
#define DEBUG_BREAK           __asm__("");
#endif

#endif /* SRC_GENERALDEFINES_H_ */
