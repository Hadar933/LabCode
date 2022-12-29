#ifndef SRC_MOTORSCHARACTERSDB_H_
#define SRC_MOTORSCHARACTERSDB_H_

#define MOTOR_SPEED_CONSTANT (25800)  // [RPM/V]
#define SPEED_TO_TORQUE_GRAD (53400) // [RPM/mNm]
#define TORQUE_CONST (0.37) // [mNm/A]
#define MOTOR_GEAR_EFFICIENCY (0.76)
#define MOTOR_EFFICIENCY (0.63 * MOTOR_GEAR_EFFICIENCY)
#define GEAR_RATIO (15.0)
#define INV_GEAR_RATIO (1.0/GEAR_RATIO)
#define POWER_SUPPLY_VOLTAGE (5.0)
#define STARTING_VOLTAGE (0.6)
// control gains
// TODO - if needed define KI, KP for each motor
#define KI (0.02)
#define KP (2.0)


#define MOTOR_SPEED_CONTROLLER_HZ (300) // TODO - change to the correct Hz

//#define QUATER_SPEED_STEPS 5
#define CHANGE_DIR_HZ (1) // * QUATER_SPEED_STEPS * 4

#endif /* SRC_MOTORSCHARACTERSDB_H_ */
