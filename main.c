/***************************************************************************//**
 * @file main.c
 * @brief main() function.
 *******************************************************************************
 * # License
 * <b>Copyright 2020 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/


//TODO - need to change the project debug setting and optimization level when
// switching to operational code


#include "sl_component_catalog.h"
#include "sl_system_init.h"
#include "app.h"
#if defined(SL_CATALOG_POWER_MANAGER_PRESENT)
#include "sl_power_manager.h"
#endif
#if defined(SL_CATALOG_KERNEL_PRESENT)
#include "sl_system_kernel.h"
#else // SL_CATALOG_KERNEL_PRESENT
#include "sl_system_process_action.h"
#endif // SL_CATALOG_KERNEL_PRESENT

#include "stdbool.h"
#include <stddef.h>


#include "generalDefines.h"
#include "generalPurposeFunctions.h"
#include "motorControlStateMachine.h"
#include "callBacks.h"
#include "motorDriverMain.h"
#include<unistd.h>
#include "stdio.h"

//#include "tests/pwm/test_pwm.h"
//#include "tests/gpio_int/test_gpio_int.h"
//#include "tests/read_hulls/read_hulls.h"
//#include "tests/no_hulls/test_no_hulls.h"
//#include "tests/change_dir/testChangeDir.h"
//#include "tests/movingAverage/testMovingAverage.h"
//#include "tests/SPI/test_spi.h"


extern SMotorsData motors[NUM_OF_MOTORS];

int main(void)
{
  motorDriverPhaseConfigurationInit();

  // Initialize Silicon Labs device, system, service(s) and protocol stack(s).
  // Note that if the kernel is present, processing task(s) will be created by
  // this call.
  sl_system_init();

  // Initialize the application. For example, create periodic timer(s) or
  // task(s) if the kernel is present.
//  app_init();


  // Initialize callbacks
  GPIO_PinOutSet(SL_EMLIB_GPIO_INIT_PF9_PORT, SL_EMLIB_GPIO_INIT_PF9_PIN);
  setTimedCallBacksDB();
  init_callbacks_GPIO();
  counter_default_cfg( );
  delay_ms(300);

  motors[left].speedControler.refSpeed = 50.0; // [Rad/sec]

// Test functions for Debug
//  test_gpio_init();
//  test_pwm();
//  read_hulls();
//  test_motors_handle();
//  runMotorNoHulls(left);
//  init_test_callbacks_timed();
//  test_moving_average();
//  test_counter_spi();
//  test_arduino_spi();
//  test_counter_spi_once();
 // int iter =100;

  //TODO: export real speeds to excel file
  //outData.open("motorTest1");

  float real_speeds[200]={-1} ;
  float real_times[200]={0} ;
  float speeds[]= {50,
      40,
      30,
      40,
      40,
      -40,
      -35,
      -40,
      -50,
      -45,
      -40,
      -35,
      -30,
      30,
      35,
      0,
      0,


} ;//[rad/sec],


#if defined(SL_CATALOG_KERNEL_PRESENT)
  // Start the kernel. Task(s) created in app_init() will start running.
  sl_system_kernel_start();
#else // SL_CATALOG_KERNEL_PRESENT
  uint64_t OuterStartTime = getuSec();

  uint64_t currentTime = getuSec();
  int i = 0;
  int j = 0;
  int run_sec = 10;
  //int run_usec = run_sec * 1000000;

    while (currentTime - OuterStartTime < 10000000){
        uint64_t InnerStartTime = getuSec();
//      executeTimedFunctionsTest();
        motors[left].speedControler.refSpeed =speeds[j%17];

        while (getuSec() - InnerStartTime < 1000000){
            // do nothing
            //printf("boom1");
        }
        //printf("boom2");
        //TODO: supposedly works, but make sure that an array of speeds if correctly fed into the motor (every one sec, speed increments)
        //motors[left].speedControler.refSpeed =30.0;
        executeTimedFunctions();
        real_speeds[i]=  motors[left].speedControler.speedFromEncoder;
        real_times[i]=  currentTime - OuterStartTime;
        i++;
        j++;
        currentTime = getuSec();

    // Do not remove this call: Silicon Labs components process action routine
    // must be called from the super loop.
//    sl_system_process_action();

    // Application process.
//    app_process_action();

      #if defined(SL_CATALOG_POWER_MANAGER_PRESENT)
          // Let the CPU go to sleep if the system allows it.
          //sl_power_manager_sleep();
      #endif
  }

  while (1) {
        getuSec() ;
    }


    //return 0;
      //motors[left].PWMCommand=0
    //sendPWMCommadToAllMotors()

#endif // SL_CATALOG_KERNEL_PRESENT
}
