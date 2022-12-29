#ifndef COUNTER_H
#define COUNTER_H

#include <string.h>
#include <stdio.h>
#include "spidrv.h"
#include "em_core.h"
#include "sl_spidrv_instances.h"
#include "generalDefines.h"
#include "sl_emlib_gpio_init_PF9_config.h"
#include "sl_emlib_gpio_init_PF12_config.h"
#include "motorDriverMain.h"

// use SPI handle for EXP header (configured in project settings)
#define SPI_HANDLE                  sl_spidrv_encoderSPI_handle

/**
 * \defgroup error_code Error Code
 * \{
 */
#define COUNTER_RETVAL  uint8_t

#define COUNTER_OK           0x00
#define COUNTER_INIT_ERROR   0xFF
/** \} */

/**
 * \defgroup register_commands Register commands
 * \{
 */
#define COUNTER_CMD_CLR               0x00    
#define COUNTER_CMD_WR                0x80    
#define COUNTER_CMD_LOAD              0xC0    
#define COUNTER_CMD_RD                0X40    
/** \} */

/**
 * \defgroup counter_click_registers Counter click registers
 * \{
 */
#define COUNTER_MDR0                  0x08    
#define COUNTER_MDR1                  0x10    
#define COUNTER_DTR                   0x18    
#define COUNTER_CNTR                  0x20    
#define COUNTER_OTR                   0x28    
#define COUNTER_STR                   0x30    
/** \} */

/**
 * \defgroup mdr0_count_mode_setting MDR0 count mode setting
 * \{
 */
#define COUNTER_NON_QUAD              0x00    
#define COUNTER_1X_QUAD               0x01    
#define COUNTER_2X_QUAD               0x02    
#define COUNTER_4X_QUAD               0x03    
/** \} */

/**
 * \defgroup  mdr0_count_type_setting MDR0 count type setting
 * \{
 */
#define COUNTER_FREE_RUN              0x00    
#define COUNTER_SINGLE_COUNT          0x04    
#define COUNTER_RANGE_LIMIT           0x08    
#define COUNTER_N_MODULE              0x0C    
/** \} */

/**
 * \defgroup mdr0_index_setting MDR0 index setting
 * \{
 */
#define COUNTER_INDEX_DISABLED        0x00    
#define COUNTER_INDEX_LOAD_CNTR       0x10     
#define COUNTER_INDEX_RESET_CNTR      0x20     
#define COUNTER_INDEX_LOAD_OTR        0x30     
/** \} */

/**
 * \defgroup mdr0_index_sign_setting MDR0 index sign setting
 * \{
 */
#define COUNTER_INDEX_NEG             0x00    
#define COUNTER_INDEX_POS             0x40    
/** \} */

/**
 * \defgroup mdr0_clock_filter_division_factor MDR0 clock filter division factor
 */
#define COUNTER_FILTER_CLOCK_DIV1     0x00    
#define COUNTER_FILTER_CLOCK_DIV2     0x80    
/** \} */

/**
 * \defgroup mdr1_buffer_size_setting MDR1 buffer size setting
 * \{
 */
#define COUNTER_MODE_32               0x00    
#define COUNTER_MODE_24               0x01    
#define COUNTER_MODE_16               0x02    
#define COUNTER_MODE_8                0x03    
/** \} */

/**
 * \defgroup mdr1_enable_disable_counting_setting MDR1 enable/disable counting setting
 * \{
 */
#define COUNTER_ENABLE                0x00    
#define COUNTER_DISABLE               0x04    

/** \} */

/**
 * \defgroup mdr1_flags_settings MDR1 flags settings
 * \{
 */
#define COUNTER_FLAG_DISABLE          0x00    
#define COUNTER_FLAG_IDX              0x10    
#define COUNTER_FLAG_CMP              0x20    
#define COUNTER_FLAG_BW               0x40    
#define COUNTER_FLAG_FY               0x80    
/** \} */

/** \} */ // End group macro 
// --------------------------------------------------------------- PUBLIC TYPES
///**
// * \defgroup type Types
// * \{
// */
//
///**
// * @brief Click ctx object definition.
// */
//typedef struct
//{
//    // Output pins
//
//    digital_out_t en;
//    digital_out_t cs;
//
//    // Input pins
//
//    digital_in_t int_pin;
//
//    // Modules
//
//    spi_master_t spi;
//    pin_name_t chip_select;
//
//    // Variables
//
//    uint8_t buffer_size;
//
//} counter_t;
//
///**
// * @brief Click configuration structure definition.
// */
//typedef struct
//{
//    // Communication gpio pins
//
//    pin_name_t miso;
//    pin_name_t mosi;
//    pin_name_t sck;
//    pin_name_t cs;
//
//    // Additional gpio pins
//
//    pin_name_t en;
//    pin_name_t int_pin;
//
//    // static variable
//
//    uint32_t spi_speed;
//    spi_master_mode_t spi_mode;
//    spi_master_chip_select_polarity_t cs_polarity;
//
//} counter_cfg_t;

/** \} */ // End types group
// ----------------------------------------------- PUBLIC FUNCTION DECLARATIONS

/**
 * \defgroup public_function Public function
 * \{
 */
#ifdef __cplusplus
extern "C"{
#endif

/**
 * @brief Config Object Initialization function.
 *
 * @param cfg  Click configuration structure.
 *
 * @description This function initializes click configuration structure to init state.
 * @note All used pins will be set to unconnected state.
 */
//void counter_cfg_setup ( counter_cfg_t *cfg );

/**
 * @brief Initialization function.
 * @param counter Click object.
 * @param cfg Click configuration structure.
 * 
 * @description This function initializes all necessary pins and peripherals used for this click.
 */
//COUNTER_RETVAL counter_init ( counter_t *ctx, counter_cfg_t *cfg );

/**
 * @brief Click Default Configuration function.
 *
 * @description This function executes default configuration for Counter click.
 */
void counter_default_cfg ( void );

/**
 * @brief Generic transfer function.
 *
 * @param ctx          Click object.
 * @param wr_buf       Write data buffer
 * @param wr_len       Number of byte in write data buffer
 * @param rd_buf       Read data buffer
 * @param rd_len       Number of byte in read data buffer
 *
 * @description Generic SPI transfer, for sending and receiving packages
 */
//void counter_generic_transfer
//(
//    counter_t *ctx,
//    uint8_t *wr_buf,
//    uint16_t wr_len,
//    uint8_t *rd_buf,
//    uint16_t rd_len
//);

/**
 * @brief Writes data into MDR0
 *
 * @param settings     MDR0 settings.
 * 
 * @note The MDR0 (Mode Register 0) is an 8-bit read/write register that sets up the
 * operating mode for the LS7366. Upon power up MDR0 is cleared to zero.
 *
 * @description This function writes data into MDR0, using click object.
 */
void counter_write_mdr0 ( uint8_t settings );

/**
 * @brief Writes data into MDR1
 *
 * @param settings     MDR1 settings.
 * 
 * @note The MDR1 (Mode Register 1) is an 8-bit read/write register which is appended
 * to MDR0 for additional modes. Upon power-up MDR1 is cleared to zero.
 *
 * @description This function writes data into MDR1, using click object.
 */
void counter_write_mdr1 ( uint8_t settings );

/**
 * @brief Writes data into DTR
 * 
 * @param buffer       Data to be written into DTR.
 *
 * @note The DTR is a software configurable 8, 16, 24 or 32-bit input data register
 * which can be written into directly from MOSI, the serial input. The DTR data
 * can be transferred into the 32-bit counter (CNTR) under program control or
 * by hardware index signal.
 *
 * @description This function writes data into DTR, using click object.
 */
void counter_write_dtr ( uint8_t buffer_size, uint8_t* buffer );

/**
 * @brief Loads CNTR from DTR
 * 
 * @param ctx          Click object.
 *
 * @note The CNTR is a software configurable 8, 16, 24 or 32-bit up/down counter which
 * counts the up/down pulses resulting from the quadrature clocks applied at the
 * A and B inputs, or alternatively, in non-quadrature mode, pulses applied at
 * the A input.
 *
 * @description This function loads CNTR from DTR, using click object.
 */
void counter_load_cntr ( );

/**
 * @brief Loads OTR from CNTR
 *
 * 
 * @note The OTR is a software configuration 8, 16, 24 or 32-bit register which can be
 * read back on the MISO output. Since instantaneous CNTR value is often needed
 * to be read while the CNTR continues to count, the OTR serves as a convenient
 * dump site for instantaneous CNTR data which can then be read without
 * interfering with the counting process.
 * 
 * @description This function loads OTR from CNTR, using click object.
 */
void counter_load_otr ( );

/**
 * @brief Reads MDR0
 * 
 * @returns            Data from MDR0.
 * 
 * @description This function reads MDR0, using click object.
 */
uint8_t counter_read_mdr0 ( );

/**
 * @brief Reads MDR1
 * 
 * @returns            Data from MDR1.
 * 
 * @description This function reads MDR1, using click object.
 */
uint8_t counter_read_mdr1 ( );

/**
 * @brief Reads OTR
 * 
 * @param ctx          Click object.
 * @returns            OTR register data.
 * 
 * @description This function reads OTR, using click object.
 */
int32_t counter_read_otr ( );

/**
 * @brief Reads CNTR
 *
 * @returns            CNTR register data.
 * 
 * @description This function reads CNTR, using click object.
 */
void counter_read_cntr ( EMotor motor);

/**
 * @brief Reads DTR
 *
 * @param ctx          Click object.
 * @returns            DTR register data.
 * 
 * @description This function reads DTR, using click object.
 */
int32_t counter_read_dtr ( );

/**
 * @brief Reads STR
 *
 * @returns            Status register data.
 * 
 * @note
 * <pre>
 * The STR is an 8-bit status register which stores count related status
 * information.
 *
 * B0 - Sign bit. 1: negative, 0: positive
 * B1 - Count direction indicator: 0: count down, 1: count up
 * B2 - Power loss indicator latch; set upon power up
 * B3 - Count enable status: 0: counting disabled, 1: counting enabled
 * B4 - Index latch
 * B5 - Compare (CNTR = DTR) latch
 * B6 - Borrow (CNTR underflow) latch
 * B7 - Carry (CNTR overflow) latch
 * </pre>
 * 
 * @description This function reads STR, using click object.
 */
uint8_t counter_read_str ( );

/**
 * @brief Clears MDR0
 * 
 * @description This function clears MDR0.
 */
void counter_clear_mrd0 ( );

/**
 * @brief Clears MDR1
 * 
 * @description This function clears MDR1.
 */
void counter_clead_mrd1 ( );

/**
 * @brief Clears CNTR
 * 
 * @description This function clears CNTR.
 */
void counter_clear_cntr ( );

/**
 * @brief Clears STR
 *
 * @note
 * <pre>
 * Resets all status bits except CEN and U/D. In quadrature mode, if the
 * quadrature clocks have been halted, the status bits CY, BW and CMP are not
 * affected by a “CLR STR” command under the following conditions:
 *      CY: If CNTR = FFFFFFFF with status bit U/D = 1
 *      BW: If CNTR = 0 with status bit U/D = 0
 *      CMP: If CNTR = DTR
 * In non-quadrature mode the same rules apply if input A is held at
 * logic low.
 * </pre>
 * 
 * @description This function clears STR
 */
void counter_clear_str ( );

/**
 * @brief Initialisation of click board with default settings.
 * 
 * @note
 * <pre>
 * MDR0 settings :
 *      4X quad
 *      Free run
 *      Counter index disabled
 *      Filter clock division 1
 *
 * MDR1 settings :
 *      Counter mode 32
 *      Counter disabled
 *      Flags disabled
 * </pre>
 * 
 * @description This function initialize of click board with default settings.
 */
void counter_initialisation ( );

/**
 * @brief Initialisation of click board with specific settings.
 * @param mdr0_set     Settings for mdr0.
 * @param mdr1_set     Settings for mdr1.
 * 
 * @description This function initialize of click board with specific settings.
 */
uint8_t counter_init_advanced ( uint8_t mdr0_set, uint8_t mdr1_set );

/**
 * @brief Enables counting ( Software mode )
 * 
 * @description This function enables counting in software mode.
 */
void counter_enable ( );

/**
 * @brief Disables counting ( Software mode )
 * 
 * @description This function disables counting in software mode.
 */
void counter_disable ( );

/**
 * @brief Enables counting ( Hardware mode )
 * 
 * @description This function enables counting in hardware mode.
 */
//void counter_chip_enable ( );

/**
 * @brief Disables counting ( Hardware mode )
 * 
 * @description This function disables counting in hardware mode.
 */
//void counter_chip_disable ( );

/**
 * @brief Function for get int-pin state
 * 
 * @returns            Int pin state.
 * 
 * @description This function gets int-pin state.
 */
//uint8_t get_int_state ( );



uint8_t counter_read_register( uint8_t command );
void counter_read_data ( uint8_t command, uint8_t *data_buff, uint8_t count );
void counter_write_data (uint8_t command, uint8_t *data_buff, uint8_t count);
void counter_write_command ( uint8_t command );

#ifdef __cplusplus
}
#endif
#endif  // _COUNTER_H_

/** \} */ // End public_function group
/// \}    // End click Driver group  
/*! @} */
// ------------------------------------------------------------------------- END
