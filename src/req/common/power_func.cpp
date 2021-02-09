#include "power_func.h"

void getRegisterPowerArea(float cycle_time,
                          float* internal_power_per_bit,
                          float* switch_power_per_bit,
                          float* leakage_power_per_bit,
                          float* area_per_bit) {
  switch ((int)cycle_time) {  // cycleTime in ns
    case 10:
      *internal_power_per_bit = REG_10ns_int_power;
      *switch_power_per_bit = REG_10ns_switch_power;
      *leakage_power_per_bit = REG_10ns_leakage_power;
      *area_per_bit = REG_10ns_area;
      break;
    case 6:
      *internal_power_per_bit = REG_6ns_int_power;
      *switch_power_per_bit = REG_6ns_switch_power;
      *leakage_power_per_bit = REG_6ns_leakage_power;
      *area_per_bit = REG_6ns_area;
      break;
    case 5:
      *internal_power_per_bit = REG_5ns_int_power;
      *switch_power_per_bit = REG_5ns_switch_power;
      *leakage_power_per_bit = REG_5ns_leakage_power;
      *area_per_bit = REG_5ns_area;
      break;
    case 4:
      *internal_power_per_bit = REG_4ns_int_power;
      *switch_power_per_bit = REG_4ns_switch_power;
      *leakage_power_per_bit = REG_4ns_leakage_power;
      *area_per_bit = REG_4ns_area;
      break;
    case 3:
      *internal_power_per_bit = REG_3ns_int_power;
      *switch_power_per_bit = REG_3ns_switch_power;
      *leakage_power_per_bit = REG_3ns_leakage_power;
      *area_per_bit = REG_3ns_area;
      break;
    case 2:
      *internal_power_per_bit = REG_2ns_int_power;
      *switch_power_per_bit = REG_2ns_switch_power;
      *leakage_power_per_bit = REG_2ns_leakage_power;
      *area_per_bit = REG_2ns_area;
      break;
    case 1:
      *internal_power_per_bit = REG_1ns_int_power;
      *switch_power_per_bit = REG_1ns_switch_power;
      *leakage_power_per_bit = REG_1ns_leakage_power;
      *area_per_bit = REG_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power_per_bit = REG_6ns_int_power;
      *switch_power_per_bit = REG_6ns_switch_power;
      *leakage_power_per_bit = REG_6ns_leakage_power;
      *area_per_bit = REG_6ns_area;
      break;
  }
}
void getAdderPowerArea(float cycle_time,
                       float* internal_power,
                       float* switch_power,
                       float* leakage_power,
                       float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = ADD_10ns_int_power;
      *switch_power = ADD_10ns_switch_power;
      *leakage_power = ADD_10ns_leakage_power;
      *area = ADD_10ns_area;
      break;
    case 6:
      *internal_power = ADD_6ns_int_power;
      *switch_power = ADD_6ns_switch_power;
      *leakage_power = ADD_6ns_leakage_power;
      *area = ADD_6ns_area;
      break;
    case 5:
      *internal_power = ADD_5ns_int_power;
      *switch_power = ADD_5ns_switch_power;
      *leakage_power = ADD_5ns_leakage_power;
      *area = ADD_5ns_area;
      break;
    case 4:
      *internal_power = ADD_4ns_int_power;
      *switch_power = ADD_4ns_switch_power;
      *leakage_power = ADD_4ns_leakage_power;
      *area = ADD_4ns_area;
      break;
    case 3:
      *internal_power = ADD_3ns_int_power;
      *switch_power = ADD_3ns_switch_power;
      *leakage_power = ADD_3ns_leakage_power;
      *area = ADD_3ns_area;
      break;
    case 2:
      *internal_power = ADD_2ns_int_power;
      *switch_power = ADD_2ns_switch_power;
      *leakage_power = ADD_2ns_leakage_power;
      *area = ADD_2ns_area;
      break;
    case 1:
      *internal_power = ADD_1ns_int_power;
      *switch_power = ADD_1ns_switch_power;
      *leakage_power = ADD_1ns_leakage_power;
      *area = ADD_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = ADD_6ns_int_power;
      *switch_power = ADD_6ns_switch_power;
      *leakage_power = ADD_6ns_leakage_power;
      *area = ADD_6ns_area;
      break;
  }
}
void getMultiplierPowerArea(float cycle_time,
                            float* internal_power,
                            float* switch_power,
                            float* leakage_power,
                            float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = MUL_10ns_int_power;
      *switch_power = MUL_10ns_switch_power;
      *leakage_power = MUL_10ns_leakage_power;
      *area = MUL_10ns_area;
      break;
    case 6:
      *internal_power = MUL_6ns_int_power;
      *switch_power = MUL_6ns_switch_power;
      *leakage_power = MUL_6ns_leakage_power;
      *area = MUL_6ns_area;
      break;
    case 5:
      *internal_power = MUL_5ns_int_power;
      *switch_power = MUL_5ns_switch_power;
      *leakage_power = MUL_5ns_leakage_power;
      *area = MUL_5ns_area;
      break;
    case 4:
      *internal_power = MUL_4ns_int_power;
      *switch_power = MUL_4ns_switch_power;
      *leakage_power = MUL_4ns_leakage_power;
      *area = MUL_4ns_area;
      break;
    case 3:
      *internal_power = MUL_3ns_int_power;
      *switch_power = MUL_3ns_switch_power;
      *leakage_power = MUL_3ns_leakage_power;
      *area = MUL_3ns_area;
      break;
    case 2:
      *internal_power = MUL_2ns_int_power;
      *switch_power = MUL_2ns_switch_power;
      *leakage_power = MUL_2ns_leakage_power;
      *area = MUL_2ns_area;
      break;
    case 1:
      *internal_power = MUL_1ns_int_power;
      *switch_power = MUL_1ns_switch_power;
      *leakage_power = MUL_1ns_leakage_power;
      *area = MUL_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = MUL_6ns_int_power;
      *switch_power = MUL_6ns_switch_power;
      *leakage_power = MUL_6ns_leakage_power;
      *area = MUL_6ns_area;
      break;
  }
}
void getBitPowerArea(float cycle_time,
                     float* internal_power,
                     float* switch_power,
                     float* leakage_power,
                     float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = BIT_10ns_int_power;
      *switch_power = BIT_10ns_switch_power;
      *leakage_power = BIT_10ns_leakage_power;
      *area = BIT_10ns_area;
      break;
    case 6:
      *internal_power = BIT_6ns_int_power;
      *switch_power = BIT_6ns_switch_power;
      *leakage_power = BIT_6ns_leakage_power;
      *area = BIT_6ns_area;
      break;
    case 5:
      *internal_power = BIT_5ns_int_power;
      *switch_power = BIT_5ns_switch_power;
      *leakage_power = BIT_5ns_leakage_power;
      *area = BIT_5ns_area;
      break;
    case 4:
      *internal_power = BIT_4ns_int_power;
      *switch_power = BIT_4ns_switch_power;
      *leakage_power = BIT_4ns_leakage_power;
      *area = BIT_4ns_area;
      break;
    case 3:
      *internal_power = BIT_3ns_int_power;
      *switch_power = BIT_3ns_switch_power;
      *leakage_power = BIT_3ns_leakage_power;
      *area = BIT_3ns_area;
      break;
    case 2:
      *internal_power = BIT_2ns_int_power;
      *switch_power = BIT_2ns_switch_power;
      *leakage_power = BIT_2ns_leakage_power;
      *area = BIT_2ns_area;
      break;
    case 1:
      *internal_power = BIT_1ns_int_power;
      *switch_power = BIT_1ns_switch_power;
      *leakage_power = BIT_1ns_leakage_power;
      *area = BIT_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = BIT_6ns_int_power;
      *switch_power = BIT_6ns_switch_power;
      *leakage_power = BIT_6ns_leakage_power;
      *area = BIT_6ns_area;
      break;
  }
}
void getShifterPowerArea(float cycle_time,
                         float* internal_power,
                         float* switch_power,
                         float* leakage_power,
                         float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = SHIFTER_10ns_int_power;
      *switch_power = SHIFTER_10ns_switch_power;
      *leakage_power = SHIFTER_10ns_leakage_power;
      *area = SHIFTER_10ns_area;
      break;
    case 6:
      *internal_power = SHIFTER_6ns_int_power;
      *switch_power = SHIFTER_6ns_switch_power;
      *leakage_power = SHIFTER_6ns_leakage_power;
      *area = SHIFTER_6ns_area;
      break;
    case 5:
      *internal_power = SHIFTER_5ns_int_power;
      *switch_power = SHIFTER_5ns_switch_power;
      *leakage_power = SHIFTER_5ns_leakage_power;
      *area = SHIFTER_5ns_area;
      break;
    case 4:
      *internal_power = SHIFTER_4ns_int_power;
      *switch_power = SHIFTER_4ns_switch_power;
      *leakage_power = SHIFTER_4ns_leakage_power;
      *area = SHIFTER_4ns_area;
      break;
    case 3:
      *internal_power = SHIFTER_3ns_int_power;
      *switch_power = SHIFTER_3ns_switch_power;
      *leakage_power = SHIFTER_3ns_leakage_power;
      *area = SHIFTER_3ns_area;
      break;
    case 2:
      *internal_power = SHIFTER_2ns_int_power;
      *switch_power = SHIFTER_2ns_switch_power;
      *leakage_power = SHIFTER_2ns_leakage_power;
      *area = SHIFTER_2ns_area;
      break;
    case 1:
      *internal_power = SHIFTER_1ns_int_power;
      *switch_power = SHIFTER_1ns_switch_power;
      *leakage_power = SHIFTER_1ns_leakage_power;
      *area = SHIFTER_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = SHIFTER_6ns_int_power;
      *switch_power = SHIFTER_6ns_switch_power;
      *leakage_power = SHIFTER_6ns_leakage_power;
      *area = SHIFTER_6ns_area;
      break;
  }
}
void getSinglePrecisionFloatingPointAdderPowerArea(float cycle_time,
                                                   float* internal_power,
                                                   float* switch_power,
                                                   float* leakage_power,
                                                   float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = FP_SP_3STAGE_ADD_10ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_10ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_10ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_10ns_area;
      break;
    case 6:
      *internal_power = FP_SP_3STAGE_ADD_6ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_6ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_6ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_6ns_area;
      break;
    case 5:
      *internal_power = FP_SP_3STAGE_ADD_5ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_5ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_5ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_5ns_area;
      break;
    case 4:
      *internal_power = FP_SP_3STAGE_ADD_4ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_4ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_4ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_4ns_area;
      break;
    case 3:
      *internal_power = FP_SP_3STAGE_ADD_3ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_3ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_3ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_3ns_area;
      break;
    case 2:
      *internal_power = FP_SP_3STAGE_ADD_2ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_2ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_2ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_2ns_area;
      break;
    case 1:
      *internal_power = FP_SP_3STAGE_ADD_1ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_1ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_1ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = FP_SP_3STAGE_ADD_6ns_int_power;
      *switch_power = FP_SP_3STAGE_ADD_6ns_switch_power;
      *leakage_power = FP_SP_3STAGE_ADD_6ns_leakage_power;
      *area = FP_SP_3STAGE_ADD_6ns_area;
      break;
  }
}
void getDoublePrecisionFloatingPointAdderPowerArea(float cycle_time,
                                                   float* internal_power,
                                                   float* switch_power,
                                                   float* leakage_power,
                                                   float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = FP_DP_3STAGE_ADD_10ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_10ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_10ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_10ns_area;
      break;
    case 6:
      *internal_power = FP_DP_3STAGE_ADD_6ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_6ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_6ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_6ns_area;
      break;
    case 5:
      *internal_power = FP_DP_3STAGE_ADD_5ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_5ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_5ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_5ns_area;
      break;
    case 4:
      *internal_power = FP_DP_3STAGE_ADD_4ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_4ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_4ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_4ns_area;
      break;
    case 3:
      *internal_power = FP_DP_3STAGE_ADD_3ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_3ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_3ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_3ns_area;
      break;
    case 2:
      *internal_power = FP_DP_3STAGE_ADD_2ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_2ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_2ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_2ns_area;
      break;
    case 1:
      *internal_power = FP_DP_3STAGE_ADD_1ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_1ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_1ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = FP_DP_3STAGE_ADD_6ns_int_power;
      *switch_power = FP_DP_3STAGE_ADD_6ns_switch_power;
      *leakage_power = FP_DP_3STAGE_ADD_6ns_leakage_power;
      *area = FP_DP_3STAGE_ADD_6ns_area;
      break;
  }
}
void getSinglePrecisionFloatingPointMultiplierPowerArea(float cycle_time,
                                                        float* internal_power,
                                                        float* switch_power,
                                                        float* leakage_power,
                                                        float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = FP_SP_3STAGE_MUL_10ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_10ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_10ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_10ns_area;
      break;
    case 6:
      *internal_power = FP_SP_3STAGE_MUL_6ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_6ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_6ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_6ns_area;
      break;
    case 5:
      *internal_power = FP_SP_3STAGE_MUL_5ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_5ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_5ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_5ns_area;
      break;
    case 4:
      *internal_power = FP_SP_3STAGE_MUL_4ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_4ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_4ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_4ns_area;
      break;
    case 3:
      *internal_power = FP_SP_3STAGE_MUL_3ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_3ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_3ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_3ns_area;
      break;
    case 2:
      *internal_power = FP_SP_3STAGE_MUL_2ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_2ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_2ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_2ns_area;
      break;
    case 1:
      *internal_power = FP_SP_3STAGE_MUL_1ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_1ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_1ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = FP_SP_3STAGE_MUL_6ns_int_power;
      *switch_power = FP_SP_3STAGE_MUL_6ns_switch_power;
      *leakage_power = FP_SP_3STAGE_MUL_6ns_leakage_power;
      *area = FP_SP_3STAGE_MUL_6ns_area;
      break;
  }
}
void getDoublePrecisionFloatingPointMultiplierPowerArea(float cycle_time,
                                                        float* internal_power,
                                                        float* switch_power,
                                                        float* leakage_power,
                                                        float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = FP_DP_3STAGE_MUL_10ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_10ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_10ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_10ns_area;
      break;
    case 6:
      *internal_power = FP_DP_3STAGE_MUL_6ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_6ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_6ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_6ns_area;
      break;
    case 5:
      *internal_power = FP_DP_3STAGE_MUL_5ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_5ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_5ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_5ns_area;
      break;
    case 4:
      *internal_power = FP_DP_3STAGE_MUL_4ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_4ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_4ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_4ns_area;
      break;
    case 3:
      *internal_power = FP_DP_3STAGE_MUL_3ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_3ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_3ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_3ns_area;
      break;
    case 2:
      *internal_power = FP_DP_3STAGE_MUL_2ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_2ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_2ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_2ns_area;
      break;
    case 1:
      *internal_power = FP_DP_3STAGE_MUL_1ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_1ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_1ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_1ns_area;
      break;
    default:
      std::cerr << " Current power model supports accelerators running"
                << " at 1, 2, 3, 4, 5, 6, and 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 6ns power model instead." << std::endl;
      *internal_power = FP_DP_3STAGE_MUL_6ns_int_power;
      *switch_power = FP_DP_3STAGE_MUL_6ns_switch_power;
      *leakage_power = FP_DP_3STAGE_MUL_6ns_leakage_power;
      *area = FP_DP_3STAGE_MUL_6ns_area;
      break;
  }
}

void getTrigonometricFunctionPowerArea(float cycle_time,
                                      float* internal_power,
                                      float* switch_power,
                                      float* leakage_power,
                                      float* area) {
  switch ((int)cycle_time) {  // cycle_time in ns
    case 10:
      *internal_power = FP_3STAGE_TRIG_10ns_int_power;
      *switch_power   = FP_3STAGE_TRIG_10ns_switch_power;
      *leakage_power  = FP_3STAGE_TRIG_10ns_leakage_power;
      *area           = FP_3STAGE_TRIG_10ns_area;
      break;
    default:
      std::cerr << " Current power model supports trig functions running"
                << " at 10 ns. " << std::endl;
      std::cerr << " Cycle time: " << cycle_time << " is not supported yet."
                << " Use 10ns power model instead." << std::endl;
      *internal_power = 10*FP_3STAGE_TRIG_10ns_int_power;
      *switch_power   = 10*FP_3STAGE_TRIG_10ns_switch_power;
      *leakage_power  = 10*FP_3STAGE_TRIG_10ns_leakage_power;
      *area           = FP_3STAGE_TRIG_10ns_area;
      break;
  }
}

uca_org_t cactiWrapper(unsigned num_of_bytes, unsigned wordsize, unsigned num_ports) {
  int cache_size = num_of_bytes;
  int line_size = wordsize;  // in bytes
  if (wordsize < 4)          // minimum line size in cacti is 32-bit/4-byte
    line_size = 4;
  if (cache_size / line_size < 64)
    cache_size = line_size * 64;  // minimum scratchpad size: 64 words
  int associativity = 1;
  int rw_ports = num_ports;
  if (rw_ports == 0)
    rw_ports = 1;
  int excl_read_ports = 0;
  int excl_write_ports = 0;
  int single_ended_read_ports = 0;
  int search_ports = 0;
  int banks = 1;
  double tech_node = 45;  // in nm
  //# following three parameters are meaningful only for main memories
  int page_sz = 0;
  int burst_length = 8;
  int pre_width = 8;
  int output_width = wordsize * 8;
  //# to model special structure like branch target buffers, directory, etc.
  //# change the tag size parameter
  //# if you want cacti to calculate the tagbits, set the tag size to "default"
  int specific_tag = false;
  int tag_width = 0;
  int access_mode = 2;  // 0 normal, 1 seq, 2 fast
  int cache = 0;        // scratch ram 0 or cache 1
  int main_mem = 0;
  // assign weights for CACTI optimizations
  int obj_func_delay = 0;
  int obj_func_dynamic_power = 0;
  int obj_func_leakage_power = 100;
  int obj_func_area = 0;
  int obj_func_cycle_time = 0;
  // from CACTI example config...
  int dev_func_delay = 20;
  int dev_func_dynamic_power = 100000;
  int dev_func_leakage_power = 100000;
  int dev_func_area = 1000000;
  int dev_func_cycle_time = 1000000;

  int ed_ed2_none = 2;  // 0 - ED, 1 - ED^2, 2 - use weight and deviate
  int temp = 300;
  int wt = 0;  // 0 - default(search across everything), 1 - global, 2 - 5%
               // delay penalty, 3 - 10%, 4 - 20 %, 5 - 30%, 6 - low-swing
  int data_arr_ram_cell_tech_flavor_in =
      0;  // 0(itrs-hp) 1-itrs-lstp(low standby power)
  int data_arr_peri_global_tech_flavor_in = 0;  // 0(itrs-hp)
  int tag_arr_ram_cell_tech_flavor_in = 0;      // itrs-hp
  int tag_arr_peri_global_tech_flavor_in = 0;   // itrs-hp
  int interconnect_projection_type_in = 1;      // 0 - aggressive, 1 - normal
  int wire_inside_mat_type_in = 1;   // 2 - global, 0 - local, 1 - semi-global
  int wire_outside_mat_type_in = 1;  // 2 - global
  int REPEATERS_IN_HTREE_SEGMENTS_in =
      1;  // TODO for now only wires with repeaters are supported
  int VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in = 0;
  int BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in = 0;
  int force_wiretype = 1;
  int wiretype = 30;
  int force_config = 0;
  int ndwl = 1;
  int ndbl = 1;
  int nspd = 0;
  int ndcm = 1;
  int ndsam1 = 0;
  int ndsam2 = 0;
  int ecc = 0;
  return cacti_interface(cache_size,
                         line_size,
                         associativity,
                         rw_ports,
                         excl_read_ports,
                         excl_write_ports,
                         single_ended_read_ports,
                         search_ports,
                         banks,
                         tech_node,  // in nm
                         output_width,
                         specific_tag,
                         tag_width,
                         access_mode,  // 0 normal, 1 seq, 2 fast
                         cache,        // scratch ram or cache
                         main_mem,
                         obj_func_delay,
                         obj_func_dynamic_power,
                         obj_func_leakage_power,
                         obj_func_area,
                         obj_func_cycle_time,
                         dev_func_delay,
                         dev_func_dynamic_power,
                         dev_func_leakage_power,
                         dev_func_area,
                         dev_func_cycle_time,
                         ed_ed2_none,
                         temp,
                         wt,
                         data_arr_ram_cell_tech_flavor_in,
                         data_arr_peri_global_tech_flavor_in,
                         tag_arr_ram_cell_tech_flavor_in,
                         tag_arr_peri_global_tech_flavor_in,
                         interconnect_projection_type_in,
                         wire_inside_mat_type_in,
                         wire_outside_mat_type_in,
                         REPEATERS_IN_HTREE_SEGMENTS_in,
                         VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in,
                         BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in,
                         page_sz,
                         burst_length,
                         pre_width,
                         force_wiretype,
                         wiretype,
                         force_config,
                         ndwl,
                         ndbl,
                         nspd,
                         ndcm,
                         ndsam1,
                         ndsam2,
                         ecc);
}
uca_org_t cactiWrapper(unsigned num_of_bytes, unsigned wordsize, unsigned read_ports, unsigned write_ports) {
  int cache_size = num_of_bytes;
  int line_size = wordsize;  // in bytes
  if (wordsize < 4)          // minimum line size in cacti is 32-bit/4-byte
    line_size = 4;
  if (cache_size / line_size < 64)
    cache_size = line_size * 64;  // minimum scratchpad size: 64 words
  int associativity = 1;
  int excl_read_ports,excl_write_ports;
  int rw_ports = std::min(read_ports,write_ports);
  if (rw_ports == 0)
    rw_ports = 1;
  if(read_ports>write_ports){
     excl_read_ports = read_ports-write_ports;
     excl_write_ports = 0;}
  else {
     excl_read_ports = 0;
     excl_write_ports = write_ports-read_ports;}
  int single_ended_read_ports = 0;
  int search_ports = 0;
  int banks = 1;
  double tech_node = 45;  // in nm
  //# following three parameters are meaningful only for main memories
  int page_sz = 0;
  int burst_length = 8;
  int pre_width = 8;
  int output_width = wordsize * 8;
  //# to model special structure like branch target buffers, directory, etc.
  //# change the tag size parameter
  //# if you want cacti to calculate the tagbits, set the tag size to "default"
  int specific_tag = false;
  int tag_width = 0;
  int access_mode = 2;  // 0 normal, 1 seq, 2 fast
  int cache = 0;        // scratch ram 0 or cache 1
  int main_mem = 0;
  // assign weights for CACTI optimizations
  int obj_func_delay = 0;
  int obj_func_dynamic_power = 0;
  int obj_func_leakage_power = 100;
  int obj_func_area = 0;
  int obj_func_cycle_time = 0;
  // from CACTI example config...
  int dev_func_delay = 20;
  int dev_func_dynamic_power = 100000;
  int dev_func_leakage_power = 100000;
  int dev_func_area = 1000000;
  int dev_func_cycle_time = 1000000;

  int ed_ed2_none = 2;  // 0 - ED, 1 - ED^2, 2 - use weight and deviate
  int temp = 300;
  int wt = 0;  // 0 - default(search across everything), 1 - global, 2 - 5%
               // delay penalty, 3 - 10%, 4 - 20 %, 5 - 30%, 6 - low-swing
  int data_arr_ram_cell_tech_flavor_in =
      0;  // 0(itrs-hp) 1-itrs-lstp(low standby power)
  int data_arr_peri_global_tech_flavor_in = 0;  // 0(itrs-hp)
  int tag_arr_ram_cell_tech_flavor_in = 0;      // itrs-hp
  int tag_arr_peri_global_tech_flavor_in = 0;   // itrs-hp
  int interconnect_projection_type_in = 1;      // 0 - aggressive, 1 - normal
  int wire_inside_mat_type_in = 1;   // 2 - global, 0 - local, 1 - semi-global
  int wire_outside_mat_type_in = 1;  // 2 - global
  int REPEATERS_IN_HTREE_SEGMENTS_in =
      1;  // TODO for now only wires with repeaters are supported
  int VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in = 0;
  int BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in = 0;
  int force_wiretype = 1;
  int wiretype = 30;
  int force_config = 0;
  int ndwl = 1;
  int ndbl = 1;
  int nspd = 0;
  int ndcm = 1;
  int ndsam1 = 0;
  int ndsam2 = 0;
  int ecc = 0;
  return cacti_interface(cache_size,
                         line_size,
                         associativity,
                         rw_ports,
                         excl_read_ports,
                         excl_write_ports,
                         single_ended_read_ports,
                         search_ports,
                         banks,
                         tech_node,  // in nm
                         output_width,
                         specific_tag,
                         tag_width,
                         access_mode,  // 0 normal, 1 seq, 2 fast
                         cache,        // scratch ram or cache
                         main_mem,
                         obj_func_delay,
                         obj_func_dynamic_power,
                         obj_func_leakage_power,
                         obj_func_area,
                         obj_func_cycle_time,
                         dev_func_delay,
                         dev_func_dynamic_power,
                         dev_func_leakage_power,
                         dev_func_area,
                         dev_func_cycle_time,
                         ed_ed2_none,
                         temp,
                         wt,
                         data_arr_ram_cell_tech_flavor_in,
                         data_arr_peri_global_tech_flavor_in,
                         tag_arr_ram_cell_tech_flavor_in,
                         tag_arr_peri_global_tech_flavor_in,
                         interconnect_projection_type_in,
                         wire_inside_mat_type_in,
                         wire_outside_mat_type_in,
                         REPEATERS_IN_HTREE_SEGMENTS_in,
                         VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in,
                         BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in,
                         page_sz,
                         burst_length,
                         pre_width,
                         force_wiretype,
                         wiretype,
                         force_config,
                         ndwl,
                         ndbl,
                         nspd,
                         ndcm,
                         ndsam1,
                         ndsam2,
                         ecc);
}

uca_org_t multiWrapper(unsigned num_of_bytes, unsigned wordsize, unsigned read_ports, unsigned write_ports, std::string& policy) {
  uca_org_t cacti_result=cactiWrapper(num_of_bytes,wordsize,read_ports,write_ports);
  unsigned long area_ntx[12][6]={{ 10000,17000,	31000,	56000,	112000,	216000},{
                    57000	,89000	,141000,	243000,	454000,	807000},{
                    190000,	320000,	523000,	781000,	1310000,	2200000},{
                    650000,	1080000,	1720000,	3260000,	5770000,	7970000},{
                    25909	,40454,	64090,	110454,	206363,	366818},{
                    130000,	200000	,330000,	570000,	1050000,	1960000},{
                    416000,	640000	,1280000,	1960000,	3460000,	5760000},{
                    1372800,	2112000,	4224000,	6468000,	11418000,	19008000},{
                    57000	,89000	,141000,	243000,	454000,	807000},{
                    190000,	320000,	523000,	781000,	1310000,	2200000},{
                    650000,	1080000,	1720000,	3260000,	5770000,	7970000},{
                    2340000,	3888000,	6192000,	11736000,	20772000,	28692000}};
  double power_ntx[12][6]= {{0.0001,	0.00024,	0.00033,	0.00045,	0.00071,	0.00113},{
                      0.00045,	0.0006,	0.00086,	0.00147,	0.00236, 0.00383},{
                      0.0006, 0.00106,	0.00171,	0.0033,	0.00543,	0.0108},{
                      0.00125,	0.00218,	0.00346,	0.00653,	0.0118,	0.0219},{
                      0.0003,	0.0004,	0.00057333,	0.00098,	0.0015733,	0.0025533},{
                      0.00187,	0.00225,	0.00285,	0.00414,	0.00601,	0.00873},{
                      0.002805,	0.003375,	0.004275,	0.00621,	0.009015,	0.013095},{
                      0.00631125,	0.00759375,	0.009618,	0.0139725,	0.02028,	0.02946375},{
                      0.00045,	0.0006,	0.00086, 0.00147,	0.00236,	0.00383},{
                      0.0006,	0.00106,	0.00171,	0.0033, 0.00543,	0.0108},{
                      0.00125,	0.00218,	0.00346,	0.00653,	0.0118,	0.0219},{
                      0.0028125,	0.004905,	0.007785,	0.0146925,	0.02655,	0.049275}};
  int i,j;
  i=j=0;
  switch(num_of_bytes)
      {
      case 1024:
                  {   i=0;
                  break;}    
        case 2048:
                {  i=1;
                    break;}
        case 4096:
                {   i=2;
                    break;}

        case 8192:
                {   i=3;
                    break;}
        case 16384:
                  {   i=4;
                    break;}
        case 32768:
                  {   i=5;
                    break;}
    
        default:
        break;
    }
    switch(write_ports)
      {
      case 1:
                  {   j=4*std::log2(read_ports)-4;
                  break;}    
        case 2:
                {  j=4*std::log2(read_ports)-3;
                    break;}
        case 4:
                {   j=4*std::log2(read_ports)-2;
                    break;}

        case 8:
                {  j=4*std::log2(read_ports)-1;
                    break;}   
        default:
        break;
    }
  // cacti_result.power.readOp.dynamic = 0;
  // cacti_result.power.writeOp.dynamic = 0;
  //  std::cout<<"cache_size "<<cache_size<<std::endl;
  // std::cout<<"line_size "<<line_size<<std::endl;
  // std::cout<<"associativity "<<associativity<<std::endl;
  // std::cout<<"rw_ports "<<rw_ports<<std::endl;
  // std::cout<<"excl_read_ports "<<excl_read_ports<<std::endl;
  // std::cout<<"excl_write_ports "<<excl_write_ports<<std::endl;
  // std::cout<<"single_ended_read_ports "<<single_ended_read_ports<<std::endl;
  // std::cout<<"banks "<<banks<<std::endl;
  cacti_result.power.readOp.leakage = power_ntx[j][i];
  // cacti_result.area = area_ntx[j][i];
  int cache_size = num_of_bytes;
  return cacti_result;
}
