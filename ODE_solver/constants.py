# SIMULATION CONSTANTS:

PI = 3.141592653589793
MASS = 2.6e-4  # from the hummingbird paper
WING_LENGTH = 0.07  # meters
GYRATION_RADIUS = 0.6 * WING_LENGTH  # we use this for moment of inertia
MoI = MASS * GYRATION_RADIUS ** 2
AIR_DENSITY = 1.2  # From Arion's simulatio
WING_AREA = 0.5 * WING_LENGTH * (0.5 * WING_LENGTH) * PI  # 1/2 ellipse with minor radios ~ 1/2 major = length/2
# drag coefficients from whitney & wood (JFM 2010):
C_D_MAX = 3.4
C_D_0 = 0.4
C_L_MAX = 1.8
ZERO_CROSSING = 1
RADIAN45 = PI / 4
RADIAN135 = 3 * RADIAN45

# ENVIRONMENT CONSTANTS:

MIN_TORQUE = -1
MAX_TORQUE = 1
MIN_PHI = 0
MAX_PHI = PI
HISTORY_SIZE = 10
STEP_TIME = 0.01
STEPS_PER_EPISODE = 20
MAX_APPROX_TORQUE = 0.02
ACTION_ERROR_PERCENTAGE = 0.05
INITIAL_PHI0 = 0
INITIAL_PHI_DOT0 = 2e-4
LIFT_WEIGHT = 1
PHI_WEIGHT = 1
TORQUE_WEIGHT = 80
POWER_WEIGHT = 0

# ENVIRONMENT KEYWORDS:

REWARD_KEY = 'reward'
LIFT_REWARD_KEY = 'lift_reward'
ANGLE_REWARD_KEY = 'angle_reward'
TORQUE_REWARD_KEY = 'torque_reward'
POWER_REWARD_KEY = 'power_reward'
TOTAL_REWARD_KEY = 'total'
PHI_KEY = 'phi'
STATE_KEY = 'state'
PHI_DOT_KEY = 'phi_dot'
LIFT_FORCE_KEY = 'lift_force'
TIME_KEY = 'time'
ACTION_KEY = 'action'
ITERATION_KEY = 'iter'
TORQUE_KEY = 'torque'
STEP_SIMULATION_OUT_KEY = 'curr_simulation_output'
