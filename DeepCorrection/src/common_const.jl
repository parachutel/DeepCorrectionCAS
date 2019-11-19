using ArgParse

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table s begin
		"--dest_range"
			help = "Distance between the initial position and the destination of the ego agent"
			arg_type = Float64
			default = 10000.0 # [m]
		"--sensing_range"
			help = "Radius of the alerting perimeter"
			arg_type = Float64
			default = 1000.0 # [m]

		"--penalty_action" # max =  - 100 * pen
			help = "Reward function parameter: penalty on action"
			arg_type = Float64
			default = 2.0
		"--penalty_conflict" # max =  - pen
			help = "Reward function parameter: penalty on conflict"
			arg_type = Float64
			default = 50.0
		"--penalty_closeness" # max = - pen (on NMAC) or - pen * 2.71 (on collision)
			help = "Reward function parameter: penalty on closeness between agents"
			arg_type = Float64
			default = 500.0
		"--penalty_nmac" # max = - pen
			help = "Reward function parameter: penalty on near mid-air collisions"
			arg_type = Float64
			default = 7000.0
		
		"--penalty_deviation" # max = - 180 * pen
			help = "Reward function parameter: penalty on direction deviation from destination"
			arg_type = Float64
			default = 0.8
		"--penalty_digression" # max = 
			help = "Reward function parameter: penalty on digression from destination"
			arg_type = Float64
			default = 50.0
		"--reward_destination" # max = rew
			help = "Reward function parameter: reward on reaching destination"
			arg_type = Float64
			default = 0.0
	end
	return parse_args(s)
end

parsed_args = parse_commandline()

const SPAWN_RATE_SET = [25, 35, 45]
const CONST_NUM_AC_SET = [1]
# SPAWN_RATE = rand(SPAWN_RATE_SET)
const BEARING_DIST =   [[0.1047 0.0489];
   						[0.3142 0.0474];
   						[0.5236 0.0474];
   						[0.7330 0.0459];
   						[0.9425 0.0425];
   						[1.1519 0.0403];
   						[1.3614 0.0365];
   						[1.5708 0.0333];
   						[1.7802 0.0304];
   						[1.9897 0.0269];
   						[2.1991 0.0237];
   						[2.4086 0.0210];
   						[2.6180 0.0194];
   						[2.8274 0.0183];
   						[3.0369 0.0181];
   						[3.2463 0.0181];
   						[3.4558 0.0183];
   						[3.6652 0.0194];
   						[3.8746 0.0210];
   						[4.0841 0.0237];
   						[4.2935 0.0269];
   						[4.5029 0.0304];
   						[4.7124 0.0333];
   						[4.9218 0.0365];
   						[5.1313 0.0403];
   						[5.3407 0.0425];
   						[5.5501 0.0459];
   						[5.7596 0.0474];
   						[5.9690 0.0474];
   						[6.1785 0.0489];]

const DT = 1.0 # [s]
const V_MIN = 50.0 # [m/s]
const V_MAX = 50.0 # [m/s], currently using single value for speed
const AIRSPACE_DIM = 1500  # [m]
const STATE_DIM = 5
const AUG_STATE_DIM = 3

const DEST_CRITERION = 120.0 # [m]
const NMAC_RANGE = 150.0 # [m]
const NUM_SECTIONS = 4
const RELAX_FACTOR = 1.2 # for train
# const RELAX_FACTOR = 1.0 # for resTime evaluation

const TERM_VAR = terminal_state_var # imported from CAS_QMDP

# state ind
# s = [ρ, θ, ϕ, v_own, v_int]
const IND_rho   = 1
const IND_theta = 2
const IND_phi   = 3
const IND_v_own = 4
const IND_v_int = 5
const IND_deviation = STATE_DIM * NUM_SECTIONS + 1
const IND_dist_to_dest = STATE_DIM * NUM_SECTIONS + 2
const IND_prev_dist_to_dest = STATE_DIM * NUM_SECTIONS + 3

const SENSING_RANGE = parsed_args["sensing_range"]
const DEST_RANGE = parsed_args["dest_range"]

# Reward weights for the correction network
const PEN_ACTION = parsed_args["penalty_action"]
const PEN_CLOSENESS = parsed_args["penalty_closeness"]
const PEN_NMAC = parsed_args["penalty_nmac"]
const PEN_CONFLICT = parsed_args["penalty_conflict"]
const PEN_DEVIATION = parsed_args["penalty_deviation"]
const PEN_DIGRESSION = parsed_args["penalty_digression"]
const REW_DESTINATION = parsed_args["reward_destination"]

# Leaving out the huge positive reward from reaching destination:
const POSSIBLE_MAX_REW = V_MAX * DT * PEN_DIGRESSION 
# Keeping the huge positive reward from reaching destination:
# const POSSIBLE_MAX_REW = REW_DESTINATION + V_MAX * DT * PEN_DIGRESSION

# Leaving out the huge negative reward from NMAC:
# const POSSIBLE_MIN_REW = - PEN_CONFLICT - PEN_ACTION * 10^2 - PEN_CLOSENESS * e - 180 * PEN_DEVIATION - V_MAX * DT * PEN_DIGRESSION
# Keeping the huge negative reward from NMAC:
const POSSIBLE_MIN_REW = - PEN_CONFLICT - PEN_ACTION * 10^2 - PEN_NMAC - PEN_CLOSENESS * e - 180 * PEN_DEVIATION - V_MAX * DT * PEN_DIGRESSION

const LOW_FI_POLICY = "mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0"
const LOW_FI_POLICY_FILENAME = 
	"../../VICAS/policies/" * LOW_FI_POLICY * ".jld"

# Step reward bounds from VICAS
# These values need to match those used to train VICAS
# Read them from LOW_FI_POLICY
const penalty_action_VICAS = 3.0
const penalty_conflict_VICAS = 50.0
const penalty_closeness_VICAS = 500.0
const penalty_nmac_VICAS = 7000.0

const POSSIBLE_MAX_REW_VICAS = 0.0
const POSSIBLE_MIN_REW_VICAS = - penalty_conflict_VICAS - penalty_action_VICAS * 10.^2 - penalty_nmac_VICAS - penalty_closeness_VICAS * e


