"""
	common constants
"""
const dt = 1.0                # [s]
const sensing_range = 1000.0  # [m]
const min_speed = 40.0        # [m/s]
const max_speed = 60.0        # [m/s]
const collision_range = 150.0 # [m]

const rho_dim   = 50
const v_dim     = 3


const rho_min = 0.
const rho_max = 2000.0 # [m]

const theta_min = 0
const theta_max = 2 * pi

const phi_min = 0
const phi_max = 2 * pi

const v_min = min_speed
const v_max = max_speed

const discrete_rho = linspace(rho_min, rho_max, rho_dim)
# It is important to guarantee the appearance of π in angle discretizations!
const discrete_theta = deg2rad.(vcat(linspace(0, 30, 8), linspace(35, 90, 8), linspace(100, 260, 13), linspace(270, 325, 8), linspace(330, 360, 8)))
const discrete_phi   = deg2rad.(vcat(linspace(0, 80, 7), linspace(90, 146, 10), linspace(150, 210, 17), linspace(214, 270, 10), linspace(280, 360, 7)))
const theta_dim = length(discrete_theta)
const phi_dim   = length(discrete_phi)
const discrete_v = linspace(v_min, v_max, v_dim)

"""
	sigma-point sampling
	non-cooperative
	increase noise, more robust MDP
"""
const sigma_v = 5.0
const sigma_alert_deg = 5.0
const sigma_coc_deg = 3.0
const sigma_theta_deg = 3.0
const sigma_phi_deg = 7.0

const sigma_alert = deg2rad(sigma_alert_deg)
const sigma_coc = deg2rad(sigma_coc_deg)
const sigma_theta = deg2rad(sigma_theta_deg)
const sigma_phi = deg2rad(sigma_phi_deg)

const sigma_str = "_Sigma_v_" * string(sigma_v) * 
				  "_alert_" * string(sigma_alert_deg) *
				  "_theta_" * string(sigma_theta_deg) *
				  "_phi_" * string(sigma_phi_deg) * 
				  "_coc_" * string(sigma_coc_deg)

# const sigmas = [0 sigma_alert -sigma_alert       0        0       0        0;
# 				0           0            0 sigma_v -sigma_v       0        0;
# 				0           0            0       0        0 sigma_v -sigma_v]

# const sigmas = [0 sigma_alert -sigma_alert         0          0       0        0       0        0;
# 				0           0            0 sigma_phi -sigma_phi       0        0       0        0;
# 				0           0            0         0          0 sigma_v -sigma_v       0        0;
# 				0           0            0         0          0       0        0 sigma_v -sigma_v]

const sigmas = [0 sigma_alert -sigma_alert         0          0           0            0       0        0       0        0;
				0           0            0 sigma_phi -sigma_phi           0            0       0        0       0        0;
				0           0            0         0          0 sigma_theta -sigma_theta       0        0       0        0;
				0           0            0         0          0           0            0 sigma_v -sigma_v       0        0;
				0           0            0         0          0           0            0       0        0 sigma_v -sigma_v]

const turn_rate_sigma_ind = 1
const phi_sigma_ind       = 2
const theta_sigma_ind     = 3
const v_own_sigma_ind     = 4
const v_int_sigma_ind     = 5



# const sigmas = [0 sigma_phi -sigma_phi           0            0;
# 				0         0          0 sigma_theta -sigma_theta]

# const phi_sigma_ind       = 1
# const theta_sigma_ind     = 2


const num_sigmas = size(sigmas, 2)

# need to increase the weight of the first sigma that executes the exact action
const actual_weight = 0.7;
const weights = [actual_weight; (1 - actual_weight) / (num_sigmas - 1) * ones(num_sigmas - 1)]



