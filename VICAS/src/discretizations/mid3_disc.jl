"""
	common constants
"""
const dt = 1.0                # [s]
const sensing_range = 1000.0  # [m]
const min_speed = 40.0        # [m/s]
const max_speed = 60.0        # [m/s]
const collision_range = 150.0 # [m]

const rho_dim   = 65
const theta_dim = 25
const phi_dim   = 25
const v_dim     = 3


const rho_min = 0.
const rho_max = 1800.0 # [m]

const theta_min = 0
const theta_max = 2 * pi

const phi_min = 0
const phi_max = 2 * pi

const v_min = min_speed
const v_max = max_speed

const discrete_rho = linspace(rho_min, rho_max, rho_dim)
# It is important to guarantee the appearance of π in angle discretizations!
const discrete_theta = linspace(theta_min, theta_max, theta_dim)
const discrete_phi   = linspace(phi_min, phi_max, phi_dim)
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

# sigmas are tunable

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
const actual_weight = 0.5;
const weights = [actual_weight; (1 - actual_weight) / (num_sigmas - 1) * ones(num_sigmas - 1)]



