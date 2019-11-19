"""
	common constants
"""
const dt = 1.0                # [s]
const sensing_range = 1000.0  # [m]
const min_speed = 20.0        # [m/s]
const max_speed = 50.0        # [m/s]
const collision_range = 150.0 # [m]

const rho_dim   = 100
const theta_dim = 20
const phi_dim   = 20
const v_dim     = 5


const rho_min = 0.
const rho_max = 2000.0 # [m]

const theta_min = 0
const theta_max = 2 * pi

const phi_min = 0
const phi_max = 2 * pi

const v_min = min_speed
const v_max = max_speed


const discrete_rho = linspace(rho_min, rho_max, rho_dim)
const discrete_theta = linspace(theta_min, theta_max, theta_dim)
const discrete_phi   = linspace(phi_min, phi_max, phi_dim)
const discrete_v = linspace(v_min, v_max, v_dim)


"""
	sigma-point sampling
	non-cooperative
	increase noise, more robust MDP
"""
const sigma_v = 10.0
const sigma_alert_deg = 30.0
const sigma_coc_deg = 35.0

const sigma_alert = deg2rad(sigma_alert_deg)
const sigma_coc = deg2rad(sigma_coc_deg)

const sigma_str = "_Sigma_v_" * string(sigma_v) * 
				  "_alert_" * string(sigma_alert_deg) *
				  "_coc_" * string(sigma_coc_deg)

const sigmas = [0 sigma_alert -sigma_alert       0        0       0        0;
				0           0            0 sigma_v -sigma_v       0        0;
				0           0            0       0        0 sigma_v -sigma_v]
const num_sigmas = size(sigmas, 2)

const weights = [1/3; 2/(3 * (num_sigmas - 1)) * ones(num_sigmas - 1)]

const turn_rate_sigma_ind = 1
const v_own_sigma_ind     = 2
const v_int_sigma_ind     = 3