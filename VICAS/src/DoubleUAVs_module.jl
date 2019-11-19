module DoubleUAVs_module

push!(LOAD_PATH, "./")
disc = "mid3"
include("./discretizations/" * disc * "_disc.jl")
include("./discretizations/common.jl")

using Distributions

export states, advisories, weights, advisory_to_action_dict, disc, state_dim
export sigma_str, sigma_v, sigma_alert_deg, sigma_coc_deg
export terminal_state, terminal_state_var
export num_states, num_actions, num_sigmas
export get_reward, get_next_state
export is_terminal
export norm_angle
export rho_max, sensing_range, v_min, v_max, collision_range
export rho_ind, theta_ind, phi_ind, v_own_ind, v_int_ind
export grid_states
export actual_weight

"""
	tells if the state is terminal
"""
function is_terminal(state::Vector{Float64})
	rho = state[rho_ind]
	if rho > rho_max
		return true
	else
		return false
	end
end

"""
	keeps the angles in [0, 2*pi]
"""
function norm_angle(angle::Float64)
	return ((angle % (2 * pi)) + 2 * pi) % (2 * pi)
end

"""
	state update
	assuming the intruder keeps the straight path
"""
function get_next_state(state::Vector{Float64}, advisory::Symbol; isigma::Int64=1, if_sample::Bool=false)
	if is_terminal(state)
		return terminal_state
	else
		if if_sample == false
			rho = state[rho_ind]
			theta = state[theta_ind] + sigmas[theta_sigma_ind, isigma]
			phi = state[phi_ind] + sigmas[phi_sigma_ind, isigma]
			v_own = state[v_own_ind]
			v_int = state[v_int_ind]
	
			if advisory == :COC
				turn_rate = sign(sigmas[turn_rate_sigma_ind, isigma]) * sigma_coc
			else
				turn_rate = deg2rad(advisory_to_action_dict[advisory]) + sigmas[turn_rate_sigma_ind, isigma]
			end
			turn_angle = turn_rate * dt
	
			v_own += sigmas[v_own_sigma_ind, isigma]
			v_int += sigmas[v_int_sigma_ind, isigma]

		else # if_sample == true
			# rho = state[rho_ind]
			# theta = state[theta_ind]
			# phi = state[phi_ind] + clamp(rand(Normal(0, sigma_phi)), -sigma_phi, sigma_phi)
			# v_own = state[v_own_ind]
			# v_int = state[v_int_ind]

			# if advisory == :COC
			# 	turn_rate = clamp(rand(Normal(0, sigma_coc)), -sigma_coc, sigma_coc)
			# else
			# 	turn_rate = deg2rad(advisory_to_action_dict[advisory]) + 
			# 		clamp(rand(Normal(0, sigma_alert)), -sigma_alert, sigma_alert)
			# end
			# turn_angle = turn_rate * dt
	
			# v_own += clamp(rand(Normal(0, sigma_v)), -sigma_v, sigma_v)
			# v_int += clamp(rand(Normal(0, sigma_v)), -sigma_v, sigma_v)
			nothing
		end
		###################################################################
		# Using the body fixed frame origined at the ownship. 
		# The positive x-direction is coincide with the heading of the ownship.
		# desperate

		x_own = 0.
		y_own = 0.
		x_int = rho * cos(theta)
		y_int = rho * sin(theta)

		next_x_own = x_own + v_own * cos(turn_angle) * dt
		next_y_own = y_own + v_own * sin(turn_angle) * dt
		next_x_int = x_int + v_int * cos(phi) * dt
		next_y_int = y_int + v_int * sin(phi) * dt
	
		next_rho = sqrt((next_x_own - next_x_int)^2 + (next_y_own - next_y_int)^2)

		if next_rho > rho_max
			return terminal_state
		end
		
		temp_var = norm_angle(atan2(next_y_int - next_y_own, next_x_int - next_x_own))
		next_theta = norm_angle(temp_var - turn_angle)
		next_phi = norm_angle(phi - turn_angle)
	
		next_state = Vector{Float64}(state_dim)
		next_state[rho_ind] = next_rho
		next_state[theta_ind] = next_theta
		next_state[phi_ind] = next_phi
		next_state[v_own_ind] = v_own
		next_state[v_int_ind] = v_int
	
		return next_state
	end
end

"""
	reward function
"""
function get_reward(state::Vector{Float64}, advisory::Symbol; 
	penalty_action=0.02, 
	penalty_closeness=10.0, 
	penalty_nmac=1000.0, 
	penalty_conflict=1.0)
	
	possible_max_rew = 0.0
	possible_min_rew = - penalty_conflict - penalty_action * 10.^2 - penalty_nmac - penalty_closeness * e

	action = advisory_to_action_dict[advisory] # [deg]
	reward = 0.0
	if advisory == :COC
		action = 0.0
	else
		reward -= penalty_conflict
	end

	if is_terminal(state)
		reward -= penalty_action * action^2
		reward /= (possible_max_rew - possible_min_rew)
		if abs(reward) > 1
			println(reward)
		end
		return reward
	else
		reward -= penalty_action * action^2
		rho = state[rho_ind]
		if rho < collision_range
			reward -= penalty_nmac
		end
		reward -= penalty_closeness * exp(-(rho - collision_range) / collision_range)
	end

	# scale the magnitude of every step reward between 0 and 1
	reward = reward / (possible_max_rew - possible_min_rew)
	if abs(reward) > 1
		println(reward)
	end
	return reward
end


end # module



