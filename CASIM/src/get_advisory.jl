# using CAS_module
# include("../CAS_Dependencies/load_network.jl")

"""
	update the advisory of an Aircraft object with ACAS_Xu_Network and Nearest_Intruder_Tracker
"""
function get_advisory(ac::Aircraft, ::ACAS_Xu_Network, ::Nearest_Intruder_Tracker)
	if isempty(ac.sensor.obs) == false
		obs = ac.sensor.obs[1] # ac.sensor.obs is an array
		Q_values = evaluate_network(acas_xu_network_data, 
			[obs.rho, obs.theta, obs.psi, obs.sos, obs.soi, obs.tau, obs.pRA])
		ind = indmin(Q_values)
		ac.advisory = ac.cas.Ind_to_Advisory_dict[ind]
	else 
		ac.advisory = :COC
	end
end

"""
	update the advisory of an Aircraft object with ACAS_Xu_Network and Multi_Threat_Tracker
"""
function get_advisory(ac::Aircraft, ::ACAS_Xu_Network, ::Multi_Threat_Tracker)
	if isempty(ac.sensor.obs) == false
		# Utility fusion using max-min
		min_Q = []
		min_ind = []
		for obs in ac.sensor.obs
			# Q_values are actually cost (not based on reward)
			Q_values = evaluate_network(acas_xu_network_data, [obs.rho, obs.theta, 
								obs.psi, obs.sos, obs.soi, obs.tau, obs.pRA])
			push!(min_Q, minimum(Q_values))
			push!(min_ind, indmin(Q_values))
		end
		max_min_ind = indmax(min_Q)
		ind = min_ind[max_min_ind]
		ac.advisory = ac.cas.Ind_to_Advisory_dict[ind]
	else
		ac.advisory = :COC
	end
end


"""
	update the advisory of an Aircraft object with VICAS and Nearest_Intruder_Tracker
"""
function get_advisory(ac::Aircraft, ::VICAS, ::Nearest_Intruder_Tracker)
	if isempty(ac.sensor.obs) == false
		obs = ac.sensor.obs[1] # ac.sensor.obs is an array
		# for polar
		state = [ft2m(obs.rho), norm_angle(obs.theta), norm_angle(obs.psi), ft2m(obs.sos), ft2m(obs.soi)]
		# # for xy
		# x_rel = ft2m(obs.rho) * cos(obs.theta)
		# y_rel = ft2m(obs.rho) * sin(obs.theta)
		# state = [x_rel, y_rel, norm_angle(obs.psi), ft2m(obs.sos), ft2m(obs.soi)]
		Q_values = get_qvals(alphas, state)
		ac.advisory = ac.cas.Ind_to_Advisory_dict[indmax(Q_values)]
	else 
		ac.advisory = :COC
	end
end

"""
	update the advisory of an Aircraft object with corrected VICAS
"""
function get_advisory(ac::Aircraft, full_state::Vector{Float64}, lowfi_state::Vector{Float64})
	if isempty(ac.sensor.obs) == false
		# obs = ac.sensor.obs[1] # ac.sensor.obs is an array
		# lowfi_state = [ft2m(obs.rho), norm_angle(obs.theta), norm_angle(obs.psi), ft2m(obs.sos), ft2m(obs.soi)]
		ac.advisory = get_action(policy, full_state, lowfi_state)
	else 
		ac.advisory = :COC
	end
end


"""
	update the advisory of an Aircraft object with VICAS and Multi_Threat_Tracker
"""
function get_advisory(ac::Aircraft, ::VICAS, ::Multi_Threat_Tracker)
	if isempty(ac.sensor.obs) == false
		# Utility fusion using max-min
		q_matrix = Matrix{Float64}(6, 0)
		for obs in ac.sensor.obs
			# for polar
			state = [ft2m(obs.rho), norm_angle(obs.theta), norm_angle(obs.psi), 
						ft2m(obs.sos), ft2m(obs.soi)]
			# # for xy
			# x_rel = ft2m(obs.rho) * cos(obs.theta)
			# y_rel = ft2m(obs.rho) * sin(obs.theta)
			# state = [x_rel, y_rel, norm_angle(obs.psi), ft2m(obs.sos), ft2m(obs.soi)]

			qvals = get_qvals(alphas, state)
     		q_matrix = hcat(q_matrix, qvals)
		end
		min_qvals = minimum(q_matrix, 2)
		the_ind = indmax(min_qvals)
		ac.advisory = ac.cas.Ind_to_Advisory_dict[the_ind]
	else
		ac.advisory = :COC
	end
end

"""
	update the advisory of an Aircraft object with VICAS and Multi_Threat_Tracker
"""
function get_advisory(ac::Aircraft, ::VICASWeighted, ::Multi_Threat_Tracker)
	if isempty(ac.sensor.obs) == false
		# Utility fusion using max-min
		q_matrix = Matrix{Float64}(6, 0)
		for obs in ac.sensor.obs
			# for polar
			state = [ft2m(obs.rho), norm_angle(obs.theta), norm_angle(obs.psi), 
						ft2m(obs.sos), ft2m(obs.soi)]
			# # for xy
			# x_rel = ft2m(obs.rho) * cos(obs.theta)
			# y_rel = ft2m(obs.rho) * sin(obs.theta)
			# state = [x_rel, y_rel, norm_angle(obs.psi), ft2m(obs.sos), ft2m(obs.soi)]

			qvals = get_qvals(alphas, state) * (state[1] - sensing_range) / (collision_range - sensing_range)
     		q_matrix = hcat(q_matrix, qvals)
		end
		min_qvals = minimum(q_matrix, 2)
		the_ind = indmax(min_qvals)
		ac.advisory = ac.cas.Ind_to_Advisory_dict[the_ind]
	else
		ac.advisory = :COC
	end
end

"""
	update the advisory of an Aircraft object with Simple_CAS and Nearest_Intruder_Tracker
"""
function get_advisory(ac::Aircraft, ::Simple_CAS, ::Nearest_Intruder_Tracker)
	if isempty(ac.sensor.obs) == false
		obs = ac.sensor.obs[1]
		if obs.rho <= ac.cas.alert_threshold
			if obs.theta < 0 # intruder on the right side
				ac.advisory = :WL # turn left
			elseif obs.theta > 0 # intruder on the left side
				ac.advisory = :WR # turn right
			else
				ac.advisory = rand([:WL, :WR])
			end
		else
			ac.advisory = :COC
		end
	else
		ac.advisory = :COC
	end
end

"""
	update the advisory of an Aircraft object without CAS
"""
# NO CAS, always COC
function get_advisory(ac::Aircraft, ::No_CAS, ::Sensor)
	ac.advisory = :COC
end

