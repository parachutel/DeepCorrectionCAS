function closer(a::Aircraft, b::Aircraft, p::Aircraft)
	if (a.dyn.x - p.dyn.x)^2 + (a.dyn.y - p.dyn.y)^2 < 
		(b.dyn.x - p.dyn.x)^2 + (b.dyn.y - p.dyn.y)^2
		return true
	else
		return false
	end
end


function farther(a::Aircraft, b::Aircraft, p::Aircraft)
	if (a.dyn.x - p.dyn.x)^2 + (a.dyn.y - p.dyn.y)^2 > 
		(b.dyn.x - p.dyn.x)^2 + (b.dyn.y - p.dyn.y)^2
		return true
	else
		return false
	end
end

"""
	sort the Aircraft in an array
	using quick sort, returns ascending order
"""
function sort_agents!(p::Aircraft, agents_arr::Vector{Aircraft}, lo::Int64, hi::Int64)
    i, j = lo, hi
    while i < hi
        pivot = agents_arr[(lo + hi) >>> 1]
        while i <= j
            while closer(agents_arr[i], pivot, p); i += 1; end
            while farther(agents_arr[j], pivot, p); j -= 1; end
            if i <= j
                agents_arr[i], agents_arr[j] = agents_arr[j], agents_arr[i]
                i, j = i + 1, j - 1
            end
        end
        if lo < j; sort_agents!(p, agents_arr, lo, j); end
        lo, j = i, hi
    end
end

function get_section_id(ac::Aircraft, intruder::Aircraft, num_sections::Int64)
	ac_pos_angle = norm_angle(norm_angle(atan2(intruder.dyn.y - ac.dyn.y, intruder.dyn.x - ac.dyn.x)) - norm_angle(ac.dyn.heading))
	# ac_pos_angle = norm_angle(atan2(intruder.dyn.y - ac.dyn.y, intruder.dyn.x - ac.dyn.x))
	section = Int(floor(ac_pos_angle / (2 * pi / num_sections)) + 1)
	return section
end

function get_state(ac, intruder)
	intruder_pos_angle = atan2(intruder.dyn.y - ac.dyn.y, intruder.dyn.x - ac.dyn.x)

	return [ft2m(sqrt((ac.dyn.x - intruder.dyn.x)^2 + (ac.dyn.y - intruder.dyn.y)^2)), 
			norm_angle(intruder_pos_angle - ac.dyn.heading), 
			norm_angle(intruder.dyn.heading - ac.dyn.heading), 
			ft2m(ac.dyn.v), 
			ft2m(intruder.dyn.v)]
end

function get_full_and_low_fi_state(correction_type::Symbol, ac::Aircraft, All_AC::Vector{Aircraft}, num_sections::Int64)

	intruders = get_observation!(ac, All_AC, ac.sensor) # all intruders are within the sensing range
	
	if correction_type == :closest || policy
		# Method using num_sections closest intruders
		full_state = zeros(num_sections * STATE_DIM + AUG_STATE_DIM)
		hi = length(intruders)
		lo = (hi == 0) ? 0 : 1
		if hi > 0 
			# # 1. Using multiple close intruders
			# # Ascending order in terms of distance from ego_agent
			sort_agents!(ac, intruders, lo, hi) 
			# # 2. Using multiple dangerous intruders
			# # Ascending order in terms of maximum q_vals wrt ego_agent 
			# # The minimum maximum q_vals => the most dangerous
			# sort_agents_q_vals!(env, intruders, lo, hi)
	
			for i in 1 : num_sections
				if i <= length(intruders)
					full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] .= get_state(ac, intruders[i])
				else
					full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] .= terminal_state
				end
			end
		else # no intruder
			full_state[1 : STATE_DIM * num_sections] .= TERM_VAR
		end
		
	elseif correction_type == :sector
		# # Method using num_sections sections
		section_ac_arr = Dict{Int64, Vector{Aircraft}}()
		for isection in 1:num_sections
			section_ac_arr[isection] = Aircraft[]
		end
	
		for int in intruders
			if int.id != ac.id
				section_id = get_section_id(ac, int, num_sections)
				push!(section_ac_arr[section_id], int)
			end
		end
	
		full_state = zeros(num_sections * STATE_DIM + AUG_STATE_DIM)
	
		# sort ac in each section
		for isection in 1:num_sections
			hi = length(section_ac_arr[isection])
			lo = (hi == 0) ? 0 : 1
			if hi > 0 
				sort_agents!(ac, section_ac_arr[isection], lo, hi)
				# resulting section_ac_arr is in ascending order,
				# grab ac that is closest to the ego ac for each section
				# to update the full_state
				full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
					get_state(ac, section_ac_arr[isection][1]) # [m]
			else
				# no intruder in the section
				# set the corresponding state var to terminal values
				full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
					[TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR]
			end
		end
		#
	end # if correction_type
	
	# append the destination information to the full_state
	# 1. deviation from the destination, measured by rad
	deviation = norm_angle(ac.dyn.heading - 
		norm_angle(atan2(ac.y_dest - ac.dyn.y, ac.x_dest - ac.dyn.x)))
	if deviation > π
		deviation = deviation - 2 * π # ∈ [-π, π]
		# deviation = 2 * π - deviation # ∈ [0, π], deprecated
	end 

	# 2. current distance to destination
	dist_to_dest = sqrt((ac.dyn.y - ac.y_dest)^2 + (ac.dyn.x - ac.x_dest)^2) # [ft]
	# 3. previous distance to destination
	prev_dist_to_dest = ac.dist_to_dest # [ft]
	## update ego_agent_dist_to_dest
	ac.dist_to_dest = dist_to_dest # [ft]

	# update state entries
	full_state[end - AUG_STATE_DIM + 1 : end] .= [deviation, ft2m(dist_to_dest), ft2m(prev_dist_to_dest)] # [m]
	
	# find the lowfi_state from the full_state
	closest_intruder_section_id = indmin(full_state[[(i - 1) * STATE_DIM + IND_rho for i in 1:num_sections]])
	lowfi_state = full_state[STATE_DIM * (closest_intruder_section_id - 1) + 1 : STATE_DIM * closest_intruder_section_id]

	return full_state, lowfi_state
end

function normalize_obs!(ac::Aircraft, num_sections::Int64, obs::Vector{Float64})
	for isection in 0 : num_sections - 1
		if obs[isection * STATE_DIM + IND_rho] == TERM_VAR
			obs[isection * STATE_DIM + 1 : isection * STATE_DIM + STATE_DIM] *= 0 # if terminal, set empty
			# obs[isection * STATE_DIM + 1 : isection * STATE_DIM + STATE_DIM] = 1 # if terminal, set to 1
		else
			obs[isection * STATE_DIM + IND_rho] /= SENSING_RANGE
			obs[isection * STATE_DIM + IND_theta] /= (2 * pi)
			obs[isection * STATE_DIM + IND_phi] /= (2 * pi)
			# obs[isection * STATE_DIM + IND_v_own] = (obs[isection * STATE_DIM + IND_v_own] - V_MIN) / (V_MAX - V_MIN + 1e-12)
			obs[isection * STATE_DIM + IND_v_own] /= V_MAX
			obs[isection * STATE_DIM + IND_v_int] /= V_MAX
		end
	end
	obs[IND_deviation] /= pi
	obs[IND_dist_to_dest] /= ft2m(ac.nominal_route_len)
	obs[IND_prev_dist_to_dest] /= ft2m(ac.nominal_route_len)
end

"""
	update the observation of an Aircraft object with Nearest_Intruder_Tracker
"""
function get_observation!(ac::Aircraft, All_AC::Array{Aircraft, 1}, ::Nearest_Intruder_Tracker)
	# search for the closest intruder from ownship
	intruders = Aircraft[]
	closest_intruder = nothing
	closest_dist = Inf
	num_intruders_at_this_step = 0
	for other_ac in All_AC
		if other_ac.id != ac.id
			dist = sqrt((ac.dyn.x - other_ac.dyn.x)^2 + (ac.dyn.y - other_ac.dyn.y)^2)
			if dist <= ac.sensor.sensing_range
				push!(intruders, other_ac)
				num_intruders_at_this_step += 1
				ac.cumulative_num_ac_encountered += 1
				if dist < closest_dist 
					closest_dist = dist
					closest_intruder = other_ac
				end
			end
		end
	end

	push!(ac.num_ac_encountered_arr, num_intruders_at_this_step)

	if closest_intruder == nothing
		ac.sensor.obs = []
	else
		ac.sensor.obs = [construct_obs(closest_dist, ac, closest_intruder)]
	end

	return intruders
end

"""
	Update the observation of an Aircraft object with Multi_Threat_Tracker
"""
function get_observation!(ac::Aircraft, All_AC::Array{Aircraft, 1}, ::Multi_Threat_Tracker)
	ac.sensor.obs = [] # dump previous obs
	intruders = Aircraft[]
	for other_ac in All_AC
		if other_ac.id != ac.id
			dist = sqrt((ac.dyn.x - other_ac.dyn.x)^2 + (ac.dyn.y - other_ac.dyn.y)^2)
		    if dist <= ac.sensor.sensing_range
				obs = construct_obs(dist, ac, other_ac)
		    	push!(ac.sensor.obs, obs)
		    	push!(intruders, other_ac)
		    end
		end
	end
	ac.cumulative_num_ac_encountered += length(intruders)
	push!(ac.num_ac_encountered_arr, length(intruders))
	return intruders
end

"""
	Helper: construct the Observation object
"""
function construct_obs(dist::Float64, ac::Aircraft, other_ac::Aircraft)
	rho = dist
	intruder_pos_angle = atan2(other_ac.dyn.y - ac.dyn.y, other_ac.dyn.x - ac.dyn.x)
	theta = intruder_pos_angle - ac.dyn.heading
	psi = other_ac.dyn.heading - ac.dyn.heading
	# constrain theta and psi
	if theta > pi
		theta = theta - 2 * pi
	elseif theta < - pi
		theta = theta + 2 * pi
	end
	if psi > pi
		psi = psi - 2 * pi
	elseif psi < - pi
		psi = psi + 2 * pi
	end

	sos = ac.dyn.v
	soi = other_ac.dyn.v
	tau = 0 # coaltitude
	pRA = ac.cas.Advisory_to_Ind_dict[ac.advisory]

	obs = Observation(rho, theta, psi, sos, soi, tau, pRA, intruder_pos_angle)
	return obs
end

