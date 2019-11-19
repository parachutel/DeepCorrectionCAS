include("get_observation.jl")
include("get_advisory.jl")

"""
	Update the dynamics of an Aircraft object when there is alert
	ac.advisory != :COC
"""
function update_dynamics_with_alert(ac::Aircraft)
	# get turn rate
	ac.dyn.turn_rate = deg2rad(ac.cas.Advisory_to_Action_dict[ac.advisory])
	# update heading
	ac.dyn.heading += ac.dyn.turn_rate * ac.dyn.dt
	# constrain heading
	if ac.dyn.heading > pi
		ac.dyn.heading -= 2 * pi
	elseif ac.dyn.heading < - pi
		ac.dyn.heading += 2 * pi
	end
	# update coordinates
	ac.dyn.x += cos(ac.dyn.heading) * ac.dyn.v * ac.dyn.dt
	ac.dyn.y += sin(ac.dyn.heading) * ac.dyn.v * ac.dyn.dt
	ac.actual_route_len += ac.dyn.v * ac.dyn.dt
end

"""
	Update the dynamics of an Aircraft object when there is NO alert
	ac.advisory == :COC || isempty(ac.sensor.obs)
"""
function go_to_dest(ac::Aircraft)
	# set threshold for confirming the correct heading towards destination
	threshold = deg2rad(10.)

	heading_vec = [cos(ac.dyn.heading); sin(ac.dyn.heading); 0.]
	desti_dir_angle = atan2(ac.y_dest - ac.dyn.y, ac.x_dest - ac.dyn.x)
	heading_error = abs(desti_dir_angle - ac.dyn.heading)

	if heading_error < threshold
		ac.dyn.turn_rate = 0.
		ac.dyn.heading = desti_dir_angle
	else
		desti_vec = [cos(desti_dir_angle); sin(desti_dir_angle); 0.]
		cross_result = cross(heading_vec, desti_vec)
	
		if cross_result[end] < 0
			side = :Right
		elseif cross_result[end] > 0
			side = :Left
		else
			side = :OnDir
		end

		# get turn rate
		if side == :Right
			ac.dyn.turn_rate = deg2rad(ac.cas.Advisory_to_Action_dict[:SR])
		elseif side == :Left
			ac.dyn.turn_rate = deg2rad(ac.cas.Advisory_to_Action_dict[:SL])
		else # dest on heading
			ac.dyn.turn_rate = 0. # keep heading
		end
	end
	# update heading
	ac.dyn.heading = ac.dyn.heading + ac.dyn.turn_rate * ac.dyn.dt
	# constrain heading
	if ac.dyn.heading > pi
		ac.dyn.heading = ac.dyn.heading - 2 * pi
	elseif ac.dyn.heading < - pi
		ac.dyn.heading = ac.dyn.heading + 2 * pi
	end
	# update coordinates
	ac.dyn.x = ac.dyn.x + cos(ac.dyn.heading) * ac.dyn.v * ac.dyn.dt
	ac.dyn.y = ac.dyn.y + sin(ac.dyn.heading) * ac.dyn.v * ac.dyn.dt
	ac.actual_route_len += ac.dyn.v * ac.dyn.dt
end

"""
	Warp up
	update dynamics for all the agents
"""
function update_all_dynamics(All_AC::Array{Aircraft, 1}; correction=false)
	for ac in All_AC
		if correction == false
			# update observation (state)
			get_observation!(ac, All_AC, ac.sensor)
			# update action (advisory)
			get_advisory(ac, ac.cas, ac.sensor)
		else
			full_state, lowfi_state = get_full_and_low_fi_state(correction_type, ac, All_AC, policy.env.num_sections)
			normalize_obs!(ac, policy.env.num_sections, full_state)
			get_advisory(ac, full_state, lowfi_state)
		end
	end

	for ac in All_AC
		if isempty(ac.sensor.obs) == true || ac.advisory == :COC
			ac.advisory = :COC
			go_to_dest(ac)
		else
			update_dynamics_with_alert(ac)
		end
	end
end