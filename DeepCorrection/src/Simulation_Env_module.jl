__precompile__()

module Simulation_Env_module

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "../")
push!(LOAD_PATH, "../../VICAS/src/")


using DeepRL
using POMDPs
using PyPlot
using CAS_QMDP
using Distributions

include("common_const.jl")

export Simple_CASIM_Env, Dummy_Problem
export step!, sample_action
export convert_ac_state, get_qvals, lowfi_values, normalize_qvals, normalize_obs!, actions
export IND_rho, IND_theta, IND_phi, IND_v_own, IND_v_int, IND_deviation, IND_dist_to_dest, IND_prev_dist_to_dest
export STATE_DIM, AUG_STATE_DIM, TERM_VAR, LOW_FI_POLICY, LOW_FI_POLICY_FILENAME, DEST_RANGE
export V_MAX, NMAC_RANGE, SENSING_RANGE, SPAWN_RATE_SET, CONST_NUM_AC_SET, RELAX_FACTOR
export POSSIBLE_MAX_REW_VICAS, POSSIBLE_MIN_REW_VICAS
export terminal_state

"""
	keeps the angles in [0, 2*pi]
"""
function norm_angle(angle::Float64)
	return ((angle % (2 * pi)) + 2 * pi) % (2 * pi)
end

"""
	state of an Aircraft object, reflecting the observation of the environemnt
	s = [ρ, θ, ϕ, v_own, v_int]
"""
mutable struct Aircraft_State
	rho::Float64
	theta::Float64
	phi::Float64
	v_own::Float64
	v_int::Float64
end

"""
	convert Aircraft_State to Float64 array
	s = [ρ, θ, ϕ, v_own, v_int]
"""
function convert_ac_state(s::Aircraft_State) 
	return Float64[s.rho, s.theta, s.phi, s.v_own, s.v_int]
end

"""
	Define the dynamics of the Aircraft, self property
"""
mutable struct Dynamics
	x::Float64
	y::Float64
	dt::Float64
	v::Float64
	heading::Float64
	turn_rate::Float64
end

"""
	agent object
"""
mutable struct Aircraft
	id::Base.Random.UUID
	is_ego::Bool
	dist_to_ego::Float64
	section::Int64
	x_init::Float64
	y_init::Float64
	x_dest::Float64
	y_dest::Float64
	advisory::Symbol
	dyn::Dynamics
	state::Aircraft_State
end # module

"""
	initiate the ego agent, i.e., the learning agent
"""
function init_ego_agent()
	id = Base.Random.uuid1()
	is_ego = true
	section = 0
	dist_to_ego = 0.

	x_init = 0.
	y_init = 0.
	th = 2 * pi * rand()
	x_dest = cos(th) * DEST_RANGE
	y_dest = sin(th) * DEST_RANGE

	# dynamics
	x = x_init
	y = y_init
	v = (V_MAX - V_MIN) * rand() + V_MIN
	heading = norm_angle(atan2(y_dest, x_dest))
	turn_rate = 0.

	advisory = :COC
	dyn = Dynamics(x, y, DT, v, heading, turn_rate)
	state = Aircraft_State(0., 0., 0., 0., 0.)
	ego_agent = Aircraft(id, is_ego, dist_to_ego, section, x_init, y_init, 
							x_dest, y_dest, advisory, dyn, state)
	return ego_agent
end

"""
	Generate init and dest coords for env agents
	This has to be truly representitive of the CASIM simulation
"""
function get_init_and_dest_coords(ego_agent::Aircraft)
	# Use encounter model:
	intruder_dist = Categorical(vec(BEARING_DIST[:, 2]))
	intruder_angle = BEARING_DIST[:, 1][rand(intruder_dist)]
	if intruder_angle > π
		intruder_angle -= 2 * π
	end # intruder_angle ∈ [-pi, pi], centered at 0
	init_angle = norm_angle(ego_agent.dyn.heading + intruder_angle) 
	dest_angle = norm_angle(init_angle + deg2rad(rand() * 270 + 45)) # 45 - 315 uniform?
	x_init = cos(init_angle) * SENSING_RANGE * RELAX_FACTOR
	y_init = sin(init_angle) * SENSING_RANGE * RELAX_FACTOR
	x_dest = cos(dest_angle) * SENSING_RANGE * RELAX_FACTOR
	y_dest = sin(dest_angle) * SENSING_RANGE * RELAX_FACTOR

	# Use Gaussian:
	# RELAX_FACTOR = 1.5
	# init_angle = norm_angle(ego_agent.dyn.heading + rand(Normal(0, deg2rad(70))))
	# dest_angle = norm_angle(init_angle + rand(Normal(pi, deg2rad(90)))) # TODO
	# x_init = cos(init_angle) * SENSING_RANGE * RELAX_FACTOR
	# y_init = sin(init_angle) * SENSING_RANGE * RELAX_FACTOR
	# x_dest = cos(dest_angle) * SENSING_RANGE * RELAX_FACTOR
	# y_dest = sin(dest_angle) * SENSING_RANGE * RELAX_FACTOR

	# init_angle = norm_angle(ego_agent.dyn.heading + rand(Normal(0, deg2rad(90))))
	# dest_angle = norm_angle(init_angle + rand(Normal(pi, deg2rad(100)))) # TODO

	# # @ outdated @
	# x_init = cos(init_angle) * SENSING_RANGE * RELAX_FACTOR * (1 - rand() * 0.4)
	# y_init = sin(init_angle) * SENSING_RANGE * RELAX_FACTOR * (1 - rand() * 0.4)
	# x_dest = cos(dest_angle) * SENSING_RANGE * RELAX_FACTOR * (1 - rand() * 0.4)
	# y_dest = sin(dest_angle) * SENSING_RANGE * RELAX_FACTOR * (1 - rand() * 0.4)

	return x_init, y_init, x_dest, y_dest
end

"""
	initiate single env agent
"""
function init_one_env_agent(ego_agent::Aircraft, num_sections::Int64)
	id = Base.Random.uuid1()
	is_ego = false
	section = 0

	x_init, y_init, x_dest, y_dest = get_init_and_dest_coords(ego_agent)

	# dynamics
	x = x_init
	y = y_init
	v = (V_MAX - V_MIN) * rand() + V_MIN
	# adding noise to the initial heading of env_agents
	heading = norm_angle(atan2(y_dest - y_init, x_dest - x_init) + rand(Normal(0, deg2rad(20))))
	turn_rate = 0.
	dist_to_ego = sqrt(x^2 + y^2)

	advisory = :COC
	dyn = Dynamics(x, y, DT, v, heading, turn_rate)
	state = Aircraft_State(0., 0., 0., 0., 0.)
	return Aircraft(id, is_ego, dist_to_ego, section, x_init, y_init, 
						x_dest, y_dest, advisory, dyn, state)
end

"""
	initiate an array of env agents
"""
function init_env_agents(ego_agent::Aircraft, init_num_env_agents::Int64, num_sections::Int64)
	env_agents = Vector{Aircraft}(init_num_env_agents)
	for i in 1:init_num_env_agents
		env_agents[i] = init_one_env_agent(ego_agent, num_sections)
	end
	return env_agents
end

"""
	define an interface helper
"""
struct Dummy_Problem 
	discount_factor::Float64
end

"""
	define the simulation environemnt object
"""
mutable struct Simple_CASIM_Env <: AbstractEnvironment
	scheme::AbstractString
	t::Int64
	sim_horizon::Int64
	num_sections::Int64
	init_num_env_agents::Int64
	const_num_env_agents::Int64
	ego_agent::Aircraft
	ego_agent_dist_to_dest::Float64
	env_agents::Vector{Aircraft}
	section_ac_arr::Dict{Int64, Vector{Aircraft}}
	full_state::Vector{Float64}
	action_space::Vector{Symbol}
	advisory_to_ind_dict::Dict{Symbol, Int64}
	advisory_to_action_dict::Dict{Symbol, Float64}
	discount_factor::Float64
	low_fi_policy::Matrix{Float64}
	reward_weights::Dict{AbstractString, Float64}
	stochastic_env_policy::Bool
	stochastic_ego_policy::Bool
	# POMDPs
	rng::AbstractRNG
	problem::Dummy_Problem
	# visualization option
	render::Bool
	# deep correction
	correction::Bool
	correction_weight::Float64
	nn_policy::Bool
	random_populate_replay_buffer::Bool

	function Simple_CASIM_Env(;
		scheme::AbstractString="_TEST_",
		sim_horizon::Int64=5000,
		num_sections::Int64=NUM_SECTIONS,
		init_num_env_agents::Int64=6,
		const_num_env_agents::Int64=rand(CONST_NUM_AC_SET),
		discount_factor::Float64=0.95,
		low_fi_policy_filename::AbstractString=LOW_FI_POLICY_FILENAME,
		rng::MersenneTwister=MersenneTwister(1),
		render::Bool=false,
		correction::Bool=false,
		stochastic_env_policy::Bool=false,
		stochastic_ego_policy::Bool=false,
		nn_policy::Bool=false,
		correction_weight::Float64=0.5,
		random_populate_replay_buffer::Bool=false)

		this = new()
		this.scheme = scheme
		this.t = 1
		this.sim_horizon = sim_horizon
		this.num_sections = num_sections
		this.init_num_env_agents = init_num_env_agents
		this.const_num_env_agents = const_num_env_agents
		this.ego_agent = init_ego_agent()
		this.ego_agent_dist_to_dest = DEST_RANGE
		this.env_agents = init_env_agents(this.ego_agent, init_num_env_agents, num_sections)
		this.section_ac_arr = Dict{Int64, Vector{Aircraft}}()
		this.full_state = zeros(num_sections * STATE_DIM + AUG_STATE_DIM)
		this.action_space = Symbol[:SR, :WR, :KEEP, :WL, :SL, :COC]
		this.advisory_to_ind_dict = Dict(:SR => 1, :WR => 2, :KEEP => 3, :WL => 4, :SL => 5, :COC => 6)
		this.advisory_to_action_dict = Dict(:SR => deg2rad(-10.0), :WR => deg2rad(-5.0), :KEEP => 0.0, 
											:WL => deg2rad(5.0), :SL => deg2rad(10.0), :COC => -1.0)
		this.discount_factor = discount_factor
		this.low_fi_policy = load_alphas(low_fi_policy_filename)
		# POMDPs
		this.rng = rng
		this.problem = Dummy_Problem(discount_factor)
		this.render = render
		if render == true
			show()
		end
		this.correction = correction
		this.correction_weight = correction_weight
		this.reward_weights = Dict("penalty_action" => PEN_ACTION, 
								   "penalty_closeness" => PEN_CLOSENESS, 
								   "penalty_nmac" => PEN_NMAC, 
								   "penalty_conflict" => PEN_CONFLICT, 
								   "penalty_deviation" => PEN_DEVIATION, 
								   "penalty_digression" => PEN_DIGRESSION, 
								   "reward_destination" => REW_DESTINATION)
		this.stochastic_env_policy = stochastic_env_policy
		this.stochastic_ego_policy = stochastic_ego_policy
		this.nn_policy = nn_policy
		this.random_populate_replay_buffer = random_populate_replay_buffer
		return this
	end
end # struct

"""
	visualize the environemnt
"""
function render_env(env::Simple_CASIM_Env)
	All_AC = vcat(env.env_agents, env.ego_agent)
	for ac in All_AC
		if ac.advisory == :COC
			scatter(ac.dyn.x, ac.dyn.y, marker="o", color="blue", s=12)
		else
			scatter(ac.dyn.x, ac.dyn.y, marker="o", color="red", s=12)
		end

		arrow_len = AIRSPACE_DIM / 6
		arrow(
			ac.dyn.x, ac.dyn.y,
			cos(ac.dyn.heading) * arrow_len * 0.8,
			sin(ac.dyn.heading) * arrow_len * 0.8,
			head_width=AIRSPACE_DIM / 20,
			width=1,
			head_length=AIRSPACE_DIM / 20,
			overhang=0.5,
			head_starts_at_zero="true",
			facecolor="black",
			length_includes_head="true")

		scatter(ac.x_dest, ac.y_dest, marker=",", color="magenta", s=12)

		plot([ac.dyn.x; ac.x_dest], [ac.dyn.y; ac.y_dest], 
			linestyle="--", color="black", linewidth=0.5)

		# text(ac.dyn.x, ac.dyn.y, string(ac.section))
	end

	# alert circle
	th = linspace(-pi, pi, 30)
	if env.ego_agent.advisory == :COC
		plot(env.ego_agent.dyn.x + SENSING_RANGE * cos.(th), 
			env.ego_agent.dyn.y + SENSING_RANGE * sin.(th), 
			linestyle="--", color="green", linewidth=0.4)
	else
		plot(env.ego_agent.dyn.x + SENSING_RANGE * cos.(th), 
			env.ego_agent.dyn.y + SENSING_RANGE * sin.(th), 
			linestyle="--", color="red", linewidth=0.4)
	end

	# NMAC circle
	plot(env.ego_agent.dyn.x + NMAC_RANGE * cos.(th), 
			env.ego_agent.dyn.y + NMAC_RANGE * sin.(th), 
			linestyle="--", color="red", linewidth=0.4)

	# plot sections and highlight the closest intruder in each section
	# section_angle = 2 * pi / env.num_sections
	# for i in 0 : env.num_sections / 2 - 1
	# 	th_1 = i * section_angle + env.ego_agent.dyn.heading
	# 	th_2 = th_1 + pi
	# 	plot(SENSING_RANGE * [cos(th_1), cos(th_2)], SENSING_RANGE * [sin(th_1), sin(th_2)], 
	# 		linestyle="--", color="red", linewidth=0.4)
	# end

	# println(env.full_state)
	for i in 1 : env.num_sections
		rho = env.full_state[(i - 1) * STATE_DIM + IND_rho]
		if rho != TERM_VAR
			theta = env.full_state[(i - 1) * STATE_DIM + IND_theta]
			scatter(rho * cos(theta), rho * sin(theta), 
				marker="o", color="green", s=40, linewidths=0.6)
		end
	end

	xlabel("X (m)")
	ylabel("Y (m)")
	title(string(env.ego_agent.advisory) * string(env.const_num_env_agents / 100) * ", " * string(env.t) * ": " * string(length(env.env_agents)))
	axis("equal")
	ax = gca()
	ax[:set_xlim]([-AIRSPACE_DIM * 1.5, AIRSPACE_DIM * 1.5])
	ax[:set_ylim]([-AIRSPACE_DIM * 1.5, AIRSPACE_DIM * 1.5])
	pause(0.1)
	clf()
end

"""
	overload :< operator for the Aircraft object
"""
function Base.:<(a::Aircraft, b::Aircraft)
	if a.dyn.x^2 + a.dyn.y^2 < b.dyn.x^2 + b.dyn.y^2
		return true
	else
		return false
	end
end

"""
	overload :> operator for the Aircraft object
"""
function Base.:>(a::Aircraft, b::Aircraft)
	if a.dyn.x^2 + a.dyn.y^2 > b.dyn.x^2 + b.dyn.y^2
		return true
	else
		return false
	end
end

"""
	sort the Aircraft in an array, wrt distance to the ego agent
	using quick sort, returns ascending order in terms of the distance to the ego agent
"""
function sort_agents!(agents_arr::Vector{Aircraft}, lo::Int64, hi::Int64)
    i, j = lo, hi
    while i < hi
        pivot = agents_arr[(lo + hi) >>> 1]
        while i <= j
            while agents_arr[i] < pivot; i += 1; end
            while agents_arr[j] > pivot; j -= 1; end
            if i <= j
                agents_arr[i], agents_arr[j] = agents_arr[j], agents_arr[i]
                i, j = i + 1, j - 1
            end
        end
        if lo < j; sort_agents!(agents_arr, lo, j); end
        lo, j = i, hi
    end
end

"""
	sort the Aircraft in an array, wrt the maximum q_vals wrt the eo agent
	using quick sort, returns ascending order in terms of the maximum q_vals
"""
function sort_agents_q_vals!(env::Simple_CASIM_Env, agents_arr::Vector{Aircraft}, lo::Int64, hi::Int64)
	i, j = lo, hi
    while i < hi
        pivot = agents_arr[(lo + hi) >>> 1]
        pivot_max_q = maximum(get_qvals(env.low_fi_policy, get_state(env.ego_agent, pivot)))
        while i <= j
        	agent_i_max_q = maximum(get_qvals(env.low_fi_policy, get_state(env.ego_agent, agents_arr[i])))
        	agent_j_max_q = maximum(get_qvals(env.low_fi_policy, get_state(env.ego_agent, agents_arr[j])))
            while agent_i_max_q < pivot_max_q 
            	i += 1
            	agent_i_max_q = maximum(get_qvals(env.low_fi_policy, get_state(env.ego_agent, agents_arr[i])))
            end
            while agent_j_max_q > pivot_max_q 
            	j -= 1 
            	agent_j_max_q = maximum(get_qvals(env.low_fi_policy, get_state(env.ego_agent, agents_arr[j])))
            end
            if i <= j
                agents_arr[i], agents_arr[j] = agents_arr[j], agents_arr[i]
                i, j = i + 1, j - 1
            end
        end
        if lo < j; sort_agents_q_vals!(env, agents_arr, lo, j); end
        lo, j = i, hi
    end
end


"""
	update the state of an Aircraft object with Nearest_Intruder_Tracker
	OR with the most dangerous intruder traker
	OR with the Multi_Intruder_Tracker
"""
function get_state!(ac::Aircraft, env::Simple_CASIM_Env)
	# search for the closest intruder from ownship
	if length(env.env_agents) > 0

		# # The closest intruder
		# closest_intruder = nothing
		# closest_dist = Inf
		# All_AC = vcat(env.env_agents, env.ego_agent)
		# for other_ac in All_AC
		# 	if other_ac.id != ac.id
		# 		dist = sqrt((ac.dyn.x - other_ac.dyn.x)^2 + (ac.dyn.y - other_ac.dyn.y)^2)
		# 		if dist < closest_dist 
		# 			closest_dist = dist
		# 			closest_intruder = other_ac
		# 		end
		# 	end
		# end

		# # The most dangerous intruder in terms of low max q_vals
		closest_intruder = nothing
		min_max_q = Inf
		All_AC = vcat(env.env_agents, env.ego_agent)
		for other_ac in All_AC
			if other_ac.id != ac.id
				max_q = maximum(get_qvals(env.low_fi_policy, get_state(ac, other_ac)))
				if max_q < min_max_q 
					min_max_q = max_q
					closest_intruder = other_ac
				end
			end
		end
		closest_dist = sqrt((ac.dyn.x - closest_intruder.dyn.x)^2 + (ac.dyn.y - closest_intruder.dyn.y)^2)
	
		# s = [ρ, θ, ϕ, v_own, v_int]
		intruder_pos_angle = norm_angle(atan2(closest_intruder.dyn.y - ac.dyn.y, 
											  closest_intruder.dyn.x - ac.dyn.x))
		theta = norm_angle(intruder_pos_angle - ac.dyn.heading)
		phi = norm_angle(closest_intruder.dyn.heading - ac.dyn.heading)
	
		ac.state = Aircraft_State(closest_dist, theta, phi, ac.dyn.v, closest_intruder.dyn.v)
	else # no env_agents
		# following VICAS, set the state to terminal state
		ac.state = Aircraft_State(TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR)
	end
end

"""
	helper: get state for ego_agent
"""
function get_state(ego::Aircraft, int::Aircraft)
	rho = sqrt(int.dyn.x^2 + int.dyn.y^2)
	theta = norm_angle(atan2(int.dyn.y, int.dyn.x))
	phi = norm_angle(int.dyn.heading - ego.dyn.heading)
	return [rho, theta, phi, ego.dyn.v, int.dyn.v]
end


"""
	get section id of an ac
"""
function get_section_id!(ac::Aircraft, env::Simple_CASIM_Env)
	# ac_pos_angle = norm_angle(atan2(ac.dyn.y, ac.dyn.x))
	# Relative to the heading of the ego_agent
	ac_pos_angle = norm_angle(norm_angle(atan2(ac.dyn.y, ac.dyn.x)) - norm_angle(env.ego_agent.dyn.heading))
	ac.section = floor(ac_pos_angle / (2 * pi / env.num_sections)) + 1
end

"""
	update the states of each agent in the env as well as the full_state
"""
function update_states!(env::Simple_CASIM_Env)
	# Two other different ways of forming full_state:
	get_state!(env.ego_agent, env) # update intruder observation: closest or most dangerous

	#
	# intruders = Vector{Aircraft}(0)
	# for ac in env.env_agents
	# 	get_state!(ac, env) # update state for env_agents using closest
	# 	if check_within_sensing(ac, env) # collect intruders for ego_agent
	# 		push!(intruders, ac) 
	# 	end
	# end

	# hi = length(intruders)
	# lo = (hi == 0) ? 0 : 1
	
	# if hi > 0 
	# 	# 1. Using multiple closest intruders
	# 	# Ascending order in terms of distance from ego_agent
	# 	sort_agents!(intruders, lo, hi) 
	# 	# 2. Using multiple dangerous intruders
	# 	# Ascending order in terms of maximum q_vals wrt ego_agent 
	# 	# The minimum maximum q_vals => the most dangerous
	# 	# sort_agents_q_vals!(env, intruders, lo, hi)

	# 	for i in 1 : env.num_sections
	# 		if i <= length(intruders)
	# 			env.full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] .= get_state(env.ego_agent, intruders[i])
	# 		else
	# 			env.full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] .= terminal_state
	# 		end
	# 	end
	# else # no intruder
	# 	env.full_state[1 : STATE_DIM * env.num_sections] .= TERM_VAR
	# end
	#


	# 3. Using closest intruders in each sections 
	# clear section_ac_arr

	env.section_ac_arr = Dict{Int64, Vector{Aircraft}}()
	for isection in 1:env.num_sections
		env.section_ac_arr[isection] = Aircraft[]
	end

	# update state for the ego agent
	get_state!(env.ego_agent, env)

	for ac in env.env_agents
		get_state!(ac, env) # update state for env_agents using closest
		if check_within_sensing(ac, env)
			get_section_id!(ac, env)
			push!(env.section_ac_arr[ac.section], ac)
		else
			ac.section = 0
		end
	end

	# sort ac in each section
	for isection in 1:env.num_sections
		hi = length(env.section_ac_arr[isection])
		lo = (hi == 0) ? 0 : 1
		if hi > 0 
			sort_agents!(env.section_ac_arr[isection], lo, hi)
			# OR
			# sort_agents_q_vals!(env, env.section_ac_arr[isection], lo, hi)

			# Resulting section_ac_arr is in ascending order,
			# taking ac that is the closest OR the most dangerous (indexed by 1)
			# wrt the ego ac for each section to update the full_state
			env.full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
				get_state(env.ego_agent, env.section_ac_arr[isection][1])
		else # no intruder in the section
			# set the corresponding state var to terminal values
			env.full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
				[TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR]
		end
	end
	#

	# Append the destination information to the full_state
	# 1. deviation from the destination, measured by rad, ∈[0, π]
	deviation = norm_angle(env.ego_agent.dyn.heading - 
							norm_angle(atan2(env.ego_agent.y_dest - env.ego_agent.dyn.y, 
						 					env.ego_agent.x_dest - env.ego_agent.dyn.x)))# ∈ [0, 2*π)
	if deviation > π
		deviation = deviation - 2 * π
	end # deviation ∈ [-π, π]

	# 2. Current distance to destination
	dist_to_dest = sqrt((env.ego_agent.dyn.y - env.ego_agent.y_dest)^2 + 
		(env.ego_agent.dyn.x - env.ego_agent.x_dest)^2)

	# 3. previous distance to destination
	prev_dist_to_dest = env.ego_agent_dist_to_dest
	## update ego_agent_dist_to_dest
	env.ego_agent_dist_to_dest = dist_to_dest

	# update state entries
	env.full_state[end - AUG_STATE_DIM + 1 : end] .= [deviation, dist_to_dest, prev_dist_to_dest]

	return env.full_state
end



"""
	update advisory for env_agents following low_fi_policy (VICAS)
"""
function get_env_agent_advisory!(ac::Aircraft, env::Simple_CASIM_Env)
	qvals = get_qvals(env.low_fi_policy, convert_ac_state(ac.state))
	if env.stochastic_env_policy == false
		# definitive
		q_ind = indmax(qvals)
	else 
		# softmax 
		qvals -= maximum(qvals) # preventing extreme values for power
		exp_qvals = exp.(qvals)
		cat_dist = Categorical(exp_qvals / sum(exp_qvals))
		q_ind = rand(cat_dist)
	end
	ac.advisory = env.action_space[q_ind]
end

"""
	 travel towards destination if there is no conflict
"""
function go_to_dest!(ac::Aircraft, env::Simple_CASIM_Env)
	# set threshold for confirming the correct heading towards destination
	threshold = deg2rad(10.)
	heading_vec = [cos(ac.dyn.heading); sin(ac.dyn.heading); 0.]
	desti_dir_angle = norm_angle(atan2(ac.y_dest - ac.dyn.y, ac.x_dest - ac.dyn.x))
	heading_error = norm_angle(abs(desti_dir_angle - ac.dyn.heading))

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
			ac.dyn.turn_rate = env.advisory_to_action_dict[:SR]
		elseif side == :Left
			ac.dyn.turn_rate = env.advisory_to_action_dict[:SL]
		else # dest on heading
			ac.dyn.turn_rate = 0. # keep heading
		end
	end
end

"""
	update dynamics for env_agents, env_agents follow VICAS by default
"""
function update_agents_dynamics!(env::Simple_CASIM_Env, advisory::Symbol)
	# get advisory for ego_agent
	env.ego_agent.advisory = advisory
	if advisory != :COC # conflict!
		env.ego_agent.dyn.turn_rate = env.advisory_to_action_dict[advisory]
	else # COC
		go_to_dest!(env.ego_agent, env)
	end

	env.ego_agent.dyn.heading = 
		norm_angle(env.ego_agent.dyn.heading + env.ego_agent.dyn.turn_rate * env.ego_agent.dyn.dt)
	ego_x_disp = cos(env.ego_agent.dyn.heading) * env.ego_agent.dyn.v * env.ego_agent.dyn.dt
	ego_y_disp = sin(env.ego_agent.dyn.heading) * env.ego_agent.dyn.v * env.ego_agent.dyn.dt
	# update destination coords
	env.ego_agent.x_dest -= ego_x_disp
	env.ego_agent.y_dest -= ego_y_disp

	for ac in env.env_agents
		# update destination coords
		ac.x_dest -= ego_x_disp
		ac.y_dest -= ego_y_disp
		# get advisories for env_agents
		get_env_agent_advisory!(ac, env)
		if ac.advisory != :COC # conflict!
			ac.dyn.turn_rate = env.advisory_to_action_dict[ac.advisory]
		else # COC
			go_to_dest!(ac, env)
		end
		ac.dyn.heading = norm_angle(ac.dyn.heading + ac.dyn.turn_rate * ac.dyn.dt)
		ac_x_disp = cos(ac.dyn.heading) * ac.dyn.v * ac.dyn.dt
		ac_y_disp = sin(ac.dyn.heading) * ac.dyn.v * ac.dyn.dt
		ac.dyn.x += ac_x_disp - ego_x_disp
		ac.dyn.y += ac_y_disp - ego_y_disp
	end
end

"""
	check if an env_agent goes out of the sensing range
"""
function check_within_sensing(ac::Aircraft, env::Simple_CASIM_Env)
	if sqrt((ac.dyn.x - env.ego_agent.dyn.x)^2 + (ac.dyn.y - env.ego_agent.dyn.y)^2) <= SENSING_RANGE
		return true
	else
		return false
	end
end

function check_within_extended_sensing(ac::Aircraft, env::Simple_CASIM_Env)
	if sqrt((ac.dyn.x - env.ego_agent.dyn.x)^2 + (ac.dyn.y - env.ego_agent.dyn.y)^2) <= SENSING_RANGE * RELAX_FACTOR
		return true
	else
		return false
	end
end

"""
	check if an agent reaches destination
"""
function check_dest(ac::Aircraft)
	if sqrt((ac.dyn.x - ac.x_dest)^2 + (ac.dyn.y - ac.y_dest)^2) <= DEST_CRITERION
		return true
	else
		return false
	end
end

"""
	control (maintain) the traffic density in the airspace as constant
"""
function airspace_control!(env::Simple_CASIM_Env)
	# # constant number of_env agents
	# while length(env.env_agents) < env.const_num_env_agents
	# 	ac = init_one_env_agent(env.ego_agent, env.num_sections)
	# 	push!(env.env_agents, ac)
	# end
	# for i = 1:length(env.env_agents)
	# 	if check_within_sensing(env.env_agents[i], env) == false || check_dest(env.env_agents[i]) == true
	# 		deleteat!(env.env_agents, i)
	# 		ac = init_one_env_agent(env.ego_agent, env.num_sections)
	# 		push!(env.env_agents, ac)
	# 	end
	# end

	
	i = 1
	while i <= length(env.env_agents)
		if check_within_extended_sensing(env.env_agents[i], env) == false || check_dest(env.env_agents[i]) == true
		# if check_within_sensing(env.env_agents[i], env) == false || check_dest(env.env_agents[i]) == true
			deleteat!(env.env_agents, i)
		end
		i += 1
	end

	# Number of env_agents controlled by Poisson distribution parameterized by spawn rates
	# SPAWN_RATE = env.const_num_env_agents / 100
	# num_new_ac = rand(Poisson(SPAWN_RATE))
	# for i = 1 : num_new_ac
	# 	ac = init_one_env_agent(env.ego_agent, env.num_sections)
	# 	push!(env.env_agents, ac)
	# end

	# Number of env_agents controlled by constant number
	const_num_ac = rand(CONST_NUM_AC_SET)
	while length(env.env_agents) < const_num_ac
		ac = init_one_env_agent(env.ego_agent, env.num_sections)
		push!(env.env_agents, ac)
	end

end

"""
	reward function
	obs is not normalized
"""
function reward_function(env::Simple_CASIM_Env, obs::Vector{Float64}, advisory::Symbol)
	reward = 0.0
	num_nmac = 0

	action = rad2deg(env.advisory_to_action_dict[advisory]) # in [deg]

	# 1. penalize conflict
	if advisory == :COC
		action = 0.0
	else
		reward -= PEN_CONFLICT
	end

	# 2. penalize action
	reward -= PEN_ACTION * action^2

	# 3. penalize distance between intruders
	int_reward = 0.0
	for isection in 1:env.num_sections
		rho = obs[(isection - 1) * STATE_DIM + IND_rho]
		if rho < NMAC_RANGE
			num_nmac += 1
			int_reward -= PEN_NMAC
		end
		int_reward -= PEN_CLOSENESS * exp(- (rho - NMAC_RANGE) / NMAC_RANGE)
	end
	# average the int_reward over sections?
	int_reward /= env.num_sections
	reward += int_reward

	# Additional reward apart from those the same as VICAS:

	# 4. reward reaching destination
	if check_dest(env.ego_agent) == true
		reward += REW_DESTINATION
	end

	# Experiment on the following rewards:

	# 5. penalize pointing far from destination
	reward -= rad2deg(abs(obs[IND_deviation])) * PEN_DEVIATION

	# # 6. penalize digression from destination in terms of distance
	reward -= (obs[IND_dist_to_dest] - obs[IND_prev_dist_to_dest]) * PEN_DIGRESSION
	# OR
	# 6. rewarding getting closer to the destination
	# reward += exp(- obs[IND_dist_to_dest] / DEST_RANGE) * PEN_DIGRESSION

    # scale the step reward (scale reward, do not shift reward!)
    # norm_reward = reward / (POSSIBLE_MAX_REW - POSSIBLE_MIN_REW)
    norm_reward = reward / (POSSIBLE_MAX_REW_VICAS - POSSIBLE_MIN_REW_VICAS)
    # println("norm_rew = ", norm_reward)

    # # non-scaled step reward
    # norm_reward = reward

	return norm_reward, num_nmac
end

"""
	normalize the input to the network
	s = [ρ, θ, ϕ, v_own, v_int]
"""
function normalize_obs!(env::Simple_CASIM_Env, obs::Vector{Float64})
	for isection in 0 : env.num_sections - 1
		if obs[isection * STATE_DIM + IND_rho] == TERM_VAR
			obs[isection * STATE_DIM + 1 : isection * STATE_DIM + STATE_DIM] *= 0 # if terminal, set empty
			# obs[isection * STATE_DIM + 1 : isection * STATE_DIM + STATE_DIM] .= 1 # if terminal, set to 1
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
	obs[IND_dist_to_dest] /= DEST_RANGE
	obs[IND_prev_dist_to_dest] /= DEST_RANGE
end


"""
	for deep correction
"""
function normalize_qvals(qvals::Vector{Float64}; epsilon::Float64=1e-12)
    # return qvals / sum(qvals)
    return qvals / sqrt(max(sum(qvals .^ 2), epsilon))
end

"""
	Utility fusion using max-min
	This function gives min_qvals over multiple intruders
	Action is obtained by taking argmax over actions
"""
function lowfi_values(env::Simple_CASIM_Env, s::Array{Float64, 1}) 
    # s is the normalized full_state, term_state set to empty (zeros)
    q_matrix = Matrix{Float64}(length(env.action_space), 0) # initiate an empty q_matrix
    rhos = [s[(i - 1) * STATE_DIM + IND_rho] for i in 1 : env.num_sections]
    for i in 1 : env.num_sections
    	if rhos[i] != 0 # section not empty, since s is normalized, empty state is set to zeros
    		section_state = s[STATE_DIM * (i - 1) + 1 : STATE_DIM * i]
    		# de-normalize the state (restore to the original scales)
        	section_state[IND_rho]   *= SENSING_RANGE
        	section_state[IND_theta] *= (2 * pi)
        	section_state[IND_phi]   *= (2 * pi)
        	section_state[IND_v_own] *= V_MAX
        	section_state[IND_v_int] *= V_MAX
        	# only appending qvals to q_matrix if the section_state is non-terminal
        	qvals = get_qvals(env.low_fi_policy, section_state)
        	q_matrix = hcat(q_matrix, qvals)
    	end
    end
    if size(q_matrix, 2) > 0
    	min_qvals = minimum(q_matrix, 2)
    else # q_matrix is empty, then return terminal state values
    	# This only happens when there is no intruder.
    	min_qvals = get_qvals(env.low_fi_policy, terminal_state)
    end

	# q_lo = normalize_qvals(vec(min_qvals)) # deprecated method
	q_lo = vec(min_qvals)

    return reshape(q_lo, 1, :) # Array{Float64, 2}, convert to row vector
end

##################################################################################
## DeepRL.jl interface implementation                                           ##
##################################################################################

function Base.reset(env::Simple_CASIM_Env)
	# env.const_num_env_agents = rand(SPAWN_RATE_SET) # reset SPAWN_RATE for the episode
	env.const_num_env_agents = rand(CONST_NUM_AC_SET) 
	env.t = 1
	env.ego_agent = init_ego_agent()
	env.env_agents = init_env_agents(env.ego_agent, env.init_num_env_agents, env.num_sections)
	# env.const_num_env_agents = env.init_num_env_agents # + rand(0:5)
	update_states!(env) # update observation
	obs = env.full_state
	normalize_obs!(env, obs)
	return obs
end

function POMDPs.actions(env::Simple_CASIM_Env)
	return env.action_space
end

function DeepRL.sample_action(env::Simple_CASIM_Env)
    # return rand(env.rng, actions(env))
    return rand(actions(env)) # ignoring rng
end

function POMDPs.n_actions(env::Simple_CASIM_Env)
	return length(env.action_space)
end

function DeepRL.obs_dimensions(env::Simple_CASIM_Env)
	# obs = full_state
	return size(env.full_state)
end

function POMDPs.discount(problem::Dummy_Problem)
	return problem.discount_factor
end

function POMDPs.action_index(problem::Dummy_Problem, action::Symbol)
	# Symbol[:SR, :WR, :KEEP, :WL, :SL, :COC]
	if action == :SR
		return 1
	elseif action == :WR
		return 2
	elseif action == :KEEP
		return 3
	elseif action == :WL
		return 4
	elseif action == :SL
		return 5
	else
		return 6
	end
end

function DeepRL.step!(env::Simple_CASIM_Env, advisory::Symbol)
	update_agents_dynamics!(env, advisory)
	done = check_dest(env.ego_agent)
	obs = update_states!(env) # update observation, non-normalized (for getting reward)
	# println(obs)
	if env.render
    	render_env(env)
    end
	reward, num_nmac = reward_function(env, obs, advisory) 
	# reward is scaled to fit the output of lowfi_values and correction network
	env.t += 1
    info = env.t
	if !done
    	airspace_control!(env)
    end
    normalize_obs!(env, obs)
    # the returned obs is normalized, therefore, the states in the replay buffer is normalized
	return obs, reward, num_nmac, done, info
end

end # module


