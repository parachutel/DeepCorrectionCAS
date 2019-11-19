"""
	A global agents control method
"""
module Airspace_Control_module

push!(LOAD_PATH, "../src")

using Constants_module, Stats_module
using Distributions, CAS_module, Aircraft_module

include("check_status.jl")

export Airspace_Controller, Constant_Number_Controller, Constant_Spawn_Rate_Controller
export CAS_Property_Controller, Uniform_CAS, Random_CAS
export Sensor_Property_Controller, Uniform_Sensor, Random_Sensor
export airspace_initialization, airspace_control


abstract type Airspace_Controller end
mutable struct Constant_Number_Controller <: Airspace_Controller
	ac_number::Int64
end
mutable struct Constant_Spawn_Rate_Controller <: Airspace_Controller
	ac_number::Int64 # initial value
	spawn_rate::Float64 # num new ac per time step
end

abstract type CAS_Property_Controller end
mutable struct Uniform_CAS <: CAS_Property_Controller 
	the_cas::CAS
end
struct Random_CAS <: CAS_Property_Controller 
	the_cas::Void
	function Random_CAS()
		return new(nothing)
	end
end

abstract type Sensor_Property_Controller end
mutable struct Uniform_Sensor <: Sensor_Property_Controller 
	the_sensor::Sensor
end
struct Random_Sensor <: Sensor_Property_Controller end

"""
	Initialize airspace
"""
function airspace_initialization(stats::Stats, init_num_ac::Int64, cas_ctrl::CAS_Property_Controller, 
	sensor_ctrl::Sensor_Property_Controller, t::Int64)
	All_AC = Array{Aircraft, 1}(0);
	for i = 1 : init_num_ac
		ac = create_ac(cas_ctrl, sensor_ctrl, t)
		push!(All_AC, ac)
	end
	return All_AC
end

"""
	Control the airspace traffic density using constant ac number
"""
# Use airspace_control() after dynamics updated for All_AC
function airspace_control(stats::Stats, All_AC::Array{Aircraft, 1}, as_ctrl::Constant_Number_Controller, 
	cas_ctrl::CAS_Property_Controller, sensor_ctrl::Sensor_Property_Controller, t::Int64)
	while length(All_AC) < as_ctrl.ac_number
		ac = create_ac(cas_ctrl, sensor_ctrl, t)
		push!(All_AC, ac)
	end
	for i = 1:length(All_AC)
		All_AC[i].desti = check_desti(All_AC[i])
		if All_AC[i].desti == true
			stats.total_nominal_route_len += All_AC[i].nominal_route_len
			stats.total_route_len += (All_AC[i].actual_route_len + dest_range)
			deleteat!(All_AC, i)
			ac = create_ac(cas_ctrl, sensor_ctrl, t)
			push!(All_AC, ac)
		end
	end
	return All_AC
end

"""
	Control the airspace traffic density using constant spawn rate
	Analogying Poisson process
"""
function airspace_control(stats::Stats, All_AC::Array{Aircraft, 1}, as_ctrl::Constant_Spawn_Rate_Controller, 
	cas_ctrl::CAS_Property_Controller, sensor_ctrl::Sensor_Property_Controller, t::Int64)
	# Correction weight control
	# if correction_type == :closest && as_ctrl.spawn_rate > 5900./3600
	# 	policy.env.correction_weight = correction_weight
	# elseif correction_type == :sector && as_ctrl.spawn_rate > 7900./3600
	# 	policy.env.correction_weight = correction_weight
	# end


	# deactivate unqualified ac
	i = 1
	while i <= length(All_AC)
		All_AC[i].desti = check_desti(All_AC[i])
		if All_AC[i].desti == true
			stats.total_nominal_route_len += All_AC[i].nominal_route_len
			stats.total_route_len += (All_AC[i].actual_route_len + dest_range)
			deleteat!(All_AC, i)
		else
			i += 1
		end
	end
	# add new ac following Poisson process
	num_new_ac = rand(Poisson(as_ctrl.spawn_rate))
	for i = 1 : num_new_ac
		ac = create_ac(cas_ctrl, sensor_ctrl, t)
		# while check_colli(ac, All_AC) == true
		# 	ac = create_ac(cas_ctrl, sensor_ctrl, t)
		# end
		push!(All_AC, ac)
	end
	return All_AC
end


##################################################################################
"""
	Helpers:
"""
function init_ac_helper()
	id = Base.Random.uuid1()
	x_0 = rand() * airspace_dim
	y_0 = rand() * airspace_dim
	heading_0 = rand() * 2 * pi - pi
	turn_rate_0 = 0.
	x_dest_0 = rand() * airspace_dim
	y_dest_0 = rand() * airspace_dim
	colli_0 = false
	desti_0 = false
	advisory_0 = :COC # COC
	nominal_route_len = sqrt((x_dest_0 - x_0)^2 + (y_dest_0 - y_0)^2)
	actual_route_len = 0.
	t_to_nominal_t_0 = 0.
	d_to_nominal_d_0 = 1.
	cumulative_num_ac_encountered = 0
	num_ac_encountered_arr = Int64[]
	dist_to_dest = nominal_route_len
	return (id, x_0, y_0, heading_0, turn_rate_0, x_dest_0, y_dest_0, 
		colli_0, desti_0, advisory_0, nominal_route_len, actual_route_len,
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, num_ac_encountered_arr, dist_to_dest)
end

function init_random_cas()
	num_cas = length(subtypes(CAS))
	return random_cas = CAS_DICT[subtypes(CAS)[rand(1:num_cas)]]
end

function init_random_sensor()
	num_sensor = length(subtypes(Sensor))
	return random_sensor = subtypes(Sensor)[rand(1:num_sensor)](sensing_range)
end


function create_ac(cas_ctrl::Uniform_CAS, sensor_ctrl::Uniform_Sensor, t::Int64)
	(id, x_0, y_0, heading_0, turn_rate_0, x_dest_0, y_dest_0, 
		colli_0, desti_0, advisory_0, nominal_route_len, actual_route_len,
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, 
		num_ac_encountered_arr, dist_to_dest) = init_ac_helper()
	v_0 = rand() * (max_speed - min_speed) + min_speed
	dyn_0 = Dynamics(x_0, y_0, dt, v_0, heading_0, turn_rate_0)
	return Aircraft(id, dyn_0, x_0, y_0, x_dest_0, y_dest_0, colli_0, desti_0, 
		advisory_0, cas_ctrl.the_cas, sensor_ctrl.the_sensor, t, nominal_route_len, actual_route_len, dist_to_dest, 
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, num_ac_encountered_arr)
end

function create_ac(::Random_CAS, sensor_ctrl::Uniform_Sensor, t::Int64)
	(id, x_0, y_0, heading_0, turn_rate_0, x_dest_0, y_dest_0, 
		colli_0, desti_0, advisory_0, nominal_route_len, actual_route_len,
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, 
		num_ac_encountered_arr, dist_to_dest) = init_ac_helper()
	rand_cas = init_random_cas()
	v_0 = rand() * (max_speed - min_speed) + min_speed
	dyn_0 = Dynamics(x_0, y_0, dt, v_0, heading_0, turn_rate_0)
	return Aircraft(id, dyn_0, x_0, y_0, x_dest_0, y_dest_0, colli_0, desti_0, 
		advisory_0, rand_cas, sensor_ctrl.the_sensor, t, nominal_route_len, actual_route_len, dist_to_dest, 
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, num_ac_encountered_arr)
end

function create_ac(cas_ctrl::Uniform_CAS, ::Random_Sensor, t::Int64)
	(id, x_0, y_0, heading_0, turn_rate_0, x_dest_0, y_dest_0, 
		colli_0, desti_0, advisory_0, nominal_route_len, actual_route_len,
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, 
		num_ac_encountered_arr, dist_to_dest) = init_ac_helper()
	rand_sensor = init_random_sensor()
	v_0 = rand() * (max_speed - min_speed) + min_speed
	dyn_0 = Dynamics(x_0, y_0, dt, v_0, heading_0, turn_rate_0)
	return Aircraft(id, dyn_0, x_0, y_0, x_dest_0, y_dest_0, colli_0, desti_0, 
		in_airspace_0, advisory_0, rand_cas, rand_sensor, t, nominal_route_len, actual_route_len, dist_to_dest, 
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, num_ac_encountered_arr)
end

function create_ac(::Random_CAS, ::Random_Sensor, t::Int64)
	(id, x_0, y_0, heading_0, turn_rate_0, x_dest_0, y_dest_0, 
		colli_0, desti_0, advisory_0, nominal_route_len, actual_route_len,
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, 
		num_ac_encountered_arr, dist_to_dest) = init_ac_helper()
	rand_cas = init_random_cas()
	rand_sensor = init_random_sensor()
	v_0 = rand() * (max_speed - min_speed) + min_speed
	dyn_0 = Dynamics(x_0, y_0, dt, v_0, heading_0, turn_rate_0)
	return Aircraft(id, dyn_0, x_0, y_0, x_dest_0, y_dest_0, colli_0, desti_0, 
		in_airspace_0, advisory_0, rand_cas, rand_sensor, t, nominal_route_len, actual_route_len, dist_to_dest, 
		t_to_nominal_t_0, d_to_nominal_d_0, cumulative_num_ac_encountered, num_ac_encountered_arr)
end

end # module