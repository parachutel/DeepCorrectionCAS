
module CASIM

# force recompiling by deleting the cached lib
# use /home/lisheng/.julia/lib/v0.6 on SISL server
# if ispath("/Users/shengli/.julia/lib/v0.6/CAS_QMDP.ji")
# 	rm("/Users/shengli/.julia/lib/v0.6/CAS_QMDP.ji")
# end
# if ispath("/Users/shengli/.julia/lib/v0.6/DoubleUAVs_module.ji")
# 	rm("/Users/shengli/.julia/lib/v0.6/DoubleUAVs_module.ji")
# end
# if ispath("/Users/shengli/.julia/lib/v0.6/Simulation_Env_module.ji")
# 	rm("/Users/shengli/.julia/lib/v0.6/Simulation_Env_module.ji")
# end

push!(LOAD_PATH, "../src")

using CAS_module
using Aircraft_module
using Airspace_Control_module
using Stats_module
using Constants_module


push!(LOAD_PATH, "../../DeepCorrection")
using DeepQLearning
push!(LOAD_PATH, "../../DeepCorrection/src")
using Simulation_Env_module


export Dynamics, Observation, Sensor, Aircraft
export Nearest_Intruder_Tracker, Multi_Threat_Tracker

export Airspace_Controller, Constant_Number_Controller, Constant_Spawn_Rate_Controller
export CAS_Property_Controller, Uniform_CAS, Random_CAS
export Sensor_Property_Controller, Uniform_Sensor, Random_Sensor
export airspace_initialization, airspace_control

export CAS
export ACAS_Xu_Network, Simple_CAS, No_CAS, VICAS, Corrected_VICAS, VICASWeighted
export CAS_DICT
export acas_xu_network_data

export Stats, run_stats

export get_advisory
export get_observation
export update_all_dynamics

export sensing_range
export airspace_dim
export dest_range
export collision_range

export Render_Option, Render_Airspace, Not_Render

export simulate

export correction_version, correction_type, correction_weight, VICAS_policyname, policy

"""
	convert ft to m 
"""
function ft2m(ft::Float64)
	return ft * 0.3048
end

include("update_dynamics.jl")
include("check_status.jl")
include("get_advisory.jl")
include("get_observation.jl")
include("write_stats.jl")
include("render.jl")
# include("plots_animation.jl")

function simulate(
	exp::AbstractString,
	scheme::AbstractString,
	sim_horizon::Int64, 
	airspace_controller::Airspace_Controller,
	cas_ctrl::CAS_Property_Controller, 
	sensor_ctrl::Sensor_Property_Controller, 
	render_option::Render_Option;
	correction::Bool=false)

	T = time()
	stats = Stats(T)
	t = 0
	All_AC = airspace_initialization(stats, airspace_controller.ac_number, cas_ctrl, sensor_ctrl, t)

	for t = 1 : sim_horizon
		render_airspace(t, render_option, All_AC)
		# Step:
		if t % 1000 == 0
			println(string(t) * " : " * string(length(All_AC)))
		end
		update_all_dynamics(All_AC, correction=correction)
		run_stats(All_AC, stats, t)
		All_AC = airspace_control(stats, All_AC, airspace_controller, cas_ctrl, sensor_ctrl, t)
	end
	write_stats(exp, scheme, stats, sim_horizon, airspace_controller, cas_ctrl, sensor_ctrl)

end


end # module