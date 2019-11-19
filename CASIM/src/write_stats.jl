push!(LOAD_PATH, "../src")

using JLD
using Stats_module

"""
	generated_time::Float64

	# Metrics within simulation wrt time steps:
	t::Array{Int64,1}
	num_ac::Array{Int64,1}
	num_NMAC_t::Array{Int64,1}
	num_NMAC_cum::Array{Int64,1}
	average_flight_time::Array{Float64,1}
	num_ac_arrived::Array{Int64,1}
	num_ac_created::Array{Int64,1}
	traffic_density::Array{Float64,1}
	avg_ac_encountered::Array{Float64,1}
	ac_encountered_per_step::Array{Float64,1}

	# Simulation-wise metrics:
	# Safety:
	total_NMACs::Int64
	total_flight_hours::Int64
	final_traffic_density::Float64

	# Efficiency:
	landed_over_created::Float64
	final_avg_flight_hours::Float64
	total_route_len::Float64
	total_nominal_route_len::Float64
"""
function write_stats(
	exp::AbstractString,
	scheme::AbstractString, 
	stats::Stats, 
	sim_horizon::Int64,
	airspace_controller::Airspace_Controller, 
	cas_ctrl::CAS_Property_Controller, 
	sensor_ctrl::Sensor_Property_Controller)

	time_str = Dates.format(now(), "e-dd-u-HH.MM.SS") # file tag, read .jld by inputing time string
	
	airspace_controller_type_str = string(typeof(airspace_controller))
	cas_type_str = string(typeof(cas_ctrl))
	sensor_type_str = string(typeof(sensor_ctrl))


	exp_path = "../simulation_results/" * exp
	if !ispath(exp_path)
		mkdir(exp_path)
	end
	filename = exp_path * "/" * scheme * ".jld"

	stats_dict = Dict(
		"generated_time_str" => time_str,
		"generated_time" => stats.generated_time,
		"time_steps_arr" => stats.t,
		"num_ac_arr" => stats.num_ac,
		"num_NMAC_t_arr" => stats.num_NMAC_t,
		"num_NMAC_cum_arr" => stats.num_NMAC_cum,
		"average_flight_time_arr" => stats.average_flight_time,
		"num_ac_arrived_arr" => stats.num_ac_arrived,
		"num_ac_created_arr" => stats.num_ac_created,
		"traffic_density_arr" => stats.traffic_density,
		"avg_ac_encountered_arr" => stats.avg_ac_encountered,
		"total_NMACs" => stats.total_NMACs,
		"total_flight_hours" => stats.total_flight_hours,
		"final_traffic_density" => stats.final_traffic_density,
		"landed_over_created" => stats.landed_over_created,
		"final_avg_flight_hours" => stats.final_avg_flight_hours,
		"total_route_len" => stats.total_route_len,
		"total_nominal_route_len" => stats.total_nominal_route_len,
		"actual_over_nominal_route_len" => stats.total_route_len / stats.total_nominal_route_len,
		"ac_encountered_per_step" => stats.ac_encountered_per_step_arr,
		"encounter_distribution" => stats.encounter_distribution
		)
	
	println("Writing simulation results to: \n" * filename)
	jldopen(filename, "w") do file
    	write(file, "sim_results", stats_dict)
	end

	print("************************************************\nSummary: \n")
	if airspace_controller_type_str == "Airspace_Control_module.Constant_Spawn_Rate_Controller"
		println("Spawn rate = ", airspace_controller.spawn_rate * 3600)
	else
		println("Num AC = ", airspace_controller.ac_number)
	end
	print("Simlation time horizon = ")
	print(sim_horizon, "\n")
	print("Airspace controller = ")
	print(airspace_controller_type_str, "\n")
	print("CAS type = ")
	print(cas_type_str, "\n")
	print("Sensor type = ")
	print(sensor_type_str, "\n")
	print("\n")
	print("Totol NMACs = ")
	print(stats.total_NMACs, "\n")
	print("Totol flight hours = ")
	print(stats.total_flight_hours, "\n")
	print("# Created = ")
	print(stats.num_ac_created[end], "\n")
	print("# Arrived = ")
	print(stats.num_ac_arrived[end], "\n")
	print("# Final = ")
	print(stats.num_ac[end], "\n")
	print("Final traffic density = ")
	print(stats.final_traffic_density, "\n")
	print("Avg ac enconutered = ")
	print(stats.avg_ac_encountered[end-20:end], "\n")
	print("Landed / Created = ")
	print(stats.landed_over_created, "\n")
	print("Final average flight hours = ")
	print(stats.final_avg_flight_hours, "\n")
	print("|Actual route| / |Nominal route| = ")
	print(stats_dict["actual_over_nominal_route_len"], "\n")
	print("Encounter Distribution = ")
	println(stats_dict["encounter_distribution"])
	println("\n\n")
end