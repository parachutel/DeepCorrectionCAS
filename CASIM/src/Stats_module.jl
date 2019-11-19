module Stats_module

push!(LOAD_PATH, "../src")

using JLD, Aircraft_module, Constants_module
include("check_status.jl")

export Stats, run_stats

mutable struct Stats
	# airspace_controller::Airspace_Controller
	# cas_ctrl_type::DataType
	# sensor_ctrl::Sensor_Property_Controller
	# dyn_update_rule::Update_Rule

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
	ac_encountered_per_step_arr::Array{Float64,1}

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
	encounter_distribution::Vector{Float64}



	function Stats(
		generated_time::Float64)
		return new(generated_time, 
			[], [], [], [], [], [], [], [], [], [],
			0, 0, 0., 0., 0., 0., 0., zeros(6))
	end
end

function run_stats(All_AC::Array{Aircraft,1}, stats::Stats, t::Int64)
	push!(stats.t, t)
	push!(stats.num_ac, length(All_AC))
	push!(stats.num_NMAC_t, check_NMAC(All_AC))
	push!(stats.num_NMAC_cum, sum(stats.num_NMAC_t))
	avg_flight_time = 0
	if isempty(stats.num_ac_arrived)
		num_ac_arrived = 0
	else
		num_ac_arrived = stats.num_ac_arrived[t-1]
	end
	avg_ac_encountered = 0.
	temp_num_intruders_per_step = 0.
	for ac in All_AC
		avg_flight_time += (t - ac.created_time)
		if check_desti(ac) == true
			num_ac_arrived += 1
		end
		ac.t_to_nominal_t = (t - ac.created_time) / (ac.nominal_route_len / ac.dyn.v)
		ac.d_to_nominal_d = sqrt((ac.dyn.x - ac.x_dest)^2 + (ac.dyn.y - ac.y_dest)^2) / ac.nominal_route_len
		avg_ac_encountered += ac.cumulative_num_ac_encountered / (t - ac.created_time)
		temp_num_intruders_per_step += ac.num_ac_encountered_arr[end]
		if ac.num_ac_encountered_arr[end] > 5
			stats.encounter_distribution[end] += 1
		elseif ac.num_ac_encountered_arr[end] >= 1 && ac.num_ac_encountered_arr[end] <= 5
			stats.encounter_distribution[ac.num_ac_encountered_arr[end]] += 1
		end
	end
	avg_flight_time /= length(All_AC)
	push!(stats.average_flight_time, avg_flight_time)
	push!(stats.num_ac_arrived, num_ac_arrived)
	push!(stats.num_ac_created, num_ac_arrived + length(All_AC))
	push!(stats.traffic_density, length(All_AC) / airspace_dim^2)
	stats.total_flight_hours += length(All_AC) * dt
	stats.total_NMACs = stats.num_NMAC_cum[end]
	stats.final_traffic_density = stats.traffic_density[end]
	stats.landed_over_created = stats.num_ac_arrived[end] / stats.num_ac_created[end]
	stats.final_avg_flight_hours = stats.total_flight_hours / stats.num_ac_created[end]
	avg_ac_encountered /= length(All_AC)
	push!(stats.avg_ac_encountered, avg_ac_encountered)
	temp_num_intruders_per_step /= length(All_AC)
	push!(stats.ac_encountered_per_step_arr, temp_num_intruders_per_step)
end

end # module