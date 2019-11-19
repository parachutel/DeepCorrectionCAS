module Aircraft_module

push!(LOAD_PATH, "./")

using CAS_module, Constants_module

export Dynamics, Observation, Sensor, Aircraft
export Nearest_Intruder_Tracker, Multi_Threat_Tracker

"""
	Define the Dynamics object
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
	Define the Observation object
"""
mutable struct Observation
	rho::Float64 # Distance from ownship to intruder
	theta::Float64 # Angle to intruder relative to ownship heading direction
	psi::Float64 # Heading angle of intruder relative to ownship heading direction
	sos::Float64 # Speed of ownship
	soi::Float64 # Speed of intruder
	tau::Float64 # Time until loss of vertical separation
	pRA::Int64 # Previous advisory
	intruder_pos_angle::Float64
end

"""
	Define the Sensor object
"""
abstract type Sensor end

"""
	Define the tracking mode: nearest intruder
"""
mutable struct Nearest_Intruder_Tracker <: Sensor
	sensing_range::Float64
	obs::Array{Observation, 1}
	# default constructor
	function Nearest_Intruder_Tracker(; r::Float64=sensing_range)
		this = new()
		this.sensing_range = r
		this.obs = []
		return this
	end
end

"""
	Define the tracking mode: multi-threat
"""
mutable struct Multi_Threat_Tracker <: Sensor
	sensing_range::Float64
	obs::Array{Observation, 1}
	# default constructor
	function Multi_Threat_Tracker(; r::Float64=sensing_range)
		this = new()
		this.sensing_range = r
		this.obs = [] # empty array
		return this
	end
end


"""
	Define the Aircraft object
"""
mutable struct Aircraft
	id::Base.Random.UUID
	dyn::Dynamics

	x_init::Float64
	y_init::Float64
	
	x_dest::Float64 # within airspace
	y_dest::Float64 # within airspace

	colli::Bool 
	desti::Bool 

	advisory::Symbol # Current resolution advisory
	cas::CAS
	sensor::Sensor

	created_time::Int64
	nominal_route_len::Float64
	actual_route_len::Float64

	dist_to_dest::Float64

	# statistics for each individual agent (not global stats)
	t_to_nominal_t::Float64
	d_to_nominal_d::Float64
	cumulative_num_ac_encountered::Int64
	num_ac_encountered_arr::Array{Int64,1}
end

end # module







