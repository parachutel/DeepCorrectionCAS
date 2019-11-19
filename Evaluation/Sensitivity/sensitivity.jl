push!(LOAD_PATH, "../../DeepCorrection")
push!(LOAD_PATH, "../../DeepCorrection/src")
push!(LOAD_PATH, "../../VICAS/src/")
push!(LOAD_PATH, "../TrajectoryViz")

include("../TrajectoryViz/trajViz.jl")


headingSet = deg2rad.([30 * i - 360 for i in 1:23])

function sensitivity(;
	numInt::Int64=1,
	numSamples::Int64=10000,
	CASType::Symbol=:correctedSector,
	policyFile::AbstractString="",
	printOption::Bool=false,
	correction_weight::Float64=0.5)

	problem_file = "../../DeepCorrection/logs/" * policyFile * "/final_problem.jld"
	weights_file = "../../DeepCorrection/logs/" * policyFile * "/final_weights.jld"
	policy = restore(problem_file=problem_file, weights_file=weights_file)
	policy.env.correction_weight = correction_weight
	
	x0 = 0.0
	y0 = 0.0
	x_dest = 10000.0
	y_dest = 0.0
	ownship = initOwnship(x0, y0, x_dest, y_dest, CASType, policy)

	sensitiveRate = 0
	for t in 1 : numSamples
		intCoords = Vector{Float64}()
		for i in 1 : numInt
			intruder_x = rand(-1000 : 50 : 1000)
			y = sqrt(SENSING_RANGE^2 - intruder_x^2)
			intruder_y = rand(trunc(- y) : 50 : trunc(y))
			intruder_heading = rand(headingSet)
			intCoords = vcat(intCoords, [intruder_x, intruder_y, intruder_heading])
		end
		getState!(ownship, numInt, intCoords)
		getAdvisory!(ownship)
		if ownship.advisory != :COC 
			sensitiveRate += 1
		end
		if printOption == true
			println(intCoords, ", ", ownship.advisory)
		end
	end
	return sensitiveRate / numSamples
end