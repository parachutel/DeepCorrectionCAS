#=
	This is a file for simulation based conflict resolution time computation.
=#

using Distributions, PyPlot
push!(LOAD_PATH, "../../VICAS/src/")
push!(LOAD_PATH, "../../DeepCorrection")
push!(LOAD_PATH, "../../DeepCorrection/src")
push!(LOAD_PATH, "../TrajectoryViz")

include("../TrajectoryViz/trajViz.jl")

const BEARING_DIST =   [[0.1047 0.0489];
   						[0.3142 0.0474];
   						[0.5236 0.0474];
   						[0.7330 0.0459];
   						[0.9425 0.0425];
   						[1.1519 0.0403];
   						[1.3614 0.0365];
   						[1.5708 0.0333];
   						[1.7802 0.0304];
   						[1.9897 0.0269];
   						[2.1991 0.0237];
   						[2.4086 0.0210];
   						[2.6180 0.0194];
   						[2.8274 0.0183];
   						[3.0369 0.0181];
   						[3.2463 0.0181];
   						[3.4558 0.0183];
   						[3.6652 0.0194];
   						[3.8746 0.0210];
   						[4.0841 0.0237];
   						[4.2935 0.0269];
   						[4.5029 0.0304];
   						[4.7124 0.0333];
   						[4.9218 0.0365];
   						[5.1313 0.0403];
   						[5.3407 0.0425];
   						[5.5501 0.0459];
   						[5.7596 0.0474];
   						[5.9690 0.0474];
   						[6.1785 0.0489];]


function distance(own::Ownship, int::Ownship)
	return sqrt((own.x - int.x)^2 + (own.y - int.y)^2)
end

function checkWithinSensing(own::Ownship, int::Ownship)
	if distance(own, int) < SENSING_RANGE * 1.02
		return true
	else
		return false
	end
end

function checkDest(ac::Ownship)
	if sqrt((ac.x - ac.x_dest)^2 + (ac.y - ac.y_dest)^2) < 50
		return true
	else
		return false
	end
end

function getIntruderStartDestCoords(ownship::Ownship)
	# Use encounter model:
	intruder_dist = Categorical(vec(BEARING_DIST[:, 2]))
	intruder_angle = BEARING_DIST[:, 1][rand(intruder_dist)]
	if intruder_angle > π
		intruder_angle -= 2 * π
	end # intruder_angle ∈ [-pi, pi], centered at 0
	init_angle = norm_angle(ownship.heading + intruder_angle) 
	dest_angle = norm_angle(init_angle + deg2rad(rand() * 270 + 45)) # 45 - 315 uniform?
	x_init = ownship.x + cos(init_angle) * SENSING_RANGE
	y_init = ownship.y + sin(init_angle) * SENSING_RANGE
	x_dest = ownship.x + cos(dest_angle) * SENSING_RANGE * 1.2
	y_dest = ownship.y + sin(dest_angle) * SENSING_RANGE * 1.2
	return x_init, y_init, x_dest, y_dest
end

function addIntruders!(sr::Float64, ownship::Ownship, intArr::Vector{Ownship}, CASType::Symbol, policy)
	numNewIntruders = rand(Poisson(sr))
	for i = 1:numNewIntruders
		x0, y0, x_dest, y_dest = getIntruderStartDestCoords(ownship)
		ac = initOwnship(x0, y0, x_dest, y_dest, CASType, policy)
		push!(intArr, ac)
	end
	return numNewIntruders
end

function reset(sr::Float64, CASType::Symbol, policy)
	x0 = 0.0
	y0 = 0.0
	x_dest = 10000.0
	y_dest = 0.0
	ownship = initOwnship(x0, y0, x_dest, y_dest, CASType, policy)
	intArr = Ownship[]
	addIntruders!(sr, ownship, intArr, CASType, policy)
	return ownship, intArr
end

function step!(sr::Float64, ownship::Ownship, intArr::Vector{Ownship}, CASType::Symbol, policy)
	stepResTime = length(intArr) * dt
	
	acArr = vcat(ownship, intArr)

	for i in 1 : length(acArr)
		intCoords = Vector{Float64}()
		for j in 1 : length(acArr)
			if j != i
				intCoords = vcat(intCoords, [acArr[j].x, acArr[j].y, acArr[j].heading])
			end
		end

		numInt = length(intArr)
		if numInt > 0 && CASType != :NOCAS
			# Get state
			numObservedInt = getState!(acArr[i], numInt, intCoords)
			if numObservedInt > 0
				# Get action and advisory
				getAdvisory!(acArr[i])
			else
				acArr[i].advisory = :COC
			end
		else
			acArr[i].advisory = :COC
		end
	end

	# Update dynamics
	for ac in acArr
		updateOwnshipCoords!(ac)
	end

	# Check within sensing and check dest for intruders
	i = 1
	while i <= length(intArr)
		if checkDest(intArr[i]) || !checkWithinSensing(ownship, intArr[i])
		   deleteat!(intArr, i) 
		end
		i += 1
	end

	numNewIntruders = addIntruders!(sr, ownship, intArr, CASType, policy)

	return stepResTime, numNewIntruders
end


function render_env(sr::Float64, t::Int64, ownship::Ownship, intruders::Vector{Ownship})
	All_AC = vcat(ownship, intruders)
	AIRSPACE_DIM = 1500
	num_sections = ownship.policy.env.num_sections
	for ac in All_AC
		if ac.advisory == :COC
			scatter(ac.x, ac.y, marker="o", color="blue", s=12)
		else
			scatter(ac.x, ac.y, marker="o", color="red", s=12)
		end

		arrow_len = AIRSPACE_DIM / 6
		arrow(
			ac.x, ac.y,
			cos(ac.heading) * arrow_len * 0.8,
			sin(ac.heading) * arrow_len * 0.8,
			head_width=AIRSPACE_DIM / 20,
			width=1,
			head_length=AIRSPACE_DIM / 20,
			overhang=0.5,
			head_starts_at_zero="true",
			facecolor="black",
			length_includes_head="true")

		scatter(ac.x_dest, ac.y_dest, marker=",", color="magenta", s=12)

		PyPlot.plot([ac.x; ac.x_dest], [ac.y; ac.y_dest], 
			linestyle="--", color="black", linewidth=0.5)
	end

	# alert circle
	th = linspace(-pi, pi, 30)
	if ownship.advisory == :COC
		PyPlot.plot(ownship.x + SENSING_RANGE * cos.(th), 
			ownship.y + SENSING_RANGE * sin.(th), 
			linestyle="--", color="green", linewidth=0.4)
	else
		PyPlot.plot(ownship.x + SENSING_RANGE * cos.(th), 
			ownship.y + SENSING_RANGE * sin.(th), 
			linestyle="--", color="red", linewidth=0.4)
	end

	# NMAC circle
	PyPlot.plot(ownship.x + NMAC_RANGE * cos.(th), 
			ownship.y + NMAC_RANGE * sin.(th), 
			linestyle="--", color="red", linewidth=0.4)

	# plot sections and highlight the closest intruder in each section
	section_angle = 2 * pi / num_sections
	for i in 0 : num_sections / 2 - 1
		th_1 = i * section_angle + ownship.heading
		th_2 = th_1 + pi
		PyPlot.plot(ownship.x + SENSING_RANGE * [cos(th_1), cos(th_2)], ownship.y + SENSING_RANGE * [sin(th_1), sin(th_2)], 
			linestyle="--", color="red", linewidth=0.4)
	end

	# for i in 1 : env.num_sections
	# 	rho = env.full_state[(i - 1) * STATE_DIM + IND_rho]
	# 	if rho != TERM_VAR
	# 		theta = env.full_state[(i - 1) * STATE_DIM + IND_theta]
	# 		scatter(rho * cos(theta), rho * sin(theta), 
	# 			marker="o", color="green", s=40, linewidths=0.6)
	# 	end
	# end

	xlabel("X (m)")
	ylabel("Y (m)")
	title(string(sr) * ", " * string(t) * ": " * string(length(intruders)))
	axis("equal")
	# ax = gca()
	# ax[:set_xlim]([-AIRSPACE_DIM * 1.5, AIRSPACE_DIM * 1.5])
	# ax[:set_ylim]([-AIRSPACE_DIM * 1.5, AIRSPACE_DIM * 1.5])
	pause(0.1)
	clf()
end



function resTime(;
				spawnRateSet::Union{Vector{Int64},Int64}=[25, 35, 45],
				CASType::Symbol=:correctedSector,
				policyFile::AbstractString="",
				numEvalEposide::Int64=50,
				maxEpisodeLen::Int64=2000,
				render::Bool=false)
	
	println("CASType = ", string(CASType))
	avgResTimeArr = Dict{Float64, Float64}()
	stdResTimeArr = Dict{Float64, Float64}()
	problem_file = "../../DeepCorrection/logs/" * policyFile * "/final_problem.jld"
	weights_file = "../../DeepCorrection/logs/" * policyFile * "/final_weights.jld"
	policy = restore(problem_file=problem_file, weights_file=weights_file)

	for sr in spawnRateSet
		sr /= 100
		epResTimeArr = Float64[]

		for ep in 1 : numEvalEposide
			# initiate ownship and intruders (reset)
			ownship, intArr = reset(sr, CASType, policy)
			totalResTime = 0.0
			totalNumIntruders = 0
			numNewIntruders = length(intArr)
			totalNumIntruders += numNewIntruders

			for t = 1 : maxEpisodeLen
				stepResTime, numNewIntruders = step!(sr, ownship, intArr, CASType, policy)
				totalResTime += stepResTime
				totalNumIntruders += numNewIntruders
				if render
					render_env(sr, t, ownship, intArr)
				end
				if checkDest(ownship) # ownship reaches destination
					break
				end
			end # for t
			push!(epResTimeArr, totalResTime / totalNumIntruders)
		end # for ep

		avgResTimeArr[sr] = mean(epResTimeArr)
		stdResTimeArr[sr] = std(epResTimeArr)
		println("sr = ", sr, ", avgResTime = ", avgResTimeArr[sr])
	end # for sr
	return avgResTimeArr, stdResTimeArr
end

