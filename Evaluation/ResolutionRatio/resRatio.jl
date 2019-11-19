# using Distributions, PyPlot
push!(LOAD_PATH, "../../VICAS/src/")
push!(LOAD_PATH, "../../DeepCorrection")
push!(LOAD_PATH, "../../DeepCorrection/src")
push!(LOAD_PATH, "../TrajectoryViz")

include("../TrajectoryViz/trajViz.jl")

th = linspace(0, 2*pi, 50)


conflictRange = SENSING_RANGE * 1.5

function check_NMAC(i, acArr, finished)
	ac = acArr[i]
	for j in 1:length(acArr)
		if j != i && finished[i] == false && finished[j] == false
			d = sqrt((ac.x - acArr[j].x)^2 + (ac.y - acArr[j].y)^2)
			if d <= NMAC_RANGE
				acArr[i].dmin = d
				acArr[j].dmin = d
				return true, d
			end
		end
	end
	return false, Inf
end

function check_colli(ac, acArr)
	for existing_ac in acArr
		if sqrt((ac.x - existing_ac.x)^2 + (ac.y - existing_ac.y)^2) <= NMAC_RANGE
			return true
		end
	end
end

function create_ac(spawn_range, CASType, policy)
	# Annulus type, centralized encounter
	spawn_range = rand(2000:4000)
	start_angle = rand(th)
	x0 = spawn_range * cos(start_angle)
	y0 = spawn_range * sin(start_angle)
	# conflictPoint_x = conflictRange * (- 2 * rand() + 1)
	# conflictPoint_y  = sqrt(conflictRange^2 - conflictPoint_x^2) * (- 2 * rand() + 1)
	# dest_angle = atan2(conflictPoint_y - y0, conflictPoint_x - x0)
	# dest_dist = (4000 - 2 * spawn_range) * rand() + 2 * spawn_range
	dest_dist = rand(2000:4000)
	# x_dest = x0 + cos(dest_angle) * dest_dist
	# y_dest = y0 + sin(dest_angle) * dest_dist
	x_dest = dest_dist * cos(start_angle + pi)
	y_dest = dest_dist * sin(start_angle + pi)

	# Circular type encounter
	# x0 = spawn_range * (- 2 * rand() + 1)
	# y0max  = sqrt(spawn_range^2 - x0^2)
	# y0     = y0max * (- 2 * rand() + 1)
	# dest_angle = rand(th)
	# dest_dist = 2*spawn_range
	# x_dest = cos(dest_angle) * dest_dist
	# y_dest = sin(dest_angle) * dest_dist

	# dest_angle = rand(th)
	# x_dest = spawn_range * cos(dest_angle)
	# y_dest = spawn_range * sin(dest_angle)
	ac = initOwnship(x0, y0, x_dest, y_dest, CASType, policy)
	return ac
end

function resRatio(;
				CASType::Symbol=:correctedSector,
				policyFile::AbstractString="",
				numAC::Int64=2,
				numSim::Int64=10000,
				timeout::Int64=200,
				correction_weight::Float64=0.09)

	problem_file = "../../DeepCorrection/logs/" * policyFile * "/final_problem.jld"
	weights_file = "../../DeepCorrection/logs/" * policyFile * "/final_weights.jld"
	policy = restore(problem_file=problem_file, weights_file=weights_file)
	policy.env.correction_weight = correction_weight
	spawn_range = SENSING_RANGE * 1.5
	resolvedStats = 0
	unresolved_NMAC_Stats = Int64[]
	unresolved_Timeout_Stats = 0
	resTimeArr = Vector{Float64}()
	avgResTime = 0
	criticality = Float64[]
	v_closing_v_cruise = Float64[]

	for sim in 1 : numSim
		# Initiate
		t = 1
		NMACFlag = false
		fail = false
		finished = falses(numAC)
		acArr = Ownship[]
		for i = 1 : numAC
			ac = create_ac(spawn_range, CASType, policy)
			while check_colli(ac, acArr) == true
				ac = create_ac(spawn_range, CASType, policy)
			end
			push!(acArr, ac)
		end

		push!(unresolved_NMAC_Stats, 0)
	
		while !all(finished) && NMACFlag == false
			for i in 1 : numAC
				if !finished[i]
					ownship = acArr[i]
					# Get intruder coords for ownship
					intCoords = Vector{Float64}()
					for j in 1 : length(acArr)
						if j != i && !finished[j] # not ownship and not finished
							intCoords = vcat(intCoords, [acArr[j].x, acArr[j].y], acArr[j].heading)
						end
					end
					numInt = Int(length(intCoords) / 3)
	
					if numInt > 0 && CASType != :NOCAS
						# Get state
						numObservedInt = getState!(ownship, numInt, intCoords)
						if numObservedInt > 0
							# Get action and advisory
							getAdvisory!(ownship)
						else
							ownship.advisory = :COC
						end
					else
						ownship.advisory = :COC
					end
					# print(i, " : ", ownship.state, " : ", ownship.advisory, "\n")
				end # if !finished[i]
			end # for
	
			for i in 1 : numAC
				if !finished[i]
					dest_angle = norm_angle(atan2(acArr[i].y_dest - acArr[i].y, acArr[i].x_dest - acArr[i].x))
					heading_error = norm_angle(dest_angle - acArr[i].heading)
					push!(v_closing_v_cruise, cos(heading_error))
					updateOwnshipCoords!(acArr[i])
				end
	
				if sqrt((acArr[i].x - acArr[i].x_dest)^2 + (acArr[i].y - acArr[i].y_dest)^2) < 80
					finished[i] = true
				end
			end # for
	
			for i in 1 : numAC
				nmac, d = check_NMAC(i, acArr, finished)
				if finished[i] == false && nmac == true # FAIL: NMAC
					push!(criticality, max(0.0, 1 - acArr[i].dmin / NMAC_RANGE))
					fail = true
					NMACFlag = true
					unresolved_NMAC_Stats[sim] += 1
				end
			end
	
			if all(finished) && NMACFlag == false # SUCCESS
				resolvedStats += 1
			end
	
			t += 1
			if t > timeout && NMACFlag == false # FAIL: timeout
				fail = true
				unresolved_Timeout_Stats += 1
				break # while
			end

		end #while 

		if t <= timeout && NMACFlag == false # only count SUCCESS cases
			avgResTime += t
		end
	end # for sim

	return resolvedStats, unresolved_NMAC_Stats ./ 2, unresolved_Timeout_Stats, avgResTime / resolvedStats, criticality, v_closing_v_cruise
end


