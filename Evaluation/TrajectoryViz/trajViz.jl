#=
	This is a file for trajectories visualization in Jupyter Notebook.
=#

using PGFPlots, CSV
push!(LOAD_PATH, "../../VICAS/src/")
push!(LOAD_PATH, "../../DeepCorrection")
push!(LOAD_PATH, "../../DeepCorrection/src")
using DeepQLearning
# using Simulation_Env_module_nnpolicy
using Simulation_Env_module

# Constants
const v = 50
const dt = 1
#

mutable struct Ownship 
	x::Float64
	y::Float64
	v::Float64
	heading::Float64
	x_dest::Float64
	y_dest::Float64
	dist_to_dest::Float64
	CASType::Symbol
	policy # Any
	state::Vector{Float64}
	advisory::Symbol
	action::Float64
	dmin::Float64
end

mutable struct Intruder
	x::Float64
	y::Float64
	v::Float64
	heading::Float64
end


"""
	keeps the angles in [0, 2*pi]
"""
function norm_angle(angle::Float64)
	return ((angle % (2 * pi)) + 2 * pi) % (2 * pi)
end


function initOwnship(x0, y0, x_dest, y_dest, CASType, policy)
	dist_to_dest = sqrt(x_dest^2 + y_dest^2)
	heading0 = norm_angle(atan2(y_dest - y0, x_dest - x0))

	if CASType == :correctedSector || CASType ==  :correctedClosest
		assert(policy.env.correction == true)
	elseif CASType == :VICASMulti || CASType == :VICASMultiWeighted || CASType ==  :VICASClosest
		policy.env.correction = false
		assert(policy.env.correction == false)
	end

	state = zeros(STATE_DIM)
	advisory = :COC
	action = policy.env.advisory_to_action_dict[:COC]

	ownship = Ownship(x0, y0, v, heading0, x_dest, y_dest, dist_to_dest, 
					CASType, policy, state, advisory, action, Inf)
	return ownship
end

"""
	overload :< operator for the Aircraft object
"""
function closer(a::Union{Intruder, Ownship}, b::Union{Intruder, Ownship}, ref::Ownship)
	if (a.x - ref.x)^2 + (a.y - ref.y)^2 < (b.x - ref.x)^2 + (b.y - ref.y)^2
		return true
	else
		return false
	end
end

"""
	overload :> operator for the Aircraft object
"""
function farther(a::Union{Intruder, Ownship}, b::Union{Intruder, Ownship}, ref::Ownship)
	if (a.x - ref.x)^2 + (a.y - ref.y)^2 > (b.x - ref.x)^2 + (b.y - ref.y)^2
		return true
	else
		return false
	end
end

"""
	sort the Aircraft in an array, wrt distance to the ego agent
	using quick sort, returns ascending order in terms of the distance to the ego agent
"""
function sort_agents!(ref::Ownship, agents_arr::Union{Vector{Intruder}, Vector{Ownship}}, lo::Int64, hi::Int64)
    i, j = lo, hi
    while i < hi
        pivot = agents_arr[(lo + hi) >>> 1]
        while i <= j
            while closer(agents_arr[i], pivot, ref); i += 1; end
            while farther(agents_arr[j], pivot, ref); j -= 1; end
            if i <= j
                agents_arr[i], agents_arr[j] = agents_arr[j], agents_arr[i]
                i, j = i + 1, j - 1
            end
        end
        if lo < j; sort_agents!(ref, agents_arr, lo, j); end
        lo, j = i, hi
    end
end

function get_pairwise_state(ownship::Ownship, intruder::Union{Intruder, Ownship})
	intruder_pos_angle = atan2(intruder.y - ownship.y, intruder.x - ownship.x)

	return [sqrt((ownship.x - intruder.x)^2 + (ownship.y - intruder.y)^2), 
			norm_angle(intruder_pos_angle - ownship.heading), 
			norm_angle(intruder.heading - ownship.heading), 
			ownship.v, 
			intruder.v]
end

function get_section_id(ownship::Ownship, int::Union{Intruder, Ownship}, num_sections::Int64)
	ac_pos_angle = norm_angle(norm_angle(atan2(int.y, int.x)) - norm_angle(ownship.heading))
	return section_id = floor(ac_pos_angle / (2 * pi / num_sections)) + 1
end

function getState!(ownship::Ownship, numInt::Int64, intCoords::Vector{Float64})
	num_sections = ownship.policy.env.num_sections
	if ownship.CASType == :correctedSector || 
		ownship.CASType == :VICASMulti || ownship.CASType == :VICASMultiWeighted ||
		ownship.CASType == :VICASClosest ||
		ownship.CASType == :correctedClosest || ownship.CASType == :nnCAS

		full_state = zeros(num_sections * STATE_DIM + AUG_STATE_DIM)

		# Construct intruder array
		intArr = Intruder[]
		for i in 1:numInt
			x = intCoords[3 * i - 2]
			y = intCoords[3 * i - 1]
			if sqrt((x - ownship.x)^2 + (y - ownship.y)^2) <= SENSING_RANGE
				heading = norm_angle(intCoords[3 * i])
				push!(intArr, Intruder(x, y, v, heading))
			end
		end

		# Identify the sector ID for each intruder
		# # Method using num_sections sections
		section_ac_arr = Dict{Int64, Vector{Intruder}}()
		for isection in 1:num_sections
			section_ac_arr[isection] = Intruder[]
		end
	
		all_int_state = Float64[]
		for int in intArr
			all_int_state = vcat(all_int_state, get_pairwise_state(ownship, int))
			section_id = get_section_id(ownship, int, num_sections)
			push!(section_ac_arr[section_id], int)
		end

		if ownship.CASType == :correctedSector || ownship.CASType == :VICASClosest
			# sort ac in each section
			for isection in 1:num_sections
				hi = length(section_ac_arr[isection])
				lo = (hi == 0) ? 0 : 1
				if hi > 0 
					sort_agents!(ownship, section_ac_arr[isection], lo, hi)
					# resulting section_ac_arr is in ascending order,
					# grab ac that is closest to the ego ac for each section
					# to update the full_state
					full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
						get_pairwise_state(ownship, section_ac_arr[isection][1]) # [m]
				else
					# no intruder in the section
					# set the corresponding state var to terminal values
					full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
						[TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR, TERM_VAR]
				end
			end
			closest_intruder_section_id = indmin(full_state[[(i - 1) * STATE_DIM + IND_rho for i in 1:num_sections]])
			lowfi_state = full_state[STATE_DIM * (closest_intruder_section_id - 1) + 1 : STATE_DIM * closest_intruder_section_id]
		
		elseif ownship.CASType == :correctedClosest || ownship.CASType == :nnCAS
			hi = length(intArr)
			lo = (hi == 0) ? 0 : 1
			
			if hi > 0 
				# Ascending order in terms of distance from ownship
				sort_agents!(ownship, intArr, lo, hi) 
		
				for i in 1 : num_sections
					if i <= length(intArr)
						full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] .= get_pairwise_state(ownship, intArr[i])
					else
						full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] .= terminal_state
					end
				end
			else # no intruder
				full_state[1 : STATE_DIM * num_sections] .= TERM_VAR
			end
		end
	
		# append the destination information to the full_state
		# 1. deviation from the destination, measured by rad
		deviation = norm_angle(ownship.heading - 
			norm_angle(atan2(ownship.y_dest - ownship.y, ownship.x_dest - ownship.x)))
		if deviation > π
			deviation = deviation - 2 * π # ∈ [-π, π]
			# deviation = 2 * π - deviation # ∈ [0, π], deprecated
		end 
	
		# 2. current distance to destination
		dist_to_dest = sqrt((ownship.y - ownship.y_dest)^2 + (ownship.x - ownship.x_dest)^2) # [ft]
		# 3. previous distance to destination
		prev_dist_to_dest = ownship.dist_to_dest # [ft]
		## update ego_agent_dist_to_dest
		ownship.dist_to_dest = dist_to_dest # [ft]
	
		# update state entries
		full_state[end - AUG_STATE_DIM + 1 : end] .= [deviation, dist_to_dest, prev_dist_to_dest] # [m]
		
		# find the lowfi_state from the full_state
		# closest_intruder_section_id = indmin(full_state[[(i - 1) * STATE_DIM + IND_rho for i in 1:num_sections]])
		# lowfi_state = full_state[STATE_DIM * (closest_intruder_section_id - 1) + 1 : STATE_DIM * closest_intruder_section_id]
		
		normalize_obs!(ownship.policy.env, full_state)

		if ownship.CASType == :correctedSector || ownship.CASType == :correctedClosest || ownship.CASType == :nnCAS
			ownship.state = full_state # normalized
		elseif ownship.CASType == :VICASMulti || ownship.CASType == :VICASMultiWeighted
			ownship.state = all_int_state # non-normalized
		elseif ownship.CASType == :VICASClosest
			ownship.state = lowfi_state # non-normalized
		end

		return length(intArr)

	end # if ownship.CASType
end

function getAdvisory!(ownship::Ownship)
	if ownship.CASType == :correctedSector || ownship.CASType == :correctedClosest
		low_fi_state = zeros(STATE_DIM) # low_fi_state is not used, give it a dummy value
		ownship.advisory = get_action(ownship.policy, ownship.state, low_fi_state)
		ownship.action = ownship.policy.env.advisory_to_action_dict[ownship.advisory]
	elseif ownship.CASType == :nnCAS
		low_fi_state = zeros(STATE_DIM) # low_fi_state is not used, give it a dummy value
		s = ownship.state[vcat(1:3, 6:8, 11:13, 16:18)]
		ownship.advisory = get_action(ownship.policy, s, low_fi_state)
		ownship.action = ownship.policy.env.advisory_to_action_dict[ownship.advisory]

	elseif ownship.CASType == :VICASMulti || ownship.CASType == :VICASMultiWeighted
		q_matrix = Matrix{Float64}(6, 0)
		for i in 1 : Int(length(ownship.state) / STATE_DIM)
			state = ownship.state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i]
			if ownship.CASType == :VICASMulti
				qvals = get_qvals(ownship.policy.env.low_fi_policy, state)
			else
				qvals = get_qvals(ownship.policy.env.low_fi_policy, state) * (state[IND_rho] - SENSING_RANGE) / (NMAC_RANGE - SENSING_RANGE)
			end
     		q_matrix = hcat(q_matrix, qvals)
		end
		min_qvals = minimum(q_matrix, 2)
		the_ind = indmax(min_qvals)
		ownship.advisory =ownship.policy.env.action_space[the_ind]
		ownship.action = ownship.policy.env.advisory_to_action_dict[ownship.advisory]

	elseif ownship.CASType == :VICASClosest
		qvals = get_qvals(ownship.policy.env.low_fi_policy, ownship.state)
		ownship.advisory = ownship.policy.env.action_space[indmax(qvals)]
		ownship.action = ownship.policy.env.advisory_to_action_dict[ownship.advisory]

	elseif ownship.CASType == :NOCAS
		ownship.advisory = :COC
		ownship.action = ownship.policy.env.advisory_to_action_dict[ownship.advisory]
	end
end

function toDest!(ac::Ownship)
	# set threshold for confirming the correct heading towards destination
	threshold = deg2rad(8.)

	heading_vec = [cos(ac.heading); sin(ac.heading); 0.]
	desti_dir_angle = norm_angle(atan2(ac.y_dest - ac.y, ac.x_dest - ac.x))
	heading_error = abs(desti_dir_angle - ac.heading)
	# println("heading err = ", heading_error)

	if heading_error < threshold
		ac.action = 0.
		ac.heading = desti_dir_angle
		# println("here")
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
		# println("side = ", side)
		# get turn rate
		if side == :Right
			ac.action = ac.policy.env.advisory_to_action_dict[:SR]
		elseif side == :Left
			ac.action = ac.policy.env.advisory_to_action_dict[:SL]
		else # dest on heading
			ac.action = 0. # keep heading
		end
		# update heading
		ac.heading += ac.action * dt
		ac.heading = norm_angle(ac.heading)
	end
	# update coordinates
	ac.x += cos(ac.heading) * ac.v * dt
	ac.y += sin(ac.heading) * ac.v * dt
end

function updateOwnshipCoords!(ac::Ownship)
	if ac.advisory != :COC
		ac.heading += ac.action * dt
		ac.heading = norm_angle(ac.heading)
		ac.x += cos(ac.heading) * ac.v * dt
		ac.y += sin(ac.heading) * ac.v * dt
	else
		# println("toDest")
		toDest!(ac)
	end
end

"""
	Non-interactive
	Intruder trajectories predifined
"""
function trajViz(;
				x0::Float64=0.0,
				y0::Float64=0.0,
				x_dest::Float64=10000.0,
				y_dest::Float64=0.0, 
				CASType::Symbol=:correctedSector,
				policyFile::AbstractString="",
				intTrajFile::AbstractString="intTraj.csv",
				figName::AbstractString="",
				colorBar::Bool=false,
				figSize::Float64=5.0,
				legend::Bool=false,
				correction_weight::Float64=0.0)

	problem_file = "../../DeepCorrection/logs/" * policyFile * "/final_problem.jld"
	weights_file = "../../DeepCorrection/logs/" * policyFile * "/final_weights.jld"
	policy = restore(problem_file=problem_file, weights_file=weights_file)
	policy.env.correction_weight = correction_weight

	# Load predefined intruder trajectories
	if intTrajFile != ""
		intTrajFile = "./coordsFiles/" * intTrajFile
		intTraj = convert(Matrix{Float64}, CSV.read(intTrajFile, datarow=1))
	else
		exit()
	end
	numInt = Int(size(intTraj, 2) / 3)
	maxTimeSteps = size(intTraj, 1)

	# Initiate onwship trajectory (x, y, a), time step given by row number
	ownTraj = Matrix{Float64}(maxTimeSteps, 3)

	# Initiate Ownship
	ownship = initOwnship(x0, y0, x_dest, y_dest, CASType, policy)

	for t = 1 : maxTimeSteps
		# Get intruder coords
		intCoords = intTraj[t, :] # returns Vector
		# Get state
		if CASType != :NOCAS
			numObservedInt = getState!(ownship, numInt, intCoords)
		else
			numObservedInt = 0
		end

		if numObservedInt > 0 && CASType != :NOCAS
			# Get action and advisory
			getAdvisory!(ownship)
		else
			ownship.advisory = :COC
		end
		# println(t, " ", numObservedInt)
		# update ownship coords
		updateOwnshipCoords!(ownship)
		# Append ownship coords to ownTraj
		ownTraj[t, 1:2] = [ownship.x ownship.y]
		ownTraj[t, 3] = rad2deg(ownship.action)
		if sqrt((ownship.x - ownship.x_dest)^2 + (ownship.y - ownship.y_dest)^2) < 50
			ownTraj = ownTraj[1:t, :]
			# println(ownTraj)
			break
		end
	end
	
	# Plot trajectories
	# println(ownTraj)
	ownTrajLen = size(ownTraj, 1)
	distanceMatrix = Matrix{Float64}(ownTrajLen, numInt)
	for i = 1:numInt
		delta = ownTraj[:, 1:2] - intTraj[1:ownTrajLen, 3*i-2 : 3*i-1]
		distanceMatrix[:,i] = sqrt.(delta[:,1].^2 + delta[:,2].^2)
	end

	actionArr = vec(ownTraj[:,3])
	if length(unique(actionArr)) > 1
		if colorBar == true
			colorbarOption = true
		else
			colorbarOption = false
		end
		actionsLeg = "Ownship Actions"
	else
		colorbarOption = false
		actionsLeg = "Ownship Action = COC"
	end

	if legend == true
		legCommand = ""
	else
		legCommand = "\\legend{}"
	end

	timeTicks = [10 * i for i in 1:Int(floor(min(maxTimeSteps, ownTrajLen) / 10))]

	# g = GroupPlot(1, 2, groupStyle = "vertical sep = 2cm")

	trajFig = Axis([
    	Plots.Linear(vec(intTraj[:,1]), vec(intTraj[:,2]), style="forget plot, red, very thick", mark="none"),
    	Plots.Linear(vec(intTraj[:,4]), vec(intTraj[:,5]), style="forget plot, magenta, very thick", mark="none"),
    	Plots.Linear(vec(intTraj[:,7]), vec(intTraj[:,8]), style="forget plot, pink, very thick", mark="none"),
    	Plots.Linear(vec(ownTraj[:,1]), vec(ownTraj[:,2]), mark="none", style="forget plot, black, very thick"),
    	Plots.Scatter(vec(ownTraj[:,1]), vec(ownTraj[:,2]), vec(ownTraj[:,3]), legendentry=actionsLeg, mark="*", markSize=1.2),

    	Plots.Scatter(0, 0, legendentry="Start", style="white, mark options={fill=black},mark=square*", markSize=3),
    	Plots.Scatter(x_dest, y_dest, legendentry="Destination", style="white, mark options={fill=green},mark=square*", markSize=3),
    	Plots.Scatter(intTraj[1,1], intTraj[1,2], style="forget plot,white, mark options={fill=black},mark=square*", markSize=3),
    	Plots.Scatter(intTraj[end,1], intTraj[end,2], style="forget plot,white, mark options={fill=green},mark=square*", markSize=3),
    	Plots.Scatter(intTraj[1,4], intTraj[1,5], style="forget plot,white, mark options={fill=black},mark=square*", markSize=3),
    	Plots.Scatter(intTraj[end,4], intTraj[end,5], style="forget plot,white, mark options={fill=green},mark=square*", markSize=3),
    	Plots.Scatter(intTraj[1,7], intTraj[1,8], style="forget plot,white, mark options={fill=black},mark=square*", markSize=3),
    	Plots.Scatter(intTraj[end,7], intTraj[end,8], style="forget plot,white, mark options={fill=green},mark=square*", markSize=3),

    	Plots.Scatter(vec(intTraj[timeTicks,1]), vec(intTraj[timeTicks,2]), style="very thick,solid,red,mark=diamond", legendentry="Intruder 1", markSize=3),
    	Plots.Scatter(vec(intTraj[timeTicks,4]), vec(intTraj[timeTicks,5]), style="very thick,solid,magenta,mark=diamond", legendentry="Intruder 2", markSize=3),
    	Plots.Scatter(vec(intTraj[timeTicks,7]), vec(intTraj[timeTicks,8]), style="very thick,solid,pink,mark=diamond", legendentry="Intruder 3", markSize=3),
    	Plots.Scatter(vec(ownTraj[timeTicks,1]), vec(ownTraj[timeTicks,2]), style="very thick,solid,black,mark=diamond", legendentry="Ownship Trajectory", markSize=3),
    	Plots.Command(legCommand)
    	],
    	colorbar=colorbarOption,
    	legendStyle="{at={(0.02,0.02)}, anchor=south west}",
    	title=string(CASType),
    	xlabel="x (m)",
    	ylabel="y (m)",
    	width=string(figSize) * "in", height=string(figSize) * "in", axisEqual=true
    ) # Axis

	minDistance = minimum(vcat(vec(distanceMatrix[:,1]), vec(distanceMatrix[:,2]), vec(distanceMatrix[:,3])))
    distFig = Axis([
    	Plots.Linear(1:ownTrajLen, vec(distanceMatrix[:,1]), legendentry="Intruder 1", style="red, very thick", mark="none"),
    	Plots.Linear(1:ownTrajLen, vec(distanceMatrix[:,2]), legendentry="Intruder 2", style="magenta, very thick", mark="none"),
    	Plots.Linear(1:ownTrajLen, vec(distanceMatrix[:,3]), legendentry="Intruder 3", style="pink, very thick", mark="none"),
    	Plots.Linear(1:ownTrajLen, ones(ownTrajLen)*NMAC_RANGE, legendentry="NMAC", style="black, very thick", mark="none"),
    	Plots.Command(legCommand)
    	],
    	legendStyle="{at={(0.05,0.95)}, anchor=north west}",
    	title="Minimum Distance = " * string(minDistance)[1:6] * " m",
    	xlabel="Time Step (s)",
    	ylabel="Distance (m)",
    	width=string(figSize) * "in", height="2in"
    ) # Axis

    return trajFig, distFig
end


"""
	Interactive
	Start and destination are given
	Free flight
"""
function interactiveTrajViz(;
				CASType::Symbol=:correctedSector,
				policyFile::AbstractString="",
				startDestCoordsFile::AbstractString="",
				figName::AbstractString="",
				colorBar::Bool=false,
				figSize::Float64=5.0,
				legend::Bool=false,
				ylabel::Bool=false,
				correction_weight::Float64=0.0,
				show_action::Bool=true)

	problem_file = "../../DeepCorrection/logs/" * policyFile * "/final_problem.jld"
	weights_file = "../../DeepCorrection/logs/" * policyFile * "/final_weights.jld"
	# problem_file = "../../DeepCorrection/logs/" * policyFile * "/problem.jld"
	# weights_file = "../../DeepCorrection/logs/" * policyFile * "/weights.jld"
	policy = restore(problem_file=problem_file, weights_file=weights_file)
	policy.env.correction_weight = correction_weight

	if startDestCoordsFile != ""
		startDestCoordsFile = "./coordsFiles/" * startDestCoordsFile
		startDestCoords = convert(Matrix{Float64}, CSV.read(startDestCoordsFile, datarow=1))
	else
		exit()
	end
	numAC = size(startDestCoords, 2)
	# Initiate
	t = 1
	finished = falses(numAC)
	acArr = Ownship[]
	ownTraj = Dict{Int64, Matrix{Float64}}()
	for i = 1 : numAC
		x0     = startDestCoords[1, i]
		y0     = startDestCoords[2, i]
		x_dest = startDestCoords[3, i]
		y_dest = startDestCoords[4, i]
		ac = initOwnship(x0, y0, x_dest, y_dest, CASType, policy)
		push!(acArr, ac)
		ownTraj[i] = Matrix{Float64}(0, 3)
	end

	while !all(finished)
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
				updateOwnshipCoords!(acArr[i])
				ownTraj[i] = vcat(ownTraj[i], [acArr[i].x acArr[i].y rad2deg(acArr[i].action)])
			end

			if sqrt((acArr[i].x - acArr[i].x_dest)^2 + (acArr[i].y - acArr[i].y_dest)^2) < 70
				finished[i] = true
			end
		end # for

		t += 1
		if t > 1000
			break
		end
	end #while 

	if legend == true
		legCommand = ""
	else
		legCommand = "\\legend{}"
	end

	if ylabel == true
		trajYlabelStr = "y (m)"
		distYlabelStr = "Distance (m)"
	else
		trajYlabelStr = ""
		distYlabelStr = ""
	end

	if numAC == 2
		distanceDim = min(size(ownTraj[1], 1), size(ownTraj[2], 1))
		distance = Matrix{Float64}(distanceDim, 1)
		delta = ownTraj[1][1:distanceDim, 1:2] - ownTraj[2][1:distanceDim, 1:2]
		distance = sqrt.(delta[:,1].^2 + delta[:,2].^2)

		timeTicks1 = [10 * i for i in 1:Int(floor(size(ownTraj[1], 1) / 10))]
		timeTicks2 = [10 * i for i in 1:Int(floor(size(ownTraj[2], 1) / 10))]

		actionArr = vcat(vec(ownTraj[1][:,3]), vec(ownTraj[2][:,3]))
		if length(unique(actionArr)) > 1
			if colorBar == true
				colorbarOption = true
			else
				colorbarOption = false
			end
			actionsLeg = "Actions"
		else
			colorbarOption = false
			actionsLeg = "Action = COC"
		end
	
		# g = GroupPlot(1, 2, groupStyle = "vertical sep = 2cm")
	
		tragFig = Axis([
    		Plots.Scatter(vec(ownTraj[1][:,1]), vec(ownTraj[1][:,2]), vec(ownTraj[1][:,3]), legendentry=actionsLeg, mark="*", markSize=1.2),
    		Plots.Linear(vec(ownTraj[1][:,1]), vec(ownTraj[1][:,2]), mark="none", style="forget plot, red, very thick"),
    		Plots.Scatter(vec(ownTraj[2][:,1]), vec(ownTraj[2][:,2]), vec(ownTraj[2][:,3]), mark="*", markSize=1.2, style="forget plot"),
    		Plots.Linear(vec(ownTraj[2][:,1]), vec(ownTraj[2][:,2]), mark="none", style="forget plot, blue, very thick"),

    		Plots.Scatter(startDestCoords[1, 1], startDestCoords[2, 1], legendentry="Start", style="white,mark options={fill=black},mark=square*", markSize=3),
    		Plots.Scatter(startDestCoords[3, 1], startDestCoords[4, 1], legendentry="Destination", style="white,mark options={fill=green},mark=square*", markSize=3),
    		Plots.Scatter(startDestCoords[1, 2], startDestCoords[2, 2], style="forget plot,white,mark options={fill=black},mark=square*", markSize=3),
    		Plots.Scatter(startDestCoords[3, 2], startDestCoords[4, 2], style="forget plot,white,mark options={fill=green},mark=square*", markSize=3),

    		Plots.Scatter(vec(ownTraj[1][timeTicks1,1]), vec(ownTraj[1][timeTicks1,2]), style="solid,red,mark=diamond", legendentry="AC 1 Trajectory", markSize=3),
    		Plots.Scatter(vec(ownTraj[2][timeTicks2,1]), vec(ownTraj[2][timeTicks2,2]), style="solid,blue,mark=diamond", legendentry="AC 2 Trajectory", markSize=3),
    		Plots.Command(legCommand)
    		],
    		colorbar=colorbarOption,
    		legendStyle="{at={(0.02,0.02)}, anchor=south west}",
    		title=string(CASType) * ", " * L"$D_\mathrm{min} = $" * string(minimum(vec(distance)))[1:5] * " m",
    		style=">=stealth', y tick label style={/pgf/number format/.cd,
          				scaled y ticks = false,
          				set thousands separator={},fixed},
					  x tick label style={/pgf/number format/.cd,
						scaled x ticks = false,
          				set thousands separator={},fixed},
          			  colormap/jet,
          			  colorbar style={at={(1.1,0.0)}, anchor=south west, ylabel={Action (" * L"$^\circ$" * "/s)}}",
    		xlabel="x (m)",
    		ylabel=trajYlabelStr,
    		width="5in", height="5in", axisEqual=true
    	) # Axis

    	distFig = Axis([
    		Plots.Linear(1:distanceDim, vec(distance), legendentry="Distance", mark="none", style="red, very thick"),
    		Plots.Linear(1:distanceDim, ones(distanceDim)*NMAC_RANGE, legendentry="NMAC (150 m)", style="black, very thick", mark="none"),
    		],
    		legendStyle="{at={(0.98,0.98)}, anchor=north east}",
    		xlabel="Time Step (s)",
    		ylabel=distYlabelStr,
    		title="Minimum Distance = " * string(minimum(vec(distance)))[1:6] * " m",
    		width="5in", height="2in"
    	) # Axis


    elseif numAC == 3
    	distanceDim12 = min(size(ownTraj[1], 1), size(ownTraj[2], 1))
		delta12 = ownTraj[1][1:distanceDim12, 1:2] - ownTraj[2][1:distanceDim12, 1:2]
		distanceDim13 = min(size(ownTraj[1], 1), size(ownTraj[3], 1))
		delta13 = ownTraj[1][1:distanceDim13, 1:2] - ownTraj[3][1:distanceDim13, 1:2]
		distanceDim23 = min(size(ownTraj[2], 1), size(ownTraj[3], 1))
		delta23 = ownTraj[2][1:distanceDim23, 1:2] - ownTraj[3][1:distanceDim23, 1:2]
		distance12 = sqrt.(delta12[:,1].^2 + delta12[:,2].^2)
		distance13 = sqrt.(delta13[:,1].^2 + delta13[:,2].^2)
		distance23 = sqrt.(delta23[:,1].^2 + delta23[:,2].^2)
		maxDistanceDim = max(distanceDim12, distanceDim13, distanceDim23)

		minDistance = minimum(vcat(vec(distance12), vec(distance13), vec(distance23)))

		timeTicks1 = [10 * i for i in 1:Int(floor(size(ownTraj[1], 1) / 10))]
		timeTicks2 = [10 * i for i in 1:Int(floor(size(ownTraj[2], 1) / 10))]
		timeTicks3 = [10 * i for i in 1:Int(floor(size(ownTraj[3], 1) / 10))]

	
		# g = GroupPlot(1, 2, groupStyle = "vertical sep = 2cm")

		destHeading1 = atan2(-(startDestCoords[2, 1] - startDestCoords[4, 1]), -(startDestCoords[1, 1] - startDestCoords[3, 1]))
		destHeading2 = atan2(-(startDestCoords[2, 2] - startDestCoords[4, 2]), -(startDestCoords[1, 2] - startDestCoords[3, 2]))
		destHeading3 = atan2(-(startDestCoords[2, 3] - startDestCoords[4, 3]), -(startDestCoords[1, 3] - startDestCoords[3, 3]))

		actionArr = vcat(vec(ownTraj[1][:,3]), vec(ownTraj[2][:,3]), vec(ownTraj[3][:,3]))
		if length(unique(actionArr)) > 1
			if colorBar == true
				colorbarOption = true
			else
				colorbarOption = false
			end
			actionsLeg = "Actions"
			scatterColor = ""
		else
			colorbarOption = false
			actionsLeg = "Action = COC"
			scatterColor = ", orange, mark options={fill=orange}"
		end

			ownTraj[1][1,3]=10
			ownTraj[1][end,3]=-10
			ownTraj[2][1,3]=10
			ownTraj[2][end,3]=-10
			ownTraj[3][1,3]=10
			ownTraj[3][end,3]=-10

			if show_action == true
				tragFig = Axis([
    				Plots.Scatter(vec(ownTraj[1][:,1]), vec(ownTraj[1][:,2]), vec(ownTraj[1][:,3]), legendentry=actionsLeg, style="mark=*", markSize=0.8),
    				Plots.Linear(vec(ownTraj[1][:,1]), vec(ownTraj[1][:,2]), mark="none", style="forget plot, red, very thick"),
    				Plots.Scatter(vec(ownTraj[2][:,1]), vec(ownTraj[2][:,2]), vec(ownTraj[2][:,3]), style="forget plot,mark=*", markSize=0.8),
    				Plots.Linear(vec(ownTraj[2][:,1]), vec(ownTraj[2][:,2]), mark="none", style="forget plot, blue, very thick"),
    				Plots.Scatter(vec(ownTraj[3][:,1]), vec(ownTraj[3][:,2]), vec(ownTraj[3][:,3]), style="forget plot,mark=*", markSize=0.8),
    				Plots.Linear(vec(ownTraj[3][:,1]), vec(ownTraj[3][:,2]), mark="none", style="forget plot, green, very thick"),
	
		
    				Plots.Scatter(startDestCoords[1, 1], startDestCoords[2, 1], legendentry="Start", style="white,mark options={fill=black},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[3, 1], startDestCoords[4, 1], legendentry="Destination", style="white,mark options={fill=magenta},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[1, 2], startDestCoords[2, 2], style="forget plot,white,mark options={fill=black},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[3, 2], startDestCoords[4, 2], style="forget plot,white,mark options={fill=magenta},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[1, 3], startDestCoords[2, 3], style="forget plot,white,mark options={fill=black},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[3, 3], startDestCoords[4, 3], style="forget plot,white,mark options={fill=magenta},mark=square*", markSize=2.3),
		
    				Plots.Scatter(vec(ownTraj[1][timeTicks1,1]), vec(ownTraj[1][timeTicks1,2]), style="solid,red,mark=diamond", legendentry="AC 1 Trajectory", markSize=2),
    				Plots.Scatter(vec(ownTraj[2][timeTicks2,1]), vec(ownTraj[2][timeTicks2,2]), style="solid,blue,mark=diamond", legendentry="AC 2 Trajectory", markSize=2),
    				Plots.Scatter(vec(ownTraj[3][timeTicks3,1]), vec(ownTraj[3][timeTicks3,2]), style="solid,black,mark=diamond", legendentry="AC 3 Trajectory", markSize=2),
	
    				# Plots.Command("\\begin{pgfonlayer}{foreground}\\draw[->,-triangle 60,red] (axis cs:" * string(startDestCoords[1, 1]) * ", " * string(startDestCoords[2, 1]) * 
    				# 	") to (axis cs:" * string(startDestCoords[1, 1] + 400 * cos(destHeading1)) * ", " * string(startDestCoords[2, 1] + 400 * sin(destHeading1)) * ");\\end{pgfonlayer}{foreground}"),
    				# Plots.Command("\\begin{pgfonlayer}{foreground}\\draw[->,-triangle 60,blue] (axis cs:" * string(startDestCoords[1, 2]) * ", " * string(startDestCoords[2, 2]) * 
    				# 	") to (axis cs:" * string(startDestCoords[1, 2] + 400 * cos(destHeading2)) * ", " * string(startDestCoords[2, 2] + 400 * sin(destHeading2)) * ");\\end{pgfonlayer}{foreground}"),
    				# Plots.Command("\\begin{pgfonlayer}{foreground}\\draw[->,-triangle 60,black] (axis cs:" * string(startDestCoords[1, 3]) * ", " * string(startDestCoords[2, 3]) * 
    				# 	") to (axis cs:" * string(startDestCoords[1, 3] + 400 * cos(destHeading3)) * ", " * string(startDestCoords[2, 3] + 400 * sin(destHeading3)) * ");\\end{pgfonlayer}{foreground}"),
    				Plots.Command(legCommand)
    				],
    				colorbar=colorbarOption,
    				legendStyle="{at={(1.6,0.02)}, anchor=south west}",
    				title=string(CASType) * ", " * L"$D_\mathrm{min} = $ " * string(minDistance)[1:5] * " m",
    				xlabel="x (m)",
    				ylabel=trajYlabelStr,
    				style=">=stealth', y tick label style={/pgf/number format/.cd,
          					scaled y ticks = false,
          					set thousands separator={},fixed},
						  x tick label style={/pgf/number format/.cd,
							scaled x ticks = false,
          					set thousands separator={},fixed},
          				  colormap/jet,
          				  colorbar style={at={(1.1,0.0)}, anchor=south west, ylabel={Action (" * L"$^\circ$" * "/s)}}",
    				width=string(figSize) * "in", height=string(figSize) * "in", axisEqual=true
    			) # Axis
			else
				tragFig = Axis([
    				Plots.Linear(vec(ownTraj[1][:,1]), vec(ownTraj[1][:,2]), mark="none", style="forget plot, red, very thick"),
    				Plots.Linear(vec(ownTraj[2][:,1]), vec(ownTraj[2][:,2]), mark="none", style="forget plot, blue, very thick"),
    				Plots.Linear(vec(ownTraj[3][:,1]), vec(ownTraj[3][:,2]), mark="none", style="forget plot, green, very thick"),
	
		
    				Plots.Scatter(startDestCoords[1, 1], startDestCoords[2, 1], legendentry="Start", style="white,mark options={fill=black},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[3, 1], startDestCoords[4, 1], legendentry="Destination", style="white,mark options={fill=magenta},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[1, 2], startDestCoords[2, 2], style="forget plot,white,mark options={fill=black},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[3, 2], startDestCoords[4, 2], style="forget plot,white,mark options={fill=magenta},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[1, 3], startDestCoords[2, 3], style="forget plot,white,mark options={fill=black},mark=square*", markSize=2.3),
    				Plots.Scatter(startDestCoords[3, 3], startDestCoords[4, 3], style="forget plot,white,mark options={fill=magenta},mark=square*", markSize=2.3),
		
    				Plots.Scatter(vec(ownTraj[1][timeTicks1,1]), vec(ownTraj[1][timeTicks1,2]), style="solid,red,mark=diamond", legendentry="Aircraft 1 Trajectory", markSize=2),
    				Plots.Scatter(vec(ownTraj[2][timeTicks2,1]), vec(ownTraj[2][timeTicks2,2]), style="solid,blue,mark=diamond", legendentry="Aircraft 2 Trajectory", markSize=2),
    				Plots.Scatter(vec(ownTraj[3][timeTicks3,1]), vec(ownTraj[3][timeTicks3,2]), style="solid,green,mark=diamond", legendentry="Aircraft 3 Trajectory", markSize=2),
	
    				Plots.Command("\\draw[->,-triangle 60,red] (axis cs:" * string(startDestCoords[1, 1]) * ", " * string(startDestCoords[2, 1]) * 
    					") to (axis cs:" * string(startDestCoords[1, 1] + 400 * cos(destHeading1)) * ", " * string(startDestCoords[2, 1] + 400 * sin(destHeading1)) * ");"),
    				Plots.Command("\\draw[->,-triangle 60,blue] (axis cs:" * string(startDestCoords[1, 2]) * ", " * string(startDestCoords[2, 2]) * 
    					") to (axis cs:" * string(startDestCoords[1, 2] + 400 * cos(destHeading2)) * ", " * string(startDestCoords[2, 2] + 400 * sin(destHeading2)) * ");"),
    				Plots.Command("\\draw[->,-triangle 60,green] (axis cs:" * string(startDestCoords[1, 3]) * ", " * string(startDestCoords[2, 3]) * 
    					") to (axis cs:" * string(startDestCoords[1, 3] + 400 * cos(destHeading3)) * ", " * string(startDestCoords[2, 3] + 400 * sin(destHeading3)) * ");"),
    				Plots.Command(legCommand)
    				],
    				colorbar=colorbarOption,
    				legendStyle="{at={(1.3,0.0)}, anchor=south west}",
    				title=string(CASType) * ", " * L"$D_\mathrm{min} = $ " * string(minDistance)[1:5] * " m",
    				xlabel="x (m)",
    				ylabel=trajYlabelStr,
    				style=">=stealth', y tick label style={/pgf/number format/.cd,
          					scaled y ticks = false,
          					set thousands separator={},fixed},
						  x tick label style={/pgf/number format/.cd,
							scaled x ticks = false,
          					set thousands separator={},fixed},
          				  colormap/jet,
          				  colorbar style={at={(1.1,0.0)}, anchor=south west, ylabel={Action (" * L"$^\circ$" * "/s)}}",
    				width=string(figSize) * "in", height=string(figSize) * "in", axisEqual=true
    			) # Axis
			end
    	distFig = Axis([
    		Plots.Linear(1:distanceDim12, vec(distance12), legendentry="Distance12", mark="none", style="red, very thick"),
    		Plots.Linear(1:distanceDim13, vec(distance13), legendentry="Distance13", mark="none", style="blue, very thick"),
    		Plots.Linear(1:distanceDim23, vec(distance23), legendentry="Distance23", mark="none", style="green, very thick"),
    		Plots.Linear(1:maxDistanceDim, ones(maxDistanceDim)*NMAC_RANGE, legendentry="NMAC (150 m)", style="black, dashed, thick", mark="none"),
    		Plots.Command(legCommand)
    		],
    		legendStyle="{at={(1.15,0.02)}, anchor=south west}",
    		xlabel="Time Step (s)",
    		ylabel=distYlabelStr,
    		title="Minimum Distance = " * string(minDistance)[1:6] * " m",
    		width=string(figSize) * "in", height="1.5in"
    	) # Axis
    end
    # legendStyle="{at={(0.02,0.02)}, anchor=south west}",
    return tragFig, distFig
end


