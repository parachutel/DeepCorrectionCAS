push!(LOAD_PATH, "../src")
push!(LOAD_PATH, "../Distances")

using Distances, Aircraft_module, Constants_module

function check_NMAC(All_AC::Array{Aircraft,1})
	All_Coords = Matrix{Float64}(2, 0)

	R = Matrix{Float64}(length(All_AC), length(All_AC))
	for ac in All_AC
		All_Coords = hcat(All_Coords, [ac.dyn.x; ac.dyn.y])
	end

	pairwise!(R, Euclidean(), All_Coords) #inline

	# iterate through the upper right side of R (not including diag)
	# so that not double count
	TOTAL_num_colli = 0
	for col = 2 : length(All_AC)
		for row = 1 : col - 1
			if R[row, col] <= collision_range
				TOTAL_num_colli = TOTAL_num_colli + 1
			end
		end
	end
	
	return TOTAL_num_colli
end

function check_desti(ac::Aircraft)
	dist = sqrt((ac.dyn.x - ac.x_dest) ^ 2 + (ac.dyn.y - ac.y_dest) ^ 2)
	if dist <= dest_range
		return true
	else
		return false
	end
end

function check_in_airspace(ac::Aircraft)
	if ac.dyn.x >= 0 && ac.dyn.x <= airspace_dim  && 
		ac.dyn.y >= 0 && ac.dyn.y <= airspace_dim
		return true
	else
		return false
	end
end

function check_colli(new_ac::Aircraft, All_AC::Array{Aircraft,1})
	for ac in All_AC
		dist = sqrt((ac.dyn.x - new_ac.dyn.x) ^ 2 + (ac.dyn.y - new_ac.dyn.y) ^ 2)
		if dist <= collision_range
			break
			return true 
		end
	end
end