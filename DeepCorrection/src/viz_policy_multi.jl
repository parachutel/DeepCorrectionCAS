push!(LOAD_PATH, "./")
push!(LOAD_PATH, "../")
push!(LOAD_PATH, "../logs/")
push!(LOAD_PATH, "../../VICAS/src/")

using PyPlot
using DeepQLearning
using Simulation_Env_module
using CAS_QMDP
using Interact

rc("font", family="Times New Roman")
rc("text", usetex=true)

function get_section_id(x::Float64, y::Float64, num_sections::Int64)
	ac_pos_angle = norm_angle(atan2(y, x))
	return floor(ac_pos_angle / (2 * pi / num_sections)) + 1
end

function viz_policy_multi(policy::DQNPolicy; 
	CASType::AbstractString="",
	state_formation::Symbol=:closest, 
	correctionOffOption::Symbol=:multi,
	num_intruders::Int64=1, 
	resolution::Int64=100, 
	correction::Bool=true, 
	colorbarOption::Bool=false,
	xlabelOption::Bool=true,
	ylabelOption::Bool=false,
	legendOption::Bool=false,
	labelleft::Bool=true,
	labelbottom::Bool=true,
	fontsize::Int64=12,
	titleOption::Bool=true,
    format::AbstractString="pdf",
    figsize::Float64=3.5)

	rc("font", size=fontsize)

	XMIN = -1200
	XMAX = 1200
	YMIN = XMIN
	YMAX = XMAX
	v = 50.0
	viz = figure()
	

	save_button = button("Save Policy Slice", value=0)
	display(save_button)
	fig_name_input = textbox("Figure Name", value=CASType*"PolicySlice")
	display(fig_name_input)
	print_state_button = button("Print full_state", value=0)
	display(print_state_button)
	int_pos_input = textbox("Intruder Position", value="0 0")
	display(int_pos_input)
	
	@manipulate for int_1_x = XMIN : 150 : XMAX,
					int_1_y = YMIN : 150 : YMAX,
					int_1_p = 0. : 30. : 360,
					int_2_x = XMIN : 150 : XMAX,
					int_2_y = YMIN : 150 : YMAX,
					int_2_p = 0. : 30. : 360,
					p = 0. : 30. : 360,
					deviation = -180. : 30. : 180.,
					dist_to_dest = 100. : 500. : DEST_RANGE
					# ...

		num_fix_intruders = 0
		lowfi_state_matrix = ones(STATE_DIM, 5) * TERM_VAR # 5 is a defautl max num of intruders

		withfig(viz) do
			int_1_rho = sqrt(int_1_x^2 + int_1_y^2)
			if int_1_rho <= SENSING_RANGE
				lowfi_state_matrix[:, 1] .= [int_1_rho, norm_angle(atan2(int_1_y, int_1_x)), deg2rad(int_1_p), v, v]
				num_fix_intruders += 1
			end
			int_2_rho = sqrt(int_2_x^2 + int_2_y^2)
			if int_2_rho <= SENSING_RANGE
				lowfi_state_matrix[:, 2] .= [int_2_rho, norm_angle(atan2(int_2_y, int_2_x)), deg2rad(int_2_p), v, v]
				num_fix_intruders += 1
			end
			# ...

			function get_heat(x::Float64, y::Float64) # of the moving intruder
    			rho = sqrt(x^2 + y^2)
    			theta = norm_angle(atan2(y, x))
    			if rho <= SENSING_RANGE
    				num_intruders = num_fix_intruders + 1
    				lowfi_state_matrix[:, num_intruders] .= [rho, theta, deg2rad(p), v, v]
    			else # if rho >= SENSING_RANGE
    				num_intruders = num_fix_intruders
    			end

    			if num_intruders > 0
					closest_intruder_ind = indmin(lowfi_state_matrix[IND_rho, :])
    				low_fi_state = vec(lowfi_state_matrix[:, closest_intruder_ind])
	
    				if correction == true
    					full_state = ones(policy.env.num_sections * STATE_DIM + AUG_STATE_DIM) * TERM_VAR
    					
    					
    					# full_state formualtion method: 4 closest intruders
    					# Sort lowfi_state_matrix by rho, resulting in acsending order
    					if state_formation == :closest
    						temp_lowfi = copy(lowfi_state_matrix)
    						# lowfi_state_matrix = transpose(sortrows(transpose(lowfi_state_matrix), by=x->x[IND_rho]))
    						for i in 1 : policy.env.num_sections
    							# if !is_terminal(vec(lowfi_state_matrix[:, i]))
    							closestInd = indmin(temp_lowfi[IND_rho, :])
    							if temp_lowfi[IND_rho, closestInd] <= SENSING_RANGE
    								full_state[STATE_DIM * (i - 1) + 1 : STATE_DIM * i] = vec(temp_lowfi[:, closestInd])
    								temp_lowfi[IND_rho, closestInd] = Inf
    							end
    						end

    					else # full_state formulation method: closest intruders in each section
    						section_ac_arr = Dict{Int64, Array{Float64,2}}()
    						for isection in 1 : policy.env.num_sections
    							section_ac_arr[isection] = Array{Float64,2}(STATE_DIM, 0)
    						end	
			
    						for i in 1 : num_intruders
    							# if !is_terminal(vec(lowfi_state_matrix[:, i]))
    							if vec(lowfi_state_matrix[:, i])[IND_rho] <= SENSING_RANGE
    								ix = lowfi_state_matrix[IND_rho, i] * cos(lowfi_state_matrix[IND_theta, i]) 
    								iy = lowfi_state_matrix[IND_rho, i] * sin(lowfi_state_matrix[IND_theta, i])
    								section_id_i = get_section_id(ix, iy, policy.env.num_sections)
    								section_ac_arr[section_id_i] = hcat(section_ac_arr[section_id_i], lowfi_state_matrix[:, i])
    							end
    						end
			
    						for isection in 1 : policy.env.num_sections
    							if length(section_ac_arr[isection]) > 0
    								closest_intruder_ind = indmin(section_ac_arr[isection][IND_rho, :])
    								full_state[STATE_DIM * (isection - 1) + 1 : STATE_DIM * isection] .= 
										vec(section_ac_arr[isection][:, closest_intruder_ind])
    							end
    						end
						end # if state_formation

						# Augment states
    					prev_dist_to_dest = dist_to_dest + v
    					full_state[end - AUG_STATE_DIM + 1 : end] .= [deg2rad(deviation), dist_to_dest, prev_dist_to_dest]
		
    					normalize_obs!(policy.env, full_state)
    					action = get_action(policy, full_state, low_fi_state) # returns symbol
		
     				else # if correction == false
     					full_state = ones(policy.env.num_sections * STATE_DIM + AUG_STATE_DIM) * TERM_VAR # dummy variable
     					if correctionOffOption == :multi
     						q_matrix = Matrix{Float64}(length(policy.env.action_space), 0)
     						for i in 1 : num_intruders
     							if vec(lowfi_state_matrix[:, i])[IND_rho] <= SENSING_RANGE
     								state_i = vec(lowfi_state_matrix[:, i])
     								qvals = get_qvals(policy.env.low_fi_policy, state_i)
     								q_matrix = hcat(q_matrix, qvals)
     							end
     						end
     						if size(q_matrix, 2) == 0
     							q_matrix = hcat(q_matrix, terminal_state)
     						end
     						min_qvals = minimum(q_matrix, 2)
							the_ind = indmax(min_qvals)
		
     						action = Simulation_Env_module.actions(policy.env)[the_ind]

     					elseif correctionOffOption == :closest
							qvals = get_qvals(policy.env.low_fi_policy, low_fi_state)
							action = Simulation_Env_module.actions(policy.env)[indmax(qvals)]
     					end
     				end # if

	
     				if action == :COC
    					return 2.0, full_state, lowfi_state_matrix
    				end
     				return rad2deg(policy.env.advisory_to_action_dict[action]), full_state, lowfi_state_matrix # get numerical value of action

     			else # if num_intruders == 0, i.e. no intruder
     				return 2.0, full_state, lowfi_state_matrix
     			end
			end # get_heat


			# plot 
			action_map = zeros(resolution, resolution)
			x_arr = linspace(XMIN, XMAX, resolution)
			y_arr = linspace(YMIN, YMAX, resolution)

			for j in 1:resolution
				for i in 1:resolution
					action_map[j, i], _, _ = get_heat(x_arr[i], y_arr[j])
				end
			end

			action_map[Int(resolution / 2), Int(resolution / 2)] = 10.0
			action_map[Int(resolution / 2), Int(resolution / 2)+1] = -10.0

			action_map = flipdim(action_map, 1)

			g = imshow(action_map, cmap="jet", extent=(XMIN, XMAX, YMIN, YMAX))

			if colorbarOption == true
				cbar = colorbar(g, fraction=0.088, pad=0.04, ticks=[-10, -5, 0, 2, 5, 10], aspect=10, label="Advisory of Ownship")
				cbar[:set_ticklabels](["SR", "WR", "KEEP", "COC", "WL", "SL"])
			end

			# plot Ownship triangle
			scatter(0, 0 , marker=">", color="green", s=180, label="Ownship")

			# plot intruder arrow
			airspace_dim = XMAX - XMIN
            arrow_len = airspace_dim / 12
            arrow(0.8 * XMAX, 0.8 * YMAX,
                arrow_len * cos(deg2rad(p)), # dx
                arrow_len * sin(deg2rad(p)), # dy
                head_width = airspace_dim / 30,
                width = airspace_dim / 90,
                head_length = airspace_dim / 25,
                head_starts_at_zero = "true",
                facecolor = "red",
                length_includes_head = "true")
            
            scatter(0.8 * XMAX, 0.8 * XMAX , color="red", s=40, label="Free Intruder")

			# plot existing intruder 1 arrow
			airspace_dim = XMAX - XMIN
			arrow_len = airspace_dim / 12

			# plot fixed intruder 1
			arrow(int_1_x, int_1_y,
				arrow_len * cos(deg2rad(int_1_p)), # dx
				arrow_len * sin(deg2rad(int_1_p)), # dy
				head_width = airspace_dim / 20,
				width = airspace_dim / 70,
				head_length = airspace_dim / 18,
				head_starts_at_zero = "true",
				facecolor = "blue",
				length_includes_head = "true")
			scatter(int_1_x, int_1_y , color="blue", s=60, label="Fixed Intruder 1")

			# plot fixed intruder 2
			arrow(int_2_x, int_2_y,
				arrow_len * cos(deg2rad(int_2_p)), # dx
				arrow_len * sin(deg2rad(int_2_p)), # dy
				head_width = airspace_dim / 20,
				width = airspace_dim / 70,
				head_length = airspace_dim / 18,
				head_starts_at_zero = "true",
				facecolor = "magenta",
				length_includes_head = "true")
			scatter(int_2_x, int_2_y , color="magenta", s=60, label="Fixed Intruder 2")
			# ...

			# plot NMAC circle
			th = linspace(0, 2*pi, 50)
			PyPlot.plot(NMAC_RANGE * cos.(th), NMAC_RANGE * sin.(th), linestyle="--", color="red", label="NMAC Range = 150 m")

			# plot sensing circle
			PyPlot.plot(SENSING_RANGE * cos.(th), SENSING_RANGE * sin.(th), linestyle="--", color="blue", label="Sensing Range = 1000 m")

			if ylabelOption == true
				ylabel("\$y\$ (m)")
			end
			if xlabelOption == true
				xlabel("\$x\$ (m)")
			end

			if titleOption == true
				title(CASType)
			end
    		
    		
    		ax = gca()
    		ax[:set_xlim]([-1200,1200])
    		ax[:set_ylim]([-1200,1200])
    		ax[:tick_params](direction="in", color="gray", labelleft=labelleft, labelbottom=labelbottom)
    		fig = gcf()
			fig[:set_size_inches](figsize, figsize)

			if legendOption == true
				legend(loc="center left", bbox_to_anchor=(1.05, 0.2), fancybox=false, edgecolor="black", shadow=false)
			end

			o_saveButton = observe(save_button)
    		o_figNameTextbox = observe(fig_name_input)
			function save_slice(signal::Int64, filename=o_figNameTextbox[])
				if signal > 0
					savefig(filename * "." * format, bbox_inches="tight", pad_inches=0.01)
				end
			end
			on(save_slice, o_saveButton)

    		o_button = observe(print_state_button)
    		o_textbox = observe(int_pos_input)
			function print_state(signal::Int64, pos_str=o_textbox[])
				pos = [parse(Float64, ps) for ps in split(pos_str)]
				if signal > 0
					_, fs, lowfi_state_matrix = get_heat(pos[1], pos[2])
					println("fs = ", fs)
					println("lowfi_state_matrix = ", lowfi_state_matrix)
					println()
				end
			end
			on(print_state, o_button)
    		
    		return viz

		end # withfig
	end
end

