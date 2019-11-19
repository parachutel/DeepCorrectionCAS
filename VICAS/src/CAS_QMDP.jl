module CAS_QMDP

push!(LOAD_PATH, "./")

using DoubleUAVs_module, GridInterpolations, JLD, PGFPlots, Interact, PyPlot

export qmdp, write_alphas, write_alphas_to_csv, load_alphas, get_qvals 
export viz_policy, viz_policy_multi
export norm_angle, advisories, advisory_to_action_dict, disc
export sigma_str, sigma_v, sigma_alert_deg, sigma_coc_deg
export actual_weight, discount, terminal_state_var
export is_terminal, terminal_state

const max_iteration = 3000
const alphaTol = 1e-7

"""
	save alpha vectors to jld file
"""
function write_alphas(alphas::Matrix{Float64}, filename::AbstractString)
	path = "../policies/"
	if !ispath(path)
		mkdir(path)
	end
	filename = path * filename * ".jld"
	println("Writing alpha vectors to: " * filename)
	jldopen(filename, "w") do file
    	write(file, "alphas", alphas)
	end
end


"""
	save alphas vectors to csv file
"""
function write_alphas_to_csv(alphas::Matrix{Float64}, filename::AbstractString)
	path = "../policies/"
	if !ispath(path)
		mkdir(path)
	end
	filename = path * filename * ".csv"
	println("Writing alpha vectors to: " * filename)
	open(filename, "w") do io
        for row in 1 : size(alphas, 1)
        	for col in 1 : size(alphas, 2)
        		if col != size(alphas, 2)
            		@printf(io, "%f, ", alphas[row, col])
            	else
            		@printf(io, "%f\n", alphas[row, col])
            	end
            end
        end
    end
end

"""
	read alpha vectors from file
"""
function load_alphas(filename::AbstractString)
	alphas = jldopen(filename, "r") do file
        read(file, "alphas")
    end
	return alphas
end


function update_max_alphas!(max_alphas::Vector{Float64}, alphas::Matrix{Float64})
	for istate in 1:num_states
		max_alphas[istate] = alphas[istate, 1]
		for iaction in 2:num_actions
			if max_alphas[istate] < alphas[istate, iaction]
				max_alphas[istate] = alphas[istate, iaction]
			end
		end
	end
	# max_alphas = maximum(alphas, 2)
end

function qmdp(alphas_filename::AbstractString;
	discount::Float64=0.95,
	if_sample::Bool=false,
	verbose::Bool=true, 
	penalty_action::Float64=0.02, 
	penalty_conflict::Float64=1.0, 
	penalty_closeness::Float64=10.0, 
	penalty_nmac::Float64=1000.0)

	println("Scheme: ", alphas_filename)
	println("Running value iteration alpha vectors approximation...")
	alphas = zeros(num_states, num_actions)
	cputime = 0
	max_alphas = zeros(num_states)

	for iter in 1:max_iteration
		residual = 0
		tic()
		for istate in 1:num_states
			# println("s = ", istate)
			state = terminal_state
			if istate != num_states
				state = ind2x(grid_states, istate)
			end
			
			for iaction in 1:num_actions
				# print("a = ", iaction)
				advisory = advisories[iaction]
				vnext = 0

				if if_sample == false
					# state transition through sigma-sampling
					for isigma in 1 : num_sigmas
						next_state = get_next_state(state, advisory, isigma=isigma, if_sample=if_sample)
						if is_terminal(next_state)
							vnext += max_alphas[end] * weights[isigma]
						else
							vnext += interpolate(grid_states, max_alphas, next_state) * weights[isigma]
						end
					end # for isigma
				else # if_sample == true
					# random sampled noise method
					next_state = get_next_state(state, advisory, if_sample=if_sample)
					if is_terminal(next_state)
						vnext += max_alphas[end]
					else
						vnext += interpolate(grid_states, max_alphas, next_state)
					end
				end 

				prev_alpha_sa = alphas[istate, iaction]
				alphas[istate, iaction] = 
					get_reward(state, advisory, 
						penalty_action=penalty_action, 
						penalty_conflict=penalty_conflict,
						penalty_closeness=penalty_closeness,
						penalty_nmac=penalty_nmac) + discount * vnext
				residual += (alphas[istate, iaction] - prev_alpha_sa)^2
			end # for iaction
		end # for istate

		update_max_alphas!(max_alphas, alphas)

		iterTime = toq()
		cputime = cputime + iterTime
        if verbose
            @printf("Iteration %d: residual = %.2e, Qmin = %.2f, cputime = %.2e\n", iter, residual, minimum(alphas), iterTime)
        end

        if iter % 10 == 0
        	write_alphas(alphas, alphas_filename)
        end

        if residual < alphaTol
        	break
        end

        if iter == max_iteration
        	println("Warning: maximum number of iterations reached;", "solution may be inaccurate")
        end
	end # for iter

	@printf("Value iteration done!\ncputime = %.2e sec\n\n", cputime)
	# modify the values of alphas[end, :] (the terminal state values)
	# to make it numerically small.
	# alphas[] .= minimum(alphas) ?
	return alphas
end


"""
	get belief from state
"""
function get_belief(state::Vector{Float64})
	belief = spzeros(num_states, 1)
	if is_terminal(state)
		belief[end] = 1.0
	else
		indices, interp_weights = interpolants(grid_states, state) # ? state[1:3]
        for i = 1:length(indices)
            belief[indices[i]] = interp_weights[i]
        end
	end
	return belief
end

"""
	get Q values given a state (observation if uncertainty in state presents)
"""
function get_qvals(alphas::Matrix{Float64}, state::Vector{Float64})
	belief = get_belief(state)

	qvals = zeros(num_actions)
	for iaction in 1:num_actions
		for ibelief in 1:length(belief.rowval) # dot product
			qvals[iaction] += belief.nzval[ibelief] * alphas[belief.rowval[ibelief], iaction]
		end
	end
	return qvals
end


# Auxiliary #################################################################################
"""
	Visualize policy using PGFPlots
	@ outdated @
"""
function viz_policy(alphas::Matrix{Float64}, phi::Float64, v_own::Float64, v_int::Float64; 
						resolution::Int64=250, mode="interact")
	# phi in [deg]
	XMIN = -2000
	XMAX = 2000
	YMIN = XMIN
	YMAX = XMAX
	function get_heat(x::Float64, y::Float64)
    	rho = sqrt(x^2 + y^2)
    	theta = norm_angle(atan2(y, x))
    	state = [rho, theta, deg2rad(phi), v_own, v_int]
    	qvals = get_qvals(alphas, state)
    	action = advisory_to_action_dict[advisories[indmax(qvals)]] # [deg]
    	if action == advisory_to_action_dict[:COC]
    	    return 2.0
    	end
     	return action
	end

	if mode == "pgf"
		th = linspace(0, 2*pi, 50)
		viz = Axis(
    		[
    	    	Plots.Image(get_heat, (Int(XMIN), Int(XMAX)), (Int(YMIN), Int(YMAX)), 
    	    	    xbins = resolution, ybins = resolution,
    	    	    colormap = ColorMaps.Named("RdBu"), colorbar = true), 
    	    	Plots.Node(L">", 0, 0, style="rotate=0,font=\\Huge,color=magenta"),
    	    	Plots.Node(L">", 1800, 1800, style=string("rotate=", phi, ",font=\\Huge,color=red")),
    	    	Plots.Linear(sensing_range * cos.(th), sensing_range * sin.(th), style="dashed,color=red", mark="none")
    		],
    		title="Ownship Action with Intruder Heading = " * string(phi) * " deg",
    		xlabel="x (m)", ylabel="y (m)",
    		width="10cm", height="10cm")
		return viz

	elseif mode == "csv"
		open("../viz/viz_data.csv", "w") do io
			for y in linspace(YMIN, YMAX, resolution)
				for x in linspace(XMIN, XMAX, resolution)
            		@printf(io, "%s", get_heat(x, y))
            		if x < XMAX
            			@printf(io, ", ")
            		else
            			@printf(io, "\n")
            		end
    			end
			end
		end

	elseif mode == "interact"
		th = linspace(0, 2*pi, 50)
		@manipulate for p = 0. : 30. : 360,
						v0 = v_min : 10. : v_max,
						v1 = v_min : 10. : v_max
			function get_heat_interact(x::Float64, y::Float64)
    			rho = sqrt(x^2 + y^2)
    			theta = norm_angle(atan2(y, x))
    			state = [rho, theta, deg2rad(p), v0, v1]
    			qvals = get_qvals(alphas, state)
    			action = advisory_to_action_dict[advisories[indmax(qvals)]] # [deg]
    			if action == advisory_to_action_dict[:COC]
    			    return 2.0
    			end
     			return action
			end

			viz = Axis(
    			[
    	    		Plots.Image(get_heat_interact, (Int(XMIN), Int(XMAX)), (Int(YMIN), Int(YMAX)), 
    	    		    xbins = resolution, ybins = resolution,
    	    		    colormap = ColorMaps.Named("RdBu"), colorbar = true), 
    	    		Plots.Node(L">", 0, 0, style="rotate=0,font=\\Huge,color=magenta"),
    	    		Plots.Node(L">", 1800, 1800, style=string("rotate=", p, ",font=\\Huge,color=red")),
    	    		Plots.Linear(sensing_range * cos.(th), sensing_range * sin.(th), style="dashed,color=red", mark="none"),
    	    		Plots.Linear(collision_range * cos.(th), collision_range * sin.(th), style="color=red", mark="none")
    			],
    			title="Ownship Action with Intruder Heading = " * string(p) * " deg",
    			xlabel="x (m)", ylabel="y (m)",
    			width="10cm", height="10cm")
			return viz
		end # @manipulate for 
	end # if mode
end

"""
	Visualize pairwise policy slice using PyPlot
"""
function viz_policy(alphas::Matrix{Float64}; resolution::Int64=250)
	XMIN = -2000
	XMAX = 2000
	YMIN = XMIN
	YMAX = XMAX
	viz = figure()

	save_button = button("Save Policy Slice", value=0)
	display(save_button)
	fig_name_input = textbox("Figure Name", value="policy_slice_filename")
	display(fig_name_input)

	@manipulate for p = 0. : 30. : 360,
					v0 = v_min : 10. : v_max,
					v1 = v_min : 10. : v_max
		withfig(viz) do

			function get_heat(x::Float64, y::Float64)
    			rho = sqrt(x^2 + y^2)
    			theta = norm_angle(atan2(y, x))
    			state = [rho, theta, deg2rad(p), v0, v1]
    			qvals = get_qvals(alphas, state)
    			action = advisory_to_action_dict[advisories[indmax(qvals)]] # [deg]
    			if action == advisory_to_action_dict[:COC]
    			    return 2.0
    			end
     			return action
			end

			action_map = zeros(resolution, resolution)
			x_arr = linspace(XMIN, XMAX, resolution)
			y_arr = linspace(YMIN, YMAX, resolution)

			for j in 1:resolution
				for i in 1:resolution
					action_map[j, i] = get_heat(x_arr[i], y_arr[j])
				end
			end

			action_map[1, 1] = 10.0
			action_map[1, 2] = -10.0

			action_map = flipdim(action_map, 1)

			imshow(action_map, cmap="RdBu", extent=(XMIN, XMAX, YMIN, YMAX))
			cbar = colorbar(label="Advisory", ticks=[-10, -5, 0, 2, 5, 10])
			cbar[:set_ticklabels](["-10", "-5", "0", "COC", "5", "10"])

			# plot Ownship triangle
			scatter(0, 0 , marker=">", color="green", s=180)

			# plot intruder arrow
			airspace_dim = XMAX - XMIN
			arrow_len = airspace_dim / 8
			arrow(0.7 * XMAX, 0.7 * YMAX,
				arrow_len * cos(deg2rad(p)), # dx
				arrow_len * sin(deg2rad(p)), # dy
				head_width = airspace_dim / 16,
				width = airspace_dim / 50,
				head_length = airspace_dim / 12,
				head_starts_at_zero = "true",
				facecolor = "red",
				length_includes_head = "true")

			# plot NMAC circle
			th = linspace(0, 2*pi, 50)
			PyPlot.plot(collision_range * cos.(th), collision_range * sin.(th), linestyle="--", color="red")

			# plot sensing circle
			PyPlot.plot(sensing_range * cos.(th), sensing_range * sin.(th), linestyle="--", color="red")

			title("Ownship Action with Intruder Heading = " * string(p) * " deg")
    		xlabel("x (m)") 
    		ylabel("y (m)")

    		o_button = observe(save_button)
    		o_textbox = observe(fig_name_input)
			function save_slice(signal::Int64, filename=o_textbox[])
				if signal > 0
					savefig(filename * ".pdf")
				end
			end
			on(save_slice, o_button)
    		
    		return viz

		end # withfig
	end # @manipulate for
end

"""
	Visualize multi-intruder policy slice using PyPlot
"""
function viz_policy_multi(alphas::Matrix{Float64}; num_intruders::Int64=1, resolution::Int64=250, method::Symbol=:closest)
	XMIN = -2000
	XMAX = 2000
	YMIN = XMIN
	YMAX = XMAX
	v = 50.0
	viz = figure()
	save_button = button("Save Policy Slice", value=0)
	display(save_button)
	fig_name_input = textbox("Figure Name", value="policy_slice_filename")
	display(fig_name_input)

	state_matrix = ones(state_dim, 5) * Inf # at most 5 intruders
	
	@manipulate for int_1_x = XMIN : 250 : XMAX,
					int_1_y = YMIN : 250 : YMAX,
					int_1_p = 0. : 30. : 360,
					p       = 0. : 30. : 360
					# more intruders to be added ...

		withfig(viz) do

			state_matrix[:, 1] .= [sqrt(int_1_x^2 + int_1_y^2), norm_angle(atan2(int_1_y, int_1_x)), deg2rad(int_1_p), v, v]
			# more intruders to be added ...

			function get_heat(x::Float64, y::Float64)
    			rho = sqrt(x^2 + y^2)
    			theta = norm_angle(atan2(y, x))
    			state_matrix[:, num_intruders + 1] .= [rho, theta, deg2rad(p), v, v]
    			if method == :closest
    				closest_intruder_ind = indmin(state_matrix[rho_ind, :])
    				state = Vector(state_matrix[:, closest_intruder_ind])
     				qvals = get_qvals(alphas, state)
     				action = advisories[indmax(qvals)] # symbol
     		    else # if method != :closest
     				q_matrix = Matrix{Float64}(num_actions, 0)
     				sum_qvals = zeros(num_actions)
     				for i in 1 : num_intruders + 1
     					state_i = Vector(state_matrix[:, i])
     					qvals = get_qvals(alphas, state_i)
     					q_matrix = hcat(q_matrix, qvals)
     					sum_qvals += qvals
     				end
     				if method == :maxmin
     					min_qvals = minimum(q_matrix, 2)
						the_ind = indmax(min_qvals)
					elseif method == :maxsum
						the_ind = indmax(sum_qvals)
					else
						error("No such method!")
					end # if method
     				action = advisories[the_ind]
     			end # if method

     			if action == :COC
    				return 2.0
    			end
     			return advisory_to_action_dict[action] # get numerical value of action
			end # get_heat


			# plot 
			action_map = zeros(resolution, resolution)
			x_arr = linspace(XMIN, XMAX, resolution)
			y_arr = linspace(YMIN, YMAX, resolution)

			for j in 1:resolution
				for i in 1:resolution
					action_map[j, i] = get_heat(x_arr[i], y_arr[j])
				end
			end

			action_map[1, 1] = 10.0
			action_map[1, 2] = -10.0

			action_map = flipdim(action_map, 1)

			imshow(action_map, cmap="RdBu", extent=(XMIN, XMAX, YMIN, YMAX))
			cbar = colorbar(label="Advisory", ticks=[-10, -5, 0, 2, 5, 10])
			cbar[:set_ticklabels](["-10", "-5", "0", "COC", "5", "10"])

			# plot Ownship triangle
			scatter(0, 0 , marker=">", color="green", s=180)

			# plot intruder arrow
			airspace_dim = XMAX - XMIN
			arrow_len = airspace_dim / 10
			arrow(0.7 * XMAX, 0.7 * YMAX,
				arrow_len * cos(deg2rad(p)), # dx
				arrow_len * sin(deg2rad(p)), # dy
				head_width = airspace_dim / 16,
				width = airspace_dim / 50,
				head_length = airspace_dim / 14,
				head_starts_at_zero = "true",
				facecolor = "red",
				length_includes_head = "true")

			# plot existing intruder 1 arrow
			airspace_dim = XMAX - XMIN
			arrow_len = airspace_dim / 12
			arrow(int_1_x, int_1_y,
				arrow_len * cos(deg2rad(int_1_p)), # dx
				arrow_len * sin(deg2rad(int_1_p)), # dy
				head_width = airspace_dim / 20,
				width = airspace_dim / 70,
				head_length = airspace_dim / 18,
				head_starts_at_zero = "true",
				facecolor = "blue",
				length_includes_head = "true")
			# more intruders to be added ...

			# plot NMAC circle
			th = linspace(0, 2*pi, 50)
			PyPlot.plot(collision_range * cos.(th), collision_range * sin.(th), linestyle="--", color="red")

			# plot sensing circle
			PyPlot.plot(sensing_range * cos.(th), sensing_range * sin.(th), linestyle="--", color="red")

			title("Ownship Action with Intruder Heading = " * string(p) * " deg, " * string(method))
    		xlabel("x (m)") 
    		ylabel("y (m)")

    		o_button = observe(save_button)
    		o_textbox = observe(fig_name_input)
			function save_slice(signal::Int64, filename=o_textbox[])
				if signal > 0
					savefig(filename * ".pdf")
				end
			end
			on(save_slice, o_button)
    		
    		return viz

		end # withfig
	end
end



end # module




