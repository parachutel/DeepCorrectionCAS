using PyPlot

abstract type Render_Option end

mutable struct Render_Airspace <: Render_Option
	pause_time::Float64
	if_annotate::Bool
	function Render_Airspace(; pause_time::Float64=0.1, annotate::Bool=false)
		show()
		this = new()
		this.pause_time = pause_time
		this.if_annotate = annotate
		return this
	end
end

struct Not_Render <: Render_Option
end

function render_airspace(::Any, render_option::Not_Render, ::Any)
	nothing
end

function render_airspace(t::Int64,
				render_option::Render_Airspace, 
				All_AC::Array{Aircraft, 1})

	# fig = figure("Airspace", figsize = (7, 5))

	for ac in All_AC
		if  ac.desti == false  # && ac.in_airspace == true
				
			if ac.advisory == :COC
				scatter(ac.dyn.x, ac.dyn.y, 
					marker = "o", color = "blue", s = 8)
			else
				scatter(ac.dyn.x, ac.dyn.y, 
					marker = "o", color = "red", s = 8)
			end

			th = linspace(-pi, pi, 30)
			# print(ac.sensor.sensing_range)
			if ac.advisory == :COC
				plot(ac.dyn.x + ac.sensor.sensing_range * cos.(th), 
					ac.dyn.y + ac.sensor.sensing_range * sin.(th), 
					linestyle = "--", color = "green",
					linewidth = 0.4)
			else
				plot(ac.dyn.x + ac.sensor.sensing_range * cos.(th), 
					ac.dyn.y + ac.sensor.sensing_range * sin.(th), 
					linestyle = "--", color = "red",
					linewidth = 0.4)
			end

			arrow_len = airspace_dim / 12
	
			arrow(
				ac.dyn.x,
				ac.dyn.y,
				cos(ac.dyn.heading) * arrow_len * 0.8, # dx
				sin(ac.dyn.heading) * arrow_len * 0.8, # dy
				head_width = airspace_dim / 70,
				width = 1,
				head_length = airspace_dim / 70,
				overhang = 0.5,
				head_starts_at_zero = "true",
				facecolor = "black",
				length_includes_head = "true")
	
			scatter(ac.x_dest, ac.y_dest, 
					marker = ",", color = "magenta", s = 8)
	
			plot([ac.dyn.x; ac.x_dest], [ac.dyn.y; ac.y_dest], 
				linestyle = "--", color = "black", linewidth = 1)

			if render_option.if_annotate
				annotate(string(ac.advisory),
					xy = [ac.dyn.x; ac.dyn.y],# Arrow tip
					xytext = [ac.dyn.x + arrow_len/5; ac.dyn.y + arrow_len/5 * 2],
					xycoords = "data")
	
				turn_rate_str = string(rad2deg(ac.dyn.turn_rate))
				if ac.dyn.turn_rate != 0
					turn_rate_str = turn_rate_str[1:4]
				end
	
				annotate(turn_rate_str,
					xy = [ac.dyn.x; ac.dyn.y],# Arrow tip
					xytext = [ac.dyn.x + arrow_len/5; ac.dyn.y],
					xycoords = "data")
	
				annotate(string(round(rad2deg(ac.dyn.heading))),
					xy = [ac.dyn.x; ac.dyn.y],# Arrow tip
					xytext = [ac.dyn.x + arrow_len/5; ac.dyn.y - arrow_len/5 * 2],
					xycoords = "data")
			end
		end
	end

	xlabel("X (ft)")
	ylabel("Y (ft)")
	title(string(t) * ": " * string(length(All_AC)))
	axis("equal")
	ax = gca()
	ax[:set_xlim]([0, airspace_dim])
	ax[:set_ylim]([0, airspace_dim])

	pause(render_option.pause_time)
	clf()
end



# function CAS_plot(t::Float64,
# 				plot_option::Plot_Evaluation_Results,
# 				stats::Stats)
# 	title = "Number of Aircraft at Each Time " * string(t)
# 	figure(title)
# 	plot(stats.t, stats.num_ac, linestyle = "-", color = "b", label = "Number of Aircraft")
# 	xlabel("Simulation Time")
# 	ylabel("Number of Aircraft at t")
# 	savefig("../Simulation_Results/eval_plots/" * title * ".pdf")

# 	title = "Cumulative Number of Aircraft Arrived" * string(t)
# 	figure(title)
# 	plot(stats.t, stats.num_ac_arrived, linestyle = "-", color = "r", label = "Number of Aircraft Arrived")
# 	xlabel("Simulation Time")
# 	ylabel("Number of Aircraft Arrived")
# 	savefig("../Simulation_Results/eval_plots/" * title * ".pdf")

# 	title = "Number of NMAC at Each Step " * string(t)
# 	figure(title)
# 	plot(stats.t, stats.num_NMAC_t, linestyle = "-", color = "b", linewidth = 0.3)
# 	xlabel("Simulation Time")
# 	ylabel("Number of NMAC at Each Step")
# 	savefig("../Simulation_Results/eval_plots/" * title * ".pdf")

# 	title = "Cumulative Number of NMAC " * string(t)
# 	figure(title)
# 	plot(stats.t, stats.num_NMAC_cum, linestyle = "-", color = "r", label = "Cumulative Number of NMAC")
# 	xlabel("Simulation Time")
# 	ylabel("Cumulative Number of NMAC")
# 	savefig("../Simulation_Results/eval_plots/" * title * ".pdf")

# 	title = "Average Flight Time " * string(t)
# 	figure(title)
# 	plot(stats.t, stats.average_flight_time, linestyle = "-", color = "b")
# 	xlabel("Simulation Time")
# 	ylabel("Average Flight Time")
# 	savefig("../Simulation_Results/eval_plots/" * title * ".pdf")

# 	show()
# end



