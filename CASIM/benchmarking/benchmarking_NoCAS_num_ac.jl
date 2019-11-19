push!(LOAD_PATH, "../src")

using CASIM

render_option = Not_Render()
sim_horizon = 5000
num_run = 3
init_num_ac = 100
const_num_ac = [150, 200, 250, 300, 350, 400, 450]


"""
	NoCAS
	num_ac
"""
exp = "ConstNumAC_NoCAS_10km_init_ac_" * string(init_num_ac)
if !ispath("../simulation_results/ConstNumAC_NoCAS_10km_init_ac_" * string(init_num_ac))
	mkdir("../simulation_results/ConstNumAC_NoCAS_10km_init_ac_" * string(init_num_ac))
end
cas = No_CAS()
cas_ctrl = Uniform_CAS(cas)
sensor = Nearest_Intruder_Tracker()
sensor_ctrl = Uniform_Sensor(sensor)

for run in 1:num_run
	for num_ac in const_num_ac
		scheme =  "NoCAS_num_ac_" * 
			string(Int(num_ac)) * "_run_" * string(run)
		airspace_ctrl = Constant_Number_Controller(num_ac)
		simulate(exp, scheme, sim_horizon, airspace_ctrl, cas_ctrl, sensor_ctrl, render_option, correction=false)
	end
end
