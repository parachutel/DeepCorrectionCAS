push!(LOAD_PATH, "../src")

using CASIM

render_option = Not_Render()
sim_horizon = 5000
num_run = 10
init_num_ac = 100
# spawn_rates = [250., 500., 1000., 1500., 2000., 4000., 5000., 6000., 8000., 10000.] ./ 3600
# spawn_rates = [1000., 2000., 4000., 6000., 8000.] ./ 3600
spawn_rates = [250., 500.] ./ 3600
# spawn_rates = [8000., 10000.] ./ 3600

"""
	Corrected_VICAS
	spawn_rate
"""
exp = "SpawnRates_correctionOff_CorrectedVICAS_10km_init_ac_" * string(init_num_ac) * "/" * correction_version * "_final"
if !ispath("../simulation_results/SpawnRates_correctionOff_CorrectedVICAS_10km_init_ac_" * string(init_num_ac))
	mkdir("../simulation_results/SpawnRates_correctionOff_CorrectedVICAS_10km_init_ac_" * string(init_num_ac))
end
cas = Corrected_VICAS()
cas_ctrl = Uniform_CAS(cas)
sensor = Multi_Threat_Tracker()
sensor_ctrl = Uniform_Sensor(sensor)

for run in 1:num_run
	for spawn_rate in spawn_rates
		scheme = "Corrected_sr_" * string(Int(spawn_rate * 3600)) * "_run_" * string(run)
		airspace_ctrl = Constant_Spawn_Rate_Controller(init_num_ac, spawn_rate)
		simulate(exp, scheme, sim_horizon, airspace_ctrl, cas_ctrl, sensor_ctrl, render_option, correction=true)
	end
end