push!(LOAD_PATH, "./")
using CAS_QMDP
using ArgParse

# Default:
	# penalty_action=0.02, 
	# penalty_closeness=10.0, 
	# penalty_nmac=1000.0, 
	# penalty_conflict=1.0

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table s begin
		"--filename"
			help = "Filename of the alpha matrix"
			arg_type = String
			default = "default_filename"
		"--sample"
			help = "If using random sampled noise"
			arg_type = Bool
			default = false
		"--discount"
			help = "Discount factor"
			arg_type = Float64
			default = 0.95
		"--penalty_action"
			help = "Reward function parameter: penalty on action"
			arg_type = Float64
			default = 5.0
		"--penalty_closeness"
			help = "Reward function parameter: penalty on closeness between agents"
			arg_type = Float64
			default = 1.0
		"--penalty_nmac"
			help = "Reward function parameter: penalty on near mid-air collisions"
			arg_type = Float64
			default = 1000.0
		"--penalty_conflict"
			help = "Reward function parameter: penalty on conflict"
			arg_type = Float64
			default = 50.0
	end
	return parse_args(s)
end


function main()
	parsed_args = parse_commandline()

	filename = parsed_args["filename"]
	if_sample = parsed_args["sample"]
	discount = parsed_args["discount"]
	penalty_action = parsed_args["penalty_action"]
	penalty_closeness = parsed_args["penalty_closeness"]
	penalty_nmac = parsed_args["penalty_nmac"]
	penalty_conflict = parsed_args["penalty_conflict"]

	if filename == "default_filename"
		filename = disc * "_maxdist_2km_nmac_150m" *
			"_Discount_" * string(discount) * 
			"_ActWeight_" * string(actual_weight)[1 : min(length(string(actual_weight)), 4)] * 
			"_IfSample_" * string(if_sample) * 
			"_PenAction_" * string(penalty_action) * 
			"_PenConflict_" * string(penalty_conflict) * 
			"_PenCloseness_" * string(penalty_closeness) *
			"_PenNmac_" * string(penalty_nmac) *
			sigma_str
	end
	# filename = "test_mid2_2"

	open("./VICAS_parameter_sweep_logs.csv", "a") do io
        	@printf(io, "\n")
            @printf(io, "%s,%s,%f,%f,%f,%f,%f,%f,%f", 
            	string(Dates.now()), disc, penalty_action, penalty_closeness, penalty_nmac, penalty_conflict,
            	sigma_v, sigma_alert_deg, sigma_coc_deg)
    end

	policy = qmdp(filename, 
		discount=discount,
		if_sample=if_sample,
		penalty_action=penalty_action, 
		penalty_conflict=penalty_conflict, 
		penalty_closeness=penalty_closeness,
		penalty_nmac=penalty_nmac)
	write_alphas(policy, filename)
end

main()