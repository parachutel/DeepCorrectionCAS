push!(LOAD_PATH, "./")
push!(LOAD_PATH, "../")
push!(LOAD_PATH, "../../VICAS/src/")

using POMDPs
using DeepQLearning
using Simulation_Env_module
using ArgParse

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table s begin
		"--random_seed"
			help = "Random seed"
			arg_type = Int64
			default = 1
		"--init_num_env_agents"
			help = "Initial number of agents"
			arg_type = Int64
			default = 8
		"--const_num_env_agents"
			help = "Constant number of agents"
			arg_type = Int64
			default = 8
		"--train"
			help = "If train neural network"
			arg_type = Bool
			default = true
		"--num_sections"
			help = "Number of sections, must be even"
			arg_type = Int64
			default = 4
		"--correction"
			help = "If correct the policy"
			arg_type = Bool
			default = true
		"--correction_weight"
			help = "Set the weight of the correction, 0 - 1"
			arg_type = Float64
			default = 0.5
		"--nn_policy"
			help = "If using pure neural network policy, overrides correction and VICAS"
			arg_type = Bool
			default = false
		"--render"
			help = "If render the simulation"
			arg_type = Bool
			default = false
		"--max_steps"
			help = "Total number of training step"
			arg_type = Int64
			default = 1000000
		"--target_update_freq"
			help = "Frequency at which the target network is updated"
			arg_type = Int64
			default = 3000
		"--max_episode_length"
			help = "Maximum length of a training episode"
			arg_type = Int64
			default = 1000
		"--train_start"
			help = "Number of steps used to fill in the replay buffer initially"
			arg_type = Int64
			default = 20000
		"--buffer_size"
			help = "Size of the experience replay buffer"
			arg_type = Int64
			default = 40000
		"--random_populate_replay_buffer"
			help = "Wether randomly populate experience replay buffer"
			arg_type = Bool
			default = false
		"--batch_size"
			help = "Experience replay buffer size"
			arg_type = Int64
			default = 32
		"--lr"
			help = "Learning rate"
			arg_type = Float64
			default = 1e-4
		"--eps_fraction"
			help = "Fraction of epsilon annealing"
			arg_type = Float64
			default = 0.5
		"--eps_start"
			help = "Epsilon starting value"
			arg_type = Float64
			default = 1.0
		"--eps_end"
			help = "Epsilon ending value"
			arg_type = Float64
			default = 0.01
		"--train_freq"
			help = "Frequency at which the active network is updated"
			arg_type = Int64
			default = 2
		"--eval_freq"
			help = "Frequency at which to eval the network"
			arg_type = Int64
			default = 5000
		"--num_ep_eval"
			help = "Number of episodes to evaluate the policy"
			arg_type = Int64
			default = 5
		"--log_freq"
			help = "Frequency at which to log info"
			arg_type = Int64
			default = 1000
		"--save_freq"
			help = "Frequency at which to save problem and weights"
			arg_type = Int64
			default = 10000
		"--arch_fc"
			help = "Specify the architecture of the fully connected Q network"
			arg_type = Vector{Int64}
			default = [64, 32, 32, 32, 32]
		"--double_q"
			help = "If using double q learning udpate"
			arg_type = Bool
			default = false
		"--dueling"
			help = "if using dueling structure for the q network"
			arg_type = Bool
			default = false
		"--prioritized_replay"
			help = "If using prioritized exp replay"
			arg_type = Bool
			default = true
		"--verbose"
			help = "If print info in terminal"
			arg_type = Bool
			default = true
		"--exp"
			help = "Description of the training setups"
			arg_type = AbstractString
			default = "default"
		"--if_log"
			help = "If log the training history"
			arg_type = Bool
			default = true
		"--stochastic_env_policy"
			help = "If using soft max stochastic policy for env agents"
			arg_type = Bool
			default = false
		"--stochastic_ego_policy"
			help = "If using soft max stochastic policy for the ego agent"
			arg_type = Bool
			default = false
		"--notes"
			help = "Miscellaneous notes"
			arg_type = AbstractString
			default = " "
		"--weighted_correction_sum"
			help = "If using weighted sum for additional correction"
			arg_type = Bool
			default = false
	end
	return parse_args(s)
end

function print_settings(parsed_args)
	println("-------------------------------------------------------------------------------------------")
	println("Training Settings:")
	println("train = ", parsed_args["train"])
	println("render = ", parsed_args["render"])
	println("random_seed = ", parsed_args["random_seed"])
	println("init_num_env_agents = ", parsed_args["init_num_env_agents"])
	println("const_num_env_agents = ", parsed_args["const_num_env_agents"])
	println("correction = ", parsed_args["correction"])
	println("weighted_correction_sum = ", parsed_args["weighted_correction_sum"])
	println("correction_weight = ", parsed_args["correction_weight"])
	println("nn_policy = ", parsed_args["nn_policy"])
	println("max_steps = ", parsed_args["max_steps"])
	println("target_update_freq = ", parsed_args["target_update_freq"])
	println("max_episode_length = ", parsed_args["max_episode_length"])
	println("train_start = ", parsed_args["train_start"])
	println("buffer_size = ", parsed_args["buffer_size"])
	println("random_populate_replay_buffer = ", parsed_args["random_populate_replay_buffer"])
	println("batch_size = ", parsed_args["batch_size"])
	println("lr = ", parsed_args["lr"])
	println("eps_fraction = ", parsed_args["eps_fraction"])
	println("eps_start = ", parsed_args["eps_start"])
	println("train_freq = ", parsed_args["train_freq"])
	println("eval_freq = ", parsed_args["eval_freq"])
	println("num_ep_eval = ", parsed_args["num_ep_eval"])
	println("log_freq = ", parsed_args["log_freq"])
	println("save_freq = ", parsed_args["save_freq"])
	println("arch = ", parsed_args["arch_fc"])
	println("double_q = ", parsed_args["double_q"])
	println("dueling = ", parsed_args["dueling"])
	println("prioritized_replay = ", parsed_args["prioritized_replay"])
	println("verbose = ", parsed_args["verbose"])
	println("if_log = ", parsed_args["if_log"])
	println("stochastic_env_policy = ", parsed_args["stochastic_env_policy"])
	println("stochastic_ego_policy = ", parsed_args["stochastic_ego_policy"])
end

function main()
	parsed_args = parse_commandline()

	env = Simple_CASIM_Env(init_num_env_agents=parsed_args["init_num_env_agents"], 
		const_num_env_agents=parsed_args["const_num_env_agents"], 
		num_sections=parsed_args["num_sections"],
		correction=parsed_args["correction"], 
		correction_weight=parsed_args["correction_weight"],
		render=parsed_args["render"],
		stochastic_env_policy=parsed_args["stochastic_env_policy"],
		stochastic_ego_policy=parsed_args["stochastic_ego_policy"],
		nn_policy=parsed_args["nn_policy"],
		random_populate_replay_buffer=parsed_args["random_populate_replay_buffer"],
		rng=MersenneTwister(parsed_args["random_seed"]))
	
	exp = parsed_args["exp"]
	if exp == "default"
		exp = "LOWFI_" * LOW_FI_POLICY * "_" * string(Dates.now())
	end

	print_settings(parsed_args)
	println("lowfi_policy = ", LOW_FI_POLICY_FILENAME)
	println("exp = ", exp)
	println("-------------------------------------------------------------------------------------------")

	# record settings to text file
	if !ispath("../logs/" * exp) && parsed_args["if_log"] == true
		mkdir("../logs/" * exp)
	end
	if parsed_args["if_log"]
		open("../logs/" * exp * "/training_settings.txt", "w") do io
			@printf(io, "lowfi_policy = %s\n", LOW_FI_POLICY_FILENAME)
			@printf(io, "SPAWN_RATE_SET = %s\n", SPAWN_RATE_SET)
			@printf(io, "CONST_NUM_AC_SET = %s\n", CONST_NUM_AC_SET)
			@printf(io, "RELAX_FACTOR = %s\n", RELAX_FACTOR)

        	for key in keys(parsed_args)
            	@printf(io, "%s = %s\n", key, parsed_args[key])
        	end
        	
        	@printf(io, "\n")
        	for key in keys(env.reward_weights)
            	@printf(io, "%s = %s\n", key, env.reward_weights[key])
        	end
    	end
    end

	solver = DeepQLearningSolver(train=parsed_args["train"],
								max_steps=parsed_args["max_steps"], 
								target_update_freq=parsed_args["target_update_freq"],
								max_episode_length=parsed_args["max_episode_length"],
								train_start=parsed_args["train_start"],
								buffer_size=parsed_args["buffer_size"],
								batch_size=parsed_args["batch_size"],
								lr=parsed_args["lr"],
								eps_fraction=parsed_args["eps_fraction"],
								eps_start=parsed_args["eps_start"],
								eps_end=parsed_args["eps_end"],
								train_freq=parsed_args["train_freq"],
								eval_freq=parsed_args["eval_freq"],
								num_ep_eval=parsed_args["num_ep_eval"],
								log_freq=parsed_args["log_freq"],
								save_freq=parsed_args["save_freq"],
								arch=QNetworkArchitecture(conv=[], fc=parsed_args["arch_fc"]),
								double_q=parsed_args["double_q"],
								dueling=parsed_args["dueling"],
								prioritized_replay=parsed_args["prioritized_replay"],
								verbose=parsed_args["verbose"],
								if_log=parsed_args["if_log"],
								logdir="../logs/" * exp * "/",
								weighted_correction_sum=parsed_args["weighted_correction_sum"])

	final_policy = solve(solver, env)
	
	if solver.if_log
		JLD.save(solver, final_policy, 
                    	weights_file=solver.logdir * "final_weights.jld", 
                    	problem_file=solver.logdir * "final_problem.jld")
	end
end

main()

