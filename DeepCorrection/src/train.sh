/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia \
	train.jl \
	--random_seed 16 \
	--init_num_env_agents 6 \
	--const_num_env_agents 6 \
	--num_sections 4 \
	--num_ep_eval 50 \
	--train_freq  2 \
	--target_update_freq 5000 \
	--max_steps 5000000 \
	--save_freq 5000 \
	--buffer_size 300000 \
	--train_start 100000 \
	--batch_size 64 \
	--eps_fraction 0.5 \
	--if_log false \
	--lr 1e-4 \
	--render true \
	--stochastic_env_policy true \
	--stochastic_ego_policy false \
	--nn_policy true \
	--correction true \
	--weighted_correction_sum true \
	--correction_weight 0.25 \
	--prioritized_replay true \
	--double_q true \
	--dueling true \
	--random_populate_replay_buffer true \
	--notes "non-normalized q_corr and q_lo, weighted sum, populate buffer: multi, reward scaling: max left out extreme, min kept extreme, full_state: closest"




	# experiment on dueling
	# --exp get_render
	# --exp correct_loss_debug_corr_weight_0.5
	

# normalize the state vector!
# simplify the reward function structure
# get rid of constant v state

# increase buffer_size on the order of 100k
# smaller update freq

# stacking the input to the network
