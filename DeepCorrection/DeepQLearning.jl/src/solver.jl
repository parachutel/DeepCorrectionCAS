
function POMDPs.solve(solver::DeepQLearningSolver, env::AbstractEnvironment)
    train_graph = build_graph(solver, env)

    # init and populate replay buffer
    if solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)

    # init variables
    run(train_graph.sess, global_variables_initializer())
    # train 
    dqn_train(solver, env, train_graph, replay)
    policy = DQNPolicy(train_graph.q, train_graph.s, env, train_graph.sess)
    return policy
end

function POMDPs.solve(solver::DeepQLearningSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.rng)
    # init session and build graph Create a TrainGraph object with all the tensors
    return solve(solver, env)
end



"""
    main training component
"""
function dqn_train(solver::DeepQLearningSolver,
                   env::AbstractEnvironment,
                   graph::TrainGraph,
                   replay::Union{ReplayBuffer, PrioritizedReplayBuffer})
    
    if solver.if_log == true
        summary_writer = tf.summary.FileWriter(solver.logdir)
    end

    # initialization
    obs = reset(env)
    done = false
    step = 0
    rtot = 0
    episode_rewards = Float64[0.0]
    episode_lengths = Int64[0]
    saved_best_reward = - Inf
    scores_eval = 0.
    non_corrected_scores_eval = 0.
    num_nmac = 0.
    non_corrected_num_nmac = 0.
    eval_episode_length = 0
    eval_action_stats = ones(n_actions(env)) / n_actions(env)
    eps = solver.eps_start
    weights = ones(solver.batch_size)
    model_saved = false
    policy = DQNPolicy(graph.q, graph.s, env, graph.sess)

    # train start:
    for t = 1:solver.max_steps
        # epsilon greedy policy
        if solver.train
            # if rand(solver.rng) > eps
            if rand() > eps
                action = get_action(graph, solver, env, obs) # feed into DQN
            else
                action = sample_action(env)
            end
        else
            action = get_action(graph, solver, env, obs)
        end
        # update epsilon
        if t < solver.eps_fraction * solver.max_steps
            eps = 1 - (1 - solver.eps_end) / (solver.eps_fraction * solver.max_steps) * t # decay
        else
            eps = solver.eps_end
        end
        ai = action_index(env.problem, action)

        # render_option = env.render
        # env.render = false
        op, rew, _,  done, info = step!(env, action)
        # restore render_option for visualiztion during eval
        # env.render = render_option

        exp = DQExperience(obs, ai, rew, op, done)
        add_exp!(replay, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        episode_lengths[end] += 1

        if done || step >= solver.max_episode_length
            obs = reset(env)
            push!(episode_rewards, 0.0)
            push!(episode_lengths, 0)
            done = false
            step = 0
            rtot = 0
        end
        num_episodes = length(episode_rewards)
        avg100_reward = mean(episode_rewards[max(1, num_episodes - 101):end])

        # batch_train:
        if t % solver.train_freq == 0
            if solver.prioritized_replay
                s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
            else
                weights = ones(replay.batch_size)
                s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
            end

            # q_lo's are normalized
            q_lo_batch, q_lo_p_batch = get_q_lo_batch(graph, env, s_batch, sp_batch) # returns Array{Float64}

            feed_dict = Dict(graph.s => s_batch,
                             graph.a => a_batch,
                             graph.sp => sp_batch,
                             graph.r => r_batch,
                             graph.done_mask => done_batch,
                             graph.importance_weights => weights,
                             graph.q_lo => q_lo_batch,
                             graph.q_lo_p => q_lo_p_batch)

            loss_val, td_errors, grad_val, _ = 
                run(graph.sess, [graph.loss, graph.td_errors, graph.grad_norm, graph.train_op], feed_dict)
            # loss_val, td_errors, grad_val, _, q = 
            #     run(graph.sess, [graph.loss, graph.td_errors, graph.grad_norm, graph.train_op, graph.q], feed_dict)
        end

        if t % solver.target_update_freq == 0
            println("Target updated!")
            run(graph.sess, graph.update_op)
        end

        if t % solver.eval_freq == 0
            print(@sprintf("%5d / %5d  ", t, solver.max_steps))
            scores_eval, non_corrected_scores_eval, num_nmac, non_corrected_num_nmac, eval_episode_length, eval_action_stats = 
                eval_q(graph, solver, env, t,
                    total_n_steps=solver.max_steps,
                    n_eval=solver.num_ep_eval, 
                    max_episode_length=solver.max_episode_length,
                    verbose=solver.verbose)
        end

        if t % solver.log_freq == 0 && t >= solver.eval_freq
            if solver.if_log == true
                # log to tensorboard
                tb_avgr = logg_scalar(avg100_reward, "avg_reward")
                tb_evalr = logg_scalar(scores_eval, "eval_reward")
                tb_noncorrected_evalr = logg_scalar(non_corrected_scores_eval, "non_corrected_eval_reward")
                tb_num_nmac = logg_scalar(num_nmac, "num_nmac")
                tb_noncorrected_num_nmac = logg_scalar(non_corrected_num_nmac, "non_corrected_num_nmac")
                tb_loss = logg_scalar(loss_val, "loss")
                tb_tderr = logg_scalar(mean(td_errors), "mean_td_error")
                tb_grad = logg_scalar(grad_val, "grad_norm")
                if done
                    tb_epreward = logg_scalar(episode_rewards[end], "episode_reward")
                    tb_eplength = logg_scalar(episode_lengths[end], "episode_length")
                else # if the episode is not yet done
                    tb_epreward = logg_scalar(episode_rewards[end - 1], "episode_reward")
                    tb_eplength = logg_scalar(episode_lengths[end - 1], "episode_length")
                end
                tb_eps = logg_scalar(eps, "epsilon")

                tb_eval_episode_length = logg_scalar(eval_episode_length, "eval_episode_length")
                write(summary_writer, tb_eval_episode_length, t)

                for ai in 1:n_actions(env)
                    write(summary_writer, 
                        logg_scalar(eval_action_stats[ai], "eval_action_freq_" * string(actions(env)[ai])), t)
                end

                write(summary_writer, tb_avgr, t)
                write(summary_writer, tb_evalr, t)
                write(summary_writer, tb_noncorrected_evalr, t)
                write(summary_writer, tb_num_nmac, t)
                write(summary_writer, tb_noncorrected_num_nmac, t)
                write(summary_writer, tb_loss, t)
                write(summary_writer, tb_tderr, t)
                write(summary_writer, tb_grad, t)
                write(summary_writer, tb_epreward, t)
                write(summary_writer, tb_eplength, t)
                write(summary_writer, tb_eps, t)
            end # if_log

            if solver.verbose
                logg = @sprintf("%5d / %5d eps %0.3f | avgR %1.3f | Loss %2.3f | Grad %2.6f | num_episodes %3d" ,
                                 t, solver.max_steps, eps, avg100_reward, loss_val, grad_val, num_episodes)
                println(logg)
            end
        end

        if t % solver.save_freq == 0 && solver.if_log == true
            if scores_eval >= saved_best_reward
                if solver.verbose
                    println("Saving new model with the best eval reward ", scores_eval)
                end
                policy = DQNPolicy(graph.q, graph.s, env, graph.sess)
                JLD.save(solver, policy, 
                    weights_file=solver.logdir*"weights.jld", 
                    problem_file=solver.logdir*"problem.jld")
                model_saved = true
                saved_best_reward = scores_eval
            end
        end

    end # for t = 1:solver.max_steps

    if model_saved
        if solver.verbose
            println("Restore model with eval reward ", saved_best_reward)
        end
    end

    return policy
end


"""
Evaluate a Q network
"""
function eval_q(graph::TrainGraph,
                solver::DeepQLearningSolver,
                env::AbstractEnvironment,
                t::Int64;
                total_n_steps::Int64=4000000,
                n_eval::Int64=100,
                max_episode_length::Int64=2000,
                verbose::Bool=true)
    # Evaluation
    dummy_problem = Dummy_Problem(0.99)
    avg_r = 0
    avg_num_nmac = 0
    eval_action_stats = zeros(n_actions(env))
    eval_episode_length = 0

    for i = 1 : n_eval
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        while !done && step <= max_episode_length
            action = get_action(graph, solver, env, obs)
            ai = action_index(dummy_problem, action)
            eval_action_stats[ai] += 1
            obs, rew, num_nmac, done, info = step!(env, action)
            avg_num_nmac += num_nmac
            avg_r += rew
            step += 1
            eval_episode_length += 1
        end
    end
    avg_num_nmac /= n_eval
    avg_r /= n_eval
    eval_episode_length /= n_eval
    eval_action_stats /= sum(eval_action_stats)


    # Evaluate uncorrected env (using original VICAS on ego agent):
    # record the original correction option and nn_policy option
    correction_option = env.correction
    nn_policy_option = env.nn_policy
    stochastic_ego_policy_option = env.stochastic_ego_policy

    # temporarily deactivate settings
    env.correction = false
    env.nn_policy = false
    env.stochastic_ego_policy = false


    non_corrected_avg_r = 0
    non_corrected_avg_num_nmac = 0
    non_corrected_n_eval = n_eval
    for i = 1 : non_corrected_n_eval # less n_eval for non-corrected policy to save time
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        while !done && step <= max_episode_length
            action = get_action(graph, solver, env, obs)
            obs, rew, num_nmac ,done, info = step!(env, action)
            non_corrected_avg_r += rew
            non_corrected_avg_num_nmac += num_nmac
            step += 1
        end
    end
    non_corrected_avg_num_nmac /= non_corrected_n_eval
    non_corrected_avg_r /= non_corrected_n_eval

    # restore correction option and nn_policy option
    env.correction = correction_option
    env.nn_policy = nn_policy_option
    env.stochastic_ego_policy = stochastic_ego_policy_option

    if verbose
        println("\nEvaluation ................. Avg Reward ", avg_r)
        println("Evaluation ... Non Corrected Avg Reward ", non_corrected_avg_r)
    end

    return  avg_r, non_corrected_avg_r, avg_num_nmac, non_corrected_avg_num_nmac, 
            eval_episode_length, eval_action_stats
end
































# unmodified code for recurrent net policy
##############################################################################################################

function POMDPs.solve(solver::DeepRecurrentQLearningSolver, problem::Union{MDP,POMDP})
    if !isa(problem, POMDP)
        env = MDPEnvironment(problem, rng=solver.rng)
    else
        env = POMDPEnvironment(problem, rng=solver.rng)
    end
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepRecurrentQLearningSolver, env::AbstractEnvironment)
    #init session and build graph Create a TrainGraph object with all the tensors
    train_graph = build_graph(solver, env)
    # init and populate replay buffer
    replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    #TODO save the training log somewhere
    drqn_train(solver, env, train_graph, replay)
    policy = train_graph.lstm_policy
    policy.sess = train_graph.sess
    return policy
end


function drqn_train(solver::DeepRecurrentQLearningSolver,
                   env::AbstractEnvironment,
                   graph::RecurrentTrainGraph,
                   replay::EpisodeReplayBuffer)
    summary_writer = tf.summary.FileWriter(solver.logdir)
    obs = reset(env)
    reset_hidden_state!(graph.lstm_policy)
    done = false
    step = 0
    rtot = 0
    episode = DQExperience[]
    sizehint!(episode, solver.max_episode_length)
    episode_rewards = Float64[0.0]
    saved_mean_reward = 0.
    model_saved = false
    scores_eval = 0.0
    eps = 1.0
    weights = ones(solver.batch_size*solver.trace_length)
    init_c = zeros(solver.batch_size, solver.arch.lstm_size)
    init_h = zeros(solver.batch_size, solver.arch.lstm_size)
    grad_val, loss_val = -1, -1 # sentinel value
    for t=1:solver.max_steps
        if rand(solver.rng) > eps
            action = get_action!(graph.lstm_policy, obs, graph.sess)
        else
            action = sample_action(env)
        end
        # update epsilon
        if t < solver.eps_fraction*solver.max_steps
            eps = 1 - (1 - solver.eps_end)/(solver.eps_fraction*solver.max_steps)*t # decay
        else
            eps = solver.eps_end
        end
        ai = action_index(env.problem, action)
        op, rew, _, done, info = step!(env, action)
        exp = DQExperience(obs, ai, rew, op, done)
        push!(episode, exp)
        obs = op
        step += 1
        episode_rewards[end] += rew
        if done || step >= solver.max_episode_length
            add_episode!(replay, episode)
            episode = DQExperience[] # empty episode
            obs = reset(env)
            reset_hidden_state!(graph.lstm_policy)
            push!(episode_rewards, 0.0)
            done = false
            step = 0
            rtot = 0
            # log episode reward

        end
        num_episodes = length(episode_rewards)
        avg100_reward = mean(episode_rewards[max(1, length(episode_rewards) - 101) : end])
        if t%solver.train_freq == 0
            s_batch, a_batch, r_batch, sp_batch, done_batch, trace_mask_batch = sample(replay)
            feed_dict = Dict(graph.s => s_batch,
                             graph.a => a_batch,
                             graph.sp => sp_batch,
                             graph.r => r_batch,
                             graph.done_mask => done_batch,
                             graph.trace_mask => trace_mask_batch,
                             graph.importance_weights => weights,
                             graph.hq_in.c => init_c,
                             graph.hq_in.h => init_h,
                             graph.hqp_in.c => init_c,
                             graph.hqp_in.h => init_h,
                             graph.target_hq_in.c => init_c,
                             graph.target_hq_in.h => init_h
                             )
            loss_val, td_errors_val, grad_val, _ = run(graph.sess,
                                                       [graph.loss, graph.td_errors, graph.grad_norm, graph.train_op],
                                                       feed_dict)
        end

        if t%solver.target_update_freq == 0
            run(graph.sess, graph.update_op)
        end

        if t%solver.eval_freq == 0
            # save hidden state before
            hidden_state = graph.lstm_policy.state_val
            scores_eval = eval_lstm(graph.lstm_policy,
                                     env,
                                     graph.sess,
                                     n_eval=solver.num_ep_eval,
                                     max_episode_length=solver.max_episode_length,
                                     verbose = solver.verbose)
            # reset hidden state
            graph.lstm_policy.state_val = hidden_state
        end

        if t%solver.log_freq == 0
            # log to tensorboard
            tb_avgr = logg_scalar(avg100_reward, "avg_reward")
            tb_evalr = logg_scalar(scores_eval[end], "eval_reward")
            tb_loss = logg_scalar(loss_val, "loss")
            tb_tderr = logg_scalar(mean(td_errors_val), "mean_td_error")
            tb_grad = logg_scalar(grad_val, "grad_norm")
            tb_epreward = logg_scalar(episode_rewards[end], "episode_reward")
            tb_eps = logg_scalar(eps, "epsilon")
            write(summary_writer, tb_avgr, t)
            write(summary_writer, tb_evalr, t)
            write(summary_writer, tb_loss, t)
            write(summary_writer, tb_tderr, t)
            write(summary_writer, tb_grad, t)
            write(summary_writer, tb_epreward, t)
            write(summary_writer, tb_eps, t)
            if  solver.verbose
                logg = @sprintf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3f | Grad %2.3f | num_episodes %f",
                                 t, solver.max_steps, eps, avg100_reward, loss_val, grad_val, length(episode_rewards))
                println(logg)
            end
        end
        if t > solver.train_start && t%solver.save_freq == 0
            if scores_eval[end] >= saved_mean_reward
                if solver.verbose
                    println("Saving new model with eval reward ", scores_eval[end])
                end
                saver = tf.train.Saver()
                train.save(saver, graph.sess, solver.logdir*"weights.jld")
                model_saved = true
                saved_mean_reward = scores_eval[end]
            end
        end
    end
    if model_saved
        if solver.verbose
            println("Restore model with eval reward ", saved_mean_reward)
        end
    end
    return
end

function eval_lstm(policy::LSTMPolicy,
                env::AbstractEnvironment,
                sess;
                n_eval::Int64=100,
                max_episode_length::Int64=100,
                verbose::Bool=false)
    # Evaluation
    avg_r = 0
    for i=1:n_eval
        done = false
        r_tot = 0.0
        step = 0
        obs = reset(env)
        reset_hidden_state!(policy)
        # println("start at t=0 obs $obs")
        # println("Start state $(env.state)")
        while !done && step <= max_episode_length
            action = get_action!(policy, obs, sess)
            # println(action)
            obs, rew, _, done, info = step!(env, action)
            # println("state ", env.state, " action ", a)
            # println("Reward ", rew)
            # println(obs, " ", done, " ", info, " ", step)
            r_tot += rew
            step += 1
        end
        avg_r += r_tot
        # println(r_tot)
    end
    if verbose
        println("Evaluation ... Avg Reward ", avg_r/n_eval)
    end
    return  avg_r /= n_eval
end
