mutable struct DQNPolicy <: Policy
    q::Tensor # Q network
    s::Tensor # placeholder
    env::AbstractEnvironment
    sess
end

function get_action(policy::DQNPolicy, o::Array{Float64})
    # cannot take a batch of observations
    o = reshape(o, (1, size(o)...))
    TensorFlow.set_def_graph(policy.sess.graph)
    q_val = run(policy.sess, policy.q, Dict(policy.s => o))
    ai = indmax(q_val)
    return actions(policy.env)[ai]
end

"""
    method for getting aciton with corrected policy
"""
function normalize_qvals(qvals::Vector{Float64})
    return qvals / sqrt(sum(qvals .^ 2))
end

function get_action(policy::DQNPolicy, o::Array{Float64}, low_fi_state::Vector{Float64})
    if policy.env.nn_policy == false
        q_low_fi = vec(lowfi_values(policy.env, o))
    
        if policy.env.correction == true
            # cannot take a batch of observations
            o = reshape(o, (1, size(o)...))
            TensorFlow.set_def_graph(policy.sess.graph)
            q_correction = run(policy.sess, policy.q, Dict(policy.s => o) ) # Array{Float32,2}
            # reshape q_correction
            temp_arr = zeros(n_actions(policy.env)) # Array{Float64,1}
            for i in 1 : n_actions(policy.env)
                temp_arr[i] = q_correction[1, i]
            end
            q_correction = temp_arr # non-normalized
            # q_correction = normalize_qvals(temp_arr) # Array{Float64,1}
            # corrected qvals:
            q_decision = policy.env.correction_weight * q_correction + (1 - policy.env.correction_weight) * q_low_fi
            # q_decision = q_correction + q_low_fi
            
        else # no correction
            q_decision = q_low_fi
        end
    else # if policy.env.nn_policy == true
        o = reshape(o, (1, size(o)...))
        TensorFlow.set_def_graph(policy.sess.graph)
        q_decision = run(policy.sess, policy.q, Dict(policy.s => o) ) # Array{Float32,2}
    end
    return actions(policy.env)[indmax(q_decision)]
end

function get_value(policy::DQNPolicy, o::Array{Float64})
    o = reshape(o, (1, size(o)...))
    TensorFlow.set_def_graph(policy.sess.graph)
    q_val = run(policy.sess, policy.q, Dict(policy.s => o))
    return q_val
end

function get_value(sess, q, o)
    TensorFlow.set_def_graph(policy.sess.graph)
    o = reshape(o, (1, size(o)...))
    q_val = run(sess, q, Dict(s => o) )
    return q_val
end

function get_value_batch(sess, q, o)
    TensorFlow.set_def_graph(policy.sess.graph)
    q_val = run(sess, q, Dict(s => o) )
    return q_val
end

function POMDPs.action(policy::DQNPolicy, s::S) where S
    obs = nothing
    if isa(policy.env.problem, POMDP)
        obs = convert_o(Vector{Float64}, s, policy.env.problem)
    else
        obs = convert_s(Vector{Float64}, s, policy.env.problem)
    end
    return get_action(policy, obs)
end

function POMDPs.value(policy::DQNPolicy, s)
    obs = nothing
    if isa(policy.env.problem, POMDP)
        obs = convert_o(Array{Float64}, s, policy.env.problem)
    else
        obs = convert_s(Array{Float64}, s, policy.env.problem)
    end
    return get_value(policy, obs)
end






############################################
############################################
############################################




mutable struct LSTMPolicy <: Policy
    q::Tensor # Q network
    state::LSTMStateTuple
    s::Tensor # placeholder
    state_ph::LSTMStateTuple # hidden state placeholder
    state_val::LSTMStateTuple
    env::AbstractEnvironment
    sess
end

function LSTMPolicy(sess, env, arch, dueling)
    obs_dim = obs_dimensions(env)
    n_outs = n_actions(env)
    init_c = zeros(1, arch.lstm_size)
    init_h = zeros(1, arch.lstm_size)
    # bs x trace_length x dim
    s1 = placeholder(Float32, shape=[-1, 1, obs_dim...])
    q1, q1_init, q1_state = build_recurrent_q_network(s1,
                                                   arch.convs,
                                                   arch.fc_in,
                                                   arch.fc_out,
                                                   arch.lstm_size,
                                                   n_actions(env),
                                                   final_activation = identity,
                                                   scope= Q_SCOPE,
                                                   reuse=true,
                                                   dueling = dueling)
    state_val = LSTMStateTuple(init_c, init_h)
    return LSTMPolicy(q1, q1_state, s1, q1_init, state_val, env, sess)
end

function get_action!(policy::LSTMPolicy, o::Array{Float64}, sess) # update hidden state
    # cannot take a batch of observations
    o = reshape(o, (1, 1, size(o)...))
    feed_dict = Dict(policy.s => o, policy.state_ph => policy.state_val)
    q_val, state_val = run(sess, [policy.q, policy.state], feed_dict)
    policy.state_val = state_val
    ai = indmax(q_val)
    return actions(policy.env)[ai]
end

function POMDPs.action(policy::LSTMPolicy, o::Array{Float64})
    return get_action!(policy, o, policy.sess)
end


function get_value(policy::LSTMPolicy, o::Array{Float64}, sess) # update hidden state
    # cannot take a batch of observations
    o = reshape(o, (1, 1, size(o)...))
    feed_dict = Dict(policy.s => o, policy.state_ph => policy.state_val)
    q_val, state_val = run(sess, [policy.q, policy.state], feed_dict)
    return q_val
end

function get_value!(policy::LSTMPolicy, o::Array{Float64}, sess) # update hidden state
    # cannot take a batch of observations
    o = reshape(o, (1, 1, size(o)...))
    feed_dict = Dict(policy.s => o, policy.state_ph => policy.state_val)
    q_val, state_val = run(sess, [policy.q, policy.state], feed_dict)
    policy.state_val = state_val
    return q_val
end

function reset_hidden_state!(policy::LSTMPolicy)# could use zero_state from tf
    hidden_size = get(get_shape(policy.state_ph.c).dims[end])
    init_c = zeros(1, hidden_size)
    init_h = zeros(1, hidden_size)
    policy.state_val = LSTMStateTuple(init_c, init_h)
end

# XXX be careful!! this change the hidden state of the policy
function POMDPs.value(policy::LSTMPolicy, o::Array{Float64})
    return get_value!(policy, o, policy.sess)
end
