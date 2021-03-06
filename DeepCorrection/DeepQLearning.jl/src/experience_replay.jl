struct DQExperience
    s::Array{Float64}
    a::Int64
    r::Float64
    sp::Array{Float64}
    done::Bool
end


mutable struct ReplayBuffer
    max_size::Int64 # maximum size of the buffer
    batch_size::Int64
    rng::AbstractRNG
    _curr_size::Int64
    _idx::Int64
    _experience::Vector{DQExperience}

    _s_batch::Array{Float64}
    _a_batch::Vector{Int64}
    _r_batch::Vector{Float64}
    _sp_batch::Array{Float64}
    _done_batch::Vector{Bool}

    function ReplayBuffer(env::AbstractEnvironment,
                          max_size::Int64,
                          batch_size::Int64,
                          rng::AbstractRNG = MersenneTwister(0))
        s_dim = obs_dimensions(env)
        experience = Vector{DQExperience}(max_size)
        _s_batch = zeros(batch_size, s_dim...)
        _a_batch = zeros(Int64, batch_size)
        _r_batch = zeros(batch_size)
        _sp_batch = zeros(batch_size, s_dim...)
        _done_batch = zeros(Bool, batch_size)
        return new(max_size, batch_size, rng, 0, 1, experience,
                   _s_batch, _a_batch, _r_batch, _sp_batch, _done_batch)
    end
end

is_full(r::ReplayBuffer) = r._curr_size == r.max_size

max_size(r::ReplayBuffer) = r.max_size

function add_exp!(r::ReplayBuffer, expe::DQExperience)
    r._experience[r._idx] = expe
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
end

# convert array of DQExp to batches
function _encode_sample!(r::ReplayBuffer, samples::Vector{DQExperience})
    @assert length(samples) == size(r._s_batch[1])
    for (i, sample) in enumerate(samples)
        r._s_batch[i] = sample.s
        r._a_batch[i] = sample.a
        r._r_batch[i] = sample.r
        r._sp_batch[i] = sample.sp
        r._done_batch[i] = sample.done
    end
end


function StatsBase.sample(r::ReplayBuffer)
    @assert r._curr_size >= r.batch_size
    @assert r.max_size >= r.batch_size # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, r.batch_size, replace=false)
    @assert length(sample_indices) == size(r._s_batch)[1]
    for (i, idx) in enumerate(sample_indices)
        r._s_batch[Base.setindex(indices(r._s_batch), i, 1)...] = r._experience[idx].s
        r._a_batch[i] = r._experience[idx].a
        r._r_batch[i] = r._experience[idx].r
        r._sp_batch[Base.setindex(indices(r._sp_batch), i, 1)...] = r._experience[idx].sp
        r._done_batch[i] = r._experience[idx].done
    end
    return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch
end

function populate_replay_buffer!(replay::ReplayBuffer, env::AbstractEnvironment;
                                 max_pop::Int64=replay.max_size, max_steps::Int64=2000)
    o = reset(env)
    done = false
    step = 0
    println("Start populating replay buffer...")
    for t=1:(max_pop - replay._curr_size)
        if !env.random_populate_replay_buffer
            q_low_fi = get_qvals(env.low_fi_policy, convert_ac_state(env.ego_agent.state))
            action = actions(env)[indmax(q_low_fi)]
        else
            action = sample_action(env)
        end
        ai = action_index(env.problem, action)
        op, rew, _, done, info = step!(env, action)
        exp = DQExperience(o, ai, rew, op, done)
        add_exp!(replay, exp)
        o = op
        # println(o, " ", action, " ", rew, " ", done, " ", info) #TODO verbose?
        step += 1
        if done || step >= max_steps
            o = reset(env)
            done = false
            step = 0
        end
    end
    @assert replay._curr_size >= replay.batch_size
    println("Done populating replay buffer!")
end
