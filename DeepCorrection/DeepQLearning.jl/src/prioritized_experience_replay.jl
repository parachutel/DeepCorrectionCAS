# Naive implementation

# using Simulation_Env_module

mutable struct PrioritizedReplayBuffer
    max_size::Int64
    batch_size::Int64
    rng::AbstractRNG
    α::Float64
    β::Float64
    ϵ::Float64
    _curr_size::Int64
    _idx::Int64
    _priorities::Vector{Float64}
    _experience::Vector{DQExperience}

    _s_batch::Array{Float64}
    _a_batch::Vector{Int64}
    _r_batch::Vector{Float64}
    _sp_batch::Array{Float64}
    _done_batch::Vector{Bool}
    _weights_batch::Vector{Float64}

    function PrioritizedReplayBuffer(env::AbstractEnvironment,
                                    max_size::Int64,
                                    batch_size::Int64;
                                    rng::AbstractRNG = MersenneTwister(0),
                                    α::Float64 = 0.6,
                                    β::Float64 = 0.4,
                                    ϵ::Float64 = 1e-3)
        s_dim = obs_dimensions(env)
        experience = Vector{DQExperience}(max_size)
        priorities = Vector{Float64}(max_size)
        _s_batch = zeros(batch_size, s_dim...)
        _a_batch = zeros(Int64, batch_size)
        _r_batch = zeros(batch_size)
        _sp_batch = zeros(batch_size, s_dim...)
        _done_batch = zeros(Bool, batch_size)
        _weights_batch = zeros(Float64, batch_size)
        return new(max_size, batch_size, rng, α, β, ϵ, 0, 1, priorities, experience,
                   _s_batch, _a_batch, _r_batch, _sp_batch, _done_batch, _weights_batch)
    end
end


is_full(r::PrioritizedReplayBuffer) = r._curr_size == r.max_size

max_size(r::PrioritizedReplayBuffer) = r.max_size

function add_exp!(r::PrioritizedReplayBuffer, expe::DQExperience, td_err::Float64=abs(expe.r))
    @assert td_err + r.ϵ > 0.
    priority = (td_err + r.ϵ)^r.α
    r._experience[r._idx] = expe
    r._priorities[r._idx] = priority
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
end

function update_priorities!(r::PrioritizedReplayBuffer, indices::Vector{Int64}, td_errors::Float64)
    @assert td_errors + r.ϵ .> 0.
    new_priorities = (td_err + r.ϵ)^r.α
    r._priorities[indices] = new_priorities
end

function StatsBase.sample(r::PrioritizedReplayBuffer)
    @assert r._curr_size >= r.batch_size
    @assert r.max_size >= r.batch_size # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, Weights(r._priorities[1:r._curr_size]), r.batch_size, replace=false)
    @assert length(sample_indices) == size(r._s_batch)[1]
    for (i, idx) in enumerate(sample_indices)
        r._s_batch[Base.setindex(indices(r._s_batch), i, 1)...] = r._experience[idx].s
        r._a_batch[i] = r._experience[idx].a
        r._r_batch[i] = r._experience[idx].r
        r._sp_batch[Base.setindex(indices(r._sp_batch), i, 1)...] = r._experience[idx].sp
        r._done_batch[i] = r._experience[idx].done
        r._weights_batch[i] = r._priorities[idx]
    end
    r._weights_batch = (r._weights_batch.*r._curr_size).^-r.β./maximum(r._weights_batch)
    return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch, sample_indices, r._weights_batch
end

function populate_replay_buffer!(replay::PrioritizedReplayBuffer, env::AbstractEnvironment;
                                 max_pop::Int64=replay.max_size, max_steps::Int64=2000)
    render_option = env.render
    env.render = false
    println("Start populating PRIORITIZED replay buffer...")
    # println("Using closest intruder detection...")
    if !env.random_populate_replay_buffer
        println("Using multi intruder detection ...")
    else
        println("Using random ...")
    end
    o = reset(env)
    done = false
    step = 0
    for t = 1:(max_pop - replay._curr_size)
        if !env.random_populate_replay_buffer
            q_low_fi = lowfi_values(env, o) # multi-threat
            # q_low_fi = get_qvals(env.low_fi_policy, convert_ac_state(env.ego_agent.state)) # closest
            action = actions(env)[indmax(q_low_fi)]
        else
            action = sample_action(env)
        end
        ai = action_index(env.problem, action)
        op, rew, _, done, info = step!(env, action)
        exp = DQExperience(o, ai, rew, op, done)
        add_exp!(replay, exp, abs(rew)) # assume initial td error is r
        o = op
        step += 1
        if done || step >= max_steps
            o = reset(env)
            done = false
            step = 0
        end
    end
    @assert replay._curr_size >= replay.batch_size
    println("Done populating replay buffer!")
    env.render = render_option
end
