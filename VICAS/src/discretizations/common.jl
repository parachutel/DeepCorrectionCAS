using GridInterpolations
"""
	dimensions of state and action spaces
"""
const state_dim = 5
const action_dim = 2

"""
	state discretization
	s = [ρ, θ, ϕ, v_own, v_int]
"""
const rho_ind   = 1
const theta_ind = 2
const phi_ind   = 3
const v_own_ind = 4
const v_int_ind = 5

const num_states = rho_dim * theta_dim * phi_dim * v_dim * v_dim + 1

states = zeros(state_dim, num_states)
istate = 1
for iv_int in 1:v_dim
	for iv_own in 1:v_dim
		for iphi in 1:phi_dim
			for itheta in 1:theta_dim
				for irho in 1:rho_dim
					states[rho_ind, istate] = discrete_rho[irho]
					states[theta_ind, istate] = discrete_theta[itheta]
					states[phi_ind, istate] = discrete_phi[iphi]
					states[v_own_ind, istate] = discrete_v[iv_own]
					states[v_int_ind, istate] = discrete_v[iv_int]
					istate += 1
				end
			end
		end
	end
end

const terminal_state_var = 1e5
const terminal_state = terminal_state_var * ones(state_dim)
states[:, end] = terminal_state

const grid_states = RectangleGrid(sort(unique(vec(states[rho_ind,   1:end - 1]))),
								  sort(unique(vec(states[theta_ind, 1:end - 1]))),
                      			  sort(unique(vec(states[phi_ind,   1:end - 1]))),
                      			  sort(unique(vec(states[v_own_ind, 1:end - 1]))),
                      			  sort(unique(vec(states[v_int_ind, 1:end - 1]))))


"""
	action space (individual action)
	non-cooperative, the ownship has no knowledge about the action of the intruder
"""
const advisories = [:SR, :WR, :KEEP, :WL, :SL, :COC]
const advisory_to_ind_dict = Dict(:SR => 1, :WR => 2, :KEEP => 3, :WL => 4, 
								:SL => 5, :COC => 6)
const advisory_to_action_dict = Dict(:SR => -10.0, :WR => -5.0, :KEEP => 0.0, 
									:WL => 5.0, :SL => 10.0, :COC => -1.0)
const num_actions = length(advisories)


