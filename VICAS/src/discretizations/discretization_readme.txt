The Q table is of the shape (num_states, num_actions), where num_states = 1,950,001
and num_actions = 6.

The state vector is defined by [rho, theta, phi, v_own, v_int, nmac_r], where 
	- rho [m] is the distance between the ownship and the intruder
	- theta [rad] is the relative angle of the intruder with respect to the heading of the ownship
	- phi [rad] is the difference between the headings of the ownship and the intruder, with
	  respect to the heading of the ownship
	- v_own [m/s] is the speed of the ownship
	- v_int [m/s] is the speed of the intruder
	- nmac_r [m] is the NMAC range

The range of the state variables are given by
	- 0 <= rho <= 1800 m
	- 0 <= theta <= 2 * pi
	- 0 <= phi <= 2 * pi
	- 20 m/s <= v_own <= 50 m/s
	- 20 m/s <= v_int <= 50 m/s
	- 100 m <= nmac_r <= 200 m

The resolution for state discretization is
	- rho:    rho_dim    = 65
	- theta:  theta_dim  = 25
	- phi:    phi_dim    = 25
	- v_own:  v_dim      = 4
	- v_int:  v_dim      = 4
	- nmac_r: nmac_r_dim = 3

In addition with 1 terminal state (when the intruder is beyond the upper bound of rho), the number 
of states (num_states) is 1,950,001. 

The terminal state is defined by [term_var, term_var, term_var, term_var, term_var], with term_var = 1E5.

The ordering of the states follows the following routine:

	states = zeros(state_dim, num_states)
	istate = 1
	for inmac_r in 1:nmac_r_dim # namc_r
		for iv_int in 1:v_dim # v_int
			for iv_own in 1:v_dim # v_own
				for iphi in 1:phi_dim # ϕ
					for itheta in 1:theta_dim # θ
						for irho in 1:rho_dim # ρ
							states[rho_ind, istate] = discrete_rho[irho]
							states[theta_ind, istate] = discrete_theta[itheta]
							states[phi_ind, istate] = discrete_phi[iphi]
							states[v_own_ind, istate] = discrete_v[iv_own]
							states[v_int_ind, istate] = discrete_v[iv_int]
							states[nmac_r_ind, istate] = discrete_nmac_r[inmac_r]
							istate += 1
						end
					end
				end
			end
		end
	end
	states[:, end] = terminal_state

I.e., first traversing through rho, then theta, phi, v_own , v_int and finally nmac_r.
The terminal state is appended at the end.

The action space is defined by actions = [:SR, :WR, :KEEP, :WL, :SL, :COC], with the order being enforced.
The mapping to numerical turn rates is 
	:SR => -10.0, :WR => -5.0, :KEEP => 0.0, :WL => 5.0, :SL => 10.0, :COC => -1.0


