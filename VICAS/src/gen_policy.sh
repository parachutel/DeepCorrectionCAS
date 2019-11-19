/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia \
	gen_policy.jl \
	--discount 0.95 \
	--penalty_action 3.0 \
	--penalty_conflict 50.0 \
	--penalty_closeness 500.0 \
	--penalty_nmac 7000.0 \
	--filename debug_mid3
	

