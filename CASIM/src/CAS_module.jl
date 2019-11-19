module CAS_module 

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "../")
push!(LOAD_PATH, "../ACASXU_Dependencies")
push!(LOAD_PATH, "../../VICAS/src")
push!(LOAD_PATH, "../../DeepCorrection")
push!(LOAD_PATH, "../../DeepCorrection/src")

using Simulation_Env_module
using JLD, CAS_QMDP, DeepQLearning

export CAS
export ACAS_Xu_Network, Simple_CAS, No_CAS, VICAS, Corrected_VICAS, VICASWeighted # NNCAS
export CAS_DICT
export acas_xu_network_data, evaluate_network
export alphas, get_qvals, norm_angle
export policy, STATE_DIM, AUG_STATE_DIM, TERM_VAR
export correction_version, VICAS_policyname, correction_type, correction_weight

abstract type CAS end


include("../ACASXU_Dependencies/load_network.jl")
const nnfile_name = "../ACASXU_Dependencies/ACASXU_TF12_run3_DS_Minus15_Online_08172017_200Epochs_largerRange.nnet"
const acas_xu_network_data = NNet(nnfile_name)

mutable struct ACAS_Xu_Network <: CAS
	Ind_to_Advisory_dict::Dict{Int64, Symbol}
	Advisory_to_Ind_dict::Dict{Symbol, Int64}
	Advisory_to_Action_dict::Dict{Symbol, Float64}

	function ACAS_Xu_Network()
		this = new()
		this.Ind_to_Advisory_dict = 
				Dict(1 => :COC, 2 => :WL, 3 => :WR, 4 => :SL, 5 => :SR)
		this.Advisory_to_Ind_dict = 
				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
		this.Advisory_to_Action_dict = 
				Dict(:COC => 0., :WL => 5., :WR => -5., :SL => 10., :SR => -10.)
		return this
	end
end


# const VICAS_policyname = "test_mid3"
const VICAS_policyname = "mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0"
const alphas_filename = 
	"../../VICAS/policies/" * VICAS_policyname * ".jld" # requires disc = mid3
const alphas = load_alphas(alphas_filename)

mutable struct VICAS <: CAS
	Ind_to_Advisory_dict::Dict{Int64, Symbol}
	Advisory_to_Ind_dict::Dict{Symbol, Int64}
	Advisory_to_Action_dict::Dict{Symbol, Float64}

	function VICAS()
		this = new()
		this.Ind_to_Advisory_dict = 
				Dict(1 => :SR, 2 => :WR, 3 => :KEEP, 4 => :WL, 5 => :SL, 6 => :COC)
		this.Advisory_to_Ind_dict = 
				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
		this.Advisory_to_Action_dict = 
				Dict(:COC => -1.0, :WL => 5., :WR => -5., :SL => 10., :SR => -10., :KEEP => 0.0)
		return this
	end
end

mutable struct VICASWeighted <: CAS
	Ind_to_Advisory_dict::Dict{Int64, Symbol}
	Advisory_to_Ind_dict::Dict{Symbol, Int64}
	Advisory_to_Action_dict::Dict{Symbol, Float64}

	function VICASWeighted()
		this = new()
		this.Ind_to_Advisory_dict = 
				Dict(1 => :SR, 2 => :WR, 3 => :KEEP, 4 => :WL, 5 => :SL, 6 => :COC)
		this.Advisory_to_Ind_dict = 
				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
		this.Advisory_to_Action_dict = 
				Dict(:COC => -1.0, :WL => 5., :WR => -5., :SL => 10., :SR => -10., :KEEP => 0.0)
		return this
	end
end

# const correction_type = :sector
# const correction_version = "LOWFI_mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0_2018-11-13T18\:55\:43.491" # best sector
# const correction_version = "LOWFI_mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0_2018-11-15T22\:38\:58.856"
# const correction_version = "LOWFI_mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0_2018-11-28T10\:55\:10.697"
const correction_type = :closest
const correction_version = "LOWFI_mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0_2018-11-28T10\:42\:31.084" # best closest
# const correction_version = "LOWFI_mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0_2018-11-29T22\:18\:30.068" 
# requires disc = mid3
const problem_file = "../../DeepCorrection/logs/" * correction_version * "/final_problem.jld"
const weights_file = "../../DeepCorrection/logs/" * correction_version * "/final_weights.jld"
policy = restore(problem_file=problem_file, weights_file=weights_file)
correction_weight = policy.env.correction_weight

mutable struct Corrected_VICAS <: CAS
	Ind_to_Advisory_dict::Dict{Int64, Symbol}
	Advisory_to_Ind_dict::Dict{Symbol, Int64}
	Advisory_to_Action_dict::Dict{Symbol, Float64}

	function Corrected_VICAS()
		this = new()
		this.Ind_to_Advisory_dict = 
				Dict(1 => :SR, 2 => :WR, 3 => :KEEP, 4 => :WL, 5 => :SL, 6 => :COC)
		this.Advisory_to_Ind_dict = 
				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
		this.Advisory_to_Action_dict = 
				Dict(:COC => -1.0, :WL => 5., :WR => -5., :SL => 10., :SR => -10., :KEEP => 0.0)
		return this
	end
end



# const nn_version = ""
# const nn_problem_file = "../../DeepCorrection/logs/" * nn_version * "/final_problem.jld"
# const nn_weights_file = "../../DeepCorrection/logs/" * nn_version * "/final_weights.jld"
# nn_policy = restore(problem_file=nn_problem_file, weights_file=nn_weights_file)
# nn_policy.env.correction = true

# mutable struct NNCAS <: CAS
# 	Ind_to_Advisory_dict::Dict{Int64, Symbol}
# 	Advisory_to_Ind_dict::Dict{Symbol, Int64}
# 	Advisory_to_Action_dict::Dict{Symbol, Float64}

# 	function NNCAS()
# 		this = new()
# 		this.Ind_to_Advisory_dict = 
# 				Dict(1 => :SR, 2 => :WR, 3 => :KEEP, 4 => :WL, 5 => :SL, 6 => :COC)
# 		this.Advisory_to_Ind_dict = 
# 				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
# 		this.Advisory_to_Action_dict = 
# 				Dict(:COC => -1.0, :WL => 5., :WR => -5., :SL => 10., :SR => -10., :KEEP => 0.0)
# 		return this
# 	end
# end


"""
	outdated
"""
mutable struct Simple_CAS <: CAS
	alert_threshold::Int64
	Ind_to_Advisory_dict::Dict{Int64, Symbol}
	Advisory_to_Ind_dict::Dict{Symbol, Int64}
	Advisory_to_Action_dict::Dict{Symbol, Float64}

	function Simple_CAS()
		this = new()
		set_alert_threshold = 1320. * 0.6 # [ft] ≈ 400 [m]
		this.alert_threshold = set_alert_threshold
		this.Ind_to_Advisory_dict = 
				Dict(1 => :COC, 2 => :WL, 3 => :WR, 4 => :SL, 5 => :SR)
		this.Advisory_to_Ind_dict = 
				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
		this.Advisory_to_Action_dict = 
				Dict(:COC => 0., :WL => 5., :WR => -5., :SL => 10., :SR => -10.)
		return this
	end
end

mutable struct No_CAS <: CAS
	dt::Float64
	Ind_to_Advisory_dict::Dict{Int64, Symbol}
	Advisory_to_Ind_dict::Dict{Symbol, Int64}
	Advisory_to_Action_dict::Dict{Symbol, Float64}

	function No_CAS()
		this = new()
		this.Ind_to_Advisory_dict = 
				Dict(1 => :COC, 2 => :WL, 3 => :WR, 4 => :SL, 5 => :SR)
		this.Advisory_to_Ind_dict = 
				Dict(value => key for (key, value) in this.Ind_to_Advisory_dict)
		this.Advisory_to_Action_dict = 
				Dict(:COC => 0., :WL => 5., :WR => -5., :SL => 10., :SR => -10.)
		return this
	end
end

# construct all CAS' in advance to avoid repetitive contruction
CAS_DICT = Dict{DataType, CAS}()
for i = 1:length(subtypes(CAS))
	CAS_DICT[subtypes(CAS)[i]] = subtypes(CAS)[i]()
end

println("All CAS loaded!")

end # module
