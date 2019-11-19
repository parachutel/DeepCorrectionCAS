push!(LOAD_PATH, "./")
using CAS_QMDP


jld_full_filename = "../policies/fine_maxdist_2km_colli_150m_pen_action_5.0_pen_conflict_50.0_pen_closeness_0.1.jld"
alphas = load_alphas(jld_full_filename)
csv_filename = "fine_maxdist_2km_colli_150m_pen_action_5.0_pen_conflict_50.0_pen_closeness_0.1"
write_alphas_to_csv(alphas, csv_filename)