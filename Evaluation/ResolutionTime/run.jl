include("resTime.jl")

resTime(spawnRateSet=30,
        CASType=:correctedSector,
        policyFile="LOWFI_mid3_maxdist_2km_nmac_150m_Discount_0.95_ActWeight_0.5_IfSample_false_PenAction_3.0_PenConflict_50.0_PenCloseness_500.0_PenNmac_7000.0_Sigma_v_5.0_alert_5.0_theta_3.0_phi_7.0_coc_3.0_2018-11-13T18\:55\:43.491",
        numEvalEposide=1,
        maxEpisodeLen=1000,
        render=false)