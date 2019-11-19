push!(LOAD_PATH, "./")

using CASIM
using PyPlot
using PyCall
@pyimport matplotlib.animation as animation

rc("font", family="Times New Roman")
rc("font", size=16)
rc("text", usetex=true)

# Settings:
sim_horizon = 1000
init_num_ac = 30
TAKEOFF_RATE = 50
spawn_rate = TAKEOFF_RATE * 100 / 3600
airspace_controller = Constant_Spawn_Rate_Controller(init_num_ac, spawn_rate)


# cas = Corrected_VICAS()
# policy.env.correction_weight = 0.075
# policy_name = "CorrectedClosest"
# global correction = true
# global factor = 1.0

# cas = VICAS()
# global correction = false
# policy_name = "VICASMulti"
# global factor = 1.0

# cas = No_CAS()
# global correction = false
# policy_name = "NOCAS"
# global factor = 1.0

cas = VICAS()
global correction = false
policy_name = "VICASClosest"
global factor = 1.0


cas_ctrl = Uniform_CAS(cas)
# sensor = Multi_Threat_Tracker()
sensor = Nearest_Intruder_Tracker()
sensor_ctrl = Uniform_Sensor(sensor)


fig = figure("SimAnim", figsize=(15, 8))


if correction_type == :closest 
    if airspace_controller.spawn_rate > 4900./3600
        policy.env.correction = true
        println("Correction turned on")
    else 
        policy.env.correction = false
        println("Correction turned off")
    end
elseif correction_type == :sector  
    if airspace_controller.spawn_rate > 7900./3600
        policy.env.correction = true
        println("Correction turned on")
    else 
        policy.env.correction = false
        println("Correction turned off")
    end
end


T = time()
global stats = Stats(T)
global t = 0
global All_AC = airspace_initialization(stats, airspace_controller.ac_number, cas_ctrl, sensor_ctrl, t)

function ft2m(ft::Float64)
    return ft * 0.3048
end

function animate(i)
    global All_AC
    global t
    global stats
    global correction
    global factor
    # Plot:
    clf()
    ax1 = subplot2grid((2, 2), (0, 0), rowspan=3)
    for ac in All_AC
        ax1 = scatter(ft2m(ac.dyn.x), ft2m(ac.dyn.y), marker="o", color="blue", s=6)
        ax1 = scatter(ft2m(ac.x_dest), ft2m(ac.y_dest), marker=",", color="magenta", s=3)
        ax1 = plot([ft2m(ac.dyn.x), ft2m(ac.x_dest)], [ft2m(ac.dyn.y), ft2m(ac.y_dest)], linestyle="--", color="black", linewidth=0.15)
    end
    ax1 = xlabel("\$x\$ (m)")
    ax1 = ylabel("\$y\$ (m)")
    ax1 = title(policy_name * ", Take-off Rate = " * string(TAKEOFF_RATE) * " flight/km\$^2\$-hour", fontsize=18)
    ax = gca()
    ax[:set_xlim]([0, 10000])
    ax[:set_ylim]([0, 10000])
    ax[:set_aspect]("equal")

    # Step:
    if t % 100 == 0
        println(string(t) * " : " * string(length(All_AC)))
    end
    update_all_dynamics(All_AC, correction=correction)
    t += 1
    run_stats(All_AC, stats, t)
    All_AC = airspace_control(stats, All_AC, airspace_controller, cas_ctrl, sensor_ctrl, t)


    ax2 = subplot2grid((2, 2), (0, 1))
    ax2 = plot(1 : length(stats.num_ac), stats.num_ac, color="red")
    ax2 = xlabel("Time (sec)")
    ax2 = ylabel("Number of Aircraft")
    ax = gca()
    ax[:set_xlim]([0, 1000])
    ax[:set_ylim]([0, 1000])

    ax3 = subplot2grid((2, 2), (1, 1))
    ax3 = plot(1 : length(stats.num_NMAC_t), stats.num_NMAC_t, color="red")
    println(stats.num_NMAC_t[end])
    ax3 = xlabel("Time (sec)")
    ax3 = ylabel("NMAC/sec")
    ax = gca()
    ax[:set_xlim]([0, 1000])
    ax[:set_ylim]([0, 25])

    frame = (ax1, ax2, ax3)
    

    return frame
end

sim_anim = animation.FuncAnimation(fig, animate, frames=sim_horizon, interval=200)
sim_anim[:save]("test" * policy_name * "_" * string(Int(spawn_rate * 3600)) * ".mp4", fps=24, extra_args=["-vcodec", "libx264"])
