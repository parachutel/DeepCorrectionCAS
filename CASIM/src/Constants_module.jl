module Constants_module

export min_speed, max_speed, dt
export sensing_range, airspace_dim, dest_range, collision_range

const dt = 1. # sec
const min_speed = 163. # ft/s, 50 m/s
const max_speed = 163. # ft/s, 50 m/s
const sensing_range = 3280.8 # ft, 1 km
const airspace_dim = 32808.4 # ft, 10 km (65616.8 ft = 20 km)
const dest_range = 492. # ft, 150 m
const collision_range = 492. # ft, 150 m

end # module