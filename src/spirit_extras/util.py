def set_output_folder(p_state, folder, tag="<time>"):
    from spirit.parameters import ema, gneb, llg, mc, mmf
    from spirit import log
    import os

    if not os.path.exists(folder):
        os.makedirs(folder)

    log.set_output_folder(p_state, folder)
    log.set_output_file_tag(p_state, tag)

    # ema missing

    llg.set_output_folder(p_state, folder)
    llg.set_output_tag(p_state, tag)

    gneb.set_output_folder(p_state, folder)
    gneb.set_output_tag(p_state, tag)

    mc.set_output_folder(p_state, folder)
    mc.set_output_tag(p_state, tag)

    mmf.set_output_folder(p_state, folder)
    mmf.set_output_tag(p_state, tag)


def estimate_mc_runtime(p_state, total_iterations, n_iterations_test=500):
    """Estimates the total run time of 'total iterations' monte carlo steps by measuring `n_iterations_test` monte carlo steps"""
    from spirit import simulation
    from spirit import parameters

    parameters.mc.set_iterations(p_state, n_iterations_test, n_iterations_test) # We want n_iterations iterations and only a single log message
    info = simulation.start(p_state, simulation.METHOD_MC) # Start a MC simulation

    # total_iterations = len(sample_temperatures) * (n_samples*n_decorrelation + n_thermalisation)
    runtime_seconds = total_iterations / info.total_ips

    hours   = int( (runtime_seconds / (60**2)) )
    minutes = int( (runtime_seconds - hours * 60**2) / 60 )
    seconds = int( (runtime_seconds - hours * 60**2 - minutes*60) )

    print("Estimated runtime = {:.0f}h:{:.0f}m:{:.0f}s".format( hours, minutes, seconds ) )
    print("Total iterations  = {:.0f}".format( total_iterations ) )
    print("IPS               = {:.0f}".format( info.total_ips ) )

    return runtime_seconds

def find_saddle_point_only():
    pass

def optimize_skyrmion_radius():
    pass