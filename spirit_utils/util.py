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

def find_saddle_point_only():
    pass

def optimize_skyrmion_radius():
    pass