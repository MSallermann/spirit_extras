def chain_write_between(p_state, filename, idx_start, idx_stop, fileformat=None):
    from spirit import io, chain

    if not fileformat:
        fileformat=io.FILEFORMAT_OVF_TEXT

    noi = chain.get_noi(p_state)

    if(idx_start > idx_stop or idx_start < 0 or idx_stop>= noi):
        raise Exception("Error in idx_start and/or idx_stop")

    io.image_write(p_state, filename, idx_image = idx_start, fileformat=fileformat)
    for i in range(idx_start+1, idx_stop+1):
        io.image_append(p_state, filename, idx_image = i, fileformat=fileformat)


def chain_write_split_at(p_state, filename_list, idx_list, fileformat=None):
    from spirit import io, chain

    if not fileformat:
        fileformat=io.FILEFORMAT_OVF_TEXT

    if(len(filename_list) != len(idx_list)-1):
        raise Exception("Length of filename list ({}) and length of idx_list ({}) not compatible".format(len(filename_list), len(idx_list)))

    for i in range(1,len(idx_list)):
        if(idx_list[i-i] > idx_list[i] ):
            raise Exception("idx_list not monotonously increasing")

    for i,f in enumerate(filename_list):
        chain_write_between(p_state, f, idx_start=idx_list[i], idx_stop=idx_list[i+1], fileformat=fileformat)