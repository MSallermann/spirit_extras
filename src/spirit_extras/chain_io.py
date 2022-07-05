def chain_write_between(p_state, filename, idx_start, idx_stop, fileformat=None):
    """Writes the chain between idx_start and idx_stop to a file. Includes the endpoints!"""
    from spirit import io, chain

    if fileformat is None:
        fileformat = io.FILEFORMAT_OVF_TEXT

    noi = chain.get_noi(p_state)

    if(idx_start > idx_stop or idx_start < 0 or idx_stop>= noi):
        raise Exception("Error in idx_start and/or idx_stop")

    io.image_write(p_state, filename, idx_image = idx_start, fileformat = fileformat)
    for i in range(idx_start+1, idx_stop+1):
        io.image_append(p_state, filename, idx_image = i, fileformat = fileformat)


def chain_write_split_at(p_state, filename_list, idx_list, fileformat=None):
    """Writes a chain split at each index in idx_list"""
    from spirit import io, chain

    if fileformat is None:
        fileformat = io.FILEFORMAT_OVF_TEXT

    if(len(filename_list) != len(idx_list)-1):
        raise Exception("Length of filename list ({}) and length of idx_list ({}) not compatible".format(len(filename_list), len(idx_list)))

    for i in range(1,len(idx_list)):
        if(idx_list[i-i] > idx_list[i]):
            raise Exception("idx_list not monotonously increasing")

    for i,f in enumerate(filename_list):
        chain_write_between(p_state, f, idx_start=idx_list[i], idx_stop=idx_list[i+1], fileformat=fileformat)


def chain_append_from_file(p_state, filename):
    # TODO: chain_read with insert_idx seems to be broken
    raise NotImplementedError()
    from spirit import io, chain
    noi_file = io.n_images_in_file(p_state, filename)
    noi = chain.get_noi(p_state)
    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, noi + noi_file)
    io.chain_read(p_state, filename, insert_idx=noi)
    chain.update_data(p_state)


def chain_append_to_file_from_file(p_state, filename_out, filename_in, fileformat=None):
    """Reads the chain from `filename_in` and appends it to `filename_out`"""
    from spirit import io, chain

    if fileformat is None:
        fileformat = io.FILEFORMAT_OVF_TEXT

    chain.image_to_clipboard(p_state)

    noi_file = io.n_images_in_file(p_state, filename_in)
    chain.set_length(p_state, noi_file)
    io.chain_read(p_state, filename_in)

    # TODO: io.chain_append seems to be broken, so we use image_append...
    for i in range(noi_file):
        io.image_append(p_state, filename_out, idx_image=i, fileformat=fileformat)


def swap_images(p_state, idx1, idx2):
    """Swaps the place of two images in the chain"""
    from spirit import chain
    chain.image_to_clipboard(p_state, idx1) # idx1 to clipboard
    chain.insert_image_after(p_state, idx2) # insert after 2
    chain.image_to_clipboard(p_state, idx2) # idx2 to clipboard
    chain.replace_image(p_state, idx1) # replace idx1 with idx2 from clipboard
    chain.delete_image(p_state, idx2)


def invert_chain(p_state, idx_start=0, idx_end=-1):
    """Inverts the chain between idx_start and idx_end. 'idx_end' < 0 is equivalent to idx_end=noi-1"""
    from spirit import chain

    if idx_end<0:
        idx_end = chain.get_noi(p_state) - 1

    idx_stop_swap = int( idx_start + (idx_end - idx_start - 1) / 2 )
    for img in range(idx_start, idx_stop_swap+1):
        swap_images(p_state, img, idx_end - idx_start - img)