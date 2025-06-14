def max_chain_length(chains):
    # put the chains in a dictionary
    chains_dict = {}
    for chain in chains:
        if chain[0] not in chains_dict:
            chains_dict[chain[0]] = chain[1]
        else:
            if chain[1] > chains_dict[chain[0]]:
                chains_dict[chain[0]] = chain[1]
    # find the longest chain
    max_len = 0
    for key, value in chains_dict.items():