def match_num(string):
    # Check if the first character is '5'
    if string[0] == '5':
        # If it is, return the string
        return string
    else:
        # If it's not, remove the first character and return it
        return string[1:]