def contain_star(s):
    for c in s:
        if c=='*':
            return True
    return False

def contains_star_l(l):
    for s in l:
        if contain_star(s):
            return True

    return False