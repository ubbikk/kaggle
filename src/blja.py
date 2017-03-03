fp = '../trash/std_for_mngr_id.json'

def do_work():
    with open(fp) as f:
        lines = f.readlines()
        lines = [l[l.index(':')+1: l.index(',')] for l in lines]
        return [float(l) for l in lines]