import distribution as d

def boundary_test(p, boundaries, slack=0):
    Sn = 0
    n = 0

    final_time = boundaries[-1].n
    for bdy in boundaries:
        # number of samples to draw:
        m = bdy.n - n
        X = d.Binomial(m, p).sample()
        Sn += X
        n += m
        if Sn >= bdy.accept:
            return 1, n, Sn
        if Sn <= bdy.reject - slack or n > final_time:
            return 0, n, Sn


