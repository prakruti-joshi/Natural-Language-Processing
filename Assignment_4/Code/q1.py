import pprint as pp
from collections import Counter


def counts(o,s,t,x,y):
    l_x = len(x)
    for j in range(l_x):
        x_tag_pair = (y[j + 1], x[j])

        # Updating x tag pair count
        if x_tag_pair in o:
            o[x_tag_pair] += 1
        else:
            o[x_tag_pair] = 1

    l_y = len(y)
    for j in range(0, l_y - 1):
        # Updating state count
        if y[j] in s:
            s[y[j]] += 1
        else:
            s[y[j]] = 1

        y_pair = (y[j + 1], y[j])
        # Updating state pair count
        if y_pair in t:
            t[y_pair] += 1
        else:
            t[y_pair] = 1

    return o,s,t

def estimate_mle(o_mle, t_mle, o,s,t, x, y):
    l_x = len(x)
    for j in range(l_x):
        x_tag_count = o[(y[j + 1], x[j])]
        s_count = s[y[j + 1]]
        o_mle[(y[j + 1], x[j])] = float(x_tag_count)/float(s_count)

    l_y = len(y)
    for j in range(0, l_y - 1):
        y_pair = (y[j + 1], y[j])
        t_mle[y_pair] = t[y_pair]/s[y[j]]

    return o_mle, t_mle

def calculate_potential(o_mle,t_mle, s_prev, s, j, x):
    if (s,s_prev) not in t_mle:
        return 0
    t_val =  t_mle[(s,s_prev)]
    if (s,x[j-1]) not in o_mle:
        return 0
    e_val = o_mle[(s,x[j-1])]
    return t_val * e_val


def forward(o_mle,t_mle, state_dict, x):
    S = list(state_dict.keys())
    S.remove("*")
    m = len(x)
    alpha = [Counter() for _ in range(m+1)]

    s_init = "*"
    for s in S:
        alpha[1][s] = calculate_potential(o_mle,t_mle,s_init,s,1,x)

    for j in range(2,m+1):
        for s in S:
            a_sum = 0
            for _s in S:
                a_sum += ( alpha[j-1][_s] * calculate_potential(o_mle,t_mle,_s,s,j,x) )
            alpha[j][s] = a_sum
    return alpha

def backward(o_mle,t_mle, state_dict, x):
    S = list(state_dict.keys())
    S.remove("*")
    m = len(x)
    beta = [Counter() for _ in range(m + 1)]

    s_end = "STOP"
    for s in S:
        if (s_end,s) not in t_mle:
            beta[m][s] = 0
        else:
            beta[m][s] = t_mle[(s_end,s)]

    for j in range(m-1,0,-1):
        for s in S:
            b_sum = 0
            for _s in S:
                b_sum += (beta[j + 1][_s] * calculate_potential(o_mle, t_mle, s, _s, j+1, x))
            beta[j][s] = b_sum
    return beta

def forward_backward_algo(alpha,beta,state_dict,m):
    S = list(state_dict.keys())
    S.remove("*")
    s_len = len(S)
    u = [Counter() for _ in range(m + 1)]

    for j in range(m+1):
        for s in S:
            u[j][s] = (alpha[j][s] * beta[j][s])
    return  u

def main():

    x1 = ('the man saw the cut').split()
    x2 = ('the saw cut the man').split()
    x3 = ('the saw').split()

    y1 = ('* D N V D N STOP').split()
    y2 = ('* D N V D N STOP').split()
    y3 = ('* N N STOP').split()

    X = [x1,x2,x3]
    Y = [y1,y2,y3]
    training_exp = len(X)
    assert (len(X) == len(Y))

    t = {}      # dict containing count of (y_j, y_j-1)
    o = {}      # dict containing count of (s->x)
    s = {}      # dict of state counts

    o_mle = {}
    t_mle = {}

    for i in range(training_exp):
        x = X[i]
        y = Y[i]

        o,s,t = counts(o,s,t,x,y)

    for i in range(training_exp):
        x = X[i]
        y = Y[i]
        o_mle, t_mle = estimate_mle(o_mle, t_mle, o,s,t, x, y)

    print("Non-zero MLE estimates of o:")
    pp.pprint(o_mle)
    print("**********************************************************")
    print("Non-zero MLE estimates of t:")
    pp.pprint(t_mle)
    print("**********************************************************")

    print("Conditioning on x2:")
    m = len(x2)
    alpha = forward(o_mle,t_mle, s, x2)
    beta = backward(o_mle,t_mle, s, x2)
    u = forward_backward_algo(alpha,beta,s,m)

    print("Forward backward probabilities")
    pp.pprint(u)
    print("**********************************************************")

    print(u[3]["V"])

    ################################

    print("Conditioning on x1:")
    m = len(x1)
    alpha = forward(o_mle,t_mle, s, x1)
    beta = backward(o_mle,t_mle, s, x1)
    u = forward_backward_algo(alpha,beta,s,m)

    print("Forward backward probabilities")
    pp.pprint(u)
    print("**********************************************************")

    print(u[5]["N"])



main()

















