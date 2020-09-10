
from collections import Counter
from collections import defaultdict
import  pprint as pp


def data():
    N = ["S","NP","VP","PP","D","N","V","P"]
    x = ('the boy saw the man with a telescope').split()
    R = [("S",("NP","VP")),("NP",("D","N")),("VP",("V","NP")),("NP",("NP","PP")),("PP",("P","NP")),("VP",("VP","PP"))]
    N_Count = Counter({'S': 2, 'NP': 7,'VP': 3,'PP': 2,'D': 6,'N': 6,'V': 2,'P': 2})
    r2 = Counter({("D","the") : 4,("N","boy") : 2,("V","saw") : 2,("N","man") : 2,("P","with") : 2,("D","a") : 2,("N","telescope") : 2 })
    potential = Counter({("S",("NP","VP")):1,("NP",("D","N")):0.857,("VP",("V","NP")):0.67,("NP",("NP","PP")):0.143,("PP",("P","NP")):1,("VP",("VP","PP")):0.33,("D","the") : 0.67,("N","boy") : 0.33,("V","saw") : 1,("N","man") : 0.33,("P","with") : 1,("D","a") : 0.33,("N","telescope") : 0.33 })

    return N, x, R, potential

def main():
    N, X, R, potential = data()
    n = len(X)

    # Inside algorithm:
    alpha = defaultdict(dict)

    # Base case
    for _N in N:
        alpha[_N] = {}
        for i in range(n):
            x = X[i]
            alpha[_N][(i,i)] = potential[(_N,x)]

    # Recursive term
    for _N in N:
        for l in range(2,n+1):
            for i in range(n-l+1):
                j = i + l - 1
                sum = 0
                for rule in R:
                    if rule[0] == _N:
                        # A = rule[0]
                        B = rule[1][0]
                        C = rule[1][1]
                        for k in range(i,j):
                            if (k+1,j) not in alpha[C]:
                                alpha[C][(k + 1,j)] = 0
                            if (i,k) not in alpha[B]:
                                alpha[B][(i,k)] = 0
                            sum += potential[rule]*alpha[B][(i,k)]*alpha[C][(k+1,j)]
                alpha[_N][(i,j)] = sum

    print("Alpha values:")
    pp.pprint(alpha)
    print("*******************************************************************")


    # Outside Algorithm:
    beta = {}

    # Base case
    for _N in N:
        beta[_N] = {}
        beta[_N][(0,n-1)] = 0
    beta["S"][(0,n-1)] = 1

    #  Recursive term
    for _N in N:
        for j in range(n-1,-1,-1):
            for i in range(j+1):
                if (i,j) != (0,n-1):
                    sum = 0
                    for rule in R:
                        B = rule[0]
                        if _N == rule[1][1]:
                            C = rule[1][0]
                            for k in range(i):
                                if (k,i-1) not in alpha[C]:
                                    alpha[C][(k,i-1)] = 0
                                if (k,j) not in alpha[B]:
                                    alpha[B][(k,j)] = 0
                                sum += potential[rule]*alpha[C][(k,i-1)]*alpha[B][(k,j)]
                        if _N == rule[1][0]:
                            # A = rule[1][0]
                            C = rule[1][1]
                            for k in range(j+1,n):
                                if (j+1,k) not in alpha[C]:
                                    alpha[C][(j+1,k)] = 0
                                if (i,k) not in alpha[B]:
                                    alpha[B][(i,k)] = 0
                                sum += potential[rule] * alpha[C][(j+1,k)] * alpha[B][(i,k)]
                    beta[_N][(i,j)] = sum

    print("Beta values:")
    pp.pprint(beta)
    print("*******************************************************************")


    u = defaultdict(dict)
    for _N in N:
        for i in range(n):
            for j in range(i,n):
                u[_N][(i,j)] = alpha[_N][(i,j)]*beta[_N][(i,j)]

    print("U values:")
    pp.pprint(u)
    print("*******************************************************************")



main()



