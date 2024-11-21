import sympy as sp
import numpy as np
def ophobning(f, varsAndVals, show=True):
        f = sp.sympify(f)

        gradient = [f.diff(v) for v in varsAndVals]
        symbols = [[str(i) for i in g.free_symbols] for g in gradient]
        bidrag = [(varsAndVals[v][1]*g.subs([(i, varsAndVals[i][0]) for i in s]))**2 for v, g, s in zip(varsAndVals, gradient, symbols)]
        bidragflt = [float(b) for b in bidrag]

        usikkerhed = np.sqrt(np.sum(bidragflt))

        if show==True:
                print('Bidrag:')
                for var, val in zip(varsAndVals, bidragflt):
                        print(var,':',val)

                print('\nUsikkerhed:', usikkerhed)

        return usikkerhed
