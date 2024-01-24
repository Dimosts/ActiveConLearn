import random
import time

from main import *


def remove_scope_from_bias(B, scope):  # remove constraint network C from B

    removing = [c for c in B if set(get_scope(c)) == set(scope)]

    prev_B_length = len(B)
    B = [c for c in B if not (set(get_scope(c)) == set(scope))]

    if len(B) == prev_B_length:
        print(B)
        print(scope)
        raise Exception("Removing constraints from Bias did not result in reducing its size")

    return B

def construct_random_problem(Nvars, max_domain, Ncons, gamma):

    grid = intvar(1, max_domain, shape=(1, Nvars), name="grid")

    B = construct_bias(list(grid.flatten()), gamma)

    model = Model()

    for con in range(Ncons):

        found = False

        while not found:
            r = random.randint(0,len(B)-1)

            model2 = model.copy()

            model2 += B[r]

            if model2.solve():
                model += B[r]
                print(f"Adding {B[r]} to the model ")
                B = remove_scope_from_bias(B, get_scope(B[r]))
                found = True
            else:
                print(f"skipping {B[r]}, makes it UNSAT")
                B.pop(r)

    C_T = list(model.constraints)

    print(len(C_T))

    return grid, C_T, model


gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]
grid, C_T, model = construct_random_problem(100,5,200,gamma)



print(grid)
print(C_T)
print(model)

if model.solve():
    print("has solution")
else:
    print("UNSAT")

start = time.process_time()
#ca = Conacq(gamma, grid, C_T, qg=2, time_limit=1, findscope_version=2, findc_version=1, qgen_blimit=5000)
#ca.mquacq2(analyzeNlearn=True)
#save_results(alg="mquacq2-a", qg=2, tl=1, blimit=5000, fs=2, bench='new-random', start_time=start, conacq=ca)



with open('grid.pickle', 'wb') as handle:
    pickle.dump(grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
