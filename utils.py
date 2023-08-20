import math
import time

from cpmpy import intvar, boolvar, Model, all, sum, SolverLookup
from sklearn.utils import class_weight
import numpy as np

import cpmpy
import re
from cpmpy.expressions.utils import all_pairs
from itertools import chain

from ConAcq import SOLVER


def check_value(c):
    return bool(c.value())


def get_con_subset(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y)]


def get_kappa(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is False]


def get_lambda(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is True]


def gen_pairwise(v1, v2):
    return [v1 == v2, v1 != v2, v1 < v2, v1 > v2]


# to create the binary oracle
def gen_pairwise_ineq(v1, v2):
    return [v1 != v2]


def alldiff_binary(grid):
    for v1, v2 in all_pairs(grid):
        for c in gen_pairwise_ineq(v1, v2):
            yield c


def gen_scoped_cons(grid):
    # rows
    for row in grid:
        for v1, v2 in all_pairs(row):
            for c in gen_pairwise_ineq(v1, v2):
                yield c
    # columns
    for col in grid.T:
        for v1, v2 in all_pairs(col):
            for c in gen_pairwise_ineq(v1, v2):
                yield c

    # DT: Some constraints are not added here, I will check it and fix  TODO
    # subsquares
    for i1 in range(0, 4, 2):
        for i2 in range(i1, i1 + 2):
            for j1 in range(0, 4, 2):
                for j2 in range(j1, j1 + 2):
                    if (i1 != i2 or j1 != j2):
                        for c in gen_pairwise_ineq(grid[i1, j1], grid[i2, j2]):
                            yield c


def gen_all_cons(grid):
    # all pairs...
    for v1, v2 in all_pairs(grid.flat):
        for c in gen_pairwise(v1, v2):
            yield c


def construct_bias(X, gamma):
    all_cons = []

    X = list(X)

    for relation in gamma:

        if relation.count("var") == 2:

            for v1, v2 in all_pairs(X):
                constraint = relation.replace("var1", "v1")
                constraint = constraint.replace("var2", "v2")
                constraint = eval(constraint)

                all_cons.append(constraint)

        elif relation.count("var") == 4:

            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    for x in range(j + 1, len(X) - 1):
                        for y in range(x + 1, len(X)):
                            if (y != i and x != j and x != i and y != j):
                                #            for v1, v2 in all_pairs(X):
                                #                for v3, v4 in all_pairs(X):
                                constraint = relation.replace("var1", "X[i]")
                                constraint = constraint.replace("var2", "X[j]")
                                constraint = constraint.replace("var3", "X[x]")
                                constraint = constraint.replace("var4", "X[y]")
                                constraint = eval(constraint)

                                all_cons.append(constraint)

    return all_cons


def construct_bias_for_var(X, gamma, v1):
    all_cons = []

    for relation in gamma:
        if relation.count("var") == 2:
            for v2 in X:
                if not (v1 is v2):
                    constraint = relation.replace("var1", "v1")
                    constraint = constraint.replace("var2", "v2")
                    constraint = eval(constraint)

                    all_cons.append(constraint)

        elif relation.count("var") == 4:
            X = X.copy()
            X.reverse()
            print(X)
            for j in range(0, len(X)):
                for x in range(j + 1, len(X) - 1):
                    for y in range(x + 1, len(X)):
                        # if (y != i and x != j and x != i and y != j):
                        #            for v1, v2 in all_pairs(X):
                        #                for v3, v4 in all_pairs(X):
                        constraint = relation.replace("var1", "v1")
                        constraint = constraint.replace("var2", "X[j]")
                        constraint = constraint.replace("var3", "X[x]")
                        constraint = constraint.replace("var4", "X[y]")
                        constraint = eval(constraint)

                        all_cons.append(constraint)

    return all_cons


def get_scopes_vars(C):
    return set([x for scope in [get_scope(c) for c in C] for x in scope])


def get_scopes(C):
    return list(set([tuple(get_scope(c)) for c in C]))


def get_scope(constraint):
    # this code is much more dangerous/too few cases then get_variables()
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                # non-recursive shortcut
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def get_arity(constraint):
    return len(get_scope(constraint))


def get_min_arity(C):
    if len(C) > 0:
        return min([get_arity(c) for c in C])
    return 0


def get_max_arity(C):
    if len(C) > 0:
        return max([get_arity(c) for c in C])
    return 0


def get_relation(c, gamma):
    scope = get_scope(c)

    for i in range(len(gamma)):
        relation = gamma[i]

        if relation.count("var") != len(scope):
            continue

        constraint = relation.replace("var1", "scope[0]")
        for j in range(1, len(scope)):
            constraint = constraint.replace("var" + str(j + 1), "scope[" + str(j) + "]")

        constraint = eval(constraint)

        if hash(constraint) == hash(c):
            return i

    return -1


def get_var_name(var):
    name = re.findall("\[\d+[,\d+]*\]", var.name)
    name = var.name.replace(name[0], '')
    return name


def get_var_ndims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims


def get_var_dims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    dims = re.split("[\[\]]", dims_str)[1]
    dims = [int(dim) for dim in re.split(",", dims)]
    return dims


def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def join_con_net(C1, C2):
    C3 = [[c1 & c2 if c1 is not c2 else c1 for c2 in C2] for c1 in C1]
    C3 = list(chain.from_iterable(C3))
    C3 = remove_redundant_conj(C3)
    return C3


def remove_redundant_conj(C1):
    C2 = list()

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        flag_eq = False
        flag_neq = False
        flag_geq = False
        flag_leq = False
        flag_ge = False
        flag_le = False

        for c1 in conj_args:
            print(c1.name)
            # Tias is on 3.9, no 'match' please!
            if c1.name == "==":
                flag_eq = True
            elif c1.name == "!=":
                flag_neq = True
            elif c1.name == "<=":
                flag_leq = True
            elif c1.name == ">=":
                flag_geq = True
            elif c1.name == "<":
                flag_le = True
            elif c1.name == ">":
                flag_ge = True
            else:
                raise Exception("constraint name is not recognised")

            if not ((flag_eq and (flag_neq or flag_le or flag_ge)) or (
                    (flag_leq or flag_le) and ((flag_geq or flag_ge)))):
                C2.append(c)
    return C2


def get_max_conjunction_size(C1):
    max_conj_size = 0

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        max_conj_size = max(len(conj_args), max_conj_size)

    return max_conj_size


def get_delta_p(C1):
    max_conj_size = get_max_conjunction_size(C1)

    Delta_p = [[] for _ in range(max_conj_size)]

    for c in C1:

        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        Delta_p[len(conj_args) - 1].append(c)

    return Delta_p


def compute_sample_weights(Y):
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    sw = []

    for i in range(len(Y)):
        if Y[i] == False:
            sw.append(c_w[0])
        else:
            sw.append(c_w[1])

    return sw


class Metrics:

    def __init__(self):
        self.queries_count = 0
        self.top_lvl_queries = 0
        self.generated_queries = 0
        self.findscope_queries = 0
        self.findc_queries = 0

        self.average_size_queries = 0

        self.start_time_query = time.time()
        self.max_waiting_time = 0
        self.generation_time = 0

        self.converged = 1

    def increase_queries_count(self, amount=1):
        self.queries_count += amount

    def increase_top_queries(self, amount=1):
        self.top_lvl_queries += amount

    def increase_generated_queries(self, amount=1):
        self.generated_queries += amount

    def increase_findscope_queries(self, amount=1):
        self.findscope_queries += amount

    def increase_findc_queries(self, amount=1):
        self.findc_queries += amount

    def increase_generation_time(self, amount):
        self.generation_time += self.generation_time

    def increase_queries_size(self, amount):
        self.average_size_queries += 1

    def aggreagate_max_waiting_time(self, max2):
        if self.max_waiting_time < max2:
            self.max_waiting_time = max2

    def aggregate_convergence(self, converged2):
        if self.converged + converged2 < 2:
            self.converged = 0

    def __add__(self, other):

        new = self

        new.increase_queries_count(other.queries_count)
        new.increase_top_queries(other.top_lvl_queries)
        new.increase_generated_queries(other.generated_queries)
        new.increase_findscope_queries(other.findscope_queries)
        new.increase_findc_queries(other.findc_queries)
        new.increase_generation_time(other.generation_time)
        new.increase_queries_size(other.average_size_queries)

        new.aggreagate_max_waiting_time(other.max_waiting_time)
        new.aggregate_convergence(other.converged)

        return new


def find_suitable_vars_subset2(l, B, Y):
    if len(Y) <= get_min_arity(B) or len(B) < 1:
        return Y

    scope = get_scope(B[0])
    Y_prime = list(set(Y) - set(scope))

    l2 = int(l) - len(scope)

    if l2 > 0:
        Y1 = Y_prime[:l2]
    else:
        Y1 = []

    [Y1.append(y) for y in scope]

    return Y1


def generate_findc_query(L, delta):
    # constraints from  B are taken into account as soft constraints who we do not want to satisfy (i.e. we want to violate)
    # This is the version of query generation for the FindC function that was presented in "Constraint acquisition via Partial Queries", IJCAI 2013

    tmp = Model(L)

    objective = sum([c for c in delta])  # get the amount of satisfied constraints from B

    # at least 1 violated and at least 1 satisfied
    # we want this to assure that each answer of the user will reduce
    # the set of candidates
    # Difference with normal query generation: if all are violated, we already know that the example will
    # be a non-solution due to previous answers

    tmp += objective < len(delta)
    tmp += objective > 0

    # Try first without objective
    s = SolverLookup.get(SOLVER, tmp)
    flag = s.solve()

    if not flag:
        # UNSAT, stop here
        return flag

    Y = get_scope(delta[0])
    Y = list(dict.fromkeys(Y))  # remove duplicates

    # Next solve will change the values of the variables in the lY2
    # so we need to return them to the original ones to continue if we dont find a solution next
    values = [x.value() for x in Y]

    # so a solution was found, try to find a better one now
    s.solution_hint(Y, values)

    # run with the objective
    s.minimize(abs(objective - round(len(delta) / 2)))  # we want to try and do it like a dichotomic search
    # s.minimize(objective)  # we want to minimize the violated cons

    flag2 = s.solve(time_limit=0.2)

    if not flag2:
        tmp = Model()
        i = 0
        for x in Y:
            tmp += x == values[i]
            i = i + 1

        tmp.solve()

        return flag

    else:
        return flag2


def generate_findc2_query(L, delta):
    # This is the version of query generation for the FindC function that was presented in "Constraint acquisition through Partial Queries", AIJ 2023

    tmp = Model(L)

    max_conj_size = get_max_conjunction_size(delta)
    delta_p = get_delta_p(delta)

    p = intvar(0, max_conj_size)
    kappa_delta_p = intvar(0, len(delta), shape=(max_conj_size,))
    p_soft_con = boolvar(shape=(max_conj_size,))

    for i in range(max_conj_size):
        tmp += kappa_delta_p[i] == sum([c for c in delta_p[i]])
        p_soft_con[i] = (kappa_delta_p[i] > 0)

    tmp += p == min([i for i in range(max_conj_size) if (kappa_delta_p[i] < len(delta_p[i]))])

    objective = sum([c for c in delta])  # get the amount of satisfied constraints from B

    # at least 1 violated and at least 1 satisfied
    # we want this to assure that each answer of the user will reduce
    # the set of candidates
    tmp += objective < len(delta)
    tmp += objective > 0

    # Try first without objective
    s = SolverLookup.get(SOLVER, tmp)

    print("p: ", p)

    # run with the objective
    s.minimize(100 * p - p_soft_con[p])

    flag = s.solve()
    #        print("OPT solve", s.status())

    return flag