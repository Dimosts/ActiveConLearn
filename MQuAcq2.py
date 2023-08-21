import time

import numpy as np
from cpmpy.expressions.utils import all_pairs

from ConAcq import ConAcq
from utils import construct_bias, get_kappa, get_scope, get_relation

cliques_cutoff = 0.5

class MQuAcq2(ConAcq):
    def __init__(self, gamma, grid, ct=list(), bias=list(), X=set(), C_l=set(), qg="pqgen", obj="proba",
                 time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000, perform_analyzeAndLearn=False):
        super().__init__(gamma, grid, ct, bias, X, C_l, qg, obj, time_limit, findscope_version,
                    findc_version, tqgen_t, qgen_blimit)
        self.perform_analyzeAndLearn = perform_analyzeAndLearn

    def learn(self):

        answer = True

        if len(self.B) == 0:
            self.B = construct_bias(self.X, self.gamma)

        while True:
            if self.debug_mode:
                print("Size of CL: ", len(self.C_l.constraints))
                print("Size of B: ", len(self.B))
                print("Number of queries: ", self.metrics.queries_count)
                print("MQuAcq-2 Queries: ", self.metrics.top_lvl_queries)
                print("FindScope Queries: ", self.metrics.findscope_queries)
                print("FindC Queries: ", self.metrics.findc_queries)

            gen_start = time.time()

            gen_flag, Y = self.call_query_generation(answer)

            gen_end = time.time()

            if not gen_flag:
                # if no query can be generated it means we have converged to the target network -----
                break

            kappaB = get_kappa(self.B, Y)
            Yprime = Y.copy()
            answer = True

            self.metrics.increase_generation_time(gen_end - gen_start)
            self.metrics.increase_generated_queries()

            while len(kappaB) > 0:

                self.metrics.increase_top_queries()

                if self.ask_query(Yprime):
                    # it is a solution, so all candidates violated must go
                    # B <- B \setminus K_B(e)
                    self.remove_from_bias(kappaB)
                    kappaB = set()

                else:  # user says UNSAT

                    answer = False

                    scope = self.call_findscope(Yprime, kappaB)
                    self.call_findc(scope)
                    NScopes = set()
                    NScopes.add(tuple(scope))

                    if self.perform_analyzeAndLearn:
                        NScopes = NScopes.union(self.analyze_and_learn(Y))

                    Yprime = [y2 for y2 in Yprime if not any(y2 in set(nscope) for nscope in NScopes)]

                    kappaB = get_kappa(self.B, Yprime)

    def analyze_and_learn(self, Y):

        NScopes = set()
        QCliques = set()

        # Find all neighbours (not just a specific type)
        self.cl_neighbours = self.get_neighbours(self.C_l.constraints)

        # Gamma precentage in FindQCliques is set to 0.8
        self.find_QCliques(self.X.copy(), set(), set(), QCliques, 0.8, 0)

        # [self.isQClique(clique, 0.8) for clique in QCliques]

        cliques_relations = self.QCliques_relations(QCliques)

        # Find the scopes that have a constraint in B violated, which can fill the incomplete cliques
        if len(QCliques) == 0:
            return set()

        Cq = [c for c in get_kappa(self.B, Y) if any(
            set(get_scope(c)).issubset(clique) and get_relation(c, self.gamma) in cliques_relations[i] for i, clique in
            enumerate(QCliques))]

        PScopes = {tuple(get_scope(c)) for c in Cq}

        for pscope in PScopes:

            if self.ask_query(pscope):
                # It is a solution, so all candidates violated must go
                # B <- B \setminus K_B(e)
                kappaB = get_kappa(self.B, pscope)
                self.remove_from_bias(kappaB)

            else:  # User says UNSAT

                # c <- findC(e, findScope(e, {}, grid, false))
                c = self.call_findc(pscope)

                NScopes.add(tuple(pscope))

        if len(NScopes) > 0:
            NScopes = NScopes.union(self.analyze_and_learn(Y))

        return NScopes

    def get_neighbours(self, C, type=None):

        # In case a model is given in the function instead of a list of constraints
        if not (isinstance(C, list) or isinstance(C, set)):
            C = C.constraints

        neighbours = np.zeros((len(self.X), len(self.X)), dtype=bool)

        for c in C:

            flag = False

            if type is not None:
                if self.gamma[get_relation(c, self.gamma)] == type:
                    flag = True
            else:
                flag = True

            if flag:
                scope = get_scope(c)

                i = self.hashX.index(hash(scope[0]))
                j = self.hashX.index(hash(scope[1]))

                neighbours[i][j] = True
                neighbours[j][i] = True

        return neighbours

    def QCliques_relations(self, QCliques):

        cl_relations = [get_relation(c, self.gamma) for c in self.C_l.constraints]
        cliques_relations = [[rel for i, rel in enumerate(cl_relations)
                              if set(get_scope(self.C_l.constraints[i])).issubset(clique)] for clique in QCliques]

        return cliques_relations

    # For debugging
    def is_QClique(self, clique, gammaPerc):

        edges = 0

        q = len(clique)
        q = gammaPerc * (q * (q - 1) / 2)  # number of edges needed to be considered a quasi-clique

        for var1, var2 in all_pairs(clique):
            k = self.hashX.index(hash(var1))
            l = self.hashX.index(hash(var2))

            if self.cl_neighbours[k, l]:
                edges = edges + 1

        if edges < q:
            raise Exception(
                f'findQCliques returned a clique that is not a quasi clique!!!! -> {clique} \nedges = {edges}\nq = {q}')


    def find_QCliques(self, A, B, K, QCliques, gammaPerc, t):
        """
            Find quasi cliques

            A: a mutable list of all variables (nodes in the graph)
            gammaPerc: percentage of neighbors to be considered a quasi clique
            t: total time counter
        """
        global cliques_cutoff

        start = time.time()

        if len(A) == 0 and len(K) > 2:
            if not any(K.issubset(set(clique)) for clique in QCliques):
                QCliques.add(tuple(K))

        while len(A) > 0:

            end = time.time()
            t = t + end - start
            start = time.time()

            if t > cliques_cutoff:
                return

            x = A.pop()

            K2 = K.copy()
            K2.add(x)

            A2 = set(self.X) - K2 - B
            A3 = set()

            # calculate the number of existing edges on K2
            edges = 0
            for var1, var2 in all_pairs(K2):
                k = self.hashX.index(hash(var1))
                l = self.hashX.index(hash(var2))

                if self.cl_neighbours[k, l]:
                    edges = edges + 1

            q = len(K2) + 1
            q = gammaPerc * (q * (q - 1) / 2)  # number of edges needed to be considered a quasi-clique

            # for every y in A2, check if K2 U y is a gamma-clique (if so, store in A3)
            for y in list(A2):  # take (yet another) copy

                edges_with_y = edges

                # calculate the number of from y to K2
                for var in K2:

                    k = self.hashX.index(hash(var))
                    l = self.hashX.index(hash(y))

                    if self.cl_neighbours[k, l]:
                        edges_with_y = edges_with_y + 1

                if edges_with_y >= q:
                    A3.add(y)

            self.find_QCliques(A3, B.copy(), K2.copy(), QCliques, gammaPerc, t)

            B.add(x)