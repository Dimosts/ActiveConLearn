import time

from ConAcq import ConAcq
from utils import get_kappa, construct_bias


class MQuAcq(ConAcq):
    def __init__(self, gamma, grid, ct, bias, X, C_l, qg="pqgen", gqg= False, gfs=False, gfc=False, obj="proba", classifier=None,
                 classifier_name=None, time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000):
        super().__init__(gamma, grid, ct, bias, X, C_l, qg, gqg, gfs, gfc, obj, classifier, classifier_name, time_limit, findscope_version,
                    findc_version, tqgen_t, qgen_blimit)

    def learn(self):

        answer = True

        if len(self.B) == 0:
            self.B = construct_bias(self.X, self.gamma)

        while True:

            if self.debug_mode:
                print("Size of CL: ", len(self.C_l.constraints))
                print("Size of B: ", len(self.B))

            gen_start = time.time()
            # generate e in D^X accepted by C_l and rejected by B
            gen_flag, Y = self.call_query_generation(answer)
            gen_end = time.time()

            if not gen_flag:
                # if no query can be generated it means we have converged to the target network -----
                return self.C_l

            self.metrics.increase_generation_time(gen_end - gen_start)
            self.metrics.increase_generated_queries()
            learned_scopes = self.find_all_cons(list(Y), set())
            if len(learned_scopes) > 0:
                answer = False


    def find_all_cons(self, Y, Scopes):
        kappa = get_kappa(self.B, Y)
        if len(kappa) == 0:
            return set()

        NScopes = set()

        if len(Scopes) > 0:
            s = Scopes.pop()
            for x in s:
                Y2 = set(Y.copy())
                if x in Y2:
                    Y2.remove(x)

                scopes = self.find_all_cons(list(Y2), NScopes.union(Scopes))
                NScopes = NScopes.union(scopes)

        else:
            self.metrics.increase_top_queries()
            if self.ask_query(Y):
                self.remove_from_bias(kappa)
            else:
                scope = self.call_findscope(Y, kappa)
                self.call_findc(scope)

                NScopes.add(frozenset(scope))

                NScopes = NScopes.union(self.find_all_cons(Y, NScopes.copy()))

        return NScopes