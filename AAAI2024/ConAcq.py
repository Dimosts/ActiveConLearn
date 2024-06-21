from statistics import mean, stdev

SOLVER = "ortools"

from cpmpy import *
from cpmpy.transformations.get_variables import get_variables
from utils import *
import math
from ortools.sat.python import cp_model as ort
from sklearn.preprocessing import MinMaxScaler

from utils import find_suitable_vars_subset2, find_optimal_vars_subset

partial = False
cliques_cutoff = 0.5


class ConAcq:

    def __init__(self, gamma, grid, ct, bias, X, C_l, qg="pqgen", gqg= False, gfs=False, gfc=False, obj="proba", classifier=None,
                 classifier_name=None, time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000):

        self.debug_mode = False

        # Target network
        self.C_T = ct

        self.grid = grid
        self.gamma = gamma
        self.B = bias

        # Guery generation, FindScope and FindC versions
        self.qg = qg
        self.fs = findscope_version
        self.fc = findc_version

        # Guide query generation
        self.gqg = gqg
        # Guide FindScope
        self.gfs = gfs
        # Guide FindC
        self.gfc = gfc

        # Objective
        self.obj = obj

        # For the counts
        self.counts = [0] * len(gamma)
        self.countsB = [0] * len(gamma)

        # For guiding using (probabilistic) classification
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.trained = False

        # Initialize learned network
        if len(C_l) > 0:
            self.C_l = Model(C_l)
        else:
            self.C_l = Model()

        # Initialize variables
        if len(X) > 0:
            self.X = list(X)
        else:
            self.X = list(grid.flatten())

        # Hash the variables
        self.hashX = [hash(x) for x in self.X]

        # For TQ-Gen
        self.alpha = 0.8
        self.l = len(self.X)

        # Decide to use predicted classes, or the predicted class probabilities
        if self.obj in ["class", "proba"] and self.classifier_name != "counts":
            self.classify = True
        else:
            self.classify = False

        # Query generation time limit
        if time_limit is None:
            time_limit = 1
        self.time_limit = time_limit

        # TQGen's time limit tau
        if tqgen_t is None:
            tqgen_t = 0.20
        self.tqgen_t = tqgen_t

        # Bias size limit for determining type of query generation (denoted with 'l' in paper)
        self.qgen_blimit = qgen_blimit

        # Time limit for during FindScope with objective function
        self.fs_limit = 0.5

        self.dataset_X = []
        self.dataset_Y = []

        self.metrics = Metrics()

    def train_classifier(self):

        # Train the specified machine learning classifier

        if self.classifier_name == "GaussianNB":  # If GNB, try to use class weights, to handle the imbalance
            # (doesn't really improve though)
            self.classifier.fit(self.dataset_X, self.dataset_Y, sample_weight=compute_sample_weights(self.dataset_Y))
        elif self.classifier_name == "SVM":  # if SVM, scale data to reduce (massively) the training time
            scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
            X = scaler.fit_transform(self.dataset_X)
            self.classifier.fit(X, self.dataset_Y)
        else:
            self.classifier.fit(self.dataset_X, self.dataset_Y)

        self.trained = True

    def set_counts(self, counts, countsB):

        self.counts = counts
        self.countsB = countsB

    def increase_count_cl(self, c):

        # Increase the count for the relation of a learned constraint

        rel = get_relation(c, self.gamma)
        self.counts[rel] += 1

    def increase_count_bias(self, c):

        # Increase the count for the relation of a constraint removed from the bias
        rel = get_relation(c, self.gamma)
        self.countsB[rel] += 1
        return True

    def get_counts(self):

        return self.counts, self.countsB

    def set_dataset(self, dataset_X, dataset_Y):

        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y

    def get_dataset(self):

        return self.dataset_X, self.dataset_Y

    def remove_from_bias(self, C):

        # Remove all the constraints from network C from B
        if self.obj == "proba" or "class":
            for c in C:
                if self.classifier_name == "counts":
                    self.increase_count_bias(c)
                else:
                    x_c = self.get_con_features(c)
                    self.dataset_X.append(x_c)
                    y_c = 0
                    self.dataset_Y.append(y_c)

        prev_B_length = len(self.B)
        self.B = list(set(self.B) - set(C))

        if not (prev_B_length - len(C) == len(self.B)):
            raise Exception(f'something was not removed properly')

    def add_to_cl(self, c):

        # Add a constraint c to the learned network

        if self.debug_mode:
            print(f"adding {c} to C_L")
        self.C_l += c
        self.increase_count_cl(c)
        x_c = self.get_con_features(c)
        self.dataset_X.append(x_c)
        y_c = 1
        self.dataset_Y.append(y_c)

    def get_con_features(self, c):

        # Returns the features associated with constraint c
        features = []
        scope = get_scope(c)

        var_name = get_var_name(scope[0])
        var_name_same = all([(get_var_name(var) == var_name) for var in scope])
        features.append(var_name_same)

        # Global var dimension properties
        vars_ndims = [get_var_ndims(var) for var in scope]
        ndims_max = max(vars_ndims)

        vars_dims = [get_var_dims(var) for var in scope]
        dim = []
        for j in range(ndims_max):
            dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])

            dimj_has = len(dim[j]) > 0
            # features.append(dimj_has)

            if dimj_has:
                dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
                features.append(dimj_same)
                dimj_max = max(dim[j])
                features.append(dimj_max)
                dimj_min = min(dim[j])
                features.append(dimj_min)
                dimj_avg = mean(dim[j])
                features.append(dimj_avg)
                dimj_dev = stdev(dim[j])
                features.append(dimj_dev)

            else:
                features.append(True)
                for i in range(3): features.append(0) 
                features.append(0.0)

        con_in_gamma = get_relation(c, self.gamma)
        features.append(con_in_gamma)

        arity = len(scope)
        features.append(arity)

        con_name = self.gamma[get_relation(c, self.gamma)]
        num = re.findall("[^r]\d+", con_name)
        has_const = len(num) > 0
        features.append(has_const)

        if has_const:
            features.append(int(num[0]))
        else:
            features.append(0)

        return features

    def remove(self, B, C):

        # Remove all constraints from network C from B
        lenB = len(B)
        i = 0

        while i < lenB:

            if any(B[i] is c2 for c2 in C):
                # B[i] in set(C) in condition is slower

                B.pop(i)
                i -= 1
                lenB -= 1
            i += 1

    def remove_scope_from_bias(self, scope):

        # Remove all constraints with the given scope from B
        scope_set = set(scope)
        learned_con_rel = get_relation(self.C_l.constraints[-1], self.gamma)
        B_scopes_set = [set(get_scope(c)) for c in self.B]

        removing = [c for i, c in enumerate(self.B) if B_scopes_set[i] == scope_set
                    and get_relation(c, self.gamma) != learned_con_rel]

        for c in removing:
            self.increase_count_bias(c)
            x_c = self.get_con_features(c)
            self.dataset_X.append(x_c)
            y_c = 0
            self.dataset_Y.append(y_c)

        prev_B_length = len(self.B)
        self.B = [c for i, c in enumerate(self.B) if not (B_scopes_set[i] == scope_set)]

        if len(self.B) == prev_B_length:
            if self.debug_mode:
                print(self.B)
                print(scope)
            raise Exception("Removing constraints from Bias did not result in reducing its size")

    def set_cl(self, C_l):

        if isinstance(C_l, Model):
            self.C_l = C_l
        elif isinstance(C_l, set) or isinstance(C_l, list):
            self.C_l = Model(C_l)

    def call_findc(self, scope):

        # Call the specified findC function

        if self.fc == 1:
            # Initialize delta
            delta = get_con_subset(self.B, scope)
            delta = [c for c in delta if check_value(c) is False]

            if not self.gfc:
                c = self.findC(scope, delta)
            else:

                # Use classifier to predict probabilities of constraints in kappaB
                if self.trained:

                    data_pred = [self.get_con_features(c) for c in delta]
                    myscore = self.classifier.predict_proba(data_pred)
                    P_c = [myscore[i][1] for i in range(len(myscore))]

                    for i in range(len(P_c)):
                        if P_c[i] == 0:
                            P_c[i] = 0.01

                else:
                    P_c = [0.1 for c in delta]

                c = self.findC(scope, delta, P_c)

        else:
            c = self.findC2(scope)

        if c is None:
            raise Exception("findC did not find any, collapse")
        else:
            self.add_to_cl(c)
            self.remove_scope_from_bias(scope)

    def call_findscope(self, Y, kappa):


        # Call the specified findScope function
        if self.fs == 1:
            scope = self.findScope(self.grid.value(), set(), Y, do_ask=False)
        else:

            # if we want to guide findscope
            if self.gfs:
                # Use classifier to predict probabilities of constraints in kappaB
                if self.trained:
                    data_pred = [self.get_con_features(c) for c in kappa]
                    myscore = self.classifier.predict_proba(data_pred)
                    P_c = [myscore[i][1] for i in range(len(myscore))]

                    for i in range(len(P_c)):
                        if P_c[i] == 0:
                            P_c[i] = 0.01

                else:
                    P_c = [0.1 for _ in kappa]
            else:
                P_c = None

            scope = self.findScope2(self.grid.value(), set(), Y, kappa, P_c)


        return scope

    def adjust_tq_gen(self, l, a, answer):

        # Adjust the number of variables taken into account in the next iteration of TQGen
        if answer:
            l = min([int(math.ceil(l / a)), len(self.X)])
        else:
            l = int((a * l) // 1)  # //1 to round down
            if l < get_min_arity(self.B):
                l = 2

        return l

    def tq_gen(self, alpha, t, l):
        # Generate a query using TQGen
        # alpha: reduction factor
        # t: solving timeout
        # l: expected query size

        ttime = 0

        while ttime < self.time_limit and len(self.B) > 0:

            t = min([t, self.time_limit - ttime])
            l = max([l, get_min_arity(self.B)])

            Y = find_suitable_vars_subset2(l, self.B, self.X)

            B = get_con_subset(self.B, Y)
            Cl = get_con_subset(self.C_l.constraints, Y)

            m = Model(Cl)
            s = SolverLookup.get(SOLVER, m)

            # Create indicator variables upfront
            V = boolvar(shape=(len(B),))
            s += (V != B)

            # We want at least one constraint to be violated
            if self.debug_mode:
                print("length of B: ", len(B))
                print("l: ", l)
            s += sum(V) > 0

            t_start = time.time()
            flag = s.solve(time_limit=t)
            ttime = ttime + (time.time() - t_start)

            if flag:
                return flag, Y

            if s.ort_status == ort.INFEASIBLE:
                [self.add_to_cl(c) for c in B]
                self.remove_from_bias(B)
            else:
                l = int((alpha * l) // 1)  # //1 to round down

        if len(self.B) > 0:
            self.converged = 0

        return False, []

    def generate_query(self):
        # A basic version of query generation for small problems. May lead
        # to premature convergence, so generally not used

        if len(self.B) == 0:
            return False

        # B are taken into account as soft constraints that we do not want to satisfy (i.e., that we want to violate)
        m = Model(self.C_l.constraints)  # could use to-be-implemented m.copy() here...

        # Get the amount of satisfied constraints from B
        objective = sum([c for c in self.B])

        # We want at least one constraint to be violated to assure that each answer of the
        # user will reduce the set of candidates
        m += objective < len(self.B)

        s = SolverLookup.get(SOLVER, m)
        flag = s.solve(time_limit=600)

        if not flag:
            # If a solution is found, then continue optimizing it
            if s.ort_status == ort.UNKNOWN:
                self.converged = 0

        return flag

    def pqgen(self, time_limit=1):
        # Generate a query using PQGen

        # Start time (for the cutoff t)
        t0 = time.time()

        # Project down to only vars in scope of B
        Y = frozenset(get_variables(self.B))
        lY = list(Y)

        if len(Y) == len(self.X):
            B = self.B
            Cl = self.C_l.constraints
        else:
            B = get_con_subset(self.B, Y)
            Cl = get_con_subset(self.C_l.constraints, Y)

        global partial

        # If no constraints left in B, just return
        if len(B) == 0:
            return False, set()

        # If no constraints learned yet, start by just generating an example in all the variables in Y
        if len(Cl) == 0:
            Cl = [sum(Y) >= 1]

        if not partial and len(B) > self.qgen_blimit:

            m = Model(Cl)
            flag = m.solve()  # no time limit to ensure convergence

            if flag and not all([c.value() for c in B]):
                return flag, lY
            else:
                partial = True

        m = Model(Cl)
        s = SolverLookup.get(SOLVER, m)

        # Create indicator variables upfront
        V = boolvar(shape=(len(B),))
        s += (V == B)

        # We want at least one constraint to be violated to assure that each answer of the user
        # will lead to new information
        s += ~all(V)

        # Solve first without objective (to find at least one solution)
        flag = s.solve()

        t1 = time.time() - t0
        if not flag or (t1 > time_limit):
            # UNSAT or already above time_limit, stop here --- cannot maximize
            if self.debug_mode:
                print("RR1Time:", time.time() - t0, len(B), len(Cl))
            return flag, lY

        # Next solve will change the values of the variables in the lY2
        # so we need to return them to the original ones to continue if we dont find a solution next
        values = [x.value() for x in lY]

        # So a solution was found, try to find a better one now
        s.solution_hint(lY, values)

        if self.obj == "max":
            objective = sum([~v for v in V])

        elif self.obj == "proba":

            if self.classifier_name == "counts":
                theta = [(self.counts[i] + 0.25) / (self.countsB[i] + 0.5) for i in range(len(self.gamma))]
                P_c = [theta[get_relation(c, self.gamma)] for c in B]

            else:
                # Use classifier to predict probabilities of constraints in B
                if len(self.C_l.constraints) > 0:
                    t_ml = time.time()
                    self.train_classifier()
                    t_ml_end = time.time() - t_ml

                    data_pred = [self.get_con_features(c) for c in B]
                    myscore = self.classifier.predict_proba(data_pred)
                    myscore = [m if len(m) > 1 else [0, m[0]] for m in myscore]

                    P_c = [myscore[i][1] for i in range(len(myscore))]

                    for i in range(len(P_c)):
                        if P_c[i] == 0:
                            P_c[i] = 0.01

                else:
                    P_c = [0.1 for _ in B]

            objective = sum(
                [~v * (1 - len(self.gamma) * ((1 / P_c[c]) <= math.log2(len(Y)))) for
                 v, c in zip(V, range(len(B)))])

        else:  # i.e., if self.obj == "class":

            if self.classifier_name == "counts":
                theta = [(self.counts[i] + 0.25) / (self.countsB[i] + 0.5) for i in range(len(self.gamma))]
                P_c = [theta[get_relation(c, self.gamma)] for c in B]
                C_c = [0 if P_c[i] < 0.5 else 1 for i,c in enumerate(B)]

            else:
                # Use classifier to predict probabilities of constraints in B
                if len(self.C_l.constraints) > 0:

                    t_ml = time.time()
                    self.train_classifier()
                    t_ml_end = time.time() - t_ml

                    data_pred = [self.get_con_features(c) for c in B]
                    C_c = self.classifier.predict(data_pred)

                else:
                    C_c = [0 for _ in B]

            objective = sum(
                [~v * (1 - len(self.gamma) * C_c[c]) for
                 v, c in zip(V, range(len(B)))])

        # Run with the objective
        s.maximize(objective)

        flag2 = s.solve(time_limit=(time_limit - t1))

        if flag2:
            if self.debug_mode:
                print("RR2Time:", time.time() - t0, len(B), len(Cl))
            return flag2, lY
        else:
            tmp = Model()
            i = 0
            for x in lY:
                tmp += x == values[i]
                i = i + 1

            tmp.solve()
            if self.debug_mode:
                print("RR3Time:", time.time() - t0, len(B), len(Cl))
            return flag, lY

    def call_query_generation(self, answer=None):
        # Call the specified query generation method

        # Generate e in D^X accepted by C_l and rejected by B
        if self.qg == "base":
            gen_flag = self.generate_query()
            Y = self.X
        elif self.qg == "pqgen":
            gen_flag, Y = self.pqgen(time_limit=self.time_limit)
        elif self.qg == "tqgen":

            if self.queries_count > 0:
                self.l = self.adjust_tq_gen(self.l, self.alpha, answer)

            gen_flag, Y = self.tq_gen(self.alpha, self.tqgen_t, self.l)

        else:
            raise Exception("Error: No available query generator was selected!!")

        return gen_flag, Y

    def ask_query(self, value):

        if not (isinstance(value, list) or isinstance(value, set) or isinstance(value, frozenset) or
                isinstance(value, tuple)):
            Y = set()
            Y = Y.union(self.grid[value != 0])
        else:
            Y = value
            e = self.grid.value()

            value = np.zeros(e.shape, dtype=int)

            # Create a truth table numpy array
            sel = np.array([item in set(Y) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            # Variables present in the partial query
            value[sel] = e[sel]

        # Post the query to the user/oracle
        if self.debug_mode:
            print(f"Query({self.metrics.queries_count}: is this a solution?")
            print(value)
            print(f"Query: is this a solution?")
            print(np.array([[v if v != 0 else -0 for v in row] for row in value]))

            print("violated from B: ", get_kappa(self.B, Y))
            print("violated from C_T: ", get_kappa(self.C_T, Y))
            print("violated from C_L: ", get_kappa(self.C_l.constraints, Y))

        # Need the oracle to answer based only on the constraints with a scope that is a subset of Y
        suboracle = get_con_subset(self.C_T, Y)

        # Check if at least one constraint is violated or not
        ret = all([check_value(c) for c in suboracle])

        if self.debug_mode:
            print("Answer: ", ("Yes" if ret else "No"))

        # For the evaluation metrics

        # Increase the number of queries
        self.metrics.increase_queries_count()
        self.metrics.increase_queries_size(len(Y))

        # Measuring the waiting time of the user from the previous query
        end_time_query = time.time()
        # To measure the maximum waiting time for a query
        waiting_time = end_time_query - self.metrics.start_time_query
        self.metrics.aggreagate_max_waiting_time(waiting_time)
        self.metrics.start_time_query = time.time()  # to measure the maximum waiting time for a query

        return ret

    # This is the version of the FindScope function that was presented in "Constraint acquisition via Partial Queries", IJCAI 2013
    def findScope(self, e, R, Y, do_ask):
        #if self.debug_mode:
            # print("\tDEBUG: findScope", e, R, Y, do_ask)
        if do_ask:
            # if ask(e_R) = yes: B \setminus K(e_R)
            # need to project 'e' down to vars in R,
            # will show '0' for None/na/" ", should create object nparray instead

            e_R = np.zeros(e.shape, dtype=int)
            sel = np.array([item in set(R) for item in list(self.grid.flatten())]).reshape(self.grid.shape)
            if self.debug_mode:
                print(sel)
            if sum(sel) == 0:
                raise Exception("ERR, FindScope, Nothing to select, something went wrong...")

            self.metrics.increase_findscope_queries()

            e_R[sel] = e[sel]
            if self.ask_query(e_R):
                kappaB = get_kappa(self.B, R)
                self.remove_from_bias(kappaB)

            else:
                return set()

        if len(Y) == 1:
            return set(Y)

        s = len(Y) // 2
        Y1, Y2 = Y[:s], Y[s:]

        S1 = self.findScope(e, R.union(Y1), Y2, True)
        S2 = self.findScope(e, R.union(S1), Y1, len(S1) > 0)

        return S1.union(S2)

    # This is the version of the FindScope function that was presented in "Constraint acquisition through Partial Queries", AIJ 2023
    def findScope2(self, e, R, Y, kappaB, proba=None):

        if not frozenset(kappaB).issubset(self.B):
            raise Exception(f"kappaB given in findscope {call} is not part of B: \nkappaB: {kappaB}, \nB: {self.B}")

        if proba is None:
            proba = []

        # if ask(e_R) = yes: B \setminus K(e_R)
        # need to project 'e' down to vars in R,
        # will show '0' for None/na/" ", should create object nparray instead
        kappaBR = get_con_subset(kappaB, list(R))
        if len(kappaBR) > 0:

            e_R = np.zeros(e.shape, dtype=int)

            sel = np.array([item in set(R) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            if sum(sel) == 0:
                raise Exception("ERR, FindScope, Nothing to select, something went wrong...")

            self.metrics.increase_findscope_queries()

            e_R[sel] = e[sel]
            if self.ask_query(e_R):
                self.remove_from_bias(kappaBR)
                kappaB_prev = kappaB.copy()
                self.remove(kappaB, kappaBR)

                if self.gfs:
                    proba = [proba[i] for i in range(len(proba)) if any(kappaB_prev[i] is c for c in kappaB)]

            else:
                return set()

        if len(Y) == 1:
            return set(Y)

        # Create Y1, Y2 -------------------------
        if self.gfs:
            Y1, Y2 = find_optimal_vars_subset(R, Y, kappaB, proba)
        else:
            s = len(Y) // 2
            Y1, Y2 = Y[:s], Y[s:]

        S1 = set()
        S2 = set()

        # R U Y1
        RY1 = R.union(Y1)

        kappaBRY = kappaB.copy()
        kappaBRY_prev = kappaBRY.copy()
        kappaBRY1 = get_con_subset(kappaBRY, RY1)

        if len(kappaBRY1) < len(kappaBRY):
            S1 = self.findScope2(e, RY1, Y2, kappaBRY, proba.copy())

        # remove from original kappaB
        kappaBRY_removed = set(kappaBRY_prev) - set(kappaBRY)
        if self.gfs:
            proba = [proba[i] for i in range(len(proba)) if all(kappaB[i] is not c for c in kappaBRY_removed)]
        self.remove(kappaB, kappaBRY_removed)

        # R U S1
        RS1 = R.union(S1)

        kappaBRS1 = get_con_subset(kappaBRY, RS1)
        kappaBRS1Y1 = get_con_subset(kappaBRY, RS1.union(Y1))
        kappaBRS1Y1_prev = kappaBRS1Y1.copy()
        probaBRS1Y1 = [proba[i] for i in range(len(proba)) if any(kappaB[i] is c for c in kappaBRS1Y1)]

        if len(kappaBRS1) < len(kappaBRY):
            S2 = self.findScope2(e, RS1, Y1, kappaBRS1Y1, probaBRS1Y1)

        # remove from original kappaB
        kappaBRS1Y1_removed = set(kappaBRS1Y1_prev) - set(kappaBRS1Y1)
        self.remove(kappaB, kappaBRS1Y1_removed)

        return S1.union(S2)

    # This is the version of the FindC function that was presented in
    # "Constraint acquisition via Partial Queries", IJCAI 2013
    def findC(self, scope, delta, P_c=None):
        # This function works only for normalised target networks!
        # A modification that can also learn conjunction of constraints in each scope is described in the
        # article "Partial Queries for Constraint Acquisition" that is published in AIJ !

        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.C_l.constraints, scope)

        scope_values = [x.value() for x in scope]

        while True:
            # Try to generate a counter example to reduce the candidates
            if self.gfc:
                flag = generate_findc_query3(sub_cl, delta, P_c)
            else:
                flag = generate_findc_query(sub_cl, delta)

            if flag is False:
                # If no example could be generated
                # check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    print("Collapse, the constraint we seek is not in B")
                    exit(-2)

                # FindC changes the values of the variables in the scope,
                # so we need to return them to the original ones to continue
                tmp = Model()
                i = 0
                for x in scope:
                    tmp += x == scope_values[i]
                    i = i + 1

                tmp.solve()

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            # Ask the partial counter example and update delta depending on the answer of the oracle
            e = self.grid.value()

            sel = np.array([item in set(scope) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            e_S = np.zeros(e.shape, dtype=int)
            e_S[sel] = e[sel]

            self.metrics.increase_findc_queries()

            if self.ask_query(e_S):
                # delta <- delta \setminus K_{delta}(e)
                delta_prev = delta.copy()
                delta = [c for c in delta if check_value(c) is not False]
                if self.gfc:
                    P_c = [P_c[i] for i in range(len(P_c)) if any(delta_prev[i] is c for c in delta)]

            else:  # user says UNSAT
                # delta <- K_{delta}(e)
                delta_prev = delta.copy()
                delta = [c for c in delta if check_value(c) is False]
                if self.gfc:
                    P_c = [P_c[i] for i in range(len(P_c)) if any(delta_prev[i] is c for c in delta)]

    # This is the version of the FindC function that was presented in
    # "Constraint acquisition through Partial Queries", AIJ 2023
    def findC2(self, scope):
        # This function works also for non-normalised target networks!!!

        # Initialize delta
        delta = get_con_subset(self.B, scope)
        delta = join_con_net(delta, [c for c in delta if check_value(c) is False])

        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.C_l.constraints, scope)

        scope_values = [x.value() for x in scope]

        while True:

            # Try to generate a counter example to reduce the candidates
            if generate_findc2_query(sub_cl, delta) is False:

                # If no example could be generated
                # check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    print("Collapse, the constraint we seek is not in B")
                    exit(-2)

                # FindC changes the values of the variables in the scope,
                # so we need to return them to the original ones to continue
                tmp = Model()
                i = 0
                for x in scope:
                    tmp += x == scope_values[i]
                    i = i + 1

                tmp.solve()

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            # Ask the partial counter example and update delta depending on the answer of the oracle
            e = self.grid.value()
            sel = np.array([item in set(scope) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            e_S = np.zeros(e.shape, dtype=int)
            e_S[sel] = e[sel]

            self.metrics.findc_queries()

            if self.ask_query(e_S):
                # delta <- delta \setminus K_{delta}(e)
                delta = [c for c in delta if check_value(c) is not False]
            else:  # user says UNSAT
                # delta <- joint(delta,K_{delta}(e))

                kappaD = [c for c in delta if check_value(c) is False]

                scope2 = self.call_findscope(list(scope), kappaD)

                if len(scope2) < len(scope):
                    self.call_findc(scope2)
                else:
                    delta = join_con_net(delta, kappaD)