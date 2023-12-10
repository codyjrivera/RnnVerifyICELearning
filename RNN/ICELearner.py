from timeit import default_timer as timer
from RNN.ICEMultiLayerBase import ICEMultiLayer
from RNN.MarabouRnnModel import RnnMarabouModel
from fractions import Fraction
import cvc5
from cvc5 import Kind
import code
import sys

TIMEOUT = 60


class ICELearner:
    def __init__(self):
        self.solver = cvc5.Solver()
        self.solver.setLogic("LRA")
        self.solver.setOption("sygus", "true")
        self.solver.setOption("incremental", "false")
        self.solver.setOption("tlimit", str(TIMEOUT * 1000))

    def make_template_fun(self, c1, c2, c3, c4, t, m):
        solver = self.solver

        lb_term = solver.mkTerm(
            Kind.GEQ,
            m,
            solver.mkTerm(Kind.ADD, solver.mkTerm(Kind.MULT, c1, t), c2),
        )

        ub_term = solver.mkTerm(
            Kind.LEQ,
            m,
            solver.mkTerm(Kind.ADD, solver.mkTerm(Kind.MULT, c3, t), c4),
        )

        body = solver.mkTerm(Kind.AND, lb_term, ub_term)
        boolean = solver.getBooleanSort()
        return solver.defineFun("Inv", [c1, c2, c3, c4, t, m], boolean, body)

    def make_const_grammar(self, inc):
        solver = self.solver
        real = solver.getRealSort()

        # Declare non-terminals
        c_nt = solver.mkVar(real, "C")
        cp_nt = solver.mkVar(real, "Cp")
        cn_nt = solver.mkVar(real, "Cn")

        # Make grammar
        g = solver.mkGrammar([], [c_nt, cp_nt, cn_nt])

        # Productions for C
        g.addRules(c_nt, [cp_nt, cn_nt])

        # Productions for Cp
        cp_prod0 = solver.mkTerm(Kind.ADD, cp_nt, inc)
        g.addRules(cp_nt, [solver.mkReal(0), cp_prod0])

        # Productions for Cn
        cn_prod0 = solver.mkTerm(Kind.SUB, cn_nt, inc)
        g.addRules(cn_nt, [solver.mkReal(0), cn_prod0])

        return g

    def do_round(self, rnnModel: RnnMarabouModel, algorithm: ICEMultiLayer, bk=False):
        solver = self.solver
        real = solver.getRealSort()
        boolean = solver.getBooleanSort()
        c1 = solver.mkVar(real, "C1")
        c2 = solver.mkVar(real, "C2")
        c3 = solver.mkVar(real, "C3")
        c4 = solver.mkVar(real, "C4")
        self.t = solver.mkVar(real, "t")
        self.m = solver.mkVar(real, "m")

        increments = [
            solver.mkReal(0),
            solver.mkTerm(Kind.DIVISION, solver.mkReal(1), solver.mkReal(10)),
            solver.mkTerm(Kind.DIVISION, solver.mkReal(1), solver.mkReal(100))
        ]

        lower_limits = [
            solver.mkReal(-500),
            solver.mkTerm(Kind.DIVISION, solver.mkReal(-9), solver.mkReal(10)),
            solver.mkTerm(Kind.DIVISION, solver.mkReal(-9), solver.mkReal(100))
        ]

        upper_limits = [
            solver.mkReal(500),
            solver.mkTerm(Kind.DIVISION, solver.mkReal(9), solver.mkReal(10)),
            solver.mkTerm(Kind.DIVISION, solver.mkReal(9), solver.mkReal(100))
        ]

        const_grammars = []
        for inc in increments:
            const_grammars.append(self.make_const_grammar(inc))

        inv_template = self.make_template_fun(c1, c2, c3, c4, self.t, self.m)

        invariants = []
        for l in range(rnnModel.num_rnn_layers):
            rnn_start_idxs, _ = rnnModel.get_start_end_idxs(rnn_layer=l)
            invariants_inner = []

            # Declare invariants to synthesize
            for i in range(len(rnn_start_idxs)):
                al = []
                for j, g in enumerate(const_grammars):
                    al.append(
                        solver.synthFun(
                            "al_" + str(l) + "_" + str(i) + "_" + str(j), 
                            [], 
                            real, 
                            g
                        )
                    )
                bl = []
                for j, g in enumerate(const_grammars):
                    bl.append(
                        solver.synthFun(
                            "bl_" + str(l) + "_" + str(i) + "_" + str(j), 
                            [], 
                            real, 
                            g
                        )
                    )
                ah = []
                for j, g in enumerate(const_grammars):
                    ah.append(
                        solver.synthFun(
                            "ah_" + str(l) + "_" + str(i) + "_" + str(j), 
                            [], 
                            real, 
                            g
                        )
                    )
                bh = []
                for j, g in enumerate(const_grammars):
                    bh.append(
                        solver.synthFun(
                            "bh_" + str(l) + "_" + str(i) + "_" + str(j), 
                            [], 
                            real, 
                            g
                        )
                    )
                invariants_inner.append(
                    [al, bl, ah, bh]
                )

            # Add positive counterexamples
            #print("pos cex", algorithm.alphas_algorithm_per_layer[l].pos_cex)
            for i, t, p in algorithm.alphas_algorithm_per_layer[l].pos_cex:
                solver.addSygusConstraint(
                    solver.mkTerm(
                        Kind.APPLY_UF,
                        inv_template,
                        solver.mkTerm(Kind.ADD, *(invariants_inner[i][0])),
                        solver.mkTerm(Kind.ADD, *(invariants_inner[i][1])),
                        solver.mkTerm(Kind.ADD, *(invariants_inner[i][2])),
                        solver.mkTerm(Kind.ADD, *(invariants_inner[i][3])),
                        solver.mkReal(t),
                        solver.mkReal(Fraction(str(p))),
                    )
                )

            # Add implication counterexamples
            #print("imp cex", algorithm.alphas_algorithm_per_layer[l].imp_cex)
            for i, t, (r, s) in algorithm.alphas_algorithm_per_layer[l].imp_cex:
                t = solver.mkTerm(
                        Kind.IMPLIES,
                        solver.mkTerm(
                            Kind.APPLY_UF,
                            inv_template,
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][0])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][1])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][2])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][3])),
                            solver.mkTerm(Kind.SUB, solver.mkReal(t), solver.mkReal(1)),
                            solver.mkReal(Fraction(str(r))),
                        ),
                        solver.mkTerm(
                            Kind.APPLY_UF,
                            inv_template,
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][0])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][1])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][2])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][3])),
                            solver.mkReal(t),
                            solver.mkReal(Fraction(str(s))),
                        )
                    )
                solver.addSygusConstraint(
                    t
                )

            # Add negative counterexamples
            #print("neg cex", algorithm.alphas_algorithm_per_layer[l].neg_cex)
            for cex in algorithm.alphas_algorithm_per_layer[l].neg_cex:
                conjuncts = []
                for i, t, n in cex:
                    conjuncts.append(
                        solver.mkTerm(
                            Kind.APPLY_UF,
                            inv_template,
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][0])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][1])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][2])),
                            solver.mkTerm(Kind.ADD, *(invariants_inner[i][3])),
                            solver.mkReal(t),
                            solver.mkReal(Fraction(str(n))),
                        )
                    )
                solver.addSygusConstraint(
                    solver.mkTerm(
                        Kind.NOT,
                        solver.mkTerm(
                            Kind.AND,
                            *conjuncts
                        )
                    )
                )

            invariants.append(invariants_inner)
        
        res = solver.checkSynth()
        if res.hasSolution():
            for l in range(rnnModel.num_rnn_layers):
                alpha_l = []
                beta_l = []
                alpha_u = []
                beta_u = []

                for inner_invariant in invariants[l]:
                    sols = solver.getSynthSolutions(inner_invariant[0])
                    a_l = sum(map(eval_constants, sols))
                    sols = solver.getSynthSolutions(inner_invariant[1])
                    b_l = sum(map(eval_constants, sols))
                    sols = solver.getSynthSolutions(inner_invariant[2])
                    a_u = sum(map(eval_constants, sols))
                    sols = solver.getSynthSolutions(inner_invariant[3])
                    b_u = sum(map(eval_constants, sols))

                    alpha_l.append(float(a_l))
                    beta_l.append(float(b_l))
                    alpha_u.append(float(a_u))
                    beta_u.append(float(b_u))

                algorithm.alphas_algorithm_per_layer[l].ice_alphas = alpha_l + alpha_u
                algorithm.alphas_algorithm_per_layer[l].ice_betas = beta_l + beta_u
                algorithm.alphas_algorithm_per_layer[l].ice_is_upper\
                    = ([False] * len(invariants[l])) + ([True] * len(invariants[l]))
                # print(algorithm.alphas_algorithm_per_layer[l].ice_alphas)
                # print(algorithm.alphas_algorithm_per_layer[l].ice_betas)
                # print(algorithm.alphas_algorithm_per_layer[l].ice_is_upper)
            return True               
        else:
            return False

    def do_example(self):
        solver = self.solver
        real = solver.getRealSort()
        boolean = solver.getBooleanSort()
        t = solver.mkVar(real, "t")
        m = solver.mkVar(real, "m")
        g = self.makeGrammar(t, m)
        inv = solver.synthFun("Inv", [t, m], boolean, g)

        solver.addSygusConstraint(
            solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(0), solver.mkReal(0))
        )
        solver.addSygusConstraint(
            solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(1), solver.mkReal(3))
        )
        solver.addSygusConstraint(
            solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(2), solver.mkReal(3))
        )
        solver.addSygusConstraint(
            solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(3), solver.mkReal(3.4))
        )
        solver.addSygusConstraint(
            solver.mkTerm(
                Kind.NOT,
                solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(1), solver.mkReal(-5)),
            )
        )
        solver.addSygusConstraint(
            solver.mkTerm(
                Kind.NOT,
                solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(2), solver.mkReal(42)),
            )
        )
        solver.addSygusConstraint(
            solver.mkTerm(
                Kind.NOT,
                solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(2), solver.mkReal(4)),
            )
        )
        solver.addSygusConstraint(
            solver.mkTerm(
                Kind.NOT,
                solver.mkTerm(Kind.APPLY_UF, inv, solver.mkReal(3), solver.mkReal(3.5)),
            )
        )

        if solver.checkSynth().hasSolution():
            # Output should be equivalent to:
            # (define-fun max ((x Int) (y Int)) Int (ite (<= x y) y x))
            # (define-fun min ((x Int) (y Int)) Int (ite (<= x y) x y))
            terms = [inv]
            sols = solver.getSynthSolutions(terms)
            print(extract_constants(sols[0]))
            print_synth_solutions(terms, sols)


# Only works on linear templatesx in the strict format.
def eval_constants(c):
    if c.isRealValue():
        return c.getRealValue()
    if c.getKind() == Kind.ADD:
        return sum(map(lambda x: eval_constants(x), c))
    if c.getKind() == Kind.MULT:
        return eval_constants(c[0]) * eval_constants(c[1])
    if c.getKind() == Kind.SUB:
        return eval_constants(c[0]) - eval_constants(c[1])
    if c.getKind() == Kind.DIVISION:
        return eval_constants(c[0]) / eval_constants(c[1])
    print("Weird expression encountered: ", c, file=sys.stderr)
    exit(0)

def extract_constants(fun):
    body = fun[1]
    lb, ub = body
    al_term = lb[1][0][0]
    bl_term = lb[1][1]
    au_term = ub[1][0][0]
    bu_term = ub[1][1]
    al, bl = eval_constants(al_term), eval_constants(bl_term)
    au, bu = eval_constants(au_term), eval_constants(bu_term)
    # does float suffice?
    return (float(al), float(bl)), (float(au), float(bu))


def define_fun_to_string(f, params, body):
    sort = f.getSort()
    if sort.isFunction():
        sort = f.getSort().getFunctionCodomainSort()
    result = "(define-fun " + str(f) + " ("
    for i in range(0, len(params)):
        if i > 0:
            result += " "
        result += "(" + str(params[i]) + " " + str(params[i].getSort()) + ")"
    result += ") " + str(sort) + " " + str(body) + ")"
    return result


def print_synth_solutions(terms, sols):
    result = "(\n"
    for i in range(0, len(terms)):
        params = []
        body = sols[i]
        if sols[i].getKind() == Kind.LAMBDA:
            params += sols[i][0]
            body = sols[i][1]
        result += "  " + define_fun_to_string(terms[i], params, body) + "\n"
    result += ")"
    print(result)


if __name__ == "__main__":
    learner = ICELearner()
    learner.do_example()
