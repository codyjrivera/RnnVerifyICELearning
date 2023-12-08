from timeit import default_timer as timer
from RNN.ICEMultiLayerBase import ICEMultiLayer
from RNN.MarabouRnnModel import RnnMarabouModel
from fractions import Fraction
import cvc5
from cvc5 import Kind
import code
import sys

TIMEOUT = 60


class ICESMTLearner:
    def __init__(self):
        self.solver = cvc5.Solver()
        self.solver.setLogic("NRA")
        self.solver.setOption("incremental", "false")
        self.solver.setOption("produce-models", "true")
        # Decision procedure (we may still want a timeout)
        #self.solver.setOption("tlimit", str(TIMEOUT * 1000))

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

    def do_round(self, rnnModel: RnnMarabouModel, algorithm: ICEMultiLayer, bk=False):
        solver = self.solver
        real = solver.getRealSort()
        c1 = solver.mkVar(real, "c1")
        c2 = solver.mkVar(real, "c2")
        c3 = solver.mkVar(real, "c3")
        c4 = solver.mkVar(real, "c4")
        t = solver.mkVar(real, "t")
        m = solver.mkVar(real, "m")
        
        inv_temp = self.make_template_fun(c1, c2, c3, c4, t, m)

        invariants = []
        for l in range(rnnModel.num_rnn_layers):
            rnn_start_idxs, _ = rnnModel.get_start_end_idxs(rnn_layer=l)
            invariants_inner = []

            # Declare invariants to synthesize
            for i in range(len(rnn_start_idxs)):
                al = solver.mkConst(real, "al_"+str(l)+"_"+str(i))
                bl = solver.mkConst(real, "bl_"+str(l)+"_"+str(i))
                ah = solver.mkConst(real, "ah_"+str(l)+"_"+str(i))
                bh = solver.mkConst(real, "bh_"+str(l)+"_"+str(i))
                invariants_inner.append(((al, bl), (ah, bh)))

            # Add positive counterexamples
            print("pos cex", algorithm.alphas_algorithm_per_layer[l].pos_cex)
            for i, t, p in algorithm.alphas_algorithm_per_layer[l].pos_cex:
                (al, bl), (ah, bh) = invariants_inner[i]
                solver.assertFormula(
                    solver.mkTerm(
                        Kind.APPLY_UF,
                        inv_temp,
                        al, bl, ah, bh,
                        solver.mkReal(t),
                        solver.mkReal(Fraction(str(p))),
                    )
                )

            # Add implication counterexamples
            print("imp cex", algorithm.alphas_algorithm_per_layer[l].imp_cex)
            for i, t, (r, s) in algorithm.alphas_algorithm_per_layer[l].imp_cex:
                (al, bl), (ah, bh) = invariants_inner[i]
                t = solver.mkTerm(
                        Kind.IMPLIES,
                        solver.mkTerm(
                            Kind.APPLY_UF,
                            inv_temp,
                            al, bl, ah, bh,
                            solver.mkTerm(Kind.SUB, solver.mkReal(t), solver.mkReal(1)),
                            solver.mkReal(Fraction(str(r))),
                        ),
                        solver.mkTerm(
                            Kind.APPLY_UF,
                            inv_temp,
                            al, bl, ah, bh,
                            solver.mkReal(t),
                            solver.mkReal(Fraction(str(s))),
                        )
                    )
                solver.assertFormula(
                    t
                )

            # Add negative counterexamples
            
            print("neg cex", algorithm.alphas_algorithm_per_layer[l].neg_cex)
            for i, t, n in algorithm.alphas_algorithm_per_layer[l].neg_cex:
                solver.assertFormula(
                    solver.mkTerm(
                        Kind.NOT,
                        solver.mkTerm(
                            Kind.APPLY_UF,
                            inv_temp,
                            al, bl, ah, bh,
                            solver.mkReal(t),
                            solver.mkReal(Fraction(str(n))),
                        ),
                    )
                )

            invariants.append(invariants_inner)
        
        res = solver.checkSat()
        if res.isSat():
            for l in range(rnnModel.num_rnn_layers):
                alpha_l = []
                beta_l = []
                alpha_u = []
                beta_u = []

                for (a_l, b_l), (a_u, b_u) in invariants[l]:
                    a_l = solver.getValue(a_l)
                    b_l = solver.getValue(b_l)
                    a_u = solver.getValue(a_u)
                    b_u = solver.getValue(b_u)
                    alpha_l.append(float(a_l.getRealValue()))
                    beta_l.append(float(b_l.getRealValue()))
                    alpha_u.append(float(a_u.getRealValue()))
                    beta_u.append(float(b_u.getRealValue()))

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
