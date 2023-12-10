from timeit import default_timer as timer
from typing import List, Union

import numpy as np
import tensorflow as tf

from RNN.MarabouRnnModel import MARABOU_TIMEOUT
from RNN.MarabouRnnModel import RnnMarabouModel
from RNN.ICELearner import ICELearner
from maraboupy import MarabouCore

large = 5000.0
small = 10 ** -4
TOLERANCE_VALUE = 0.01
ALPHA_IMPROVE_EACH_ITERATION = 5


def marabou_solve_negate_eq(query, debug=False, print_vars=False, return_vars=False):
    '''
    Run marabou solver
    :param query: query to execute
    :param debug: if True printing all of the query equations
    :return: True if UNSAT (no valid assignment), False otherwise
    '''
    verbose = 0
    #if debug:
    # query.dump()

    # print("{}: start query".format(str(datetime.now()).split(".")[0]), flush=True)
    vars1, stats1 = MarabouCore.solve(query, "", MARABOU_TIMEOUT, verbose)
    # print("{}: finish query".format(str(datetime.now()).split(".")[0]), flush=True)
    if stats1.hasTimedOut():
        print("Marabou has timed out")
        raise TimeoutError()
    if len(vars1) > 0:
        if print_vars:
            print("SAT")
            # print(vars1)
            # query.dump()
            # exit(1)
        res = False
    else:
        # print("UNSAT")
        res = True

    if return_vars:
        # if len(vars1) > 0:
        #     print(vars1)
        return res, vars1
    else:
        return res


def negate_equation(eq: MarabouCore.Equation):
    '''
    negates the equation
    :param eq: equation
    :return: new equation which is exactly (not eq)
    '''
    assert eq is not None
    not_eq = MarabouCore.Equation(eq)
    if eq.getType() == MarabouCore.Equation.GE:
        not_eq.setType(MarabouCore.Equation.LE)
        not_eq.setScalar(eq.getScalar() - small)
    elif eq.getType() == MarabouCore.Equation.LE:
        not_eq.setType(MarabouCore.Equation.GE)
        not_eq.setScalar(eq.getScalar() + small)
    elif eq.setType(MarabouCore.Equation.EQ):
        raise NotImplementedError("can't negate equal equations")
    else:
        raise NotImplementedError("got {} type which is not implemented".format(eq.getType()))
    return not_eq


def add_loop_indices_equations(network, loop_indices):
    '''
    Adds to the network equations that make all loop variabels to be equal
    :param network: marabou quert that the equations will be appended
    :param loop_indices: variables that needs to be equal
    :return: None
    '''
    # Make sure all the iterators are in the same iteration, we create every equation twice
    step_loop_eq = []
    # for idx in loop_indices:
    if isinstance(loop_indices, list):
        loop_indices = [i for ls in loop_indices for i in ls]
    idx = loop_indices[0]
    for idx2 in loop_indices[1:]:
        if idx < idx2:
            temp_eq = MarabouCore.Equation()
            temp_eq.addAddend(1, idx)
            temp_eq.addAddend(-1, idx2)
            # step_loop_eq.append(temp_eq)
            network.addEquation(temp_eq)


def create_invariant_equations(loop_indices: List[int],
                               invariant_eq: Union[MarabouCore.Equation, List[MarabouCore.Equation]]):
    '''
    create the equations needed to prove using induction from the invariant_eq
    :param loop_indices: List of loop variables (i's), which is the first variable for an RNN cell
    :param invariant_eq: the invariant we want to prove, might be a list
    :return: [base equations], [step equations]
    '''

    def create_induction_hypothesis_from_invariant_eq():
        '''
        for example our invariant is that s_i f <= i, the induction hypothesis will be s_i-1 f <= i-1
        :return: the induction hypothesis
        '''
        scalar_diff = 0
        hypothesis_eq = []

        cur_temp_eq = MarabouCore.Equation(invariant_eq.getType())
        for addend in invariant_eq.getAddends():
            # if for example we have s_i f - 2*i <= 0 we want s_i-1 f - 2*(i-1) <= 0 <--> s_i-1 f -2i <= -2
            if addend.getVariable() in loop_indices:
                scalar_diff = addend.getCoefficient()
            # here we change s_i f to s_i-1 f
            if addend.getVariable() in rnn_output_indices:
                cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable() - 2)
            else:
                cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable())
        cur_temp_eq.setScalar(invariant_eq.getScalar() + scalar_diff)
        hypothesis_eq.append(cur_temp_eq)
        return hypothesis_eq

    if isinstance(invariant_eq, list):
        # recursion
        base = []
        step = []
        hypothesis = []
        for eq in invariant_eq:
            assert isinstance(eq, MarabouCore.Equation)
            b, s, h = create_invariant_equations(loop_indices, eq)
            base += b
            step += s
            hypothesis += h
        return base, step, hypothesis

    if isinstance(loop_indices[0], list):
        # TODO: delete isinstance
        assert False
        rnn_input_indices = [idx + 1 for ls in loop_indices for idx in ls]
        rnn_output_indices = [idx + 3 for ls in loop_indices for idx in ls]
    else:
        rnn_input_indices = [idx + 1 for idx in loop_indices]
        rnn_output_indices = [idx + 3 for idx in loop_indices]

    induction_step = negate_equation(invariant_eq)

    # make sure i == 0 (for induction base)
    loop_equations = []
    if isinstance(loop_indices[0], list):
        loop_indices = [i for ls in loop_indices for i in ls]
    for i in loop_indices:
        loop_eq = MarabouCore.Equation()
        loop_eq.addAddend(1, i)
        loop_eq.setScalar(0)
        loop_equations.append(loop_eq)

    # s_i-1 f == 0
    zero_rnn_hidden = []
    for idx in rnn_input_indices:
        base_hypothesis = MarabouCore.Equation()
        base_hypothesis.addAddend(1, idx)
        base_hypothesis.setScalar(0)
        zero_rnn_hidden.append(base_hypothesis)

    step_loop_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    step_loop_eq.addAddend(1, loop_indices[0])
    step_loop_eq.setScalar(1)
    # step_loop_eq.append(step_loop_eq_more_1)

    induction_base_equations = [induction_step] + loop_equations + zero_rnn_hidden

    # The way it is now
    induction_hypothesis = create_induction_hypothesis_from_invariant_eq()
    induction_step_equations = [step_loop_eq] + [induction_step]

    # # The way it was
    # induction_hypothesis = create_induction_hypothesis_from_invariant_eq() + [step_loop_eq]
    # induction_step_equations = [induction_step]

    # induction_step_equations = induction_step + step_loop_eq

    return induction_base_equations, induction_step_equations, induction_hypothesis


def prove_invariant_multi(network, rnn_start_idxs: List[int],
                          invariant_equations: Union[List[MarabouCore.Equation], List[List[MarabouCore.Equation]]],
                          return_vars=False):
    '''
    Prove invariants where we need to assume multiple assumptions and conclude from them.
    For each of the invariant_equations creating 3 sets: base_equations, hyptosis_equations, step_equations
    First proving on each of the base equations seperatly, Then assuming all the hyptosis equations and proving
    on the step_equations set by set
    At the end of the function the network will be exactly the same as before
    :param network:
    :param rnn_start_idxs:
    :param invariant_equations:
    :return:
    '''

    # TODO: Get rid of this
    if isinstance(rnn_start_idxs[0], list) and len(rnn_start_idxs) == 1:
        rnn_start_idxs = rnn_start_idxs[0]

    proved_invariants = [None] * len(invariant_equations)
    base_eq = []
    step_eq = []  # this needs to be a list of lists, each time we work on all equations of a list
    hypothesis_eq = []
    assignments = []

    rnn_node_ids = list(range(len(rnn_start_idxs))) * 2
    pos_cex = []
    imp_cex = []

    for i in range(len(invariant_equations)):
        cur_base_eq, cur_step_eq, cur_hypothesis_eq = create_invariant_equations(rnn_start_idxs, invariant_equations[i])
        base_eq.append(cur_base_eq)
        step_eq.append(cur_step_eq)
        hypothesis_eq += cur_hypothesis_eq

    # first prove base case for all equations
    for i, ls_eq in enumerate(base_eq):
        for eq in ls_eq:
            #eq.dump()
            network.addEquation(eq)
        marabou_result, sat_vars = marabou_solve_negate_eq(network, print_vars=True, return_vars=True)
        assignments.append(sat_vars)
        #network.dump()

        for eq in ls_eq:
            network.removeEquation(eq)

        proved_invariants[i] = marabou_result
        if not marabou_result:
            print("induction base fail, on invariant {}".format(i))
            # I'm just going with rnn_start_idx[i] + 3 blindly here, as the
            # value of the memory node.
            # print("here!!!", rnn_start_idxs[rnn_node_ids[i]])
            # print(sat_vars[rnn_start_idxs[rnn_node_ids[i]]])
            pos_cex.append(
                (rnn_node_ids[i],
                 sat_vars[rnn_start_idxs[rnn_node_ids[i]]], # should be zero
                 sat_vars[rnn_start_idxs[rnn_node_ids[i]] + 3])
            )
            # Formulate positive counterexample.
            # for eq in ls_eq:
            #     eq.dump()
            # assert False

    if not all(proved_invariants):
        if return_vars:
            return proved_invariants, assignments, pos_cex, imp_cex
        else:
            return proved_invariants

    proved_invariants = [False] * len(invariant_equations)
    assignments = []
    # print("proved induction base for all invariants")

    # add all hypothesis equations
    # print("adding hypothesis_eq")
    for eq in hypothesis_eq:
        #eq.dump()
        network.addEquation(eq)

    hypothesis_fail = False
    # TODO: DEBUG
    # marabou_result, cur_vars = marabou_solve_negate_eq(network, print_vars=False, return_vars=True)
    # if marabou_result:
    #     # UNSAT Conflict in the hypothesis
    #     assert False
    #     proved_invariants = [False] * len(proved_invariants)
    #     hypothesis_fail = True

    if not hypothesis_fail:
        for i, steq_eq_ls in enumerate(step_eq):
            for eq in steq_eq_ls:
                network.addEquation(eq)

            marabou_result, cur_vars = marabou_solve_negate_eq(network, print_vars=True, return_vars=True)
            assignments.append(cur_vars)
            # print("Querying for induction step: {}".format(marabou_result))
            # network.dump()

            # proved_invariants[i] = marabou_result
            if not marabou_result:
                proved_invariants[i] = False
                # rnn_start_idx[i] + 1: lhs implication
                # rnn_start_idx[i] + 3: rhs implication
                imp_cex.append(
                    (rnn_node_ids[i],
                     cur_vars[rnn_start_idxs[rnn_node_ids[i]]],
                    (cur_vars[rnn_start_idxs[rnn_node_ids[i]] + 1],
                     cur_vars[rnn_start_idxs[rnn_node_ids[i]] + 3]))
                )
                #print(imp_cex)
            else:
                proved_invariants[i] = True

            for eq in steq_eq_ls:
                network.removeEquation(eq)
    for eq in hypothesis_eq:
        network.removeEquation(eq)

    if return_vars:
        return proved_invariants, assignments, pos_cex, imp_cex
    else:
        return proved_invariants


def alphas_to_equations(rnn_start_idxs, rnn_output_idxs, initial_values, inv_type, alphas):
    '''
    Create a list of marabou equations, acording to the template: \\alpha*i \\le R_i OR \\alpha*i \\ge R_i
    For parameter look at alpha_to_equation, this is just a syntax sugar to remove the loop from outer functions
    :return: List of marabou equations
    '''
    assert len(rnn_start_idxs) == len(rnn_output_idxs)
    assert len(rnn_start_idxs) == len(initial_values)
    assert len(rnn_start_idxs) == len(alphas)
    invariants = []
    if not isinstance(inv_type, list):
        inv_type = [inv_type] * len(rnn_start_idxs)

    for i in range(len(rnn_start_idxs)):
        invariants.append(
            alpha_to_equation(rnn_start_idxs[i], rnn_output_idxs[i], initial_values[i], alphas[i], inv_type[i]))

    return invariants


def alpha_to_equation(start_idx, output_idx, initial_val, new_alpha, inv_type):
    '''
    Create an invariant equation according to the simple template \\alpha*i \\le R_i OR \\alpha*i \\ge R_i
    :param start_idx: index of the rnn iterator (i)
    :param output_idx: index of R_i
    :param initial_val: If inv_type = GE the max value of R_1 if inv_type = LE the min of R_1
    :param new_alpha: alpha to use
    :param inv_type: Marabou.Equation.GE / Marabou.Equation.LE
    :return: marabou equation
    '''
    # Need the invariant from both side because they are all depndent in each other
    invariant_equation = MarabouCore.Equation(inv_type)
    invariant_equation.addAddend(1, output_idx)  # b_i
    if inv_type == MarabouCore.Equation.LE:
        ge_better = -1
    else:
        # TODO: I don't like this either
        ge_better = 1
        # ge_better = -1

    invariant_equation.addAddend(new_alpha * ge_better, start_idx)  # i
    # TODO: Why isn't it ge_better * initial_val? if it's LE we want:
    # not ( alpha * i + beta \le R ) \iff -alpha * i - beta > R
    invariant_equation.setScalar(initial_val)
    # invariant_equation.dump()
    return invariant_equation


def double_list(ls):
    '''
    create two items from each item in the list
    i.e. if the input is: [1,2,3] the output is: [1,1,2,2,3,3]
    '''
    import copy
    new_ls = []
    for i in range(len(ls)):
        new_ls += [copy.deepcopy(ls[i]), copy.deepcopy(ls[i])]
    return new_ls


def invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, return_vars=False):
    '''
    Creates a function that can verify invariants accodring to the network and rnn indcies
    :param network: Marabou format of a network
    :param rnn_start_idxs: Indcies of the network where RNN cells start
    :param rnn_output_idxs: Output indcies of RNN cells in the network
    :return: A pointer to a function that given a list of equations checks if they stand or not
    '''

    def invariant_oracle(equations_to_verify):
        return prove_invariant_multi(network, rnn_start_idxs, equations_to_verify, return_vars=return_vars)

    return invariant_oracle


def property_oracle_generator(network, property_equations):
    def property_oracle(invariant_equations, last_rnn_indices,
                        return_vars=False):

        for eq in invariant_equations:
            if eq is not None:
                network.addEquation(eq)

        # TODO: This is only for debug
        # before we prove the property, make sure the invariants does not contradict each other, expect SAT from marabou
        # assert not marabou_solve_negate_eq(network, False, False)

        for eq in property_equations:
            if eq is not None:
                network.addEquation(eq)
        res, sat_vars = marabou_solve_negate_eq(network, False, print_vars=True, return_vars=True)
        #network.dump()
        if res:
            pass
        # Get negative counterexamples
        neg_cex = []
        if not res:
            for i, rnn_idx in enumerate(last_rnn_indices):
                neg_cex.append(
                    (i,
                    sat_vars[rnn_idx],
                    sat_vars[rnn_idx + 3]) # Again, this weird +3 thing.
                )
        
        for eq in invariant_equations + property_equations:
            if eq is not None:
                network.removeEquation(eq)

        if return_vars:
            return res, sat_vars, neg_cex
        else:
            return res

    return property_oracle


def prove_multidim_property(rnnModel: RnnMarabouModel, property_equations, algorithm, return_alphas=False,
                            number_of_steps=5000, debug=False, return_queries_stats=False, stats=None):
    network = rnnModel.network
    rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(rnn_layer=None)
    add_loop_indices_equations(network, rnn_start_idxs)
    # invariant_oracle = invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, return_vars=True)
    property_oracle = property_oracle_generator(network, property_equations)

    res = False
    unsat = False
    if stats is None:
        stats = {}
    if 'invariant_times' not in stats:
        stats['invariant_times'] = []
        stats['property_times'] = []
        stats['step_times'] = []

    for i in range(number_of_steps):
        invariant_results = []
        proved_equations = []

        # Obtain an invariant from CVC5 via SyGuS

        start_step = timer()

        learner = ICELearner()
        learner_res = learner.do_round(rnnModel, algorithm)
        if not learner_res:
            res = False
            print("unrealizable")
            for l in range(rnnModel.num_rnn_layers):
                print("Layer", l)
                print("pos:", algorithm.alphas_algorithm_per_layer[l].pos_cex)
                print("neg:", algorithm.alphas_algorithm_per_layer[l].neg_cex)
                print("imp:", algorithm.alphas_algorithm_per_layer[l].imp_cex)
            break

        # Test case
        # if i == 0:
    
        # else:
        #     learner = ICELearner()
        #     learner_res = learner.do_round(rnnModel, algorithm)
        #     if not learner_res:
        #         res = False
        #         break

        # # # Another invariant? (yes)
        # algorithm.alphas_algorithm_per_layer[0].ice_alphas = [0.0, 0.0, 0.0, 0.0]
        # algorithm.alphas_algorithm_per_layer[0].ice_betas = [0.0, 0.0, 0.0, 1.0]
        # algorithm.alphas_algorithm_per_layer[0].ice_is_upper = [False, False, True, True]
        # algorithm.alphas_algorithm_per_layer[1].ice_alphas = [0.0, 0.0, 1.0, 0.0]
        # algorithm.alphas_algorithm_per_layer[1].ice_betas = [0.0, 0.0, 1.0, 1.0]
        # algorithm.alphas_algorithm_per_layer[1].ice_is_upper = [False, False, True, True]

        # for l in range(rnnModel.num_rnn_layers):
        #     print(algorithm.alphas_algorithm_per_layer[l].ice_alphas)
        #     print(algorithm.alphas_algorithm_per_layer[l].ice_betas)
        #     print(algorithm.alphas_algorithm_per_layer[l].ice_is_upper)

        # if i > 0:
        #     exit(1)
        # # Correct case
        # algorithm.alphas_algorithm_per_layer[0].ice_alphas = [0.0, 0.0, 0.0, 0.18445917338170847]
        # algorithm.alphas_algorithm_per_layer[0].ice_betas = [-0.0002, -0.0002, 0.019999999999868123, 0.19543422519421938]
        # algorithm.alphas_algorithm_per_layer[0].ice_is_upper = [False, False, True, True]
        # algorithm.alphas_algorithm_per_layer[1].ice_alphas = [0.277017253090813, 0.19006014976501465, 0.5331825524177967, 0.19187598175435117]
        # algorithm.alphas_algorithm_per_layer[1].ice_betas = [0.8006336043514816, -0.570380449295044, 0.8597897887226245, 0.19287598175435117]
        # algorithm.alphas_algorithm_per_layer[1].ice_is_upper = [False, False, True, True]

        end_step = timer()
        print("Time taken in Sygus", end_step - start_step)
        stats['step_times'].append(end_step - start_step)

        ice_test = True

        invariant_okay = True
        for l in range(rnnModel.num_rnn_layers):
            #print("layer: ", l)
            # start_step = timer()
            if hasattr(algorithm, 'support_multi_layer') and algorithm.support_multi_layer:
                rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(rnn_layer=l)
                if ice_test:
                    equations = algorithm.get_ice_equations(layer_idx=l)
                else:
                    equations = algorithm.get_equations(layer_idx=l)
                    alphas = algorithm.get_alphas(layer_idx=l)
                    print(alphas)
            else:
                rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(rnn_layer=None)
                equations = algorithm.get_equations()
            # end_step = timer()
            # stats['step_times'].append(end_step - start_step)
            if equations is None: # This is the case where no invariants in the concept
                                  # class are found.
                unsat = True
                break
            start_invariant = timer()
            invariant_oracle = invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, return_vars=True)
            layer_invariant_results, sat_vars, pos_cex, imp_cex = invariant_oracle(equations)
            algorithm.alphas_algorithm_per_layer[l]\
                     .pos_cex.extend(pos_cex)
            algorithm.alphas_algorithm_per_layer[l]\
                     .imp_cex.extend(imp_cex)
            end_invariant = timer()
            stats['invariant_times'].append(end_invariant - start_invariant)
            invariant_results += layer_invariant_results
            if all(layer_invariant_results):
                # print('proved an invariant: {}'.format(algorithm.get_alphas(l)))
                if ice_test:
                    if hasattr(algorithm, 'proved_ice_invariant'):
                        algorithm.proved_ice_invariant(l, equations=equations)
                else:
                    if hasattr(algorithm, 'proved_invariant'):
                        algorithm.proved_invariant(l, equations=equations)
                # When we prove layer l+1 we need to proved equations on layer l
                eqs = [eq for eq_ls in equations for eq in eq_ls] if isinstance(equations[0],
                                                                                              list) else equations
                proved_equations += eqs
                # print(proved_equations)
                for eq in eqs:
                    network.addEquation(eq)
            else:
                print('layer: {}, fail in one (or more) invariants: {}'.format(l, invariant_results))
                invariant_okay = False
                #break
                # Still propagate the invariant forward.
                if ice_test:
                    if hasattr(algorithm, 'proved_ice_invariant'):
                        algorithm.proved_ice_invariant(l, equations=equations)
                else:
                    if hasattr(algorithm, 'proved_invariant'):
                        algorithm.proved_invariant(l, equations=equations)
                # When we prove layer l+1 we need to proved equations on layer l
                eqs = [eq for eq_ls in equations for eq in eq_ls] if isinstance(equations[0],
                                                                                              list) else equations
                proved_equations += eqs
                for eq in eqs:
                    network.addEquation(eq)

        if unsat:
            res = False
            break
        for eq in proved_equations:
            network.removeEquation(eq)
        # print(invariant_results)
        if all(invariant_results) and invariant_okay:
            # print('proved an invariant: {}'.format(algorithm.get_alphas()))
            start_property = timer()
            last_rnn_idxs, _ = rnnModel.get_start_end_idxs(rnn_layer=rnnModel.num_rnn_layers - 1)
            prop_res, sat_vars, neg_cex = property_oracle(
                proved_equations, 
                last_rnn_idxs,
                return_vars=True
            )
            if neg_cex != []:
                algorithm.alphas_algorithm_per_layer[l]\
                            .neg_cex.append(neg_cex)
            end_property = timer()
            stats['property_times'].append(end_property - start_property)
            if prop_res and invariant_okay:
                print("proved property after {} iterations, using the following invariants:".format(i + 1))
                # Test case
                num_pos, num_neg, num_imp = 0, 0, 0
                for l in range(rnnModel.num_rnn_layers):
                    print(algorithm.alphas_algorithm_per_layer[l].ice_alphas)
                    print(algorithm.alphas_algorithm_per_layer[l].ice_betas)
                    print(algorithm.alphas_algorithm_per_layer[l].ice_is_upper)
                    num_pos += len(algorithm.alphas_algorithm_per_layer[l].pos_cex)
                    num_imp += len(algorithm.alphas_algorithm_per_layer[l].imp_cex)
                    num_neg += len(algorithm.alphas_algorithm_per_layer[l].neg_cex)
                num_total = num_pos + num_neg + num_imp
                print("and {} counterexamples: ({} +) ({} -) ({} * => *)".format(num_total, num_pos, num_neg, num_imp))
                res = True
                break
            else:
                print("failed to prove property at iteration", i + 1)
                res = False
        else:
            print('fail in one (or more) invariants:', invariant_results)
            res = False
        

        #  print progress for debug
        if debug:
            if i > 0 and i % 300 == 0:
                print('iteration {}, alphas: {}'.format(i, algorithm.get_alphas()))

    if debug:
        if len(stats['property_times']) > 0:
            # print("did {} invariant queries that took on avg: {}, and {} property, that took: {} on avg".format(
            #     len(stats['invariant_times']), sum(stats['invariant_times']) / len(stats['invariant_times']), len(stats['property_times']),
            #                           sum(stats['property_times']) / len(stats['property_times'])))
            pass
        else:
            avg_inv_time = sum(stats['invariant_times']) / len(stats['invariant_times']) if len(stats['invariant_times']) > 0 else 0
            print("{}\t{} invariant queries that took on avg: {}, and {} property".format(
                'SUCCESS' if res else 'FAIL', len(stats['invariant_times']), avg_inv_time, len(stats['property_times'])))
    queries_stats = {}
    if return_queries_stats:
        safe_percentile = lambda func, x: func(x) if len(x) > 0 else 0
        queries_stats['property_times'] = {'avg': safe_percentile(np.mean, stats['property_times']),
                                           'median': safe_percentile(np.median, stats['property_times']), 'raw': stats['property_times']}
        queries_stats['invariant_times'] = {'avg': safe_percentile(np.mean, stats['invariant_times']),
                                            'median': safe_percentile(np.median, stats['invariant_times']),
                                            'raw': stats['invariant_times']}
        queries_stats['step_times'] = {'avg': safe_percentile(np.mean, stats['step_times']),
                                       'median': safe_percentile(np.median, stats['step_times']), 'raw': stats['step_times']}
        queries_stats['step_queries'] = len(stats['step_times'])
        queries_stats['property_queries'] = len(stats['property_times'])
        queries_stats['invariant_queries'] = len(stats['invariant_times'])
        queries_stats['number_of_updates'] = i + 1  # one based counting
        queries_stats['algorithm'] = []
        if hasattr(algorithm, 'get_stats'):
            queries_stats['algorithm'] = algorithm.get_stats()
    if not return_alphas:
        if not return_queries_stats:
            return res
        if return_queries_stats:
            return res, queries_stats
    else:
        if not return_queries_stats:
            return res, algorithm.get_alphas()
        if return_queries_stats:
            return res, algorithm.get_alphas(), queries_stats
        
def calc_min_max_by_radius(x, radius):
    '''
    :param x: base_vector (input vector that we want to find a ball around it), need to be valid shape for the model
    :param radius: determines the limit of the inputs around the base_vector, non negative number
    :return: xlim -  list of tuples each tuple is (lower_bound, upper_bound)
    '''
    assert radius >= 0
    xlim = []
    for val in x:
        if val > 0:
            xlim.append((val * (1 - radius), val * (1 + radius)))
        elif val < 0:
            xlim.append((val * (1 + radius), val * (1 - radius)))
        else:
            xlim.append((-radius, radius))
    return xlim


def get_output_vector(h5_file_path: str, x: list, n_iterations: int):
    '''
    predict on the model using x (i.e. the matrix will be of shape (1, n_iterations, len(x))
    :param h5_file_path: path to the model file
    :param x: list of values
    :param n_iterations: number of iteration to create
    :return: ndarray, shape is according to the model output
    '''
    model = tf.keras.models.load_model(h5_file_path)
    tensor = np.repeat(np.array(x)[None, None, :], n_iterations, axis=1)
    return model.predict(tensor)


def get_out_idx(x, n_iterations, h5_file_path, other_index_func=lambda vec: np.argmin(vec)):
    # TODO:Change name to get_out_idx_keras
    '''
    Calcuate the output vector of h5_file for n_iterations repetations of x vector
    :param x: input vector
    :param n_iterations: how many times to repeat x
    :param h5_file_path: model
    :param other_index_func: function to pointer that gets the out vector (1d) and returns the other inedx
    :return: max_idx, other_idx (None if they are the same)
    '''
    out = np.squeeze(get_output_vector(h5_file_path, x, n_iterations))
    other_idx = other_index_func(out)  # np.argsort(out)[-2]
    y_idx_max = np.argmax(out)
    # assert np.argmax(out) == np.argsort(out)[-1]
    # print(y_idx_max, other_idx)
    if y_idx_max == other_idx:
        # This means all the enteris in the out vector are equal...
        return None, None
    return y_idx_max, other_idx


def adversarial_query(x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str, algorithm_ptr,
                      n_iterations=10, steps_num=5000):
    '''
    Query marabou with adversarial query
    :param x: base_vector (input vector that we want to find a ball around it)
    :param radius: determines the limit of the inputs around the base_vector
    :param y_idx_max: max index in the output layer
    :param other_idx: which index to compare max idx
    :param h5_file_path: path to keras model which we will check on
    :param algorithm_ptr: TODO
    :param n_iterations: number of iterations to run
    :return: True / False, and queries_stats
    '''

    if y_idx_max is None or other_idx is None:
        y_idx_max, other_idx = get_out_idx(x, n_iterations, h5_file_path)
        if y_idx_max == other_idx or y_idx_max is None or other_idx is None:
            # This means all the enteris in the out vector are equal...
            return False, None, None

    start_initialize_query = timer()
    xlim = calc_min_max_by_radius(x, radius)
    rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
    rnn_model.set_input_bounds(xlim)


    # output[y_idx_max] >= output[0] <-> output[y_idx_max] - output[0] >= 0, before feeding marabou we negate this
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(-1, rnn_model.output_idx[other_idx])
    adv_eq.addAddend(1, rnn_model.output_idx[y_idx_max])
    adv_eq.setScalar(0)

    time_eq = MarabouCore.Equation()
    time_eq.addAddend(1, rnn_model.get_start_end_idxs(0)[0][0])
    time_eq.setScalar(n_iterations - 1) #see if this is okay
    end_initialize_query = timer()

    start_initial_alg = timer()
    algorithm = algorithm_ptr(rnn_model, xlim)
    end_initial_alg = timer()
    # rnn_model.network.dump()

    res, queries_stats = prove_multidim_property(rnn_model, [negate_equation(adv_eq), time_eq], algorithm, debug=1,
                                                 return_queries_stats=True, number_of_steps=steps_num)
    if queries_stats:
        step_times = queries_stats['step_times']['raw']
        step_times.insert(0, end_initial_alg - start_initial_alg)
        queries_stats['step_times'] = {'avg': np.mean(step_times), 'median': np.median(step_times), 'raw': step_times}
        queries_stats['step_queries'] = len(step_times)
        queries_stats['query_initialize'] = end_initialize_query - start_initialize_query

    # if 'invariant_queries' in queries_stats and 'property_queries' in queries_stats and \
    #         queries_stats['property_queries'] != queries_stats['invariant_queries']:
    #     print("What happened?\n", x)
    del rnn_model
    return res, queries_stats, algorithm.alpha_history
