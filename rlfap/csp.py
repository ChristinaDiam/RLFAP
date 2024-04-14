# Christina Diamanti 1115201800046

"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)"""
# Edited version from the original

import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend

class CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

        global my_domains, my_variables, my_neighbors
        my_domains = domains
        my_variables = variables
        my_neighbors = neighbors

    # ------------ CUSTOM FUNCTIONS START ------------
    # Functions for reading input files
            
    # function to read and process "var" type files
    def read_file_var(file_path):

        # read the variables from var.txt file and save their variable and domain in a list.
        var = []

        with open(file_path, 'r') as file:
            # read first line to get number of variables
            num_var = file.readline()

            for i in range(int(num_var)):
                # get line content: "var_index" "domain"
                line = file.readline()

                line_split = line.split(" ")

                # Convert each string element to an integer
                int_values = []
                for x in line_split:
                    int_values.append(int(x))

                # Append the list of integers to the 'variables' list
                var.append(int_values)

        return var
    

    # function to read and process "dom" type files
    def read_file_dom(file_path,variables):

        global my_domains 
        my_domains = {}

        # list to save lists with variables for every domain
        dom = []

        with open(file_path, 'r') as file:
            # read first line to get number of domains
            num_dom = file.readline()

            for i in range(int(num_dom)):
                # get line content
                line = file.readline()

                line_split = line.split()

                domain = line_split[0]           # get domain
                numofvariables = line_split[1]   # get number of variables (of domain 'i')

                d = []                           # list to save the variables (of domain 'i')
                for j in range(int(numofvariables)):
                    # add variables to the list
                    d.append(int(line_split[j+2]))

                dom.append(d)

        # matching domains and variables
        for v in variables:
            # v = [variable, variable's domain]
            my_domains[v[0]] = dom[v[1]]

        return my_domains


    # function to read and process "ctr" type files
    def read_file_ctr(file_path, variables):

        global my_constraints, my_neighbors
        my_constraints = {}
        my_neighbors = {}

        for v in range(len(variables)):
        # for v in range(variables table size):
            my_neighbors[v] = []

        
        with open(file_path, 'r') as file:
            # read first line to get number of restrictions
            num_restrictions = file.readline()

            for i in range(int(num_restrictions)):

                # get line content
                line = file.readline()

                # split and save each variable separately (|x-y|>k or |x-y|=k)
                line_split = line.split()
                
                x = int(line_split[0])          # get x
                y = int(line_split[1])          # get y
                k = int(line_split[3])          # get k

                operator = line_split[2]        # get operator "=" or ">"

                my_constraints[(x,y)] = (operator, k)
                my_constraints[(y,x)] = (operator, k)

                my_neighbors[x].append(y)
                my_neighbors[y].append(x)


        # Initialize weight variable
        for c in my_constraints:
            weight[c] = 1 

        for v in variables:
            conf_set[v] = set()
            order[v] = 0

        return my_neighbors
    
    # ------------ CUSTOM FUNCTIONS END --------------
    

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


# (changes: added the last "if condition" and update of weights)
def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True

    # check if the domain of Xi becomes empty after pruning
    if not csp.curr_domains[Xi]:
        # update weights
        weight[(Xi, Xj)] += 1
        weight[(Xj, Xi)] += 1

    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic

def AC3b(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # or queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4

def AC4(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])


# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True



# ------------ CUSTOM FUNCTIONS START ------------


# Initialize constraintsChecked to zero
constraintsChecked = 0
# global weight, conf_set, order
weight = {}
conf_set = {}
order = {}

# evaluates a constraint based on the given parameters.
def constraint(A, a, B, b):
    global constraintsChecked
    constraintsChecked += 1

    # retrieve constraint information from the dictionary
    constraint_info = my_constraints.get((A, B))

    if constraint_info is not None:
        operator = constraint_info[0]  # operator
        k = constraint_info[1]         # constant value

        # check the constraint based on the operator
        if operator == '=':               # "equals" operator
            return abs(a - b) == k
        elif operator == '>':             # "greater than" operator
            return abs(a - b) > k

    # print message if constraint information is not found and return False
    print("Constraint information not found for ({}, {})".format(A, B))
    return False

# help function for printing checks number
def printconstraints():
    print("\nChecks: {}".format(constraintsChecked))


# the dom/wdeg heuristic function
def dom_wdeg_heuristic(assignment, csp):
    # initialize variables
    weighted_degrees = {}       # dictionary to store weighted degrees
    min_ratio = 1e300           # initialize min_ratio to a very large number
    best_variable = 0           # initialize the best variable to 0

    # Calculate weighted degrees
    for variable in csp.variables:
        if variable in assignment:
            continue

        # Initialize sum value to zero
        weighted_degrees[variable] = 0

        # add weight to weighted_degrees variable
        for neighbor in my_neighbors[variable]:
            weighted_degrees[variable] += weight[(variable, neighbor)]

    # select variable with minimum ratio
    for variable in csp.variables:
        if variable in assignment:
            continue

        # find current domain size
        if csp.curr_domains:
            current_domain = csp.curr_domains[variable]
        else:
            current_domain = my_domains[variable]

        # calculate ratio and update best variable
        ratio = len(current_domain) / weighted_degrees[variable]
        if ratio < min_ratio:
            min_ratio = ratio
            best_variable = variable

    # return the variable with the minimum ratio
    return best_variable  


# ------------ CUSTOM FUNCTIONS END --------------

# (changes: added the update of conf_set and weight)
def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""

    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)

            # check if the domain is empty after pruning
            if not csp.curr_domains[B]:

                # update conflict set (conf_set) and weights (weight) due to domain wipeout
                conf_set[B].add(var)    
                weight[(var, B)] += 1
                weight[(B, var)] += 1
                return False
    return True

# (changes: changed AC3b to AC3)
def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)


# The search, proper
def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result


# ------------ CUSTOM FUNCTIONS START ------------

# global counter variable
counter = 1

# backtracking search with CBJ and FC
def fc_cbj(csp, assignment, select_unassigned_variable, order_domain_values, inference, counter, conf_set, order, visited):
    # if all variables are assigned, a solution is found
    if len(assignment) == len(csp.variables):
        return assignment, None

    var = select_unassigned_variable(assignment, csp)
    order[var] = counter
    counter += 1

    for value in order_domain_values(var, assignment, csp):
        if csp.nconflicts(var, value, assignment) == 0:
            # if the value is consistent with the assignment
            csp.assign(var, value, assignment)
            removals = csp.suppose(var, value)

            # attempt to continue the search with the chosen inference mechanism
            if inference(csp, var, value, assignment, removals):
                result, conflict_var = fc_cbj(csp, assignment, select_unassigned_variable, order_domain_values, inference, counter, conf_set, order, visited)
                
                # Solution found
                if result is not None:
                    return result, None
                elif (var in visited) and (var != conflict_var):
                    # Conflict backjumping
                    conf_set[var].clear()
                    visited.discard(var)
                    csp.restore(removals)
                    csp.unassign(var, assignment)
                    return None, conflict_var

            csp.restore(removals)

    # backtrack if no consistent value is found
    csp.unassign(var, assignment)
    visited.add(var)

    conflict_var = None
    max_order = 0

    # update conflict set
    if len(conf_set[var]):
        for c in conf_set[var]:
            # check if the order of the variable is higher than the maximum
            if order[c] > max_order:
                max_order = order[c]
                conflict_var = c

        # update the conflict set and remove the conflict variable
        conf_set[conflict_var].update(conf_set[var])
        conf_set[conflict_var].discard(conflict_var)

    return None, conflict_var


# main function for Backjumping Search with CBJ and FC
def backjumping_fc(csp, select_unassigned_variable=first_unassigned_variable, order_domain_values=unordered_domain_values, inference=forward_checking):
    # variables initialization
    visited = set()
    conf_set = {}
    order = {}
    counter = 0

    for var in csp.variables:
        conf_set[var] = set()
        order[var] = 0

    # only use the first element returned
    result, unused = fc_cbj(csp, {}, select_unassigned_variable, order_domain_values, inference, counter, conf_set, order, visited)
    assert result is None or csp.goal_test(result)

    return result


# ------------ CUSTOM FUNCTIONS END --------------


# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs

# (changes: added a remaining conflicts counter (remaining_conf_count), decreased max_steps)
def min_conflicts(csp, max_steps=1000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}

    # variable to save the count of remaining conflicts after reaching the maximum steps
    remaining_conf_count = 0

    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)

    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)

        if not conflicted:
            return current
        
        # save the number of remaining conflicts on the last iteration
        if i == max_steps-1:     
            remaining_conf_count = len(conflicted)  # get the number of conflicted variables

        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)

    # return 'None' and the count of remaining conflicts (if the maximum steps are reached)
    return None , remaining_conf_count


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))