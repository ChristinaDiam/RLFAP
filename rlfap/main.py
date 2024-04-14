# Christina Diamanti - 1115201800046

# main program for RLFAP
import argparse
import time
from csp import CSP, constraint, backtracking_search, forward_checking, unordered_domain_values, dom_wdeg_heuristic, mac, backjumping_fc, min_conflicts, printconstraints
from collections import defaultdict


#-------------- MAIN ---------------
def main():
    # create an argument parser
    parser = argparse.ArgumentParser(description="using argparse to take arguments")
    
    # add arguments to the parser
    parser.add_argument("instance", type=str, help="instance")
    parser.add_argument("method", type=str, help="method (FC, MAC, FC-CBJ, MINCONFLICTS)")

    # parse the arguments from the command line
    args = parser.parse_args()

    # create file path for .txt files
    file_path_var = 'rlfap_data/var{}.txt'.format(args.instance)
    file_path_dom = 'rlfap_data/dom{}.txt'.format(args.instance)
    file_path_ctr = 'rlfap_data/ctr{}.txt'.format(args.instance)

    var = CSP.read_file_var(file_path_var)

    # separate variables from domains and keep only variables in a list
    variables = []
    for v in var:
        variables.append(v[0])

    domains = CSP.read_file_dom(file_path_dom, var)
    neighbors = CSP.read_file_ctr(file_path_ctr, variables)

    # parse file data to the CSP class (variables, domains, neighbors and constraint)
    my_csp = CSP(variables, domains, neighbors, constraint)


    # checking which method to use
    match args.method:

        case "FC":
            print("Solving instance {}, using {} method and dom/wdeg heuristic.\n".format(args.instance, args.method))
            # start execution time for FC
            start = time.time()

            solution = backtracking_search(
                my_csp, 
                select_unassigned_variable=dom_wdeg_heuristic,
                order_domain_values=unordered_domain_values, 
                inference = forward_checking)
            
            # end execution time for FC
            end = time.time()
           
            execution_time = end-start

            print(solution)

        case "MAC":
            print("Solving instance {}, using {} method and dom/wdeg heuristic.\n".format(args.instance, args.method))
            # start execution time for MAC
            start = time.time()

            solution = backtracking_search(
                my_csp, 
                select_unassigned_variable=dom_wdeg_heuristic, 
                order_domain_values=unordered_domain_values, 
                inference=mac)

            # end execution time for MAC
            end = time.time()
            execution_time = end-start

            print(solution)

        case "FC-CBJ":
            print("Solving instance {}, using {} method and dom/wdeg heuristic.\n".format(args.instance, args.method))
            # start execution time for FC-CBJ
            start = time.time()

            solution = backjumping_fc(
                my_csp, 
                select_unassigned_variable=dom_wdeg_heuristic, 
                order_domain_values=unordered_domain_values, 
                inference=forward_checking)

            # end execution time for FC-CBJ
            end = time.time()
            execution_time = end-start

            print(solution)
        
        case "MINCONFLICTS":
            print("Solving instance {}, using {} method and dom/wdeg heuristic.\n".format(args.instance, args.method))
            # start execution time for MINCONFLICTS
            start = time.time()

            solution = min_conflicts(my_csp)

            # end execution time for MINCONFLICTS
            end = time.time()
            execution_time = end-start

            print("Constraints violated: {}".format(solution[1]))

        case _:
            print("{} doesn't match any of the methods. Try typing: FC, MAC, FC-CBJ or MINCONFLICTS\n".format(args.method))
            start = time.time()
            end = time.time()
            execution_time = end-start


    print("\nVisited Nodes: {}".format(my_csp.nassigns))
    printconstraints()
    print("\nThe code executed in {:.3f} seconds.\n".format(execution_time))

if __name__ == "__main__":
    main()
