## RLFAP 

#### Execution command: 
```
python3 main.py [instance] [method]
```

Where [instance] is the name of the instance and [method] is the corresponding name (in uppercase letters) of the algorithm you want to run.

#### for example:
```
python3 main.py 11 FC
python3 main.py 2-f25 MAC
python3 main.py 2-f24 FC-CBJ
python3 main.py 6-w2 MINCONFLICTS
```
There is also the script run_all.py for easier execution of the program if you want to run it for all .txt files without typing them one by one. 
#### This script can be used as follows:
```
python3 run_all.py
```
### File csp.py:

This file contains the functions for reading the .txt files, which are as follows:

1. read_file_var: Reads the variable information.
2. read_file_dom: Reads the domain information.
3. ead_file_ctr: Reads the constraint information.

Additionally, the same file contains the algorithms FC, MAC, FC-CBJ, MINCONFLICTS and dom/wdeg heuristic.

### dom/wdeg heuristc:

The heuristic dom/wdeg prioritizes variables with smaller values and higher weights. The function initializes the weighted_degrees to store the weights of each variable, starting with an initial value of 0. Then, it iterates through the variables in the CSP, excluding those that have already been assigned. For each unassigned variable, it calculates the weighted_degree considering its neighbors and their respective weights. The function computes the ratio of the current domain size to the weighted_degree and updates the best_variable if the ratio is less than the current minimum ratio. Finally, the function returns the variable with the minimum ratio, which the heuristic suggests as the best choice for assignment.

