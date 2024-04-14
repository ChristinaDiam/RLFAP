# Christina Diamanti - 1115201800046

# helpful script to run all different instances one after the other

import subprocess

# list with all instances name
all_inst = ["2-f24", "2-f25", "3-f10", "3-f11", "6-w2", "7-w1-f4", "7-w1-f5", "8-f10", "8-f11", "11", "14-f27", "14-f28"]

# list with all algorithm names
all_alg = ["FC", "MAC", "FC-CBJ", "MINCONFLICTS"]

# run loop through all different instances
print("Running all instances...")

for i in range(len(all_inst)):
    
    # use different algorithms for each instance
    for j in range(len(all_alg)):
        print(">> {} | {}\n".format(all_inst[i], all_alg[j]))

        # execute python script for each instance and algorithm
        subprocess.run(["python3", "main.py", all_inst[i], all_alg[j]])
        
