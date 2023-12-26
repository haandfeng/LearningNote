# Starvation
Definition: a process is perpetually denied necessary resources to process its work. 
Example:  A system always chooses the process with the shortest file to execute. If there is a constant stream of processes with short files, the process with long file will never be executed.
## vs. Deadlock
Starvation: thread waits indefinitely
Deadlock: circular waiting for resource.
What are the differences between them?
-  Deadlock -> starvation but not vice versa.
-  Starvation may end，but deadlock can’t end without external intervention. 

Deadlock is not always deterministic.

# Deadlock Detection, Deadlock Avoidance, and Deadlock 
Deadlock Detection: To make sure whether there is a deadlock now. 

Deadlock Avoidance:  to ensure the system won’t enter an unsafe state.  The system dynamically considers every request and decides whether it is safe to grant it at this point.

Deadlock Prevention: to ensure that at least one of the necessary conditions for deadlock can never hold.

# Attacking the Mutual Exclusion Condition
No resource were assigned exclusively to a single process.

Problem?
This method is generally impossible, because the mutual-exclusion condition must hold for non-sharable resources. e.g., a printer can not be shared by multiple processes simultaneously. 
# Attacking the Hold and Wait Condition
Require processes to request resources before starting
a process never has to wait for what it needs
e.g. In the dining philosophers’ problem, each philosopher is required to pick up both forks at the same time. If he fails, he has to release the fork(s) (if any) he has acquired. 

Problems?
It is difficult to know required resources at start of run

# Attacking the No Preemption Condition
