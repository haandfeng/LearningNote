# Starvation
Definition: a process is perpetually denied necessary resources to process its work. 
Example:  A system always chooses the process with the shortest file to execute. If there is a constant stream of processes with short files, the process with long file will never be executed.
## vs. Deadlock
Starvation: thread waits indefinitely
Deadlock: circular waiting for resource.
What are the differences between them?
-  Deadlock -> starvation but not vice versa.
-  Starvation may end，but deadlock can’t end without external intervention. 

