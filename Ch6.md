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

# Deadlock Detection, Deadlock Avoidance, and Deadlock Prevention 
Deadlock Detection: To make sure whether there is a deadlock now. 

Deadlock Avoidance:  to ensure the system won’t enter an unsafe state.  The system dynamically considers every request and decides whether it is safe to grant it at this point.

Deadlock Prevention: to ensure that at least one of the necessary conditions for deadlock can never hold.

# Deadlock Prevention
## Attacking the Mutual Exclusion Condition
No resource were assigned exclusively to a single process.

Problem?
This method is generally impossible, because the mutual-exclusion condition must hold for non-sharable resources. e.g., a printer can not be shared by multiple processes simultaneously. 
## Attacking the Hold and Wait Condition
Require processes to request resources before starting a process never has to wait for what it needs
e.g. In the dining philosophers’ problem, each philosopher is required to pick up both forks at the same time. If he fails, he has to release the fork(s) (if any) he has acquired. 

Problems?
It is difficult to know required resources at start of run
## Attacking the No Preemption Condition
If a process is holding some resources and requests another resource that cannot be allocated to it, then all resources are released.
Problem?
The method can be applied to resources whose state can be save and restored later, e.g., memory.
It cannot be applied to resources such as printers.

## Attacking the Circular Wait Condition
1. A process is entitled only to a single resource at any moment. 一个时刻只能占用一种资源
2. Impose a total ordering of all resource types and to require that each process requests resources in an increasing order of enumeration.
# Other Issues:
## Two-Phase Locking
![[Pasted image 20231226214756.png]]
Similar to requesting all resources at once
## Non-resource Deadlocks
![[Pasted image 20231226215006.png]]

# Memory Management	
Ideally programmers want memory that is 
- large
- fast
- non volatile

Memory hierarchy 
- small amount of fast, expensive memory – cache 
- some medium-speed, medium price main memory
- gigabytes of slow, cheap disk storage

Memory manager handles the memory hierarchy

## Multiprogramming with Fixed Partitions
![[Pasted image 20231226220711.png]]

## Swapping & Virtual Memory
Two approaches to overcome the limitation of memory: 
- Swapping puts a process back and forth in memory and on the disk.
- Virtual memory allows programs to run even when they are only partially in main memory.
## Swapping

![[Pasted image 20231226221141.png]]

Memory allocation changes as processes come into memory and leave memory.

When swapping creates multiple holes in memory, memory compaction can be used to combine them into a big one by moving all processes together.

## Memory Management with Bit Maps & List
![[Pasted image 20231226221739.png]]
![[Pasted image 20231226221814.png]]

## Memory Management with Linked Lists
