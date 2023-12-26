# Computer Deadlock
## What is deadlock?
A set of processes is in a deadlock state when every process in the set is waiting for a resource that can only be released by another process in the set.
![[Pasted image 20231226183607.png]]

### Resources
A resource is anything that can be acquired, used, and released over the course of time.  

Sequence of events required to use a resource
1. request the resource
2. use the resource
3. release the resource

Must wait if request is denied
- requesting process may be blocked
- may fail with error code

Potential deadlocks that involve Preempt able resources can usually be resolved by reallocating resources from one process to another.
#### Preempt able resources
can be taken away from a process with no ill effects (e.g. memory)
#### Non preempt able resources
will cause the process to fail if taken away (e.g. CD recorder)

## Conditions for Deadlocks
1. Mutual exclusion
Resources are held by  processes in a non-sharable (exclusive) mode.
 
2. Hold and Wait
A process holds a resource while waiting for another resource.
 
3. No Preemption
There is only voluntary release of a resource - nobody else can make a process give up a resource.
 
4. Circular Wait
Process A waits for Process B waits for Process C .... waits for Process A.

**ALL of these four conditions must happen simultaneously for a deadlock to occur.**


## Resource-Allocation Graph
If graph contains no cycles => no deadlock.
If graph contains a cycle =>
- if only one instance per resource type, then deadlock.
- if several instances per resource type, possibility of deadlock.

## The Ostrich Algorithm
Pretend that there is no problem
Reasonable if 
1. deadlocks occur very rarely 
2. cost of prevention is high

## Detection with One Resource of Each Type  
画图就好检查有没有圈，每种类型资源只有一个的情况
![[Pasted image 20231226194839.png]]
算法，第二段是算法本质
![[Pasted image 20231226195134.png]]



## Detection with Multiple Resources of Each Type  
类似于银行家算法，看ppt26。简单来说就是比对现有资源A每列是不是大于需求资源R的某行每列，如果大于了，就分配资源，然后释放资源，把C对应的某行和A相加
![[Pasted image 20231226195923.png]]

## Recovery from Deadlock
### Recovery through preemption
- take a resource from some other processes
- depends on the nature of the resource
### Recovery through rollback
- checkpoint a process periodically
- use this saved state 
- restart the process if it is found deadlocked

### Recovery through killing processes
- crudest but simplest way to break a deadlock
- kill one of the processes in the deadlock cycle
- the other processes get its resources 
- choose process that can be rerun from the beginning