# Inter Process Communication IPC
问题：
Race conditions: situations in which several processes access shared data and the final result depends on the order of operations.
With increasing parallelism due to increasing number of cores, race condition are becoming more common.
解决办法：
Key idea to avoid race condition：prohibit more than one process from reading and writing the shared data at the same time.
Four conditions to support a good solution
1. No two processes may be simultaneously in critical region. 
2. No assumption made about speeds or numbers of CPUs
3. No process running outside its critical region may block another process
4. No process must wait forever to enter its critical region
![[Pasted image 20231224111404.png]]


# Hardware solution Disabling Interrupts

The CPU is only switch from process to process when clock or other interrupts happen; Hence, by disabling all interrupts, the CPU will not be switched to another process.

However, it is unwise to allow user processes to disable interrupts.
- One thread may never turn on interrupt;
- Problem still exist for multiprocessor systems.



# Lock Variable
```c++ 
  shared int lock = 0;
	/* entry_code: execute before entering critical section */
	while (lock != 0) ; // do nothing
	lock = 1;
	- critical section -
	/* exit_code: execute after leaving critical section */
	lock = 0;


```
问题：If a context switch occurs after one process executes the while statement, but before setting lock = 1, then two (or more) processes may be able to enter their critical sections at the same time.  while 和 lock 语句的操作分开了 

# Strict Alternation
![[Pasted image 20231223204745.png]]
问题：Since the processes must strictly alternate entering their critical sections, a process wanting to enter its critical section twice will be blocked until the other process decides to enter (and leave) its critical section.（同一个不能连续两次访问 critical_region，必须交替访问 ）

# Peterson’s Solution
互相谦让的算法  没有遵守让权等待
This solution satisfies all 4 properties of a good solution. Unfortunately, this solution involves busy waiting in the while loop. 
![[Pasted image 20231223205046.png]]


# Hardware solution: Test-and-Set Locks (TSL)
The hardware must support a special instruction, TSL, which does two things in a single atomic action:
1. copy a value in memory (flag) to a CPU register
2. set flag to 1.
![[Pasted image 20231224111737.png]]
问题: 不满足让权等待的原则，会导致忙等的问题出现
# Busy waiting的问题
BUSY-WAITING：a process executing the entry code will sit in a tight loop using up CPU cycles, testing some condition over and over, until it becomes true.

Busy-waiting may lead to the priority-inversion problem .
![[Pasted image 20231224112016.png]]

# Semaphores 
A SEMAPHORE, S, is a structure consisting of two parts:
    (a) an integer counter, COUNT
    (b) a queue of pids of blocked processes, Q

There are two operations on semaphores, UP and DOWN (PV).  These operations must be executed atomically (that is in mutual exclusion). Suppose that P is the process  making the system call. The operations are defined  as follows: 	
![[Pasted image 20231224132625.png]]
![[Pasted image 20231224132705.png]]

Semaphores do not require busy-waiting, instead they involve BLOCKING.
![[Pasted image 20231224133455.png]]
Mutex：
A mutex is a semaphore that can be in one of two states: unlocked (0) or locked (1).

![[Pasted image 20231224133537.png]]
##  同步问题
## 生产者消费者问题
## 
## 读者写者问题