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

# Disabling Interrupts

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
互相谦让的算法
![[Pasted image 20231223205046.png]]

