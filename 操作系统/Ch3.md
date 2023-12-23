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


