# Monitors
A monitor is a collection of procedures, variables, and data structures that **can only be accessed by one process at a time**

To allow a process to wait within the monitor, a condition variable must be declared, as condition x, y;
Condition variable can only be used with the operations wait and signal (for the purpose of synchronization).
The operation：
x.wait(); means that the process invoking this operation is suspended until another process invokes
x.signal(); The x.signal operation resumes exactly one suspended process.  If no process is suspended, then the signal operation has no effect.	
## Producer-consumer 样例
Monitor
![[Pasted image 20231226161241.png|200]]
使用
![[Pasted image 20231226161623.png|200]]

## Pros and Cons
Monitors in Java 
supports user-level threads  and methods (procedures) to be grouped together into classes.
By adding the keyword synchronized to a method, Java guarantees that once any thread has started executing that method, no other thread can execute that method. 
Advantages: Ease of programming. 
Disadvantages:
Monitors are a programming language concept, so they are difficult to add to an existing language; e.g., how can a compiler know which procedures were in monitor?
Monitors are too expensive to implement.

## Monitor 和 Semaphores比较
共同点：
The wait and signal operations on condition variables in a monitor are similar to P and V operations on counting semaphores. A wait statement can block a process's execution, while a signal statement can cause another process to be unblocked. However, there are some differences between them. 

不同点：
- When a process executes a P operation, it does not necessarily block that process because the counting semaphore may be greater than zero. In contrast, when a wait statement is executed, it always blocks the process. 
- When a task executes a V operation on a semaphore, it either unblocks a task waiting on that semaphore or increments the semaphore counter if there is no task to unlock. On the other hand, if a process executes a signal statement when there is no other process to unblock, there is no effect on the condition variable.
- 
- 
- Another difference between semaphores and monitors is that users awaken by a V operation can resume execution without delay. Contrarily, users awaken by a signal operation are restarted only when the monitor is unlocked.

# Message Passing 
Message passing may be blocking or non-blocking
Blocking is considered synchronous
Blocking send has the sender block until the message is received
Blocking receive has the receiver block until a message is available
