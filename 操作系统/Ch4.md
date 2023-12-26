# Monitors
A monitor is a collection of procedures, variables, and data structures that **can only be accessed by one process at a time**

To allow a process to wait within the monitor, a condition variable must be declared, as condition x, y;
Condition variable can only be used with the operations wait and signal (for the purpose of synchronization).
The operation：
x.wait(); means that the process invoking this operation is suspended until another process invokes
x.signal(); The x.signal operation resumes exactly one suspended process.  If no process is suspended, then the signal operation has no effect.	
## Producer-consumer 样例
![[Pasted image 20231226161241.png]]
