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
- Blocking send has the sender block until the message is received
- Blocking receive has the receiver block until a message is available

Non-blocking is considered asynchronous
- Non-blocking send has the sender send the message and continue
- Non-blocking receive has the receiver receive a valid message or null

Sender: it is more natural not to be blocked after issuing send:
- can send several messages to multiple destinations.
- but sender usually expect acknowledgment of message receipt (in case receiver fails).

Receiver: it is more natural to be blocked after issuing receive:
- the receiver usually needs the information before proceeding.
- but could be blocked indefinitely if sender process fails to send.

Other methods are offered, e.g., blocking send, blocking receive: 
- both are blocked until the message is received.
- provides tight synchronization (rendezvous聚会).
 
So, there are  three combinations that make sense:
(1) Blocking send, Blocking receive;
(2) Nonblocking send, Nonblocking receive;
(3) Nonblocking send, Blocking receive – most popular


# Process Scheduling
**Scheduler**: A part of the operating system that decides which process is to be run next.
**Scheduling Algorithm**: a policy used by the scheduler to make that decision.
- To make sure that no process runs too long, a clock is used to cause a periodic interrupt (usually around 50-60 Hz); that is, about every 20 msec.
**Preemptive Scheduling**: allows processes that are runnable to be temporarily suspended so that other processes can have a chance to use the CPU.

## Process Behavior
bursts 密集

![[Pasted image 20231226164159.png]]
中文书解释
![[Pasted image 20231226164317.png]]


## When to  Schedule
When a new process is created;
When a process exist;
When a process blocks on I/O;
When an I/O interrupt occurs (e.g., clock interrupt).

## Scheduling Algorithm Goals
1. Fairness - each process gets its fair share of time with the CPU.
2. Efficiency - keep the CPU busy doing productive work.
3. Response Time - minimize the response time for interactive users.
4. Turn around Time - minimize the average time from a batch job being submitted until it is completed.
5. Throughput - maximize the number of jobs processed per hour.

## First-Come, First-Served (FCFS) Scheduling
| Process | Burst Time |
| ------- | ---------- |
| P1      | 24         |
| P2      | 3          |
| P3      | 3          |
Suppose that the processes arrive in the order: P1 , P2 , P3  
Waiting time for  P1, P2, and P3  = 0，24，27
Average waiting time = (0+24+27)/3 = 17

## Shortest-Job-First (SJF) Scheduling

Associate with each process the length of its next CPU burst.  Use these lengths to schedule the process with the shortest time.
![[Pasted image 20231226165753.png]]
nonpreemptive – once CPU given to the process it cannot be preempted until it completes its CPU burst.
preemptive – if a new process arrives with CPU burst length less than remaining time of current executing process, preempt.  This scheme is know as the Shortest-Remaining-Time-First (SRTF).


Adv:
SJF is optimal – gives minimum average waiting time for a given set of processes.
Drawback:
- The real difficulty with the SJF algorithm is knowing the length of the next CPU request.
- Can cause starvation 
## Round Robin (RR) Scheduling
Each process gets a small unit of CPU time (time quantum), usually 10-100 milliseconds.  After this time has elapsed, the process is preempted and added to the end of the ready queue.
If there are n processes in the ready queue and the time quantum is q, then each process gets 1/n of the CPU time in chunks of at most q time units at once.  No process waits 
more than (n-1)q time units.
Performance
q large => FIFO
q small => q must be large with respect to context switch, otherwise overhead is too high.

| Process | Burst Time |
| ------- | ---------- |
| P1      | 53         |
| P2      | 17         |
| P3      | 68         |
| P4      | 24         |

![[Pasted image 20231226174036.png]]
Typically, higher average turnaround than SJF, but better response.

## Priority Scheduling
A priority number (integer) is associated with each process
The CPU is allocated to the process with the highest priority (smallest integer == highest priority).
Preemptive
nonpreemptive
SJF is a priority scheduling where priority is the predicted next CPU burst time.
Problem: Starvation – low priority processes may never execute.
Solution: Aging – as time progresses increase the priority of the process.
![[Pasted image 20231226174539.png]]

## Highest Response Ratio Next Scheduling 
considers both the process waiting time and execution time
is a non-preemptive CPU Scheduling algorithm.
Response Ratio = (W + S)/S
Here, W is the waiting time of the process so far and S is the Burst time of the process.

## More Scheduling
### Shortest Process Next
SJF can be used in an interactive environment by estimating the runtime based on past behavior. Aging is a method used to estimate runtime by taking a weighted average of the current runtime and the previous estimate.
Example: Let a = estimate weight, then the current estimate is: a *T0 + (1-a) * T1
where T0 is the previous estimate and T1 is the current runtime.

Guaranteed Scheduling
Suppose 1/n of the CPU cycles.
Compute ratio = actual CPU time consumed / CPU time entitled
Run the process with the lowest ratio
### Lottery Scheduling
Give processes lottery tickets for various system resources
When a scheduling decision is made, a lottery ticket is chosen, and the process holding that ticket gets the resource.
### Fair-Share Scheduling
Take into account how many processes a user owns.
Example: User 1 – A, B, C, D and Use 2 – E
Round-robin: ABCDEABCDE...
Fair-Share: if use 1 is entitled to the same CPU time as user 2
   AEBECEDEAEBECEDE….





# Thread Scheduling
The process scheduling algorithms can also be used in thread scheduling. In practice, round-robin and priority scheduling are commonly used.

User-level and kernel-level threads
- A major difference between user-level threads and kernel-level threads is the performance.
- User-level threads can employ an application-specific thread scheduler.
