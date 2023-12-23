# 复习题
## What are the major functions of OS? Please list some distinct features of OS.
![[Pasted image 20231223142920.png]]


## What is a process and what are the differences between a process and a program?
![[Pasted image 20231223143004.png]]
![[Pasted image 20231223143026.png]]

# Monolithic System(单体系统)
All operating system operations are put into a single file. The operating system is a collection of procedures, each of which can call any of the others. (e.g., Linux, windows)
![[Pasted image 20231223143627.png]]
# Layered System
The operating system is organized as a hierarchy of layers of processes.
![[Pasted image 20231223143907.png]]

# Microkernel
- Split the OS into modules, only one runs in kernel mode and the rest run in user mode;  (a lot communication)
- Put as little as possible into kernel model to improve the reliability of the system.
不让系统经常死机
![[Pasted image 20231223145428.png]]
# Client-Server Model
微内核的变体
Contains two classes of processes, the servers and the clients. Communication between servers and clients is done by message passing.  It is an abstraction that can be used for a single machine or a network of machines
![[Pasted image 20231223145806.png]]

# Process
## Multiprogramming
- when the system is booted, many processes are running simultaneously
- The CPU switches from process to process quickly, running each for tens or hundreds of milliseconds. 
- At anytime, the CPU is running only one process.

## Process Creation
1. System initialization
2.  Created by a running process.
3. User request to create a new process
4. Initiation of a batch job
- Foreground processes: processes that interact with users and perform work for them. 前台进程 和人进行交互
- Background processes that handle some incoming request are called daemons. 后台进程

## Process terminal的原因
1. Normal exit (voluntary)
"Exit” in UNIX and “ExitProcess” in Windows.
4. Error exit (voluntary)
Example: input file is not exist.
3. Error exit (voluntary)
Example: input file is not exist.
4. Fatal error (involuntary)
Example: referencing nonexistent memory.
5. Killed by another process (involuntary)
“Kill” in UNIX and “TerminateProcess” in Windows.


## Process States
- Running：using the CPU at that instant.
- Ready：runnable; temporarily stopped to let another process run.
- Blocked：unable to run until some external event happens.
![[Pasted image 20231223155910.png]]



## Process Scheduling
The OS maintains a Process Table with one entry (called a process control block (PCB)) for each process.(用来实现进程的管理)
![[Pasted image 20231223160307.png]]
## Context switch
When a context switch occurs between processes P1 and P2, the current state of the RUNNING process, say P1, is saved in the PCB for process P1 and the state of a READY process, say P2, is restored from the PCB for process P2 to the CPU registers, etc. Then, process P2 begins RUNNING.

## Pseudo- parallelism
the rapid switching between processes gives the illusion of true parallelism and is called pseudo-parallelism.
