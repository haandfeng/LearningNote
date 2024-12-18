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
见中文书122
![[Pasted image 20231226224033.png]]

## Virtual Memory
Problem:  Program too large to fit in memory
Solutions: 
- Virtual memory - OS keeps the part of the program currently in use in memory

Paging is a technique used to implement virtual memory.
Virtual Address is a program generated address.
### MMU
The MMU  (memory management unit) translates a virtual address into a physical address.
The position and function of the MMU
![[Pasted image 20231226230627.png]]



### Page Table 
The relation between virtual addresses and physical  memory addresses     given by page table
![[Pasted image 20231226230717.png|250]]
Example: Virtual address = 4097 = 0001  0000 0000 0001
                                              Virtual page #   12-bit offset
The purpose of the page table is to map virtual pages into page frames. The page table is a function to map the virtual page to the page frame.

 Two major issues : 
1. Page tables may be extremely large (e.g. most computers use) 32-bit address(内存空间4GB) with 4k page size(页面空间4KB), 12-bit offset(4kb的页面空间需要12 bit的offset
      => 20 bits for virtual page number
      =>1 million entries!
2. The mapping must be fast because it is done on every memory access!!

#### Remarks
- Most OSs allocate a page table for each process.
- Single page table consisting of an array of hardware registers.  As a process is loaded, the registers are loaded with page table.
		Advantage - simple
		Disadvantage - expensive if table is large and loading the full page table at every context switch hurts performance. 
- Leave page table in memory - a single register points to the table
		Advantage - context switch cheap
		Disadvantage - one or more memory references to read table entries

#### Structure of a Page Table Entry
![[Pasted image 20231226234531.png]]
- Page frame number: map the frame number
- Present/absent bit: 1/0 indicates valid/invalid entry
- Protection bit: what kinds of access are permitted.
- Modified – set when modified and writing to the disk occur
- Referenced - Set when page is referenced (help decide which page to evict(swap))
- Caching disabled - used to keep data that logically belongs on the disk in memory to  improve performance

### Pure paging 
![[Pasted image 20231226232714.png|450]]


### Multilevel Page Tables
Multilevel page tables - reduce the table size. Also, don't keep page tables in memory that are not needed.
![[Pasted image 20231226233305.png|350]]
32 bit address with 2 page table fields
![[Pasted image 20231226233710.png]]

A logical address (on 32-bit machine with 4K page size) is divided into:
- a page number consisting of 20 bits.
- a page offset consisting of 12 bits.
Since the page table is paged, the page number is further divided into:
- a 10-bit page number. 
- a 10-bit page offset.
![[Pasted image 20231226234156.png]]
Thus, a logical address is as follows:
   where p1 is an index into the outer page table, and p2 is the displacement within the page of the outer page table.
![[Pasted image 20231226234233.png|250]]

### TLB
Observation: 
	Most programs make a large number of references to a small number of pages.

Solution:  
	Equip computers with a small hardware device, called Translation Look-aside Buffer (TLB) or associative memory, to map virtual addresses to physical addresses without using the page table.
把虚拟地址转化成物理地址，首先看快表有没有，在看page table 。相当于register cache 之于memory
![[Pasted image 20231226235440.png]]

### Inverted Page Table（shared by all process）
具体见ppt
- Usually, each process has a page table associated with it. One of drawbacks of this method is that each page table may consist of millions of entries.
- To solve this problem, an inverted page table can be used. There is one entry for each real page (frame) of memory.
- Each entry consists of the virtual address of the page stored in that real memory location, with information about the process that owns that page.
一个page frame 对应一个表项
![[Pasted image 20231227000601.png]]
![[Pasted image 20231227000606.png]]

