# Page Replacement Introduction
Page fault forces choice 
- which page must be removed- 
- make room for incoming page
Modified page must first be saved
- unmodified just overwritten
Better not to choose an often used page
- will probably need to be brought back in soon
# Optimal Page Replacement Algorithm
Replace the page which will be referenced at the farthest point.(把最远才可能使用的page 换出去)
Optimal but impossible to implement.
# Not Recently Used Page Replacement Algorithm

Each page has Reference bit (R) and Modified bit (M).
- bits are set when page is referenced (read or written recently), modified (written to)
- when a process starts, both bits R and M are set to 0 for all pages.
- periodically, (on each clock interval (20msec) ), the R bit is cleared. (i.e. R=0).

Pages are classified
- Class 0: not referenced, not modified
- Class 1: not referenced, modified
- Class 2: referenced, not modified
- Class 3: referenced, modified
NRU removes page at random
- from lowest numbered non-empty class

# FIFO Page Replacement Algorithm
Maintain a linked list of all pages 
- Pages came into memory with the oldest page at the front of the list.
Page at beginning of list replaced
Advantage?
- easy to implement
Disadvantage?
- page in memory the longest (perhaps often used) may be evicted







# Second Chance Page Replacement Algorithm
![[Pasted image 20231227095835.png]]
Inspect R bit:
    if R = 0 => evict the page
    if R = 1 => set R = 0 and put page at end (back) of list. The page is treated like a newly loaded page.
![[Pasted image 20231227095534.png]]
![[Pasted image 20231227095531.png]]
# The Clock Page Replacement Algorithm
![[Pasted image 20231227100046.png]]
![[Pasted image 20231227095934.png]]

# Least Recently Used (LRU)
Assume pages used recently will used again soon
- throw out page that has been unused for longest time
## Software Solution
- Must keep a linked list of pages: most recently used at front, least at rear; update this list every memory reference  Too expensive!!
## Hardware solution
Equip hardware with a 64 bit counter. 
- That is incrementing after each instruction. 
- The counter value is stored in the page table entry of  the page that was just referenced.
- choose page with lowest value counter
- periodically zero the counter
Problem? 
- page table is very large, become even larger.
Maintain a matrix of n *x* n bits for a machine with n page frames. 
 When page frame K is referenced:
     (i)  Set row K to all 1s.
     (ii) Set column K to all 0s.
 The row whose binary value is smallest is the LRU page.

![[Pasted image 20231227102400.png]]
LRU using a matrix – pages referenced in order 0,1,2,3,2,1,0,3,2,3


# Simulating LRU in Software
LRU hardware is not usually available. NFU (Not Frequently Used) is implemented in software.
- At each clock interrupt, the R bit is added to the counter associated with each page. When a page fault occurs, the page with the lowest counter is replaced.
- Difference? Problem?
 NFU never forgets, so a page referenced frequency long ago may have the highest counter.
 
Modified NFU = NFU with Aging - at each clock interrupt:
	The counters are shifted right one bit, and
	The R bits are added to the leftmost bit.
	In this way, we can give higher priority to recent R values.
![[Pasted image 20231227102742.png]]
The aging algorithm simulates LRU in software
Note 6 pages for 5 clock ticks, (a) – (e)

# Working-Set Model
- Pages are loaded only on demand. This strategy is called demand paging.
- During the phase of execution the process references relatively small fraction of its pages. This is called a locality of reference.
- The set of pages that a process is using currently is called its ***working set.***
- A program causing page faults every few instructions is said to be thrashing.
- Paging systems keep each process’s working set in memory before letting the process run. This approach is called the ***working set model.***
- Loading the pages before letting processes run is called ***pre paging***.
- The working set is the set of pages used by the k most recent memory references, w(k,t) is the size of the working set at time  t.
![[Pasted image 20231227110628.png]]

The idea is to examine the most recent page references. Evict a page that is not in the working set.

The working set of a process is the set of pages it has referenced during the past τ seconds of virtual time (the amount of CPU time a process has actually used).

Scan the entire page table and evict the page:
	R= 0, its age is greater than τ.
	R = 0, its age is not greater than τ and its age is largest.
	R = 1, randomly choose a page.
The basic working set algorithm is expensive. Instead, WSCLock is used in practice.
具体操作方法见书137
![[Pasted image 20231227111815.png]]

![[Pasted image 20231227112247.png]]

# Modeling Page Replacement Algorithms

## Belady’s anomaly:
只有FIFO会产生
More page frames might not always have fewer page faults.
![[Pasted image 20231227113048.png]]

## Modeling LRU Algorithm
When a page is referenced, it is always moved to the top entry in pages in memory.
If the page referenced was already in memory, all pages above it move down one position. Pages that below the referenced page are not moved.
![[Pasted image 20231227113430.png]]

## Stack Replacement Algorithms
A page replacement algorithm is called a stack replacement algorithm if the set of pages in a k-frame memory is always a subset of the pages in a (k + 1) frame memory.
M is the set of memory array, after each item in reference string is processed，m is the number of page frames, then  M(m)⊆ M(m+1).


# Local /Global allocation

## Local versus Global Allocation Policies
Global algorithms dynamically allocate page frames among all runnable processes. Local algorithms allocate pages for a single process.
![[IMG_4716.jpeg]]
A global algorithm is used to prevent thrashing and keep the paging rate within acceptable bounds: 
A：too  high => assign more page frames to the process. 
B：too  low => assign process fewer page frames.
![[Pasted image 20231227115129.png]]

# Page Size
Small page size
Advantages: 
   less internal fragmentation

Disadvantages:
   programs need many pages  => larger page tables
![[Pasted image 20231227131841.png]]

# Separate Instruction and Data Spaces
Most systems separate address spaces for instructions (program text) and data. A process can have two pointers in its process table: one to the instruction page and one to the data page.
A shared code can be pointed by two processes.
![[Pasted image 20231227132418.png]]
# Shared Pages
![[Pasted image 20231227132549.png|300]]

Two processes sharing same program sharing its page table

# Cleaning Policy
Paging Daemon(分页守护进程):
- A background process, in sleep state in most of the time;
- Periodically woken up to inspect state of memory
- When too few frames are free, selects pages to evict using a replacement algorithm.
# Page Fault Handling
1. Hardware traps to kernel
2. Save general registers
3. Determines which virtual page needed
4. Seeks page frame
5. If the selected frame is dirty(修改了), write it to disk
6. Brings new page in from disk
7. Page tables updated
8. Instruction backed up to when it began 
9. Faulting process scheduled(调度引发缺页中断的程序)
10. Registers restored & Program continues
![[Pasted image 20231227134326.png|300]]

Process issues call for read from device into buffer
- while waiting for  I/O, another process starts up
- has a page fault
- buffer for the first process may be chosen to be paged out
If a page transferring data through the I/O is paged out, it will cause part of the data in buffer and part in the newly loaded page. In this case, the page need to be locked (pinning).




# Backing Store
Two approaches to allocate page space on the disk:
1）Paging to static swap area 
2）Backing up pages dynamically with a disk map

![[Pasted image 20231227135412.png]]
(a) Paging to static swap area；    (b) Backing up pages dynamically.
