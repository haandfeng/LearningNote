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
More page frames might not always have fewer page faults.
![[Pasted image 20231227113048.png]]

