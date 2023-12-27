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


