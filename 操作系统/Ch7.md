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
