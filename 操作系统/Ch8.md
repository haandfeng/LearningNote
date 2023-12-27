# Segmentation
Programmer’s view memory is not usually as a single linear address space:
- A segment is a logically independent address space.
- segments may have different sizes
- their sizes may change dynamically
- the address space uses 2-dimensional memory addresses and has 2 parts:
	(segment #, offset within segment)
- segments may have different protections
- allows for the sharing of procedures and data between processes. 

![[Pasted image 20231227142810.png]]
Programmer doesn’t know how large these will be, or how they will grow, and doesn’t want to manage where they go in virtual memory.
![[Pasted image 20231227141511.png]]
Segmentation maintains multiple separate virtual address spaces per process.  
Allows each table to grow or shrink, independently.
## Addressing
### Pure Segmentation
![[Pasted image 20231227142636.png]]

### Segmentation with Paging (MULTICS)
先分段，再分页
![[Pasted image 20231227143317.png]]
段表和页表的对应关系，一个进程== 1 段表，>=1页表
![[Pasted image 20231227143419.png]]

![[Pasted image 20231227143933.png]]


