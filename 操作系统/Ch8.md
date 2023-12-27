# Segmentation
Programmer’s view memory is not usually as a single linear address space:

Programmer doesn’t know how large these will be, or how they will grow, and doesn’t want to manage where they go in virtual memory.
![[Pasted image 20231227141511.png]]
Segmentation maintains multiple separate virtual address spaces per process.  
Allows each table to grow or shrink, independently.
## Addressing
### Pure Segmentation
![[Pasted image 20231227141609.png]]

