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


# I/O Device

Two common kinds of I/O devices:
1. Block device: stores information in fixed-size blocks.
2. Character device：delivers or accepts a stream of characters, without regard to any block structure.
- Special device: e.g., clocks.

I/O devices cover a huge range in speeds

## Components of I/O devices: 
1. Mechanical component ;
2. Electronic component：i.e., device controller



## Device Controllers
A device controller is a part of a computer system that makes sense of the signals going to, and coming from the CPU. 

Each device controller has a local buffer and some registers. It communicates with the CPU by interrupts. A device's controller plays as a bridge between the device and the operating system.

## Memory-Mapped I/O
Three approaches:
1. Each control register is assigned an I/O port number.
2. All the control registers are mapped into the memory space. This is called memory-mapped I/O.
3. Mapping I/O data buffers into memory space but separating I/O ports from memory
## Programmed I/O
A method of transferring data between the CPU and a peripheral.
Software running on the CPU uses instructions to perform data transfers to or from an I/O device. 
见书p211
![[Pasted image 20231227164620.png]]
![[Pasted image 20231227165522.png]]

## Interrupt-Driven I/O
Writing a string to the printer using interrupt-driven I/O
Code executed when print system call is made
Interrupt service procedure
