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
2. Electronic component：i.e., device controller ![[截屏2023-12-28 12.39.12.png]]
## Device Controllers
A device controller is a part of a computer system that makes sense of the signals going to, and coming from the CPU. 

Each device controller has a local buffer and some registers. It communicates with the CPU by interrupts. A device's controller plays as a bridge between the device and the operating system.
![[Pasted image 20231228134856.png]]
控制过程
![[Pasted image 20231228135639.png]]
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
![[Pasted image 20231228140025.png]]

每次读写一个字
![[Pasted image 20231228140207.png|400]]

## Interrupt-Driven I/O
Writing a string to the printer using interrupt-driven I/O
1. Code executed when print system call is made
2. Interrupt service procedure
![[Pasted image 20231227172535.png]]
每次读写一个字
![[Pasted image 20231228140605.png]]
How interrupts happens?
Connections between devices and interrupt controller actually use interrupt lines on the bus rather than dedicated wires


## I/O Using DMA
Printing a string using DMA
(a) code executed when the print system call is made
(b) interrupt service procedure
![[Pasted image 20231227174953.png]]

![[Pasted image 20231228141120.png]]
数据不经过cpu
![[Pasted image 20231228141348.png]]


## I/O Software Layers
![[Pasted image 20231227175341.png]]
中间三层再操作系统里面
![[Pasted image 20231228143140.png]]

![[Pasted image 20231227194202.png]]
   Layers of the I/O system and the main functions of each layer

### User IO software
![[Pasted image 20231227194319.png]]
![[Pasted image 20231228143443.png]]
### Device-Independent I/O Software
The basic function of the device-independent software is to perform the ***I/O functions that are common to all devices*** and to provide a uniform interface to the user-level software. 

![[Pasted image 20231227194540.png]]
![[Pasted image 20231228144114.png]]
### Device Drivers
Communications between drivers and device controllers goes over the bus; Logical position of device drivers is shown in the following figure.
![[Pasted image 20231228144226.png]]
![[Pasted image 20231227175444.png|350]]
### Interrupt Handlers 

 Interrupt handlers are best hidden：have driver starting an I/O operation block until interrupt notifies of completion
 Interrupt procedure does its task，then unblocks driver that started it 

Steps must be performed in software after interrupt completed:
Save regs not already saved by interrupt hardware
Set up context for interrupt service procedure
Set up stack for interrupt service procedure
Ack interrupt controller, reenable interrupts
Copy registers from where saved
Run service procedure 
Set up MMU context for process to run next
Load new process' registers
Start running the new process
 
![[Pasted image 20231228144545.png]]

