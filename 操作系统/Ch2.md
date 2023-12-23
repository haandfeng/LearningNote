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
