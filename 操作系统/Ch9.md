# Magnetic Disk
Hard disks and floppy disks
Organized into cylinders, tracks, and sectors.
![[Pasted image 20231227203445.png]]
## Disk Formatting
- A low-level format operation should be done on a disk before the disk can be used.
- Each track consists of a number of sectors, with short gaps between the sectors.
![[Pasted image 20231227204142.png]]
## Cylinder Skew
Cylinder skew: the position of sector 0 on each track is offset from the previous track when the low-level format is laid down. 可以改善性能
## Disk Interleaving
Motivation: when the copy to memory is completed (need some time cost), the controller will have to wait almost an entire rotation time for the second sector to come around again.  改善性能
![[Pasted image 20231227205319.png]]
(a) No interleaving;          (b) Single interleaving;         (c) Double interleaving

## Disk Arm Scheduling Algorithms
Time required to read or write a disk block determined by 3 factors
1. Seek time
2. Rotational delay
3. Actual transfer time
***Seek time dominates***
Error checking is done by controllers
![[Pasted image 20231227213531.png|400]]
### Shortest Seek First, SSF
可能会starvation， 找离当前最近的磁道
![[Pasted image 20231227212746.png]]
### Elevator Algorithm
对不同磁道响应不平均
![[IMG_4719.jpeg]]

## Redundant Array of Independent Disk(RAID)
prevent data lost, when disk crush
Key idea: data are distributed over the drives, to allow parallel operation. 
![[Pasted image 20231227221606.png]]
![[Pasted image 20231227221949.png|250]]
![[Pasted image 20231227222056.png|250]]
![[Pasted image 20231227222628.png|250]]
![[Pasted image 20231227222547.png|250]]
![[Pasted image 20231227222655.png|250]]

## Error Handling
Why do errors always happen?
As soon as manufacturing technology has improved to the point where it is possible to operate flawlessly at certain densities, disk designers will go to higher densities to increase the capacity.
![[Pasted image 20231227223256.png]]
Two solutions: Replace bad sector with spare sectors, or shift the sectors. 
(a) A disk track with a bad sector;  
(b) Substituting a spare for the bad sector; 
(c) Shifting all the sectors to bypass the bad one.

## Stable Storage 
RAIDs  do not protect write errors laying down bad data in the first place.
In some applications, it is essential that data never be lost or corrupted.
Stable storage: the goal is to keep the disk consistent at all costs.
![[Pasted image 20231227223925.png]]
### Stable Writes
1. Write the block on drive 1, then read it to verify. 
2. If something wrong, write and reread again up to n times until they work. 
3. After n consecutive failures, the block is remapped onto a spare and the operation repeated until it succeeds.
4. After the write to drive 1 has succeeded, the corresponding block on drive 2 is written and reread, repeatedly if need be, until it, too, finally succeeds.

### Stable Read
1. Read the block on drive 1, if this yields an incorrect ECC, the read is tried again, up to n times. 
2. If all n times fail, then read from drive 2. 
The probability of the same block going bad on both drivers is negligible.

# Clock in Computer
## Clock Hardware
 The crystal oscillator can generate a periodic signal in the range of several hundred MHz. 
 Two modes: one-shot mode, and square-wave mode.
 Clock ticks: periodic interrupts caused by the programmable clock.

![[Pasted image 20231227232853.png]]
![[Pasted image 20231227233036.png]]

## Clock Software
The functions of clock driver
- Maintaining the time of day.
- Preventing processes from running too long.
- Handling the alarm system calls (e.g., ACK).
- Others.
![[Pasted image 20231227233418.png]]
Three ways to maintain the time of day.




# Input Software
## Keyboard
An interrupt is generated when a key is pressed or released. 
The key board driver extracts the Scan code from the I/O port, and translates it to ASCII code.
Two modes: canonical(规范) mode, Non-canonical mode.
## Mouse Software
Two common types: 
- Mouse with a rubber ball
- Optical mouse 

When a mouse has moved a certain minimum distance or a button is depressed or released, a message is sent to the computer. (Dx, Dy, status of buttons)

# Output Software
## : GUI Windows




