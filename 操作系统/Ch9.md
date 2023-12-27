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
Motivation: when the copy to memory is completed (need some time cost), the controller will have to wait almost an entire rotation time for the second sector to come around again. 
![[Pasted image 20231227205319.png]]
(a) No interleaving;          (b) Single interleaving;         (c) Double interleaving.
