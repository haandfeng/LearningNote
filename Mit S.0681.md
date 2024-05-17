命令

# GDB调试
跳到代码所在文件
窗口1：
make qemu-gdb
窗口2:

brew install riscv64-elf-gdb
要先到那个目录启动才可以执行file kernel/kernel
riscv64-elf-gdb
target remote localhost:25501
b sys_sleep
c
file kernel/kernel
layout split
p $fp
在vscode上调试
[参考链接](https://www.cnblogs.com/KatyuMarisaBlog/p/13727565.html)
参考上面链接后半部分，配置好自己的gdb和对应的端口
然后在终端 make qemu gdb
然后在vscode 上debg
__attribute__((noreturn))
-exec file /user/_ sleep

![[Pasted image 20240516163026.png]]




![[Pasted image 20240516230615.png]]




![[Pasted image 20240517103147.png]]