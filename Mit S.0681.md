命令

# GDB调试
跳到代码所在文件
窗口1：
make qemu-gdb
窗口2:

brew install riscv64-elf-gdb
riscv64-elf-gdb
target remote localhost:25501
file kernel/kernel