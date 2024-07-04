# lecture2
tldr的使用
更好的man
[mac下载使用链接](https://blog.csdn.net/weixin_49268249/article/details/120403250)

fd的使用
find简介版
[fd使用](https://cloud.tencent.com/developer/article/1435433)

ripgrep的使用
彩色&高级版grep

Ctrl+R 查找历史记录

fzf 模糊搜索
tree命令显示树形结构

autojump 帮助cd
[autojump的下载和使用](https://blog.csdn.net/liaowenxiong/article/details/121044809)

[xargs](https://man7.org/linux/man-pages/man1/xargs.1.html) 命令，它可以使用标准输入中的内容作为参数。 例如 ls | xargs rm 会删除当前目录中的所有文件

-d option doesn't exist on macOS for xargs
This doesn't work on macOS as it is a GNU extension. Had to figure this out. The alternative is using `-print0` option on `find` and `-0` option on `xargs`.