# Response设置响应
**http**响应部分可以分为三部分:响应行,响应头,响应体
活动
![[截屏2023-11-13 21.57.57.png]]
## 响应行

响应协议    状态码  状态描述

## 响应头
Content-Type:响应内容的类型(MIME)
![[截屏2023-11-13 22.08.16.png]]

# Servlet的继承结构
# Servlet的生命周期
# ServletContext和ServletConfig
一个是全局储存数据 一个是单个Servlet储存数据
![[截屏2023-11-13 22.58.25.png]]
# URL路径匹配规则
```xml
   <servlet-mapping>

        <servlet-name>servlet1</servlet-name>

        <!--精确匹配-->

        <url-pattern>/servlet1.do</url-pattern>

        <url-pattern>/x.do</url-pattern>

        <!--拓展名匹配-->

        <!--<url-pattern>*.do</url-pattern>-->

        <!--路径匹配-->

        <!-- <url-pattern>/a/b/*</url-pattern>-->

        <!--任意匹配  不包含jsp-->

        <!--<url-pattern>/</url-pattern>-->

        <!--任意匹配 包含了JSP 一般不推荐-->

        <!--<url-pattern>/*</url-pattern>-->

</servlet-mapping>
```


# 注解模式开发Servlet
