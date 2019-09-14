# MCM-2019-B
同心鼓．．．

## 项目包含以下文件：
 - **venv** 运行本项目所需的Python虚拟环境(考虑到项目体积，某些版本中可能缺失)
 - **simulatou** 仿真器
    - `drum_simulator.py` : **主要的仿真框架和工具**
    - `drum_simulator_for_server.py` : 在服务器上运行的分支版本，简单优化了OI和输出．并且使用的不是项目的标定坐标方向
    - `drum_simulator_for_analyse.py` : 专门用于收集数据的分支版本，简单优化了数据采集．但不包含数据处理
 - **analyse** 分析工具
    - `grid1.ipynb` 和 `grid2.ipynb` : 用于完成最优化的网格搜索脚本,后者是前者的高精度版本
    - `ploy.ipynb` : 分析仿真产生的数据,并绘制图表
 - **res** 产出的一些数据和图表
    - **csv** : 产出的数据(仅作为备份,直接使用时具有风险)
    - **pic** : 产出的图表
    
## 项目中的规定值
在`.\simulatou\drum_simulator.py`中陈述

