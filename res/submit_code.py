import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from math import *
from tabulate import tabulate

###############################################################################
# python3
# -*- coding: utf-8 -*-
# @File    : drum_simulator.py
# @Desc    : 同心鼓仿真器-主版本(OI优化和数据分析版本见附件)
# @Project : MCM-2019

"""
备忘:
 - 时间:    S,以０为标准发力时刻
 - 坐标轴：　ｘ轴方向朝第１人,右手系，z=0位置在绳水平处
 - 力:      F,使用施，受力点控制方向
 - 高度:    M,Ｈ轴以绳子水平位置为0,向上
 - 角速度:  弧度/s,以右手螺旋法则为正,输出转化为角度制
 - 速度:    m/s,沿Ｈ轴,向上为正
 - 施力点:　forcer, 人手的位置,三阶np.array
 - 受力点:  forcee, 鼓连接绳的位置,三阶np.array

代码结构设计:
 - Situation:  物理环境，控制所有物理变量状态
 - Drum:       鼓，控制鼓本身的状态
 - Force:      力，控制力的状态，
 - main:       主程序，实例化类并配置以完成仿真

注意:
 - update_forcee : 使用法向量进行forcee仿射变换在全周期仿真时存在风险，但在0.25周期仿真中会提供比角度逼近更高的精确度
"""

# 全局日志
REPORT = pd.DataFrame(
    index=[
        "角度X/度",
        "角度Y/度",
        "高度H/米",
        "角速度X/弧度每秒",
        "角速度Y/弧度每秒",
        "平动速度H/米每秒",
        "角加速度X/弧度每二次方秒",
        "角加速度Y/弧度每二次方秒",
        "平动加速度H/弧度每二次方秒",
    ]
)


class Situation:
    """
    整个物理情景
    """

    person_num = 0
    line_length = 0

    drum_radius = 0.2
    drum_start_height = -0.11

    frequency = 100

    def __init__(self, forces_magnitude, forces_time, line_length=None):

        Situation.person_num = len(forces_magnitude)
        if len(forces_magnitude) != len(forces_time):
            print(
                "ERROR:力的输入个数不匹配,力度和开始时间长度分别为：{}{}".format(
                    len(forces_magnitude), len(forces_time)
                )
            )

        if line_length is not None:
            Situation.line_length = line_length

        self.forces_magnitude = forces_magnitude
        self.forces_time = forces_time

        # 初始化力和鼓
        self.forcer = self.init_forcer()
        self.forcee = self.init_forcee()
        self.drum = Drum(
            self.forcer,
            self.forcee,
            self.forces_magnitude,
            self.forces_time,
            self.frequency,
            self.drum_start_height,
        )

    @staticmethod
    def init_forcer():
        """
        计算施力点位置
        :return: 施力点位置列表:np.array
        """
        forcer = []
        for i in range(Situation.person_num):
            forcer.append(
                np.array(
                    [
                        (Situation.drum_radius + Situation.line_length)
                        * cos(2 * i * pi / Situation.person_num),
                        (Situation.drum_radius + Situation.line_length)
                        * sin(2 * i * pi / Situation.person_num),
                        0,
                    ]
                )
            )
        return forcer

    @staticmethod
    def init_forcee():
        """
        计算受力点位置
        :return: 受力点位置列表:np.array
        """
        forcee = []
        for i in range(Situation.person_num):
            forcee.append(
                np.array(
                    [
                        Situation.drum_radius * cos(2 * i * pi / Situation.person_num),
                        Situation.drum_radius * sin(2 * i * pi / Situation.person_num),
                        Situation.drum_start_height,
                    ]
                )
            )
        return forcee

    def run(self, time_limit=None, height_limit=None):
        global REPORT
        self.drum.run(time_limit, height_limit)
        REPORT.T.to_csv("./brief_report.csv")


class Force:
    """力"""

    forcees = []
    forcers = []

    @classmethod
    def set(cls, forcers, forcees):
        """
        设置力ee,力er的位置，以便力的属性更新
        """
        cls.forcees = forcees
        cls.forcers = forcers

    def __init__(self, id, force, is_started):
        self.force = force
        self.id = id
        self.forcee = Force.forcees[self.id]
        self.forcer = Force.forcers[self.id]
        self.force_now = 69
        self.update(is_started)

    def update(self, is_started):
        if is_started:
            self.force_now = self.force
        else:
            self.force_now = 69
        self.forcee = Force.forcees[self.id]
        self.forcer = Force.forcers[self.id]


class Drum:
    """鼓"""

    frequency = 1
    console_report = [
        "时间",
        "角度X",
        "角度Y",
        "总倾斜角度",
        "高度H",
        "角速度X",
        "角速度Y",
        "平动速度H",
        "角加速度X",
        "角加速度Y",
        "平动加速度H",
    ]

    def __init__(
        self, forcer, forcee, forces_magnitude, forces_time, frequency, height
    ):
        """
        初始化鼓的相关物理量
        :param forcer: 施力点列表
        :param forcee: 受力点列表
        :param forces_magnitude: 力的大小列表
        :param forces_time: 力的开始时间列表
        :param frequency: 采样频率
        :param height: 鼓的初始位置
        """

        # 采样率
        Drum.frequency = frequency

        # 时间
        self.current_time = min(forces_time)

        # 零阶量
        self.forcer = forcer
        self.rota_intertia = 0.0507
        self.weight = 3.6
        self.forces_time = forces_time

        # 负一阶量
        self.angle = {"x": 0, "y": 0}
        self.height = height
        self.forcee = forcee

        # 负二阶量
        self.angel_velocity = {"x": 0, "y": 0}
        self.velocity = 0

        # 力
        Force.set(self.forcer, self.forcee)
        self.force = [
            Force(i, forces_magnitude[i], forces_time[i] <= self.current_time)
            for i in range(len(forcee))
        ]

    def set_status(self, height=None, angle=None, angel_velocity=None, velocity=None):
        """
        设置鼓的状态
        :param height: 高度
        :param angle: 角度
        :param angel_velocity:　角速度
        :param velocity: 速度
        :return:
        """
        if height is not None:
            self.height = height
        if angle is not None:
            self.angle = angle
        if angel_velocity is not None:
            self.angel_velocity = angel_velocity
        if velocity is not None:
            self.velocity = velocity

        self.update_forcee()
        Force.set(self.forcer, self.forcee)

    def run(self, time=None, height=None):
        """
        开始仿真
        :param time:停止时间
        :param height: 停止高度
        :return:
        """

        Force.set(self.height, self.angle)

        if time is None and height is not None:
            while self.height <= height:
                self.next_moment()

        elif time is not None and height is None:
            while self.current_time <= time:
                self.next_moment()
        else:
            print("ERROR: Running limit error!")

        pd.DataFrame(
            [
                [
                    self.current_time,
                    self.angle["x"],
                    self.angle["y"],
                    self.angel_velocity["x"],
                    self.height,
                    self.angel_velocity["x"],
                    self.angel_velocity["y"],
                    self.velocity,
                    self.angular_acce["x"],
                    self.angular_acce["x"],
                    self.translation_acca,
                ]
            ],
            columns=Drum.console_report,
        ).to_csv("./res.csv", index=Force)
        print(str(self))

    def next_moment(self):
        """
        更新至下一次采样
        :return:
        """
        # 更新一阶量
        self.angle["x"] += self.angel_velocity["x"] * 0.1 / Drum.frequency
        self.angle["y"] += self.angel_velocity["y"] * 0.1 / Drum.frequency
        self.height += self.velocity * 0.1 / Drum.frequency
        self.update_forcee()

        # 更新二阶量
        self.angel_velocity["x"] += self.angular_acce["x"] * 0.1 / Drum.frequency
        self.angel_velocity["y"] += self.angular_acce["y"] * 0.1 / Drum.frequency
        self.velocity += self.translation_acca * 0.1 / Drum.frequency

        # 更新三阶量
        Force.set(self.forcer, self.forcee)
        for i, f in enumerate(self.force):
            f.update(self.forces_time[i] <= self.current_time)

        # 更新时间
        self.current_time += 0.1 / Drum.frequency

        # 报告
        self.save_report()

    def update_forcee(self):
        """
        根据鼓位置及角度计算受力点
        :return: None
        """
        ################################################################
        # 定义旋转操作符
        rotate_x = lambda r: r[0] * np.array(
            [
                [1, 0, 0, 0],
                [0, cos(r[1]), sin(r[1]), 0],
                [0, -sin(r[1]), cos(r[1]), 0],
                [0, 0, 0, 1],
            ]
        )
        rotate_y = lambda r: r[0] * np.array(
            [
                [1, 0, 0, 0],
                [0, cos(r[1]), sin(r[1]), 0],
                [0, -sin(r[1]), cos(r[1]), 0],
                [0, 0, 0, 1],
            ]
        )
        rotate_z = lambda r: r[0] * np.array(
            [
                [cos(r[1]), sin(r[1]), 0, 0],
                [-sin(r[1]), cos(r[1]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        ################################################################
        # 忽略转动
        # for i, e in enumerate(self.forcee):
        #     self.forcee[i] = (e * np.array([1, 1, 0])) + np.array([0, 0, self.height])

        ################################################################
        # 使用法向量计算
        # normal_vector = -np.cross(
        #     np.array([0, 1, tan(self.angle["y"])]),
        #     np.array([1, 0, tan(self.angle["x"])]),
        # )
        # normal_vector = normal_vector / sqrt(sum(normal_vector ** 2))
        #
        # alpha = -(
        #     atan(normal_vector[0] / normal_vector[1])
        #     if round(normal_vector[1], 6) != 0
        #     else 0
        # )
        # beta = (
        #     atan(normal_vector[2] / normal_vector[1])
        #     if round(normal_vector[1], 6) != 0
        #     else 0
        # )
        # s = rotate_x((np.eye(4), alpha))
        # s = rotate_y((s, beta))
        # s = rotate_y((s, -alpha))

        ################################################################
        # 直接使用角度计算
        s = rotate_x((np.eye(4), -self.angle["x"]))
        s = rotate_y((s, -self.angle["y"]))

        ################################################################
        # 更新所有forcee
        forcee = []
        for i in range(Situation.person_num):
            forcee.append(
                np.array(
                    [
                        Situation.drum_radius * cos(2 * i * pi / Situation.person_num),
                        Situation.drum_radius * sin(2 * i * pi / Situation.person_num),
                        0,
                    ]
                )
            )
        for i, e in enumerate(forcee):
            forcee[i] = (np.append(e, 1) @ s)[:-1] + np.array([0, 0, self.height])
        self.forcee = forcee

    @property
    def angular_acce(self):
        """
        转动加速度计算
        :return: {"x": x转动加速度, "y": y转动加速度}
        """
        # X轴
        x_moment_sum = 0
        for f in self.force:
            f_vector = (
                f.force_now
                * (f.forcer - f.forcee)
                / sqrt(np.sum((f.forcer - f.forcee) ** 2))
            )
            f_arm = f.forcee - np.array([0, 0, self.height])
            f_moment = np.cross(f_vector, f_arm)[0]
            x_moment_sum += f_moment

        # Y轴
        y_moment_sum = 0
        for f in self.force:
            f_vector = (
                f.force_now
                * (f.forcer - f.forcee)
                / sqrt(np.sum((f.forcer - f.forcee) ** 2))
            )
            f_arm = f.forcee - np.array([0, 0, self.height])
            f_moment = np.cross(f_vector, f_arm)[1]
            y_moment_sum += f_moment

        return {
            "x": x_moment_sum / self.rota_intertia,
            "y": y_moment_sum / self.rota_intertia,
        }

    @property
    def translation_acca(self):
        """
        平动加速度计算
        :return: 平动加速度
        """
        h_moment_sum = 0
        for f in self.force:
            f_vector = (
                f.force_now
                * (f.forcer - f.forcee)
                / sqrt(np.sum((f.forcer - f.forcee) ** 2))
            )
            f_moment = f_vector[2]
            h_moment_sum += f_moment
        return h_moment_sum / self.weight

    def save_report(self):
        """
        收集鼓状态,输出报告
        :return: None
        """
        global REPORT
        REPORT[self.current_time] = [
            round(self.angle["x"] / pi * 180, 6),
            round(self.angle["y"] / pi * 180, 6),
            round(self.height, 6),
            round(self.angel_velocity["x"], 6),
            round(self.angel_velocity["y"], 6),
            round(self.velocity, 6),
            round(self.angular_acce["x"], 6),
            round(self.angular_acce["y"], 6),
            round(self.translation_acca, 6),
        ]

    @property
    def angle_all(self):
        """
        组合两个方向上的角度
        :return: 总的偏转角
        """
        normal_vector = np.cross(
            np.array([0, 1, tan(-self.angle["y"])]),
            np.array([1, 0, tan(self.angle["x"])]),
        )
        return acos(abs(normal_vector[2] / sqrt(sum(normal_vector ** 2))))

    def __str__(self):
        """
        美化输出
        :return: Drum.report里要求的所有项目
        """
        return tabulate(
            pd.DataFrame(
                [
                    [
                        round(self.current_time, 5),
                        round(self.angle["x"] / pi * 180, 6),
                        round(self.angle["y"] / pi * 180, 6),
                        # 输出为红色，熬夜的时候更容易看清楚主要结果
                        "\033[0;31m%s\033[0m"
                        % str(round(self.angle_all / pi * 180, 6)),
                        round(self.height, 6),
                        round(self.angel_velocity["x"], 6),
                        round(self.angel_velocity["y"], 6),
                        round(self.velocity, 6),
                        round(self.angular_acce["x"], 6),
                        round(self.angular_acce["y"], 6),
                        round(self.translation_acca, 6),
                    ]
                ],
                columns=Drum.console_report,
            ),
            tablefmt="pipe",
            headers="keys",
        )


if __name__ == "__main__":
    ####################################################################################################################
    # 常规调用仿真
    # 初始化情景
    s = Situation(
        forces_magnitude=np.array([80, 80, 80, 90, 80, 80, 80, 80]),
        forces_time=np.array([0, 0, 0, 0, -0.1, 0, 0, 0]),
        line_length=1.7,
    )

    # 设置鼓初始状态
    s.drum.set_status(height=-0.11)

    # 开始仿真
    s.run(time_limit=0.1)

    ####################################################################################################################
    # 发力大小灵敏度分析-数据采集
    # with open("./force_angle.csv", "w") as f:
    #     f.write("force,angle_y\n")
    #
    # for force in range(80, 200, 10):
    #     # 初始化情景
    #     s = Situation(
    #         forces_magnitude=np.array([force, 80, 80, 80, 80, 80, 80, 80]),
    #         forces_time=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
    #         line_length=1.7,
    #     )
    #
    #     # 设置鼓初始状态
    #     s.drum.set_status(height=-0.11)
    #
    #     # 开始仿真
    #     valuable_data = s.search(time_limit=0.1)[1]
    #     with open("./force_angle.csv", "a") as f:
    #         f.write("{},{}\n".format(force, valuable_data))

    ####################################################################################################################
    # 发力时间灵敏度分析-数据采集
    # with open("./time_angle.csv", "w") as f:
    #     f.write("time,angle_y\n")
    #
    # time = -0.23
    # while time < 0:
    #     # 初始化情景
    #     s = Situation(
    #         forces_magnitude=np.array([80, 80, 80, 80, 80, 80, 80, 80]),
    #         forces_time=np.array([time, 0, 0, 0, 0, 0, 0, 0]),
    #         line_length=1.7,
    #     )
    #
    #     # 设置鼓初始状态
    #     s.drum.set_status(height=-0.11)
    #
    #     # 开始仿真
    #     valuable_data = s.search(time_limit=0.1)[1]
    #     with open("./time_angle.csv", "a") as f:
    #         f.write("{},{}\n".format(time, valuable_data))
    #
    #     time += 0.05

    ####################################################################################################################
    # 施力,受力分析-数据采集
    # s = Situation(
    #     forces_magnitude=np.array([79.6, 80, 80, 80, 80, 80, 80, 80, 80, 79.9]),
    #     forces_time=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #     line_length=1.7,
    # )
    #
    # s.drum.set_status(height=-0.3)
    # s.run(height_limit=0)
    #
    # er = ""
    # for fr in s.forcer:
    #     er += "{},{},{}\n".format(fr[0], fr[1], fr[2])
    # with open("./er.csv", "w") as f:
    #     f.write(er)

    ####################################################################################################################
    # 网格搜索-数据采集
    # res = ""
    # i = 0
    # with open("./log.txt", "w") as f:
    #     f.write("x1,x10,y1,y2,loss\n")
    # # a,b,x,y
    # for x in range(-20, 20):
    #     for y in range(-20, 20):
    #         s = Situation(
    #             forces_magnitude=[
    #                 80 - x / 10,
    #                 80,
    #                 80,
    #                 80,
    #                 80,
    #                 80,
    #                 80,
    #                 80,
    #                 80,
    #                 80 - y / 10,
    #             ],
    #             forces_time=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             line_length=2,
    #         )
    #         i += 1
    #         s_x, s_y = s.search(height_limit=0)
    #         if i % 20 == 0:
    #             with open("./log.txt", "a") as f:
    #                 f.write(res)
    #             res = ""
    #             print(
    #                 i,
    #                 x,
    #                 y,
    #                 round(s_x, 2),
    #                 round(s_y, 2),
    #                 round((s_x ** 2 + (s_y - 0.4175) ** 2), 6),
    #             )
    #         res += "{},{},{},{},{}\n".format(
    #             x, y, s_x, s_y, s_x ** 2 + (s_y - 0.4175) ** 2
    #         )
    # with open("./log.txt", "a") as f:
    #     f.write(res)

###############################################################################
# python3
# -*- coding: utf-8 -*-
# @File    : plot.ipynb
# @Desc    : 绘图工具
# @Project : MCM-2019

# Q1 v-t图

plt.figure(figsize=(10, 5))
plt.grid()
x = np.linspace(0, 6 * np.pi, 300)
y = np.sin(x)
d = np.pi / 2

plt.yticks(np.linspace(-1, 1, 6))
plt.xticks(np.linspace(0, 15 * d, 16))

x1 = [0, d, d, 5 * d, 5 * d, 9 * d, 9 * d]
y1 = [0, -0.5, 1, -1, 1, -1, 1]

x2 = [d, d, 5 * d, 5 * d, 9 * d, 9 * d]
y2 = [-0.5, 1, -1, 1, -1, 1]

plt.fill_between([0, d], [0, -0.5], label="球的初始下落距离")
plt.fill_between([3 * d, 5 * d], [0, -1], label="球的半周期下落距离")
plt.fill_between(
    np.linspace(0, 0.5 * np.pi, 100),
    np.sin(np.linspace(0, 0.5 * np.pi, 100)),
    label="鼓的初始下落距离",
)

plt.plot(x1, y1, color="black", linestyle="-.", label="球的速度")
plt.plot(x, y, linestyle="--", label="鼓的速度")

plt.scatter(x2, y2, color="r", s=50, label="碰撞点")
plt.legend(loc="upper right")

plt.savefig("../res/pic/q1-v-t.png", dpi=720)

# Q3 分析力的大小和发力时间对角度的关系图

f_a = pd.read_csv("../res/csv/force_angle.csv")
plt.figure(figsize=(10, 5))
plt.grid()

plt.yticks(np.linspace(0, 10, 10))
plt.xticks(np.linspace(80, 200, 10))

plt.xlabel("发力大小(s)")
plt.ylabel("倾斜角度(度)")

plt.plot(f_a["force"], f_a["angle_y"])

plt.savefig("../res/pic/q3-f-a.png", dpi=720)

f_a = pd.read_csv("../res/csv/time_angle.csv")
plt.figure(figsize=(10, 5))
plt.grid()

plt.yticks(np.linspace(0, 5, 10))
plt.xticks(np.linspace(-0.3, 0, 12))

plt.xlabel("提前发力时间(s)")
plt.ylabel("倾斜角度(度)")

plt.plot(f_a["time"], f_a["angle_y"])

plt.savefig("../res/pic/q3-t-a.png", dpi=720)

# Q4 小球运动角度

plt.figure(figsize=(10, 5))
plt.grid()

theta = 30 * 0.4 * pi / 180
l = 2.5
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("z")

plt.plot(
    [cos(theta) * l, -cos(theta) * l], [-sin(theta) * l, sin(theta) * l], label="鼓面"
)

plt.plot([-l * 1.5, l * 1.5], [0, 0], color="black", label="x-o-y平面")
plt.plot([0, 0], [0, 1], label="碰撞前小球方向")

plt.plot([0, l * sin(theta)], [0, l * cos(theta)], "-.", label="碰撞平面法线")

plt.plot([0, 1 * sin(2 * theta)], [0, 1 * cos(2 * theta)], label="碰撞后小球方向")
plt.legend(loc="upper right")
plt.annotate(s="θ", xy=(l / 2 * cos(theta / 2 + 0.03), -l / 2 * sin(theta / 2 + 0.03)))
plt.annotate(s="θ", xy=(l / 2 * sin(theta / 2), 1 / 2 * cos(theta / 2)))
plt.annotate(s="θ", xy=(l / 2 * sin(theta / 8 - 0.03), 1 / 2 * cos(theta / 8 - 0.03)))
plt.annotate(s="θ=0.5°", xy=(2.1, 1.1))
plt.savefig("../res/pic/q4-1.png", dpi=720)

plt.figure(figsize=(10, 5))
plt.grid()

theta = -30 * 0.4 * pi / 180
l = 2.5
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("z")

plt.plot(
    [cos(theta) * l, -cos(theta) * l], [-sin(theta) * l, sin(theta) * l], label="鼓面"
)

plt.plot([-l * 1.5, l * 1.5], [0, 0], color="black", label="x-o-y平面")

x, y = (l / 3 * cos(theta), -l / 3 * sin(theta))
plt.plot([0 + x, 0 + x], [0 + y, 1 + y], label="碰撞后小球方向")
plt.plot([0 + x, l * sin(theta) + x], [0 + y, l * cos(theta) + y], "-.", label="碰撞平面法线")
plt.plot(
    [0 + x, 1 * sin(2 * theta) + x], [0 + y, 1 * cos(2 * theta) + y], label="碰撞前小球方向"
)
plt.legend(loc="upper right")

plt.annotate(s="θ", xy=(l / 2 * cos(theta / 2 + 0.03), -l / 2 * sin(theta / 2 + 0.03)))
plt.annotate(
    s="θ", xy=(l / 2 * sin(theta / 2) + x + 0.015, 1 / 2 * cos(theta / 2) + y + 0.015)
)

plt.annotate(s="θ", xy=(l / 2 * sin(theta / 4) - 0.18 + x, 1 / 2 * cos(theta / 4) + y))
plt.annotate(s="θ=0.4187°", xy=(2.1, 1.1))
plt.savefig("../res/pic/q4-2.png", dpi=720)

# Q2 受力分析图

plt.figure(figsize=(12, 2))
plt.grid()
plt.xlabel("x(m)")
plt.ylabel("z(m)")

plt.plot([0.2, 0.2, -0.2, -0.2, 0.2], [-0.33, -0.11, -0.11, -0.33, -0.33], label="鼓")
l = sqrt(1.7 ** 2 - 0.11 ** 2) + 0.2

plt.xticks(np.linspace(-2, 2, 11))
plt.yticks(np.linspace(-0.6, 0, 4))
plt.axis("equal")
plt.scatter([l, -l], [0, 0], color="black", label="施力点")
plt.scatter([-0.2, 0.2], [-0.21, -0.21], color="b", label="受力点")

plt.plot([l, 0.2], [0, -0.21], "-.", color="orange", label="绳")
plt.plot([-l, -0.2], [0, -0.21], "-.", color="orange")

plt.plot([0, 0], [-0.21, -0.51], color="b", label="重力")
plt.plot([-0.2, -1.5], [-0.21, -0.05], color="black", label="拉力")
plt.plot([0.2, 1.5], [-0.21, -0.05], color="black")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig("../res/pic/q2-f.png", dpi=720)

# Q2 仿真结果图


er8 = pd.read_csv("../res/csv/er_8.csv", header=None)
ee8 = pd.read_csv("../res/csv/ee_8.csv", header=None)

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
person_num = 8
for i in range(1, person_num):
    ax.plot(ee8[3 * i][10:], ee8[3 * i + 1][10:], ee8[3 * i + 2][10:], ":", color="red")
ax.plot(ee8[0][10:], ee8[1][10:], ee8[2][10:], ":", color="red", label="鼓上受力点的移动轨迹")

for j in range(10, len(ee8[1]), 20):
    x = [ee8[3 * i][j] for i in range(person_num)] + [ee8[0][j]]
    y = [ee8[3 * i + 1][j] for i in range(person_num)] + [ee8[1][j]]
    z = [ee8[3 * i + 2][j] for i in range(person_num)] + [ee8[2][j]]
    ax.plot(
        x, y, z, label="t={}s时鼓的位置".format(round(0.1 * j / 2 / 100, 3)), linewidth=2
    )

for i in range(person_num):
    for j in range(10, len(ee8[1]), 20):
        x = [ee8[3 * i][j], er8[0][i]]
        y = [ee8[3 * i + 1][j], er8[1][i]]
        z = [ee8[3 * i + 2][j], er8[2][i]]
        ax.plot(x, y, z, color="orange", linewidth="0.5")

ax.set_xlabel("y(m)")
ax.set_ylabel("x(m)")
ax.set_zlabel("z(m)")

ax.scatter(er8[0], er8[1], er8[2], color="black", label="施力者")
plt.legend(loc="upper right")

plt.savefig("../res/pic/q2-res.png", dpi=720)

# Q4 仿真结果图


er = pd.read_csv("../res/csv/er_10.csv", header=None)
ee = pd.read_csv("../res/csv/ee_10.csv", header=None)

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
person_num = 10
for i in range(1, person_num):
    ax.plot(ee[3 * i][10:], ee[3 * i + 1][10:], ee[3 * i + 2][10:], ":", color="red")
ax.plot(ee[0][10:], ee[1][10:], ee[2][10:], ":", color="red", label="鼓上受力点的移动轨迹")

for j in range(10, len(ee[1]), 20):
    x = [ee[3 * i][j] for i in range(person_num)] + [ee[0][j]]
    y = [ee[3 * i + 1][j] for i in range(person_num)] + [ee[1][j]]
    z = [ee[3 * i + 2][j] for i in range(person_num)] + [ee[2][j]]
    ax.plot(
        x,
        y,
        z,
        label="t={}s时鼓的位置".format(round(j / len(ee[1]) * 0.139, 3)),
        linewidth=2,
    )

for i in range(person_num):
    for j in range(10, len(ee[1]), 20):
        x = [ee[3 * i][j], er[0][i]]
        y = [ee[3 * i + 1][j], er[1][i]]
        z = [ee[3 * i + 2][j], er[2][i]]
        ax.plot(x, y, z, color="orange", linewidth="0.5")

ax.set_xlabel("y(m)")
ax.set_ylabel("x(m)")
ax.set_zlabel("z(m)")

ax.scatter(er[0], er[1], er[2], color="black", label="施力者")
plt.legend(loc="upper right")

plt.savefig("../res/pic/q4-res.png", dpi=720)

# Q2 情景展示及坐标轴图


fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)

# drum
r = 0.2
theta = np.arange(0, 2 * np.pi, 0.1)
x = 0 + r * np.cos(theta)
y = 0 + r * np.sin(theta)
x = np.append(x, x[0])
y = np.append(y, y[0])

for i in range(-22, 0):
    ax.plot(x, y, i / 100, "peru", alpha=0.8)
ax.plot(x, y, i / 100, "peru", alpha=0.8, label="鼓")

angel = np.arange(0, 2 * np.pi, 2 * pi / 8)
x = 0 + r * np.cos(angel)
y = 0 + r * np.sin(angel)

ax.scatter(x, y, -0.11, color="navy", linewidth=0.1, label="受力点(绳鼓连接处)")

# 人和绳
ax.scatter(er8[0], er8[1], er8[2], color="black", label="施力点(人手)")
for i in range(8):
    ax.plot(
        [er8[0][i], x[i]], [er8[1][i], y[i]], [er8[2][i], -0.11], "orange", linewidth=1
    )
ax.plot(
    [er8[0][i], x[i]],
    [er8[1][i], y[i]],
    [er8[2][i], -0.11],
    "orange",
    linewidth=1,
    label="绳",
)

# 坐标轴
ax.plot([-2.5, 2.5], [0, 0], [0, 0], "black", linewidth=0.8)
ax.plot([0, 0], [-2.5, 2.5], [0, 0], "black", linewidth=0.8)
ax.plot([0, 0], [0, 0], [-1, 1], "black", linewidth=0.8, label="坐标轴")

ax.set_xlabel("y(m)")
ax.set_ylabel("x(m)")
ax.set_zlabel("z(m)")

plt.legend(loc="upper right")

plt.savefig("../res/pic/q1-situation.png", dpi=720)

# Q4 情景展示图


fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)

# drum
r = 0.2
theta = np.arange(0, 2 * np.pi, 0.1)
x = 0 + r * np.cos(theta)
y = 0 + r * np.sin(theta)
x = np.append(x, x[0])
y = np.append(y, y[0])

for i in range(-22, 0):
    ax.plot(x, y, i / 100, "peru", alpha=0.8)
ax.plot(x, y, i / 100, "peru", alpha=0.8, label="鼓")

angel = np.arange(0, 2 * np.pi, 2 * pi / 10)
x = 0 + r * np.cos(angel)
y = 0 + r * np.sin(angel)

ax.scatter(x, y, -0.11, color="navy", linewidth=0.1, label="受力点(绳鼓连接处)")

# 人和绳
ax.scatter(er[0], er[1], er[2], color="black", label="施力点(人手)")
for i in range(10):
    ax.plot(
        [er[0][i], x[i]], [er[1][i], y[i]], [er[2][i], -0.11], "orange", linewidth=1
    )
ax.plot(
    [er[0][i], x[i]],
    [er[1][i], y[i]],
    [er[2][i], -0.11],
    "orange",
    linewidth=1,
    label="绳",
)

# 坐标轴
ax.plot([-2.5, 2.5], [0, 0], [0, 0], "black", linewidth=0.8)
ax.plot([0, 0], [-2.5, 2.5], [0, 0], "black", linewidth=0.8)
ax.plot([0, 0], [0, 0], [-1, 1], "black", linewidth=0.8, label="坐标轴")

ax.set_xlabel("y(m)")
ax.set_ylabel("x(m)")
ax.set_zlabel("z(m)")

plt.legend(loc="upper right")

plt.savefig("../res/pic/q4-situation.png", dpi=720)

# Q2 力矩分析图


fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)

# drum
r = 0.2
theta = np.arange(0, 2 * np.pi, 0.1)
x = 0 + r * np.cos(theta)
y = 0 + r * np.sin(theta)
x = np.append(x, x[0])
y = np.append(y, y[0])

for i in range(-11, 11):
    ax.plot(x, y, i / 100, "peru", alpha=0.8)
ax.plot(x, y, i / 100, "peru", alpha=0.8, label="鼓")

l = -0.2 / sqrt(2)
ax.scatter([-l], [l], 0, label="受力点位置")
ax.plot([0, -l], [0, l], 0, color="red", label="力臂")
ax.plot([-l, -l], [l, l], [0, 0.6], label="拉力在z轴方向上的分量")
ax.plot([0, 3 * l], [0, 3 * l], 0, color="navy", label="力矩")

ax.plot([0, 0], [0, 3 * l], 0, "-.", color="navy", label="力矩在轴方向上的分量")
ax.plot([0, 3 * l], [0, 0], 0, "-.", color="navy")

ax.plot([3 * l, 3 * l], [0, 3 * l], 0, "-.", color="black", linewidth=0.8)
ax.plot([0, 3 * l], [3 * l, 3 * l], 0, "-.", color="black", linewidth=0.8)

# 坐标轴
ax.plot([-1, 1], [0, 0], [0, 0], "black", linewidth=0.5)
ax.plot([0, 0], [-1, 1], [0, 0], "black", linewidth=0.5)
ax.plot([0, 0], [0, 0], [-1, 1], "black", linewidth=0.5, label="坐标轴")

ax.set_xlabel("y(m)")
ax.set_ylabel("x(m)")
ax.set_zlabel("z(m)")

plt.legend(loc="upper right")

# plt.savefig("../res/pic/q2-f-r.png", dpi=720)


# 平动坐标轴显示图

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)

# drum
r = 0.2
theta = np.arange(0, 2 * np.pi, 0.1)
x = 0 + r * np.cos(theta)
y = 0 + r * np.sin(theta)
x = np.append(x, x[0])
y = np.append(y, y[0])

for i in range(-22, 0):
    ax.plot(x, y, i / 100, "peru", alpha=0.8)
ax.plot(x, y, i / 100, "peru", alpha=0.8, label="鼓")

angel = np.arange(0, 2 * np.pi, 2 * pi / 8)
x = 0 + r * np.cos(angel)
y = 0 + r * np.sin(angel)

ax.scatter(x, y, -0.11, color="navy", linewidth=0.1, label="受力点(绳鼓连接处)")

# 人和绳
ax.scatter(er8[0], er8[1], er8[2], color="black", label="施力点(人手)")
for i in range(8):
    ax.plot(
        [er8[0][i], x[i]], [er8[1][i], y[i]], [er8[2][i], -0.11], "orange", linewidth=1
    )
ax.plot(
    [er8[0][i], x[i]],
    [er8[1][i], y[i]],
    [er8[2][i], -0.11],
    "orange",
    linewidth=1,
    label="绳",
)

# 坐标轴
ax.plot([-2.5, 2.5], [0, 0], [0, 0], "black", linewidth=0.8)
ax.plot([0, 0], [-2.5, 2.5], [0, 0], "black", linewidth=0.8)
ax.plot([0, 0], [0, 0], [-1, 1], "black", linewidth=0.8, label="静止坐标轴")

ax.plot([0, 2.5], [0, 0], [-0.11, -0.11], "navy", linewidth=0.8)
ax.plot([0, 0], [0, -2.5], [-0.11, -0.11], "navy", linewidth=0.8, label="平动坐标轴")

ax.set_xlabel("y(m)")
ax.set_ylabel("x(m)")
ax.set_zlabel("z(m)")

plt.legend(loc="upper right")

plt.savefig("../res/pic/q1-situation1.png", dpi=720)

###############################################################################
# python3
# -*- coding: utf-8 -*-
# @File    : grid1.ipynb & grid2.ipynb
# @Desc    : 网格搜索结果分析工具
# @Project : MCM-2019
log = pd.read_csv("./grid1.txt")
a = log.drop_duplicates().sort_values(by="loss").reset_index()
delta = 1

x = np.arange(-6.0, 6.0, delta)
y = np.arange(-6.0, 6.0, delta)
l = len(x)
Z = np.zeros(shape=(l, l))
X, Y = np.meshgrid(x, y)
for i in range(l):
    Z[i] = a[a["x10"] == -6 + i].sort_values(by="x1")["loss"]

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("rainbow"))

log = pd.read_csv("./grid2.txt")
a = log.drop_duplicates().sort_values(by="loss").reset_index()

delta = 0.1

x = np.arange(-2.0, 2.0, delta)
y = np.arange(-2.0, 2.0, delta)
l = len(x)
Z = np.zeros(shape=(l, l))
X, Y = np.meshgrid(x, y)

for i in range(l):
    Z[i] = a[a["x10"] == -20 + i].sort_values(by="x1")["loss"]

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("rainbow"))
