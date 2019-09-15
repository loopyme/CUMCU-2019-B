# python3
# -*- coding: utf-8 -*-
# @File    : drum_simulator.py
# @Desc    : 同心鼓仿真器
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

import pandas as pd
import numpy as np
from math import *
from tabulate import tabulate

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
        self.rota_intertia = 0.08
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

        ################################################################
        # 忽略转动
        # for i, e in enumerate(self.forcee):
        #     self.forcee[i] = (e * np.array([1, 1, 0])) + np.array([0, 0, self.height])

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
        return h_moment_sum / self.weight - 9.8

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
    # 初始化情景
    s = Situation(
        forces_magnitude=np.array([80, 80, 80, 80, 80, 80, 80, 80]),
        forces_time=np.array([-0, -0, 0, -0, -0, 0, 0, -0]),
        line_length=1.7,
    )

    # 设置鼓初始状态
    s.drum.set_status(height=-0.11)

    # 开始仿真
    s.run(time_limit=0.1)
