from manim import *

class CreateCircle(Scene):
    def construct(self):

        left = Dot(LEFT * 4)
        right = Dot(RIGHT * 4)

        steps = 5
        for i in range(steps):
            dot = Dot(color=GRAY)
            dot.shift(LEFT * 4 + RIGHT * 4 * i / steps)
            self.add(dot)
            text = Tex(r"$x_{}$".format(i)).next_to(dot, DOWN)
            self.add(text)


        line = Line(left, right)

        particle = Dot(color=RED)

        x_0_text = Tex(r"$x_0$").next_to(left, DOWN)
        x_1_text = Tex(r"$x_n$").next_to(right, DOWN)

        self.add(x_0_text, x_1_text)

        self.play(MoveAlongPath(particle, line), run_time=5, rate_func=linear)


        