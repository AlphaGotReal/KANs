from manim import *

class Debug(Scene):
    def construct(self):
        self.play(
            Write(Text("hello world"), run_time=3)
        )
