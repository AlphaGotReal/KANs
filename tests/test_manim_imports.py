def test_manim_imports():
    from manim import Scene
    class Debug(Scene):
        def construct(self):
            self.play(Write(Text("hello world"), run_time=3.0))
