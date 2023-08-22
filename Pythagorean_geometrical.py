from manimlib.imports import *

class Grid1515(VGroup):
    CONFIG = {
        "height": 6.0,
        "width": 6.0,
    }

    def __init__(self, rows, columns, **kwargs):
        digest_config(self, kwargs, locals())
        super().__init__(**kwargs)

        x_step = self.width / self.columns
        y_step = self.height / self.rows

        for x in np.arange(0, self.width + x_step, x_step):
            self.add(Line(
                [x - self.width / 2., -self.height / 2., 0],
                [x - self.width / 2., self.height / 2., 0],
            ))
        for y in np.arange(0, self.height + y_step, y_step):
            self.add(Line(
                [-self.width / 2., y - self.height / 2., 0],
                [self.width / 2., y - self.height / 2., 0]
            ))


class ScreenGrid(VGroup):
    CONFIG = {
        "rows": 8,
        "columns": 14,
        "height": FRAME_Y_RADIUS * 2,
        "width": 14,
        "grid_stroke": 0.5,
        "grid_color": WHITE,
        "axis_color": RED,
        "axis_stroke": 2,
        "labels_scale": 0.25,
        "labels_buff": 0,
        "number_decimals": 2
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rows = self.rows
        columns = self.columns
        grid = Grid(width=self.width, height=self.height, rows=rows, columns=columns)
        grid.set_stroke(self.grid_color, self.grid_stroke)

        vector_ii = ORIGIN + np.array((- self.width / 2, - self.height / 2, 0))
        vector_si = ORIGIN + np.array((- self.width / 2, self.height / 2, 0))
        vector_sd = ORIGIN + np.array((self.width / 2, self.height / 2, 0))

        axes_x = Line(LEFT * self.width / 2, RIGHT * self.width / 2)
        axes_y = Line(DOWN * self.height / 2, UP * self.height / 2)

        axes = VGroup(axes_x, axes_y).set_stroke(self.axis_color, self.axis_stroke)

        divisions_x = self.width / columns
        divisions_y = self.height / rows

        directions_buff_x = [UP, DOWN]
        directions_buff_y = [RIGHT, LEFT]
        dd_buff = [directions_buff_x, directions_buff_y]
        vectors_init_x = [vector_ii, vector_si]
        vectors_init_y = [vector_si, vector_sd]
        vectors_init = [vectors_init_x, vectors_init_y]
        divisions = [divisions_x, divisions_y]
        orientations = [RIGHT, DOWN]
        labels = VGroup()
        set_changes = zip([columns, rows], divisions, orientations, [0, 1], vectors_init, dd_buff)
        for c_and_r, division, orientation, coord, vi_c, d_buff in set_changes:
            for i in range(1, c_and_r):
                for v_i, directions_buff in zip(vi_c, d_buff):
                    ubication = v_i + orientation * division * i
                    coord_point = round(ubication[coord], self.number_decimals)
                    label = Text(f"{coord_point}",font="Arial",stroke_width=0).scale(self.labels_scale)
                    label.next_to(ubication, directions_buff, buff=self.labels_buff)
                    labels.add(label)

        self.add(grid, axes, labels)



class WhatIsTransform(Scene):
    def construct(self):
        M1 = TextMobject("A")
        M2 = TextMobject("B")
        M3 = TextMobject("C")
        M4 = TextMobject("D")
        self.add(M1)
        self.wait()
 
        self.play(Transform(M1,M2))
        self.wait()
 
        self.play(Transform(M1,M3))
        self.wait()
 
        self.play(Transform(M1,M4))
        self.wait()
 
        self.play(FadeOut(M1))
 
class WhatIsReplacementTransform(Scene):
    def construct(self):
        M1 = TextMobject("A")
        M2 = TextMobject("B")
        M3 = TextMobject("C")
        M4 = TextMobject("D")
        self.add(M1)
        self.wait()
 
        self.play(ReplacementTransform(M1,M2))
        self.wait()
 
        self.play(ReplacementTransform(M2,M3))
        self.wait()
 
        self.play(ReplacementTransform(M3,M4))
        self.wait()
 
        self.play(FadeOut(M4))

class TransformationText1V1(Scene):
	def construct(self):
		texto1 = TextMobject("First text")
		texto2 = TextMobject("Second text")
		self.play(Write(texto1))
		self.wait()
		self.play(Transform(texto1,texto2))
		self.wait()

class TransformationText1V2(Scene):
	def construct(self):
		texto1 = TextMobject("First text")
		texto1.to_edge(UP)
		texto2 = TextMobject("Second text")
		self.play(Write(texto1))
		self.wait()
		self.play(Transform(texto1,texto2))
		self.wait()

class TransformationText2(Scene):
	def construct(self):
		text1 = TextMobject("Function")
		text2 = TextMobject("Derivative")
		text3 = TextMobject("Integral")
		text4 = TextMobject("Transformation")
		self.play(Write(text1))
		self.wait()
		#Trans text1 -> text2
		self.play(ReplacementTransform(text1,text2))
		self.wait()
		#Trans text2 -> text3
		self.play(ReplacementTransform(text2,text3))
		self.wait()
		#Trans text3 -> text4
		self.play(ReplacementTransform(text3,text4))
		self.wait()

class CopyTextV1(Scene):
	def construct(self):
		formula = TexMobject(
			"\\frac{d}{dx}", #0
			"(", #1
			"u", #2
			"+", #3
			"v", #4
			")", #5
			"=", #6
			"\\frac{d}{dx}", #7
			"u", #8
			"+", #9
			"\\frac{d}{dx}", #10
			"v" #11
			)
		formula.scale(2)
		self.play(Write(formula[0:7]))
		self.wait()
		self.play(
			ReplacementTransform(formula[2].copy(),formula[8]),
			ReplacementTransform(formula[4].copy(),formula[11]),
			ReplacementTransform(formula[3].copy(),formula[9])
			)
		self.wait()
		self.play(
			ReplacementTransform(formula[0].copy(),formula[7]),
			ReplacementTransform(formula[0].copy(),formula[10])
			)
		self.wait()

class CopyTextnew(Scene): #A nicer way of writing the above code
	def construct(self):
		formula = TexMobject(
			"\\frac{d}{dx}", #0
			"(", #1
			"u", #2
			"+", #3
			"v", #4
			")", #5
			"=", #6
			"\\frac{d}{dx}", #7
			"u", #8
			"+", #9
			"\\frac{d}{dx}", #10
			"v" #11
			)
		formula.scale(2)
		self.play(Write(formula[0:7]))
		self.wait()
		changes=[
			[(2,4,3),
	#Step2    | | |
			(8,11,9)],

			[(0,0),	
		#	  | |
			(7,10)]
		]
		for positional_list,index_in_list in changes:
			self.play(
				*[
					ReplacementTransform(formula[i].copy(),formula[j])
					for i,j in zip(positional_list,index_in_list)

				]
			)
			self.wait()
		self.wait()


class CopyTextV2(Scene):
	def construct(self):
		formula = TexMobject("\\frac{d}{dx}",
			"(","u","+","v",")","=",
			"\\frac{d}{dx}","u","+","\\frac{d}{dx}","v"
			)
		formula.scale(2)
		self.play(Write(formula[0:7]))
		self.wait()
		self.play(
			ReplacementTransform(formula[2].copy(),formula[8]),
			ReplacementTransform(formula[4].copy(),formula[11]),
			ReplacementTransform(formula[3].copy(),formula[9]),
			run_time=3
			)
		self.wait()
		self.play(
			ReplacementTransform(formula[0].copy(),formula[7]),
			ReplacementTransform(formula[0].copy(),formula[10]),
			run_time=3
			)
		self.wait()

class CopyTextV3(Scene):
	def construct(self):
		formula = TexMobject("\\frac{d}{dx}",
			"(","u","+","v",")","=",
			"\\frac{d}{dx}","u","+","\\frac{d}{dx}","v"
			)
		formula.scale(2)
		formula[8].set_color(RED)
		formula[11].set_color(BLUE)
		self.play(Write(formula[0:7]))
		self.wait()
		self.play(
			ReplacementTransform(formula[2].copy(),formula[8]),
			ReplacementTransform(formula[4].copy(),formula[11]),
			ReplacementTransform(formula[3].copy(),formula[9]),
			run_time=3
			)
		self.wait()
		self.play(
			ReplacementTransform(formula[0].copy(),formula[7]),
			ReplacementTransform(formula[0].copy(),formula[10]),
			run_time=3
			)
		self.wait()

class CopyTextV4(Scene):
	def construct(self):
		formula = TexMobject("\\frac{d}{dx}",
			"(","u","+","v",")","=",
			"\\frac{d}{dx}","u","+","\\frac{d}{dx}","v"
			)
		formula.scale(2)
		for letter,color in [("u",RED),("v",BLUE)]:
			formula.set_color_by_tex(letter,color)
		self.play(Write(formula[0:7]))
		self.wait()
		self.play(
			ReplacementTransform(formula[2].copy(),formula[8]),
			ReplacementTransform(formula[4].copy(),formula[11]),
			ReplacementTransform(formula[3].copy(),formula[9]),
			run_time=3
			)
		self.wait()
		self.play(
			ReplacementTransform(formula[0].copy(),formula[7]),
			ReplacementTransform(formula[0].copy(),formula[10]),
			run_time=3
			)
		self.wait()

class CopyTwoFormulas1(Scene):
	def construct(self):
		formula1 = TexMobject(
				"\\neg",		#0
				"\\forall",		#1
				"x",			#2
				":",			#3
				"P(x)"			#4
			)
		formula2 = TexMobject(
				"\\exists",		#0
				"x",			#1
				":",			#2
				"\\neg",		#3
				"P(x)"			#4
			)
		for size,pos,formula in [(2,2*UP,formula1),(2,2*DOWN,formula2)]:
			formula.scale(size)
			formula.move_to(pos)
		self.play(Write(formula1))
		self.wait()
		changes = [
			[(0,1,2,3,4),
			# | | | | |
			# v v v v v
			 (3,0,1,2,4)],
		]
		for pre_ind,post_ind in changes:
			self.play(*[
				ReplacementTransform(
					formula1[i].copy(),formula2[j]
					)
				for i,j in zip(pre_ind,post_ind)
				],
				run_time=2
			)
			self.wait()

class CopyTwoFormulas2(Scene):
	def construct(self):
		formula1 = TexMobject(
				"\\neg","\\forall","x",":","P(x)"
			)
		formula2 = TexMobject(
				"\\exists","x",":","\\neg","P(x)"
			)
		for tam,pos,formula in [(2,2*UP,formula1),(2,2*DOWN,formula2)]:
			formula.scale(tam)
			formula.move_to(pos)
		self.play(Write(formula1))
		self.wait()
		changes = [
			# First time
			[(2,3,4),
			# | | |
			# v v v
			 (1,2,4)],
			# Second time
			[(0,),
			# | 
			# v
			 (3,)],
			# Third time
			[(1,),
			# | 
			# v
			 (0,)]
		]
		for pre_ind,post_ind in changes:
			self.play(*[
				ReplacementTransform(
					formula1[i].copy(),formula2[j]
					)
				for i,j in zip(pre_ind,post_ind)
				],
				run_time=2
			)
			self.wait()

class CopyTwoFormulas2Color(Scene):
	def construct(self):
		formula1 = TexMobject(
				"\\neg","\\forall","x",":","P(x)"
			)
		formula2 = TexMobject(
				"\\exists","x",":","\\neg","P(x)"
			)
		parametters = [(2,2*UP,formula1,GREEN,"\\forall"),
					  (2,2*DOWN,formula2,ORANGE,"\\exists")]
		for size,pos,formula,col,sim in parametters:
			formula.scale(size)
			formula.move_to(pos)
			formula.set_color_by_tex(sim,col)
			formula.set_color_by_tex("\\neg",PINK)
		self.play(Write(formula1))
		self.wait()
		changes = [
			[(2,3,4),(1,2,4)],
			[(0,),(3,)],
			[(1,),(0,)]
		]
		for pre_ind,post_ind in changes:
			self.play(*[
				ReplacementTransform(
					formula1[i].copy(),formula2[j]
					)
				for i,j in zip(pre_ind,post_ind)
				],
				run_time=2
			)
			self.wait()

class CopyTwoFormulas3(Scene):
	def construct(self):
		formula1 = TexMobject(
				"\\neg","\\forall","x",":","P(x)"
			)
		formula2 = TexMobject(
				"\\exists","x",":","\\neg","P(x)"
			)
		parametters = [(2,2*UP,formula1,GREEN,"\\forall"),
					  (2,2*DOWN,formula2,ORANGE,"\\exists")]
		for size,pos,formula,col,sim in parametters:
			formula.scale(size)
			formula.move_to(pos)
			formula.set_color_by_tex(sim,col)
			formula.set_color_by_tex("\\neg",PINK)
		self.play(Write(formula1))
		self.wait()
		changes = [
			[(2,3,4),(1,2,4)],
			[(0,),(3,)],
			[(1,),(0,)]
		]
		for pre_ind,post_ind in changes:
			self.play(*[
				ReplacementTransform(
					formula1[i],formula2[j]
					)
				for i,j in zip(pre_ind,post_ind)
				],
				run_time=2
			)
			self.wait()

class ChangeTextColorAnimation(Scene):
	def construct(self):
		text = TextMobject("Text")
		text.scale(3)
		self.play(Write(text))
		self.wait()
		self.play(
                text.set_color, YELLOW,
                run_time=2
            )
		self.wait()

class ChangeTextColorAnimationnew(Scene):
	def construct(self):
		text = TextMobject("Text")
		text.scale(3)
		self.play(Write(text))
		self.wait()
		self.play(
			ApplyMethod(text.set_color, YELLOW),
                run_time=2
            )
		self.wait()

class ChangeSizeAnimation(Scene):
	def construct(self):
		text = TextMobject("Text")
		text.scale(2)
		self.play(Write(text))
		self.wait()
		self.play(
                text.scale, 3,
                run_time=2
            )
		self.wait()

class MoveText(Scene):
	def construct(self):
		text = TextMobject("Text")
		text.scale(2)
		text.shift(LEFT*2)
		self.play(Write(text))
		self.wait()
		self.play(
                text.shift, RIGHT*2,
                run_time=2,
                path_arc=0 #Change 0 by -np.pi
            )
		self.wait()

class ChangeColorAndSizeAnimation(Scene):
	def construct(self):
		text = TextMobject("Text")
		text.scale(2)
		text.shift(LEFT*2)
		self.play(Write(text))
		self.wait()

		text.generate_target()
		text.target.shift(RIGHT*2)
		text.target.set_color(RED)
		text.target.scale(2)

		self.play(MoveToTarget(text),
                #text.shift, RIGHT*2,
                #text.scale, 2,
                #text.set_color, RED,
                run_time=2,
            )
		self.wait()

class Pythagoras(Scene):
	def construct(self):
		#cube_out= Square(fill_color = GOLD_B, fill_opacity=1, color=GOLD)
		#cube_in = Square(fill_color=BLACK, fill_opacity=1, color=BLACK)
		#cube_in.rotate(20*DEGREES)
		#self.play(ShowCreation(cube_out), ShowCreation(cube_in))
		#self.wait()
		#t = Polygon([2,2,0],
		#			[-2,0,0],
		#			[2,0,0],
		#			[2,2,0])
		#sidetext = TexMobject(r"a^2")
		#sidetext.next_to(t, LEFT)
		#self.play(ShowCreation(t), ShowCreation(sidetext))
		#self.wait()
		#grid = ScreenGrid()

		Heading=TextMobject("Pythagorean Visual Proof")
		Heading.to_edge(UL)
		self.play(Write(Heading), run_time=0.8)
		self.wait(0.5)

		'''First cube and filling in triangles'''
		cube_out = Polygon([-2,2,0],
						  [2,2,0],
						  [2,-2,0],
						  [-2,-2,0],
						   color=WHITE)

		cube_copy = cube_out.copy()

		cube_in = Polygon([-2, 2 - np.sqrt(2),0],
						  [-2 + np.sqrt(6),2,0],
						  [2, 2 - np.sqrt(6),0],
						  [2 - np.sqrt(6),-2,0],
						  color=RED
							)

		cube_in.shift(LEFT*4)

		triangle1 = Polygon([-2, 2, 0],
            [-2 + math.sqrt(6), 2, 0],
            [-2, 2 - math.sqrt(2), 0],
            color=YELLOW, fill_opacity=0.7
		)

		triangle2 = Polygon(
            [2, 2, 0],
            [-2 + math.sqrt(6), 2, 0],
            [2, 2 - math.sqrt(6), 0],
            color=YELLOW, fill_opacity=0.7
        )

		triangle3 = Polygon(
            [2, 2 - math.sqrt(6), 0],
            [2, -2, 0],
            [2 - math.sqrt(6), -2, 0],
            color=YELLOW, fill_opacity=0.7
        )

		triangle4 = Polygon(
            [-2, 2 - math.sqrt(2), 0],
            [-2, -2, 0],
            [2 - math.sqrt(6), -2, 0],
            color=YELLOW, fill_opacity=0.7
        )

		cube_small_left = Polygon(
								[-2 + np.sqrt(2.4),-2,0],
								[-2,-2,0],
								[-2, -2 + np.sqrt(2.4),0],
								[-2 + np.sqrt(2.4), -2 + np.sqrt(2.4),0],
								color=RED	
		)

		cube_small_left.shift(RIGHT*4)

		cube_small_right = Polygon(
								[2 - np.sqrt(6),2,0],
								[2,2,0],
								[2, 2 - np.sqrt(6),0],
								[2 - np.sqrt(6), 2 - np.sqrt(6),0],
								color=RED	
		)

		cube_small_right.shift(RIGHT*4)
		triangles = [triangle1, triangle2,triangle3,triangle4]
		for t in triangles:
			t.set_stroke(None, 1.5) #makes lines match with edges of cube
		
		self.play(ShowCreation(cube_out), ShowCreation(triangle1), ShowCreation(triangle2), ShowCreation(triangle3), ShowCreation(triangle4), run_time=0.8)
		self.wait(0.5)

		'''Make group of triangles and cube and move'''
		group_cube_and_triangles = Group(cube_out, triangle1,triangle2,triangle3,triangle4)

		self.play(ApplyMethod(group_cube_and_triangles.shift, LEFT*4),run_time=0.8)
		self.wait(0.5)
		self.add(cube_copy)
		cube_copy.shift(LEFT*4)
		self.play(ApplyMethod(cube_copy.shift, RIGHT*8),run_time=0.8)
		#self.play(ReplacementTransform(cube_out.copy(),cube_copy, buff=10))
		self.wait(0.5)

		'''Make triangles in the left cube'''
		triangle2.generate_target()
		triangle2.target.shift(RIGHT * (8 - math.sqrt(6)))

		triangle1.generate_target()
		triangle1.target = triangle2.target.copy().rotate(PI)

		triangle3.generate_target()
		triangle3.target.shift(RIGHT*8)

		triangle4.generate_target()
		triangle4.target = triangle3.target.copy().rotate(PI)

		self.play(MoveToTarget(triangle1.copy()), MoveToTarget(triangle2.copy()), MoveToTarget(triangle3.copy()), MoveToTarget(triangle4.copy()),run_time=0.8)
		self.wait(0.5)
		self.add(cube_in)
		self.add(cube_small_left)
		self.add(cube_small_right)
		self.play(ApplyMethod(cube_in.set_opacity, 0.7, run_time=0.8), ApplyMethod(cube_small_left.set_opacity, 0.7, run_time=0.8), ApplyMethod(cube_small_right.set_opacity, 0.7, run_time=0.8))
		self.wait(0.5)

		'''Insert formulas in cubes'''
		formula = TexMobject(r"a^2", r"=", r"b^2",r"+",r"c^2", color=BLUE).to_edge(DOWN)
		group_2 = VGroup(TexMobject(r"a^2").move_to(cube_in),
						TexMobject(r"b^2").move_to(cube_small_left),
						TexMobject(r"c^2").move_to(cube_small_right))
		self.play(Write(group_2),run_time=0.8)
		self.wait(0.5)
		self.play(Write(formula[1]), Write(formula[3]),ReplacementTransform(group_2[0].copy(), formula[0]),
				  ReplacementTransform(group_2[1].copy(), formula[2]),
				  ReplacementTransform(group_2[2].copy(),formula[4]),run_time=0.8)
		self.wait()

		# textend = TextMobject("Q.E.D")
		# textend.scale(3)
		# self.play(Write(textend))
		#self.add(grid)
		self.wait()



class testing(Scene):
	def construct(self):
		#T = Triangle()
		#T.set_color(YELLOW)
		#T.set_opacity(0.8)
		#self.play(ShowCreation(T))
		#self.wait()
		#s = Square()
		#s2 = s.copy()
		#self.add(s)
		#self.wait()
		#self.play(ApplyMethod(s.to_edge))
		#self.wait()
		#self.play(ReplacementTransform(s.copy(),s2))
		
		c = Square()
		#c.set_fill(color=RED,opacity=0.7)
		self.play(ApplyMethod(c.set_fill, RED, fill_opacity=0.7))
		self.wait()

		
		#s.copy()
		#self.play(ReplacementTransform(s.copy(),s))
		self.wait()
