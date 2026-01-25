import pyvista as pv

import field

# Cilindro transparente
# cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=r_b, height=L)
# plotter.add_mesh(cyl, color="blue", opacity=0.1)


class Plot:
    # ------------------------------------------------------------
    # Plot com tamanho de seta constante
    # ------------------------------------------------------------

    def __init__(self, glyph_size=2e-3):
        self.glyph_size = glyph_size
        self.field = field.Field()
        self.plotter = pv.Plotter()

    def show(self):
        points, vectors_unit, mag = self.field.calculate_field()

        cloud = pv.PolyData(points)
        cloud["vectors"] = vectors_unit
        cloud["mag"] = mag

        glyphs = cloud.glyph(
            orient="vectors",
            scale=False,  # <<--- TAMANHO FIXO
            factor=self.glyph_size,
        )

        self.plotter.add_mesh(
            glyphs,
            scalars="mag",
            cmap="viridis",
            scalar_bar_args={"title": "|E| [V/m]"},
        )

        self.plotter.add_axes()
        self.plotter.show()
