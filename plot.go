package mbpe

import (
	"image/color"
	"log"
	"sort"

	"gonum.org/v1/gonum/interp"
	pl "gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type PlotData struct {
	xs     []float64
	ys     []float64
	line   bool
	spline bool
	label  string
	color  color.RGBA
}

func NewPlotData(xs, ys []float64, line, spline bool, label string, color color.RGBA) PlotData {
	return PlotData{
		xs:     xs,
		ys:     ys,
		line:   line,
		spline: spline,
		label:  label,
		color:  color,
	}
}

func (p PlotData) Len() int {
	return len(p.xs)
}

func (p PlotData) Less(i, j int) bool {
	return p.xs[i] < p.xs[j]
}

func (p PlotData) Swap(i, j int) {
	p.xs[i], p.xs[j] = p.xs[j], p.xs[i]
	p.ys[i], p.ys[j] = p.ys[j], p.ys[i]
}

func (p PlotData) XY(i int) (x, y float64) {
	return p.xs[i], p.ys[i]
}

func Plot(data []PlotData, rangeX, rangeY [2]float64, labelX, labelY string) {
	p := pl.New()

	p.X.Min = rangeX[0]
	p.X.Max = rangeX[1]
	p.Y.Min = rangeY[0]
	p.Y.Max = rangeY[1]

	p.X.Label.Text = labelX
	p.Y.Label.Text = labelY

	drawSpline := func(s PlotData) {
		sort.Sort(s)

		predictor := interp.FittablePredictor(&interp.AkimaSpline{})

		if err := predictor.Fit(s.xs, s.ys); err != nil {
			log.Fatal(err)
		}

		line := plotter.NewFunction(func(x float64) float64 {
			return predictor.Predict(x)
		})

		line.Color = s.color
		line.Dashes = []vg.Length{vg.Points(2), vg.Points(2)}

		p.Add(line)

		p.Legend.Add(s.label, line)
	}

	drawScatter := func(s PlotData) {
		if scatter, err := plotter.NewScatter(s); err != nil {
			log.Fatal(err)
		} else {
			scatter.Color = s.color
			scatter.Radius = 2

			p.Add(scatter)

			p.Legend.Add(s.label, scatter)
		}

		if line, err := plotter.NewLine(s); err != nil {
			log.Fatal(err)
		} else {
			line.Color = s.color

			p.Add(line)
		}
	}

	for _, s := range data {
		if s.spline {
			drawSpline(s)

			continue
		}

		drawScatter(s)
	}

	if err := p.Save(8*vg.Inch, 5*vg.Inch, "assets/plot.svg"); err != nil {
		log.Fatal(err)
	}
}
