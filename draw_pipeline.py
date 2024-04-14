import schemdraw
import schemdraw.elements as elm
from schemdraw import flow


def width(drawing):
    return drawing.E[0] - drawing.W[0]


def main():
    with schemdraw.Drawing(file="pipeline.svg") as d:
        label_fontsize = 10
        arrow_length = d.unit * 2 / 3

        flow.Data().label("Constellate")
        flow.Arrow().right(arrow_length).label("wrangler.py", fontsize=label_fontsize)
        flow.Box().label("Cleaned\ndocs")
        flow.Arrow().right(arrow_length).label("subsetter.py", fontsize=label_fontsize)
        sd = flow.Box().label("Subsetted\ndocs")
        flow.Arrow().right(arrow_length).label("sentencizer.py", fontsize=label_fontsize)
        s = flow.Box().label("Sentences")

        (elm.ArcLoop(arrow='->', radius=.75)
            .at(sd.NNW)
            .to(sd.NNE)
            .label("citations_detector.py", fontsize=label_fontsize, halign='center'))

        elm.Wire("|-", arrow="->").at(s.N).to((s.N[0] + arrow_length, s.N[1] + d.unit / 3)).label("srl.py", fontsize=label_fontsize)
        rp = flow.Box().label("Raw atomic\npropositions")

        elm.Wire("|-", arrow="->").at(s.S).to((s.N[0] + arrow_length, s.S[1] - d.unit / 3)).label("dp.py", fontsize=label_fontsize)
        dp = flow.Box().label("Dependency\nparses")

        joint_loc = (dp.E[0] + d.unit / 3, dp.E[1] + d.unit / 3 + 1)
        elm.Wire("-|").at(dp.E).to(joint_loc)
        elm.Wire("-|").at(rp.E).to(joint_loc)

        flow.Arrow().right(arrow_length).label("annotator.py", fontsize=label_fontsize)
        flow.Box().label("Annotated atomic\npropositions")
        flow.Arrow().right(arrow_length).label("clusterer.py", fontsize=label_fontsize)
        flow.Box().label("Atomic proposition\nclusters")

        # print((width(rp) - width(dp)) / 2) # For adjusting dp position


if __name__ == '__main__':
    main()
