import schemdraw
import schemdraw.elements as elm
from schemdraw import flow


def main():
    with schemdraw.Drawing(file="pipeline.svg") as d:
        label_fontsize = 10
        arrow_length = d.unit * 2 / 3

        flow.Data().label("Constellate")
        flow.Arrow().right(arrow_length).label("cleaner.py", fontsize=label_fontsize)
        flow.Box().label("Cleaned docs")
        flow.Arrow().right(arrow_length).label("subsetter.py", fontsize=label_fontsize)
        flow.Box().label("Subsetted docs")
        flow.Arrow().right(arrow_length).label("sentencizer.py", fontsize=label_fontsize)
        s = flow.Box().label("Sentences")

        elm.Wire("|-", arrow="->").at(s.N).to((s.N[0] + arrow_length, s.N[1] + d.unit / 3)).label("srl.py", fontsize=label_fontsize)
        pn = flow.Box().label("Protonarratives")

        elm.Wire("|-", arrow="->").at(s.S).to((s.N[0] + arrow_length + .28431, s.S[1] - d.unit / 3)).label("dp.py", fontsize=label_fontsize)
        dp = flow.Box().label("Dependency\nparses")

        joint_loc = (dp.E[0] + d.unit / 3, dp.E[1] + d.unit / 3 + 1)
        elm.Wire("-|").at(dp.E).to(joint_loc)
        elm.Wire("-|").at(pn.E).to(joint_loc)

        flow.Arrow().right(arrow_length).label("annotator.py", fontsize=label_fontsize)
        flow.Box().label("Annotated\nprotonarratives")
        flow.Arrow().right(arrow_length).label("clusterer.py", fontsize=label_fontsize)
        flow.Box().label("Protonarrative\nclusters")


if __name__ == '__main__':
    main()
