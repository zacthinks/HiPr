import schemdraw
import schemdraw.elements as elm
from schemdraw import flow


def main():
    with schemdraw.Drawing(file="pipeline.svg") as d:
        flow.Data().label("Constellate")
        flow.Arrow().right(d.unit * 2 / 3).label("cleaner.py", fontsize=10)
        flow.Box().label("Cleaned docs")
        flow.Arrow().right(d.unit * 2 / 3).label("subsetter.py", fontsize=10)
        flow.Box().label("Subsetted docs")
        flow.Arrow().right(d.unit * 2 / 3).label("sentencizer.py", fontsize=10)
        s = flow.Box().label("Sentences")

        elm.Wire("|-", arrow="->").at(s.N).to((s.N[0] + d.unit * 2 / 3, s.N[1] + d.unit / 3)).label("srl.py", fontsize=10)
        pn = flow.Box().label("Protonarratives")

        elm.Wire("|-", arrow="->").at(s.S).to((s.N[0] + d.unit * 2 / 3 + .28431, s.S[1] - d.unit / 3)).label("dp.py", fontsize=10)
        dp = flow.Box().label("Dependency\nparses")

        joint_loc = (dp.E[0] + d.unit / 3, dp.E[1] + d.unit / 3 + 1)
        elm.Wire("-|").at(dp.E).to(joint_loc)
        elm.Wire("-|").at(pn.E).to(joint_loc)

        flow.Arrow().right(d.unit * 2 / 3).label("annotator.py", fontsize=10)
        flow.Box().label("Annotated\nprotonarratives")
        flow.Arrow().right(d.unit * 2 / 3).label("clusterer.py", fontsize=10)
        flow.Box().label("Protonarrative\nclusters")

        dp.at(pn.S)


if __name__ == '__main__':
    main()
