# `HiPr`
- A Python pipeline for extracting hierarchical propositions (HiPr) from texts.
	- *Propositions* are things that can be true or false. For example, the sentence "María ate an apple." represents a proposition that is true if and only if María ate an apple. Moreover, that same proposition can be expressed in different ways ("The apple was eaten by María.", "María comió la manzana.", etc.). They are also the kinds of things that are believed, asserted, hoped, etc. As such, they are a useful unit of analysis.
	- Propositions can also be thought of as *hierarchical* in the sense that they entail other propositions. For example, if is true that María ate an apple, then it is also true that María ate a fruit since all apples are fruit. Likewise, it is also true that someone at a fruit, that a fruit was eaten by someone, and so on. Our HiPr pipeline attempts to capture these hierarchical relationships.
- This repository contains two main pipelines (detailed run-throughs below)
	- [A pipeline to extract sentences of interest from a Constellate dataset](#sentence-extraction-pipeline)
	- [A pipeline to extract HiPr from those sentences](#proposition-extraction-pipeline)
- Each python script has a some documentation regarding what it's supposed to do and what the various parameters are--to access that, in your command line type `python [LOCATION OF THE SCRIPT] -h`.
## Run-through of pipelines
For the following diagrams, boxes represent data represented in a certain format and arrows represent processing steps (with an accompanying Python script, all of which are in [`src`](src)).
### Sentence extraction pipeline
![[sentence_pipeline.svg]]
For the purposes of our research, we used ITHAKA's [Constellate](https://www.constellate.org/) service to get full-text documents from academic journals (analysis forthcoming). This is the pipeline we used to extract sentences of interest from the raw formats provided by Constellate. If you have your own sentences, you can skip these steps.

1. *Constellate* data is provided in `.jsonl.gz`  files.
2. Using `wrangler.py`, we filter out documents by language, extract metadata, convert everything to strings, and clean the text (join pages, remove header and footer content between pages, etc.). These *cleaned documents* and metadata are saved in `parquet` datasets.
3. We then extract windows of text around certain terms of interest (TOIs) using `subsetter.py`, giving us *subsetted documents* that are a lot more efficient to process. We also use `citations_detector.py` to flag certain documents as being citations rather than prose, which we exclude from our analysis.
4. Finally, we use `sentencizer.py` to extract *sentences* from the subsetted documents. While the proposition extraction pipeline doesn't require inputs to be sentences, sentences are a natural unit of analysis (for more discussion, see forthcoming paper).
### Proposition extraction pipeline
![[processing_pipeline.svg]]
Once we have sentences, we can extract HiPr from them. Because each of these steps require more conceptual scaffolding to fully make sense of, we opt instead to provide a toy example for what each step looks like and direct you to our paper for more conceptual explication. The output from each step is also [included in the repository](vignettes\toy_dataset).

1. We start with *sentences* that were extracted with the previous pipeline. 
   
   In our toy example, we use the following three sentences: 
   (1) The teacher wrote on the board a question for the students to solve.
   (2) The school board voted unanimously to exploit the new funding opportunity.
   (3) The city council voted to cannibalize the arts budget to build a new parking garage.
2. We then apply semantic role labelling (SRL) with `srl.py` (an implementation of AllenNLP's SRL model). The output at this stage is (i) a tokenized representation of the sentence and (ii) a list of *verbs and the respective tokens of the sentence that correspond to each verbs' semantic roles*.
   
   For sentence (1), `srl.py` identifies two verbs: "wrote" and "solve". Each of those has its roles labeled in an [IOB format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)), which can be a little hard to understand, so I represent them using square bracket and non-technical labels instead:
   (1a) \[agent: The teacher] **\[verb: wrote]** \[location: on the board] \[patient: a question for the students to solve].
   (1b) The teacher wrote on the board \[patient: a question] for \[agent: the students] to **\[verb: solve]**.
   
   Across the three sentences, we find seven verbs and their respective roles.
3. At this point, however, these roles are still tied to phrases, e.g., the whole span "a question on the board for the students to solve" is the patient in (1a). This makes it challenging for us to find commonalities between, say, that and "the questions on the worksheet" because they are two different strings. As such, we use dependency parsing and a number of other tools to *annotate* these spans to get more information. Primarily, we use `srl_annotator.py` to determine what the content head of each span, i.e., the main thing it's about (e.g., the content head of "a question on the board ..." is "question"); the lemma of that content head (removing inflections like number, person, tense); the kind of entity it is (person, geopolitical entity, etc.); and so on. We then use `wsd.py` to also disambiguate potentially ambiguous terms. In our example, both (1) and (2) have the word "board" but they refer to different things. Using Wordnet and custom synsets, we were able to tell that the first was "a surface, frame, or device for writing or posting notices" while the second is "a committee having supervisory powers." The full impact of this will be obvious in the next step.
   
   Across the seven verbs, we annotate 22 roles.
4. Finally, we use `proposition_identifier.py` to recover the *propositions* from the annotated semantic roles. From 22 roles, we recover 262 propositions. This is represented in multiple ways, first, a verb-role-matrix (VRM) where each row is a use of a verb, each column is a semantic role, and the value is true if that use of the verb has that role. Each possible combination of columns that are labeled as true represents a proposition entailed by the verb. We then do that combinatorial work to get all the possible propositions, their frequencies, and finally a verb-proposition-matrix (VPM) where each row is a use of a verb, each column is a proposition, and the value is true if that use of the verb entails the proposition.
   
   Because the VPM is too large to display here (or to visually make sense of, for that matter), we can instead represent this VPM as a tree, leveraging the hierarchical nature of propositions. I explore one such tree from our example data in the next section.
## An example
In [the previous section](#proposition-extraction-pipeline), we put a simple corpus of three sentences through the HiPr pipeline. Here, we look at one way to visualize the output.

```
DFTree with 12 nodes

00: Raw propositions (n=7)
02: ├─── ARG0: administrative_unit.n.01 (n=4, 57.14% of parent, 57.14% of root)
03: │   ├─── V: vote.v.01 (n=2, 50.00% of parent, 28.57% of root)
04: │   └─── V: use.v.01 (n=2, 50.00% of parent, 28.57% of root)
05: │       ├─── ARG1: entity.n.01 and ARG1: abstraction.n.06 and ARG1: opportunity.n.01 (n=1, 50.00% of parent, 14.29% of root)
06: │       └─── ARG1: entity.n.01 and ARG1: abstraction.n.06 and ARG1: budget.n.01 (n=1, 50.00% of parent, 14.29% of root)
01: └─── Agent is not administrative unit or omitted (n=3, 42.86% of parent, 42.86% of root)
07:     ├─── Agent omitted (n=1, 33.33% of parent, 14.29% of root)
08:     ├─── ARG0: entity.n.01 and ARG0: person.n.01 and ARG0: causal_agent.n.01 and ARG0: student.n.01 (n=1, 33.33% of parent, 14.29% of root)
10:     │   └─── V: solve.v.01 (n=1, 100.00% of parent, 14.29% of root)
09:     └─── ARG0: entity.n.01 and ARG0: person.n.01 and ARG0: causal_agent.n.01 and ARG0: teacher.n.01 (n=1, 33.33% of parent, 14.29% of root)
11:         └─── V: write.v.07 (n=1, 100.00% of parent, 14.29% of root)
```

Using the `src/utils/DFTree.py` script, we can easily create trees and manipulate them. Node ids are the two-digit numbers at the start of each row.

Starting with the root node `00`, we have all seven raw propositions, i.e., the propositions as they were extracted from the texts. We then split those based on whether `ARG0: administrative_unit.n.01` (or, in human terms, the agent of the proposition is an administrative unit) is entailed. We find that 4 out of 7 propositions (57%) satisfy that condition. This is very interesting because 'administrative unit' isn't something that is explicit in the data. The reason why it comes up is because Wordnet lists the disambiguated forms of "council" and "(school) board" as being specific kinds of administrative units. Of course, the board that the teacher was writing on is not an administrative unit.

Moving on, we can then ask, within node `02` where all the propositions are about administrative units doing things, what are they doing? Here, we find they are split evenly between two verbs, 'voting' and 'using'. Just as with 'administrative unit', 'use' is identified as a hypernym of both "exploit" and "cannibalize," allowing us to see a connection between these two sentences. We can also then ask what kinds of things administrative things are using (splitting node `04`), which we discover to be opportunities and budgets (note that both are also labeled as entities and abstractions).

Finally, we turn to node `01`, which are the other three propositions where administrative units aren't doing stuff. We see that one has no agent, one is a student solving something, and the last one is a teacher writing something.

Notably, this is not the only tree we could've constructed--nodes can be split in many different ways based on what questions are salient to the researcher, making the exploration process intentional and principled. It can also be motivated by frequencies (e.g., splitting based on the most common agents).

While this example is small, it shows how much of the information from the three sentences is retained in the VPM structure. Similarly, larger corpora can be represented as VPMs, allowing researchers to traverse the large space of meanings by asking specific questions about who is doing what to whom, when, where, and how.