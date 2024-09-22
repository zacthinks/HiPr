from allennlp_models import pretrained


def main():
    predictor = pretrained.load_predictor("structured-prediction-srl-bert", cuda_device=-1)

    while True:
        sentence = input(" Sentence: ")
        if sentence == '':
            break

        srl = predictor.predict(sentence)

        print("\nTokenized sentence:", srl['words'])
        print(f"Found {len(srl['verbs'])} verb{'' if len(srl['verbs']) == 1 else 's'}:")
        for verb_dict in srl['verbs']:
            print("Verb:", verb_dict['verb'])
            print("Roles:", verb_dict['description'])


if __name__ == '__main__':
    main()
