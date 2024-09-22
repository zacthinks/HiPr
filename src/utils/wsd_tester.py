import argparse

import src.scripts.model.wordnet_consec as consec


def main():
    args = parse_args()

    mt = consec.get_consec_module_tokenizer(args.consec_location,
                                            args.cuda_device)
    if args.wordnet_extension_location:
        wn_extension = consec.get_wn_extension(args.wordnet_extension_location)
    else:
        wn_extension = None

    while True:
        sentence = input(" Sentence: ")
        if sentence == '':
            break
        index = int(input(" Index: "))
        result = consec.wordnet_consec(mt, target_position=index, sentence=sentence,
                                       use_lemmas=True, use_both=True,
                                       wn_extension=wn_extension, full_consec=False, n=3)
        print(result)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="wordnet_wsd",
        description="Tester for the word sense disambiguator using wordnet and consec.")
    parser.add_argument('consec_location', type=str,
                        help="location of consec checkpoint")
    parser.add_argument('--wordnet_extension_location', '-x', type=str, nargs='?',
                        help="location wordnet extensions)")
    parser.add_argument('--cuda_device', '-c', nargs='?', type=int, default=0,
                        help="cuda device to use, -1 for CPU (default: 0)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
