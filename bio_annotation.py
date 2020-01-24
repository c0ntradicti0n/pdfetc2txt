import regex as re
import itertools
import more_itertools
from collections import Counter
from typing import Dict, List, Any
from helpers.color_logger import *
from helpers.nested_dict_tools import flatten


conll_line = re.compile(r"([^\s]+)  ([^\s]+)  ([^\s]+)  ([^\s]+)")
only_abc = re.compile('[^a-zA-Z0-9]|_')


def re_replace_abc(string):
    return re.sub(only_abc, '', string)


dots = re.compile('[,\.;:?!]')


def re_replace_dot(string):
    return re.sub(dots, 'PCT', string)


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return [(s, e + 1) for s, e in zip(edges, edges)]


def not_between(annotation):
    start = 0
    for tok, tag in annotation:
        if tag== 'O':
            start+=1
        else:
            break
    stop = len(annotation)
    for tok, tag in annotation[::-1]:
        if tag=='O':
            stop-=1
        else:
            break
    return annotation[min([start, stop]): max([start,stop])]


class BIO_Annotation:
    def read_annotation_from_corpus(path, different_only=False, yet=[]):
        with open(path, 'r+') as f:
            text = f.read()
            ##lines = f.readlines()
        samples = text.split("\n\n") #"[list(group) for k, group in itertools.groupby(lines[2:], lambda x: x.strip() == "") if not k]

        for sample in samples:
            if sample.replace("\n","").strip() =="" or sample == '-DOCSTART- -X- -X- O':
                continue
            try:
                sample = sample.replace('-DOCSTART- -X- -X- O','')
                tok, pos, pos_tag, tag = list(zip(*[l.strip().split('  ') for l in sample.strip().split("\n") if l.strip() ]))
            except Exception as e:
                print (f"conll3 reading error in {path} on sample\n '{str(sample)}'")

                raise
            res = list(zip(tok, tag))
            if different_only:
                actual_text = " ".join(tok for tok, tag in not_between(res))
                if actual_text in yet:
                    continue
                else:
                    yet.append(actual_text)
            yield res

    def snippets_from_annotation(annotation):
        BIO_Annotation.compute_structured_spans(annotation)

    def token_pos_tag_to_conll3(combos):
        conll_lines = []
        span_delims = []

        for i, ((token, tag), pos) in enumerate(combos):

            if not pos:
                # bad tag for spacy
                pos = "UKN"
            elif pos == 'SP':
                # del newline tokens

                continue

            span_delims.append(tag[0])

            pos_tag = "-".join([tag[0], pos] if tag[0] != 'O' else 'O')
            line = "  ".join([token, pos, pos_tag, tag])
            if not conll_line.match(line):
                raise AssertionError
            conll_lines.append(line)

        assert (all(sd in ['B', 'I', 'O'] for sd in span_delims))
        count = Counter(span_delims)
        if not count['B'] == 2:
            logging.error('Annotation contains wrong number of spanning tags!!! %s' % str(count))
            return None
        assert count['B'] == 2

        return conll_lines

    def bioul_to_bio(bioul_tags):
        """ Make BIOUL coding scheme to BIO

        >>> annotations = [('The', 'O'), ('hormones', 'O'), ('’', 'O'), ('two', 'O'), ('classifications', 'O'), ('are', 'O'), ('“', 'O'), ('amino', 'O'), ('acid', 'O'), ('-', 'O'), ('based', 'O'), ('and', 'O'), ('steroids', 'O'), ('”', 'U-SUBJECT'), ('.', 'O'), ('As', 'O'), ('for', 'O'), ('neurotransmitters', 'O'), (',', 'O'), ('it', 'O'), ('can', 'O'), ('be', 'O'), ('classified', 'O'), ('according', 'O'), ('to', 'O'), ('ion', 'O'), ('flow', 'O'), ('facilitation', 'B-CONTRAST'), (':', 'I-CONTRAST'), ('“', 'I-CONTRAST'), ('excitatory', 'I-CONTRAST'), ('and', 'I-CONTRAST'), ('inhibitory', 'L-CONTRAST'), ('”', 'U-SUBJECT'), ('and', 'B-CONTRAST'), ('according', 'I-CONTRAST'), ('to', 'I-CONTRAST'), ('structure', 'I-CONTRAST'), ('(', 'I-CONTRAST'), ('chemical', 'I-CONTRAST'), ('or', 'I-CONTRAST'), ('molecular', 'I-CONTRAST'), (')', 'I-CONTRAST'), (':', 'I-CONTRAST'), ('“', 'I-CONTRAST'), ('small', 'I-CONTRAST'), ('molecule', 'I-CONTRAST'), ('and', 'I-CONTRAST'), ('neuropeptides', 'I-CONTRAST'), ('”', 'L-CONTRAST'), ('.', 'O')]
        >>> tags = [x[1] for x in annotations]
        >>> tags
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-SUBJECT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'L-CONTRAST', 'U-SUBJECT', 'B-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'L-CONTRAST', 'O']

        >>> list(Model.bioul_to_bio(tags))
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-SUBJECT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'B-SUBJECT', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'O']

        >>> tags = ['B-SUBJECT', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'O', 'B-SUBJECT', 'I-SUBJECT', 'B-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'O']
        >>> list(Model.bioul_to_bio(tags))
['B-SUBJECT', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'O', 'B-SUBJECT', 'I-SUBJECT', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'I-CONTRAST', 'O']

        :param bioul_tags:
        :return:
        """
        assert (all([x[0] in ['B', 'I', 'O', 'U', 'L'] for x in bioul_tags]))
        started = False
        for t in bioul_tags:
            span_tag, _, annotag = t.partition('-')

            if span_tag in ['O']:
                started = False

            if annotag == 'CONTRAST':
                if span_tag in ['U']:
                    if started:
                        span_tag = 'I'
                    else:
                        started = True
                        span_tag = 'B'
                elif span_tag in ['B']:
                    if started:
                        span_tag = 'I'
                    else:
                        started = True
                        span_tag = 'B'  # is already 'B', but for completeness
                elif span_tag in ['L']:
                    span_tag = 'I'
                    started = False

            elif annotag == 'SUBJECT':
                if span_tag in ['U']:
                    if started:
                        span_tag = 'I'
                    else:
                        started = True
                        span_tag = 'B'

                elif span_tag in ['B']:
                    if started:
                        span_tag = 'I'
                    else:
                        started = True

                elif span_tag in ['L']:
                    span_tag = 'I'

            yield "".join([span_tag, _, annotag])

    def annotation_to_spans(annotations):
        ''' Get spans bases on spans annotated with the BIOL tagging scheme

        >>> annotation = [  # Some strange prediction from the model
        ...    ('The', 'O'), ('hormones', 'O'), ('’', 'O'), ('two', 'O'), ('classifications', 'O'),
        ...    ('are', 'O'), ('“', 'O'), ('amino', 'O'), ('acid', 'O'), ('-', 'O'), ('based', 'O'),
        ...    ('and', 'O'), ('steroids', 'O'), ('”', 'U-SUBJECT'), ('.', 'O'), ('As', 'O'), ('for', 'O'),
        ...    ('neurotransmitters', 'O'), (',', 'O'), ('it', 'O'), ('can', 'O'), ('be', 'O'), ('classified', 'O'),
        ...    ('according', 'O'), ('to', 'O'), ('ion', 'O'), ('flow', 'O'), ('facilitation', 'B-CONTRAST'),
        ...    (':', 'I-CONTRAST'), ('“', 'I-CONTRAST'), ('excitatory', 'I-CONTRAST'), ('and', 'I-CONTRAST'),
        ...    ('inhibitory', 'L-CONTRAST'), ('”', 'U-SUBJECT'), ('and', 'B-CONTRAST'), ('according', 'I-CONTRAST'),
        ...    ('to', 'I-CONTRAST'), ('structure', 'I-CONTRAST'), ('(', 'I-CONTRAST'), ('chemical', 'I-CONTRAST'),
        ...    ('or', 'I-CONTRAST'), ('molecular', 'I-CONTRAST'), (')', 'I-CONTRAST'), (':', 'I-CONTRAST'),
        ...    ('“', 'I-CONTRAST'), ('small', 'I-CONTRAST'), ('molecule', 'I-CONTRAST'), ('and', 'I-CONTRAST'),
        ...    ('neuropeptides', 'I-CONTRAST'), ('”', 'L-CONTRAST'), ('.', 'O')
        ...    ]
        >>> annotation = [('The', 'B-CONTRAST'), ('queen', 'I-SUBJECT'), ('’s', 'I-SUBJECT'), ('crown', 'I-SUBJECT'), (',', 'I-CONTRAST'), ('also', 'I-CONTRAST'), ('referred', 'I-CONTRAST'), ('to', 'I-CONTRAST'), ('the', 'I-CONTRAST'), ('Royal', 'I-CONTRAST'), ('crown', 'I-CONTRAST'), ('is', 'I-CONTRAST'), ('made', 'I-CONTRAST'), ('with', 'I-CONTRAST'), ('depressed', 'I-CONTRAST'), ('arches', 'I-CONTRAST'), ('.', 'O'), ('The', 'B-CONTRAST'), ('king', 'I-SUBJECT'), ('’s', 'I-SUBJECT'), ('crown', 'I-SUBJECT'), ('also', 'I-CONTRAST'), ('called', 'I-CONTRAST'), ('the', 'I-CONTRAST'), ('Imperial', 'I-CONTRAST'), ('crown', 'I-CONTRAST'), (',', 'I-CONTRAST'), ('on', 'I-CONTRAST'), ('the', 'I-CONTRAST'), ('other', 'I-CONTRAST'), ('hand', 'I-CONTRAST'), (',', 'I-CONTRAST'), ('has', 'I-CONTRAST'), ('arches', 'I-CONTRAST'), ('that', 'I-CONTRAST'), ('rise', 'I-CONTRAST'), ('to', 'I-CONTRAST'), ('the', 'I-CONTRAST'), ('centre', 'I-CONTRAST'), ('.', 'O')]
        >>> import pprint
        >>> pprint.pprint(list(BIO_Annotation.annotation_to_spans(annotation))) # doctest: +NORMALIZE_WHITESPACE
                [('CONTRAST',
                  (27, 33),
                  [(27, ('facilitation', 'B-CONTRAST')),
                   (28, (':', 'I-CONTRAST')),
                   (29, ('“', 'I-CONTRAST')),
                   (30, ('excitatory', 'I-CONTRAST')),
                   (31, ('and', 'I-CONTRAST')),
                   (32, ('inhibitory', 'L-CONTRAST'))]),
                 ('SUBJECT', (33, 34), [(33, ('”', 'U-SUBJECT'))]),
                 ('CONTRAST',
                  (34, 50),
                  [(34, ('and', 'B-CONTRAST')),
                   (35, ('according', 'I-CONTRAST')),
                   (36, ('to', 'I-CONTRAST')),
                   (37, ('structure', 'I-CONTRAST')),
                   (38, ('(', 'I-CONTRAST')),
                   (39, ('chemical', 'I-CONTRAST')),
                   (40, ('or', 'I-CONTRAST')),
                   (41, ('molecular', 'I-CONTRAST')),
                   (42, (')', 'I-CONTRAST')),
                   (43, (':', 'I-CONTRAST')),
                   (44, ('“', 'I-CONTRAST')),
                   (45, ('small', 'I-CONTRAST')),
                   (46, ('molecule', 'I-CONTRAST')),
                   (47, ('and', 'I-CONTRAST')),
                   (48, ('neuropeptides', 'I-CONTRAST')),
                   (49, ('”', 'L-CONTRAST'))])]
        >>> annotation = [["The", "B-CONTRAST"], ["key", "I-CONTRAST"], ["difference", "I-CONTRAST"], ["between", "I-CONTRAST"], ["distillation", "I-CONTRAST"], ["and", "I-CONTRAST"], ["condensation", "I-CONTRAST"], ["is", "O"], ["that", "O"], ["the", "B-CONTRAST"], ["distillation", "I-SUBJECT"], ["is", "I-CONTRAST"], ["a", "I-CONTRAST"], ["separation", "I-CONTRAST"], ["technique", "I-CONTRAST"], ["whereas", "O"], ["the", "B-CONTRAST"], ["condensation", "I-SUBJECT"], ["is", "I-CONTRAST"], ["a", "I-CONTRAST"], ["process", "I-CONTRAST"], ["of", "I-CONTRAST"], ["changing", "I-CONTRAST"], ["the", "I-CONTRAST"], ["phase", "I-CONTRAST"], ["of", "I-CONTRAST"], ["matter", "I-CONTRAST"], [".", "O"]]
        >>> pprint.pprint(list(BIO_Annotation.annotation_to_spans(annotation))) # doctest: +NORMALIZE_WHITESPACE

        :param annotations: list of words and tags
        :return:
        '''

        # Divide by 'B'eginning tags
        parts = list(BIO_Annotation.compute_parts(annotations))

        # Get the spans from the parts: parts mean, that if the 'B' tag appeared, all tokens, that come until the next 'B'
        # can are of the same kind

        yield from BIO_Annotation.spans_from_partitions_flat(parts)

    def compute_structured_spans(annotations):

        parts = list(BIO_Annotation.compute_parts(annotations))
        # get single spans within these parts
        subordinated_spans = list(BIO_Annotation.spans_from_partitions_nested(parts))
        if not all(len(i) == len(subordinated_spans[0]) for i in subordinated_spans):
            logging.warning('different len of annotation parts')

        # nur subjekt oder mit beginning in der mitte

        # Divide by 'B'eginning tags
        parts = list(BIO_Annotation.compute_parts(annotations))

        # get single spans within these parts
        subordinated_spans = list(BIO_Annotation.spans_from_partitions_nested(parts))



        return subordinated_spans

    def apply_on_side(structured_spans, f):
        for part in structured_spans:
            yield f(part)

    def apply_on_kinds(structured_spans, f):
        sorted_by_kinds = itertools.groupby(sorted(more_itertools.flatten(structured_spans), key=lambda x: x[0]), key=lambda x: x[0])
        for kind, group in sorted_by_kinds:
            yield f(list(group))

    def structured_span_to_text(span):
        return " ".join(w[1][0] for w in span[2])

    def compute_parts(annotations):
        return [part
                for part in more_itertools.split_before(enumerate(annotations), lambda x: x[1][1][0] == 'B')
                if part[0][1][1][0] == 'B']

    def spans_from_partitions_flat(parts):
        for part in parts:
            spans = BIO_Annotation.spans_from_part(part)
            yield from spans

    def spans_from_partitions_nested(parts):
        for part in parts:
            spans = list(BIO_Annotation.spans_from_part(part))
            yield spans

    def kind_from_tag(tag):
        try:
            return tag[1][1][2:]
        except IndexError:
            raise IndexError

    def spans_from_part(part):
        part = sorted(part, key=BIO_Annotation.kind_from_tag)
        partitions = {
            kind: list(tokens)
            for kind, tokens in itertools.groupby(part, key=BIO_Annotation.kind_from_tag)
        }
        for kind, tokens in partitions.items():
            if kind not in ['O', '']:
                positions = sorted([t[0] for t in tokens])

                yield (kind, (min(positions), max(positions) + 1), tokens)

    def pair_spans(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        indices = sorted(list(set(flatten([list(range(d['start'], d['end'])) for d in spans if d['able']]))))
        rs = ranges(indices)
        paired_spans = [
            [
                d
                for d in spans
                if d['start'] >= r_start and d['end'] <= r_end
                   and
                d['able']
            ]
            for r_start, r_end in rs
        ]

        return paired_spans

    importance_list = ['SUBJECT','ASPECT','CONTRAST',
                            'SUBJECT_EXCEPT', 'CONTRAST_MARKER','COMPARISON_MARKER', '']


    def annotation_from_spans(tokens: List[str], paired_spans: List[List[Dict[str, Any]]]):
        sorted_paired_spans = sorted(paired_spans,
                                     key=lambda l:
                                     min(l, key=lambda x: x['start'])['start'])

        all_tags = [[]] * len(tokens)
        # read list of spans backwards to overwrie the contrast with the subject tags
        for spans in sorted_paired_spans:

            beginning = min(spans, key=lambda x: x['start'])['start']
            for d in spans[::-1]:
                these_tags = ['O'] * len(tokens)

                if d['able']:
                    if not (d['end'] < len(tokens)):
                        import pprint
                        logging.error('Exceeding annotation length')
                        length = len(tokens)
                        logging.error('length %d' % length)
                        logging.error(pprint.pformat(paired_spans))
                    for i in range(d['start'], d['end']):
                        these_tags[i] = "-".join(['B' if i == beginning else 'I', d['kind']])
                all_tags = [x + [y] for x, y in zip(all_tags, these_tags)]

        print ('ALL TAGS', all_tags)
        tags = [max(row_tags, key=lambda x: - BIO_Annotation.importance_list.index(x[2:]))
                for row_tags in all_tags]

        annotation = list(zip(tokens, tags))
        return annotation
