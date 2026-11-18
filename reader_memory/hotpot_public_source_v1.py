"""Public-only HotpotQA context adapter; does not allocate a panel or read labels."""
from .replacement_core_v1 import sentence_spans


def validate_public(source):
    if set(source) != {'question_id', 'native_id', 'mode', 'question', 'paragraphs'}:
        raise ValueError('Unregistered public source fields')
    if source['mode'] != 'multi' or not isinstance(source['question'], str) or not source['question'].strip():
        raise ValueError('Invalid public question')
    paragraphs = source['paragraphs']
    # Native distractor validation has 2--10 contexts; never pad or duplicate to 20.
    if not 2 <= len(paragraphs) <= 10:
        raise ValueError('Expected native HotpotQA distractor contexts')
    if [p['source_id'] for p in paragraphs] != [str(i) for i in range(len(paragraphs))]:
        raise ValueError('Source order or IDs changed')
    if len({p['title'] for p in paragraphs}) != len(paragraphs):
        raise ValueError('Duplicate native title')
    for p in paragraphs:
        if set(p) != {'source_id', 'title', 'timestamp', 'text', 'spans'}:
            raise ValueError('Unregistered paragraph fields')
        if not isinstance(p['title'], str) or not p['title'].strip() or not isinstance(p['text'], str):
            raise ValueError('Invalid source text or title')
        if p['timestamp'] != 'undated' or not p['text'].strip():
            raise ValueError('Invalid timestamp or empty context')
        if p['spans'] != sentence_spans(p['text']):
            raise ValueError('Sentence provenance or segmentation changed')


def from_public_row(row):
    """Accept exactly id/question/context, after the caller excludes private columns."""
    if set(row) != {'id', 'question', 'context'}:
        raise ValueError('Pass only id, question, context; labels must stay outside this adapter')
    if not isinstance(row['id'], str) or not row['id'] or not isinstance(row['context'], dict):
        raise ValueError('Invalid native ID or context schema')
    context = row['context']
    if set(context) != {'title', 'sentences'} or len(context['title']) != len(context['sentences']):
        raise ValueError('Native context columns do not align')
    paragraphs = []
    for i, (title, sentences) in enumerate(zip(context['title'], context['sentences'])):
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise ValueError('Expected native sentence strings')
        # Same exact concatenation as the source-overlap audit. Sentence strings remain intact.
        text = ' '.join(sentences)
        paragraphs.append(dict(source_id=str(i), title=title, timestamp='undated',
                               text=text, spans=sentence_spans(text)))
    source = dict(question_id='hotpot:' + row['id'], native_id=row['id'], mode='multi',
                  question=row['question'], paragraphs=paragraphs)
    validate_public(source)
    return source


def model_payload(source):
    validate_public(source)
    return dict(question=source['question'], sources=[dict(id=p['source_id'], title=p['title'], text=p['text'])
                                                    for p in source['paragraphs']])
