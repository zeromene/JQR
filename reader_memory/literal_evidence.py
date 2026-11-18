"""Reversible source views. Literal retention is not proof of task sufficiency."""
import json

FIELDS = {'source_id', 'start', 'end', 'role', 'date', 'text'}


def validate(spans):
    if not isinstance(spans, list):
        raise ValueError('Expected source span list')
    for s in spans:
        if (not isinstance(s, dict) or set(s) != FIELDS
                or not isinstance(s['source_id'], str) or not s['source_id']
                or s['role'] not in {'user', 'assistant'} or not isinstance(s['date'], str)
                or type(s['start']) is not int or type(s['end']) is not int
                or not 0 <= s['start'] < s['end']
                or not isinstance(s['text'], str) or len(s['text']) != s['end'] - s['start']):
            raise ValueError('Invalid literal source span')
    return spans


def canonical(turns, spans):
    """Union only overlapping/touching ranges, preserving source order and gaps."""
    validate(spans)
    source = {t.source_id: t for t in turns}
    if len(source) != len(turns):
        raise ValueError('Duplicate native turn ID')
    ranges = {}
    for s in spans:
        t = source[s['source_id']]
        if (s['end'] > len(t.content) or t.content[s['start']:s['end']] != s['text']
                or s['role'] != t.role or s['date'] != t.date):
            raise ValueError('Source mismatch')
        ranges.setdefault(t.source_id, []).append((s['start'], s['end']))
    result = []
    for t in turns:
        merged = []
        for start, end in sorted(ranges.get(t.source_id, [])):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        result.extend({'source_id': t.source_id, 'role': t.role, 'date': t.date,
                       'start': a, 'end': b, 'text': t.content[a:b]} for a, b in merged)
    return result


def render(spans, view):
    validate(spans)
    if view == 'structured':
        return json.dumps({'source_spans': spans}, ensure_ascii=False, separators=(',', ':'))
    if view != 'raw':
        raise ValueError('Unknown view')
    # Payload length comes from source character offsets, so even fake headers
    # inside quoted dialogue cannot be mistaken for a new turn by the decoder.
    return ''.join('SOURCE ' + json.dumps({k: s[k] for k in ('source_id', 'role', 'date', 'start', 'end')},
        ensure_ascii=False, separators=(',', ':')) + '\n' + s['text'] + '\n' for s in spans)


def decode(text, view):
    if view == 'structured':
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) != {'source_spans'}:
            raise ValueError('Invalid structured envelope')
        return validate(obj['source_spans'])
    if view != 'raw':
        raise ValueError('Unknown view')
    spans = []; cursor = 0
    while cursor < len(text):
        if not text.startswith('SOURCE ', cursor):
            raise ValueError('Invalid source header')
        end = text.index('\n', cursor)
        meta = json.loads(text[cursor + 7:end])
        if not isinstance(meta, dict) or set(meta) != FIELDS - {'text'}:
            raise ValueError('Invalid source metadata')
        if type(meta['start']) is not int or type(meta['end']) is not int or meta['end'] <= meta['start']:
            raise ValueError('Invalid source length')
        begin = end + 1; finish = begin + meta['end'] - meta['start']
        if text[finish:finish+1] != '\n':
            raise ValueError('Truncated source payload')
        spans.append({**meta, 'text': text[begin:finish]}); cursor = finish + 1
    return validate(spans)


def includes(container, needed):
    validate(container); validate(needed)
    for s in needed:
        cursor = s['start']
        for a, b in sorted((x['start'], x['end']) for x in container if x['source_id'] == s['source_id']):
            if a <= cursor:
                cursor = max(cursor, b)
        if cursor < s['end']:
            return False
    return True


def characters(spans):
    return sum(len(s['text']) for s in spans)
