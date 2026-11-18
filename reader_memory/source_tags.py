"""Bounded, query-blind source cues; no oracle or semantic identity inference."""
from __future__ import annotations

import re

from .schema import fingerprint

LEGEND = ('Source cues only interpret literal sources shown below. withdraw_preference means stop using that '
          'preference for personalization, not dislike or a ban. external_quotation and assistant text are not '
          'automatically the user\'s preferences. text_operation has unresolved attribution. '
          'Unmarked material is unresolved, not a self-fact assertion. Historical assertions are not guaranteed current.\n')
VERBS = r'like|love|enjoy|prefer'
SELF = re.compile(rf'I\s+(?P<verb>{VERBS})\s+(?P<object>[^.!?\n]+)\.?',re.I)
FORGET = re.compile(rf'(?:please\s+)?forget\s+that\s+I\s+(?P<verb>{VERBS})\s+(?P<object>[^.!?\n]+)\.?',re.I)
OTHER = re.compile(rf'(?P<owner>my\s+(?:friend|sister|brother|mother|father|partner|wife|husband|colleague))\s+'
                   rf'(?P<verb>likes|loves|enjoys|prefers)\s+(?P<object>[^.!?\n]+)\.?',re.I)
COMPLEX = re.compile(r'[,;:"“”]|\b(?:and|or|but|if|unless|except|until|when|not|never|no|sometimes|maybe|'
                     r'this|that|these|those|it|them|his|her|their|its)\b',re.I)
OPERATION = re.compile(r'\b(?:translate|rewrite|rephrase|polish|proofread|edit)\b',re.I)
REQUEST = re.compile(r'^(?:can|could|would|will)\s+you\b|^please\b|^help\s+me\b',re.I)
EXTERNAL = re.compile(r'\bI\s+(?:came\s+across|found|read|saw)\b[^\n.!?]{0,160}\b(?:article|book|passage|text|magazine)\b',re.I)
OWN_MATERIAL = re.compile(r'\b(?:my|our|own|wrote|authored)\b|\bwritten\s+by\s+me\b',re.I)


def normalize_object(text):
    return ' '.join(text.casefold().split())


def citation(turn,start,end):
    if not 0<=start<end<=len(turn.content): raise ValueError('Invalid cue source range')
    return {'source_id':turn.source_id,'start':start,'end':end,'text':turn.content[start:end]}


def _event(turn,kind,owner,state,support,scope,**extra):
    value = {'kind':kind,'owner':owner,'state':state,'support':support,'scope':scope,**extra}
    return {'event_id':'e'+fingerprint(value)[:20],**value}


def extract_turn(turn):
    if turn.role!='user': return []
    text = turn.content;stripped = text.strip();offset = len(text)-len(text.lstrip())
    for pattern,kind,owner,state in ((FORGET,'forget_request','self','withdraw_preference'),
            (SELF,'direct_preference','self','asserted_at_source'),(OTHER,'other_report','explicit_other','reported_not_self')):
        match = pattern.fullmatch(stripped)
        if match:
            obj = match['object'].strip()
            if not 2<=len(re.findall(r'[A-Za-z]+',obj))<=12 or COMPLEX.search(obj): return []
            start = offset+match.start('object');end = offset+match.end('object')
            while start<end and text[start].isspace(): start+=1
            while end>start and text[end-1].isspace(): end-=1
            support = citation(turn,offset,offset+len(stripped))
            scope = citation(turn,start,end)
            return [_event(turn,kind,owner,state,[support],scope,rule='whole-user-single-clause-v1',
                verb=match['verb'].casefold(),object_key=normalize_object(scope['text']),
                owner_literal=match['owner'] if kind=='other_report' else 'I')]
    # Only the request frame can justify a text-operation cue. Material alone
    # never supplies an external origin or a self/other preference assertion.
    paragraph = text.find('\n\n')
    quote = re.search(r'["“]',text)
    boundary = paragraph
    if quote and (paragraph<0 or quote.start()<paragraph):
        boundary = quote.start()
        closing = '”' if quote.group()=='“' else '"'
        if text.find(closing,boundary+1)<0: return []
    if boundary<=0 or boundary>512: return []
    frame = text[:boundary].strip()
    if not REQUEST.search(frame) or not OPERATION.search(frame): return []
    frame_start = text.index(frame);support = citation(turn,frame_start,frame_start+len(frame))
    start = boundary
    while start<len(text) and text[start].isspace(): start+=1
    end = len(text.rstrip())
    if start>=end: return []
    # A body after a request is retained as material, never promoted to a
    # preference. Unknown or nested quotations remain unresolved attribution.
    external = bool(EXTERNAL.search(frame)) and not OWN_MATERIAL.search(frame)
    return [_event(turn,'external_quotation' if external else 'text_operation',
            'external_material' if external else 'unresolved','not_self_asserted' if external else 'unresolved',
            [support],citation(turn,start,end),rule='explicit-text-operation-frame-v1')]


def build_memory(turns):
    turns = list(turns)
    parsed = [re.fullmatch(r'p(\d+):m(\d+):t\d+',t.source_id) for t in turns]
    if (not turns or any(p is None for p in parsed) or len({p[1] for p in parsed})!=1
        or len({t.source_id for t in turns})!=len(turns)
        or any(t.role not in {'user','assistant'} for t in turns)
        or any(int(a[2])>=int(b[2]) for a,b in zip(parsed,parsed[1:]))):
        raise ValueError('Require one user and strictly ordered native source turns')
    events = [];links = [];active = {}
    for turn in turns:
        for event in extract_turn(turn):
            events.append(event)
            if event['kind'] not in {'direct_preference','forget_request'}: continue
            key = (event['verb'],event['object_key'])
            if event['kind']=='direct_preference': active[key] = event
            else:
                previous = active.pop(key,None)
                if previous:
                    links.append({'kind':'exact_scope_withdrawal','earlier_event':previous['event_id'],
                        'later_event':event['event_id'],'support':[*previous['support'],*event['support']],
                        'state':'withdraw_preference','owner':'self'})
    return {'version':'source-tags-v1','source_identity':parsed[0][1],
        'turn_fingerprints':{t.source_id:fingerprint([t.role,t.content,t.date]) for t in turns},
        'events':events,'links':links,'active_exact_scope_event_ids':sorted(e['event_id'] for e in active.values()),
        'inputs':'ordered history turns only','human_gold':False,'semantic_correctness_guaranteed':False}


def validate_memory(turns,memory):
    if build_memory(turns)!=memory: raise ValueError('Cue memory or source provenance changed')


def covered(cite,spans):
    cursor = cite['start']
    for span in sorted((s for s in spans if s['source_id']==cite['source_id']),key=lambda s:s['start']):
        if len(span['text'])!=span['end']-span['start']: raise ValueError('Invalid source span length')
        a,b = max(span['start'],cite['start']),min(span['end'],cite['end'])
        if a<b and span['text'][a-span['start']:b-span['start']]!=cite['text'][a-cite['start']:b-cite['start']]:
            raise ValueError('Cue citation differs from selected source')
        if span['start']<=cursor<span['end']:
            cursor = max(cursor,span['end'])
        if cursor>=cite['end']: return True
    return False


def visible_cues(spans,memory):
    values = []
    for event in memory['events']:
        if not all(covered(c,spans) for c in event['support']): continue
        scope = event['scope']
        if event['kind'] in {'direct_preference','forget_request','other_report'}:
            if not covered(scope,spans): continue
            scopes = [scope]
        else:
            scopes = []
            for s in spans:
                if s['source_id']!=scope['source_id']: continue
                start,end = max(s['start'],scope['start']),min(s['end'],scope['end'])
                if start<end:
                    text = s['text'][start-s['start']:end-s['start']]
                    if text!=scope['text'][start-scope['start']:end-scope['start']]: raise ValueError('Material source changed')
                    scopes.append({'source_id':s['source_id'],'start':start,'end':end,'text':text})
            if not scopes: continue
        values.append({'event_id':event['event_id'],'kind':event['kind'],'owner':event['owner'],
            'state':event['state'],'support':event['support'],'scopes':scopes})
    visible = {v['event_id'] for v in values}
    for link in memory['links']:
        if link['earlier_event'] in visible and link['later_event'] in visible and all(covered(c,spans) for c in link['support']):
            values.append({'event_id':link['earlier_event']+'>'+link['later_event'],'kind':link['kind'],
                'owner':link['owner'],'state':link['state'],'support':link['support'],'scopes':[]})
    return values


def ref(cite):
    return f"{cite['source_id']}@{cite['start']}:{cite['end']}"


def render(groups,memory,arm):
    from .four_reader_probe import serialize
    text = serialize(groups,'compact')
    if arm=='plain': return text
    if arm not in {'neutral','typed'}: raise ValueError('Unknown cue representation')
    spans = [s for g in groups for s in g.metadata['spans']]
    lines = []
    for cue in visible_cues(spans,memory):
        kind,owner,state = (cue['kind'],cue['owner'],cue['state']) if arm=='typed' else ('source_record','unresolved','uninterpreted')
        lines.append(f"[cue kind={kind}; owner={owner}; state={state}; scope={','.join(ref(c) for c in cue['scopes']) or '-'}; "
                     f"support={','.join(ref(c) for c in cue['support'])}]")
    return LEGEND+'\n'.join(lines)+ ('\n' if lines else '')+text
