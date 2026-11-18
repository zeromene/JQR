import unittest
from reader_memory import compact_reranker_control_core_v1 as n
from reader_memory import quote_retention_mechanism_core_v1 as q


def fixture():
    parts=['Anchor.', 'Old candidate.', 'New useful sentence.']
    text=' '.join(parts)
    spans=[dict(start=text.index(x),end=text.index(x)+len(x),text=x) for x in parts]
    source=dict(question_id='test',native_id='test',mode='multi',question='Which fact?',paragraphs=[
        dict(source_id='0',title='T',timestamp='undated',text=text,spans=spans)])
    return source,{'0:span:0':10.,'0:span:1':1.,'0:span:2':9.}


class Counter:
    exact=True
    count=staticmethod(len)


class FullPoolControlTests(unittest.TestCase):
    def test_can_select_sentence_absent_from_old_pool(self):
        source,scores=fixture()
        anchor=q.span_for(source,'0:span:0')
        wanted=q.render(source,scores,{'0':[[0,7],[23,43]]},len)
        cap=wanted['evidence_tokens']
        plan,_=n.full_pool_plan(source,scores,Counter(),cap,anchor)
        self.assertIn('0:span:2',plan['selected_spans'])
        oldpool=q.candidate_pool(source,scores,{'candidates':[]},{'selected_spans':['0:span:1']})
        old=q.pack(source,scores,oldpool,q.extend({},anchor),cap,len)
        self.assertNotIn('New useful sentence.',old['evidence_text'])

    def test_tight_cap_never_slices_long_sentence(self):
        source,scores=fixture()
        anchor=q.span_for(source,'0:span:0')
        only=q.render(source,scores,q.extend({},anchor),len)
        plan,_=n.full_pool_plan(source,scores,Counter(),only['evidence_tokens'],anchor)
        self.assertEqual(plan['selected_spans'],['0:span:0'])
        self.assertEqual(plan['evidence_text'],only['evidence_text'])

    def test_trace_preserves_production_C_and_geometry(self):
        source,scores=fixture()
        anchor=q.span_for(source,'0:span:0')
        quote=q.span_for(source,'0:span:2')
        quote['text']='New useful sentence.'
        proposal={'candidates':[quote]}
        fallback={'selected_spans':['0:span:0','0:span:1','0:span:2']}
        pool=q.candidate_pool(source,scores,proposal,fallback)
        cap=q.render(source,scores,q.extend({},anchor),len)['evidence_tokens']
        plan,steps=n.traced_pack(source,scores,pool,anchor,cap,len)
        self.assertEqual(plan,q.pack(source,scores,pool,q.extend({},anchor),cap,len))
        self.assertEqual(n.geometry(source,plan['selected_ranges'])['partly_included_sentences'],0)
        self.assertTrue(any(s['decision']=='over_cap' for s in steps))

    def test_long_sentence_crowding_witness_preserves_kept_quotes(self):
        parts=['Anchor.', 'Useful short quote with very long surrounding filler text.', 'Other needed fact.']
        text=' '.join(parts)
        spans=[dict(start=text.index(x),end=text.index(x)+len(x),text=x) for x in parts]
        source=dict(question_id='w',native_id='w',mode='multi',question='Both?',paragraphs=[
            dict(source_id='0',title='T',timestamp='undated',text=text,spans=spans)])
        scores={'0:span:0':10.,'0:span:1':9.,'0:span:2':8.}
        anchor=q.span_for(source,'0:span:0')
        first=dict(source='0',start=spans[1]['start']+7,end=spans[1]['start']+18,text=text[spans[1]['start']+7:spans[1]['start']+18])
        second=q.span_for(source,'0:span:2');second['text']=parts[2]
        proposal={'candidates':[first,second]};fallback={'selected_spans':list(scores)}
        cap=q.render(source,scores,{'0':[[0,spans[1]['end']]]},len)['evidence_tokens']
        a=q.render(source,scores,q.extend(q.extend(q.extend({},anchor),first),second),len)
        self.assertLessEqual(a['evidence_tokens'],cap)
        c,_=n.traced_pack(source,scores,q.candidate_pool(source,scores,proposal,fallback),anchor,cap,len)
        audit=n.construction_audit(source,scores,proposal,fallback,a,c,anchor,cap,len)
        self.assertEqual(len(audit['lost_quotes']),1)
        self.assertTrue(audit['single_sentence_replacement_witnesses'])


if __name__=='__main__': unittest.main()
