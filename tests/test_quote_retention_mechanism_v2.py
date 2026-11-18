import unittest
from reader_memory import quote_retention_mechanism_core_v1 as n


def fixture():
    text='Anchor entity. Distractor. Target fact.'
    parts=['Anchor entity.','Distractor.','Target fact.']
    spans=[]
    for part in parts:
        start=text.index(part)
        spans.append(dict(start=start,end=start+len(part),text=part))
    source=dict(question_id='test',question='What is the target?',paragraphs=[
        dict(source_id='0',title='Title',timestamp='undated',text=text,spans=spans)])
    scores={'0:span:0':10.,'0:span:1':9.,'0:span:2':1.}
    proposal=dict(roles=[{}],candidates=[dict(source='0',start=spans[-1]['start'],end=spans[-1]['end'],text='Target fact.',roles=[0])])
    return source,scores,proposal


class QuotationControlsTests(unittest.TestCase):
    def test_retention_uses_source_containment_and_empty_is_undefined(self):
        _,_,proposal=fixture()
        self.assertTrue(n.retention(proposal,{'0':[[0,39]]})['all_quotes_retained'])
        self.assertFalse(n.retention(proposal,{'0':[[27,38]]})['all_quotes_retained'])
        self.assertIsNone(n.retention({'candidates':[]},{})['quote_retention'])

    def test_candidate_dedup_and_no_type_specific_tie_break(self):
        s,scores,p=fixture()
        pool=n.candidate_pool(s,scores,p,{'selected_spans':['0:span:2','0:span:1']})
        self.assertEqual(len(pool),2)
        self.assertEqual(pool[0]['score'],9.)
        self.assertEqual(pool[1]['kinds'],['fallback_sentence','quote'])
        p['candidates'][0]['text']='invented'
        with self.assertRaises(AssertionError):n.candidate_pool(s,scores,p,{'selected_spans':[]})

    def test_quote_score_is_fixed_max_of_touched_sentences(self):
        s,scores,p=fixture()
        p['candidates'][0].update(start=15,end=39,text=s['paragraphs'][0]['text'][15:39])
        pool=n.candidate_pool(s,scores,p,{'selected_spans':[]})
        self.assertEqual(pool[0]['score'],9.)

    def test_controls_reproduce_production_fill_and_preserve_anchor(self):
        from reader_memory.anchored_quote_v1 import quote_plan
        s,scores,p=fixture()
        class Count:
            exact=True
            count=staticmethod(len)
        # The complete paragraph is an intentionally tight, reproducible full cap.
        full=n.render(s,scores,{'0':[[0,39]]},len)
        baseline=dict(full,selected_spans=['0:span:0','0:span:1','0:span:2'])
        fallback=dict(baseline)
        a,trace=quote_plan(s,scores,baseline,fallback,p,Count(),'quote_anchor')
        b,_=quote_plan(s,scores,baseline,fallback,p,Count(),'quote_cover')
        controls,audit=n.make_controls(s,scores,p,fallback,a,b,trace,full['evidence_tokens'],len)
        self.assertTrue(audit['admitted'])
        for value in controls.values():
            self.assertTrue(n.covered(n.span_for(s,'0:span:0'),value['selected_ranges']))
        self.assertEqual(controls['B_protect_fill']['evidence_text'],b['evidence_text'])
        self.assertLessEqual(controls['C_reselect_tight']['evidence_tokens'],a['evidence_tokens'])

    def test_fallback_is_identical_for_all_arms(self):
        s,scores,p=fixture()
        fallback=dict(evidence_text='unchanged fallback',evidence_tokens=18,selected_spans=[])
        controls,audit=n.make_controls(s,scores,p,fallback,fallback,fallback,
            {'fallback':'missing_role_quote'},512,len)
        self.assertFalse(audit['admitted'])
        self.assertEqual(set(controls),set(n.ARMS))
        self.assertTrue(all(v==fallback for v in controls.values()))

    def test_greedy_skips_overbudget_without_truncation(self):
        s,scores,p=fixture()
        seed={'0':[[0,14]]}
        start=n.render(s,scores,seed,len)
        pool=n.candidate_pool(s,scores,p,{'selected_spans':['0:span:1']})
        plan=n.pack(s,scores,pool,seed,start['evidence_tokens'],len)
        self.assertEqual(plan['evidence_text'],start['evidence_text'])
        self.assertEqual(plan['accepted_pool_indices'],[])


if __name__=='__main__':unittest.main()
