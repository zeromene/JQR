import unittest
from copy import deepcopy
from reader_memory import budget_sweep_core_v1 as n


def fixture():
    texts=['Alice was born in Lima.', 'Lima is in Peru.', 'Alice studied in Paris.', 'Paris is in France.']
    source=dict(question_id='fixture',native_id='fixture',mode='multi',question='Which country?',paragraphs=[
        dict(source_id=str(i),title='T',timestamp='undated',text=t,spans=[dict(start=0,end=len(t),text=t)]) for i,t in enumerate(texts)])
    scores={'0:span:0':4.,'2:span:0':3.,'1:span:0':2.,'3:span:0':1.}
    proposal={'roles':[{},{}], 'candidates':[dict(source=str(i),start=0,end=len(texts[i]),text=texts[i],roles=[i]) for i in [0,1]]}
    return source,scores,proposal


class Counter:
    exact=True
    count=staticmethod(len)


class SweepContracts(unittest.TestCase):
    def test_registered_old_budgets_reproduce_production(self):
        s,scores,p=fixture()
        for b in [256,512]:
            old=n.short.short_plans(s,scores,['0','1'],Counter(),b,b)
            new=n.baseline_and_fallback(s,scores,['0','1'],Counter(),b)
            self.assertEqual(old,new)

    def test_reranker_is_independent_of_quotation_proposal(self):
        s,scores,p=fixture();missing=deepcopy(p);missing['candidates']=[]
        for b in n.BUDGETS:
            a,trace=n.construct(s,scores,['0','1'],p,Counter(),b)
            z,zt=n.construct(s,scores,['0','1'],missing,Counter(),b)
            self.assertEqual(a['strong_rerank']['evidence_text'],z['strong_rerank']['evidence_text'])
            self.assertEqual(zt['fallback'],'missing_role_quote')
            for plan in a.values():self.assertLessEqual(len(plan['evidence_text']),b)
        self.assertLess(len(a['quote_anchor']['evidence_text']),len(a['strong_rerank']['evidence_text']))

    def test_overlong_collection_falls_back_without_truncation(self):
        s,scores,p=fixture()
        for i in [0,1]:
            text=s['paragraphs'][i]['text']+' x'*80
            s['paragraphs'][i].update(text=text,spans=[dict(start=0,end=len(text),text=text)])
            p['candidates'][i].update(end=len(text),text=text)
        small,trace=n.construct(s,scores,['0','1'],p,Counter(),128)
        self.assertFalse(trace['admitted'])
        self.assertTrue(trace['fallback'])
        for pid,ranges in small['quote_anchor']['selected_ranges'].items():
            self.assertEqual(ranges,[[0,len(s['paragraphs'][int(pid)]['text'])]])
        large,trace=n.construct(s,scores,['0','1'],p,Counter(),512)
        self.assertTrue(trace['admitted'])
        self.assertNotEqual(small['quote_anchor']['evidence_text'],large['quote_anchor']['evidence_text'])

    def test_no_sentence_fits_remains_a_recordable_failure(self):
        s,scores,p=fixture()
        for v in s['paragraphs']:
            text='X'*600;v.update(text=text,spans=[dict(start=0,end=len(text),text=text)])
        plans,trace=n.construct(s,scores,['0','1'],p,Counter(),128)
        self.assertFalse(trace['admitted'])
        for v in plans.values():
            self.assertTrue(v['selection_error']);self.assertEqual(v['evidence_text'],'')

    def test_cluster_bootstrap_keeps_budget_pairing_and_question_weights(self):
        import numpy as np
        estimates,boot=n.bootstrap_matrix([[1.,1.],[0.,0.],[0.,0.]],['a','b','b'],1000)
        self.assertAlmostEqual(estimates[0],1/3)
        np.testing.assert_array_equal(boot[:,0],boot[:,1])
        result,critical=n.interval_family([0.],np.zeros((1000,1)))
        self.assertEqual(result[0]['simultaneous_ci95_points'],[0.,0.])
        self.assertEqual(critical,0.)


if __name__=='__main__':unittest.main()
