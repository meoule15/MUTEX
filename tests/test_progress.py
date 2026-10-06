import unittest
import torch
from mutex.progress import GoalProgressMonitor
from mutex.pmi import rank_with_abstention,SpecificationSelector


class TestProgress(unittest.TestCase):
    def test_joint_completion_not_ever_complete(self):
        monitor=GoalProgressMonitor(2,regression_patience=2)
        self.assertFalse(monitor.update(0,[True,False])['joint_complete'])
        self.assertFalse(monitor.update(1,[False,True])['joint_complete'])
        event=monitor.update(2,[False,True])
        self.assertEqual(event['lost'],[0]);self.assertEqual(event['reason'],'regression')
        self.assertTrue(monitor.update(3,[True,True])['joint_complete'])

    def test_regression_debounce_recovery_and_cooldown(self):
        m=GoalProgressMonitor(1,regression_patience=2,cooldown_steps=4)
        m.update(0,[True]);self.assertFalse(m.update(1,[False])['reassess'])
        self.assertTrue(m.update(2,[False])['reassess'])
        self.assertFalse(m.update(3,[False])['reassess'])
        self.assertTrue(m.update(4,[True])['joint_complete'])
        self.assertFalse(m.update(5,[False])['reassess'])
        self.assertTrue(m.update(6,[False])['reassess'])

    def test_stagnation_reset_and_never_seen_predicate(self):
        m=GoalProgressMonitor(2,stall_steps=3)
        self.assertEqual(m.update(0,[False,False])['lost'],[])
        self.assertFalse(m.update(2,[True,False])['reassess'])
        self.assertFalse(m.update(4,[True,False])['reassess'])
        self.assertEqual(m.update(5,[True,False])['reason'],'stagnation')
        with self.assertRaises(ValueError):m.update(5,[True,False])
        with self.assertRaises(ValueError):m.update(6,[True])

    def test_abstention_retains_configured_fallback_and_resolved_switch(self):
        scores={'gl':dict(expected_lift=torch.tensor([2.015]),expected_lift_se=torch.tensor([.086])),
                'vid':dict(expected_lift=torch.tensor([2.]),expected_lift_se=torch.tensor([.074]))}
        self.assertIsNone(rank_with_abstention(scores)[0]['choice'])
        selector=SpecificationSelector(['gl','vid'],1,margin=0,initial_key='vid')
        self.assertEqual(selector.update(1,scores),[False]);self.assertEqual(selector.current,['vid'])
        self.assertEqual(selector.last_decisions[0]['status'],'abstained')
        scores['gl']['expected_lift'][0]=4.
        self.assertEqual(selector.update(11,scores),[True]);self.assertEqual(selector.current,['gl'])

    def test_invalid_and_single_candidate_rankings(self):
        score=dict(expected_lift=torch.tensor([1.]),expected_lift_se=torch.tensor([0.]))
        self.assertEqual(rank_with_abstention({'gl':score})[0]['choice'],'gl')
        score['expected_lift_se'][0]=-1
        with self.assertRaises(ValueError):rank_with_abstention({'gl':score})
        with self.assertRaises(ValueError):GoalProgressMonitor(0)


if __name__=='__main__':unittest.main()
