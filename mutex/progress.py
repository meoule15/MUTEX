"""Progress events from an explicit predicate adapter, separate from policy inputs.

LIBERO predicates are benchmark privileged information, not a learned predictor.
"""
class GoalProgressMonitor:
    def __init__(self,num_predicates,stall_steps=240,regression_patience=5,cooldown_steps=60):
        if min(num_predicates,stall_steps,regression_patience,cooldown_steps)<1:
            raise ValueError('Progress counts must be positive')
        self.num_predicates=num_predicates;self.stall_steps=stall_steps
        self.regression_patience=regression_patience;self.cooldown_steps=cooldown_steps
        self.previous=[False]*num_predicates;self.seen=[False]*num_predicates
        self.false_streak=[0]*num_predicates;self.last_step=-1
        self.last_progress=0;self.last_reassessment=-cooldown_steps

    def update(self,step,predicates):
        if step<0 or step<=self.last_step or len(predicates)!=self.num_predicates:
            raise ValueError('Expected increasing steps and complete predicate vector')
        if any(type(x) is not bool for x in predicates):raise ValueError('Expected boolean predicates')
        gained=[i for i,(old,new) in enumerate(zip(self.previous,predicates)) if new and not old]
        if gained:self.last_progress=step
        for i,current in enumerate(predicates):
            self.seen[i]|=current
            self.false_streak[i]=0 if current or not self.seen[i] else self.false_streak[i]+1
        lost=[i for i,n in enumerate(self.false_streak) if n>=self.regression_patience]
        complete=all(predicates);stagnant=step-self.last_progress>=self.stall_steps
        reason='regression' if lost else 'stagnation' if stagnant else None
        reassess=not complete and reason is not None and step-self.last_reassessment>=self.cooldown_steps
        if reassess:self.last_reassessment=step
        self.previous=list(predicates);self.last_step=step
        return dict(step=step,current_predicates=list(predicates),joint_complete=complete,
                    gained=gained,lost=lost,stagnant=stagnant,reassess=reassess,
                    reason=reason if reassess else None,last_progress_step=self.last_progress)
