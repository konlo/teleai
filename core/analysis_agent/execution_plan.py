"""Persist immutable task scopes and dependencies, never model-authored success."""
from copy import deepcopy
from hashlib import sha256
import json
from datetime import datetime, timezone

CONTRACT_ALIASES={'row_count':'calculation','table_list':'remote_query','chart_adjust':'chart'}


class ExecutionPlans:
    def __init__(self, db):
        self.db = db
        with db.lock, db.conn:
            db.conn.execute('CREATE TABLE IF NOT EXISTS analysis_plans (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.conn.execute('CREATE TABLE IF NOT EXISTS analysis_plan_events '
                '(sequence INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, '
                'event_key TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(request_id,event_key))')

    def sync(self, current, calls=None):
        goal = current.get('goal')
        if not goal or current.get('goal_pending') or not current.get('request_id'):
            return None
        from core.analysis_agent.completion import CONTRACTS
        contracts={c.name:c for c in CONTRACTS}
        if current.get('intent_plan'):
            tasks,plan=self._independent_plan(current,contracts)
        else:
            tasks=[]
            scope={k:deepcopy(current.get(k)) for k in ('required_sources','required_columns','scope','current_result_only','fresh_source_required')}
            for index, task in enumerate(goal.get('tasks', [])):
                contract=CONTRACT_ALIASES.get(task['capability'],task['capability'])
                satisfied=contract in contracts and contracts[contract].satisfied(current)
                key=sha256(json.dumps([current['request_id'],index,task,scope],sort_keys=True,default=str).encode()).hexdigest()[:20]
                tasks.append({'id':key, 'capability':task['capability'], 'options':task['options'],
                              'scope':scope, 'depends_on':[],
                              'postconditions':deepcopy(task['options'].get('result_contract')),
                              'verification_receipt':deepcopy(current.get(contracts[contract].evidence_key)) if satisfied else None,
                              'status':'verified' if satisfied else 'pending',
                              'completion_contract':contract,
                              'evidence_ids':current.get('evidence_ids',[]) if satisfied else [],
                              'artifact_ids':current.get('artifact_ids',[]) if satisfied else []})
            plan={'id':current['request_id'],'tasks':tasks,'goal':goal,
                  'status':current.get('status','working'),'scope_mode':'shared_goal_scope',
                  'decision_evidence':deepcopy(current.get('goal_decision_evidence'))}
        if plan['decision_evidence']:
            from core.analysis_agent.goal_grounding import bound_to_current
            if not bound_to_current(current):plan['decision_evidence']['status']='stale'
        with self.db.lock,self.db.conn:
            previous=self.db.conn.execute('SELECT payload FROM analysis_plans WHERE id=?',(plan['id'],)).fetchone()
            old_plan=json.loads(previous[0]) if previous else {}
            if previous:
                old={t['id']:t for t in old_plan['tasks']}
                for task in tasks:task['depends_on']=old.get(task['id'],{}).get('depends_on',[])
            plan['revision']=old_plan.get('revision',0)
            if self._content(plan)!=self._content(old_plan):plan['revision']+=1
            self._record_progress(plan,old_plan,current,calls or {})
            self.db.conn.execute('INSERT INTO analysis_plans VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                                 (plan['id'],json.dumps(plan,ensure_ascii=False,default=str)))
        return plan

    @staticmethod
    def _independent_plan(current,contracts):
        saved=current['intent_plan'];completed={s['id']:s for s in saved['completed']}
        tasks=[]
        for step in saved['spec']['steps']:
            goal=step['goal'];task=goal['tasks'][0];record=completed.get(step['id'])
            contract=CONTRACT_ALIASES.get(task['capability'],task['capability'])
            scope={'required_sources':deepcopy(goal['sources']),
                'required_columns':deepcopy(goal['columns']),
                'scope':{k:deepcopy(goal[k]) for k in ('conditions','any_conditions','measure_conditions','ratio')},
                'current_result_only':goal['current_result_only'],
                'fresh_source_required':goal['fresh_source_required']}
            if record:scope['scope']=deepcopy(record['scope'])
            elif step['id']==saved['active_step_id']:
                scope={k:deepcopy(current.get(k)) for k in scope}
            tasks.append({'id':saved['request_id']+':'+step['id'],
                'capability':task['capability'],'options':deepcopy(task['options']),
                'scope':scope,'depends_on':[saved['request_id']+':'+k for k in step['depends_on']],
                'status':'verified' if record else 'pending',
                'completion_contract':contract,'postconditions':deepcopy(task['options'].get('result_contract')),
                'verification_receipt':deepcopy(record['receipts']) if record else None,
                'evidence_ids':deepcopy(record['evidence_ids']) if record else [],
                'artifact_ids':deepcopy(record['artifact_ids']) if record else []})
        return tasks,{'id':current['request_id'],'tasks':tasks,'goal':deepcopy(saved['spec']),
            'status':current.get('status','working'),'scope_mode':'per_task',
            'active_task_id':current['request_id']+':'+saved['active_step_id'],
            'decision_evidence':None,'intent_plan_sha256':saved['sha256']}

    @staticmethod
    def _content(plan):
        return json.dumps({k:v for k,v in plan.items() if k!='revision'},sort_keys=True,default=str)

    def _event(self,request_id,key,kind,**facts):
        event={'kind':kind,'at':datetime.now(timezone.utc).isoformat(),**facts}
        self.db.conn.execute('INSERT OR IGNORE INTO analysis_plan_events '
            '(request_id,event_key,payload) VALUES (?,?,?)',
            (request_id,key,json.dumps(event,ensure_ascii=False,default=str)))

    def _record_progress(self,plan,previous,current,calls):
        request_id=plan['id'];revision=plan['revision']
        if not previous:self._event(request_id,'created','plan_created',revision=revision)
        old={t['id']:t for t in previous.get('tasks',[])}
        for decision in current.get('action_cycle_history',[]):
            key=sha256(str(decision['decision_id']).encode()).hexdigest()
            self._event(request_id,'model-action:'+key,'model_action_selected',
                observation_id=decision['observation_id'],choices=decision['choices'],
                outstanding=decision['outstanding'],observed_call_ids=decision['observed_call_ids'],
                revision=revision)
        for task in plan['tasks']:
            if task.get('status')!=old.get(task['id'],{}).get('status'):
                self._event(request_id,f'task:{revision}:{task["id"]}',
                    'task_verified' if task['status']=='verified' else 'task_pending',
                    task_id=task['id'],capability=task['capability'],revision=revision)
        from core.analysis_agent.tool_repair import signature_id,diagnosis
        failures=current.get('tool_failures',{})
        for call_id,call in calls.items():
            if call_id not in current.get('sent_calls',[]):continue
            if self.db.conn.execute('SELECT 1 FROM analysis_plan_events WHERE request_id=? '
                'AND event_key=?',(request_id,'call:'+call_id)).fetchone():continue
            args_hash=sha256(json.dumps(call.get('args',{}),sort_keys=True,default=str).encode()).hexdigest()
            recent=self.db.conn.execute('SELECT payload FROM analysis_plan_events WHERE request_id=? '
                'AND event_key LIKE ? ORDER BY sequence DESC LIMIT 128',(request_id,'call:%')).fetchall()
            prior=next((item for row in recent if (item:=json.loads(row[0])).get('tool')==call.get('name')), {})
            failure=next((f for f in failures.values() if f.get('call_id')==prior.get('call_id')),None)
            if (failure and prior.get('tool')==call.get('name') and prior.get('arguments_sha256')!=args_hash):
                self._event(request_id,'change:'+call_id,'tool_plan_changed',
                    call_id=call_id,tool=call.get('name'),failure_call_id=prior['call_id'],
                    error_code=failure.get('error_code'),arguments_sha256=args_hash,
                    goal_preserved=plan['goal']==previous.get('goal'),revision=revision)
            self._event(request_id,'call:'+call_id,'tool_dispatched',call_id=call_id,
                tool=call.get('name'),arguments_sha256=args_hash,revision=revision)
        for signature,failure in failures.items():
            count=current.get('failed_signatures',{}).get(signature,1)
            item=diagnosis(failure,[])
            self._event(request_id,'failure:'+signature_id(signature)+':'+str(count),
                'tool_failure_observed',tool=failure.get('tool'),call_id=failure.get('call_id'),
                error_code=failure.get('error_code'),category=item['category'],
                count=count,revision=revision)
        if plan['status']!=previous.get('status'):
            self._event(request_id,'status:'+str(revision),'plan_status_changed',
                status=plan['status'],stop_reason=current.get('stop_reason'),revision=revision)

    def history(self,request_id,limit=32):
        if type(limit) is not int or not 1<=limit<=128:raise ValueError('History limit must be 1..128')
        with self.db.lock:
            rows=self.db.conn.execute('SELECT sequence,payload FROM analysis_plan_events '
                'WHERE request_id=? ORDER BY sequence DESC LIMIT ?', (request_id,limit)).fetchall()
        return [{'sequence':row[0],**json.loads(row[1])} for row in reversed(rows)]

    def inspect(self, request_id):
        with self.db.lock:
            row=self.db.conn.execute('SELECT payload FROM analysis_plans WHERE id=?',(request_id,)).fetchone()
        if row is None:return {'status':'needs_context','error_code':'plan_not_found'}
        return {'status':'ready','plan':json.loads(row[0]),'recent_events':self.history(request_id)}

    def dependencies(self, request_id, edges):
        result=self.inspect(request_id)
        if result['status']!='ready':return result
        plan=deepcopy(result['plan']);tasks={t['id']:t for t in plan['tasks']}
        graph={k:list(t['depends_on']) for k,t in tasks.items()}
        for edge in edges:
            if set(edge)!={'task_id','depends_on'} or edge['task_id'] not in tasks:
                raise ValueError('Unknown task ID')
            if tasks[edge['task_id']]['status']=='verified':raise ValueError('Verified task cannot be replanned')
            if any(k not in tasks or k==edge['task_id'] for k in edge['depends_on']):raise ValueError('Invalid dependency')
            if (plan.get('active_task_id')==edge['task_id']
                    and any(tasks[k]['status']!='verified' for k in edge['depends_on'])):
                raise ValueError('An active task cannot acquire unverified dependencies')
            graph[edge['task_id']]=list(dict.fromkeys(edge['depends_on']))
        def visit(key, path):
            if key in path:raise ValueError('Dependency cycle')
            for parent in graph[key]:visit(parent,path|{key})
        for key in tasks:visit(key,set())
        for key in tasks:tasks[key]['depends_on']=graph[key]
        changed=self._content(plan)!=self._content(result['plan'])
        if not changed:return {'status':'ready','plan':plan}
        plan['revision']=plan.get('revision',0)+1
        with self.db.lock,self.db.conn:
            self._event(request_id,'dependencies:'+str(plan['revision']),'dependencies_revised',
                edges=deepcopy(edges),revision=plan['revision'],goal_preserved=True)
            self.db.conn.execute('UPDATE analysis_plans SET payload=? WHERE id=?',
                                 (json.dumps(plan,ensure_ascii=False),request_id))
        return {'status':'ready','plan':plan}

    def admission(self,current,tool_name):
        if current.get('intent_plan'):
            from core.analysis_agent.intent_plan import digest
            if digest(current.get('goal'))!=current.get('active_task_goal_sha256'):
                return {'error_code':'task_goal_changed','message':'Active task goal changed after admission.'}
        if current.get('goal_decision_evidence'):
            from core.analysis_agent.goal_grounding import bound_to_current
            if not bound_to_current(current):return {
                'error_code':'goal_decision_support_stale',
                'message':'The support record belongs to a different request or goal. Preserve the original intent and do not execute altered work.'}
        result=self.inspect(current.get('request_id',''))
        if result['status']!='ready':return None
        from core.analysis_agent.completion import CONTRACTS
        capabilities={c.name for c in CONTRACTS if tool_name in c.tools}
        tasks={t['id']:t for t in result['plan']['tasks']}
        for task in tasks.values():
            if result['plan'].get('active_task_id') and task['id']!=result['plan']['active_task_id']:continue
            if task['completion_contract'] in capabilities and task['status']!='verified':
                waiting=[key for key in task['depends_on'] if tasks[key]['status']!='verified']
                if waiting:return {'error_code':'task_dependency_pending','waiting_task_ids':waiting,
                                   'message':'Complete the dependency tasks without changing their scope first.'}
        return None
