"""Select behavior rules by typed obligations, never by user-text patterns."""
RULE_CAPABILITIES=({'table_list'},{'metadata'},{'metadata'},{'value_list'},{'profile'},
    {'row_preview'},{'row_count','calculation'},{'chart'},{'chart'},
    {'chart_adjust'},{'chart_adjust'},{'advanced_eda'},{'custom_analysis'},{'export'})


def for_selection(instructions,selection):
    if not selection or selection.get('mode')!='execute':return instructions
    marker='Output obligations (negation removes ONLY the forbidden action; preserve positive requests):'
    head,sep,rest=instructions.partition(marker)
    rules,tail_sep,tail=rest.partition('Chart options use axes=')
    blocks=rules.split('\n- ')[1:]
    if not sep or not tail_sep or len(blocks)!=len(RULE_CAPABILITIES):
        # A changed protocol must not silently erase a behavior rule.
        return instructions
    capabilities=set(selection.get('capabilities',[]))
    selected=['- '+block.strip() for kinds,block in zip(RULE_CAPABILITIES,blocks) if kinds & capabilities]
    return head+marker+'\n'+'\n'.join(selected)+'\n\n'+tail_sep+tail
