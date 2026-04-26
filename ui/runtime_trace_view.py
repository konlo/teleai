import streamlit as st

from utils.runtime_trace import read_latest_trace


def render_runtime_trace_debug_panel() -> None:
    trace = read_latest_trace()
    st.markdown("#### Runtime Trace")
    with st.expander("최근 Runtime Trace", expanded=False):
        if not trace:
            st.info("아직 기록된 runtime trace가 없습니다.")
            return

        summary = trace.get("summary", {})
        st.caption(f"trace_id: `{trace.get('trace_id', '')}`")
        st.write(f"- 입력: `{summary.get('user_message', '')}`")
        st.write(f"- agent_mode: `{summary.get('agent_mode', '')}`")
        st.write(f"- SQL status: `{summary.get('sql_execution_status', '')}`")
        st.write(f"- chain_triggered: `{summary.get('chain_triggered', '')}`")
        st.write(f"- matched_keywords: `{summary.get('matched_keywords', [])}`")

        compact_events = []
        for event in trace.get("events", []):
            compact_events.append(
                {
                    "seq": event.get("event_seq"),
                    "type": event.get("event_type"),
                    "agent_mode": event.get("agent_mode"),
                    "matched_keywords": event.get("matched_keywords"),
                    "keyword_forced_sql": event.get("keyword_forced_sql"),
                    "sql": event.get("sql"),
                    "decision": event.get("decision"),
                    "chain_triggered": event.get("chain_triggered"),
                    "error": event.get("error"),
                }
            )
        st.json(
            {
                "summary": summary,
                "events": compact_events,
            }
        )


__all__ = ["render_runtime_trace_debug_panel"]
