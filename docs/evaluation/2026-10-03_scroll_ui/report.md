# Scroll UI verification — 2026-10-03

Actual browser wheel scrolling reproduced the reported return to the bottom during an existing model retry. The main scrolling element was `stAppScrollToBottomContainer`; after moving up it returned to 9710.5 with height 748 and scrollHeight 10458, i.e. the bottom.

Installed Streamlit 1.63.0 activates its whole-page bottom-following hook when a chat input is in the bottom tree. The application used a root-level chat input. The frontend hook animates to the bottom and checks scroll state at intervals. This is a UI behavior independent of SQL/data loading.

The composer is now an inline, keyed native container in the main tree. A native heading anchor and link let the user move to recent results/input explicitly. The input is at the end of the conversation, rather than pinned over history. No injected script/CSS, dependency upgrade, or dataset reload was required.

The actual page now uses `stMain`, with no whole-page auto-scroll container. Browser observations: input link 10006.5 → up wheel 9258.5 → later observation 9258.5 → second up wheel 8510.5. See [screen](fixed_scroll.png) and [measurements](validation.json).

Targeted AppTest/runtime/support regression: 10 PASS + 2 subtests, 11.21s. The actual page input remains in the main tree after a stored-chart follow-up submission; no inference is invoked by that test. Compile/diff checks and health succeeded. This request did not re-run the full agent suite, launch a fresh model/SQL request, or test all browsers. The legend/color analysis failure currently in the conversation remains a separate agent issue; it is not fixed by changing scroll behavior.
