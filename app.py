"""Streamlit interface for the public MedQuAD router demo."""

import os

import streamlit as st

from src.router_rag import RouterRAGSystem, SAFETY_WARNING_BANNER


st.set_page_config(page_title="MedQuAD Router Demo", page_icon="🩺", layout="wide")


@st.cache_resource
def load_system(threshold: float, top_k: int) -> RouterRAGSystem:
    kb_path = os.path.join(os.path.dirname(__file__), "data", "medquad_sample_kb.json")
    return RouterRAGSystem(
        kb_path=kb_path,
        router_threshold=threshold,
        top_k_retrieval=top_k,
    )


def show_evidence(records):
    with st.expander("Retrieved source records", expanded=True):
        for record in records:
            st.markdown(
                f"**Rank {record['rank']}** · similarity `{record['similarity']:.4f}`  \n"
                f"**Source label:** {record.get('source', 'Unknown')}  \n"
                f"**Question:** {record['question']}  \n"
                f"**Source answer:** {record['answer']}"
            )
            st.divider()


def main() -> None:
    st.title("🩺 MedQuAD Router Demo")
    st.caption("Candidate confirmation and source-excerpt fallback over a six-record sample")
    st.warning(SAFETY_WARNING_BANNER)

    with st.sidebar:
        st.header("Router settings")
        threshold = st.slider("Similarity threshold", 0.30, 0.95, 0.68, 0.02)
        top_k = st.selectbox("Evidence records", options=[3, 5], index=0)
        st.info(
            "The default public demo does not invoke Qwen, QLoRA, or any other LLM. "
            "Low-confidence queries return the top retrieved source excerpt."
        )

    system = load_system(threshold, top_k)
    preset = st.selectbox(
        "Example",
        [
            "",
            "What are the treatments for Paroxysmal Hemicrania ?",
            "What is asthma ?",
            "I feel tired and thirsty all the time. Could this be diabetes?",
        ],
    )
    query = st.text_input("Question", value=preset)
    if not query.strip():
        return

    route = system.route_query(query)
    col1, col2, col3 = st.columns(3)
    col1.metric("Top-1 similarity", f"{route['top1_similarity']:.4f}")
    col2.metric("Threshold", f"{route['router_threshold']:.2f}")
    col3.metric("Backend", route["retrieval_backend"])

    if route["recommended_branch"] == "BRANCH_A_DIRECT_CONFIRMATION":
        st.success("High-similarity candidates found. Confirm the intended source question.")
        candidates = route["top3_candidates"]
        labels = [
            f"{item['question']} (similarity {item['similarity']:.3f})"
            for item in candidates
        ]
        labels.append("None match — use retrieved-evidence fallback")
        choice = st.radio("Candidate", labels)
        if st.button("Show answer", type="primary"):
            if choice == labels[-1]:
                response = system.answer_via_evidence_fallback(query, route["top_k_evidence"])
            else:
                response = system.answer_via_direct_database(candidates[labels.index(choice)])
            st.subheader(response["pathway"])
            st.write(response["answer"])
            st.warning(response["warning"])
    else:
        st.info("Low-similarity query: showing the top retrieved source excerpt without generation.")
        response = system.answer_via_evidence_fallback(query, route["top_k_evidence"])
        st.subheader(response["pathway"])
        st.write(response["answer"])

    show_evidence(route["top_k_evidence"])


if __name__ == "__main__":
    main()
