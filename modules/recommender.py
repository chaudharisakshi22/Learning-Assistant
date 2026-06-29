"""
modules/recommender.py
───────────────────────
Uses weak-topic data + RAG context to suggest personalised study
resources and a learning plan via Google Gemini.
"""

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from modules.llm_setup import get_llm
from modules.performance_tracker import get_weak_topics

RECOMMEND_PROMPT = PromptTemplate(
    input_variables=["context", "weak_topics", "all_topics"],
    template="""You are a personal learning coach. A student has been studying the
following topics: {all_topics}

Their weaker areas (below 60% average) are: {weak_topics}

Here is relevant content from their study materials:
{context}

Please provide:
1. A brief diagnosis (2-3 sentences) of where they should focus.
2. A prioritised 3-step study plan using the material above.
3. Two specific concepts from the context they should revisit.
4. One motivational tip.

Keep your response concise, practical, and encouraging.
"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def get_recommendations(retriever, all_topics: List[str]) -> str:
    weak_topics = get_weak_topics(threshold=60.0)

    weak_topics_str = ", ".join(weak_topics) if weak_topics else "None — great work! All topics above 60%."
    all_topics_str  = ", ".join(all_topics)  if all_topics  else "No topics yet"

    llm   = get_llm(temperature=0.7, max_new_tokens=600)
    query = " ".join(weak_topics) if weak_topics else " ".join(all_topics)

    chain = (
        {
            "context":     retriever | _format_docs,
            "weak_topics": lambda _: weak_topics_str,
            "all_topics":  lambda _: all_topics_str,
        }
        | RECOMMEND_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)
