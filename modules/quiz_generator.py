"""
modules/quiz_generator.py
─────────────────────────
Uses RAG to retrieve relevant context from the vector store,
then prompts Gemini to generate MCQ quizzes from that context.
"""

import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from modules.llm_setup import get_llm

QUIZ_PROMPT = PromptTemplate(
    input_variables=["context", "topic", "num_questions"],
    template="""You are an expert educator. Using ONLY the context below, create
{num_questions} multiple-choice questions about "{topic}".

CONTEXT:
{context}

FORMAT (follow exactly):
Q1. <question text>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <letter>
Explanation: <one sentence>

Q2. ...

Rules:
- Base every question strictly on the provided context.
- Make distractors plausible but clearly wrong.
- Do NOT add any text before Q1 or after the last explanation.
"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def generate_quiz(topic: str, retriever, num_questions: int = 5) -> str:
    llm = get_llm(temperature=0.4, max_new_tokens=1024)

    chain = (
        {
            "context": retriever | _format_docs,
            "topic": RunnablePassthrough(),
            "num_questions": lambda _: num_questions,
        }
        | QUIZ_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(topic)


def parse_quiz(raw_quiz: str) -> list[dict]:
    questions = []
    blocks = re.split(r"\nQ\d+\.", "\n" + raw_quiz)
    blocks = [b.strip() for b in blocks if b.strip()]

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 6:
            continue

        question    = lines[0]
        options     = {}
        answer      = ""
        explanation = ""

        for line in lines[1:]:
            if line.startswith(("A)", "B)", "C)", "D)")):
                options[line[0]] = line[3:].strip()
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()

        if question and options and answer:
            questions.append({
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": explanation,
            })

    return questions
