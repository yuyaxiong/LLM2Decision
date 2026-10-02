from __future__ import annotations

import re

import pytest

from llm2decision.core.schema import Question
from llm2decision.llm.service import question_items
from llm2decision.prompts import UnknownPromptVersionError, available_versions, build_messages, load_prompt_set

SYSTEM_PROMPT = load_prompt_set("v1").system

STATE = "I was charged twice and want a refund."

OUTPUT_TAIL = "答案编号（只输出一个字符，不要有任何其他内容）："


def test_choice_prompt_lists_handles_and_instructions() -> None:
    question = Question(
        type="choice",
        instructions="Choose the customer intent.",
        criteria={"billing": "A payment or refund issue", "technical": "A malfunction"},
    )
    messages = build_messages(STATE, question, question_items(question))
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    content = messages[1]["content"]
    assert STATE in content
    assert "Choose the customer intent." in content
    assert "1. billing：A payment or refund issue" in content
    assert "2. technical：A malfunction" in content
    assert "只输出所选候选对应的编号" in content


def test_noul_prompt_asks_for_yes_no_label() -> None:
    question = Question(type="noul", instructions="The customer explicitly requests a refund.")
    messages = build_messages(STATE, question, question_items(question))
    content = messages[1]["content"]
    assert "The customer explicitly requests a refund." in content
    assert "只输出 1" in content and "只输出 2" in content


def test_score_prompt_uses_scale_labels_as_handles() -> None:
    question = Question(
        type="score",
        instructions="Rate the tone.",
        scale=["0", "1", "2", "3"],
    )
    items = question_items(question)
    assert [handle for handle, _, _ in items] == ["0", "1", "2", "3"]
    content = build_messages(STATE, question, items)[1]["content"]
    assert "0. 0" in content
    assert "3. 3" in content


def test_system_prompt_keeps_few_shot_example() -> None:
    assert "示例：" in SYSTEM_PROMPT
    assert "输入：客户说钱被扣了两次。" in SYSTEM_PROMPT
    assert "候选：1. 账单问题 2. 功能故障 3. 其他" in SYSTEM_PROMPT
    assert "输出：1" in SYSTEM_PROMPT


def test_choice_rendering_is_verbatim_to_legacy() -> None:
    """Byte-locks the choice rendering to ensure behavior is unchanged after templating."""
    question = Question(
        type="choice",
        instructions="选意图",
        criteria={"billing": "账单问题", "other": "其他"},
    )
    content = build_messages(STATE, question, question_items(question))[1]["content"]
    expected = (
        f"[输入]\n{STATE}\n\n"
        "[任务]\n选意图\n\n"
        "[候选]\n1. billing：账单问题\n2. other：其他\n\n"
        "[输出要求]\n"
        "只输出所选候选对应的编号（例如 1），不要输出任何其他内容。\n"
        f"{OUTPUT_TAIL}"
    )
    assert content == expected


def test_three_question_types_keep_key_fragments_and_tail() -> None:
    choice_q = Question(type="choice", instructions="选意图", criteria={"billing": "账单问题"})
    noul_q = Question(type="noul", instructions="客户明确要求退款。")
    score_q = Question(type="score", instructions="打分", scale=["0", "1", "2"])

    choice = build_messages(STATE, choice_q, question_items(choice_q))[1]["content"]
    noul = build_messages(STATE, noul_q, question_items(noul_q))[1]["content"]
    score = build_messages(STATE, score_q, question_items(score_q))[1]["content"]

    assert "只输出所选候选对应的编号" in choice
    assert "只输出 1" in noul and "只输出 2" in noul
    assert "只输出所选档位对应的编号" in score
    assert choice.endswith(OUTPUT_TAIL) and noul.endswith(OUTPUT_TAIL) and score.endswith(OUTPUT_TAIL)


def test_v2_is_english_and_available_alongside_v1() -> None:
    """v2 is the English template and coexists with v1.

    Why not simply swap the template to English: v1's rendering is byte-locked (see the test above),
    and every published measured number was taken on v1. Replacing the default template would
    decouple those numbers from the code, so English is provided as a new version while the default
    stays v1; using v2 requires explicitly switching in config and re-evaluating.
    """
    versions = available_versions()
    assert "v1" in versions and "v2" in versions

    # All four templates must exist and load
    v2 = load_prompt_set("v2")
    assert "decision model" in v2.system and "Example:" in v2.system
    assert "答案编号" not in v2.system      # The English version must not retain Chinese

    # All three question types must render, with every placeholder substituted
    for question in (
        Question(type="choice", instructions="Which team?",
                 criteria={"billing": "Charges", "technical": "Faults"}),
        Question(type="noul", instructions="The customer asks for a refund."),
        Question(type="score", instructions="How satisfied?", scale=["Low", "High"]),
    ):
        content = build_messages(STATE, question, question_items(question), version="v2")[1]["content"]
        assert not re.search(r"\{[a-z]+\}", content), f"v2 残留未替换占位符：{content}"
        assert "{state}" not in content and content.count(STATE) >= 1
        assert content.rstrip().endswith("nothing else):")

    choice = build_messages(
        STATE,
        Question(type="choice", instructions="Which team?",
                 criteria={"billing": "Charges", "technical": "Faults"}),
        question_items(Question(type="choice", instructions="Which team?",
                                criteria={"billing": "Charges", "technical": "Faults"})),
        version="v2",
    )[1]["content"]
    assert "[Candidates]" in choice and "[Output requirement]" in choice
    assert "只输出" not in choice


def test_option_separator_and_fallback_instructions_follow_the_version() -> None:
    """The candidate-line separator and the fallback used when instructions are empty are both
    determined by the version, not baked into the template body.

    These two are assembled at render time in the loader, so they need a per-version lookup. v1 is
    **frozen**: its rendering is verbatim-locked (see test_choice_rendering_is_verbatim_to_legacy),
    because every published number was measured on v1.
    """
    both = Question(type="choice", instructions="", criteria={"billing": "Charges"})
    items = question_items(both)

    v1 = build_messages(STATE, both, items, version="v1")[1]["content"]
    v2 = build_messages(STATE, both, items, version="v2")[1]["content"]

    # The Chinese template uses a full-width colon; the English template uses a half-width colon plus a space
    assert "1. billing：Charges" in v1
    assert "1. billing: Charges" in v2
    assert "：" not in v2

    # The fallback when instructions is empty must follow the language too, otherwise English users get a Chinese prompt
    assert "从下列候选中选出最合适的一个。" in v1
    assert "Pick the most fitting option below." in v2

    score = Question(type="score", instructions="", scale=["Low", "High"])
    assert "请把输入放到下列有序档位中的某一档。" in build_messages(
        STATE, score, question_items(score), version="v1")[1]["content"]
    assert "Place the input in one of the ordered levels below." in build_messages(
        STATE, score, question_items(score), version="v2")[1]["content"]


def test_prompt_version_cached_and_unknown_version_lists_available(monkeypatch) -> None:
    from llm2decision.prompts import loader

    loader._CACHE.clear()
    calls = {"count": 0}
    original = loader._read_template

    def counting_read(path):
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(loader, "_read_template", counting_read)
    try:
        first = load_prompt_set("v1")
        second = load_prompt_set("v1")
        assert first is second
        assert calls["count"] == 4  # The four templates are read only once
        assert "v1" in available_versions()
        with pytest.raises(UnknownPromptVersionError) as error:
            load_prompt_set("v9")
        assert "v9" in str(error.value) and "v1" in str(error.value)
    finally:
        loader._CACHE.clear()
