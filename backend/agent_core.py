import json
import re
from time import perf_counter

from pydantic import ValidationError

from backend.llm_service import (
    attach_usage_to_runtime_info,
    chat,
    get_llm_usage_record_count,
    llm,
    summarize_llm_usage_since,
    track_llm_usage,
)
from backend.schemas import AgentPlan
from backend.pending_actions import (
    SUPPORTED_PENDING_ACTION_TOOLS,
    get_pending_action_repository,
    normalize_action_arguments,
)
from backend.tool_registry import ToolCategory
from backend.tools import TOOL_REGISTRY, _is_valid_agent_context


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _extract_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _validate_agent_plan(data: dict) -> dict:
    if hasattr(AgentPlan, "model_validate"):
        plan = AgentPlan.model_validate(data)
    else:
        plan = AgentPlan.parse_obj(data)

    return _model_to_dict(plan)


def _proposable_tool_specs():
    safe_specs = (
        TOOL_REGISTRY.agent_specs()
        if hasattr(TOOL_REGISTRY, "agent_specs")
        else list(TOOL_REGISTRY.values())
    )
    dangerous_specs = [
        spec
        for spec in TOOL_REGISTRY.values()
        if (
            spec.category is ToolCategory.DANGEROUS
            and spec.name in SUPPORTED_PENDING_ACTION_TOOLS
        )
    ]
    seen = {spec.name for spec in safe_specs}
    return [*safe_specs, *[spec for spec in dangerous_specs if spec.name not in seen]]


def _tool_descriptions_for_prompt() -> str:
    return "\n".join(
        f"- {tool.name}：{tool.description}"
        + ("（只创建待确认操作，不会立即执行）" if tool.category is ToolCategory.DANGEROUS else "")
        for tool in _proposable_tool_specs()
    )


def _tool_names_for_prompt() -> str:
    return "|".join(tool.name for tool in _proposable_tool_specs())


_LEGACY_STUDY_TOOLS = {"explain", "summarize", "quiz", "flashcard"}


def _registry_tool_name(name: str) -> str:
    if not hasattr(TOOL_REGISTRY, "execute"):
        return name
    if name == "rag":
        return "rag_search"
    if name in _LEGACY_STUDY_TOOLS:
        return "study"
    return name


def _registry_has_tool(name: str) -> bool:
    return TOOL_REGISTRY.get(_registry_tool_name(name)) is not None


def _fallback_agent_plan(user_input: str, reason: str = "planner json parse failed") -> dict:
    lowered = user_input.lower()
    arguments = {}

    if (
        any(word in lowered for word in ["删除知识库", "删除文件", "delete knowledge", "delete file"])
        and re.search(r"\.(?:md|txt|pdf)\b", lowered)
    ):
        tool = "delete_knowledge_file"
        try:
            arguments = normalize_action_arguments(tool, {}, user_input)
        except ValueError:
            arguments = {}
    elif any(word in lowered for word in ["删除 run", "delete run", "删除运行记录"]):
        tool = "delete_run"
        try:
            arguments = normalize_action_arguments(tool, {}, user_input)
        except ValueError:
            arguments = {}
    elif any(word in lowered for word in ["重建索引", "rebuild index", "rebuild rag"]):
        tool = "rebuild_rag_index"
    elif any(word in lowered for word in ["重置索引", "清空索引", "reset index", "reset rag"]):
        tool = "reset_rag_index"
    elif any(word in lowered for word in ["清空笔记", "清空卡片", "清空题库", "清空练习", "reset saved"]):
        tool = "reset_saved_items"
        try:
            arguments = normalize_action_arguments(tool, {}, user_input)
        except ValueError:
            arguments = {}
    elif any(word in lowered for word in ["flashcard", "卡片", "记忆卡", "抽认卡"]):
        tool = "flashcard"
    elif any(word in lowered for word in ["quiz", "题", "练习", "测试"]):
        tool = "quiz"
    elif any(word in lowered for word in ["summary", "summarize", "总结", "摘要"]):
        tool = "summarize"
    elif any(word in lowered for word in ["rag", "知识库", "根据我的资料", "根据文档"]):
        tool = "rag"
    elif any(word in lowered for word in ["解释", "什么是", "what is", "why"]):
        tool = "explain"
    else:
        tool = "chat"

    if not _registry_has_tool(tool):
        tool = "chat"

    fallback_plan = {
        "goal": "根据用户请求选择最合适的学习助手能力。",
        "fallback": True,
        "steps": [
            {
                "tool": tool,
                "input": user_input,
                "reason": "使用本地规则生成的 fallback 单步计划。",
                **({"arguments": arguments} if arguments else {}),
            }
        ],
    }
    validated_plan = _validate_agent_plan(fallback_plan)
    validated_plan["fallback_reason"] = reason
    validated_plan["planner_json_parse"] = "失败"
    validated_plan["planner_schema_validate"] = "失败"
    return validated_plan


def plan_agent_steps(user_input: str, custom_llm=None, history_context: str | None = None) -> dict:
    active_llm = track_llm_usage(custom_llm or llm)
    history_block = history_context or "无"
    planner_prompt = f"""
你是 AI 学习助手的 Planner。
请把用户请求拆成 1 到 4 个执行步骤，并只返回 JSON object。

可用工具：
{_tool_descriptions_for_prompt()}

AgentPlan schema：
{{
  "goal": "用户任务目标，非空字符串",
  "steps": [
    {{
      "tool": "{_tool_names_for_prompt()}",
      "input": "传给工具的输入，非空字符串",
      "reason": "为什么使用这个工具",
      "arguments": {{"仅危险工具需要": "填写精确结构化参数；普通工具使用空对象"}}
    }}
  ],
  "fallback": false
}}

工具名只能是：{', '.join(tool.name for tool in _proposable_tool_specs())}。
危险工具只能在用户明确要求删除、清空、重置或重建时使用。不要根据暗示推断危险操作。
危险工具参数要求：
- delete_saved_item: {{"collection": "notes|flashcards|quizzes", "item_id": "..."}}
- delete_run: {{"target_run_id": "..."}}
- delete_knowledge_file: {{"filename": "..."}}
- reset_saved_items: {{"collection": "notes|flashcards|quizzes"}}，省略 collection 表示全部
- reset_rag_index / rebuild_rag_index: {{}}

示例 1：
用户输入：什么是 RAG
输出：
{{
  "goal": "解释 RAG 的概念",
  "steps": [
    {{
      "tool": "explain",
      "input": "RAG",
      "reason": "用户询问概念定义，使用 explain 工具解释"
    }}
  ],
  "fallback": false
}}

示例 2：
用户输入：请解释 RAG，并出 3 道练习题
输出：
{{
  "goal": "解释 RAG 并生成练习题",
  "steps": [
    {{
      "tool": "explain",
      "input": "RAG",
      "reason": "先解释概念"
    }},
    {{
      "tool": "quiz",
      "input": "基于 RAG 生成 3 道练习题",
      "reason": "用户要求出题"
    }}
  ],
  "fallback": false
}}

示例 3：
用户输入：根据知识库解释 agentic rag，并出 3 道练习题
输出：
{{
  "goal": "基于知识库解释 agentic rag 并生成练习题",
  "steps": [
    {{
      "tool": "rag",
      "input": "agentic rag",
      "reason": "用户要求根据知识库回答，先检索相关内容"
    }},
    {{
      "tool": "explain",
      "input": "基于知识库内容解释 agentic rag",
      "reason": "解释检索到的概念"
    }},
    {{
      "tool": "quiz",
      "input": "基于 agentic rag 生成 3 道练习题",
      "reason": "用户要求生成练习题"
    }}
  ],
  "fallback": false
}}

示例 4：
用户输入：根据知识库解释 agentic rag，生成记忆卡片，并出 3 道练习题
输出：
{{
  "goal": "基于知识库解释 agentic rag，生成记忆卡片并出题",
  "steps": [
    {{
      "tool": "rag",
      "input": "agentic rag",
      "reason": "用户要求根据知识库回答，先检索相关内容"
    }},
    {{
      "tool": "explain",
      "input": "基于知识库内容解释 agentic rag",
      "reason": "先帮助用户理解概念"
    }},
    {{
      "tool": "flashcard",
      "input": "基于 agentic rag 生成 3-5 张复习记忆卡片",
      "reason": "用户要求生成记忆卡片用于复习"
    }},
    {{
      "tool": "quiz",
      "input": "基于 agentic rag 生成 3 道练习题",
      "reason": "用户要求生成练习题"
    }}
  ],
  "fallback": false
}}

最近对话：
{history_block}

当前用户请求：
{user_input}

最终输出要求：
你必须只输出一个 JSON object。
不要输出 Markdown。
不要输出 ```json。
不要输出任何解释文字。
不要输出 schema 以外的字段。
JSON 必须符合 AgentPlan schema。
"""
    response = active_llm.invoke(planner_prompt)
    data = _extract_json_object(response.content)

    if not data:
        return _fallback_agent_plan(user_input, reason="planner json parse failed")

    try:
        plan = _validate_agent_plan(data)
    except ValidationError as exc:
        fallback_plan = _fallback_agent_plan(
            user_input,
            reason=f"planner schema validate failed: {exc.errors()[0].get('msg')}",
        )
        fallback_plan["planner_json_parse"] = "成功"
        return fallback_plan

    if not plan.get("steps"):
        fallback_plan = _fallback_agent_plan(user_input, reason="planner returned empty steps")
        fallback_plan["planner_json_parse"] = "成功"
        return fallback_plan

    if any(not _registry_has_tool(step.get("tool", "")) for step in plan["steps"]):
        fallback_plan = _fallback_agent_plan(user_input, reason="planner returned unknown tool")
        fallback_plan["planner_json_parse"] = "成功"
        return fallback_plan

    plan["steps"] = plan["steps"][:4]
    plan["fallback"] = False
    plan["planner_json_parse"] = "成功"
    plan["planner_schema_validate"] = "成功"
    return plan


def _execute_agent_tool(
    tool: str,
    tool_input: str,
    tool_arguments: dict | None = None,
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    registry_tool = _registry_tool_name(tool)
    tool_spec = TOOL_REGISTRY.get(registry_tool)

    if tool_spec is None:
        fallback_tool = TOOL_REGISTRY["chat"]
        arguments = {
            "step_input": tool_input,
            "original_input": (shared_context or {}).get("original_input", ""),
            "custom_llm": custom_llm,
            "top_k": top_k,
            "shared_context": shared_context,
        }
        result = (
            TOOL_REGISTRY.execute("chat", actor="agent", **arguments)
            if hasattr(TOOL_REGISTRY, "execute")
            else fallback_tool.run(**arguments)
        )
        result["trace"].append(f"Agent unknown tool：{tool}，已回退到 chat")
        result["tool_name"] = "chat"
        result["tool_description"] = fallback_tool.description
        result["tool_success"] = True
        return result

    if tool_spec.category is ToolCategory.DANGEROUS:
        run_id = str((shared_context or {}).get("run_id") or "").strip()
        if not run_id:
            raise ValueError("Dangerous tool proposals require a persisted Run")
        try:
            arguments = normalize_action_arguments(
                registry_tool,
                tool_arguments,
                tool_input,
            )
        except ValueError as exc:
            return {
                "answer": str(exc),
                "trace": [f"Pending Action 未创建：{exc}"],
                "sources": [],
                "flashcards": [],
                "tool_name": tool_spec.name,
                "tool_description": tool_spec.description,
                "tool_success": False,
                "error": str(exc),
            }

        action = get_pending_action_repository().create(
            run_id=run_id,
            session_id=(shared_context or {}).get("session_id"),
            tool_name=registry_tool,
            arguments=arguments,
            request_message=(shared_context or {}).get("original_input", tool_input),
        )
        action_payload = (
            action.model_dump(mode="json")
            if hasattr(action, "model_dump")
            else action.dict()
        )
        return {
            "answer": f"我已准备好“{action.summary}”，但尚未执行。请检查下方待确认操作。",
            "trace": [f"Pending Action 已创建：{action.id}"],
            "sources": [],
            "flashcards": [],
            "pending_action": action_payload,
            "tool_name": tool_spec.name,
            "tool_description": tool_spec.description,
            "tool_success": True,
        }

    arguments = {
        "step_input": tool_input,
        "original_input": (shared_context or {}).get("original_input", ""),
        "custom_llm": custom_llm,
        "top_k": top_k,
        "shared_context": shared_context,
    }
    if tool in _LEGACY_STUDY_TOOLS and hasattr(TOOL_REGISTRY, "execute"):
        arguments["operation"] = tool
    result = (
        TOOL_REGISTRY.execute(registry_tool, actor="agent", **arguments)
        if hasattr(TOOL_REGISTRY, "execute")
        else tool_spec.run(**arguments)
    )
    result["tool_name"] = tool_spec.name
    result["tool_description"] = tool_spec.description
    result["tool_success"] = True
    return result


def run_agent(
    user_input: str,
    custom_llm=None,
    prefer_rag: bool = False,
    top_k: int = 3,
    retrieval_mode: str = "vector",
    reranker_enabled: bool = False,
    history_context: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    active_llm = track_llm_usage(custom_llm or llm)
    trace = ["Agent Planner：开始分析用户请求"]
    trace.append(f"Agent Planner 使用 history：{'是' if history_context else '否'}")
    trace.append(f"Agent Planner retrieval_mode：{retrieval_mode}")
    trace.append(f"Agent Planner reranker_enabled：{reranker_enabled}")
    planner_usage_started_at = get_llm_usage_record_count(active_llm)
    planner_started_at = perf_counter()
    plan = plan_agent_steps(
        user_input,
        custom_llm=active_llm,
        history_context=history_context,
    )
    planner_latency_ms = _elapsed_ms(planner_started_at)

    trace.append(f"Agent Planner goal：{plan.get('goal')}")
    trace.append(f"Agent Planner JSON parse：{plan.get('planner_json_parse', '成功')}")
    trace.append(f"Agent Planner schema validate：{plan.get('planner_schema_validate', '成功')}")
    trace.append(f"Agent Planner fallback：{'是' if plan.get('fallback') else '否'}")
    trace.append(f"Agent Planner latency_ms={planner_latency_ms}")
    if plan.get("fallback_reason"):
        trace.append(f"Agent Planner fallback reason：{plan['fallback_reason']}")

    if prefer_rag and not any(step.get("tool") == "rag" for step in plan["steps"]):
        plan["steps"].insert(0, {
            "tool": "rag",
            "input": user_input,
            "reason": "use_rag=true，优先尝试知识库检索",
        })
        plan["steps"] = plan["steps"][:4]
        trace.append("Agent Planner：use_rag=true，已插入 rag step")

    if run_id:
        from backend.run_repository import get_run_repository

        get_run_repository().update_run(
            run_id,
            plan=plan.get("steps", []),
            metadata={
                "planner": {
                    "goal": plan.get("goal"),
                    "fallback": bool(plan.get("fallback")),
                    "error": plan.get("fallback_reason"),
                    "mode": "agent",
                }
            },
        )

    previous_result = ""
    step_outputs = []
    all_sources = []
    all_flashcards = []
    pending_actions = []
    fallback_used = False
    retrieval_summary = {
        "retrieval_mode": retrieval_mode,
        "candidate_k": None,
        "vector_candidates": 0,
        "bm25_candidates": 0,
        "hybrid_used": retrieval_mode == "hybrid",
        "reranker_enabled": reranker_enabled,
        "reranker_used": False,
        "reranker_model": None,
        "reranker_top_n": None,
        "reranker_error": None,
    }
    tool_calls = [
        {
            "node": "planner",
            "tool": "planner",
            "description": "Agent Planner",
            "success": not bool(plan.get("fallback")),
            "used_context": bool(history_context),
            "context_sources": ["history"] if history_context else [],
            "output_length": len(plan.get("steps", [])),
            "latency_ms": planner_latency_ms,
        }
    ]
    planner_usage_delta = summarize_llm_usage_since(active_llm, planner_usage_started_at)
    if planner_usage_delta:
        tool_calls[0].update(planner_usage_delta)
    shared_context = {
        "run_id": run_id,
        "session_id": session_id,
        "original_input": user_input,
        "history_context": history_context or "",
        "retrieval_mode": retrieval_mode,
        "reranker_enabled": reranker_enabled,
        "rag_context": "",
        "sources": [],
        "step_outputs": [],
        "last_output": "",
    }

    for index, step in enumerate(plan["steps"], start=1):
        tool = step["tool"]
        tool_input = step.get("input") or user_input
        if "{previous_result}" in tool_input:
            tool_input = tool_input.replace("{previous_result}", previous_result)

        trace.append(f"Agent Step {index} tool={tool}")
        trace.append(f"Agent Step {index} plan：tool={tool}, reason={step.get('reason')}")
        tool_usage_started_at = get_llm_usage_record_count(active_llm)
        tool_started_at = perf_counter()
        result = _execute_agent_tool(
            tool,
            tool_input,
            tool_arguments=step.get("arguments"),
            custom_llm=active_llm,
            top_k=top_k,
            shared_context=shared_context,
        )
        tool_latency_ms = _elapsed_ms(tool_started_at)
        result["latency_ms"] = tool_latency_ms
        result_answer = result.get("answer", "")
        result_sources = result.get("sources", [])
        result_flashcards = result.get("flashcards", [])
        previous_result = result_answer
        all_sources.extend(result_sources)
        all_flashcards.extend(result_flashcards)
        fallback_used = fallback_used or result.get("fallback_used", False)
        trace.append(f"Agent Step {index} 工具说明：{result.get('tool_description', '')}")
        trace.append(f"Agent Step {index} 工具执行成功：{'是' if result.get('tool_success') else '否'}")
        trace.append(f"Agent Step {index} latency_ms={tool_latency_ms}")
        trace.extend(result.get("trace", []))
        trace.append(f"Agent Step {index} 使用上下文：{'是' if result.get('used_context') else '否'}")
        if result.get("context_sources"):
            trace.append(f"Agent Step {index} 上下文来源：{' + '.join(result['context_sources'])}")

        tool_call = {
            "node": result.get("tool_name") or tool,
            "tool": result.get("tool_name") or tool,
            "description": result.get("tool_description", ""),
            "success": bool(result.get("tool_success")),
            "used_context": bool(result.get("used_context")),
            "context_sources": result.get("context_sources", []),
            "output_length": len(result_answer),
            "latency_ms": tool_latency_ms,
        }
        tool_usage_delta = summarize_llm_usage_since(active_llm, tool_usage_started_at)
        if tool_usage_delta:
            tool_call.update(tool_usage_delta)
        if result.get("tool_name") and result["tool_name"] != tool:
            tool_call["requested_tool"] = tool
        if result.get("error"):
            tool_call["error"] = result["error"]
        if result.get("retrieval_info"):
            tool_call.update(result["retrieval_info"])
            retrieval_summary.update(result["retrieval_info"])
        tool_calls.append(tool_call)

        if result.get("pending_action"):
            pending_actions.append(result["pending_action"])

        if tool == "rag" and _is_valid_agent_context(result.get("context")):
            shared_context["rag_context"] = result["context"]
            shared_context["sources"] = result_sources

        shared_context["step_outputs"].append({
            "tool": tool,
            "input": tool_input,
            "answer": result_answer,
        })
        shared_context["last_output"] = result_answer
        step_outputs.append(f"步骤 {index}（{tool}）：\n{result_answer}")
        trace.append(f"Agent Step {index} done：输出长度 {len(result_answer)}")
        if result.get("pending_action"):
            trace.append("Agent execution paused：等待用户确认操作")
            break

    answer = "\n\n".join(step_outputs) if step_outputs else chat(
        user_input,
        custom_llm=active_llm,
        history_context=history_context,
    )

    return {
        "answer": answer,
        "trace": trace,
        "plan": plan,
        "sources": all_sources,
        "flashcards": all_flashcards,
        "fallback_used": fallback_used,
        "runtime_info": attach_usage_to_runtime_info({
            "runtime": "agent",
            "graph_path": ["planner", *[call["tool"] for call in tool_calls[1:]]],
            "node_count": len(tool_calls),
            "tool_calls": tool_calls,
            "finalizer_used": False,
            "planner_mode": "agent",
            "planner_fallback": bool(plan.get("fallback")),
            "planner_error": plan.get("fallback_reason") or None,
            "error": None,
            "awaiting_action": bool(pending_actions),
            **retrieval_summary,
        }, active_llm),
        "pending_actions": pending_actions,
    }


def agent_router(user_input: str, custom_llm=None, history_context: str | None = None) -> str:
    return run_agent(
        user_input,
        custom_llm=custom_llm,
        history_context=history_context,
    )["answer"]
