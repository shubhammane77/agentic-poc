from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

try:
    import dspy
    from dspy.adapters.types.tool import Tool
    from dspy.primitives.module import Module
    from dspy.signatures.signature import ensure_signature
    from dspy.utils.exceptions import ContextWindowExceededError
except ImportError:  # pragma: no cover - optional runtime dependency
    dspy = None  # type: ignore[assignment]
    Tool = Any  # type: ignore[assignment,misc]
    Module = object  # type: ignore[assignment,misc]
    ensure_signature = None  # type: ignore[assignment]
    ContextWindowExceededError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class CustomReAct(Module):
    def __init__(
        self,
        signature: type["Signature"] | str,
        tools: list[Callable[..., Any]],
        max_iters: int = 20,
        max_format_retries: int = 3,
    ) -> None:
        super().__init__()
        if dspy is None or ensure_signature is None:
            raise RuntimeError("DSPy is required to use CustomReAct.")

        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.max_format_retries = max(1, max_format_retries)

        dspy_tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tool_map = {tool.name: tool for tool in dspy_tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting next_tool_name and next_tool_args, the tool must be one of:\n",
            ]
        )

        tool_map["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=(
                "Marks the task as complete. That is, signals that all information for producing "
                f"the outputs, i.e. {outputs}, are now available to be extracted."
            ),
            args={},
        )

        for idx, tool in enumerate(tool_map.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing next_tool_args, the value must be valid JSON and parse into a dictionary.")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tool_map.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )
        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tool_map
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]) -> str:
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def _validate_react_prediction(self, pred: Any) -> None:
        if not hasattr(pred, "next_tool_name") or not hasattr(pred, "next_tool_args"):
            raise ValueError("Missing required fields `next_tool_name` and/or `next_tool_args`.")
        if pred.next_tool_name not in self.tools:
            raise ValueError(f"Unknown tool selected: {pred.next_tool_name}")
        if not isinstance(pred.next_tool_args, dict):
            raise ValueError("`next_tool_args` must be a JSON object (dict).")

    def _call_with_potential_trajectory_truncation(self, module: Any, trajectory: dict[str, Any], **input_args: Any) -> Any:
        for _ in range(3):
            try:
                return module(**input_args, trajectory=self._format_trajectory(trajectory))
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded context window, truncating oldest tool call.")
                trajectory = self.truncate_trajectory(trajectory)
        raise ValueError("Context window exceeded even after truncating trajectory 3 times.")

    async def _async_call_with_potential_trajectory_truncation(
        self, module: Any, trajectory: dict[str, Any], **input_args: Any
    ) -> Any:
        for _ in range(3):
            try:
                return await module.acall(**input_args, trajectory=self._format_trajectory(trajectory))
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded context window, truncating oldest tool call.")
                trajectory = self.truncate_trajectory(trajectory)
        raise ValueError("Context window exceeded even after truncating trajectory 3 times.")

    def _predict_react_with_retry(self, trajectory: dict[str, Any], **input_args: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_format_retries + 1):
            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
                self._validate_react_prediction(pred)
                return pred
            except Exception as err:  # noqa: BLE001 - broad by design for robust retry
                last_error = err if isinstance(err, Exception) else Exception(str(err))
                logger.warning("ReAct format/tool validation failed on attempt %s/%s: %s", attempt, self.max_format_retries, _fmt_exc(last_error))
        raise ValueError(f"Agent failed to produce a valid tool action after {self.max_format_retries} attempts: {last_error}")

    async def _async_predict_react_with_retry(self, trajectory: dict[str, Any], **input_args: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_format_retries + 1):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
                self._validate_react_prediction(pred)
                return pred
            except Exception as err:  # noqa: BLE001 - broad by design for robust retry
                last_error = err if isinstance(err, Exception) else Exception(str(err))
                logger.warning("Async ReAct format/tool validation failed on attempt %s/%s: %s", attempt, self.max_format_retries, _fmt_exc(last_error))
        raise ValueError(f"Agent failed to produce a valid tool action after {self.max_format_retries} attempts: {last_error}")

    def forward(self, **input_args: Any) -> Any:
        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._predict_react_with_retry(trajectory, **input_args)
            except ValueError as err:
                logger.warning("Ending trajectory: agent failed to select a valid tool: %s", _fmt_exc(err))
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:  # noqa: BLE001
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args: Any) -> Any:
        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._async_predict_react_with_retry(trajectory, **input_args)
            except ValueError as err:
                logger.warning("Ending trajectory: agent failed to select a valid tool: %s", _fmt_exc(err))
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:  # noqa: BLE001
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def truncate_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        keys = list(trajectory.keys())
        if len(keys) < 4:
            raise ValueError(
                "Trajectory exceeded context window but cannot be truncated because it only has one tool call."
            )
        for key in keys[:4]:
            trajectory.pop(key)
        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
