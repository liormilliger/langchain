cd ~/langchain && source venv/bin/activate
python3 -c "
import inspect
from langchain.agents.middleware.pii import PIIMiddleware
src = inspect.getsource(PIIMiddleware.before_model)
print(src[:2500])
"

(venv) root@ip-172-31-41-244:~/langchain# python3 -c "
import inspect
from langchain.agents.middleware.pii import PIIMiddleware
src = inspect.getsource(PIIMiddleware.before_model)
print(src[:2500])
"

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages = list(messages)
        any_modified = False

        # Check user input if enabled
        if self.apply_to_input:
            # Get last user message
            last_user_msg = None
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_user_msg = messages[i]
                    last_user_idx = i
                    break

            if last_user_idx is not None and last_user_msg and last_user_msg.content:
                # Detect PII in message content
                content = str(last_user_msg.content)
                new_content, matches = self._process_content(content)

                if matches:
                    updated_message: AnyMessage = HumanMessage(
                        content=new_content,
                        id=last_user_msg.id,
                        name=last_user_msg.name,
                    )

                    new_messages[last_user_idx] = updated_message
                    any_modified = True

        # Check tool results if enabled
        if self.apply_to_tool_results:
            # Find the last AIMessage, then process all `ToolMessage` objects after it
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is not None:
                # Get all tool messages after the last AI message
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMes
(venv) root@ip-172-31-41-244:~/langchain# 


=============NEW==============
python3 -c "
import inspect
from langchain.agents.middleware.pii import PIIMiddleware
print(inspect.signature(PIIMiddleware.__init__))
"

(self, pii_type: "Literal['email', 'credit_card', 'ip', 'mac_address', 'url'] | str", *, strategy: "Literal['block', 'redact', 'mask', 'hash']" = 'redact', detector: 'Callable[[str], list[PIIMatch]] | str | None' = None, apply_to_input: 'bool' = True, apply_to_output: 'bool' = False, apply_to_tool_results: 'bool' = False) -> 'None'

