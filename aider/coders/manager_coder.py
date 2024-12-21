
from aider import __version__
from aider.coders.ask_coder import AskCoder
from aider.coders.manager_coder_prompts import ManagerPrompts
from aider.coders.base_coder import FinishReasonLength, Coder
from aider.llm import litellm
from aider.sendchat import RETRY_TIMEOUT, retry_exceptions

import json
import openai
import time
import traceback
from typing import Dict, Tuple, List, Any

class TaskEndedExeption(Exception):
    pass


class ManagerCoder(AskCoder):

    edit_format = "manager"
    gpt_prompts = ManagerPrompts()

    functions = [
            dict(
                name="add_file",
                description="add a file to the chat context",
                # strict=True,
                parameters=dict(
                    type="object",
                    properties=dict(
                        explanation=dict(
                            type="string",
                            description=(
                                "Short explanation of why the file should be added to the chat context."
                            ),
                        ),
                        filepath=dict(
                            type="string",
                            description="Path to the file to add to the chat context.",
                        ),
                    ),
                    required=["explanation", "filepath"],
                    additionalProperties=False,
                ),
            ),
            dict(
                name="remove_file",
                description="remove a file from the chat context",
                # strict=True,
                parameters=dict(
                    type="object",
                    properties=dict(
                        explanation=dict(
                            type="string",
                            description=(
                                "Short explanation of why the file should be removed from the chat context."
                            ),
                        ),
                        filepath=dict(
                            type="string",
                            description="Path to the file to remove from the chat context.",
                        ),
                    ),
                    required=["explanation", "filepath"],
                    additionalProperties=False,
                ),
            ),
            dict(
                name="make_edits",
                description="instruct the editor engineer to plan and make changes to the code",
                parameters=dict(
                    type="object",
                    properties=dict(
                        explanation=dict(
                            type="string",
                            description=(
                                "A thorough explanation of the changes that need to be made."
                            ),
                        ),
                        filepath=dict(
                            type="string",
                            description="Path to the file to which the changes must be made.",
                        ),
                    ),
                    required=["explanation", "filepath"],
                    additionalProperties=False,
                ),
            ),
            dict(
                name="stop_edits",
                description="declare the end of the editing process",
            ),
        ]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = 1


    def create(*args, **kwargs):
        return ManagerCoder(*args, **kwargs)


    # TODO: might need to update this to handle multiple tool calls
    def update_cur_messages(self):

        functionName: str = self.partial_response_function_calls[0].get("name") if len(self.partial_response_function_calls) > 0 else ""
        args: Dict = self.parse_partial_args() if len(self.partial_response_function_calls) > 0 else {}
        
        self.partial_response_content = self.partial_response_content or self.functionChatOutput or f"I am going to use the `{functionName}` function with arguments: {args}\n. This is the best next step because {args.get("explanation")}\n"
        super().update_cur_messages()


    def reply_completed(self):
        if self.partial_response_function_calls:
            self.handlePossibleToolCalls()


    def handlePossibleToolCalls(self):

        args: Dict = self.parse_partial_args()

        for fnCallId, fnCall, fnCallArgs in zip(self.partial_response_function_calls_id, self.partial_response_function_calls, args):

            functionName: str = fnCall.get("name")

            content = self.partial_response_content
            
            self.functionChatOutput: str = self.execFunction(fnCallId, functionName, fnCallArgs, content)
        

        list(self.sendToolMessages()) # We need to wrap in list() so items in the returned generator are actually called


    def show_send_output(self, completion):

        try:
            if completion.choices[0].message.tool_calls:
                self.partial_response_function_calls = [ tool_call.function for tool_call in completion.choices[0].message.tool_calls ]
                self.partial_response_function_calls_id = [ tool_call.id for tool_call in completion.choices[0].message.tool_calls ]

        except AttributeError as func_err:
            show_func_err = func_err

        # functionName: str = self.partial_response_function_call.get("name")
        # args: Dict = self.parse_partial_args()

        self.cur_messages += [ completion.choices[0].message.to_dict() ]

        # completion.choices[0].message.content = f"I am going to use the `{functionName}` function with arguments: {args}\n. This is the best next step because {args.get("explanation")}\n"
        super().show_send_output(completion)


    def execFunction(self, fnCallId: str, functionName: str, args: Dict, content: str) ->  None:
        
        result: str

        if functionName == "make_edits":
            result, output = self._make_edits(args, content)

        elif functionName == "stop_edits":
            result, output = "I have declared the end of the editing process.", None
            raise 
        
        elif functionName == "add_file":
            result, output = self._add_file(args)

        elif functionName == "remove_file":
            result, output = self._remove_file(args)

        elif functionName == "check_files":
            result, output = self._check_files(args)

        else:
            raise ValueError(f"Unknown function name: {function}")
        
        self.queueToolResult(fnCallId, result, output)
        
        # TODO: do we need this?
        # self.reply_completed()
        return result
    

    def queueToolResult(self, fnCallId: str, result: str, output:Any) -> None:
        content: str|None = json.dumps(result) if isinstance(result, dict) else result

        newMsg: Dict[str, str] = dict(
            role="tool",
            content=content,
            tool_call_id=fnCallId,
        )
        self.cur_messages += [ newMsg ]


    def sendToolMessages(self):
            
        chunks = self.format_messages()
        messages = chunks.all_messages()

        while True:
            try:
                yield from self.send(messages, functions=self.functions)
                break
            except retry_exceptions() as err:
                # Print the error and its base classes
                # for cls in err.__class__.__mro__: dump(cls.__name__)

                retry_delay *= 2
                if retry_delay > RETRY_TIMEOUT:
                    self.mdstream = None
                    self.check_and_open_urls(err)
                    break
                err_msg = str(err)
                self.io.tool_error(err_msg)
                self.io.tool_output(f"Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                continue
            except KeyboardInterrupt:
                interrupted = True
                break
            except litellm.ContextWindowExceededError:
                # The input is overflowing the context window!
                exhausted = True
                break
            except litellm.exceptions.BadRequestError as br_err:
                self.io.tool_error(f"BadRequestError: {br_err}")
                return
            except FinishReasonLength:
                # We hit the output limit!
                if not self.main_model.info.get("supports_assistant_prefill"):
                    exhausted = True
                    break

                self.multi_response_content = self.get_multi_response_content()

                if messages[-1]["role"] == "assistant":
                    messages[-1]["content"] = self.multi_response_content
                else:
                    messages.append(
                        dict(role="assistant", content=self.multi_response_content, prefix=True)
                    )
            except (openai.APIError, openai.APIStatusError) as err:
                self.mdstream = None
                self.check_and_open_urls(err)
                break
            except Exception as err:
                lines = traceback.format_exception(type(err), err, err.__traceback__)
                self.io.tool_warning("".join(lines))
                self.io.tool_error(str(err))
                return


    def _add_file(self, args: Dict) -> Tuple[str, None]:
        self.run(f"/add {args['filepath']}")
        
        return f"I have added the file {args["filepath"]} to the chat.", None


    def _remove_file(self, args: Dict) -> Tuple[str, None]:
        self.run(f"/drop {args['filepath']}")

        return f"I have removed the file {args["filepath"]} from the chat.", None


    # Actually not sure if this tool is needed, since the files are already injected into LLM API calls via the repo map
    def _check_files(self, args: Dict) -> Tuple[str, List[str]]:
        self.run("/ls")

        return "I am checking the files already in the chat.", [] # TODO: return the list of files in the chat

    
    def _make_edits(self, args: Dict, content: str) -> Tuple[str, None]:
        
        kwargs = dict()

        kwargs["main_model"] = self.main_model
        kwargs["edit_format"] = self.main_model.edit_format
        kwargs["suggest_shell_commands"] = False
        kwargs["map_tokens"] = 0
        kwargs["total_cost"] = self.total_cost
        kwargs["cache_prompts"] = False
        kwargs["num_cache_warming_pings"] = 0
        kwargs["summarize_from_coder"] = False

        new_kwargs = dict(io=self.io, from_coder=self)
        new_kwargs.update(kwargs)

        arch_coder = Coder.create(**new_kwargs)
        arch_coder.cur_messages = []
        arch_coder.done_messages = []

        if self.verbose:
            arch_coder.show_announcements()

        editorContent: str = content + "\n" + self.gpt_prompts.editFocusPrompt.format(CHANGE_REQUEST=args.get("explanation", ""))

        arch_coder.run(with_message=editorContent, preproc=False)

        # Need to leave this out compared to ArchitectCoder to avoid "summarizing away" the tool messages
        # self.move_back_cur_messages("I made those changes to the files.")
        self.total_cost = arch_coder.total_cost
        self.aider_commit_hashes = arch_coder.aider_commit_hashes

        return "I have asked the editor engineer to make changes to the code.", None
