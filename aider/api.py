
from io import StringIO
import logging
from typing import List, Tuple, Any, Dict
import yaml

from aider import main
from aider.coders.base_coder import Coder
from aider.commands import SwitchCoder
from aider.utils import is_image_file
from prompt_toolkit.output import create_output
from prompt_toolkit.input.base import PipeInput

from aider.coders.manager_coder import ManagerCoder, TaskEndedException

fence = "`" * 3

def getMaxTokens() -> int:

    with open('.aider.conf.yml', 'r') as f:
        config = yaml.safe_load(f)

    selectedModel: str = config['model']

    with open('.aider.model.settings.yml', 'r') as f:
        modelSettings = yaml.safe_load(f)

    for model in modelSettings:
        if model['name'] == selectedModel:
            return model["extra_params"]['max_tokens']
    
    raise Exception(f"Could not find model with name {selectedModel} in .aider.model.settings.yml")


logger = logging.getLogger("aider.api")


class AiderInterfaceException(Exception):
    pass


class AiderInterface:

    def __init__(self, input: PipeInput, repoDir: str, conventionsFile: str|None="CONVENTIONS.md") -> None:
  
        self.input = input
        self.output = create_output(StringIO())

        self.coder: Coder = main.main(
            input=input,
            output=self.output,
            return_coder=True,
        )
        self.coder.toApiMode()
        self.coder = self._getManager(self.coder)
        self.totalCost = 0.0
        self.MAX_TOKENS = 200000

        if conventionsFile is not None:
            self.sendInput(f'"{conventionsFile}"\n', '/read')


    def _getManager(self, coder: Coder) -> Any:

        kwargs = dict(io=coder.io, from_coder=coder, for_api_usage=True)
        kwargs.update({
            "edit_format": "manager",
        })
        coder: ManagerCoder = Coder.create(**kwargs)
        return coder


    def sendInput(self, data: str, mode: str = "") -> str:

        promptStr: str = data if mode == "" else mode + " " + data

        with self.coder.io.console.capture() as capture:

            try:
                self.coder.run(promptStr)
            
            # If we need to switch the coder type due to e.g. /architect, that gets handled here
            except SwitchCoder as switch:
                kwargs = dict(io=self.coder.io, from_coder=self.coder)
                kwargs.update(switch.kwargs)
                if "show_announcements" in kwargs:
                    del kwargs["show_announcements"]

                coder = Coder.create(**kwargs)

                if switch.kwargs.get("show_announcements") is not False:
                    coder.show_announcements()

            except TaskEndedException as e:
                logger.info("Task ended. Exiting cleanup.")
                raise e
            

        response: str = capture.get()

        self.totalCost = self.coder.total_cost
        
        # self.logContextUsage()

        return response
    

    # Ask Aider to reflect on the changes just made.
    def reflect(self) -> str:
        return self.coder.reflect()
    

    def getContextUsage(self) -> Tuple[int, Dict[str, int]]:
        # response: str = self.sendInput('/tokens\n')

        contextTypeToTokens: Dict[str, int] = {}

        # System messages
        main_sys: List[str] = self.coder.fmt_system_prompt(self.coder.gpt_prompts.main_system)
        main_sys += "\n" + self.coder.fmt_system_prompt(self.coder.gpt_prompts.system_reminder)
        msgs: List[str] = [
            dict(role="system", content=main_sys),
            dict(
                role="system",
                content=self.coder.fmt_system_prompt(self.coder.gpt_prompts.system_reminder),
            ),
        ]
        contextTypeToTokens["system_messages"] = self.coder.main_model.token_count(msgs)

        # Chat history
        msgs: List[str] = self.coder.done_messages + self.coder.cur_messages
        contextTypeToTokens["chat_history"] = self.coder.main_model.token_count(msgs) if msgs else 0

        # Repo map
        other_files = set(self.coder.get_all_abs_files()) - set(self.coder.abs_fnames)
        tokens: int = 0
        if self.coder.repo_map:
            repo_content = self.coder.repo_map.get_repo_map(self.coder.abs_fnames, other_files)
            tokens = self.coder.main_model.token_count(repo_content) if repo_content else 0
    
        contextTypeToTokens["repo_map"] = tokens


        # Files
        filesTokens: int = 0
        for fname in self.coder.abs_fnames:
            relative_fname = self.coder.get_rel_fname(fname)
            content = self.coder.io.read_text(fname)
            if is_image_file(relative_fname):
                tokens = self.coder.main_model.token_count_for_image(fname)
            else:
                # approximate
                content = f"{relative_fname}\n{fence}\n" + content + "{fence}\n"
                tokens = self.coder.main_model.token_count(content)

            filesTokens += tokens

        contextTypeToTokens["files"] = filesTokens

        # Read-Only Files
        readonlyFilesTokens: int = 0
        for fname in self.coder.abs_read_only_fnames:
            relative_fname = self.coder.get_rel_fname(fname)
            content = self.coder.io.read_text(fname)
            if content is not None and not is_image_file(relative_fname):
                # approximate
                content = f"{relative_fname}\n{fence}\n" + content + "{fence}\n"
                tokens = self.coder.main_model.token_count(content)

            readonlyFilesTokens += tokens

        contextTypeToTokens["readonly_files"] = filesTokens

        totalTokens: int = sum(contextTypeToTokens.values())
        
        return totalTokens, contextTypeToTokens
    

    def logContextUsage(self) -> str:
        contextUsage, contextTypeToTokens = self.getContextUsage()

        logger.debug(f"Context usage: {contextUsage} tokens ({contextUsage/self.MAX_TOKENS*100:.2f}% of max context size)")

        for k,v in contextTypeToTokens.items():
            logger.debug(f"{k}: {v} tokens ({v/contextUsage*100:.2f}% of used context)")
        
        



