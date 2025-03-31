from argparse import ArgumentParser
import ast


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, group):
        help_dict = getattr(self, "_help", {})
        
        for key, value in vars(self).items():
            if key == "_help":
                continue
            is_short = key.startswith("_")
            arg_name = key[1:] if is_short else key
            flags = [f"--{arg_name}"]
            if is_short:
                flags.append(f"-{arg_name[0]}")

            kwargs = {"default": value}
            if isinstance(value, bool):
                kwargs["action"] = "store_true"
            elif isinstance(value, dict):
                kwargs["type"] = ast.literal_eval
            else:
                kwargs["type"] = type(value) if value is not None else str
            
            # Add help string if available
            if arg_name in help_dict:
                kwargs["help"] = help_dict[arg_name]

            group.add_argument(*flags, **kwargs)

    def extract(self, args):
        out = GroupParams()
        for k in vars(self).keys():
            if k == "_help":
                continue
            name = k[1:] if k.startswith("_") else k
            setattr(out, name, getattr(args, name))
        return out

class EvalParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        group = parser.add_argument_group("Evaluation parameters")
        self._model_path = None
        self._dataset_path = None
        self._evaluation_set_path = None
        self.json = None
        self._label_mappings = {1:1} # default
        self._Width = 400
        self._Height = 400
        self._results_path = None
        self._help = {
            "dataset_path": "Path to the dataset to be evaluated. Must contain a folder named images, and a folder named labels (optional)",
            "model_path": "Path to the trained model directory (optional)",
            "evaluation_set_path": "Path to a directory containing multiple datasets of the format required by dataset_path. Use for evaluatation of multiple datasets on the same model, comparing results.",
            "label_mappings": "Dictionary of class label keys, and values they should map to. This is used to compress multiple classes together if needed. Otherwise, map keys to values as appear in the labels folder for your dataset. Ex: {1:1, 2:2}",
            "results_path": "Path to directory for evaluation results",
            "json": "Path to the detection JSON file (optional)"
        }
        super().__init__(group)
