from argparse import ArgumentParser


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
            else:
                kwargs["type"] = type(value) if value is not None else str
            
            # Add help string if available
            if arg_name in help_dict:
                kwargs["help"] = help_dict[arg_name]

            group.add_argument(*flags, **kwargs)

    def extract(self, args):
        out = GroupParams()
        for k in vars(self).keys():
            name = k[1:] if k.startswith("_") else k
            setattr(out, name, getattr(args, name))
        return out

class EvalParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        group = parser.add_mutually_exclusive_group(required=True)
        self._model_path = None
        self.json = None
        self._help = {
            "model_path": "Path to the trained model directory (optional)",
            "json": "Path to the detection JSON file (optional)"
        }
        super().__init__(group)
