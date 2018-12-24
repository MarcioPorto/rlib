import os

import torch


class Agent:
    def save_state_dicts(self):
        r"""Save state dicts to file."""
        if not self.model_output_dir:
            raise Exception("You must provide an output directory to save state dict.")

        for comb in self.struct:
            torch.save(
                comb[0].state_dict(),
                os.path.join(self.model_output_dir, "{}.pth".format(comb[1]))
            )

    def load_state_dicts(self):
        r"""Load state dicts from file."""
        if not self.model_output_dir:
            raise Exception("You must provide an input directory to load state dict.")

        for comb in self.struct:
            comb[0].load_state_dict(
                torch.load(os.path.join(self.model_output_dir, "{}.pth".format(comb[1])))
            )
