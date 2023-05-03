"""Output class."""


class EmulatorOutput:
    r"""A class to easily use the emulator output.

    Parameters
    ----------
    output : dict
        A dict containing the outputs of the emulator.
    """

    def __init__(self, output: dict) -> None:
        self.out_keys = [
            "delta" "k",
            "brightness_temp",
            "spin_temp",
            "tau_e",
            "Muv",
            "lfunc",
            "uv_lfs_redshifts",
            "ps_redshifts",
            "redshifts",
            "xHI",
        ]

        self.defining_dict = output
        self.k = output["k"]
        self.delta = output["delta"]
        self.brightness_temp = output["brightness_temp"]
        self.spin_temp = output["spin_temp"]
        self.tau_e = output["tau_e"]
        self.Muv = output["Muv"]
        self.lfunc = output["lfunc"]
        self.uv_lfs_redshifts = output["uv_lfs_redshifts"]
        self.ps_redshifts = output["ps_redshifts"]
        self.redshifts = output["redshifts"]
        self.xHI = output["xHI"]

    def add_errors(self, errors: dict) -> None:
        r"""Method to add the errors as attributes to the class.

        Parameters
        ----------
        errors : dict
            A dict containing the errors from the emulator.

        """
        for k in errors.keys():
            self.defining_dict[k] = errors[k]
        self.delta_err = errors["delta_err"]
        self.brightness_temp_err = errors["brightness_temp_err"]
        self.xHI_err = errors["xHI_err"]
        self.spin_temp_err = errors["spin_temp_err"]
        self.tau_e_err = errors["tau_e_err"]
