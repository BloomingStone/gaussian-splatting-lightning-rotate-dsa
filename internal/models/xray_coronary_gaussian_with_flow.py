from .xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayCoronaryGaussian
from ..deform_models.deform_model import DeformsMARecoder
from ..deform_models.deform_with_flow.deform_with_flow import DeformsWithFlow

from ..deform_models.deform_with_flow.deform_with_flow import DeformsWithFlow


class XrayCoronaryGaussianWithFlow(XrayCoronaryGaussian):
    def instantiate(self, *args, **kwargs) -> "XrayCoronaryGaussianModelWithFlow":
        return XrayCoronaryGaussianModelWithFlow(self)

class XrayCoronaryGaussianModelWithFlow(XrayCoronaryGaussianModel):
    def __init__(self, config: XrayCoronaryGaussian) -> None:
        super().__init__(config)
        self.deforms_recorder = DeformsMARecoder(deforms_type=DeformsWithFlow)
